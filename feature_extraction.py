"""
feature_extraction.py
=====================

Pipeline de Transfer Learning para series de tiempo financieras
(cuentas por cobrar de SkynetSmart).

Flujo:

    Time series  ->  Pretrained model  ->  Vector embedding  ->  (ML algorithm)
    (consolidado     (LSTM Autoencoder      (feature vector
     mensual por      pre-entrenado         por cliente, listo
     cliente)         self-supervised)      para TL aguas abajo)

Pasos:
  1. Carga `dataset.csv` (salida de sql_to_csv.py).
  2. Filtra ventas RI y construye el consolidado mensual por cliente.
  3. Genera variables derivadas (porcentaje_pendiente, ratio_pago,
     ticket_promedio, dias_credito_promedio, ...).
  4. Construye un tensor 3D  X de forma [N_clientes, T_meses, F_features]
     alineando las series a los últimos T meses de actividad por cliente.
  5. Pretexto self-supervised: entrena un LSTM Autoencoder para reconstruir
     X.  El encoder aprende un embedding compacto por cliente.
  6. Congela el encoder y extrae el feature vector (embedding) de cada
     cliente -> `embeddings.csv`.
  7. Visualiza los embeddings en 2D con PCA.

Uso:
    py feature_extraction.py
    py feature_extraction.py --csv dataset.csv --t-meses 24 --emb-dim 32 --epochs 60
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

FEATURES: list[str] = [
    "n_facturas_mes",
    "total_facturado_mes",
    "total_pagado_mes",
    "total_pendiente_mes",
    "dias_credito_promedio",
    "facturas_pendientes",
    "facturas_cerradas",
    "porcentaje_pendiente",
    "ratio_pago",
    "ticket_promedio",
]


def cargar_y_consolidar(csv_path: Path) -> pd.DataFrame:
    """Carga el CSV, filtra ventas RI y construye el consolidado mensual."""
    print(f"[1/6] Cargando {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"      Filas totales: {len(df):,}")

    df["fechaFactura"] = pd.to_datetime(df["fechaFactura"], errors="coerce")
    df["fechaVencimiento"] = pd.to_datetime(df["fechaVencimiento"], errors="coerce")
    df = df.dropna(subset=["fechaFactura"])

    df_ri = df[df["tipoDoc"] == "RI"].copy()
    print(f"      Filas RI: {len(df_ri):,}")

    df_ri["dias_credito"] = (df_ri["fechaVencimiento"] - df_ri["fechaFactura"]).dt.days
    df_ri["periodo_mes"] = df_ri["fechaFactura"].dt.to_period("M")

    for col in ["importeCobrado", "importePagado", "importePendiente"]:
        df_ri[col] = pd.to_numeric(df_ri[col], errors="coerce").fillna(0)

    print("[2/6] Consolidado mensual por cliente ...")
    consolidado = (
        df_ri.groupby(["nroDireccion", "periodo_mes"])
        .agg(
            n_facturas_mes=("nroDoc", "nunique"),
            total_facturado_mes=("importeCobrado", "sum"),
            total_pagado_mes=("importePagado", "sum"),
            total_pendiente_mes=("importePendiente", "sum"),
            dias_credito_promedio=("dias_credito", "mean"),
            facturas_pendientes=("estado", lambda x: (x == "P").sum()),
            facturas_cerradas=("estado", lambda x: (x == "C").sum()),
        )
        .reset_index()
    )

    consolidado["porcentaje_pendiente"] = (
        consolidado["total_pendiente_mes"] / consolidado["total_facturado_mes"]
    ).replace([np.inf, -np.inf], 0).fillna(0)

    consolidado["ratio_pago"] = (
        consolidado["total_pagado_mes"] / consolidado["total_facturado_mes"]
    ).replace([np.inf, -np.inf], 0).fillna(0)

    consolidado["ticket_promedio"] = (
        consolidado["total_facturado_mes"] / consolidado["n_facturas_mes"]
    ).replace([np.inf, -np.inf], 0).fillna(0)

    consolidado["dias_credito_promedio"] = consolidado["dias_credito_promedio"].fillna(0)

    consolidado["periodo_mes_ts"] = consolidado["periodo_mes"].dt.to_timestamp()
    print(f"      Registros consolidados: {len(consolidado):,} "
          f"| clientes únicos: {consolidado['nroDireccion'].nunique():,}")
    return consolidado


def construir_tensor(
    consolidado: pd.DataFrame,
    t_meses: int,
    min_meses_activos: int,
    cutoff: str = "2025-01",
) -> tuple[np.ndarray, list[int], pd.DataFrame]:
    """
    Construye X [N, T, F] con separación temporal correcta para evitar data leakage.

    Separación por cutoff date:
      - FEATURES (embedding): últimos T meses del historial ANTES del cutoff.
        Son los datos que el modelo "conoce" al momento de predecir.
      - LABEL (cumple_pago_heur): comportamiento DESPUÉS del cutoff.
        Es lo que el modelo debe predecir (futuro real).

    Esto evita el sesgo donde features y label comparten la misma ventana de
    tiempo y el modelo aprende a memorizar comportamientos ya observados en vez
    de generalizar a periodos futuros.

    Solo se incluyen clientes con al menos `min_meses_activos` meses ANTES del
    cutoff (no tiene sentido predecir a clientes sin historial previo).
    Clientes sin actividad post-cutoff se excluyen (no se puede calcular su label).
    """
    print(f"[3/6] Construyendo tensor de series (T={t_meses}, F={len(FEATURES)}, cutoff={cutoff}) ...")

    cutoff_ts = pd.Timestamp(cutoff)
    consolidado_pre  = consolidado[consolidado["periodo_mes_ts"] <  cutoff_ts]
    consolidado_post = consolidado[consolidado["periodo_mes_ts"] >= cutoff_ts]

    n_pre  = consolidado_pre["nroDireccion"].nunique()
    n_post = consolidado_post["nroDireccion"].nunique()
    print(f"      Clientes con historial pre-cutoff:  {n_pre:,}")
    print(f"      Clientes con actividad post-cutoff: {n_post:,}")

    grupos_pre  = consolidado_pre.sort_values("periodo_mes_ts").groupby("nroDireccion")
    grupos_post = consolidado_post.groupby("nroDireccion")

    series: list[np.ndarray] = []
    clientes: list[int] = []
    meta: list[dict] = []

    for cliente, g_pre in grupos_pre:
        # Requiere historial previo suficiente
        if len(g_pre) < min_meses_activos:
            continue

        # Requiere actividad futura para poder calcular la etiqueta real
        if cliente not in grupos_post.groups:
            continue
        g_post = grupos_post.get_group(cliente)

        # --- FEATURES: últimos T meses ANTES del cutoff ---
        g_last = g_pre.tail(t_meses)
        x = g_last[FEATURES].to_numpy(dtype=np.float32)
        if x.shape[0] < t_meses:
            pad = np.zeros((t_meses - x.shape[0], len(FEATURES)), dtype=np.float32)
            x = np.vstack([pad, x])

        # --- LABEL: comportamiento real en los meses DESPUÉS del cutoff ---
        ratio_pago_post    = float(g_post["ratio_pago"].mean())
        pct_pendiente_post = float(g_post["porcentaje_pendiente"].mean())

        series.append(x)
        clientes.append(int(cliente))
        meta.append(
            {
                "nroDireccion": int(cliente),
                "cutoff": cutoff,
                "n_meses_pre_cutoff":  int(len(g_pre)),
                "n_meses_post_cutoff": int(len(g_post)),
                "primer_mes": g_pre["periodo_mes_ts"].min().strftime("%Y-%m"),
                "ultimo_mes_pre":  g_pre["periodo_mes_ts"].max().strftime("%Y-%m"),
                "primer_mes_post": g_post["periodo_mes_ts"].min().strftime("%Y-%m"),
                "total_facturado_pre":  float(g_pre["total_facturado_mes"].sum()),
                "total_pagado_pre":     float(g_pre["total_pagado_mes"].sum()),
                "total_pendiente_pre":  float(g_pre["total_pendiente_mes"].sum()),
                "ratio_pago_pre":       float(g_pre["ratio_pago"].mean()),
                "ratio_pago_post":      ratio_pago_post,
                "pct_pendiente_post":   pct_pendiente_post,
                # Etiqueta calculada sobre meses FUTUROS (post-cutoff)
                "cumple_pago_heur": int(
                    ratio_pago_post >= 0.95 and pct_pendiente_post <= 0.10
                ),
            }
        )

    X = np.stack(series, axis=0)
    df_meta = pd.DataFrame(meta)
    print(f"      Clientes con split temporal completo: {X.shape[0]:,}")
    print(f"      Tensor X: {X.shape}")
    print(f"      Etiqueta cumple_pago_heur=1 (post-cutoff): "
          f"{int(df_meta['cumple_pago_heur'].sum()):,} "
          f"({df_meta['cumple_pago_heur'].mean()*100:.1f}%)")
    return X, clientes, df_meta


def escalar_tensor(
    X: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, StandardScaler]:
    """
    Aplana [N,T,F] a [N*T,F], ajusta StandardScaler SOLO con clientes de train
    y lo aplica a todos (train + test). Evita que estadísticas del test
    contaminen la normalización.
    """
    print("[4/6] Escalando features (fit solo en train) ...")
    N, T, F = X.shape
    scaler = StandardScaler()
    scaler.fit(X[train_mask].reshape(-1, F))
    X_scaled = scaler.transform(X.reshape(-1, F)).reshape(N, T, F).astype(np.float32)
    return X_scaled, scaler


class LSTMAutoencoder(nn.Module):
    """
    Pretrained model: LSTM Autoencoder self-supervised.

    Encoder:  LSTM(F -> H)  ->  Linear(H -> emb_dim)   = embedding por cliente
    Decoder:  repite el embedding T veces -> LSTM(emb_dim -> H) -> Linear(H -> F)
    """

    def __init__(self, n_features: int, hidden_dim: int = 64, emb_dim: int = 32, t_steps: int = 24):
        super().__init__()
        self.t_steps = t_steps
        self.emb_dim = emb_dim

        self.enc_lstm = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.enc_to_z = nn.Linear(hidden_dim, emb_dim)

        self.dec_lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.dec_to_x = nn.Linear(hidden_dim, n_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.enc_lstm(x)
        z = self.enc_to_z(h.squeeze(0))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z_rep = z.unsqueeze(1).repeat(1, self.t_steps, 1)
        out, _ = self.dec_lstm(z_rep)
        x_hat = self.dec_to_x(out)
        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def pretrain_ae(
    X_train: np.ndarray,
    emb_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> LSTMAutoencoder:
    """
    Pre-entrena el autoencoder de forma self-supervised (reconstrucción MSE)
    usando ÚNICAMENTE las series de los clientes de entrenamiento.

    El conjunto de test nunca toca los pesos del encoder: cuando después se
    extraigan sus embeddings, el encoder los verá por primera vez, igual que
    ocurriría en producción con un cliente nuevo.
    """
    print(f"[5/6] Pre-entrenamiento self-supervised del LSTM Autoencoder ...")
    print(f"      Entrenando solo con {X_train.shape[0]} clientes de train "
          f"(los clientes de test están aislados).")
    torch.manual_seed(seed)
    np.random.seed(seed)

    N, T, F = X_train.shape
    device = torch.device("cpu")
    model = LSTMAutoencoder(n_features=F, hidden_dim=hidden_dim, emb_dim=emb_dim, t_steps=T).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Partición interna 90/10 dentro de train para monitorear MSE de reconstrucción
    X_t = torch.from_numpy(X_train)
    n_val = max(1, N // 10)
    idx = np.random.permutation(N)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    X_tr, X_val = X_t[tr_idx], X_t[val_idx]

    n_tr = X_tr.shape[0]
    best_val = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_tr)
        total = 0.0
        for i in range(0, n_tr, batch_size):
            b = X_tr[perm[i:i + batch_size]].to(device)
            opt.zero_grad()
            x_hat, _ = model(b)
            loss = loss_fn(x_hat, b)
            loss.backward()
            opt.step()
            total += loss.item() * b.shape[0]
        train_loss = total / n_tr

        model.eval()
        with torch.no_grad():
            x_hat, _ = model(X_val.to(device))
            val_loss = loss_fn(x_hat, X_val.to(device)).item()
        best_val = min(best_val, val_loss)

        if ep == 1 or ep % max(1, epochs // 10) == 0 or ep == epochs:
            print(f"      epoch {ep:3d}/{epochs} | train MSE={train_loss:.4f} | val MSE={val_loss:.4f}")

    print(f"      Mejor val MSE: {best_val:.4f}")
    return model


def extraer_embeddings(model: LSTMAutoencoder, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Aplica el encoder congelado para obtener un feature vector por cliente."""
    print("[6/6] Extrayendo embeddings (feature vectors) ...")
    model.eval()
    embs: list[np.ndarray] = []
    X_t = torch.from_numpy(X)
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch_size):
            b = X_t[i:i + batch_size]
            z = model.encode(b).cpu().numpy()
            embs.append(z)
    Z = np.concatenate(embs, axis=0)
    print(f"      Embeddings: {Z.shape}  ({Z.shape[1]} dimensiones por cliente)")
    return Z


def guardar_embeddings(
    Z: np.ndarray,
    clientes: list[int],
    df_meta: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    """Combina metadatos + embeddings y guarda como CSV listo para downstream ML."""
    emb_cols = [f"emb_{i}" for i in range(Z.shape[1])]
    df_emb = pd.DataFrame(Z, columns=emb_cols)
    df_emb.insert(0, "nroDireccion", clientes)
    df_out = df_meta.merge(df_emb, on="nroDireccion", how="inner")
    df_out.to_csv(out_path, index=False)
    print(f"      Guardado: {out_path}  ({len(df_out):,} filas, {df_out.shape[1]} columnas)")
    return df_out


def graficar_pca(Z: np.ndarray, df_meta: pd.DataFrame, out_img: Path) -> None:
    """Proyección 2D con PCA de los embeddings, coloreado por etiqueta heurística."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("      matplotlib no disponible -> se omite gráfico.")
        return
    pca = PCA(n_components=2, random_state=0)
    P = pca.fit_transform(Z)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 6))
    y = df_meta["cumple_pago_heur"].to_numpy()
    for lab, col, name in [(1, "#1f77b4", "Cumple"), (0, "#d62728", "Riesgo")]:
        m = y == lab
        ax.scatter(P[m, 0], P[m, 1], s=12, c=col, alpha=0.6, label=name)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    ax.set_title("Embeddings de clientes (PCA 2D) - pretrained LSTM-AE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_img, dpi=130)
    plt.close(fig)
    print(f"      Gráfico guardado: {out_img}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Feature extraction con LSTM Autoencoder pretrained.")
    parser.add_argument("--csv", default="dataset.csv", help="CSV de entrada.")
    parser.add_argument("--out", default="embeddings.csv", help="CSV de salida con feature vectors.")
    parser.add_argument("--img", default="embeddings_pca.png", help="Gráfico PCA de salida.")
    parser.add_argument("--t-meses", type=int, default=24, help="Longitud T de la serie por cliente.")
    parser.add_argument("--min-meses", type=int, default=3, help="Meses mínimos de actividad para incluir cliente.")
    parser.add_argument("--cutoff", default="2025-01",
                        help="Fecha de corte 'YYYY-MM'. Features=pre-cutoff, Label=post-cutoff.")
    parser.add_argument("--test-size", type=float, default=0.25,
                        help="Fracción de clientes reservados para test (default 0.25).")
    parser.add_argument("--emb-dim", type=int, default=32, help="Dimensión del vector embedding.")
    parser.add_argument("--hidden", type=int, default=64, help="Tamaño oculto del LSTM.")
    parser.add_argument("--epochs", type=int, default=60, help="Épocas de pre-entrenamiento self-supervised.")
    parser.add_argument("--batch", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla.")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    img_path = Path(args.img)

    # --- Paso 1-3: consolidado y tensor completo ---
    consolidado = cargar_y_consolidar(csv_path)
    X_raw, clientes, df_meta = construir_tensor(
        consolidado, args.t_meses, args.min_meses, args.cutoff
    )

    # --- Paso 4: split de clientes ANTES de tocar el modelo ---
    # El split se hace aquí, antes del escalado y antes del AE,
    # para que ningún cliente de test influya en los pesos del encoder
    # ni en las estadísticas del escalador.
    N = X_raw.shape[0]
    rng = np.random.default_rng(args.seed)
    idx_perm = rng.permutation(N)
    n_test = max(1, int(N * args.test_size))
    test_idx  = idx_perm[:n_test]
    train_idx = idx_perm[n_test:]

    train_mask = np.zeros(N, dtype=bool)
    train_mask[train_idx] = True
    test_mask = ~train_mask

    df_meta["split"] = np.where(train_mask, "train", "test")
    print(f"\n[split] train: {train_mask.sum()} clientes | test: {test_mask.sum()} clientes")
    print(f"        Balance en train: cumple={df_meta.loc[train_mask,'cumple_pago_heur'].mean()*100:.1f}%")
    print(f"        Balance en test:  cumple={df_meta.loc[test_mask, 'cumple_pago_heur'].mean()*100:.1f}%")

    # --- Paso 5: escalado (fit SOLO en train, transform en todos) ---
    X_scaled, _scaler = escalar_tensor(X_raw, train_mask)

    # --- Paso 6: pre-entrena AE SOLO con clientes de train ---
    model = pretrain_ae(
        X_scaled[train_mask],
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        seed=args.seed,
    )

    # --- Paso 7: extrae embeddings de TODOS (encoder congelado) ---
    # Los clientes de test son "nuevos" para el encoder: nunca vio sus series
    # durante el entrenamiento, igual que ocurriría en producción.
    Z = extraer_embeddings(model, X_scaled, batch_size=args.batch)
    df_out = guardar_embeddings(Z, clientes, df_meta, out_path)
    graficar_pca(Z, df_out, img_path)

    print("\nResumen:")
    print(f"  - Clientes totales con embedding: {len(df_out):,}")
    print(f"    · train: {train_mask.sum():,}  · test: {test_mask.sum():,}")
    print(f"  - Dim. del feature vector: {args.emb_dim}")
    print(f"  - En el notebook usa la columna `split` para separar train/test:")
    print(f"    X_train = emb[emb['split']=='train'][emb_cols]")
    print(f"    X_test  = emb[emb['split']=='test' ][emb_cols]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
