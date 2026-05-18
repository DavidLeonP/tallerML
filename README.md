# SkynetSmart — Clasificación de Comportamiento de Pago

> **Taller Final · Machine Learning**  
> Análisis de riesgo crediticio mediante Transfer Learning sobre series de tiempo financieras.

---

## Descripción del caso de negocio

**SkynetSmart** es una empresa con 12 años de trayectoria en el mercado cuencano que provee soluciones tecnológicas a pequeñas, medianas y grandes empresas. Ante la decisión estratégica de adquirir un inmueble para centralizar operaciones (horizonte de 10 años), requiere determinar su **capacidad real de endeudamiento** basada en el comportamiento de pago de su cartera de clientes.

Este proyecto construye un modelo de clasificación que predice si un cliente cumplirá con sus obligaciones de pago en los meses siguientes, utilizando su historial mensual de facturación y cobro como entrada.

---

## Pipeline de Machine Learning

```
CuentasPorCobrar.sql
        │
        ▼  sql_to_csv.py
   dataset.csv  (23 829 transacciones · 35 columnas)
        │
        ▼  feature_extraction.py
   Consolidado mensual por cliente
   (9 345 registros · 3 492 clientes únicos)
        │
        ├── cutoff = 2025-01
        │     ├── Features: historial pre-2025 → tensor [N, T=24, F=10]
        │     └── Label:    comportamiento post-2025 → cumple_pago_heur
        │
        ├── Split train / test  (75% / 25%)  ← ANTES del modelo
        │
        ├── StandardScaler  (fit solo en train)
        │
        ├── LSTM Autoencoder  (pre-entrenado self-supervised, solo train)
        │     Encoder: LSTM(10 → 64) → Linear(64 → 32)
        │     Decoder: repeat → LSTM(32 → 64) → Linear(64 → 10)
        │
        └── Feature vectors  (embeddings de 32 dims por cliente)
                │
                ▼  Taller_Final.ipynb
         Regresión Logística L2  ←→  Random Forest
         Optimización λ · Curva de aprendizaje · Wilcoxon · t-SNE
```

### Decisiones de diseño clave

| Decisión | Justificación |
|---|---|
| Cutoff temporal (2025-01) | Evita data leakage: features y label no comparten la misma ventana |
| Split antes del AE | El encoder nunca ve series de clientes de test durante el pre-entrenamiento |
| Scaler fit solo en train | Las estadísticas de normalización no se contaminan con el test |
| `class_weight="balanced"` | Compensa el desbalance de clases (≈82% Cumple / 18% Riesgo) |
| ROC-AUC como métrica principal | Robusta al desbalance; accuracy sería engañosa |

---

## Estructura del repositorio

```
├── sql_to_csv.py          # Convierte el dump SQL a dataset.csv
├── feature_extraction.py  # Pipeline completo: consolidado → embeddings
├── Taller_Final.ipynb     # Laboratorio: pipelines, λ, LR, Wilcoxon, t-SNE
├── requirements.txt       # Dependencias del proyecto
├── dataset.csv            # Generado por sql_to_csv.py  (no subir al repo)
├── embeddings.csv         # Generado por feature_extraction.py
└── embeddings_pca.png     # Visualización PCA de los embeddings
```

> **Nota**: `CuentasPorCobrar.sql` y `dataset.csv` contienen datos sensibles de la empresa. No los incluyas en el repositorio público. Agrega al `.gitignore`:
> ```
> CuentasPorCobrar.sql
> dataset.csv
> embeddings.csv
> ```

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/<usuario>/<repo>.git
cd <repo>

# Crear entorno virtual (opcional pero recomendado)
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / Mac

# Instalar dependencias
pip install -r requirements.txt

# PyTorch CPU (si pip install -r no lo resuelve automáticamente)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Uso

### Paso 1 — Convertir el dump SQL a CSV

```bash
py sql_to_csv.py
# Genera: dataset.csv (23 829 filas × 35 columnas)
```

Argumentos opcionales:
```
--sql   ruta al archivo .sql    (default: CuentasPorCobrar.sql)
--csv   ruta de salida          (default: dataset.csv)
```

### Paso 2 — Construir embeddings con Transfer Learning

```bash
py feature_extraction.py
```

Argumentos configurables:

| Argumento | Default | Descripción |
|---|---|---|
| `--csv` | `dataset.csv` | CSV de entrada |
| `--cutoff` | `2025-01` | Fecha de corte features / label |
| `--t-meses` | `24` | Longitud T de la serie por cliente |
| `--min-meses` | `3` | Meses mínimos de actividad para incluir un cliente |
| `--test-size` | `0.25` | Fracción reservada para test |
| `--emb-dim` | `32` | Dimensión del vector embedding |
| `--hidden` | `64` | Tamaño oculto del LSTM |
| `--epochs` | `60` | Épocas de pre-entrenamiento self-supervised |
| `--lr` | `1e-3` | Learning rate (Adam) |
| `--seed` | `42` | Semilla de aleatoriedad |

Genera:
- `embeddings.csv` — 204 clientes × 47 columnas (metadatos + 32 dims de embedding + columna `split`)
- `embeddings_pca.png` — Proyección PCA 2D

### Paso 3 — Laboratorio supervisado

Abrir y ejecutar `Taller_Final.ipynb` en Jupyter (Run All).

El notebook contiene las celdas del laboratorio completo y lee directamente `embeddings.csv`. En el notebook se usa la columna `split` para respetar el split ya realizado:

```python
emb_cols = [c for c in emb.columns if c.startswith("emb_")]

X_train = emb[emb["split"] == "train"][emb_cols].to_numpy()
y_train = emb[emb["split"] == "train"]["cumple_pago_heur"].to_numpy()

X_test  = emb[emb["split"] == "test"][emb_cols].to_numpy()
y_test  = emb[emb["split"] == "test"]["cumple_pago_heur"].to_numpy()
```

---

## Resultados

### Optimización de λ (Regresión Logística L2)

| λ óptimo | C óptimo | CV ROC-AUC (10×5 folds) |
|---|---|---|
| 3.16 | 0.316 | **0.9926** |

La curva de regularización muestra una meseta ancha entre λ ≈ 0.1 y λ ≈ 30, indicando robustez a la elección exacta del hiperparámetro.

### Curva de aprendizaje

- Train AUC final (537 muestras): **0.995**
- CV AUC final: **0.993**
- Gap train–CV: **< 0.003** → sin sobreajuste ni subajuste

### Comparación estadística (Wilcoxon signed-rank, 50 folds pareados)

| Modelo | AUC media | AUC std |
|---|---|---|
| Regresión Logística L2 | 0.9794 | 0.0268 |
| Random Forest | 0.9916 | 0.0057 |

**p-value = 0.0148** (α = 0.05) → diferencia estadísticamente significativa.  
Random Forest supera a LR; sin embargo, se recomienda LR para producción por su interpretabilidad.

---

## Variables del dataset consolidado

| Variable | Tipo | Descripción |
|---|---|---|
| `nroDireccion` | ID | Identificador único del cliente |
| `periodo_mes` | Fecha | Mes de actividad |
| `n_facturas_mes` | Numérica | Número de facturas en el mes |
| `total_facturado_mes` | Numérica | Suma del importe cobrado |
| `total_pagado_mes` | Numérica | Suma del importe pagado |
| `total_pendiente_mes` | Numérica | Suma del importe pendiente |
| `dias_credito_promedio` | Numérica | Promedio de días entre factura y vencimiento |
| `facturas_pendientes` | Numérica | Cantidad de facturas con estado P |
| `facturas_cerradas` | Numérica | Cantidad de facturas con estado C |
| `porcentaje_pendiente` | Derivada | `total_pendiente / total_facturado` |
| `ratio_pago` | Derivada | `total_pagado / total_facturado` |
| `ticket_promedio` | Derivada | `total_facturado / n_facturas` |

### Variable objetivo

`cumple_pago_heur` = 1 si el cliente cumple **en los meses post-cutoff**:
- `ratio_pago_post ≥ 0.95` **Y**
- `pct_pendiente_post ≤ 0.10`

---

## Arquitectura del modelo pre-entrenado

```
Input: [batch, T=24, F=10]
         │
    LSTM Encoder (hidden=64)
         │  last hidden state h: [batch, 64]
    Linear(64 → 32)
         │
    Embedding z: [batch, 32]   ← feature vector
         │
    repeat T veces → [batch, 24, 32]
         │
    LSTM Decoder (hidden=64)
         │
    Linear(64 → 10)
         │
Output reconstruido: [batch, 24, 10]

Loss: MSE(input, output)   — self-supervised, sin etiquetas
```

---

## Referencias

1. S. Hochreiter y J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997. doi: 10.1162/neco.1997.9.8.1735
2. D. Kingma y J. Ba, "Adam: A method for stochastic optimization," *ICLR*, 2015. https://arxiv.org/abs/1412.6980
3. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.
4. A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," *NeurIPS*, 2019.
5. Y. Bengio, A. Courville y P. Vincent, "Representation learning: A review and new perspectives," *IEEE TPAMI*, vol. 35, no. 8, pp. 1798–1828, 2013.
6. L. van der Maaten y G. Hinton, "Visualizing data using t-SNE," *JMLR*, vol. 9, pp. 2579–2605, 2008.
7. F. Wilcoxon, "Individual comparisons by ranking methods," *Biometrics Bulletin*, vol. 1, no. 6, pp. 80–83, 1945.
8. C. M. Bishop, *Pattern Recognition and Machine Learning*. Springer, 2006.
9. I. Goodfellow, Y. Bengio y A. Courville, *Deep Learning*. MIT Press, 2016. https://www.deeplearningbook.org

---

## Licencia

Este proyecto fue desarrollado con fines académicos para el Taller Final de Machine Learning.  
Los datos de SkynetSmart son confidenciales y no se distribuyen en este repositorio.
