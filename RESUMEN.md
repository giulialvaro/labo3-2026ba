# Labo 3 — Resumen maestro (personal)

Recorrido de clases y tareas, con qué hice, qué significa y los resultados.
Predecir las `tn` (toneladas) de **febrero-2020 (202002)** para los **780 productos** a predecir.
Métrica: **Total Error Rate** = `Σ|real−pred| / Σreal` (menor = mejor, ponderada por volumen).

**Número a batir en Kaggle:** `0.249` (AutoGluon del grupo, 7-jul).

---

## Setup / Infraestructura
- Todo corre **local en VS Code** (kernel miniconda `base`, Python 3.13), sin Colab.
- Datasets en `datasets/` (no van al git). Sell-in: 2.9M filas, 36 meses (201701→201912), 1233 productos, 597 clientes.
- Función juez de la métrica en `src/metrica.py` → mide el error en el **mundo ideal** (`z262`, que tiene `tn_real`) **sin gastar submits**.
- Kaggle configurado local (`~/.kaggle/access_token`) → submits con `kaggle competitions submit -c labo-iii-2026-ba -f <archivo> -m "<msg>"`. Límite: 20/día por grupo.

---

## Clase 1 + Tarea 01 — EDA ✅
Notebook: `src/EDA/z101_EDA_01_local.ipynb`

**Consigna:** hacer un EDA buscando hallazgos que respondan *¿cómo lo uso para predecir mejor?*

**4 hallazgos (cada uno → decisión de modelado):**

1. **Mercado en baja: −22% de 2017 a 2019** (500k → 434k → 390k tn).
   → uso features de **tendencia/deltas**, no niveles absolutos; los árboles no extrapolan → nada de `periodo` crudo.
2. **De los 780, 130 no tienen historia completa** (650 con 12 meses, 130 nuevos/con huecos).
   → **fallback**: los 650 al modelo principal; los 130 por analogía (promedio de categoría o modelo global).
3. **Concentración: 50 productos = 50% de las tn** (177 = 80%).
   → concentro esfuerzo en el top + `sample_weight` por volumen (la métrica pondera por tn).
4. **Distribución muy asimétrica** (media 42 vs mediana 10, máx 2295).
   → **target en log** (`log1p` → predecir → `expm1`).

---

## Clase 2 — Modelos Naif ✅
Notebook: `src/naif/z211_Naif_local.ipynb`

**Qué es:** reglas tontas a propósito que copian el pasado. Son el **piso** que todo modelo serio debe ganar.

**Paso a paso:** (1) cargo y agrego a producto×mes, filtro a los 780; (2) calculo 3 naif y los guardo en `exp/naif/`; (3) evalúo con el juez en el mundo ideal; (4) elijo el mejor.

**Los 3 naif y resultados:**

| Naif | Regla | Mundo ideal | Kaggle público |
|---|---|---|---|
| último mes | `tn(202002)=tn(201912)` | 0.6246 | — |
| promedio 12m | `mean(2019)` | 0.4357 | — |
| **mismo mes año anterior** | `tn(201902)` (fallback promedio) | **0.4079** ✅ | **0.271** |
| último mes | | 0.6246 | 0.342 |
| promedio 12m | | 0.4357 | 0.273 |

→ Gana **mismo mes año anterior**: la **estacionalidad** ya aporta (coincide con hallazgo 4).
Ojo: el número del mundo ideal ≠ el público real (datasets distintos); el ideal sirve para comparar **relativo**.

---

## Tarea 02 — Mejor ARIMA ✅
Scripts: `src/Estadistica/tarea2_arima_test.py` (test en ideal) y `tarea2_arima_real.py` (780 reales → submit).

**Qué hice:** como `pmdarima` no anda en py3.13, usé **`statsforecast`** (AutoARIMA de Nixtla, vectorizado, 10 cores → 58s vs 55 min del profe).
Elegí la config en el **mundo ideal** antes de gastar submits:

| Config | Mundo ideal |
|---|---|
| AutoARIMA sin log | 0.1042 |
| **AutoARIMA + log + m12** | **0.0696** ✅ |

Config final: estacionalidad 12 + **target en log** + relleno de meses en 0 + fallback al promedio.

**Resultado en Kaggle real: 0.287** 😲 → **PEOR que el naif** (0.271).

**Lección (tesis del libro):** en el mundo ideal ARIMA aplasta al naif (0.07 vs 0.41), pero en el
**barro real** el ARIMA por-serie sobreajusta el ruido y la caída del mercado le rompe los patrones,
mientras el naif "mismo mes" es **robusto**. *No existe el modelo maravilloso.*

---

## Clase 3 — AutoGluon ⬜ (pendiente)

## Tarea 03 — Mejorar AutoGluon ⬜ (pendiente)

## Clase 4 — Regresión Lineal / aplanado con lags ⬜ (pendiente)

## Clase 5 — LightGBM + Feature Engineering ⬜ (pendiente)

---

## Números en Kaggle (público) — ranking actual
1. AutoGluon (grupo, jul) — **0.249**
2. naif mismo mes — 0.271
3. naif promedio 12m — 0.273
4. AutoARIMA + log — 0.287
5. naif último — 0.342
