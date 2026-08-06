# Plan de Feature Engineering — LightGBM (Clase 5)

Catálogo de features para `agregar_features()` en `lgbm_base.py`.
Patrón: cada feature es una línea → `pl.col('tn').<func>().over(clave).alias('nombre')`.
Loop: agrego tanda → mido WAPE local (201912) + feature importance → itero.

> Juez local base (solo lags): **WAPE = 0.2417**. Objetivo: bajarlo y pasar el 0.231 de la regresión.

---

## A. Lags *(serie pura — base)*
- [x] `lag_0 … lag_12`
- [ ] extender: `lag_18`, `lag_24` (si hay historia)

## B. Rolling stats *(ventanas 3/6/12, con `.shift(1)` anti-leakage)*
- [ ] `rmean_3/6/12` — media móvil (tendencia suavizada)
- [ ] `rstd_3/6/12` — volatilidad
- [ ] `rmax_6/12`, `rmin_6/12` — techos/pisos
- [ ] `rmedian_6/12` — robusto a outliers

## C. Deltas / momentum / ratios *(serie pura)*
- [ ] `delta_1 = tn − lag_1`, `delta_12 = tn − lag_12` (YoY)
- [ ] `ratio_1 = tn / lag_1`, `ratio_12 = tn / lag_12`
- [ ] `tn / rmean_12` — anomalía (qué tan raro es el mes)
- [ ] `rmean_3 / rmean_12` — momentum corto vs largo
- [ ] `slope_6` — pendiente de los últimos 6 meses

## D. Frecuencia / intermitencia / ciclo de vida *(serie pura)*
- [ ] `meses_desde_ultima_venta` (recencia)
- [ ] `ceros_12` — cantidad de ceros en 12 meses
- [ ] `racha_ceros` — meses consecutivos sin vender
- [ ] `ADI` — promedio de meses entre compras
- [ ] `antiguedad` — meses desde la primera venta
- [ ] `es_nuevo` — flag (< N meses de historia)

## E. Estacionalidad / calendario *(serie pura)*
- [ ] `mes` (1-12) — como **categórica**
- [ ] `trimestre`

## F. Cosmos 🌌 *(necesita `tb_productos`)*
- [ ] `universo_tn` — total de TODOS los productos por mes + lags/rolling (captura el −22%)
- [ ] `cat1_tn`, `cat2_tn`, `cat3_tn` — totales por categoría + lags
- [ ] `share_cat3 = tn / cat3_tn` — importancia relativa en la categoría
- [ ] `share_universo = tn / universo_tn`

## G. Atributos del producto *(necesita `tb_productos`)*
- [ ] `cat1` (homecare), `cat2` (personal care), `cat3` (food) — **categóricas**
- [ ] `brand` — categórica
- [ ] `sku_size` — tamaño/capacidad (numérica)
- [ ] avanzado: suma del mismo producto en todas las presentaciones, tamaño anterior/posterior

## H. Peso
- [ ] `sample_weight` = volumen reciente (alinea a la métrica ponderada por tn)

## I. Opcional / trabajo futuro
- [ ] `plan_precios_cuidados`, fill-rate (`cust_request` vs `tn`) — fuera del scope actual
- [ ] **Clusters DTW** (`tslearn` → `TimeSeriesKMeans(metric="dtw")`) — cluster como categórica, o para nivel cliente-producto

---

## Orden de implementación
1. **B + C** (rolling + deltas + ratios) — barato, alto impacto
2. **D** (frecuencia / ceros)
3. **F + G** (join `tb_productos` + cosmos) — se espera el salto grande
4. **H** (sample_weight)
5. Optuna + (opcional) DTW
