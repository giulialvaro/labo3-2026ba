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

## Clase 3 — AutoGluon 🟡 (corriendo en Colab)
Notebook: `src/AutoGluon/z316_AutoGluon.ipynb` (del profe, corre en Colab con GPU T4).

**Qué es AutoGluon:** AutoML de series de tiempo ("matar una mosca con bazooka"). Le tirás los datos
y él **entrena y ensambla ~10 modelos** de las tres familias juntas:
- estadística clásica: SeasonalNaive, AutoETS, Theta
- GBDT / tabular: Recursive/DirectTabular
- deep learning + foundation models: Chronos, TFT, DeepAR

Al final arma un **WeightedEnsemble** que combina los mejores. Es global (usa todas las series juntas).

**Qué hacemos en la clase:**
1. Correr el notebook en Colab (T4 GPU): mount Drive → descargar datos → `install autogluon[all]` (pesado) → `fit` (~1h).
2. Mirar el **leaderboard** de modelos que se va armando + cuál gana.
3. Observar 3 cosas: `eval_metric='RMSE'` (que NO es la métrica del negocio ← se cambia en Tarea 3),
   `num_val_windows=2` (walk-forward, evita leakage), y cambiar el nº de `experimento` en cada corrida.

**Conclusión:** potentísimo pero **caja negra + lento (~1h) + métrica que no es la del negocio**.
Da un número sin que entiendas *por qué*. Por eso la materia va después a **LightGBM con FE artesanal** (Clase 5),
donde una controla todo.

**Resultado:** AutoGluon RMSE = **0.255** en Kaggle (público). Corrido en Colab con GPU T4.

## Tarea 03 — Mejorar AutoGluon (métrica WAPE) ✅
Cambié `eval_metric` de `RMSE` a `WAPE` (que ES la métrica de la competencia: `Σ|real-pred|/Σreal`).

**Resultado contraintuitivo:** WAPE = **0.269**, PEOR que RMSE (0.255). Alinear la métrica de
entrenamiento a la de evaluación NO garantizó mejor leaderboard. Por qué:
1. AutoGluon valida en ventanas de fin-2019, pero el target es feb-2020 (+2 meses) → la mejora en
   validación no transfirió al mes real.
2. **Ruido de una sola corrida** — 0.249 / 0.255 / 0.269 están pegadísimos; hace falta repetir con
   varias semillas para concluir (tal cual dice el libro).
3. El público no es lo que califica (el privado sí).

**Lección:** la bazooka de AutoGluon apenas le gana al naif tonto. *No existe el modelo maravilloso.*

## Clase 4 — Regresión Lineal / aplanado con lags ✅
Notebook: `src/Estadistica/z403_RegresionLineal_local.ipynb`

**La idea (el giro clave):** aplanar la serie en una tabla `(features → target)`.
- `clase = tn(t+2)` (shift -2) → lo que vende dentro de 2 meses
- `tn_0 … tn_11` (shift 0..11) → los 12 meses hacia atrás, como features
- Cada fila = "dados los últimos 12 meses, ¿cuánto vende a +2?". Eso ya es regresión clásica.

**Qué hace z403:** arma la tabla con `.shift()`, entrena OLS en 201812 (~567 filas), predice feb-2020
para 656 productos con historia completa; 124 sin historia → promedio (fallback).

**Resultado en Kaggle: 0.231 → ¡EL MEJOR DE TODOS!** 🥇
Una regresión lineal simple le ganó a AutoGluon (bazooka) y a ARIMA. **No es el modelo lo que importa,
es la reformulación del dato** (serie → tabla con lags). Esta es la razón por la que la materia empuja
al enfoque tabular + GBDT. Es la prueba de concepto del molde que escala a LightGBM (Clase 5).

## Clase 5 — LightGBM + Feature Engineering ✅ (base + FE hechos)

### El workflow que armamos (3 archivos)
- `src/lightgbm/fe.py` → **`build_features(df, cfg)`**: funciones de features modulares. Prendés/apagás
  grupos con un config. Barato agregar features (una línea).
- `src/lightgbm/experimento.py` → **runner**: carga, entrena, mide WAPE, y **loguea** cada corrida.
- `exp/lgbm/resultados.csv` → tabla que **acumula todas las corridas** (base vs FE1 vs FE2...) para comparar.
- `exp/lgbm/imp_<exp>.csv` → feature importance de cada corrida.

Tracking: **CSV simple, no MLflow** (overkill para el timeline). Es lo que pidió el profe ("crear logs").

### Las 2 granularidades
Nivel **producto** (~22k filas, la nuestra, entra local) o **cliente-producto** (~13 millones, necesita
Google Cloud + clustering DTW). La `clave` es configurable, un solo workflow. Elegimos producto por el timeline.

### Qué features probamos y qué pasó
Loop: agrego tanda → mido WAPE local → conservo lo que ayuda, tiro lo que resta.

| Features | WAPE local | Veredicto |
|---|---|---|
| solo lags (0-12) | ~0.253 | base |
| + rolling (3/6/12) | ~0.254 | no ayuda solo |
| + deltas/ratios | ~0.251 | ✅ ayuda un poco |
| + frecuencia/calendario | 0.265+ | 🔴 resta |
| **+ cosmos estacionario** | **~0.248** | ✅ el mejor |

### Las 3 LECCIONES GRANDES (lo más importante para exponer) 🎯
1. **Los árboles NO extrapolan** → nunca uses features de nivel absoluto que crecen/caen con el tiempo
   (ej. `antiguedad`, `universo_tn`, `cat3_tn` que caen con el mercado −22%). Rompen la predicción.
   **Usá siempre ratios / shares / momentum (estacionario).** El workflow lo cazó DOS veces.
2. **Reproducibilidad y ruido:** el mismo modelo da resultados distintos entre corridas (~0.005 de spread)
   por el orden de filas + bagging + threading. **La forma correcta es promediar VARIAS SEMILLAS** y solo
   creer diferencias más grandes que el ruido. Muchas "mejoras" chicas eran ruido.
3. **Más features ≠ mejor.** El FE es SELECTIVO: rolling+deltas+cosmos ayudan; frecuencia resta.

### Dónde quedamos
- Mejor LightGBM (cosmos, 5 semillas promediadas): **0.265 en Kaggle público**.
- **Todavía no le gana a la regresión (0.231) en el PÚBLICO** — pero ver la sección de Estrategia abajo:
  el público no es lo que cuenta.
- Validando sobre un **febrero real** (misma estación que el target), el LightGBM da **0.207** → el modelo
  está bien; febrero-2020 es genuinamente difícil (mercado en caída + elecciones).

### Falta / próximos pasos
- **Comparar manzanas con manzanas**: misma validación (que se parezca a febrero) para TODOS los modelos.
- **Ensamble** regresión + LightGBM + **sample_weight** por volumen (bajar varianza = mejor privado).

---

## Números en Kaggle (PÚBLICO) — ranking actual
1. **Regresión Lineal (aplanado con lags) — 0.231** 🥇
2. AutoGluon RMSE (grupo, jul) — 0.249
3. AutoGluon RMSE (hoy) — 0.255
4. LightGBM cosmos (5 semillas) — 0.265
5. AutoGluon WAPE / LightGBM base solo-lags — 0.269
6. naif mismo mes — 0.271
7. naif promedio 12m — 0.273
8. AutoARIMA + log — 0.287
9. naif último — 0.342

**Observación:** todo apretado entre 0.23 y 0.29. Ni la bazooka ni el FE le sacan ventaja clara a una
regresión simple → *no existe el modelo maravilloso* (tesis del libro).

---

## ⭐ ESTRATEGIA: Público vs Privado (lo más importante)

**El objetivo NO es ganar el Public Leaderboard.** El libro es tajante: el público vale **0% de la nota**,
solo cuenta el **Private Leaderboard** (otro subset de datos, oculto hasta el final).

| | Public LB | Private LB |
|---|---|---|
| Qué es | score sobre un subset visible | score sobre OTRO subset, oculto |
| Cuánto vale | **0%** | **TODO (tu nota)** |

**La trampa:** si elijo mi modelo mirando el público (ej. "la regresión da 0.231, me quedo con esa"),
estoy sobreajustando al subset público → puede dar PEOR en el privado. No sé si la regresión (0.231 púb)
le gana a mi LightGBM (0.265 púb) en el PRIVADO. El número público no me lo dice.

**Cómo hacer que el privado dé bien:**
1. **Validación local honesta que se parezca al target** (validar sobre febreros, no meses cualquiera).
   Si mi CV local lo aprueba y no está tuneado al público → transfiere al privado.
2. **Bajar la varianza:** promediar semillas + ensamblar. Más estable = mejor privado.
3. **No sobreajustar:** modelos simples/robustos suelen ganar el privado.
4. **Elegir la submission final por la validación local, NO por el score público** (Kaggle deja elegir
   cuál cuenta para el privado — hay que usar esa elección).

**En una frase:** construir un modelo que genuinamente prediga bien feb-2020 (validación honesta + baja
varianza), confiar en la validación local por encima del público, y elegir esa submission para el privado.

### Comparación HONESTA (misma vara walk-forward: train ≤201907, val 201908-201910, mismas filas)

| Modelo | WAPE (misma vara) |
|---|---|
| **naif promedio12** | **0.2497** 🥇 |
| regresión (lags) | 0.2584 |
| LightGBM base | 0.2629 |
| LightGBM cosmos (FE) | 0.2656 |
| naif mismo-mes | 0.2922 |
| naif último | 0.3521 |

**BOMBA:** con validación honesta, el **promedio simple de 12 meses le gana a TODO** (regresión, LightGBM con FE).
Y contradice al público (donde ganaba la regresión) → **el ranking del público NO es confiable.**

### Por qué LightGBM no le gana al promedio (iteración)
Probamos regularizar, podar features y target-ratio → TODO empeoró. El LightGBM ya tiene el promedio
como feature (`rmean_12`) y aún así lo empeora: **le agrega varianza al baseline robusto.** Cuando el mejor
predictor es un estadístico simple y estable, un modelo flexible que intenta "mejorarlo" mete ruido (bias-variance).

### La solución: BLEND (ensamble) — lo mejor para el privado ✅
No "mejorar el LightGBM solo", sino **mezclarlo** con el baseline robusto:

| | WAPE |
|---|---|
| solo naif-promedio | 0.2497 |
| solo LightGBM | 0.2649 |
| **BLEND 0.8·naif + 0.2·LightGBM** | **0.2480** ← ¡gana a ambos! |

LightGBM solo es peor, pero está **decorrelacionado** → un poquito de su señal, mezclada con el promedio,
mejora al promedio. Curva de barrido suave con mínimo en 0.7-0.8 → efecto real, no ruido.
Submission: `exp/lgbm/blend_naif80_lgbm20.csv`. **Es la apuesta robusta para el privado.**

### Pendiente / la posta real
- **Nivel cliente-producto** (~13M filas, Google Cloud) → probablemente el salto real, para la competencia (cierra 16-ago).
- Elegir en Kaggle la submission del BLEND como la que cuenta para el privado.

---

## 🏁 ROADMAP COMPETENCIA (cliente-producto — post entrega de clase)
Ya tenemos todos los modelos EXCEPTO el LightGBM a nivel cliente-producto. Infra lista: GCloud + PC (64GB).
Dataset base: `datasets/sell-in-zeroes.txt.gz` (17M filas, generado con `z601`, ceros en meses sin venta).

### 1. FE cliente-producto SUPER completo + ESCALADO
**Escalado del target (técnica alumna Rosario — lo más importante):**
```
tn_escalado = tn / promedio(mes actual + anteriores)   # por serie, sin leakage
clase = tn(t+2) escalado
```
→ el modelo aprende patrones RELATIVOS, no niveles absolutos (los árboles no extrapolan niveles).

**FE completo (hacer de más, cortar por importancia después):**
- Historia: lags, delta lags, medias móviles, tendencia, max/min
- Cosmos/categorías: suma todos los productos, cat1/2/3/marca, **suma mismo producto todos los clientes**,
  **suma mismo cliente todos los productos** (nuevas, potentes en cliente-producto)

**Hiperparámetros LightGBM:**
- `objective='tweedie'` (mayoría de datos son 0 en cliente-producto)
- `max_bin=1230` (acá alto funciona mejor, al revés que lo usual)

### 2. DTW Clustering (7 clusters)
- Clustering jerárquico con distancia **DTW** sobre las series (input **escalado**) → columna `cluster` (1-7)
- Doble propósito: (a) el dataset no entra en RAM → K clusters = K modelos más ajustados;
  (b) resuelve series mixtas (un cliente vende mayo Y sopa)
- DTW es LENTO → **samplear**. Único hiperparámetro: la **ventana de tiempo** (probar).
- Alternativa rápida si apura: clusters por **cuartiles de toneladas**.

### 3. Correr todo + ENSAMBLE
- Correr los mejores (AutoGluon, regresión, LightGBM por cluster) → **ensamblar**.
- *"Siempre lo que mejor funciona es un ensemble"* (ya visto: blend 0.2480).

**Orden:** FE+escalado → LightGBM (tweedie/max_bin) validación honesta → DTW clustering → correr todo + ensamble.

---

## Workflow final LightGBM (16-ago)

Se rearmó el pipeline completo para comparar sin mezclar granularidades ni ventanas:

- `producto_limpio.py`: control a nivel producto, validación principal febrero-2019 y control diciembre-2019.
- `fe_cp_a_producto.py`: transforma los 17M registros en estructura comercial por producto
  (compradores, altas, repetición, concentración, fill rate y regímenes).
- `dtw_clusters_cp_v2.py`: separa sparse antes de DTW, usa DTW real para asignar y preserva `null`
  fuera de la vida observada.
- `cliente_producto_hurdle.py`: clasificador de compra × regresor Tweedie de cantidad; mide WAPE
  solamente después de sumar clientes a producto.
- `ensamble_final.py`: mezcla contra el `src/Estadistica/linreg.csv` exacto de 0.231.

### DTW v2

Los 722.457 pares quedaron segmentados así:

- 348.018 nunca compraron.
- 117.437 compraron 1-2 meses.
- 83.212 compraron 3-5 meses.
- 173.790 series con 6+ compras pasan a cuatro clusters DTW.

Para no filtrar futuro, los medoides se versionan por corte: en validación se aprenden en el año de
entrenamiento y se reutilizan al asignar el año siguiente. Para el final se aprenden hasta 201812 y se
aplican a 201912.

### Control producto, misma vara

| Modelo | Febrero local | Diciembre control | Score 70/30 |
|---|---:|---:|---:|
| **LightGBM lags raw** | **0.1914** | **0.2558** | **0.2107** |
| FE limpio raw | 0.1952 | 0.2859 | 0.2224 |
| FE limpio log | 0.1986 | 0.2821 | 0.2237 |
| FE limpio + stocks | 0.1969 | 0.2899 | 0.2248 |
| Cliente-producto agregado | 0.2178 | 0.3884 | 0.2690 |

Conclusión provisional: en producto, agregar todas las features aumenta varianza; el control robusto sigue
siendo `lags_raw`. La ruta cliente-producto de dos etapas queda como challenger a ejecutar completa en GCP,
no como reemplazo automático.

### Ensamble realmente correcto

Los blends viejos no usaban la regresión exacta. Ahora se generaron contra el archivo real de 0.231:

- 90% regresión exacta + 10% LightGBM limpio.
- 80% regresión exacta + 20% LightGBM limpio.
- 70% regresión exacta + 30% LightGBM limpio.

Están en `exp/ensamble_final/`. No se suben automáticamente: primero se revisa el challenger completo.
