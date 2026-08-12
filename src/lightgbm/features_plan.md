# Plan de Feature Engineering — LightGBM cliente-producto (competencia)

Nivel: **cliente-producto** (dataset `sell-in-zeroes`, 17M filas, con ceros).
LightGBM: `objective='tweedie'`, `max_bin=1230` (SIN log — Tweedie maneja ceros/asimetría).
Regla: **features de más, cortar por importancia después**. Todo nivel absoluto → ratio/momentum (estacionario).

---

## A. 🔑 ESCALADO (la base — técnica Rosario)
- [ ] `promedio_nivel` = `rolling_mean(tn)` del mes actual + anteriores, por serie (guardar como columna)
- [ ] `tn_escalado = tn / promedio_nivel`
- [ ] **target = `tn(t+2)` escalado** → al predecir, des-escalar (× `promedio_nivel`)
- ⚠️ el promedio usa SOLO actual+pasado (sin leakage)

## B. Historia (sobre la serie escalada)
- [ ] Lags 0-12, **18, 24**
- [ ] Rolling mean/std/min/max: **3/6/9/12/24**
- [ ] Deltas/ratios: `tn−lag1`, `tn−lag12`, `tn/lag1`, `tn/lag12`
- [ ] **Delta de lags** (lag_i − lag_j)
- [ ] **Tendencia** (slope últimos N)
- [ ] **Rolling median** (6/12) + **EWMA** (varios spans)

## C. Frecuencia (básica — el resto NO)
- [ ] meses desde última venta
- [ ] ceros_12
- (racha ceros / ADI / CV² → descartados)

## D. 🌌 Cosmos / agregaciones (lo más potente acá)
Totales por mes (+ momentum YoY estacionario, NO el nivel):
- [ ] suma universo (todos los productos)
- [ ] suma cat1 / cat2 / cat3 / marca
- [ ] **suma mismo PRODUCTO, todos los clientes** (demanda total del producto)
- [ ] **suma mismo CLIENTE, todos los productos** (cuánto compra el cliente en total)

Shares (mi participación):
- [ ] `mi_tn / total_producto`
- [ ] `mi_tn / total_cliente`
- [ ] `mi_tn / total_cat3`

## E. Categóricas
- [ ] cat1, cat2, cat3, marca, sku_size
- [ ] **cluster** (del DTW — se agrega después)
- [ ] **target encoding** de customer_id / product_id (media del target por cliente/producto)
  - ⚠️ calcular out-of-fold / solo en train para no filtrar

## F. Precio / demanda (están en el sell-in)
- [ ] `plan_precios_cuidados` + historia (meses en plan)
- [ ] **fill rate** = `tn / cust_request_tn` (+ lags) → proxy quiebre de stock / demanda insatisfecha

## G. Calendario
- [ ] `mes` (categórico) + `trimestre`

## H. Relaciones (extra)
- [ ] **Amplitud del producto**: nº de clientes distintos que lo compran + tendencia
- [ ] **Diversidad del cliente**: nº de productos distintos que compra
- [ ] **Par vs total**: crece/cae este par vs el total del cliente / del producto

---

## Orden de implementación
1. A (escalado) + B (historia) → primer LightGBM tweedie, validación honesta
2. D (cosmos/agregaciones) + F (precio/demanda) → medir
3. E (target encoding) + H (relaciones) → medir
4. Podar por importancia · DTW clustering (columna cluster) · ensamble
