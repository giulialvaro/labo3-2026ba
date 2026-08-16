# Workflow LightGBM final

## Orden

- [x] Baseline exacto: `src/Estadistica/linreg.csv` = 0.231 en Kaggle.
- [x] LightGBM limpio producto con validacion febrero + control diciembre.
- [x] Features de los 17M cliente-producto agregadas a producto.
- [x] DTW v2: regimenes sparse + DTW real para series con 6+ compras.
- [x] Snapshots DTW sin leakage entre entrenamiento y prediccion.
- [x] Hurdle cliente-producto: probabilidad de compra x cantidad Tweedie.
- [x] Variante global, cluster como feature y un modelo por cluster.
- [x] Agregacion cliente-producto a producto antes de calcular WAPE.
- [x] Ensambles 90/10, 80/20 y 70/30 con la regresion exacta.
- [ ] Correr challenger completo en GCP y elegir por validacion.
- [ ] Subir a Kaggle solamente los candidatos que superen el control.

## Instalacion

```bash
pip install -r requirements-lightgbm.txt
```

En Mac, `dtaidistance` necesita OpenMP:

```bash
brew install libomp
```

## Pruebas locales

```bash
python3 src/lightgbm/test_fe.py
python3 src/lightgbm/producto_limpio.py
python3 src/lightgbm/cliente_producto_hurdle.py --sample-products 20
```

El runner cliente-producto usa una muestra por defecto. Nunca procesa las 17M filas
por accidente.

## DTW sin leakage

Genera tres pares de snapshots. Los medoides se aprenden en el corte de entrenamiento
y se reutilizan en el corte de prediccion.

```bash
python3 src/lightgbm/run_pipeline.py --dtw-snapshots
```

Segmentos finales:

- `0`: nunca compro.
- `1`: compro 1-2 meses.
- `2`: compro 3-5 meses.
- `3-6`: cuatro formas DTW entre series con 6+ compras.

## Corrida completa GCP

Desde la raiz del repo montado en la VM:

```bash
python3 -u src/lightgbm/cliente_producto_hurdle.py \
  --sample-products 0 \
  --variants one_stage,hurdle,hurdle_cluster,hurdle_per_cluster \
  --seeds 1
```

Cuando ya se eligio variante, generar submission final:

```bash
python3 -u src/lightgbm/cliente_producto_hurdle.py \
  --sample-products 0 \
  --variants hurdle_per_cluster \
  --seeds 1,7,19 \
  --min-cluster-rows 50000 \
  --min-cluster-positive 1000 \
  --final
```

## Ensamble exacto

```bash
python3 src/lightgbm/ensamble_final.py \
  --candidate exp/lgbm_cp_clean/hurdle_per_cluster.csv
```

Genera en `exp/ensamble_final/`:

- 90% regresion + 10% LightGBM.
- 80% regresion + 20% LightGBM.
- 70% regresion + 30% LightGBM.

No se hace submit automatico: primero se revisan WAPE, volumen total y archivos.
