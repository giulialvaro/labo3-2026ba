"""Clase 5 - LightGBM base workflow.

Arquitectura (segun guia del profe):
  - CLAVE configurable: nivel producto o cliente-producto (hoy: producto)
  - agregar_features(): funcion modular, barata de extender (aca se hace TODA la magia = FE)
  - validacion walk-forward: ancla 201910 -> target 201912 (conocido) -> WAPE local sin gastar submits
  - feature importance al final -> decidir que features nuevas crear (iterativo)

Correr:  python3 src/lightgbm/lgbm_base.py
"""
import polars as pl, numpy as np, lightgbm as lgb, os

PARAM = {
    'clave': ['product_id'],          # o ['customer_id', 'product_id'] (nivel cliente-producto)
    'target_lag': 2,                  # predecir t+2
    'lags': list(range(1, 13)),       # tn de 1..12 meses atras
    'log_target': True,
    'mes_val_ancla': 201910,          # ancla de validacion (target = 201912, conocido)
    'mes_pred_ancla': 201912,         # ancla de prediccion (target = 202002)
}
DATA = 'datasets'


def cargar():
    d = pl.read_csv(f'{DATA}/sell-in.txt.gz', separator='\t')
    apre = pl.read_csv(f'{DATA}/product_id_apredecir201912.txt', separator='\t')
    v = (d.group_by(*PARAM['clave'], 'periodo').agg(pl.col('tn').sum().alias('tn'))
           .join(apre, on='product_id', how='inner')
           .with_columns(pl.col('periodo').cast(pl.Utf8).str.strptime(pl.Date, '%Y%m').alias('ds'))
           .sort(*PARAM['clave'], 'ds'))
    # relleno meses faltantes con 0 (para que los shift sean correctos)
    v = (v.upsample('ds', every='1mo', group_by=PARAM['clave'], maintain_order=True)
           .with_columns(pl.col('tn').fill_null(0.0))
           .with_columns([pl.col(k).fill_null(strategy='forward') for k in PARAM['clave']])
           .with_columns((pl.col('ds').dt.year() * 100 + pl.col('ds').dt.month()).alias('periodo')))
    return v


def agregar_features(df):
    """AQUI SE HACE LA MAGIA. Hoy: solo lags. Jueves: rolling, deltas, estacionalidad, cosmos, tb_productos..."""
    k = PARAM['clave']
    exprs = []
    # --- LAGS (mayor impacto) ---
    for L in PARAM['lags']:
        exprs.append(pl.col('tn').shift(L).over(k).alias(f'lag_{L}'))
    # --- (placeholder) proximas tandas de FE van aca, mismo patron ---
    # ej: exprs.append(pl.col('tn').rolling_mean(3).over(k).alias('rmean_3'))
    df = df.with_columns(exprs)
    # target: tn a +target_lag meses
    df = df.with_columns(pl.col('tn').shift(-PARAM['target_lag']).over(k).alias('target'))
    return df


def wape(y_true, y_pred):
    return np.abs(y_true - y_pred).sum() / y_true.sum()


def main():
    os.makedirs('exp/lgbm', exist_ok=True)
    v = cargar()
    print(f'clave={PARAM["clave"]}  filas={v.height}  claves distintas={v.select(PARAM["clave"]).n_unique()}')

    tb = agregar_features(v)
    feats = [c for c in tb.columns if c.startswith('lag_')]

    # splits por mes ANCLA
    train = tb.filter((pl.col('periodo') <= 201909) & pl.col('target').is_not_null()).drop_nulls(feats)
    val   = tb.filter(pl.col('periodo') == PARAM['mes_val_ancla']).drop_nulls(feats)
    pred  = tb.filter(pl.col('periodo') == PARAM['mes_pred_ancla']).drop_nulls(feats)
    print(f'train={train.height}  val={val.height}  pred={pred.height} filas')

    ytr = np.log1p(train['target'].to_numpy()) if PARAM['log_target'] else train['target'].to_numpy()
    Xtr = train.select(feats).to_pandas()
    Xva = val.select(feats).to_pandas()
    yva = val['target'].to_numpy()

    model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=400, learning_rate=0.05,
                              num_leaves=31, feature_fraction=0.8, bagging_fraction=0.8,
                              min_child_samples=20, verbose=-1)
    model.fit(Xtr, ytr)

    pva = model.predict(Xva)
    if PARAM['log_target']: pva = np.expm1(pva)
    pva = np.clip(pva, 0, None)
    print(f'\n>>> WAPE en validacion (target 201912) = {wape(yva, pva):.4f}')

    # feature importance (para iterar)
    imp = pl.DataFrame({'feature': feats, 'importance': model.feature_importances_}).sort('importance', descending=True)
    print('\nfeature importance (top 12):')
    print(imp.head(12))

    # prediccion 202002 -> submission
    ppred = model.predict(pred.select(feats).to_pandas())
    if PARAM['log_target']: ppred = np.expm1(ppred)
    ppred = np.clip(ppred, 0, None)
    out = pred.select('product_id').with_columns(pl.Series('tn', ppred))
    # fallback promedio para los que quedaron afuera
    apre = pl.read_csv(f'{DATA}/product_id_apredecir201912.txt', separator='\t')
    prom = v.filter(pl.col('periodo').is_between(201901, 201912)).group_by('product_id').agg(pl.col('tn').mean().alias('tn_prom'))
    out = (apre.join(out, on='product_id', how='left').join(prom, on='product_id', how='left')
              .with_columns(pl.coalesce('tn', 'tn_prom').fill_null(0.0).alias('tn')).select('product_id', 'tn'))
    out.write_csv('exp/lgbm/lgbm_base.csv')
    print(f'\nsubmission: exp/lgbm/lgbm_base.csv  ({out.height} productos, tn total={out["tn"].sum():.0f})')


if __name__ == '__main__':
    main()
