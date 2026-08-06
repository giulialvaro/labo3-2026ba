"""Runner de experimentos LightGBM con logging.

run_experimento(nombre, cfg, submit=False) -> entrena, mide WAPE local, y LOGUEA:
  - exp/lgbm/resultados.csv        (una fila por corrida: WAPE, n_features, top10)
  - exp/lgbm/imp_<nombre>.csv      (feature importance completa)
  - exp/lgbm/<nombre>.csv          (submission, si submit=True)

Asi quedan las comparaciones base vs FE1 vs FE2 en un solo lugar.
"""
import polars as pl, numpy as np, lightgbm as lgb, os, datetime
import fe

os.makedirs('exp/lgbm', exist_ok=True)
LOG = 'exp/lgbm/resultados.csv'


def wape(y_true, y_pred):
    return float(np.abs(y_true - y_pred).sum() / y_true.sum())


def run_experimento(nombre, cfg, submit=False):
    v = fe.cargar(cfg['clave'])
    tb, feats, cat_feats = fe.build_features(v, cfg)

    # validacion multi-mes (robusta): entreno <=201906, valido en 4 meses ancla pooled
    VAL_MESES = [201907, 201908, 201909, 201910]   # targets 201909..201912 (conocidos)
    train = tb.filter((pl.col('periodo') <= 201906) & pl.col('target').is_not_null()).drop_nulls(feats)
    val   = tb.filter(pl.col('periodo').is_in(VAL_MESES)).drop_nulls(feats)
    pred  = tb.filter(pl.col('periodo') == 201912).drop_nulls(feats)

    ytr = np.log1p(train['target'].to_numpy()) if cfg['log_target'] else train['target'].to_numpy()
    model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=400, learning_rate=0.05,
                              num_leaves=31, feature_fraction=0.8, bagging_fraction=0.8,
                              min_child_samples=20, verbose=-1, random_state=102191)
    cats = [c for c in cat_feats if c in feats]
    model.fit(train.select(feats).to_pandas(), ytr,
              categorical_feature=cats if cats else 'auto')

    pva = model.predict(val.select(feats).to_pandas())
    if cfg['log_target']: pva = np.expm1(pva)
    pva = np.clip(pva, 0, None)
    w = wape(val['target'].to_numpy(), pva)

    # feature importance
    imp = (pl.DataFrame({'feature': feats, 'importance': model.feature_importances_})
             .sort('importance', descending=True))
    imp.write_csv(f'exp/lgbm/imp_{nombre}.csv')
    top10 = ' | '.join(imp.head(10)['feature'].to_list())

    # log
    fila = pl.DataFrame({'fecha': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M')],
                         'experimento': [nombre], 'n_features': [len(feats)],
                         'wape_local': [round(w, 4)], 'top10': [top10]})
    if os.path.exists(LOG):
        fila = pl.concat([pl.read_csv(LOG), fila], how='diagonal_relaxed')
    fila.write_csv(LOG)

    print(f'[{nombre}]  n_features={len(feats)}  WAPE_local={w:.4f}')
    print(f'  top5: {" | ".join(imp.head(5)["feature"].to_list())}')

    if submit:
        # para el submit REAL reentreno con todos los datos disponibles (target conocido)
        full = tb.filter((pl.col('periodo') <= 201910) & pl.col('target').is_not_null()).drop_nulls(feats)
        yfull = np.log1p(full['target'].to_numpy()) if cfg['log_target'] else full['target'].to_numpy()
        model.fit(full.select(feats).to_pandas(), yfull, categorical_feature=cats if cats else 'auto')
        pp = model.predict(pred.select(feats).to_pandas())
        if cfg['log_target']: pp = np.expm1(pp)
        pp = np.clip(pp, 0, None)
        out = pred.select('product_id').with_columns(pl.Series('tn', pp))
        apre = pl.read_csv(f'{fe.DATA}/product_id_apredecir201912.txt', separator='\t')
        prom = v.filter(pl.col('periodo').is_between(201901, 201912)).group_by('product_id').agg(pl.col('tn').mean().alias('tn_prom'))
        out = (apre.join(out, on='product_id', how='left').join(prom, on='product_id', how='left')
                  .with_columns(pl.coalesce('tn', 'tn_prom').fill_null(0.0).alias('tn')).select('product_id', 'tn'))
        out.write_csv(f'exp/lgbm/{nombre}.csv')
        print(f'  submission -> exp/lgbm/{nombre}.csv')
    return w


# ---- configs de experimentos ----
BASE = {'clave': ['product_id'], 'target_lag': 2, 'log_target': True,
        'lags': list(range(0, 13)), 'rolling_windows': [], 'deltas': False,
        'frecuencia': False, 'calendario': False}


def cfg(**over):
    c = dict(BASE); c.update(over); return c


if __name__ == '__main__':
    mejor = dict(rolling_windows=[3, 6, 12], deltas=True)   # el mejor de Tier 1/2
    run_experimento('t1_rolling_deltas', cfg(**mejor))
    run_experimento('t3_cosmos_estac', cfg(**mejor, cosmos=True))
    run_experimento('t3_attrs', cfg(**mejor, producto_attrs=True))
    run_experimento('t3_cosmos_attrs', cfg(**mejor, cosmos=True, producto_attrs=True))
    print('\n=== resultados acumulados ===')
    print(pl.read_csv(LOG).select('experimento', 'n_features', 'wape_local'))
