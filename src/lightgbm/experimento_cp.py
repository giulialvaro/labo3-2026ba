"""Runner LightGBM cliente-producto (para GCP / PC 64GB).

- FE cliente-producto (fe_cp) con target ESCALADO
- LightGBM objective='tweedie' + max_bin=1230 (sin log)
- prediccion escalada -> DES-ESCALAR (x promedio_nivel) -> COLLAPSE a producto (sumo clientes)
- validacion walk-forward + WAPE a nivel PRODUCTO (la metrica real)
- genera submission 202002 y loguea

Correr en GCP (dataset completo):   python3 src/lightgbm/experimento_cp.py
Para probar local rapido:           SAMPLE=100 python3 src/lightgbm/experimento_cp.py
"""
import polars as pl, numpy as np, lightgbm as lgb, os, time, datetime
import fe_cp

SAMPLE = int(os.environ.get('SAMPLE', '0')) or None   # None = dataset completo (GCP)
TRAIN_MAX = 201907
VAL_ANCLAS = [201908, 201909, 201910]                 # targets 201910/11/12
PRED_ANCLA = 201912                                   # target 202002
TARGET_CLIP = 50.0                                    # cap del target escalado (evita blowups por promedio ~0)

LGB = dict(objective='tweedie', tweedie_variance_power=1.2, max_bin=1230,
           n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=100,
           feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
           verbose=-1, random_state=1, n_jobs=-1)


def wape(y, p):
    return float(np.abs(y - p).sum() / y.sum())


def con_target_periodo(df, col='periodo', out='tperiodo', k=2):
    y = pl.col(col) // 100
    m = pl.col(col) % 100 + k
    return df.with_columns(pl.when(m > 12).then((y + 1) * 100 + (m - 12)).otherwise(y * 100 + m).alias(out))


def collapse_wape(val_pred, prod_real):
    """val_pred: filas cliente-producto con pred_tn y ancla 'periodo'. Sumo a producto y comparo."""
    v = con_target_periodo(val_pred)
    pp = v.group_by('product_id', 'tperiodo').agg(pl.col('pred_tn').sum().alias('pred_prod'))
    j = pp.join(prod_real, left_on=['product_id', 'tperiodo'], right_on=['product_id', 'periodo'], how='inner')
    return wape(j['tn_prod'].to_numpy(), j['pred_prod'].to_numpy())


def main():
    os.makedirs('exp/lgbm_cp', exist_ok=True)
    t0 = time.time()
    df = fe_cp.cargar(sample_productos=SAMPLE)
    print(f'cargado: {df.height:,} filas | {df.select(fe_cp.CLAVE).n_unique():,} series  [{time.time()-t0:.0f}s]')

    tb, feats = fe_cp.build_features(df)
    tb = tb.sort(*fe_cp.CLAVE, 'periodo')
    # ventas reales a nivel producto (para WAPE y submission fallback)
    prod_real = df.group_by('product_id', 'periodo').agg(pl.col('tn').sum().alias('tn_prod'))

    # splits (NO dropeo nulls: LightGBM maneja NaN nativo -> conservo todas las series)
    train = tb.filter((pl.col('periodo') <= TRAIN_MAX) & pl.col('target').is_not_null())
    val = tb.filter(pl.col('periodo').is_in(VAL_ANCLAS))
    pred = tb.filter(pl.col('periodo') == PRED_ANCLA)
    print(f'train={train.height:,}  val={val.height:,}  pred={pred.height:,}  feats={len(feats)}')

    ytr = np.clip(train['target'].to_numpy(), 0, TARGET_CLIP)
    Xtr = train.select(feats).to_pandas()
    t = time.time()
    model = lgb.LGBMRegressor(**LGB)
    model.fit(Xtr, ytr)
    print(f'LightGBM entrenado [{time.time()-t:.0f}s]')

    # --- validacion: predigo escalado -> des-escalo -> collapse -> WAPE ---
    val_pred = val.with_columns(
        pl.Series('pred_esc', np.clip(model.predict(val.select(feats).to_pandas()), 0, None))
    ).with_columns((pl.col('pred_esc') * pl.col('promedio_nivel')).alias('pred_tn'))
    w = collapse_wape(val_pred.select('customer_id', 'product_id', 'periodo', 'pred_tn'), prod_real)
    print(f'\n>>> WAPE a nivel PRODUCTO (walk-forward) = {w:.4f}   (ref producto: naif~0.25, blend~0.248)')

    # feature importance
    imp = pl.DataFrame({'feature': feats, 'importance': model.feature_importances_}).sort('importance', descending=True)
    imp.write_csv('exp/lgbm_cp/imp_cp.csv')
    print('top10:', ' | '.join(imp.head(10)['feature'].to_list()))

    # --- submission 202002 ---
    pr = pred.with_columns(
        pl.Series('pred_esc', np.clip(model.predict(pred.select(feats).to_pandas()), 0, None))
    ).with_columns((pl.col('pred_esc') * pl.col('promedio_nivel')).alias('pred_tn'))
    sub = pr.group_by('product_id').agg(pl.col('pred_tn').sum().alias('tn'))
    apre = pl.read_csv(f'{fe_cp.DATA}/product_id_apredecir201912.txt', separator='\t')
    prom12 = prod_real.filter(pl.col('periodo').is_between(201901, 201912)).group_by('product_id').agg(pl.col('tn_prod').mean().alias('tn_fb'))
    out = (apre.join(sub, on='product_id', how='left').join(prom12, on='product_id', how='left')
              .with_columns(pl.coalesce('tn', 'tn_fb').fill_null(0.0).alias('tn')).select('product_id', 'tn'))
    out.write_csv('exp/lgbm_cp/lgbm_cp.csv')
    print(f'submission -> exp/lgbm_cp/lgbm_cp.csv ({out.height} productos, tn total={out["tn"].sum():.0f})')

    # log
    LOG = 'exp/lgbm_cp/resultados.csv'
    fila = pl.DataFrame({'fecha': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M')],
                         'sample': [str(SAMPLE)], 'n_features': [len(feats)], 'wape_producto': [round(w, 4)]})
    if os.path.exists(LOG):
        fila = pl.concat([pl.read_csv(LOG), fila], how='diagonal_relaxed')
    fila.write_csv(LOG)
    print(f'\nTOTAL {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
