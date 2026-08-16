"""LightGBM producto con FE completa + tb_stocks.

Mantiene el mismo modelo/config que lgbm_rico para aislar el aporte del stock.
Genera submission e importance en exp/ensamble/.
"""
import os
import sys
import warnings

import lightgbm as lgb
import numpy as np
import polars as pl

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import fe


CFG = {
    'clave': ['product_id'], 'target_lag': 2, 'log_target': True,
    'lags': list(range(0, 13)), 'rolling_windows': [3, 6, 9, 12],
    'deltas': True, 'frecuencia': True, 'calendario': True,
    'cosmos': True, 'producto_attrs': True, 'stocks': True, 'eps': 1e-6,
}


def main():
    os.makedirs('exp/ensamble', exist_ok=True)
    v = fe.cargar(CFG['clave'])
    tb, feats, cat_feats = fe.build_features(v, CFG)
    tb = tb.sort('product_id', 'periodo')

    full = tb.filter(
        (pl.col('periodo') <= 201910) & pl.col('target').is_not_null()
    ).drop_nulls(feats)
    pred = tb.filter(pl.col('periodo') == 201912).drop_nulls(feats)
    y = np.log1p(full['target'].to_numpy())
    cats = [c for c in cat_feats if c in feats]

    predictions = []
    importances = []
    for seed in range(1, 8):
        model = lgb.LGBMRegressor(
            objective='regression_l1', n_estimators=700, learning_rate=0.02,
            num_leaves=63, feature_fraction=0.6, bagging_fraction=0.8,
            bagging_freq=1, min_child_samples=40, random_state=seed,
            verbose=-1,
        )
        model.fit(
            full.select(feats).to_pandas(), y,
            categorical_feature=cats if cats else 'auto',
        )
        predictions.append(np.clip(
            np.expm1(model.predict(pred.select(feats).to_pandas())), 0, None
        ))
        importances.append(model.feature_importances_)

    pp = np.mean(predictions, axis=0)
    out = pred.select('product_id').with_columns(pl.Series('tn', pp))
    apre = pl.read_csv(f'{fe.DATA}/product_id_apredecir201912.txt', separator='\t')
    prom = (v.filter(pl.col('periodo').is_between(201901, 201912))
              .group_by('product_id').agg(pl.col('tn').mean().alias('tn_prom')))
    out = (apre.join(out, on='product_id', how='left')
               .join(prom, on='product_id', how='left')
               .with_columns(pl.coalesce('tn', 'tn_prom').fill_null(0.0).alias('tn'))
               .select('product_id', 'tn'))
    out.write_csv('exp/ensamble/lgbm_stock.csv')

    imp = (pl.DataFrame({
        'feature': feats,
        'importance': np.mean(importances, axis=0),
    }).sort('importance', descending=True))
    imp.write_csv('exp/ensamble/imp_lgbm_stock.csv')

    print(f'{len(feats)} features | train={full.height} | pred={pred.height}')
    print(f'tn total={out["tn"].sum():.0f} -> exp/ensamble/lgbm_stock.csv')
    print('top 15 importance:')
    print(imp.head(15))
    print('\nfeatures de stock:')
    print(imp.filter(pl.col('feature').str.starts_with('stock')))


if __name__ == '__main__':
    main()
