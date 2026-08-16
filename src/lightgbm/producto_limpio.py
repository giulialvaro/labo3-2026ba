"""LightGBM limpio a nivel producto: control del challenger cliente-producto.

Valida principalmente diciembre-2018 -> febrero-2019 y usa octubre-2019 ->
diciembre-2019 como control temporal. Prueba bloques de FE incrementalmente,
en toneladas crudas y log, y genera la submission del ganador.

Uso:
    python3 src/lightgbm/producto_limpio.py
    python3 src/lightgbm/producto_limpio.py --validate-only
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
import common
import fe
import fe_cp_a_producto


OUT = common.EXP / 'lgbm_clean'
SEEDS = [1, 7, 19, 37, 71]
ANCHORS = [201812, 201910]  # febrero-2019 (principal) y diciembre-2019 (control)


@dataclass(frozen=True)
class Candidate:
    name: str
    cfg: dict
    target_scale: str


BASE_CFG = {
    'clave': ['product_id'], 'target_lag': 2,
    'lags': list(range(13)), 'rolling_windows': [],
    'deltas': False, 'frecuencia': False, 'calendario': False,
    'cosmos': False, 'producto_attrs': False, 'stocks': False,
}


def candidate(name, target_scale='raw', **changes):
    cfg = {**BASE_CFG, **changes}
    return Candidate(name, cfg, target_scale)


CANDIDATES = [
    candidate('lags_raw'),
    candidate('limpio_raw', rolling_windows=[3, 6, 12], deltas=True, cosmos=True),
    candidate('limpio_log', target_scale='log', rolling_windows=[3, 6, 12], deltas=True, cosmos=True),
    candidate('limpio_stock_raw', rolling_windows=[3, 6, 12], deltas=True, cosmos=True, stocks=True),
    candidate('cp_agregada_raw', cp_aggregates=True),
    candidate('cp_agregada_stock_raw', cp_aggregates=True, stocks=True),
]


def _fit_predict(train, score, feats, cat_feats, target_scale, seeds=SEEDS):
    y = train['target'].to_numpy()
    fit_y = np.log1p(y) if target_scale == 'log' else y
    x_train = train.select(feats).to_pandas()
    x_score = score.select(feats).to_pandas()
    cats = [c for c in cat_feats if c in feats]
    preds, imps = [], []
    for seed in seeds:
        model = lgb.LGBMRegressor(
            objective='regression_l1', n_estimators=700, learning_rate=0.025,
            num_leaves=31, max_depth=-1, min_child_samples=35,
            feature_fraction=0.75, bagging_fraction=0.85, bagging_freq=1,
            reg_alpha=0.05, reg_lambda=0.2,
            random_state=seed, verbose=-1, n_jobs=-1,
            deterministic=True, force_row_wise=True,
        )
        model.fit(x_train, fit_y, categorical_feature=cats if cats else 'auto')
        pred = model.predict(x_score)
        if target_scale == 'log':
            pred = np.expm1(pred)
        preds.append(np.clip(pred, 0, None))
        imps.append(model.feature_importances_)
    return np.mean(preds, axis=0), np.mean(imps, axis=0)


def validate(tb, feats, cats, candidate_, anchor):
    train_max = common.add_months(anchor, -2)
    # LightGBM maneja NaN de forma nativa: no tiramos historia ni productos.
    train = tb.filter((pl.col('periodo') <= train_max) & pl.col('target').is_not_null())
    val = tb.filter((pl.col('periodo') == anchor) & pl.col('target').is_not_null())
    pred, imp = _fit_predict(train, val, feats, cats, candidate_.target_scale)
    score = common.wape(val['target'], pred)
    vp = val.select('product_id', 'periodo', pl.col('target').alias('tn_real')).with_columns(pl.Series('tn', pred))
    return score, vp, imp, train.height, val.height


def main(validate_only=False):
    OUT.mkdir(parents=True, exist_ok=True)
    history = fe.cargar(['product_id'])
    rows = []
    prepared = {}
    cp_aggregates = cp_features = None

    for cand in CANDIDATES:
        tb, feats, cats = fe.build_features(history, cand.cfg)
        if cand.cfg.get('cp_aggregates'):
            if cp_aggregates is None:
                print('Construyendo features cliente-producto agregadas...', flush=True)
                cp_aggregates, cp_features = fe_cp_a_producto.build()
            tb = tb.join(cp_aggregates, on=['product_id', 'periodo'], how='left')
            feats += cp_features
        tb = tb.sort('product_id', 'periodo')
        prepared[cand.name] = (cand, tb, feats, cats)
        importance = None
        for anchor in ANCHORS:
            score, vp, importance, ntrain, nval = validate(tb, feats, cats, cand, anchor)
            rows.append({'candidate': cand.name, 'scale': cand.target_scale, 'anchor': anchor,
                         'wape': score, 'n_features': len(feats), 'train_rows': ntrain, 'val_rows': nval})
            vp.write_csv(OUT / f'val_{cand.name}_{anchor}.csv')
            print(f'{cand.name:18s} anchor={anchor} WAPE={score:.4f} train={ntrain:,} val={nval:,}')
        pl.DataFrame({'feature': feats, 'importance': importance}).sort('importance', descending=True).write_csv(
            OUT / f'imp_{cand.name}.csv')

    result = pl.DataFrame(rows)
    summary = (result.pivot(values='wape', index=['candidate', 'scale', 'n_features'], on='anchor')
                     .with_columns((pl.col('201812') * 0.7 + pl.col('201910') * 0.3).alias('score_seleccion'))
                     .sort('score_seleccion'))
    result.write_csv(OUT / 'validaciones.csv')
    summary.write_csv(OUT / 'ranking.csv')
    print('\nRANKING (70% febrero, 30% control diciembre)')
    print(summary)

    if validate_only:
        return

    winner_name = summary[0, 'candidate']
    cand, tb, feats, cats = prepared[winner_name]
    train = tb.filter((pl.col('periodo') <= 201910) & pl.col('target').is_not_null())
    future = tb.filter(pl.col('periodo') == 201912)
    pred, imp = _fit_predict(train, future, feats, cats, cand.target_scale)
    raw = future.select('product_id').with_columns(pl.Series('tn', pred))
    output = OUT / f'{winner_name}.csv'
    submission = common.build_submission(raw, history, output)
    pl.DataFrame({'feature': feats, 'importance': imp}).sort('importance', descending=True).write_csv(
        OUT / f'imp_final_{winner_name}.csv')
    common.write_json(OUT / 'seleccion.json', {
        'winner': winner_name, 'target_scale': cand.target_scale,
        'features': feats, 'anchors': ANCHORS, 'selection_rule': '0.7*WAPE_feb + 0.3*WAPE_dec',
        'submission': str(output.relative_to(common.ROOT)), 'tn_total': float(submission['tn'].sum()),
    })
    print(f'\nSubmission ganadora: {output} | tn total={submission["tn"].sum():.1f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    main(args.validate_only)
