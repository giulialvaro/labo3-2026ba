"""LightGBM cliente-producto: Tweedie y hurdle (compra x cantidad).

El score siempre se calcula DESPUES de sumar clientes a producto, igual que
Kaggle. Por seguridad el default usa 100 productos; el dataset completo se
activa explicitamente con ``--sample-products 0``.

Desarrollo local:
    python3 src/lightgbm/cliente_producto_hurdle.py --sample-products 100

Corrida final GCP (global + DTW como features):
    python3 -u src/lightgbm/cliente_producto_hurdle.py \
      --sample-products 0 --variants hurdle_cluster --final

Comparacion completa (costosa):
    python3 -u src/lightgbm/cliente_producto_hurdle.py \
      --sample-products 0 --variants one_stage,hurdle,hurdle_cluster,hurdle_per_cluster
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
import common
import fe_cp


OUT = common.EXP / 'lgbm_cp_clean'
TARGET_CLIP = 50.0
CLUSTER_MODEL_FEATURES = [f'cluster_v2_is_{i}' for i in range(7)]
CLUSTER_FEATURES = {'regime', 'shape_cluster', 'cluster_v2', 'cluster', *CLUSTER_MODEL_FEATURES}

# Mismo mes contra mismo mes: replica el esquema que hizo fuerte a z403.
VALIDATIONS = [
    {'name': 'febrero', 'train_anchor': 201712, 'score_anchor': 201812,
     'train_map': 'cp_clusters_v2_fit201712.csv',
     'score_map': 'cp_clusters_v2_score201812_from201712.csv'},
    {'name': 'diciembre', 'train_anchor': 201710, 'score_anchor': 201810,
     'train_map': 'cp_clusters_v2_fit201710.csv',
     'score_map': 'cp_clusters_v2_score201810_from201710.csv'},
]


def frame32(df: pl.DataFrame, features: list[str]) -> pd.DataFrame:
    out = df.select(features).to_pandas()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors='coerce').astype('float32')
    return out


def model_params(kind, seed):
    base = dict(
        n_estimators=550, learning_rate=0.035, num_leaves=63,
        min_child_samples=150, feature_fraction=0.75,
        bagging_fraction=0.85, bagging_freq=1,
        reg_alpha=0.05, reg_lambda=0.3,
        random_state=seed, verbose=-1, n_jobs=-1,
        deterministic=True, force_row_wise=True,
    )
    if kind == 'binary':
        return {**base, 'objective': 'binary', 'max_bin': 255}
    return {**base, 'objective': 'tweedie', 'tweedie_variance_power': 1.2, 'max_bin': 1230}


@dataclass
class Prediction:
    pred_scaled: np.ndarray
    probability: np.ndarray
    importance: np.ndarray


def fit_global(train, score, features, variant, seeds):
    x_train = frame32(train, features)
    x_score = frame32(score, features)
    preds, probs, imps = [], [], []

    for seed in seeds:
        if variant == 'one_stage':
            y = np.clip(train['target'].to_numpy(), 0, TARGET_CLIP)
            model = lgb.LGBMRegressor(**model_params('tweedie', seed))
            model.fit(x_train, y)
            preds.append(np.clip(model.predict(x_score), 0, None))
            probs.append(np.ones(score.height))
            imps.append(model.feature_importances_)
            continue

        classifier = lgb.LGBMClassifier(**model_params('binary', seed))
        classifier.fit(x_train, train['target_buy'].to_numpy())
        probability = classifier.predict_proba(x_score)[:, 1]

        positive = train.filter(pl.col('target_buy') == 1)
        quantity = lgb.LGBMRegressor(**model_params('tweedie', seed + 1000))
        quantity.fit(frame32(positive, features), np.clip(positive['target'].to_numpy(), 1e-8, TARGET_CLIP))
        conditional = np.clip(quantity.predict(x_score), 0, None)
        preds.append(probability * conditional)
        probs.append(probability)
        imps.append((classifier.feature_importances_ + quantity.feature_importances_) / 2)

    return Prediction(np.mean(preds, axis=0), np.mean(probs, axis=0), np.mean(imps, axis=0))


def fit_per_cluster(train, score, features, seeds, min_rows, min_positive):
    """Global como fallback; reemplaza con modelos especializados donde alcanza la muestra."""
    base = fit_global(train, score, features, 'hurdle', seeds)
    pred = base.pred_scaled.copy()
    prob = base.probability.copy()
    cluster_values = score['cluster_v2'].drop_nulls().unique().sort().to_list()
    for cluster in cluster_values:
        tr = train.filter(pl.col('cluster_v2') == cluster)
        mask = score['cluster_v2'].to_numpy() == cluster
        sc = score.filter(pl.Series(mask))
        positives = int(tr['target_buy'].sum()) if tr.height else 0
        if tr.height < min_rows or positives < min_positive or not sc.height:
            print(f'  cluster {cluster}: fallback global (train={tr.height:,}, positivos={positives:,})')
            continue
        specialized = fit_global(tr, sc, features, 'hurdle', seeds)
        pred[mask] = specialized.pred_scaled
        prob[mask] = specialized.probability
        print(f'  cluster {cluster}: modelo propio (train={tr.height:,}, positivos={positives:,})')
    return Prediction(pred, prob, base.importance)


def predict_variant(train, score, all_features, variant, seeds, min_rows, min_positive):
    use_cluster = variant in {'hurdle_cluster', 'hurdle_per_cluster'}
    features = [f for f in all_features if use_cluster or f not in CLUSTER_FEATURES]
    if variant == 'hurdle_per_cluster':
        if 'cluster_v2' not in score.columns:
            raise ValueError('DTW v2 no disponible: falta datasets/cp_clusters_v2.csv')
        result = fit_per_cluster(train, score, features, seeds, min_rows, min_positive)
    else:
        base_variant = 'one_stage' if variant == 'one_stage' else 'hurdle'
        result = fit_global(train, score, features, base_variant, seeds)
    return result, features


def attach_cluster_map(frame, filename):
    path = common.DATA / filename
    if not path.exists():
        raise FileNotFoundError(f'Falta {path}. Generar los snapshots DTW antes de correr clusters.')
    mapping = (pl.read_csv(path)
                 .with_columns(pl.col('customer_id', 'product_id').cast(pl.Int64))
                 .select('customer_id', 'product_id', 'regime', 'shape_cluster', 'cluster_v2'))
    joined = frame.join(mapping, on=['customer_id', 'product_id'], how='left')
    # Los IDs de forma no tienen orden natural: one-hot evita tratar 3<4<5 como magnitud.
    return joined.with_columns([
        (pl.col('cluster_v2') == cluster).fill_null(False).cast(pl.Int8).alias(f'cluster_v2_is_{cluster}')
        for cluster in range(7)
    ])


def evaluate(tb, all_features, variant, spec, seeds, min_rows, min_positive):
    train = tb.filter((pl.col('periodo') == spec['train_anchor']) & pl.col('target').is_not_null())
    val = tb.filter((pl.col('periodo') == spec['score_anchor']) & pl.col('target').is_not_null())
    if variant in {'hurdle_cluster', 'hurdle_per_cluster'}:
        train = attach_cluster_map(train, spec['train_map'])
        val = attach_cluster_map(val, spec['score_map'])
        features_for_run = all_features + CLUSTER_MODEL_FEATURES
    else:
        features_for_run = all_features
    fitted, features = predict_variant(train, val, features_for_run, variant, seeds, min_rows, min_positive)
    pred_tn = fitted.pred_scaled * val['promedio_nivel'].to_numpy()
    scored = val.select('product_id', 'target_tn').with_columns([
        pl.Series('pred_tn', pred_tn), pl.Series('probability', fitted.probability),
    ])
    raw_wape, product = common.product_wape(scored)
    scale, calibrated_wape = common.prediction_grid_scale(product['tn_real'], product['tn'])
    product = product.with_columns((pl.col('tn') * scale).alias('tn_calibrated'))
    return {
        'variant': variant, 'validation': spec['name'],
        'train_anchor': spec['train_anchor'], 'score_anchor': spec['score_anchor'], 'wape': raw_wape,
        'calibration': scale, 'wape_calibrated': calibrated_wape,
        'train_rows': train.height, 'val_rows': val.height,
        'positive_rate': float(train['target_buy'].mean()), 'n_features': len(features),
    }, product, fitted.importance, features


def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    sample = None if args.sample_products == 0 else args.sample_products
    seeds = [int(x) for x in args.seeds.split(',')]
    variants = [x.strip() for x in args.variants.split(',') if x.strip()]
    started = time.time()
    source = fe_cp.cargar(sample_productos=sample, seed=args.seed)
    print(f'Dataset: {source.height:,} filas | {source.select(fe_cp.CLAVE).n_unique():,} series', flush=True)
    table, all_features = fe_cp.build_features(source, {'cluster_version': None})
    table = table.sort(*fe_cp.CLAVE, 'periodo')
    print(f'FE: {len(all_features)} features [{time.time()-started:.1f}s]', flush=True)

    results, calibrations = [], {}
    last_features, last_importance = {}, {}
    for variant in variants:
        calibrations[variant] = []
        for spec in VALIDATIONS:
            print(f"\n{variant} | {spec['name']} {spec['train_anchor']} -> {spec['score_anchor']}", flush=True)
            row, product, importance, features = evaluate(
                table, all_features, variant, spec, seeds, args.min_cluster_rows, args.min_cluster_positive)
            results.append(row)
            calibrations[variant].append(row['calibration'])
            last_features[variant], last_importance[variant] = features, importance
            product.write_csv(OUT / f"val_{variant}_{spec['name']}.csv")
            print(f"  WAPE={row['wape']:.4f} | calibrado={row['wape_calibrated']:.4f} x{row['calibration']:.2f}", flush=True)
        pl.DataFrame({'feature': last_features[variant], 'importance': last_importance[variant]}).sort(
            'importance', descending=True).write_csv(OUT / f'imp_{variant}.csv')

    result = pl.DataFrame(results)
    ranking = (result.pivot(values='wape', index='variant', on='validation')
                     .with_columns((pl.col('febrero') * 0.7 + pl.col('diciembre') * 0.3).alias('score_seleccion'))
                     .sort('score_seleccion'))
    result.write_csv(OUT / 'validaciones.csv')
    ranking.write_csv(OUT / 'ranking.csv')
    print('\nRANKING')
    print(ranking)

    if not args.final:
        return

    winner = ranking[0, 'variant']
    # Final estacional: diciembre-2018 -> febrero-2019 entrena; diciembre-2019 predice.
    train = table.filter((pl.col('periodo') == 201812) & pl.col('target').is_not_null())
    future = table.filter(pl.col('periodo') == 201912)
    final_features = all_features
    if winner in {'hurdle_cluster', 'hurdle_per_cluster'}:
        train = attach_cluster_map(train, 'cp_clusters_v2_fit201812.csv')
        future = attach_cluster_map(future, 'cp_clusters_v2_score201912_from201812.csv')
        final_features = all_features + CLUSTER_MODEL_FEATURES
    fitted, features = predict_variant(
        train, future, final_features, winner, seeds, args.min_cluster_rows, args.min_cluster_positive)
    calibration = float(np.median(calibrations[winner]))
    pred_cp = fitted.pred_scaled * future['promedio_nivel'].to_numpy() * calibration
    pred_product = (future.select('product_id').with_columns(pl.Series('pred_tn', pred_cp))
                          .group_by('product_id').agg(pl.col('pred_tn').sum().alias('tn')))
    product_history = source.group_by('product_id', 'periodo').agg(pl.col('tn').sum().alias('tn'))
    output = OUT / f'{winner}.csv'
    submission = common.build_submission(pred_product, product_history, output)
    common.write_json(OUT / 'seleccion.json', {
        'winner': winner, 'features': features, 'seeds': seeds,
        'sample_products': args.sample_products, 'calibration': calibration,
        'submission': str(output.relative_to(common.ROOT)),
        'tn_total': float(submission['tn'].sum()),
    })
    print(f'\nSubmission: {output} | tn total={submission["tn"].sum():.1f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-products', type=int, default=100,
                        help='0 usa las 17M filas; default seguro=100 productos')
    parser.add_argument('--variants', default='one_stage,hurdle,hurdle_cluster,hurdle_per_cluster')
    parser.add_argument('--seeds', default='1')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--min-cluster-rows', type=int, default=500)
    parser.add_argument('--min-cluster-positive', type=int, default=50)
    parser.add_argument('--final', action='store_true')
    main(parser.parse_args())
