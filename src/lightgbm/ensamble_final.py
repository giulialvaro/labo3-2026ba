"""Ensambla el linreg exacto de 0.231 con un candidato LightGBM.

Los blends viejos usaron una reconstruccion distinta de la regresion. Este
script apunta por defecto al archivo exacto generado por z403.

Uso:
    python3 src/lightgbm/ensamble_final.py --candidate exp/lgbm_clean/lags_raw.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
import common


def load(path: Path, suffix: str) -> pl.DataFrame:
    frame = pl.read_csv(path).select('product_id', pl.col('tn').cast(pl.Float64).alias(f'tn_{suffix}'))
    if frame['product_id'].n_unique() != frame.height:
        raise ValueError(f'product_id duplicado en {path}')
    return frame


def main(args):
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    baseline = load(baseline_path, 'linreg')
    candidate = load(candidate_path, 'lgbm')
    joined = baseline.join(candidate, on='product_id', how='inner').sort('product_id')
    if joined.height != 780:
        raise ValueError(f'Se esperaban 780 productos y hay {joined.height}')

    x, y = joined['tn_linreg'].to_numpy(), joined['tn_lgbm'].to_numpy()
    corr = float(np.corrcoef(x, y)[0, 1])
    disagreement = float(np.abs(x - y).sum() / x.sum())
    outdir = common.EXP / 'ensamble_final'
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for weight in args.weights:
        pred = (1 - weight) * x + weight * y
        name = f'blend_linreg{round((1-weight)*100):02d}_lgbm{round(weight*100):02d}.csv'
        frame = joined.select('product_id').with_columns(pl.Series('tn', np.clip(pred, 0, None)))
        frame.write_csv(outdir / name)
        outputs.append({'lgbm_weight': weight, 'file': str((outdir / name).relative_to(common.ROOT)),
                        'tn_total': float(pred.sum())})
        print(f'{name}: tn total={pred.sum():.1f}')

    common.write_json(outdir / 'manifest.json', {
        'baseline': str(baseline_path), 'candidate': str(candidate_path),
        'correlation': corr, 'absolute_disagreement_over_baseline_total': disagreement,
        'outputs': outputs,
    })
    print(f'correlacion={corr:.5f} | desacuerdo relativo={disagreement:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', default='src/Estadistica/linreg.csv')
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--weights', type=float, nargs='+', default=[0.10, 0.20, 0.30])
    main(parser.parse_args())
