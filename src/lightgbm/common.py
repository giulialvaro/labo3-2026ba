"""Utilidades compartidas por los experimentos LightGBM.

La competencia predice t+2 y evalua a nivel producto con WAPE/Total Error Rate.
Este modulo evita que cada runner implemente de forma distinta los cortes y la
metrica.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parents[2]


def find_data() -> Path:
    candidates = [
        os.environ.get('DATA_DIR'),
        ROOT / 'datasets',
        Path.cwd() / 'datasets',
        Path('~/buckets/b1/datasets').expanduser(),
    ]
    for path in candidates:
        if path and Path(path).is_dir():
            return Path(path)
    return ROOT / 'datasets'


DATA = find_data()
EXP = ROOT / 'exp'


def add_months(periodo: int, months: int) -> int:
    year, month = divmod(int(periodo), 100)
    idx = year * 12 + month - 1 + months
    return (idx // 12) * 100 + idx % 12 + 1


def wape(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    den = y.sum()
    return float(np.abs(y - p).sum() / den) if den > 0 else float('nan')


def product_wape(frame: pl.DataFrame, pred_col='pred_tn', real_col='target_tn') -> tuple[float, pl.DataFrame]:
    """Agrega predicciones cliente-producto y calcula la metrica real."""
    agg = (frame.group_by('product_id')
                .agg(pl.col(pred_col).sum().alias('tn'), pl.col(real_col).sum().alias('tn_real'))
                .sort('product_id'))
    return wape(agg['tn_real'], agg['tn']), agg


def prediction_grid_scale(real, pred, lo=0.30, hi=1.30, step=0.01) -> tuple[float, float]:
    """Calibracion global elegida exclusivamente sobre validacion."""
    scales = np.arange(lo, hi + step / 2, step)
    scores = np.array([wape(real, np.asarray(pred) * s) for s in scales])
    i = int(scores.argmin())
    return float(scales[i]), float(scores[i])


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')


def build_submission(pred: pl.DataFrame, history: pl.DataFrame, output: Path) -> pl.DataFrame:
    """Completa productos sin prediccion con promedio de los ultimos 12 meses."""
    requested = pl.read_csv(DATA / 'product_id_apredecir201912.txt', separator='\t')
    fallback = (history.filter(pl.col('periodo').is_between(201901, 201912))
                       .group_by('product_id').agg(pl.col('tn').mean().alias('fallback')))
    out = (requested.join(pred.select('product_id', 'tn'), on='product_id', how='left')
                    .join(fallback, on='product_id', how='left')
                    .with_columns(pl.coalesce('tn', 'fallback').fill_null(0.0).clip(0.0, None).alias('tn'))
                    .select('product_id', 'tn').sort('product_id'))
    output.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(output)
    return out
