"""Features cliente-producto agregadas al nivel PRODUCTO evaluado por Kaggle.

Permite aprovechar los 17M registros sin entrenar contra millones de targets
puntuales cuyo error no coincide con la metrica final.
"""
from __future__ import annotations

import os
import sys

import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
import common


CORE = [
    'cp_pairs', 'cp_buyers', 'cp_buyer_rate', 'cp_tn_per_buyer',
    'cp_request_tn', 'cp_fill_rate', 'cp_hhi', 'cp_top_share',
]
LIFECYCLE = [
    'cp_new_buyers', 'cp_repeat_buyers', 'cp_churned',
    'cp_regime0_share', 'cp_regime1_share', 'cp_regime2_share', 'cp_regime3_share',
]


def build():
    requested = pl.read_csv(common.DATA / 'product_id_apredecir201912.txt', separator='\t')
    data = (pl.read_csv(common.DATA / 'sell-in-zeroes.txt.gz')
              .select('customer_id', 'product_id', 'periodo', 'tn', 'cust_request_tn')
              .join(requested, on='product_id', how='inner')
              .sort('customer_id', 'product_id', 'periodo'))
    key = ['customer_id', 'product_id']
    data = data.with_columns((pl.col('tn') > 0).cast(pl.Int8).alias('_buy'))
    data = data.with_columns([
        pl.col('_buy').shift(1).over(key).fill_null(0).alias('_buy_lag1'),
        pl.col('_buy').cum_sum().over(key).alias('_buy_cum'),
    ])
    data = data.with_columns(
        pl.col('_buy_lag1').rolling_sum(12, min_samples=1).over(key).alias('_prior12')
    ).with_columns([
        ((pl.col('_buy') == 1) & (pl.col('_prior12') == 0)).cast(pl.Int8).alias('_new'),
        ((pl.col('_buy') == 1) & (pl.col('_prior12') > 0)).cast(pl.Int8).alias('_repeat'),
        ((pl.col('_buy') == 0) & (pl.col('_buy_lag1') == 1)).cast(pl.Int8).alias('_churn'),
        pl.when(pl.col('_buy_cum') == 0).then(0)
          .when(pl.col('_buy_cum') <= 2).then(1)
          .when(pl.col('_buy_cum') <= 5).then(2).otherwise(3).cast(pl.Int8).alias('_regime'),
        (pl.col('tn') ** 2).alias('_tn_sq'),
    ])

    agg = (data.group_by('product_id', 'periodo').agg([
        pl.len().alias('cp_pairs'),
        pl.col('_buy').sum().alias('cp_buyers'),
        pl.col('tn').sum().alias('_tn_total'),
        pl.col('tn').max().alias('_tn_max'),
        pl.col('_tn_sq').sum().alias('_tn_sq_sum'),
        pl.col('cust_request_tn').sum().alias('cp_request_tn'),
        pl.col('_new').sum().alias('cp_new_buyers'),
        pl.col('_repeat').sum().alias('cp_repeat_buyers'),
        pl.col('_churn').sum().alias('cp_churned'),
        *[(pl.col('_regime') == r).sum().alias(f'_regime{r}_count') for r in range(4)],
    ]).with_columns([
        (pl.col('cp_buyers') / pl.col('cp_pairs').clip(1, None)).alias('cp_buyer_rate'),
        (pl.col('_tn_total') / pl.col('cp_buyers').clip(1, None)).alias('cp_tn_per_buyer'),
        (pl.col('_tn_total') / (pl.col('cp_request_tn') + 1e-6)).alias('cp_fill_rate'),
        (pl.col('_tn_sq_sum') / (pl.col('_tn_total') ** 2 + 1e-6)).alias('cp_hhi'),
        (pl.col('_tn_max') / (pl.col('_tn_total') + 1e-6)).alias('cp_top_share'),
        *[(pl.col(f'_regime{r}_count') / pl.col('cp_pairs').clip(1, None)).alias(f'cp_regime{r}_share')
          for r in range(4)],
    ]).sort('product_id', 'periodo'))

    # Historia de estructura comercial: core 0/1/3/12; lifecycle actual y YoY.
    expressions, features = [], []
    for col in CORE:
        for lag in [0, 1, 3, 12]:
            name = f'{col}_lag{lag}'
            expressions.append(pl.col(col).shift(lag).over('product_id').alias(name))
            features.append(name)
    for col in LIFECYCLE:
        for lag in [0, 12]:
            name = f'{col}_lag{lag}'
            expressions.append(pl.col(col).shift(lag).over('product_id').alias(name))
            features.append(name)
    out = agg.with_columns(expressions).select('product_id', 'periodo', *features)
    return out, features


if __name__ == '__main__':
    frame, cols = build()
    print(frame.shape, len(cols), cols)
