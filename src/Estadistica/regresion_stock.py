"""Fine tuning de la regresion ganadora incorporando tb_stocks.

Selecciona el grupo de stock en un fold anual comparable:
train ancla 201810 -> valida ancla 201910 (target 201912 conocido).
Luego entrena 201812 -> predice desde 201912 el target 202002.
"""
import os
import warnings

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')
DATA = 'datasets'
OUT = 'exp/ensamble'
os.makedirs(OUT, exist_ok=True)


def wape(y, p):
    return float(np.abs(y - p).sum() / y.sum())


def preparar():
    d = pl.read_csv(f'{DATA}/sell-in.txt.gz', separator='\t')
    apre = pl.read_csv(f'{DATA}/product_id_apredecir201912.txt', separator='\t')
    v = (d.group_by('product_id', 'periodo').agg(pl.col('tn').sum().alias('tn'))
           .join(apre, on='product_id', how='inner')
           .sort('product_id', 'periodo'))
    tb = v.with_columns([
        pl.col('tn').shift(i).over('product_id').alias(f'lag_{i}')
        for i in range(12)
    ] + [pl.col('tn').shift(-2).over('product_id').alias('target')])
    tb = tb.with_columns(
        pl.mean_horizontal([f'lag_{i}' for i in range(12)]).alias('mean12')
    )

    s = (pl.read_csv(f'{DATA}/tb_stocks.txt', separator='\t')
           .sort('product_id', 'periodo')
           .with_columns([
               pl.col('stock_final').shift(i).over('product_id').alias(f'stock_lag_{i}')
               for i in [1, 2]
           ]))
    tb = tb.join(s, on=['product_id', 'periodo'], how='left').sort('product_id', 'periodo')
    tb = tb.with_columns([
        (pl.col('stock_final') - pl.col('stock_lag_1')).alias('stock_delta_1'),
        (pl.col('stock_final') / (pl.col('mean12').abs() + 1e-3)).alias('stock_cobertura_12'),
        (pl.col('stock_final') < 0).cast(pl.Int8).alias('stock_negativo'),
    ])
    return v, apre, tb


def fit_predict(tb, train_periodo, pred_periodo, feats):
    tr = tb.filter(pl.col('periodo') == train_periodo).drop_nulls(feats + ['target'])
    va = tb.filter(pl.col('periodo') == pred_periodo).drop_nulls(feats)
    if not tr.height or not va.height:
        raise ValueError(f'sin filas completas: train={tr.height}, score={va.height}')
    model = LinearRegression().fit(tr.select(feats).to_numpy(), tr['target'].to_numpy())
    pred = np.clip(model.predict(va.select(feats).to_numpy()), 0, None)
    return va, pred


def main():
    v, apre, tb = preparar()
    lags = [f'lag_{i}' for i in range(12)]
    grupos = {
        'base': [],
        'stock_actual': ['stock_final'],
        'stock_cobertura': ['stock_cobertura_12'],
        'stock_actual_cobertura': ['stock_final', 'stock_cobertura_12'],
        'stock_historia': [
            'stock_final', 'stock_lag_1', 'stock_lag_2',
            'stock_delta_1', 'stock_cobertura_12', 'stock_negativo',
        ],
    }

    resultados = []
    for nombre, stock_feats in grupos.items():
        feats = lags + stock_feats
        try:
            va, pred = fit_predict(tb, 201810, 201910, feats)
        except ValueError as error:
            print(f'{nombre:25s} OMITIDO ({error})')
            continue
        score = wape(va['target'].to_numpy(), pred)
        resultados.append((nombre, score, stock_feats))
        print(f'{nombre:25s} WAPE={score:.4f} filas={va.height}')

    nombre, _, stock_feats = min(resultados, key=lambda x: x[1])
    feats = lags + stock_feats
    future, pred = fit_predict(tb, 201812, 201912, feats)
    out = future.select('product_id').with_columns(pl.Series('tn', pred))
    prom = (v.filter(pl.col('periodo').is_between(201901, 201912))
              .group_by('product_id').agg(pl.col('tn').mean().alias('tn_prom')))
    out = (apre.join(out, on='product_id', how='left')
               .join(prom, on='product_id', how='left')
               .with_columns(pl.coalesce('tn', 'tn_prom').fill_null(0.0).alias('tn'))
               .select('product_id', 'tn'))
    path = f'{OUT}/reg_stock_{nombre}.csv'
    out.write_csv(path)
    print(f'\nGanador: {nombre} ({stock_feats})')
    print(f'{path} | 780 productos | tn total={out["tn"].sum():.0f}')


if __name__ == '__main__':
    main()
