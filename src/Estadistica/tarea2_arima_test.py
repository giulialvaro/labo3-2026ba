"""Tarea 2 - test de AutoARIMA (statsforecast) en el MUNDO IDEAL.
Compara baseline vs log para elegir la mejor config antes de correr los 780 reales.
Correr:  python3 src/Estadistica/tarea2_arima_test.py
"""
import polars as pl, numpy as np, sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, 'src')
from metrica import total_error_rate
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA


def to_sf(df):
    return (df.with_columns(pl.col('periodo').cast(pl.Utf8).str.strptime(pl.Date, '%Y%m').alias('ds'))
              .select(pl.col('product_id').alias('unique_id'), 'ds', pl.col('tn').alias('y'))
              .to_pandas())


def run_arima(dfp, use_log):
    d = dfp.copy()
    if use_log:
        d['y'] = np.log1p(d['y'].clip(lower=0))
    sf = StatsForecast(models=[AutoARIMA(season_length=12)], freq='MS', n_jobs=-1)
    fc = sf.forecast(df=d, h=2)
    fc2 = fc.sort_values(['unique_id', 'ds']).groupby('unique_id').tail(1)
    pred = pl.from_pandas(fc2[['unique_id', 'AutoARIMA']]).rename({'unique_id': 'product_id', 'AutoARIMA': 'tn'})
    if use_log:
        pred = pred.with_columns((pl.col('tn').exp() - 1).alias('tn'))
    return pred.with_columns(pl.when(pl.col('tn') < 0).then(0).otherwise(pl.col('tn')).alias('tn'))


if __name__ == '__main__':
    vi = pl.read_csv('datasets/tb_ventas_ideal.csv')
    ri = pl.read_csv('datasets/tb_realidad_ideal.csv')
    base = to_sf(vi)
    t = time.time(); p1 = run_arima(base, False); print(f'AutoARIMA (sin log) = {total_error_rate(p1, ri):.4f}   [{time.time()-t:.0f}s]')
    t = time.time(); p2 = run_arima(base, True);  print(f'AutoARIMA (con log) = {total_error_rate(p2, ri):.4f}   [{time.time()-t:.0f}s]')
    print('referencia: naif mismo-mes = 0.4079')
