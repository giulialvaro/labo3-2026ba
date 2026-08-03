"""Tarea 2 - AutoARIMA (statsforecast) sobre los 780 productos REALES -> submit Kaggle.
Config elegida en el mundo ideal: AutoARIMA(season_length=12) con target en log.
Fallback: productos donde ARIMA falla -> promedio 12m.
Correr:  python3 src/Estadistica/tarea2_arima_real.py
"""
import polars as pl, numpy as np, os, sys, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, 'src')
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

DATA = 'datasets'


def main():
    os.makedirs('exp/arima', exist_ok=True)
    d = pl.read_csv(f'{DATA}/sell-in.txt.gz', separator='\t')
    apre = pl.read_csv(f'{DATA}/product_id_apredecir201912.txt', separator='\t')
    v = (d.group_by('product_id', 'periodo').agg(pl.col('tn').sum().alias('tn'))
           .join(apre, on='product_id', how='inner')
           .with_columns(pl.col('periodo').cast(pl.Utf8).str.strptime(pl.Date, '%Y%m').alias('ds'))
           .sort('product_id', 'ds'))

    # relleno meses faltantes con 0 (ARIMA necesita serie regular)
    v = (v.select('product_id', 'ds', 'tn')
           .upsample('ds', every='1mo', group_by='product_id', maintain_order=True)
           .with_columns(pl.col('tn').fill_null(0.0), pl.col('product_id').fill_null(strategy='forward')))

    # fallback promedio 12m (2019)
    prom = (v.filter(pl.col('ds') >= pl.date(2019, 1, 1))
              .group_by('product_id').agg(pl.col('tn').mean().alias('tn_prom')))

    # a formato statsforecast, con log
    sfdf = v.select(pl.col('product_id').alias('unique_id'), 'ds',
                    pl.col('tn').clip(lower_bound=0).log1p().alias('y')).to_pandas()

    t = time.time()
    sf = StatsForecast(models=[AutoARIMA(season_length=12)], freq='MS', n_jobs=-1)
    fc = sf.forecast(df=sfdf, h=2)
    print(f'ARIMA corrido en {time.time()-t:.0f}s')

    fc2 = fc.sort_values(['unique_id', 'ds']).groupby('unique_id').tail(1)  # feb-2020
    pred = (pl.from_pandas(fc2[['unique_id', 'AutoARIMA']])
              .rename({'unique_id': 'product_id', 'AutoARIMA': 'tn'})
              .with_columns((pl.col('tn').exp() - 1).alias('tn')))  # deshago el log

    # ensamblo con fallback y limpio negativos/nulos
    out = (apre.join(pred, on='product_id', how='left')
               .join(prom, on='product_id', how='left')
               .with_columns(pl.coalesce('tn', 'tn_prom').alias('tn'))
               .with_columns(pl.when(pl.col('tn') < 0).then(0).otherwise(pl.col('tn')).alias('tn'))
               .select('product_id', 'tn'))

    nulos = out['tn'].is_null().sum()
    out = out.with_columns(pl.col('tn').fill_null(0.0))
    out.write_csv('exp/arima/autoarima_log.csv')
    print(f'archivo: exp/arima/autoarima_log.csv  |  {out.height} productos  |  nulos rellenados: {nulos}  |  tn total={out["tn"].sum():.0f}')


if __name__ == '__main__':
    main()
