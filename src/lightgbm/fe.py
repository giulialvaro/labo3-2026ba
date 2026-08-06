"""Feature Engineering modular para LightGBM (Clase 5).

build_features(v, cfg) arma la tabla segun los grupos prendidos en cfg.
Cada grupo es barato de extender: se agregan expresiones polars sobre '.over(clave)'.
NOTA leakage: los rolling son TRAILING (incluyen el mes actual t, que SI se conoce
al predecir t+2). El target es t+2, las features usan datos <= t. Sin fuga.
"""
import polars as pl

DATA = 'datasets'


def cargar(clave):
    d = pl.read_csv(f'{DATA}/sell-in.txt.gz', separator='\t')
    apre = pl.read_csv(f'{DATA}/product_id_apredecir201912.txt', separator='\t')
    v = (d.group_by(*clave, 'periodo').agg(pl.col('tn').sum().alias('tn'))
           .join(apre, on='product_id', how='inner')
           .with_columns(pl.col('periodo').cast(pl.Utf8).str.strptime(pl.Date, '%Y%m').alias('ds'))
           .sort(*clave, 'ds'))
    v = (v.upsample('ds', every='1mo', group_by=clave, maintain_order=True)
           .with_columns(pl.col('tn').fill_null(0.0))
           .with_columns([pl.col(k).fill_null(strategy='forward') for k in clave])
           .with_columns((pl.col('ds').dt.year() * 100 + pl.col('ds').dt.month()).alias('periodo')))
    return v


def build_features(v, cfg):
    """Devuelve (df, feats, cat_feats) segun cfg."""
    k = cfg['clave']
    df = v.clone()
    feats, cat_feats = [], []

    # --- A. LAGS ---
    if cfg.get('lags'):
        df = df.with_columns([pl.col('tn').shift(L).over(k).alias(f'lag_{L}') for L in cfg['lags']])
        feats += [f'lag_{L}' for L in cfg['lags']]

    # --- B. ROLLING (trailing, incluye t) ---
    for w in cfg.get('rolling_windows', []):
        df = df.with_columns([
            pl.col('tn').rolling_mean(w).over(k).alias(f'rmean_{w}'),
            pl.col('tn').rolling_std(w).over(k).alias(f'rstd_{w}'),
            pl.col('tn').rolling_max(w).over(k).alias(f'rmax_{w}'),
            pl.col('tn').rolling_min(w).over(k).alias(f'rmin_{w}'),
        ])
        feats += [f'rmean_{w}', f'rstd_{w}', f'rmax_{w}', f'rmin_{w}']

    # --- C. DELTAS / RATIOS ---
    if cfg.get('deltas'):
        exprs = [
            (pl.col('tn') - pl.col('tn').shift(1).over(k)).alias('delta_1'),
            (pl.col('tn') - pl.col('tn').shift(12).over(k)).alias('delta_12'),
            (pl.col('tn') / (pl.col('tn').shift(1).over(k) + 1e-6)).alias('ratio_1'),
            (pl.col('tn') / (pl.col('tn').shift(12).over(k) + 1e-6)).alias('ratio_12'),
        ]
        if 12 in cfg.get('rolling_windows', []):
            exprs.append((pl.col('tn') / (pl.col('rmean_12') + 1e-6)).alias('tn_vs_rmean12'))
        if {3, 12} <= set(cfg.get('rolling_windows', [])):
            exprs.append((pl.col('rmean_3') / (pl.col('rmean_12') + 1e-6)).alias('momentum_3_12'))
        df = df.with_columns(exprs)
        feats += [e.meta.output_name() for e in exprs]

    # --- D. FRECUENCIA / INTERMITENCIA ---
    if cfg.get('frecuencia'):
        df = df.with_columns([
            (pl.col('tn') == 0).cast(pl.Int8).alias('_es_cero'),
            pl.int_range(pl.len()).over(k).alias('_idx'),
        ])
        df = df.with_columns([
            pl.col('_es_cero').rolling_sum(12).over(k).alias('ceros_12'),
            # meses desde ultima venta: idx actual - idx de la ultima venta (>0)
            (pl.col('_idx') - pl.when(pl.col('tn') > 0).then(pl.col('_idx')).otherwise(None)
               .forward_fill().over(k)).alias('meses_desde_venta'),
        ])
        # OJO: 'antiguedad' (idx que crece) se saca -> los arboles no extrapolan (rompe la prediccion)
        feats += ['ceros_12', 'meses_desde_venta']

    # --- E. CALENDARIO ---
    if cfg.get('calendario'):
        df = df.with_columns((pl.col('periodo') % 100).cast(pl.Int32).alias('mes'))
        feats.append('mes'); cat_feats.append('mes')

    # target
    df = df.with_columns(pl.col('tn').shift(-cfg['target_lag']).over(k).alias('target'))
    return df, feats, cat_feats
