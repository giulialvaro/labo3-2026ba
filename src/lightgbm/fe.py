"""Feature Engineering modular para LightGBM (Clase 5).

build_features(v, cfg) arma la tabla segun los grupos prendidos en cfg.
Cada grupo es barato de extender: se agregan expresiones polars sobre '.over(clave)'.
NOTA leakage: los rolling son TRAILING (incluyen el mes actual t, que SI se conoce
al predecir t+2). El target es t+2, las features usan datos <= t. Sin fuga.
"""
import os

import polars as pl


def _find_data():
    """Encuentra datasets tanto en el repo local como en la VM de GCP."""
    repo_data = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets'))
    for path in [os.environ.get('DATA_DIR'), repo_data, 'datasets', os.path.expanduser('~/buckets/b1/datasets')]:
        if path and os.path.isdir(path):
            return path
    return repo_data


DATA = _find_data()


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

    # --- F/G. COSMOS + ATRIBUTOS DE PRODUCTO (necesita tb_productos) ---
    if cfg.get('cosmos') or cfg.get('producto_attrs'):
        prod = (pl.read_csv(f'{DATA}/tb_productos.txt', separator='\t')
                  .unique(subset='product_id')
                  .select('product_id', 'cat1', 'cat2', 'cat3', 'brand', 'sku_size'))
        df = df.join(prod, on='product_id', how='left')

    if cfg.get('cosmos'):
        uni = df.group_by('periodo').agg(pl.col('tn').sum().alias('_universo_tn'))
        c3 = df.group_by('cat3', 'periodo').agg(pl.col('tn').sum().alias('_cat3_tn'))
        df = df.join(uni, on='periodo', how='left').join(c3, on=['cat3', 'periodo'], how='left').sort(*k, 'ds')
        # SOLO features estacionarias (shares y momentum YoY), NO niveles absolutos (que caen -> extrapolacion)
        df = df.with_columns([
            (pl.col('tn') / (pl.col('_universo_tn') + 1e-6)).alias('share_universo'),
            (pl.col('tn') / (pl.col('_cat3_tn') + 1e-6)).alias('share_cat3'),
            (pl.col('_universo_tn') / (pl.col('_universo_tn').shift(12).over(k) + 1e-6)).alias('universo_yoy'),
            (pl.col('_cat3_tn') / (pl.col('_cat3_tn').shift(12).over(k) + 1e-6)).alias('cat3_yoy'),
        ])
        feats += ['share_universo', 'share_cat3', 'universo_yoy', 'cat3_yoy']

    if cfg.get('producto_attrs'):
        df = df.with_columns([
            pl.col('cat1').cast(pl.Categorical).to_physical().alias('cat1'),
            pl.col('cat2').cast(pl.Categorical).to_physical().alias('cat2'),
            pl.col('cat3').cast(pl.Categorical).to_physical().alias('cat3_cat'),
            pl.col('brand').cast(pl.Categorical).to_physical().alias('brand'),
            pl.col('sku_size').cast(pl.Float64, strict=False).alias('sku_size'),
        ])
        feats += ['cat1', 'cat2', 'cat3_cat', 'brand', 'sku_size']
        cat_feats += ['cat1', 'cat2', 'cat3_cat', 'brand']

    # --- H. STOCKS (conocidos hasta el mes ancla t; target = t+2) ---
    if cfg.get('stocks'):
        stocks = (pl.read_csv(f'{DATA}/tb_stocks.txt', separator='\t')
                    .select('product_id', 'periodo', 'stock_final')
                    .unique(subset=['product_id', 'periodo']))
        df = df.join(stocks, on=['product_id', 'periodo'], how='left').sort(*k, 'ds')

        # tb_stocks comienza en 201810. La ausencia previa es falta de fuente,
        # no stock cero: conservamos ese dato en stock_disponible.
        df = df.with_columns([
            pl.col('stock_final').is_not_null().cast(pl.Int8).alias('stock_disponible'),
            (pl.col('stock_final') < 0).fill_null(False).cast(pl.Int8).alias('stock_negativo'),
            pl.col('stock_final').fill_null(0.0).alias('stock_0'),
        ])
        df = df.with_columns([
            pl.col('stock_0').shift(L).over(k).alias(f'stock_lag_{L}')
            for L in [1, 2, 3, 6, 12]
        ])
        df = df.with_columns([
            (pl.col('stock_0') - pl.col('stock_lag_1')).alias('stock_delta_1'),
            (pl.col('stock_0') - pl.col('stock_lag_12')).alias('stock_delta_12'),
            (pl.col('stock_0') / (pl.col('tn').abs() + 1e-3)).alias('stock_vs_tn'),
        ])
        stock_feats = [
            'stock_disponible', 'stock_negativo', 'stock_0',
            'stock_lag_1', 'stock_lag_2', 'stock_lag_3', 'stock_lag_6', 'stock_lag_12',
            'stock_delta_1', 'stock_delta_12', 'stock_vs_tn',
        ]
        if 'rmean_3' in df.columns:
            df = df.with_columns(
                (pl.col('stock_0') / (pl.col('rmean_3').abs() + 1e-3)).alias('stock_cobertura_3')
            )
            stock_feats.append('stock_cobertura_3')
        if 'rmean_12' in df.columns:
            df = df.with_columns(
                (pl.col('stock_0') / (pl.col('rmean_12').abs() + 1e-3)).alias('stock_cobertura_12')
            )
            stock_feats.append('stock_cobertura_12')
        feats += stock_feats

    # target
    df = df.with_columns(pl.col('tn').shift(-cfg['target_lag']).over(k).alias('target'))
    return df, feats, cat_feats
