"""Feature Engineering cliente-producto (competencia).

Dataset: datasets/sell-in-zeroes.txt.gz (17M filas, con ceros, generado por z601).
clave = (customer_id, product_id). Serie mensual por par.

Bloques implementados: A (escalado) + B (historia). Se agregan D/E/F/G/H despues.
Modo sample: cargar(sample_productos=N) para desarrollar rapido en el Mac.
"""
import polars as pl
import os


def _find_data():
    # local: datasets/  |  GCP: ~/buckets/b1/datasets  |  o via env DATA_DIR
    for d in [os.environ.get('DATA_DIR'), 'datasets', os.path.expanduser('~/buckets/b1/datasets')]:
        if d and os.path.isdir(d):
            return d
    return 'datasets'


DATA = _find_data()
CLAVE = ['customer_id', 'product_id']

DEF_CFG = {
    'escala_win': 12,          # ventana del promedio de escalado (mes actual + anteriores)
    'esc_floor': 1e-3,         # piso para no dividir por ~0
    'eps': 1e-6,
    'lags': list(range(0, 13)) + [18, 24],
    'rolling_windows': [3, 6, 9, 12, 24],
    'target_lag': 2,
    'cosmos': True,            # D: agregaciones producto/cliente/cat + shares + momentum
    'precio': True,            # F: precios cuidados + fill rate
    'buy_history': True,       # C: historia binaria de compra / intermitencia
    'stocks': True,            # stock conocido a nivel producto en el mes ancla
    'cluster_version': 'v2',   # v2 | legacy | None
}


def cargar(sample_productos=None, seed=1):
    df = pl.read_csv(f'{DATA}/sell-in-zeroes.txt.gz')  # comma + header
    # agregacion defensiva (por si hubiera duplicados por combo/mes)
    df = (df.group_by('customer_id', 'product_id', 'periodo')
            .agg(pl.col('tn').sum(),
                 pl.col('plan_precios_cuidados').max(),
                 pl.col('cust_request_tn').sum()))
    if sample_productos is not None:
        prods = (df.select('product_id').unique().sort('product_id')
                   .sample(sample_productos, seed=seed).get_column('product_id').to_list())
        df = df.filter(pl.col('product_id').is_in(prods))
    return df.sort(*CLAVE, 'periodo')


def build_features(df, cfg=None):
    cfg = {**DEF_CFG, **(cfg or {})}
    k = CLAVE
    eps = cfg['eps']
    feats = []

    # ---------- A. ESCALADO (tecnica Rosario) ----------
    # promedio de nivel: media movil (actual + anteriores), SIN futuro. Con piso para no dividir por 0.
    df = df.with_columns(
        pl.col('tn').rolling_mean(cfg['escala_win'], min_samples=1).over(k).alias('promedio_nivel')
    ).with_columns(
        pl.max_horizontal('promedio_nivel', pl.lit(cfg['esc_floor'])).alias('promedio_nivel')
    ).with_columns(
        (pl.col('tn') / pl.col('promedio_nivel')).alias('tn_esc')
    )
    feats.append('promedio_nivel')   # el nivel tambien es feature
    # Targets para hurdle: compra y cantidad escalada. Al predecir se des-escala.
    df = df.with_columns(
        pl.col('tn').shift(-cfg['target_lag']).over(k).alias('target_tn')
    ).with_columns([
        (pl.col('target_tn') / pl.col('promedio_nivel')).alias('target'),
        pl.when(pl.col('target_tn').is_null()).then(None)
          .otherwise((pl.col('target_tn') > 0).cast(pl.Int8)).alias('target_buy'),
    ])

    # ---------- B. HISTORIA (sobre la serie ESCALADA tn_esc) ----------
    df = df.with_columns([pl.col('tn_esc').shift(L).over(k).alias(f'lag_{L}') for L in cfg['lags']])
    feats += [f'lag_{L}' for L in cfg['lags']]

    for w in cfg['rolling_windows']:
        df = df.with_columns([
            pl.col('tn_esc').rolling_mean(w).over(k).alias(f'rmean_{w}'),
            pl.col('tn_esc').rolling_std(w).over(k).alias(f'rstd_{w}'),
            pl.col('tn_esc').rolling_min(w).over(k).alias(f'rmin_{w}'),
            pl.col('tn_esc').rolling_max(w).over(k).alias(f'rmax_{w}'),
            pl.col('tn_esc').rolling_median(w).over(k).alias(f'rmed_{w}'),
        ])
        feats += [f'rmean_{w}', f'rstd_{w}', f'rmin_{w}', f'rmax_{w}', f'rmed_{w}']

    # deltas, delta de lags, tendencia, ewma
    df = df.with_columns([
        (pl.col('lag_0') - pl.col('lag_1')).alias('delta_1'),
        (pl.col('lag_0') - pl.col('lag_12')).alias('delta_12'),
        (pl.col('lag_1') - pl.col('lag_2')).alias('dlag_1_2'),
        (pl.col('lag_1') - pl.col('lag_3')).alias('dlag_1_3'),
        ((pl.col('lag_0') - pl.col('lag_5')) / 5).alias('slope_5'),
        (pl.col('lag_0') / (pl.col('lag_1') + eps)).alias('ratio_1'),
        (pl.col('lag_0') / (pl.col('lag_12') + eps)).alias('ratio_12'),
        pl.col('tn_esc').ewm_mean(span=3).over(k).alias('ewm_3'),
        pl.col('tn_esc').ewm_mean(span=6).over(k).alias('ewm_6'),
    ])
    feats += ['delta_1', 'delta_12', 'dlag_1_2', 'dlag_1_3', 'slope_5',
              'ratio_1', 'ratio_12', 'ewm_3', 'ewm_6']

    # ---------- C. COMPRA / INTERMITENCIA ----------
    if cfg.get('buy_history'):
        df = df.with_columns([
            (pl.col('tn') > 0).cast(pl.Int8).alias('_buy'),
            pl.int_range(pl.len()).over(k).alias('_idx'),
        ])
        df = df.with_columns([
            pl.col('_buy').shift(L).over(k).alias(f'buy_lag_{L}') for L in [0, 1, 2, 3, 6, 12]
        ] + [
            pl.col('_buy').rolling_sum(w, min_samples=1).over(k).alias(f'buys_{w}') for w in [3, 6, 12]
        ])
        df = df.with_columns(
            (pl.col('_idx') - pl.when(pl.col('_buy') == 1).then(pl.col('_idx')).otherwise(None)
             .forward_fill().over(k)).alias('months_since_buy')
        )
        feats += [f'buy_lag_{L}' for L in [0, 1, 2, 3, 6, 12]]
        feats += [f'buys_{w}' for w in [3, 6, 12]] + ['months_since_buy']

    # ---------- D. COSMOS / AGREGACIONES (producto, cliente, categoria) ----------
    if cfg.get('cosmos'):
        prod = (pl.read_csv(f'{DATA}/tb_productos.txt', separator='\t')
                  .unique(subset='product_id').select('product_id', 'cat1', 'cat2', 'cat3', 'brand', 'sku_size'))
        df = df.join(prod, on='product_id', how='left')
        # totales por periodo (mismo mes = sin leakage)
        tot_prod = df.group_by('product_id', 'periodo').agg(pl.col('tn').sum().alias('tot_prod')).sort('product_id', 'periodo')
        tot_cli = df.group_by('customer_id', 'periodo').agg(pl.col('tn').sum().alias('tot_cli')).sort('customer_id', 'periodo')
        tot_cat3 = df.group_by('cat3', 'periodo').agg(pl.col('tn').sum().alias('tot_cat3'))
        tot_uni = df.group_by('periodo').agg(pl.col('tn').sum().alias('tot_uni')).sort('periodo')
        # momentum YoY (ESTACIONARIO, no el nivel absoluto que cae)
        tot_prod = tot_prod.with_columns((pl.col('tot_prod') / (pl.col('tot_prod').shift(12).over('product_id') + eps)).alias('yoy_prod'))
        tot_cli = tot_cli.with_columns((pl.col('tot_cli') / (pl.col('tot_cli').shift(12).over('customer_id') + eps)).alias('yoy_cli'))
        tot_uni = tot_uni.with_columns((pl.col('tot_uni') / (pl.col('tot_uni').shift(12) + eps)).alias('yoy_uni'))
        df = (df.join(tot_prod, on=['product_id', 'periodo'], how='left')
                .join(tot_cli, on=['customer_id', 'periodo'], how='left')
                .join(tot_cat3, on=['cat3', 'periodo'], how='left')
                .join(tot_uni, on='periodo', how='left'))
        # shares (mi participacion, estacionario) — NO uso los niveles absolutos
        df = df.with_columns([
            (pl.col('tn') / (pl.col('tot_prod') + eps)).alias('share_prod'),
            (pl.col('tn') / (pl.col('tot_cli') + eps)).alias('share_cli'),
            (pl.col('tn') / (pl.col('tot_cat3') + eps)).alias('share_cat3'),
        ])
        feats += ['share_prod', 'share_cli', 'share_cat3', 'yoy_prod', 'yoy_cli', 'yoy_uni']

    # ---------- F. PRECIO / DEMANDA ----------
    if cfg.get('precio'):
        df = df.sort(*k, 'periodo')   # re-ordeno antes de los shifts (los joins de D reordenan)
        df = df.with_columns([
            pl.col('plan_precios_cuidados').cast(pl.Int8).alias('ppc'),
            (pl.col('tn') / (pl.col('cust_request_tn') + eps)).alias('fill_rate'),
        ])
        df = df.with_columns([pl.col('fill_rate').shift(L).over(k).alias(f'fill_rate_lag{L}') for L in [1, 2, 3]])
        feats += ['ppc', 'fill_rate', 'fill_rate_lag1', 'fill_rate_lag2', 'fill_rate_lag3']

    # ---------- STOCK (producto x mes; disponible en t, target t+2) ----------
    if cfg.get('stocks'):
        stock = (pl.read_csv(f'{DATA}/tb_stocks.txt', separator='\t')
                   .select('product_id', 'periodo', 'stock_final')
                   .unique(subset=['product_id', 'periodo']))
        df = df.join(stock, on=['product_id', 'periodo'], how='left').sort(*k, 'periodo')
        df = df.with_columns([
            pl.col('stock_final').is_not_null().cast(pl.Int8).alias('stock_available'),
            pl.col('stock_final').fill_null(0.0).alias('stock_0'),
            (pl.col('stock_final') < 0).fill_null(False).cast(pl.Int8).alias('stock_negative'),
        ])
        df = df.with_columns([
            pl.col('stock_0').shift(L).over(k).alias(f'stock_lag_{L}') for L in [1, 2, 3, 6, 12]
        ])
        stock_exprs = [
            (pl.col('stock_0') - pl.col('stock_lag_1')).alias('stock_delta_1'),
            (pl.col('stock_0') - pl.col('stock_lag_12')).alias('stock_delta_12'),
        ]
        if 'tot_prod' in df.columns:
            stock_exprs.append((pl.col('stock_0') / (pl.col('tot_prod') + eps)).alias('stock_vs_prod'))
        df = df.with_columns(stock_exprs)
        feats += ['stock_available', 'stock_0', 'stock_negative']
        feats += [f'stock_lag_{L}' for L in [1, 2, 3, 6, 12]]
        feats += [x.meta.output_name() for x in stock_exprs]

    # ---------- DTW: v2 no reutiliza accidentalmente el clustering viejo ----------
    cluster_version = cfg.get('cluster_version')
    if cluster_version == 'v2':
        cp_path = f'{DATA}/cp_clusters_v2.csv'
        if os.path.exists(cp_path):
            cl = (pl.read_csv(cp_path)
                    .with_columns(pl.col('customer_id', 'product_id').cast(pl.Int64))
                    .select('customer_id', 'product_id', 'regime', 'shape_cluster', 'cluster_v2'))
            df = df.join(cl, on=['customer_id', 'product_id'], how='left')
            feats += ['regime', 'shape_cluster', 'cluster_v2']
    elif cluster_version == 'legacy':
        cp_path = f'{DATA}/cp_clusters.csv'
        if os.path.exists(cp_path):
            cl = pl.read_csv(cp_path).with_columns(pl.col('customer_id', 'product_id').cast(pl.Int64))
            df = df.join(cl, on=['customer_id', 'product_id'], how='left')
            feats.append('cluster')

    return df, feats


if __name__ == '__main__':
    import time
    t = time.time()
    df = cargar(sample_productos=30)
    print(f'sample: {df.height:,} filas, {df.select("product_id").n_unique()} productos, '
          f'{df.select(CLAVE).n_unique():,} series cliente-producto  [{time.time()-t:.1f}s]')
    tb, feats = build_features(df)
    print(f'features generadas: {len(feats)}')
    print(feats)
