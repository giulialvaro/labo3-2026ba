"""Feature Engineering cliente-producto (competencia).

Dataset: datasets/sell-in-zeroes.txt.gz (17M filas, con ceros, generado por z601).
clave = (customer_id, product_id). Serie mensual por par.

Bloques implementados: A (escalado) + B (historia). Se agregan D/E/F/G/H despues.
Modo sample: cargar(sample_productos=N) para desarrollar rapido en el Mac.
"""
import polars as pl

DATA = 'datasets'
CLAVE = ['customer_id', 'product_id']

DEF_CFG = {
    'escala_win': 12,          # ventana del promedio de escalado (mes actual + anteriores)
    'esc_floor': 1e-3,         # piso para no dividir por ~0
    'lags': list(range(0, 13)) + [18, 24],
    'rolling_windows': [3, 6, 9, 12, 24],
    'target_lag': 2,
}


def cargar(sample_productos=None, seed=1):
    df = pl.read_csv(f'{DATA}/sell-in-zeroes.txt.gz')  # comma + header
    # agregacion defensiva (por si hubiera duplicados por combo/mes)
    df = (df.group_by('customer_id', 'product_id', 'periodo')
            .agg(pl.col('tn').sum(),
                 pl.col('plan_precios_cuidados').max(),
                 pl.col('cust_request_tn').sum()))
    if sample_productos is not None:
        prods = df.select('product_id').unique().sample(sample_productos, seed=seed).get_column('product_id').to_list()
        df = df.filter(pl.col('product_id').is_in(prods))
    return df.sort(*CLAVE, 'periodo')


def build_features(df, cfg=None):
    cfg = {**DEF_CFG, **(cfg or {})}
    k = CLAVE
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
    # target escalado: tn(t+2) / promedio_nivel   (al predecir, des-escalar x promedio_nivel)
    df = df.with_columns(
        (pl.col('tn').shift(-cfg['target_lag']).over(k) / pl.col('promedio_nivel')).alias('target')
    )

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
        pl.col('tn_esc').ewm_mean(span=3).over(k).alias('ewm_3'),
        pl.col('tn_esc').ewm_mean(span=6).over(k).alias('ewm_6'),
    ])
    feats += ['delta_1', 'delta_12', 'dlag_1_2', 'dlag_1_3', 'slope_5', 'ewm_3', 'ewm_6']

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
