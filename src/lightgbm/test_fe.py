"""Tests del FE cliente-producto.

1) Correctitud: serie chica hecha a mano -> valores esperados de escalado/lags/target.
2) Leakage (la regla de oro): borro el futuro y las features NO deben cambiar.

Correr:  python3 src/lightgbm/test_fe.py
"""
import polars as pl
import fe_cp


def test_correctitud():
    # una serie: tn = 10,20,30,40,50,60
    df = pl.DataFrame({
        'customer_id': [1] * 6, 'product_id': [1] * 6,
        'periodo': [201701, 201702, 201703, 201704, 201705, 201706],
        'plan_precios_cuidados': [0] * 6, 'cust_request_tn': [0.0] * 6,
        'tn': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    }).sort('customer_id', 'product_id', 'periodo')
    tb, _ = fe_cp.build_features(df)
    tb = tb.sort('periodo')

    # promedio_nivel = media expansiva (win=12, min_samples=1): 10,15,20,25,30,35
    prom = tb['promedio_nivel'].to_list()
    assert prom == [10, 15, 20, 25, 30, 35], f'promedio_nivel mal: {prom}'

    # tn_esc = tn / promedio
    esc = [round(x, 4) for x in tb['tn_esc'].to_list()]
    esp = [round(t / p, 4) for t, p in zip([10, 20, 30, 40, 50, 60], prom)]
    assert esc == esp, f'tn_esc mal: {esc} vs {esp}'

    # lag_1 de tn_esc = [null, esc0, esc1, ...]
    lag1 = tb['lag_1'].to_list()
    assert lag1[0] is None and round(lag1[1], 4) == esc[0], f'lag_1 mal: {lag1}'

    # target = tn(t+2)/promedio: 30/10, 40/15, 50/20, 60/25, null, null
    tgt = tb['target'].to_list()
    esp_t = [3.0, 40 / 15, 50 / 20, 60 / 25, None, None]
    for a, b in zip(tgt[:4], esp_t[:4]):
        assert abs(a - b) < 1e-9, f'target mal: {tgt} vs {esp_t}'
    assert tgt[4] is None and tgt[5] is None, f'target futuro no es null: {tgt}'
    print('  OK correctitud (escalado, lags, target)')


def test_leakage():
    df = fe_cp.cargar(sample_productos=20)
    m = 201810   # mes ancla a chequear (hay futuro despues)
    feats_cols = fe_cp.build_features(df)[1]

    full = fe_cp.build_features(df)[0].filter(pl.col('periodo') == m)
    # borro TODO el futuro (> m) y recalculo
    trunc = fe_cp.build_features(df.filter(pl.col('periodo') <= m))[0].filter(pl.col('periodo') == m)

    full = full.sort(*fe_cp.CLAVE)
    trunc = trunc.sort(*fe_cp.CLAVE)
    assert full.height == trunc.height, 'distinta cantidad de filas'

    malas = []
    for c in feats_cols:              # OJO: NO chequeo 'target' (usa futuro a proposito)
        a = full[c].fill_null(-999.0).to_numpy()
        b = trunc[c].fill_null(-999.0).to_numpy()
        import numpy as np
        if not np.allclose(a, b, atol=1e-9, equal_nan=True):
            malas.append(c)
    assert not malas, f'LEAKAGE en features: {malas}'
    print(f'  OK sin leakage ({len(feats_cols)} features, chequeadas en mes {m})')


if __name__ == '__main__':
    print('test_correctitud:'); test_correctitud()
    print('test_leakage:'); test_leakage()
    print('\nTODOS LOS TESTS PASARON ✔')
