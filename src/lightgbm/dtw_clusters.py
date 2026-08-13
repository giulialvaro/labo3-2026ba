"""Clustering de series de tiempo con Dynamic Time Warping (7 clusters).

Agrupa los 780 productos por FORMA de su serie mensual (escalada / z-normalizada),
con distancia DTW + clustering jerarquico. Guarda datasets/product_clusters.csv (product_id, cluster).
Sin dependencias raras (numpy + scipy) -> anda en py3.14.

Hiperparametro: WINDOW (ventana Sakoe-Chiba del DTW). Correr: python3 src/lightgbm/dtw_clusters.py
"""
import polars as pl, numpy as np, os, time
from scipy.cluster.hierarchy import linkage, fcluster

WINDOW = 6          # ventana de tiempo del DTW (unico hiperparametro; probar 3/6/9)
N_CLUSTERS = 7


def _find_data():
    for d in [os.environ.get('DATA_DIR'), 'datasets', os.path.expanduser('~/buckets/b1/datasets')]:
        if d and os.path.isdir(d):
            return d
    return 'datasets'


DATA = _find_data()


def dtw(a, b, w):
    n, m = len(a), len(b)
    w = max(w, abs(n - m))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m, i + w) + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return np.sqrt(D[n, m])


def main():
    t = time.time()
    d = pl.read_csv(f'{DATA}/sell-in.txt.gz', separator='\t')
    apre = pl.read_csv(f'{DATA}/product_id_apredecir201912.txt', separator='\t')
    v = (d.group_by('product_id', 'periodo').agg(pl.col('tn').sum().alias('tn'))
           .join(apre, on='product_id', how='inner').sort('product_id', 'periodo'))
    # matriz productos x meses (pivot), relleno 0
    piv = v.pivot(values='tn', index='product_id', on='periodo').fill_null(0.0).sort('product_id')
    pids = piv['product_id'].to_list()
    M = piv.drop('product_id').to_numpy()
    # z-normalizo cada serie (DTW compara FORMA, no magnitud) -> "input escalado"
    mu = M.mean(axis=1, keepdims=True); sd = M.std(axis=1, keepdims=True) + 1e-9
    Z = (M - mu) / sd
    print(f'{len(pids)} productos x {M.shape[1]} meses. Calculando DTW (window={WINDOW})...')

    # matriz de distancias condensada (solo triangular superior)
    n = len(pids)
    cond = np.empty(n * (n - 1) // 2)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            cond[k] = dtw(Z[i], Z[j], WINDOW); k += 1
    print(f'DTW listo [{time.time()-t:.0f}s]. Clustering jerarquico...')

    Zlink = linkage(cond, method='average')
    labels = fcluster(Zlink, t=N_CLUSTERS, criterion='maxclust')
    out = pl.DataFrame({'product_id': pids, 'cluster': labels})
    out.write_csv(f'{DATA}/product_clusters.csv')
    print(f'guardado {DATA}/product_clusters.csv')
    print('tamaño de clusters:')
    print(out.group_by('cluster').len().sort('cluster'))
    print(f'TOTAL {time.time()-t:.0f}s')


if __name__ == '__main__':
    main()
