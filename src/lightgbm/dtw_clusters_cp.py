"""DTW clustering a nivel CLIENTE-PRODUCTO (con sampleo, como pide el profe).

- series = cada par (customer_id, product_id) x mes, z-normalizada
- SAMPLE de series activas -> DTW jerarquico -> 7 arquetipos (medoides)
- asigno TODAS las 260k series al medoide mas cercano (Euclidea z-norm, rapido)
- guarda datasets/cp_clusters.csv (customer_id, product_id, cluster)

Correr:  python3 src/lightgbm/dtw_clusters_cp.py
"""
import polars as pl, numpy as np, os, time
from scipy.cluster.hierarchy import linkage, fcluster

WINDOW = 6
N_CLUSTERS = 7
SAMPLE_SERIES = 800     # series activas para el DTW (samplear; DTW es O(N^2))
MIN_ACTIVOS = 3         # una serie es "activa" si tiene >= 3 meses con venta


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
            c = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return np.sqrt(D[n, m])


def main():
    t = time.time()
    d = pl.read_csv(f'{DATA}/sell-in-zeroes.txt.gz')
    v = d.group_by('customer_id', 'product_id', 'periodo').agg(pl.col('tn').sum().alias('tn'))
    piv = v.pivot(values='tn', index=['customer_id', 'product_id'], on='periodo').fill_null(0.0)
    keys = piv.select('customer_id', 'product_id')
    M = piv.drop('customer_id', 'product_id').to_numpy()
    activos = (M > 0).sum(axis=1)
    print(f'{M.shape[0]:,} series cliente-producto x {M.shape[1]} meses | activas(>= {MIN_ACTIVOS}): {(activos>=MIN_ACTIVOS).sum():,}  [{time.time()-t:.0f}s]')

    # z-normalizo (comparar FORMA); series inactivas quedan ~0
    mu = M.mean(axis=1, keepdims=True); sd = M.std(axis=1, keepdims=True) + 1e-9
    Z = (M - mu) / sd

    # sample de series ACTIVAS para el DTW
    idx_act = np.where(activos >= MIN_ACTIVOS)[0]
    rng = np.random.default_rng(1)
    samp = rng.choice(idx_act, size=min(SAMPLE_SERIES, len(idx_act)), replace=False)
    Zs = Z[samp]
    print(f'DTW sobre {len(samp)} series muestreadas (window={WINDOW})...')
    ns = len(samp); cond = np.empty(ns * (ns - 1) // 2); k = 0
    for i in range(ns):
        for j in range(i + 1, ns):
            cond[k] = dtw(Zs[i], Zs[j], WINDOW); k += 1
    labels_s = fcluster(linkage(cond, method='average'), t=N_CLUSTERS, criterion='maxclust')
    print(f'DTW + cluster listo [{time.time()-t:.0f}s]. Calculando medoides y asignando todas...')

    # medoide de cada cluster = serie del sample mas cercana al centro (media z-norm del cluster)
    medoids = []
    for c in range(1, N_CLUSTERS + 1):
        mem = Zs[labels_s == c]
        if len(mem) == 0:
            continue
        centro = mem.mean(axis=0)
        medoids.append(mem[np.argmin(((mem - centro) ** 2).sum(axis=1))])
    medoids = np.array(medoids)

    # asigno TODAS las series al medoide mas cercano (Euclidea z-norm, vectorizado y rapido)
    dists = ((Z[:, None, :] - medoids[None, :, :]) ** 2).sum(axis=2)
    cluster = dists.argmin(axis=1) + 1

    out = keys.with_columns(pl.Series('cluster', cluster))
    out.write_csv(f'{DATA}/cp_clusters.csv')
    print(f'guardado {DATA}/cp_clusters.csv')
    print(out.group_by('cluster').len().sort('cluster'))
    print(f'TOTAL {time.time()-t:.0f}s')


if __name__ == '__main__':
    main()
