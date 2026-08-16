"""DTW v2 para series cliente-producto.

Corrige cuatro problemas del primer intento:
1. las series sin historia no se fuerzan dentro de un cluster DTW;
2. se preserva la ventana de existencia (null fuera, cero dentro);
3. la muestra activa se estratifica por frecuencia y volumen;
4. la asignacion al medoide usa DTW, la misma metrica del clustering.

Con ``active_clusters=4`` se obtienen siete segmentos finales:
0 nunca compro, 1 compro 1-2 veces, 2 compro 3-5 veces, 3-6 formas DTW.

Uso final:
    python3 src/lightgbm/dtw_clusters_cp_v2.py

Pruebas de hiperparametros:
    python3 src/lightgbm/dtw_clusters_cp_v2.py --window 2 --active-clusters 4 --suffix w2k4
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, os.path.dirname(__file__))
import common


try:
    from dtaidistance import dtw as fast_dtw
    if not fast_dtw.try_import_c():
        fast_dtw = None
except ImportError:  # la prueba sintetica puede correr sin el extra
    fast_dtw = None


def dtw_python(a: np.ndarray, b: np.ndarray, window: int) -> float:
    n, m = len(a), len(b)
    window = max(window, abs(n - m))
    prev = np.full(m + 1, np.inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr = np.full(m + 1, np.inf)
        for j in range(max(1, i - window), min(m, i + window) + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            curr[j] = cost + min(curr[j - 1], prev[j], prev[j - 1])
        prev = curr
    return float(np.sqrt(prev[m]))


def distance(a: np.ndarray, b: np.ndarray, window: int) -> float:
    a = np.asarray(a, dtype=np.double, order='C')
    b = np.asarray(b, dtype=np.double, order='C')
    if fast_dtw is not None:
        return float(fast_dtw.distance_fast(a, b, window=window, use_pruning=True))
    return dtw_python(a, b, window)


def transform_sequence(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Escala forma y conserva solo la vida observada del par."""
    where = np.flatnonzero(observed)
    if not len(where):
        return np.zeros(1, dtype=np.double)
    seq = values[where[0]:where[-1] + 1].astype(np.double, copy=True)
    positive = seq[seq > 0]
    level = positive.mean() if len(positive) else 1.0
    seq = np.log1p(seq / max(level, 1e-9))
    sd = seq.std()
    return (seq - seq.mean()) / sd if sd > 1e-9 else np.zeros_like(seq)


def stratified_sample(active_idx, positive_months, total_tn, sample_size, seed):
    """Cobertura pareja de series frecuentes/infrecuentes y chicas/grandes."""
    rng = np.random.default_rng(seed)
    idx = np.asarray(active_idx)
    freq_bin = np.digitize(positive_months[idx], [8, 12, 18])
    logvol = np.log1p(total_tn[idx])
    cuts = np.unique(np.quantile(logvol, [0.25, 0.50, 0.75]))
    vol_bin = np.digitize(logvol, cuts)
    strata = freq_bin * 4 + vol_bin
    groups = [idx[strata == s] for s in np.unique(strata)]
    quota = max(1, sample_size // len(groups))
    chosen = []
    for group in groups:
        chosen.extend(rng.choice(group, min(quota, len(group)), replace=False).tolist())
    chosen = np.array(chosen, dtype=int)
    remaining = np.setdiff1d(idx, chosen, assume_unique=False)
    if len(chosen) < sample_size and len(remaining):
        extra = rng.choice(remaining, min(sample_size - len(chosen), len(remaining)), replace=False)
        chosen = np.concatenate([chosen, extra])
    return rng.permutation(chosen[:sample_size])


def condensed_distances(sequences, window):
    n = len(sequences)
    result = np.empty(n * (n - 1) // 2, dtype=np.float64)
    pos = 0
    for i in range(n):
        for j in range(i + 1, n):
            result[pos] = distance(sequences[i], sequences[j], window)
            pos += 1
    return result


def exact_medoids(sample_indices, labels, condensed, sequences):
    full = squareform(condensed)
    medoid_indices, medoid_sequences = [], []
    for cluster in sorted(np.unique(labels)):
        members = np.flatnonzero(labels == cluster)
        local = full[np.ix_(members, members)]
        winner = members[int(local.sum(axis=1).argmin())]
        medoid_indices.append(int(sample_indices[winner]))
        medoid_sequences.append(sequences[winner])
    return medoid_indices, medoid_sequences


def assign_active(active_idx, matrix, observed, medoids, window, workers):
    active_idx = np.asarray(active_idx, dtype=int)
    chunks = [x for x in np.array_split(active_idx, max(1, workers * 8)) if len(x)]

    def assign_chunk(chunk):
        out = np.empty(len(chunk), dtype=np.int16)
        for pos, row in enumerate(chunk):
            seq = transform_sequence(matrix[row], observed[row])
            ds = [distance(seq, medoid, window) for medoid in medoids]
            out[pos] = int(np.argmin(ds)) + 1
        return chunk, out

    labels = np.empty(len(matrix), dtype=np.int16)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for chunk, assigned in pool.map(assign_chunk, chunks):
            labels[chunk] = assigned
    return labels


def main(args):
    if fast_dtw is None and not args.allow_slow:
        raise SystemExit('Falta dtaidistance. Instalar: pip install dtaidistance (o usar --allow-slow).')

    started = time.time()
    source = common.DATA / 'sell-in-zeroes.txt.gz'
    data = pl.read_csv(source).select('customer_id', 'product_id', 'periodo', 'tn')
    if args.fit_until:
        data = data.filter(pl.col('periodo') <= args.fit_until)
    periods = sorted(data['periodo'].unique().to_list())

    stats = (data.group_by('customer_id', 'product_id').agg(
        (pl.col('tn') > 0).sum().alias('positive_months'),
        pl.col('tn').sum().alias('total_tn'),
        pl.len().alias('observed_months'),
    ).sort('customer_id', 'product_id'))
    pivot = (data.pivot(values='tn', index=['customer_id', 'product_id'], on='periodo', aggregate_function='sum')
                 .sort('customer_id', 'product_id'))
    period_cols = [str(p) for p in periods]
    raw = pivot.select(period_cols).to_numpy()
    observed = ~np.isnan(raw)
    matrix = np.nan_to_num(raw, nan=0.0).astype(np.float32)
    positive = stats['positive_months'].to_numpy()
    total = stats['total_tn'].to_numpy()

    regime = np.select([positive == 0, positive <= 2, positive <= 5], [0, 1, 2], default=3).astype(np.int8)
    active_idx = np.flatnonzero(regime == 3)
    sample_idx, medoid_idx = np.array([], dtype=int), []
    if args.medoids_file:
        saved = np.load(args.medoids_file)
        medoids = [saved[key] for key in sorted(saved.files)]
        print(f'{len(matrix):,} series | activas DTW={len(active_idx):,} | '
              f'medoides fijos={args.medoids_file}', flush=True)
    else:
        sample_idx = stratified_sample(active_idx, positive, total, min(args.sample_series, len(active_idx)), args.seed)
        sample_sequences = [transform_sequence(matrix[i], observed[i]) for i in sample_idx]
        print(f'{len(matrix):,} series | activas DTW={len(active_idx):,} | sample={len(sample_idx):,}', flush=True)
        print(f'Calculando DTW sample window={args.window}...', flush=True)
        condensed = condensed_distances(sample_sequences, args.window)
        labels = fcluster(linkage(condensed, method='average'), t=args.active_clusters, criterion='maxclust')
        medoid_idx, medoids = exact_medoids(sample_idx, labels, condensed, sample_sequences)
    print(f'Asignando {len(active_idx):,} activas a {len(medoids)} medoides con DTW real...', flush=True)
    shape_cluster = assign_active(active_idx, matrix, observed, medoids, args.window, args.workers)

    final_cluster = regime.astype(np.int16)
    final_cluster[active_idx] = shape_cluster[active_idx] + 2
    shape_out = np.zeros(len(matrix), dtype=np.int16)
    shape_out[active_idx] = shape_cluster[active_idx]
    names = np.array(['nunca', 'una_dos', 'tres_cinco', 'activa'])[regime]

    out = stats.select('customer_id', 'product_id', 'positive_months', 'total_tn', 'observed_months').with_columns([
        pl.Series('regime', regime), pl.Series('regime_name', names),
        pl.Series('shape_cluster', shape_out), pl.Series('cluster_v2', final_cluster),
    ])
    suffix = f'_{args.suffix}' if args.suffix else ''
    output = common.DATA / f'cp_clusters_v2{suffix}.csv'
    out.write_csv(output)
    profile = (out.group_by('cluster_v2', 'regime_name').agg(
        pl.len().alias('series'), pl.col('positive_months').mean().alias('positive_mean'),
        pl.col('total_tn').sum().alias('tn_total'))
        .with_columns((pl.col('tn_total') / pl.col('tn_total').sum()).alias('tn_share'))
        .sort('cluster_v2'))
    exp = common.EXP / 'dtw_v2'
    exp.mkdir(parents=True, exist_ok=True)
    profile.write_csv(exp / f'profile{suffix}.csv')
    if not args.medoids_file:
        np.savez(exp / f'medoids{suffix}.npz', **{f'medoid_{i+1}': x for i, x in enumerate(medoids)})
    common.write_json(exp / f'metadata{suffix}.json', {
        'source': str(source), 'output': str(output), 'fit_until': args.fit_until,
        'window': args.window, 'active_clusters': args.active_clusters,
        'total_final_clusters': int(len(np.unique(final_cluster))),
        'sample_series': len(sample_idx), 'seed': args.seed,
        'active_definition': 'positive_months >= 6',
        'medoid_source_rows': medoid_idx,
        'medoids_file': args.medoids_file,
        'assignment_metric': 'DTW', 'seconds': round(time.time() - started, 1),
    })
    print(profile)
    print(f'Guardado: {output} [{time.time()-started:.1f}s]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=int, default=4)
    parser.add_argument('--active-clusters', type=int, default=4)
    parser.add_argument('--sample-series', type=int, default=800)
    parser.add_argument('--workers', type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fit-until', type=int, default=201912)
    parser.add_argument('--suffix', default='')
    parser.add_argument('--medoids-file', default='',
                        help='Modo asignacion: reutiliza medoides entrenados en un corte anterior')
    parser.add_argument('--allow-slow', action='store_true')
    main(parser.parse_args())
