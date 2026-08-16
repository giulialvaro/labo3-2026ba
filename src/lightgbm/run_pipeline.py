"""Orquestador reproducible del workflow final.

No sube nada a Kaggle automaticamente. Genera resultados locales y CSV listos
para revisar/subir.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def run(parts):
    print('\n$', ' '.join(map(str, parts)), flush=True)
    subprocess.run(parts, cwd=ROOT, check=True)


def main(args):
    if args.product:
        run([PYTHON, 'src/lightgbm/producto_limpio.py'])
    if args.dtw:
        run([PYTHON, 'src/lightgbm/dtw_clusters_cp_v2.py',
             '--window', str(args.dtw_window), '--active-clusters', str(args.dtw_clusters)])
    if args.dtw_snapshots:
        for train_cut, score_cut in [(201710, 201810), (201712, 201812), (201812, 201912)]:
            fit_suffix = f'fit{train_cut}'
            score_suffix = f'score{score_cut}_from{train_cut}'
            run([PYTHON, 'src/lightgbm/dtw_clusters_cp_v2.py',
                 '--fit-until', str(train_cut), '--suffix', fit_suffix,
                 '--window', str(args.dtw_window), '--active-clusters', str(args.dtw_clusters)])
            run([PYTHON, 'src/lightgbm/dtw_clusters_cp_v2.py',
                 '--fit-until', str(score_cut), '--suffix', score_suffix,
                 '--window', str(args.dtw_window), '--active-clusters', str(args.dtw_clusters),
                 '--medoids-file', f'exp/dtw_v2/medoids_{fit_suffix}.npz'])
    if args.cp:
        cmd = [PYTHON, '-u', 'src/lightgbm/cliente_producto_hurdle.py',
               '--sample-products', str(args.sample_products), '--variants', args.variants]
        if args.final:
            cmd.append('--final')
        run(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--product', action='store_true')
    parser.add_argument('--dtw', action='store_true')
    parser.add_argument('--dtw-snapshots', action='store_true')
    parser.add_argument('--cp', action='store_true')
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--sample-products', type=int, default=100)
    parser.add_argument('--variants', default='one_stage,hurdle,hurdle_cluster,hurdle_per_cluster')
    parser.add_argument('--dtw-window', type=int, default=4)
    parser.add_argument('--dtw-clusters', type=int, default=4)
    args = parser.parse_args()
    if not (args.product or args.dtw or args.dtw_snapshots or args.cp):
        parser.error('Elegir al menos uno: --product, --dtw-snapshots, --dtw o --cp')
    main(args)
