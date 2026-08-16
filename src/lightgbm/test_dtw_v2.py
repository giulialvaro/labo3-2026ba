"""Pruebas chicas del preprocesamiento y asignacion DTW v2."""
import numpy as np

from dtw_clusters_cp_v2 import assign_active, distance, stratified_sample, transform_sequence


def main():
    values = np.array([[99, 0, 1, 0, 2], [0, 1, 0, 2, 0], [0, 0, 2, 0, 4]], dtype=float)
    observed = np.array([[False, True, True, True, True], [True] * 5, [True] * 5])
    a = transform_sequence(values[0], observed[0])
    b = transform_sequence(values[1], observed[1])
    assert len(a) == 4, 'No recorto el periodo fuera de existencia'
    assert abs(distance(a, b, 2) - distance(b, a, 2)) < 1e-9, 'DTW no simetrico'

    sample1 = stratified_sample(np.arange(20), np.arange(6, 26), np.arange(1, 21), 12, 1)
    sample2 = stratified_sample(np.arange(20), np.arange(6, 26), np.arange(1, 21), 12, 1)
    assert np.array_equal(sample1, sample2), 'Muestra no reproducible'

    labels = assign_active(np.arange(3), values, observed, [a, b], 2, workers=2)
    assert set(labels.tolist()) <= {1, 2}, 'Cluster fuera de rango'
    print('DTW v2: tests OK')


if __name__ == '__main__':
    main()
