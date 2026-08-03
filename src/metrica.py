"""
Metrica de la competencia: Total Error Rate (WAPE, ponderado por volumen).

    total_error_rate = sum(|tn_real - tn_pred|) / sum(tn_real)

Uso como JUEZ LOCAL sin gastar submits, contra el mundo ideal (z262):
    import polars as pl
    from metrica import total_error_rate
    ri = pl.read_csv('datasets/tb_realidad_ideal.csv')   # product_id, tn_real
    err = total_error_rate(mi_prediccion, ri)            # mi_prediccion: product_id, tn

Referencia de piso (mundo ideal):
    naif ultimo mes  -> 0.6246
    naif promedio 12 -> 0.4357   (dificil de batir)
"""
import polars as pl


def total_error_rate(pred: pl.DataFrame, real: pl.DataFrame) -> float:
    """pred: columnas [product_id, tn]  |  real: columnas [product_id, tn_real]."""
    j = (
        real.join(pred, on="product_id", how="left")
        .with_columns(pl.col("tn").fill_null(0.0))
    )
    num = (j["tn_real"] - j["tn"]).abs().sum()
    den = j["tn_real"].sum()
    return num / den
