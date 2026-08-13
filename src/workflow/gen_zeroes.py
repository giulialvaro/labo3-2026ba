"""Genera datasets/sell-in-zeroes.txt.gz (cliente-producto con ceros) = z601 en script.
Detecta la carpeta de datasets (local o GCP ~/buckets/b1/datasets).
Correr:  python3 src/workflow/gen_zeroes.py
"""
import duckdb, os, time


def find_data():
    for d in [os.environ.get('DATA_DIR'), 'datasets', os.path.expanduser('~/buckets/b1/datasets')]:
        if d and os.path.isdir(d):
            return d
    return 'datasets'


DATA = find_data()
con = duckdb.connect()
t = time.time()
con.execute(f"""CREATE OR REPLACE TABLE tb_sellin AS
  SELECT customer_id,product_id,periodo,plan_precios_cuidados,cust_request_qty,cust_request_tn,tn
  FROM read_csv_auto('{DATA}/sell-in.txt.gz')""")
con.execute("CREATE OR REPLACE TABLE tb_periodos AS SELECT DISTINCT periodo FROM tb_sellin")
con.execute("CREATE OR REPLACE TABLE tb_productos_fechas AS SELECT product_id, MIN(periodo) periodo_min, MAX(periodo) periodo_max FROM tb_sellin GROUP BY product_id")
con.execute("CREATE OR REPLACE TABLE tb_clientes_fechas AS SELECT customer_id, MIN(periodo) periodo_min FROM tb_sellin GROUP BY customer_id")
con.execute("CREATE OR REPLACE TABLE tb_precios_cuidados AS SELECT product_id, MIN(periodo) periodo_min, MAX(periodo) periodo_max FROM tb_sellin WHERE plan_precios_cuidados=1 GROUP BY product_id")
con.execute("""CREATE OR REPLACE TABLE tb_zeroes AS
  SELECT cf.customer_id, pf.product_id, per.periodo, CAST(0 AS INT) plan_precios_cuidados, CAST(0 AS INT) cust_request_qty, 0.0 cust_request_tn, 0.0 tn
  FROM tb_productos_fechas pf, tb_clientes_fechas cf, tb_periodos per
  WHERE NOT EXISTS (SELECT 1 FROM tb_sellin si WHERE si.periodo=per.periodo AND si.customer_id=cf.customer_id AND si.product_id=pf.product_id)
    AND per.periodo BETWEEN pf.periodo_min AND pf.periodo_max AND per.periodo >= cf.periodo_min""")
con.execute("""UPDATE tb_zeroes z SET plan_precios_cuidados=1
  WHERE EXISTS (SELECT 1 FROM tb_precios_cuidados p WHERE z.periodo BETWEEN p.periodo_min AND p.periodo_max AND p.product_id=z.product_id)""")
con.execute("CREATE OR REPLACE TABLE tb_sellin_zeroes AS SELECT * FROM tb_sellin UNION ALL SELECT * FROM tb_zeroes")
con.execute(f"COPY (SELECT * FROM tb_sellin_zeroes ORDER BY 1,2,3) TO '{DATA}/sell-in-zeroes.txt.gz' (FORMAT csv, COMPRESSION 'gzip', HEADER)")
n = con.sql("SELECT COUNT(*) FROM tb_sellin_zeroes").fetchone()[0]
print(f'generado {DATA}/sell-in-zeroes.txt.gz  ({n:,} filas)  [{time.time()-t:.0f}s]')
