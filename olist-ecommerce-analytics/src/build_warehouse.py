"""Build a local DuckDB warehouse (staging -> marts) from the raw CSVs."""
import duckdb

from src.config import DB_PATH, RAW_DIR, SQL_DIR


def build(db_path=DB_PATH, raw_dir=RAW_DIR) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db_path))
    for sql_file in ["01_staging.sql", "02_marts.sql"]:
        sql = (SQL_DIR / sql_file).read_text().replace("{raw}", raw_dir.as_posix())
        con.execute(sql)
    return con


def main() -> None:
    con = build()
    for table in ["stg_orders", "stg_order_items", "stg_reviews", "fct_orders", "fct_delivered"]:
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table:<16} {n:>8,} rows")
    con.close()


if __name__ == "__main__":
    main()
