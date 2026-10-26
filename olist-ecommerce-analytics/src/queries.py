"""Load the named queries in sql/03_analysis_queries.sql."""
import re

import duckdb
import pandas as pd

from src.config import SQL_DIR


def load_queries(path=SQL_DIR / "03_analysis_queries.sql") -> dict[str, str]:
    text = path.read_text()
    blocks = re.split(r"^-- name: (\w+)\s*$", text, flags=re.MULTILINE)
    # blocks = [preamble, name1, sql1, name2, sql2, ...]
    return {name: sql.strip().rstrip(";") for name, sql in zip(blocks[1::2], blocks[2::2])}


def run(con: duckdb.DuckDBPyConnection, name: str) -> pd.DataFrame:
    return con.execute(load_queries()[name]).df()
