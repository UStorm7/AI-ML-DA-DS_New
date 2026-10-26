"""Data-quality and logic tests. Uses small synthetic CSVs so tests run without the real data."""
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src import analysis
from src.build_warehouse import build
from src.queries import load_queries, run

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    for f in FIXTURES.glob("*.csv"):
        shutil.copy(f, raw / f.name)
    c = build(db_path=tmp_path_factory.mktemp("db") / "test.duckdb", raw_dir=raw)
    yield c
    c.close()


def test_one_row_per_order(con):
    n, n_distinct = con.execute("SELECT COUNT(*), COUNT(DISTINCT order_id) FROM fct_orders").fetchone()
    assert n == n_distinct == 4


def test_duplicate_reviews_deduplicated(con):
    # order o1 has two reviews in the fixture; the most recent (score 5) must win
    assert con.execute("SELECT review_score FROM fct_orders WHERE order_id='o1'").fetchone()[0] == 5


def test_item_and_payment_aggregation(con):
    row = con.execute("SELECT n_items, item_value, payment_value FROM fct_orders WHERE order_id='o1'").fetchone()
    assert row == (2, 150.0, 170.0)


def test_lateness_flag(con):
    late = dict(con.execute("SELECT order_id, is_late FROM fct_orders").fetchall())
    assert late["o2"] is True and late["o1"] is False


def test_delivered_view_excludes_canceled(con):
    ids = {r[0] for r in con.execute("SELECT order_id FROM fct_delivered").fetchall()}
    assert "o4" not in ids


def test_repeat_customer_uses_unique_id(con):
    # c1 and c1b are two customer_ids for the same customer_unique_id 'u1'
    assert con.execute("SELECT COUNT(DISTINCT customer_id) FROM fct_delivered").fetchone()[0] == 2


def test_all_named_queries_execute(con):
    for name in load_queries():
        assert isinstance(run(con, name), pd.DataFrame)


def test_cohort_matrix_month0_is_100():
    long = pd.DataFrame({
        "cohort_month": ["2017-01-01"] * 2 + ["2017-02-01"],
        "month_number": [0, 1, 0],
        "active_customers": [10, 2, 5],
    })
    m = analysis.cohort_matrix(long, max_months=2)
    assert (m[0] == 100).all()
    assert m.loc["2017-01", 1] == 20
