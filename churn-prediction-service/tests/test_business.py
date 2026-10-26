import numpy as np

from churn import business
from churn.config import CONTACT_COST, MONTHS_SAVED, SAVE_RATE


def test_profit_counts_only_contacted():
    y, p, m = np.array([1, 0, 1]), np.array([0.9, 0.8, 0.1]), np.array([100.0, 100.0, 100.0])
    expected = (SAVE_RATE * MONTHS_SAVED * 100 - CONTACT_COST) + (-CONTACT_COST)
    assert business.campaign_profit(y, p, m, threshold=0.5) == expected


def test_best_threshold_is_in_range():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.15, 500), 0, 1)
    t, profit = business.best_threshold(y, p, np.full(500, 70.0))
    assert 0.05 <= t <= 0.95 and profit > 0
