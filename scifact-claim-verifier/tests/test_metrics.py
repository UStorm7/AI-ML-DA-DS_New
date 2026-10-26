import math

import pytest

from scifact import metrics
from scifact.retrieval import reciprocal_rank_fusion


def test_recall_and_mrr():
    assert metrics.recall_at_k([5, 1, 2], {1, 9}, 2) == 0.5
    assert metrics.mrr_at_k([5, 1, 2], {1}) == 0.5
    assert metrics.mrr_at_k([5, 6], {1}) == 0.0


def test_ndcg_perfect_and_partial():
    assert metrics.ndcg_at_k([1, 2, 3], {1}) == 1.0
    assert metrics.ndcg_at_k([9, 1], {1}) == pytest.approx(1 / math.log2(3))


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[(1, 9.0), (2, 8.0)], [(2, 0.9), (3, 0.8)]])
    assert fused[0][0] == 2  # ranked by both


def test_abstract_label_f1(claims):
    preds = {10: {1: "SUPPORT"}, 11: {2: "SUPPORT"}, 12: {3: "NEI"}}
    m = metrics.abstract_label_f1(preds, claims)
    assert m["precision"] == 0.5 and m["recall"] == 0.5
