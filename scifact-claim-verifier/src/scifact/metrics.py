"""Retrieval and verification metrics."""
import math

from sklearn.metrics import classification_report, f1_score

from scifact.data import Claim


def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr_at_k(ranked: list[int], relevant: set[int], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[int], relevant: set[int], k: int = 10) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def retrieval_report(runs: dict[int, list[int]], claims: list[Claim]) -> dict:
    """Averaged over claims that have evidence (NEI claims have no relevant docs)."""
    rows = [(runs[c.id], set(c.evidence)) for c in claims if c.evidence]
    n = len(rows)
    return {
        "n_claims": n,
        "recall@1": sum(recall_at_k(r, g, 1) for r, g in rows) / n,
        "recall@3": sum(recall_at_k(r, g, 3) for r, g in rows) / n,
        "recall@10": sum(recall_at_k(r, g, 10) for r, g in rows) / n,
        "recall@100": sum(recall_at_k(r, g, 100) for r, g in rows) / n,
        "mrr@10": sum(mrr_at_k(r, g) for r, g in rows) / n,
        "ndcg@10": sum(ndcg_at_k(r, g) for r, g in rows) / n,
        # BEIR treats every claim's cited docs as relevant (incl. NEI claims);
        # reported so numbers are comparable with published BEIR baselines.
        "ndcg@10_beir": sum(ndcg_at_k(runs[c.id], set(c.cited_doc_ids)) for c in claims) / len(claims),
    }


def abstract_label_f1(predictions: dict[int, dict[int, str]], claims: list[Claim]) -> dict:
    """SciFact "abstract-level, label-only" metric: a predicted (claim, doc, label)
    is correct if the doc is a gold evidence doc for the claim with the same label.
    NEI predictions are not counted."""
    tp = n_pred = n_gold = 0
    for c in claims:
        gold = {d: e["label"] for d, e in c.evidence.items()}
        pred = {d: lab for d, lab in predictions.get(c.id, {}).items() if lab != "NEI"}
        n_gold += len(gold)
        n_pred += len(pred)
        tp += sum(1 for d, lab in pred.items() if gold.get(d) == lab)
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def claim_label_report(y_true: list[str], y_pred: list[str]) -> dict:
    labels = ["SUPPORT", "CONTRADICT", "NEI"]
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro"),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
    }
