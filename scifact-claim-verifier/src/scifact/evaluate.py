"""Run the full ablation and write reports/results.json + reports/results.md.

    python -m scifact.evaluate              # offline: BM25 / TF-IDF / hybrid + lexical verifier
    python -m scifact.evaluate --neural     # + dense retrieval, cross-encoder rerank, NLI verifier
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from scifact import metrics
from scifact.data import DATA_DIR, load_claims, load_corpus
from scifact.pipeline import ClaimVerifier
from scifact.rationale import TfidfSentenceSelector
from scifact.retrieval import BM25Retriever, HybridRetriever, TfidfRetriever
from scifact.verifier import Verifier

REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"


def eval_retrievers(retrievers, claims):
    rows = []
    for r in retrievers:
        t = time.perf_counter()
        runs = {c.id: [d for d, _ in r.search(c.text, 100)] for c in claims}
        ms = (time.perf_counter() - t) / len(claims) * 1000
        rows.append({"retriever": r.name, **metrics.retrieval_report(runs, claims), "ms_per_query": ms})
        print(f"  {r.name:<40} R@3 {rows[-1]['recall@3']:.3f}  nDCG@10 {rows[-1]['ndcg@10']:.3f}")
    return rows


def eval_rationale(selector, corpus, claims, n=3):
    """Sentence recall on gold evidence docs: share of gold rationale sentences in the top-n."""
    hit = total = 0
    for c in claims:
        for d, e in c.evidence.items():
            chosen = set(selector.select(c.text, corpus[d], n))
            hit += len(chosen & set(e["sentences"]))
            total += len(e["sentences"])
    return {"selector": type(selector).__name__, f"sentence_recall@{n}": hit / total}


def eval_verifier(cv: ClaimVerifier, claims, oracle: bool):
    preds, y_true, y_pred = {}, [], []
    for c in claims:
        doc_ids = (list(c.evidence) or c.cited_doc_ids) if oracle else None
        verdicts = cv.verify(c.text, doc_ids)
        preds[c.id] = {v.doc_id: v.label for v in verdicts}
        y_true.append(c.label)
        y_pred.append(cv.claim_label(verdicts))
    claim_rep = metrics.claim_label_report(y_true, y_pred)
    return {
        "abstract_label": metrics.abstract_label_f1(preds, claims),
        "claim_macro_f1": claim_rep["macro_f1"],
        "claim_f1_per_class": {k: claim_rep["per_class"][k]["f1-score"] for k in ("SUPPORT", "CONTRADICT", "NEI")},
    }


def to_markdown(res) -> str:
    out = ["## Retrieval (SciFact dev, claims with evidence)", "",
           "| Retriever | R@1 | R@3 | R@10 | R@100 | MRR@10 | nDCG@10 | nDCG@10 (BEIR qrels) | ms/query |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in res["retrieval"]:
        out.append(f"| {r['retriever']} | {r['recall@1']:.3f} | {r['recall@3']:.3f} | {r['recall@10']:.3f} | "
                   f"{r['recall@100']:.3f} | {r['mrr@10']:.3f} | {r['ndcg@10']:.3f} | {r['ndcg@10_beir']:.3f} | "
                   f"{r['ms_per_query']:.1f} |")
    out += ["", "## Rationale sentence selection", "", "| Selector | Sentence recall@3 |", "|---|---|"]
    for r in res["rationale"]:
        out.append(f"| {r['selector']} | {r['sentence_recall@3']:.3f} |")
    out += ["", "## Verification", "",
            "| Verifier | Retrieval | Abstract P | Abstract R | Abstract F1 | Claim macro-F1 | F1 SUP / CON / NEI |",
            "|---|---|---|---|---|---|---|"]
    for r in res["verification"]:
        a, f = r["abstract_label"], r["claim_f1_per_class"]
        out.append(f"| {r['verifier']} | {r['retrieval']} | {a['precision']:.3f} | {a['recall']:.3f} | "
                   f"{a['f1']:.3f} | {r['claim_macro_f1']:.3f} | "
                   f"{f['SUPPORT']:.2f} / {f['CONTRADICT']:.2f} / {f['NEI']:.2f} |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neural", action="store_true", help="add dense / cross-encoder / NLI components")
    ap.add_argument("--k-docs", type=int, default=3)
    args = ap.parse_args()
    np.random.seed(0)

    corpus = load_corpus()
    train, dev = load_claims("train"), load_claims("dev")
    print(f"corpus={len(corpus)} train={len(train)} dev={len(dev)}")

    print("Retrieval")
    bm25, tfidf = BM25Retriever(corpus), TfidfRetriever(corpus)
    retrievers = [bm25, tfidf, HybridRetriever([bm25, tfidf])]
    selectors = [TfidfSentenceSelector(corpus)]
    if args.neural:
        from scifact.rationale import DenseSentenceSelector
        from scifact.retrieval import CrossEncoderReranker, DenseRetriever
        dense = DenseRetriever(corpus, cache_dir=DATA_DIR)
        hybrid = HybridRetriever([bm25, dense])
        retrievers += [dense, hybrid, CrossEncoderReranker(hybrid, corpus)]
        selectors.append(DenseSentenceSelector())
    res = {"retrieval": eval_retrievers(retrievers, dev)}
    best_retriever = max(retrievers, key=lambda r: next(x["ndcg@10"] for x in res["retrieval"]
                                                         if x["retriever"] == r.name))
    print(f"  best: {best_retriever.name}")

    print("Rationale selection")
    res["rationale"] = [eval_rationale(s, corpus, dev) for s in selectors]
    best_sel = selectors[int(np.argmax([r["sentence_recall@3"] for r in res["rationale"]]))]
    for r in res["rationale"]:
        print(f"  {r}")

    print("Verification")
    verifiers = [Verifier()]
    if args.neural:
        from scifact.verifier import NLIScorer
        verifiers.append(Verifier(nli=NLIScorer()))
    res["verification"] = []
    for v in verifiers:
        cv = ClaimVerifier(corpus, best_retriever, best_sel, v, k_docs=args.k_docs).fit(train)
        for oracle in (True, False):
            r = {"verifier": v.name, "retrieval": "oracle" if oracle else best_retriever.name,
                 **eval_verifier(cv, dev, oracle)}
            res["verification"].append(r)
            print(f"  {v.name:<12} {r['retrieval']:<28} abstract-F1 {r['abstract_label']['f1']:.3f} "
                  f"claim macro-F1 {r['claim_macro_f1']:.3f}")

    REPORT_DIR.mkdir(exist_ok=True)
    suffix = "_neural" if args.neural else ""
    (REPORT_DIR / f"results{suffix}.json").write_text(json.dumps(res, indent=2))
    (REPORT_DIR / f"results{suffix}.md").write_text(to_markdown(res))
    print(f"wrote reports/results{suffix}.md")


if __name__ == "__main__":
    main()
