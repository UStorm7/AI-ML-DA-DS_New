"""retrieve -> select rationale sentences -> verify."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scifact.data import LABELS, Claim, Doc


@dataclass
class DocVerdict:
    doc_id: int
    title: str
    label: str
    confidence: float
    rationale: list[str]


class ClaimVerifier:
    def __init__(self, corpus: dict[int, Doc], retriever, selector, verifier,
                 k_docs: int = 3, n_sentences: int = 3):
        self.corpus, self.retriever, self.selector, self.verifier = corpus, retriever, selector, verifier
        self.k_docs, self.n_sentences = k_docs, n_sentences

    def _example(self, claim: str, doc_id: int, rank: int) -> dict:
        doc = self.corpus[doc_id]
        scores = self.selector.score(claim, doc)
        idx = sorted(np.argsort(-scores)[: self.n_sentences].tolist())
        return {"claim": claim, "doc_id": doc_id, "rank": rank, "sent_idx": idx,
                "sentences": [doc.sentences[i] for i in idx],
                "sent_scores": scores[idx].tolist()}

    def training_examples(self, claims: list[Claim]) -> tuple[list[dict], list[str]]:
        """Pairs for the verifier head: gold evidence docs (SUPPORT / CONTRADICT),
        plus NEI pairs from retrieved non-evidence docs and from the docs that
        NEI claims cite (topically close, so they are hard negatives)."""
        examples, labels = [], []
        for c in claims:
            ranked = [d for d, _ in self.retriever.search(c.text, 50)]
            rank_of = {d: r for r, d in enumerate(ranked)}
            pairs = {d: e["label"] for d, e in c.evidence.items()}
            for d in ranked[: self.k_docs]:
                pairs.setdefault(d, "NEI")
            for d in c.cited_doc_ids:
                pairs.setdefault(d, "NEI")
            for d, lab in pairs.items():
                if d in self.corpus:
                    examples.append(self._example(c.text, d, rank_of.get(d, 50)))
                    labels.append(lab)
        return examples, labels

    def fit(self, claims: list[Claim]) -> "ClaimVerifier":
        self.verifier.fit(*self.training_examples(claims))
        return self

    def verify(self, claim: str, doc_ids: list[int] | None = None) -> list[DocVerdict]:
        if doc_ids is None:
            doc_ids = [d for d, _ in self.retriever.search(claim, self.k_docs)]
        examples = [self._example(claim, d, r) for r, d in enumerate(doc_ids)]
        P = self.verifier.predict_proba(examples)
        out = []
        for e, p in zip(examples, P):
            j = int(np.argmax(p))
            out.append(DocVerdict(e["doc_id"], self.corpus[e["doc_id"]].title, LABELS[j],
                                  float(p[j]), e["sentences"]))
        return out

    @staticmethod
    def claim_label(verdicts: list[DocVerdict]) -> str:
        decided = [v for v in verdicts if v.label != "NEI"]
        return max(decided, key=lambda v: v.confidence).label if decided else "NEI"
