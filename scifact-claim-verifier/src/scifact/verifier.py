"""Label prediction for a (claim, abstract) pair: SUPPORT / CONTRADICT / NEI.

Feature groups (combined by a learned classifier head trained on SciFact train):
  * lexical:  overlap, coverage, negation / direction / number mismatch cues
  * nli:      entailment / contradiction / neutral probabilities from a
              pretrained NLI cross-encoder (zero-shot, optional)
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scifact.data import LABELS
from scifact.text import NEGATIONS, NUMBER, tokenize

UP = frozenset("increase increases increased increasing higher elevated enhance enhances enhanced "
               "promote promotes promoted improve improves improved greater more risk".split())
DOWN = frozenset("decrease decreases decreased decreasing lower reduced reduce reduces inhibit "
                 "inhibits inhibited suppress suppresses suppressed less impaired protective".split())


def lexical_features(claim: str, sentences: list[str], retrieval_rank: int, sent_scores: list[float]) -> list[float]:
    ev = " ".join(sentences)
    c_raw = set(tokenize(claim, stem=False, keep_stopwords=True))
    e_raw = set(tokenize(ev, stem=False, keep_stopwords=True))
    c_tok, e_tok = set(tokenize(claim)), set(tokenize(ev))
    c_neg, e_neg = len(c_raw & NEGATIONS), len(e_raw & NEGATIONS)
    c_up, c_down = len(c_raw & UP), len(c_raw & DOWN)
    e_up, e_down = len(e_raw & UP), len(e_raw & DOWN)
    c_nums = set(NUMBER.findall(claim))
    s = sorted(sent_scores, reverse=True) + [0.0, 0.0, 0.0]
    return [
        1.0 / (1 + retrieval_rank),
        s[0], np.mean(s[:3]), s[0] - s[1],
        len(c_tok & e_tok) / max(len(c_tok), 1),          # claim coverage
        len(c_tok & e_tok) / max(len(c_tok | e_tok), 1),  # jaccard
        c_neg, e_neg, float((c_neg > 0) != (e_neg > 0)),
        c_up, c_down, e_up, e_down,
        float(c_up > 0 and e_down > 0 and e_up == 0) + float(c_down > 0 and e_up > 0 and e_down == 0),
        len(c_nums), len(c_nums - set(NUMBER.findall(ev))),
    ]


class NLIScorer:
    """Zero-shot NLI probabilities from a pretrained cross-encoder (premise = evidence, hypothesis = claim)."""

    def __init__(self, model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        id2label = {i: lab.lower() for i, lab in self.model.config.id2label.items()}
        self.order = [next(i for i, lab in id2label.items() if lab.startswith(name))
                      for name in ("entail", "contra", "neutral")]

    def __call__(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(pairs), batch_size):
            prem, hyp = zip(*pairs[i:i + batch_size])
            enc = self.tok(list(prem), list(hyp), truncation="only_first", max_length=512,
                           padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                probs = self.torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy()
            out.append(probs[:, self.order])
        return np.vstack(out) if out else np.zeros((0, 3))


class Verifier:
    """Classifier head over the feature groups. `use_nli=False` gives a fully
    offline baseline that needs no model downloads."""

    def __init__(self, nli: NLIScorer | None = None, C: float = 1.0):
        self.nli = nli
        self.head = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))

    @property
    def name(self) -> str:
        return "lexical+nli" if self.nli else "lexical"

    def featurize(self, examples: list[dict]) -> np.ndarray:
        """examples: dicts with claim, sentences, rank, sent_scores."""
        X = np.array([lexical_features(e["claim"], e["sentences"], e["rank"], e["sent_scores"])
                      for e in examples], dtype=float)
        if self.nli:
            P = self.nli([(" ".join(e["sentences"]), e["claim"]) for e in examples])
            X = np.hstack([X, P, P[:, :1] - P[:, 1:2]])
        return X

    def fit(self, examples: list[dict], labels: list[str]) -> "Verifier":
        self.head.fit(self.featurize(examples), labels)
        return self

    def predict_proba(self, examples: list[dict]) -> np.ndarray:
        """Columns ordered as data.LABELS."""
        P = self.head.predict_proba(self.featurize(examples))
        cols = list(self.head.classes_)
        return P[:, [cols.index(lab) for lab in LABELS]]
