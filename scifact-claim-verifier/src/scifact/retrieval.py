"""Abstract retrievers. Every retriever implements
`search(query, k) -> list[(doc_id, score)]`, so they can be swapped and fused."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from scifact.data import Doc
from scifact.text import tokenize


class BM25Retriever:
    name = "bm25"

    def __init__(self, corpus: dict[int, Doc], k1: float = 1.2, b: float = 0.75):
        self.ids = list(corpus)
        self.bm25 = BM25Okapi([tokenize(corpus[i].text) for i in self.ids], k1=k1, b=b)

    def search(self, query: str, k: int = 10):
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top]


class TfidfRetriever:
    name = "tfidf"

    def __init__(self, corpus: dict[int, Doc]):
        self.ids = list(corpus)
        self.vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2),
                                   sublinear_tf=True, min_df=2)
        self.matrix = self.vec.fit_transform(corpus[i].text for i in self.ids)

    def search(self, query: str, k: int = 10):
        scores = (self.matrix @ self.vec.transform([query]).T).toarray().ravel()
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top]


class DenseRetriever:
    """Bi-encoder retrieval with sentence-transformers (embeddings cached to disk)."""

    def __init__(self, corpus: dict[int, Doc], model_name: str = "BAAI/bge-small-en-v1.5",
                 cache_dir=None, query_prefix: str = "Represent this sentence for searching relevant passages: "):
        from sentence_transformers import SentenceTransformer

        self.name = f"dense:{model_name.split('/')[-1]}"
        self.ids = list(corpus)
        self.model = SentenceTransformer(model_name)
        self.query_prefix = query_prefix if "bge" in model_name else ""
        cache = (cache_dir / f"emb_{model_name.replace('/', '_')}.npy") if cache_dir else None
        if cache is not None and cache.exists():
            self.emb = np.load(cache)
        else:
            self.emb = self.model.encode([corpus[i].text for i in self.ids], batch_size=64,
                                         normalize_embeddings=True, show_progress_bar=True)
            if cache is not None:
                np.save(cache, self.emb)

    def search(self, query: str, k: int = 10):
        q = self.model.encode([self.query_prefix + query], normalize_embeddings=True)[0]
        scores = self.emb @ q
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top]


def reciprocal_rank_fusion(rankings: list[list[tuple[int, float]]], k: int = 60):
    """RRF (Cormack et al., 2009): score(d) = sum over rankers of 1 / (k + rank)."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            fused[doc_id] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(self, retrievers: list, depth: int = 100):
        self.retrievers, self.depth = retrievers, depth
        self.name = "hybrid(" + "+".join(r.name for r in retrievers) + ")"

    def search(self, query: str, k: int = 10):
        return reciprocal_rank_fusion([r.search(query, self.depth) for r in self.retrievers])[:k]


class CrossEncoderReranker:
    """Re-scores the first stage's top candidates with a cross-encoder."""

    def __init__(self, first_stage, corpus: dict[int, Doc], depth: int = 30,
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.first_stage, self.corpus, self.depth = first_stage, corpus, depth
        self.model = CrossEncoder(model_name, max_length=512)
        self.name = f"{first_stage.name}+rerank"

    def search(self, query: str, k: int = 10):
        cands = [d for d, _ in self.first_stage.search(query, self.depth)]
        scores = self.model.predict([(query, self.corpus[d].text) for d in cands])
        order = np.argsort(-scores)[:k]
        return [(cands[i], float(scores[i])) for i in order]
