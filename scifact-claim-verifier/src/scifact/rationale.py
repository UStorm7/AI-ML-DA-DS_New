"""Rationale selection: pick the sentences of an abstract most likely to
support or refute the claim."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from scifact.data import Doc
from scifact.text import tokenize


class TfidfSentenceSelector:
    def __init__(self, corpus: dict[int, Doc]):
        self.vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2),
                                   sublinear_tf=True, min_df=1)
        self.vec.fit(s for d in corpus.values() for s in d.sentences)

    def score(self, claim: str, doc: Doc) -> np.ndarray:
        if not doc.sentences:
            return np.zeros(0)
        S = self.vec.transform(doc.sentences)
        return (S @ self.vec.transform([claim]).T).toarray().ravel()

    def select(self, claim: str, doc: Doc, n: int = 3) -> list[int]:
        """Indices of the top-n sentences, returned in document order."""
        return sorted(np.argsort(-self.score(claim, doc))[:n].tolist())


class DenseSentenceSelector:
    """Bi-encoder sentence scoring, usually better at paraphrases than TF-IDF."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def score(self, claim: str, doc: Doc) -> np.ndarray:
        emb = self.model.encode([claim] + doc.sentences, normalize_embeddings=True)
        return emb[1:] @ emb[0]

    def select(self, claim: str, doc: Doc, n: int = 3) -> list[int]:
        return sorted(np.argsort(-self.score(claim, doc))[:n].tolist())
