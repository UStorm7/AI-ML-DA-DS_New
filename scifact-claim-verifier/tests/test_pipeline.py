from scifact.data import LABELS
from scifact.pipeline import ClaimVerifier
from scifact.rationale import TfidfSentenceSelector
from scifact.retrieval import BM25Retriever, HybridRetriever, TfidfRetriever
from scifact.text import tokenize
from scifact.verifier import Verifier, lexical_features


def test_tokenize_stems_and_drops_stopwords():
    assert tokenize("The studies were increasing") == ["study", "increas"]


def test_bm25_finds_obvious_doc(corpus):
    assert BM25Retriever(corpus).search("vitamin C common cold", 1)[0][0] == 2


def test_hybrid_returns_k(corpus):
    r = HybridRetriever([BM25Retriever(corpus), TfidfRetriever(corpus)])
    assert len(r.search("aspirin", 2)) == 2


def test_selector_picks_relevant_sentence(corpus):
    sel = TfidfSentenceSelector(corpus)
    assert sel.select("aspirin heart attack risk", corpus[1], n=1) == [0]


def test_lexical_features_detect_direction_conflict():
    f = lexical_features("X increases Y.", ["X decreased Y in mice."], 0, [0.5])
    g = lexical_features("X increases Y.", ["X increased Y in mice."], 0, [0.5])
    assert f[13] == 1.0 and g[13] == 0.0


def test_end_to_end_fit_and_verify(corpus, claims):
    cv = ClaimVerifier(corpus, BM25Retriever(corpus), TfidfSentenceSelector(corpus), Verifier(), k_docs=2)
    cv.fit(claims)
    verdicts = cv.verify("Aspirin reduces heart attack risk")
    assert verdicts and all(v.label in LABELS and 0 <= v.confidence <= 1 for v in verdicts)
    assert cv.claim_label(verdicts) in LABELS
