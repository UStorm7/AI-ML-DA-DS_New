"""Interactive demo:  streamlit run app/demo.py   (add --  --neural for the NLI verifier)"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scifact.data import load_claims, load_corpus  # noqa: E402
from scifact.pipeline import ClaimVerifier  # noqa: E402
from scifact.rationale import TfidfSentenceSelector  # noqa: E402
from scifact.retrieval import BM25Retriever  # noqa: E402
from scifact.verifier import Verifier  # noqa: E402

NEURAL = "--neural" in sys.argv
st.set_page_config(page_title="Scientific Claim Verifier", layout="wide")


@st.cache_resource(show_spinner="Indexing 5,183 abstracts and training the verifier...")
def build():
    corpus = load_corpus()
    nli = None
    if NEURAL:
        from scifact.verifier import NLIScorer
        nli = NLIScorer()
    cv = ClaimVerifier(corpus, BM25Retriever(corpus), TfidfSentenceSelector(corpus), Verifier(nli=nli))
    return cv.fit(load_claims("train")), load_claims("dev")


cv, dev = build()
st.title("Scientific claim verifier")
st.caption("Retrieves evidence from 5,183 biomedical abstracts (SciFact), picks rationale sentences, and "
           f"predicts SUPPORT / CONTRADICT / NOT ENOUGH INFO. Verifier: **{cv.verifier.name}**")

examples = [c.text for c in dev if c.evidence][:25]
pick = st.selectbox("Try a held-out claim from the dev set, or type your own below", [""] + examples)
claim = st.text_input("Claim", value=pick)

if claim:
    verdicts = cv.verify(claim)
    label = cv.claim_label(verdicts)
    icon = {"SUPPORT": "✅", "CONTRADICT": "❌", "NEI": "❔"}[label]
    st.subheader(f"{icon} Overall verdict: {label}")
    gold = next((c for c in dev if c.text == claim), None)
    if gold:
        st.write(f"Gold label: **{gold.label}**")
    for v in verdicts:
        with st.expander(f"{v.label} ({v.confidence:.0%}) | {v.title}", expanded=v.label != "NEI"):
            for s in v.rationale:
                st.markdown(f"> {s}")
            st.caption(f"doc_id {v.doc_id}")
