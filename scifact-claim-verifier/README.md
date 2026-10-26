# Scientific Claim Verifier: Retrieval + Rationale Selection + NLI, with a Proper Evaluation Harness

Given a scientific claim such as *"Vitamin C prevents the common cold"*, the system:

1. **Retrieves** candidate abstracts from a corpus of 5,183 biomedical papers (BM25, TF-IDF, dense bi-encoder, hybrid via Reciprocal Rank Fusion, cross-encoder re-ranking).
2. **Selects rationale sentences**: the 3 sentences of each abstract most relevant to the claim.
3. **Verifies** each (claim, abstract) pair as **SUPPORT / CONTRADICT / NOT ENOUGH INFO**. A learned head combines interpretable lexical cues with probabilities from a pretrained NLI cross-encoder.
4. Returns a verdict **with cited evidence sentences**, so every answer can be checked.

This is the retrieval-and-grounding core of RAG and fact-checking systems. Every stage is **measured
separately** on the [SciFact](https://github.com/allenai/scifact) benchmark (Wadden et al., EMNLP 2020). When
the system is wrong, you can tell which stage failed.

## Results (SciFact dev set, 300 held-out claims; the official test labels are hidden)

### Retrieval
| Retriever | R@1 | R@3 | R@10 | MRR@10 | nDCG@10 | nDCG@10 (BEIR qrels) |
|---|---|---|---|---|---|---|
| **BM25** (custom tokenizer + light stemming) | **0.684** | **0.812** | **0.917** | **0.775** | **0.806** | **0.647** |
| TF-IDF (1–2 grams) | 0.501 | 0.709 | 0.845 | 0.631 | 0.678 | 0.560 |
| Hybrid BM25 + TF-IDF (RRF) | 0.605 | 0.736 | 0.888 | 0.706 | 0.746 | 0.605 |
| Dense `bge-small-en-v1.5` | *run `make eval-neural`* | | | | | |
| Hybrid BM25 + dense (RRF) | | | | | | |
| Hybrid + cross-encoder rerank | | | | | | |

*The BEIR column treats every cited document as relevant, NEI claims included. It is the convention used by
published baselines, and our BM25 (0.647) matches the published BEIR BM25 figure (~0.665). The other columns
count only documents that contain annotated evidence, which is what the verifier actually needs.*

### Verification
| Verifier | Retrieval | Abstract-level P / R / F1 | Claim macro-F1 | F1 SUPPORT / CONTRADICT / NEI |
|---|---|---|---|---|
| Lexical head | oracle docs | 0.559 / 0.498 / **0.527** | 0.573 | 0.62 / 0.37 / 0.72 |
| Lexical head | BM25 top-3 | 0.202 / 0.402 / **0.269** | 0.475 | 0.58 / 0.34 / 0.51 |
| Lexical + NLI (DeBERTa-v3 MNLI/FEVER/ANLI) | oracle docs | *run `make eval-neural`* | | |
| Lexical + NLI | best retriever | | | |

Full generated tables: [`reports/results.md`](reports/results.md) (offline run) and `reports/results_neural.md`
(neural run; use the [Colab notebook](notebooks/run_neural_on_colab.ipynb) for a free GPU).

## What the numbers say (error analysis)
- **A well-tuned BM25 is a strong baseline** on scientific text. It beats TF-IDF by 13 nDCG points, and naively fusing it with a weaker retriever *hurts* (0.806 → 0.746). Fusion only helps when the retrievers make *different* mistakes, which is the reason to test BM25 + dense.
- **Retrieval is not the bottleneck; verification is.** BM25 finds a gold abstract in the top 3 for 81% of claims. Yet abstract-level F1 halves when moving from oracle documents to retrieved ones (0.527 → 0.269). Precision collapses (0.56 → 0.20) because the lexical verifier labels *topically similar* but non-evidential abstracts as SUPPORT or CONTRADICT. `reports/demo_lexical_baseline.png` shows a vitamin D claim marked CONTRADICT on the strength of a generic vitamin D review.
- **CONTRADICT is the hardest class** (F1 0.37). Lexical cues (negations, increase/decrease direction flips, number mismatches) catch only explicit contradictions. Contradiction needs actual inference, which is where the NLI model comes in.
- Sentence recall@3 for rationale selection is 0.54 with TF-IDF. Missing rationale sentences put a ceiling on the verifier, so dense sentence selection is part of the neural run.

## Design
```
claim ─► Retriever ──────────────► top-k abstracts ─► Rationale selector ─► top-3 sentences
         (BM25 | TF-IDF | dense |                                              │
          hybrid-RRF | +rerank)                                                ▼
                                              Verifier head (logistic regression, class-balanced)
                                              ├─ lexical features: coverage, Jaccard, retrieval rank,
                                              │  negation / direction-flip / number-mismatch cues
                                              └─ NLI features: P(entail), P(contradict), P(neutral)
                                                                               │
                                     per-abstract labels + confidence + cited sentences ─► claim verdict
```
- **Training data for the head:** gold evidence abstracts (SUPPORT/CONTRADICT), plus *hard negatives* labelled NEI: retrieved non-evidence abstracts and the abstracts that NEI claims cite. All of it comes from the train split. The dev split is used only for evaluation.
- **Metrics:** Recall@k, MRR@10 and nDCG@10 for retrieval. Sentence recall@3 for rationales. The official SciFact *abstract-level label-only* P/R/F1 and claim-level macro-F1 for verification. Every metric is unit-tested.
- **Swappable components:** every retriever exposes `search(query, k)`, so new ones plug straight into the fusion, reranking and evaluation code.

## Run it
```bash
pip install -r requirements.txt pytest ruff
make eval          # offline pipeline (~40 s on a laptop CPU); downloads SciFact automatically
make demo          # Streamlit UI
make test          # 10 unit tests

pip install -r requirements-neural.txt
make eval-neural   # + dense, reranker, NLI (GPU recommended: see notebooks/)
```

## Project layout
```
src/scifact/
  data.py        SciFact download + typed loaders
  text.py        tokenizer, light stemmer, negation lexicon
  retrieval.py   BM25, TF-IDF, dense, RRF hybrid, cross-encoder reranker
  rationale.py   TF-IDF and dense sentence selectors
  verifier.py    lexical features, NLI scorer, learned verifier head
  pipeline.py    ClaimVerifier: retrieve → select → verify; hard-negative training pairs
  metrics.py     retrieval + SciFact verification metrics
  evaluate.py    full ablation → reports/results*.json / .md
app/demo.py      Streamlit demo
tests/           unit tests (metrics, RRF, retrieval, features, end-to-end on a toy corpus)
```

## Next steps
- Fine-tune the NLI model on SciFact train pairs instead of using it zero-shot.
- Add an LLM verifier (structured JSON output with cited sentence ids) and compare it with NLI on cost, latency and F1.
- Calibrate the NEI threshold to trade precision against recall, the main failure mode found above.
