## Retrieval (SciFact dev, claims with evidence)

| Retriever | R@1 | R@3 | R@10 | R@100 | MRR@10 | nDCG@10 | nDCG@10 (BEIR qrels) | ms/query |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.684 | 0.812 | 0.917 | 0.964 | 0.775 | 0.806 | 0.647 | 10.6 |
| tfidf | 0.501 | 0.709 | 0.845 | 0.972 | 0.631 | 0.678 | 0.560 | 3.1 |
| hybrid(bm25+tfidf) | 0.605 | 0.736 | 0.888 | 0.972 | 0.706 | 0.746 | 0.605 | 14.6 |

## Rationale sentence selection

| Selector | Sentence recall@3 |
|---|---|
| TfidfSentenceSelector | 0.544 |

## Verification

| Verifier | Retrieval | Abstract P | Abstract R | Abstract F1 | Claim macro-F1 | F1 SUP / CON / NEI |
|---|---|---|---|---|---|---|
| lexical | oracle | 0.559 | 0.498 | 0.527 | 0.573 | 0.62 / 0.37 / 0.72 |
| lexical | bm25 | 0.202 | 0.402 | 0.269 | 0.475 | 0.58 / 0.34 / 0.51 |
