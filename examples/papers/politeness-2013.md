# Danescu-Niculescu-Mizil, Sudhof, Jurafsky, Leskovec & Potts (2013): A computational approach to politeness

ACL 2013, [arXiv 1306.6078](https://arxiv.org/abs/1306.6078).

**Question.** Which linguistic strategies make a request read as polite (Table 3), and can a classifier that uses them
tell polite from impolite requests, in-domain and across domains (Table 4)?

**Data.** ConvoKit's `wikipedia-politeness-corpus` (4,353 requests) and `stack-exchange-politeness-corpus` (6,603;
the paper says 6,604). `meta.Normalized Score` is the paper's politeness score (mean of 5 per-worker z-scored ratings);
`meta.Binary` is ±1 for the top/bottom quartile *within each domain*, 0 otherwise. We checked it matches the score
quartiles exactly, and gives the paper's class sizes (1,089 per class for Wiki, 1,651 for SE).
`politeness-wiki-se` in `datasets.yaml` merges both corpora and tags each utterance with `meta.corpus`.

## Method as pipeline stages

`examples/politeness-2013-strategies.yaml` (Table 3, Wikipedia only, as in the paper):
`parse` → `politeness` → `strategy_table {score: meta.Normalized Score, label: meta.Binary == 1}`.
For each strategy, `strategy_table` gives the number of requests that use it, their mean score, the % of them in the top
quartile, a Mann-Whitney p (requests with the strategy vs the rest) and a binomial p (vs the 25% base rate). These are the
paper's tests.

`examples/politeness-2013-classifiers.yaml` (Table 4):
`parse` → `politeness` → `filter {meta.Binary != 0}` → 8 × `text_classify`.
`text_classify` trains a linear SVM (`LinearSVC`, C=0.1) on unigrams (dropping unigrams seen fewer than 10 times in
the training data, as in the paper's footnote 6). BOW uses unigrams only; Ling. adds the 21 ConvoKit strategy
indicators. When train and test are the same domain it runs stratified 10-fold CV; otherwise it trains on all of one
domain and tests on all of the other.

## Paper vs ours

Table 3 (Wikipedia; mean politeness score / % in top quartile; ConvoKit strategy name in brackets):

| Strategy | Paper | Ours | n |
|---|---|---|---|
| Gratitude | 0.87 / 78% | 0.88 / 77% | 286 |
| Deference | 0.78 / 70% | 0.77 / 70% | 60 |
| Greeting (Indirect_(greeting)) | 0.43 / 45% | 0.44 / 46% | 371 |
| Positive lexicon (HASPOSITIVE) | 0.12 / 32% | 0.11 / 31% | 1952 |
| Negative lexicon (HASNEGATIVE) | −0.13 / 22% | −0.14 / 22% | 1451 |
| Apologizing | 0.36 / 53% | 0.40 / 51% | 85 |
| Please | 0.49 / 57% | 0.51 / 56% | 162 |
| Please start | −0.30 / 22% | −0.28 / 24% | 33 |
| Indirect (btw) | 0.63 / 58% | 0.58 / 58% | 24 |
| Direct question | −0.27 / 15% | −0.26 / 15% | 913 |
| Direct start | −0.43 / 9% | −0.32 / 13% | 382 |
| Counterfactual modal (SUBJUNCTIVE) | 0.47 / 52% | 0.48 / 54% | 416 |
| Indicative modal (INDICATIVE) | 0.09 / 27% | 0.19 / 32% | 284 |
| 1st person start | 0.12 / 29% | 0.12 / 29% | 1246 |
| 1st person pl. | 0.08 / 27% | 0.07 / 26% | 383 |
| 1st person | 0.08 / 28% | 0.08 / 29% | 1758 |
| 2nd person | 0.05 / 30% | 0.05 / 30% | 2828 |
| 2nd person start | −0.30 / 17% | −0.30 / 16% | 245 |
| Hedges | 0.14 / 28% | 0.13 / 27% | 732 |
| Factuality | −0.38 / 13% | −0.37 / 14% | 236 |

Tolerance: within 0.15 of the mean score and 10 points of the top-quartile share, fixed before we ran anything.
0.15 is about 0.2 of the score SD (0.7). These margins absorb differences in *which* requests get a strategy: spaCy 3.8
parses vs the paper's Stanford parses, and ConvoKit 5's lexicons. They catch a polite/impolite swap and any sign flip on a strategy whose
mean score is more than 0.15 from zero; a smaller sign flip can pass. All 20 rows pass, and 18 of 20 are within 0.05 / 3 points (the exceptions are Direct start and Indicative modal).
The significance stars mostly agree. Every strategy the paper marks significant on the mean has p < .05 here, except
1st person pl. (paper p < .05, ours p = .066). Indicative modal goes the other way: the paper finds it not
significant, while ours is significant (p < .001).

Table 4 (accuracy %, random baseline 50%):

| Train → Test | BOW paper | BOW ours | Ling. paper | Ling. ours |
|---|---|---|---|---|
| Wiki → Wiki (in-domain) | 79.84 | 79.57 | 83.79 | 82.37 |
| SE → SE (in-domain) | 74.47 | **62.78** | 78.19 | **63.84** |
| Wiki → SE | 64.23 | **60.48** | 67.53 | **61.27** |
| SE → Wiki | 72.17 | 72.13 | 75.43 | 73.97 |

(bold = more than 3 points from the paper)

## Verdict: partially reproduced

- **Table 3: reproduced.** All 20 strategies match the paper's direction and size. Significance differs on 2 (see above).
- **Table 4, tested on Wikipedia: reproduced.** All 4 cells are within 1.5 points, and Ling. beats BOW in every
  setting (+1–3 points; the paper reports 3–4).
- **Table 4, tested on Stack Exchange: not reproduced.** SE in-domain is 11–14 points below the paper, and Wiki → SE
  is 4–6 points below. Tuning didn't close the gap. We tried SVM/LR, C ∈ {1, 0.1, 0.01}, count vs binary features, and
  keeping `?`/`!` as tokens: the best SE in-domain score was 64.6. The labels match the paper's class sizes and quartile
  definition exactly. So the gap is most likely in the data (ConvoKit's SE release may differ from the 2013 snapshot)
  or in unreported details of the paper's SE setup, not in model choice. We could not settle which.
- Table 5 (admin promotion) is not attempted. The politeness corpora carry no adminship metadata.

## Deviations

- **10-fold CV instead of leave-one-out.** With 2–3k examples this changes accuracy by well under a point.
- **C=0.1 instead of an unstated value.** C=1 gives Wiki BOW 76.0, about 4 points low. C=0.1 was picked after looking
  at in-domain CV for C ∈ {1, 0.1, 0.01}, so it is mildly tuned. C=0.01 is within ±3 points on the same cells.
- **Unigram filter and tokeniser.** The filter counts occurrences in the training split. The tokeniser is sklearn's
  `CountVectorizer` default: lowercased, tokens of 2+ characters, no punctuation.
- **ConvoKit 5 politeness features (spaCy) instead of the original Stanford-parser extractor.** This is the likely
  cause of the Direct start and Indicative modal gaps (0.11 and 0.10 on the mean). ConvoKit also adds a 21st feature,
  HASHEDGE, which the paper doesn't list and which Ling. uses.
- **Shared ids.** The two corpora share 39 utterance ids; merged datasets now prefix every id with its corpus name
  (`wikipedia-politeness-corpus:…`), so `to_corpus` works on the merge.

## Runtime

On a laptop with warm downloads: Table 3 pipeline ~15 s, Table 4 pipeline ~35 s (parse + politeness on 10,956
utterances ~40 s cold). Slow test: `CKVIZ_EXAMPLES=1 uv run pytest tests/test_paper_politeness.py` → 2 passed,
1 xfailed (strict; the SE-tested cells), ~70 s.
