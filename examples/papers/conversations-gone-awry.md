# Zhang, Chang, Danescu-Niculescu-Mizil, Dixon, Hua, Taraborelli & Thain (2018): Conversations Gone Awry

ACL 2018, [arXiv 1805.05345](https://arxiv.org/abs/1805.05345).

**Question.** Can we tell from the first comment-reply exchange of a civil Wikipedia talk-page conversation whether it
will later turn into a personal attack? Which pragmatic markers in that exchange signal it (Figure 2A), and how
accurate is a classifier in a paired, topic-controlled task (Table 3)?

**Data.** ConvoKit's `conversations-gone-awry-corpus` (dataset `conversations-gone-awry-corpus` in `datasets.yaml`).
This is the 2019 extended release: 4,188 conversations. We keep `annotation_year == 2018`, which leaves
**1,168 conversations = 584 pairs on 536 talk pages**. The paper has 1,270 conversations = 635 pairs on 582 pages,
so the released 2018 subset is 51 pairs short. Every awry conversation's `pair_id` points to an on-track
conversation from the same page. The mean length is 4.57 comments (paper: 4.6). Each conversation starts with a
section-header utterance, which we drop, so "comment 1" is the first real comment.

## Method as pipeline stages

`examples/awry-2018.yaml`:
`join_meta {annotation_year}` → `filter {contains 2018}` → `filter {is_section_header == false}` → `first_n {n: 2}`
→ `politeness` (the corpus ships spaCy parses) → `prompt_types` → three `exchange` steps → two `log_odds` steps → five `paired_classify` steps.

New stages (`src/ckviz/stages/awry.py`):
- `exchange`: for each conversation, one feature per (marker, comment position), e.g. `Direct_question@1`,
  binarised (did the marker occur). Politeness uses the paper's **19 strategies per comment (38 features)**.
  These are ConvoKit's 21 minus `HASPOSITIVE`/`HASNEGATIVE`, which are Liu et al.'s opinion lexicon, i.e. the
  paper's separate sentiment baseline. Sentiment is therefore those 2 × 2 comments = **4 features**, as in the paper.
- `prompt_types`: assigns each comment its nearest prompt type, using the distances to the k = 6 types that ConvoKit's
  PromptTypeWrapper stored with the corpus (`vect_info.prompt_types__prompt_dists.6`). A comment whose distance is
  ≥ 1 to every type gets the paper's null type (footnote 7). That gives 6 × 2 = **12 features**. We did not refit
  the types ourselves; using the shipped vectors costs no CPU.
- `log_odds`: share of awry vs on-track conversations that show each marker, the log-odds ratio (+0.5 smoothing),
  and McNemar's exact test on the 584 matched pairs (discordant pairs, binomial at ½). Holm-corrected across markers.
  **Deviation:** the paper's test (footnote 11) is a binomial test of the awry count against the on-track share. That
  treats the on-track share as known and ignores the pairing; in null simulations with 584 pairs it gives 11–17%
  false positives at nominal 5%. McNemar's test holds 5%. Its p-values are larger, so the table below differs
  from what an earlier version of this page reported (binomial p's of 5e-6 … .06).
- `paired_classify`: the balanced task. Each pair becomes the difference of its two feature vectors. For training
  we enter both orders (so the model does not depend on order). The model is ConvoKit's scaled logistic regression
  (C = 1), evaluated with leave-one-page-out CV (536 folds). A tie (zero difference) counts as half right.
  `unigrams` is a 5,000-word CountVectorizer over the exchange text, fitted inside each training fold.
  `tune_C: true` adds a nested grouped 3-fold search over C; the pipeline does not use it (see below).

## Paper vs ours

Table 3 (accuracy %, chance 50; ±3 tolerance fixed in advance; bold = outside it):

| Features | # | Paper | Ours (C=1) |
|---|---|---|---|
| Bag of words | 5,000 | 56.7 | 55.3 |
| Sentiment lexicon | 4 | 55.4 | **51.7** |
| Politeness strategies | 38 | 60.5 | **57.0** |
| Prompt types | 12 | 59.2 | **52.4** |
| Pragmatic (all) | 50 | 61.6 | **58.2** |
| Interlocutor | 5 | 51.2 | not run: the corpus has no edit counts or speaker meta |
| Perspective toxicity / + pragmatic | 2 / 52 | 60.5 / 64.9 | not run: needs the Perspective API (online); the corpus's `toxicity` field is not the score |
| Humans | – | 72 | n/a |

Sensitivity checks (same CV; we looked at these after the main run, so they are not results):
- Politeness with C = 0.001–1: 57.0–58.4.
- All 21 strategies: 56.9–59.4.
- Prompt types: 51.0–59.3, depending on C (C = 0.1 gives 59.3).
- Pragmatic: up to 60.5.
- Nested-tuned C: sentiment 49.5, politeness 57.5, prompts 51.0, pragmatic 60.1. Tuning C does not close the gap
  and costs about 10× the runtime.

With 584 pairs the binomial SE of one accuracy is about 2 points. Prompt types are the least stable.

Figure 2A / Section 5 claims (log-odds awry vs on-track; McNemar exact p, then Holm across the 27 markers tested):

| Marker | Paper | Ours: log-odds, p | Holm p |
|---|---|---|---|
| Direct question, comment 1 | + , p < .001 | +0.56, p = .0033 | .082 |
| 2nd person start, comment 1 | + , p < .001 | +0.52, p = .0042 | .10 |
| Gratitude, comment 1 | − , p < .05 | −0.22, p = .17 | 1 |
| Greeting, comment 1 | − , p < .001 | −0.36, p = .14 | 1 |
| Hedges, comment 1 | − , p < .01 | −0.21, p = .14 (dependency hedge `HASHEDGE`: −0.23, p = .048) | 1 |
| 2nd person, comment 2 | + , p < .001 | +0.39, p = .0007 | .019 |
| 1st person start, comment 2 | − , p < .001 | −0.22, p = .073 | 1 |

Prompt types (11 tested): `type_1@1` −0.33, p = .004 (Holm .046); `type_2@2` −0.71, p = .016; `type_5@1` +0.49,
p = .022 (neither survives Holm). The shipped vectors carry no type names, so we did not map them onto the paper's
Factual check / Coordination / Opinion.

## Verdict: partially reproduced

- **Figure 2A: reproduced in direction, only partly in significance.** All 7 markers named in the text have the
  paper's sign. The three directness markers at the core of the paper's story are significant uncorrected
  (direct question@1 p = .003, 2nd person start@1 p = .004, 2nd person@2 p = .0007), but not at the paper's p < .001,
  and only 2nd person@2 survives Holm correction (p = .019). Gratitude@1, greeting@1, hedges@1 and 1st person
  start@2 miss p < .05 (p = .07–.17). The paper's stronger p's come from its anti-conservative binomial test (see
  `log_odds` above), not just from its larger sample.
- **Table 3: BOW reproduced (55.3 vs 56.7). Politeness and pragmatic still beat BOW (within noise); prompt types do
  not** (52.4 vs 55.3 here, 59.2 vs 56.7 in the paper). **The pragmatic accuracies are 3–7 points low,** and sentiment
  is near chance.

Likely causes:
1. 51 fewer pairs (a different, smaller 2018 subset).
2. ConvoKit 5 / spaCy parses instead of the 2018 parser and politeness code.
3. Prompt types from the shipped 2019 vectors, not the paper's own fit.
4. Unreported modelling details. The paper says only "logistic regression". We do not know its regularisation,
   pair encoding or feature selection, and the prompt-type result moves about 7 points with C alone.

## Runtime

About 3 min on a laptop: politeness 7 s, BOW leave-one-page-out 143 s, the other classifiers 3–15 s each.
`CKVIZ_EXAMPLES=1 uv run pytest tests/test_paper_awry.py`.
