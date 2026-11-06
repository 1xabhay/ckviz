# Echoes of power (Danescu-Niculescu-Mizil, Lee, Pang & Kleinberg, WWW 2012)

Paper: https://arxiv.org/abs/1112.3670 · Pipelines: `examples/echoes-of-power-wiki.yaml`, `examples/echoes-of-power-supreme.yaml`
· Test: `CKVIZ_EXAMPLES=1 uv run pytest tests/test_paper_echoes.py`

## Question
Does power show in how much people echo the function-word style of the person they reply to?
- **P_target**: people coordinate more toward high-power targets.
- **P_speaker**: high-power people coordinate less. On Wikipedia the paper found the opposite (admins coordinate *more*)
  and explains it as a personality effect.

## Data
- **Wikipedia**: ConvoKit `wiki-corpus` (391,294 utterances; the paper used 240,436 talk-page exchanges). Admin status
  is `meta.is-admin` on both speakers (current status) and utterances (status when the utterance was posted).
- **Supreme Court**: ConvoKit `supreme-corpus`, cut down to the paper's 204 cases (the case list from ConvoKit's
  coordination notebook). That leaves 50,796 utterances in 202 conversations; the paper reports 50,389 exchanges.
  To use the cut, uncomment `supreme-echoes-2012: {dir: data/supreme-echoes-2012}` in `datasets.yaml`. It keeps the lines of
  `utterances.jsonl` whose `meta.case_id[5:]` is in that list, plus the matching conversations, all speakers, and
  `corpus.json`/`index.json`. Building it takes about 8 s with a short stream-filter script (not in the repo).

## Method as stages
One `coordination` stage (`src/ckviz/stages/coordination.py`) wraps ConvoKit's `Coordination`:
C^m(b, A) = P(b's reply uses marker m | the target's utterance uses m) − P(b's reply uses m), for 8 LIWC
function-word markers. These are averaged over people, and the paper's three aggregates handle missing markers:
(1) only people with all 8 markers; (2) missing markers filled with the group mean; (3) the mean of the
markers each person has. `group_by` + `side` split the targets or the speakers into groups. With two groups the
stage reports diff and an independent t-test per measure, as the paper did.

| Figure | stage config |
|---|---|
| 1(a) | `group_by: meta.is-admin, side: target, focus: targets, target_thresh: 7, split_by: [meta.is-admin]` |
| 1(b) | `group_by: meta.is-admin, side: speaker, speaker_thresh: 7, target_thresh: 7, split_by: [meta.is-admin]` |
| 4 | `group_by: meta.type, side: speaker`, speakers/targets restricted to A\|J, thresholds 3, `split_by: [meta.case_id]` (each person in each case counts separately, as in paper footnote 14) |

The Wikipedia settings are copied from ConvoKit's coordination notebook, which says it reproduces Figure 1. The
Figure 4 thresholds aren't given in the paper; 3/3 is taken from the notebook's Figure 5 cell.

## Paper vs ours
Scores ×100. Paper values are read off its bar charts (about ±0.1). Our p-values come from an independent t-test.

| Comparison | measure | paper high / low | ours high / low | ours p | paper sig. |
|---|---|---|---|---|---|
| 1(a) targets: admins vs non-admins | agg 1 | ~1.7 / 1.5 | 1.91 / 1.72 | 0.16 | * |
| | agg 2 | ~1.65 / 1.4 | 1.87 / 1.65 | 0.033 | ** |
| | agg 3 | ~1.6 / 1.2 | 1.75 / 1.44 | 0.023 | *** |
| | n (all 8, any) | (805, 1027) / (2307, 3637) | (966, 1370) / (2208, 3548) | | |
| 1(b) speakers: admins vs non-admins | agg 1 | ~1.65 / 1.5 | 1.92 / 1.68 | 0.038 | * |
| | agg 2 | ~1.65 / 1.45 | 1.88 / 1.62 | 0.0047 | ** |
| | agg 3 | ~1.55 / 1.3 | 1.76 / 1.47 | 0.014 | ** |
| | n | (805, 1027) / (2029, 2856) | (871, 1203) / (1804, 2775) | | |
| 4 lawyers→Justices vs Justices→lawyers | agg 1 | ~3.2 / 2.1 | 2.77 / 2.29 | 0.046 | *** |
| | agg 2 | ~3.1 / 1.9 | 2.73 / 2.23 | 0.027 | *** |
| | agg 3 | ~3.1 / 1.8 | 2.72 / 2.12 | 0.012 | *** |
| | ppron / ipron / auxverb | all *** | 3.68/1.89, 2.32/1.38, 1.70/0.68 | 8e-6, 0.008, 1e-4 | *** |
| | n | (500, 535) / (815, 1307) | (511, 518) / (1110, 1432) | | |

The Wikipedia populations and bar heights match ConvoKit's notebook exactly. Its saved charts show 966/1370 and
2208/3548 for 1(a), 871/1203 and 1804/2775 for 1(b), and the same aggregate heights to within 0.0005.

## Verdict: **partially reproduced**
- **Directions: reproduced.** Everyone coordinates more toward admins (1a), admins coordinate more as speakers
  (1b), and lawyers coordinate more to Justices than Justices do to lawyers (4). This holds for every aggregate.
- **Significance: weaker than the paper.** Fig. 1(a) aggregate 1 isn't significant (p = 0.16, paper *), and
  Fig. 4 is only p < 0.05 where the paper reports p < 0.001.
- **Magnitudes:** Wikipedia scores are slightly higher than the paper's, and the lawyer–Justice gap is about half
  the paper's.
- **Likely causes:**
  - The ConvoKit corpora are not the paper's exact data: different exchange counts, and current admin status
    instead of RfA-timed status.
  - ConvoKit's marker matcher differs from the paper's on purpose (see its docstring on apostrophes).
  - The paper's thresholds for Fig. 4 are unknown.
- **Not attempted:**
  - Fig. 2/3 (RfA timing, admins-to-be): needs promotion dates, which the corpus lacks.
  - Fig. 5 (favorable/unfavorable Justices): needs an utterance-pair selector that the stage doesn't expose.
  - Fig. 6 (gender), the Chief Justice comparison, and Table 2 (SVM).

## Runtime (Apple Silicon laptop)
- **Wikipedia:** about 4 min end to end. Loading takes 57 s on the first load (then cached), and each
  coordination stage takes 83–96 s, because each converts to a Corpus and refits. Peak RSS is under 1 GB.
- **Supreme subset:** 8–15 s total, peak RSS 0.44 GB.
