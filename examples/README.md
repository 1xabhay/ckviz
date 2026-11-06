# Examples

Each `*.yaml` is a ckviz pipeline. Those in the first table reproduce a ConvoKit example notebook.
`tests/test_examples.py` checks their output against the notebook's saved results, except for the two ⚠️ drift rows,
which it checks against a re-run of the notebook's own code (see the drift notes below):

```sh
CKVIZ_EXAMPLES=1 uv run pytest tests/test_examples.py   # downloads ~330 MB of corpora on first run, ~6 min
```

They all show up in the app's pipeline picker.

## Data you need

- **In the repo, ready to run:** `synthetic`, an invented corpus built by [`data/make_synthetic.py`](data/make_synthetic.py) ([README](data/synthetic/README.md)). Used by `phq9-fighting-words` and `dlatk-phq9`.
- **Downloaded by ConvoKit on first run:** every other dataset named with `convokit:` in [`datasets.yaml`](../datasets.yaml), which covers the notebook table, the politeness and Echoes `wiki` pipelines, and `awry-2018`.
- **You build it:** `absolutist-*` needs `absolutist-forums`, built from the authors' Figshare files ([steps](papers/absolutist.md)). `echoes-of-power-supreme` needs `supreme-echoes-2012`, a cut of `supreme-corpus` whose filter script is not in the repo ([description](papers/echoes-of-power.md)).
- **You bring it:** `daily-ping-engagement` and `daily-ping-descriptive` need your own longitudinal chat data, registered in `datasets.yaml`.

| Example | Notebook | Dataset | Result |
|---|---|---|---|
| `intro-fighting-words` | Introduction_to_ConvoKit | reddit-corpus-small | ✅ matches (god 9.66, sin 9.20 / money −7.96, religion −7.77) |
| `sigdial-mixed-gender-casefold` | sigdial-demo | movie-corpus | ✅ matches (the −29.13, sir −25.21, man −19.33, …) |
| `sigdial-mixed-gender` | sigdial-demo | movie-corpus | ⚠️ dataset drift: run verbatim, see below |
| `politeness-classifier` | politeness_demo | wikipedia-politeness-corpus | ✅ mean 5-fold accuracy within 0.02 (notebook folds are unseeded) |
| `politeness-admin` | Politeness_Marker_and_Summarize_Demo | wiki-corpus (first 4,001 utterances) | ⚠️ ConvoKit drift: matches ConvoKit 5 verbatim, not the saved output |
| `bow-score` | bag-of-words-demo | subreddit-Cornell | ✅ matches (hotels 1.270, hbhs 1.116, … / creation −0.785) |

## Invented corpus

These reproduce nothing: they run on [`data/synthetic`](data/synthetic/README.md), where the score is arbitrary and
the word differences are planted, so a pipeline passes if it recovers what was planted. They are demos, not findings.

| Example | Planted | Result |
|---|---|---|
| `phq9-fighting-words` | compost rises with the score, bicycle falls | ✅ top words on each side (compost z 5.32 / bicycle z −5.54) |
| `dlatk-phq9` | same | ✅ top 2 of 294 unigrams (bicycle r −0.96, compost r 0.92), same words as a direct count |

## Published papers

Write-ups with paper-vs-ours tables in [`papers/`](papers/). Slow tests: `CKVIZ_EXAMPLES=1 uv run pytest tests/test_paper_*.py plugins/ckviz-stats/tests/test_paper_absolutist_mixed.py`.

| Paper | Domain | Pipelines | Verdict |
|---|---|---|---|
| Al-Mosaiwi & Johnstone 2018, *Clin. Psych. Sci.*: [absolutist words](papers/absolutist.md) | psychology | `absolutist-study1`, `absolutist-study2-3`, `absolutist-study1-mixed` | ✅ all 9 group means and 8 Study 2/3 d's within 0.02; Study 1 mixed model (stats plugin, [write-up](papers/absolutist-mixed.md)) on the authors' data, F within 0.001 and 5 d's to every printed digit; within 3% from raw text |
| Danescu-Niculescu-Mizil et al. 2013, ACL: [politeness](papers/politeness-2013.md) | behavioral | `politeness-2013-strategies`, `politeness-2013-classifiers` | ◐ Table 3 all 20 strategies; Table 4 cells tested on Wikipedia match, cells tested on Stack Exchange 4–14 pts low |
| Danescu-Niculescu-Mizil et al. 2012, WWW: [Echoes of Power](papers/echoes-of-power.md) | organizational | `echoes-of-power-wiki`, `echoes-of-power-supreme` | ◐ all directions hold (admins, Justices); weaker significance, Supreme Court gap ~half |
| Zhang et al. 2018, ACL: [Conversations Gone Awry](papers/conversations-gone-awry.md) | behavioral / psychology | `awry-2018` | ◐ BOW 55.3 vs 56.7 and all 7 named markers have the paper's sign; with a paired (McNemar) test only 2nd person@2 survives Holm, so significance reproduces only in part; pragmatic accuracies 3–7 pts low (584 of 635 pairs released; Perspective API rows skipped) |

## Plugins

| Example | Plugin | Result |
|---|---|---|
| `dlatk-phq9` | [`ckviz-dlatk`](../plugins/ckviz-dlatk/README.md) (DLATK 1.3.15 via uvx) | ✅ see invented corpus above |
| `absolutist-study1-mixed` | [`ckviz-stats`](../plugins/ckviz-stats/README.md) (statsmodels) | ✅ see absolutist row above |

## Daily-ping pipelines

Templates for longitudinal chat data, where a bot pings each participant every day. They reproduce nothing, and you supply the data.

| Example | What it shows |
|---|---|
| `daily-ping-descriptive` | descriptive overview of the messages and participants |
| `daily-ping-engagement` | engagement over the study days |

**Drift notes.** Both ⚠️ rows were checked by running the notebook's own code, unchanged, on ConvoKit 5.0.0:
ckviz gives the same numbers as that run, so the difference is upstream, not in ckviz.

- *movie-corpus* now stores most genders lowercase (`m` 1,899, `f` 921 vs `M` 150, `F` 45). The notebook's
  case-sensitive `'M' in genders and 'F' in genders` now flags 4,574 utterances as mixed instead of 100,931.
  The `-casefold` variant (`ignore_case: true`) restores the original result.
- *politeness* strategies on wiki-corpus differ from the saved output on 8/21 features (e.g. 1st_person
  0.18 → 0.04, HASHEDGE 0.10 → 0.00) under ConvoKit 5 / spaCy 3.8; the saved output came from an older stack.

**ConvoKit 5.0.0 issues worked around** (in `src/ckviz`):
`Classifier(pred_feats=...)` and `VectorClassifier` are broken, so classifier stages build ConvoKit's default model
(`StandardScaler(with_mean=False)` + `LogisticRegression(solver="liblinear")`) directly; `Corpus.from_pandas`
mutates the DataFrames passed to it, so `to_corpus` passes copies.

Not reproduced: `scd-example` (bespoke nested-summary extraction), `coordination` (only a chart image saved),
`hyperconvo`, `prompt-types`, `speaker-convo-diversity` (need new stages), forecasting / LLM notebooks (GPU).
