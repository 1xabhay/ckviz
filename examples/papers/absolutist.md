# In an Absolute State (Al-Mosaiwi & Johnstone, Clinical Psychological Science 2018)

Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC6376956/ (6(4):529–542) · Pipelines: `examples/absolutist-study1.yaml`,
`examples/absolutist-study2-3.yaml` · Stage: `lexicon_net` (`src/ckviz/stages/absolutist.py`) ·
Test: `CKVIZ_EXAMPLES=1 uv run pytest tests/test_paper_absolutist.py`

## Question
Do people on anxiety, depression and suicidal-ideation forums use more absolutist words (*always, nothing, completely…*)
than people on control forums (Study 1)? Is the effect specific to disorders linked to absolutist thinking, BPD and eating
disorders, compared with PTSD and schizophrenia (Study 2)? Does it persist on depression-recovery forums (Study 3)?

## Data
Figshare, CC BY 4.0. Texts: doi:10.6084/m9.figshare.4743715, 6,744 `.txt` files, one per forum **member**, holding that
member's first posts. Aggregates: doi:10.6084/m9.figshare.4743547, the authors' SPSS/LIWC workbook. Its `Original`
sheet maps each file to forum + condition and gives per-member LIWC `WC`, `Absol` and `AbsolRev`, where
`AbsolRev` is the false-positive index and `Absolutist = Absol − AbsolRev`; we checked that identity to 1e-17.

`examples/papers/absolutist/to_convokit.py` builds `data/papers/absolutist/corpus`. To use it, uncomment
`absolutist-forums` in `datasets.yaml`. In the corpus a forum is a conversation (63 of them, with `meta.group` and
`meta.study`), and a member is a speaker with one utterance. The authors' LIWC numbers are kept as
`meta.liwc_wc|liwc_absolutist|liwc_fp` so ours can be checked against them. The script fixes the files' mixed
cp1252/UTF-8/UTF-16 encodings and leading no-break spaces in the sheet's file names. It drops 2 files the sheet lists
but Figshare lacks (`Anxiety802`, `Suicide076`). Build it:

```sh
mkdir -p data/papers/absolutist/raw/texts   # the zip unpacked into texts/, the xlsx saved as raw/stats.xlsx
uv run python -I examples/papers/absolutist/to_convokit.py data/papers/absolutist/raw data/papers/absolutist
```

**Unit of analysis.** The unit is the **member**, not the forum. The paper's Ns (2,373 control, 597 anxiety…) count
members with ≥ 100 words, and its M/SD are member-level. Forums enter its SPSS `MIXED` model as a random effect, with
residuals weighted by WC and the outcome log10(x+1) transformed (supplement `.sps`).

## Method as stages
1. `lexicon_net`. This is a new generic stage: the `lexicon` rate minus the rate of excluded phrases. It uses the same
   tokens as `lexicon`, which match LIWC's word count to within 1% for 97% of members.
   - **words**: the 19 words of Table S2 (verified from the supplement). Four of them become LIWC wildcards:
     `absolut*`, `entire*`, `total*`, `everyone*`. The published list alone matched the authors' per-member scores for
     93.0% of members. A forward search over candidate stems found that these four raise it to 97.6%: the authors'
     `.dic` evidently had them, e.g. it counts *absolute hell* and *total number*.
   - **exclude**: the paper describes the false-positive phrases but never publishes them (they are not in the supplement
     either). We rebuilt the list (63 phrases) from the authors' per-member FP index:
     - *hi/hello/hey/dear/thanks/thank you* + *all/everyone*
     - *not/almost* + an absolutist word
     - *full time*
     - *every few/2/3/two…*

     Our FP counts equal theirs for 96.6% of members, r = .92, with 1,381 vs 1,417 total hits.
2. `filter n_tokens >= 100`. The paper kept members with ≥ 100 words.
3. `compare_groups` at the utterance (= member) level, once per contrast. It gives means and Cohen's d (pooled SD).
   The group is constant within each forum, so the stage withholds its Mann-Whitney p at this level.
4. Study 1 only: `rollup` to conversation (= forum) mean, then `compare_groups level: conversation`.

## Paper vs ours
Means are % of words, members with ≥ 100 words.

| group | N paper / ours | M paper | M ours |
|---|---|---|---|
| control (19 forums) | 2373 / 2374 | 0.97 | 0.97 |
| anxiety | 597 / 596 | 1.45 | 1.45 |
| depression | 529 / 529 | 1.45 | 1.45 |
| suicidal ideation | 368 / 367 | 1.80 | 1.81 |
| PTSD | 534 / 533 | 1.13 | 1.13 |
| schizophrenia | 591 / 593 | 1.14 | 1.14 |
| eating disorder | 547 / 547 | 1.25 | 1.25 |
| BPD | 326 / 326 | 1.47 | 1.46 |
| recovery | 558 / 558 | 1.31 | 1.30 |

**Study 2/3 effect sizes.** The paper's d here is member-level Cohen's d: it equals (M1−M2)/pooled SD from its own
descriptives. Ours is the same estimator.

| contrast | d paper | d ours | p paper (mixed) | p ours (MWU), from an earlier run, before the nesting check; the current pipeline withholds it |
|---|---|---|---|---|
| BPD > PTSD | 0.36 | 0.37 | < .001 | 1.4e-6 |
| BPD > schizophrenia | 0.35 | 0.35 | < .001 | 1.6e-6 |
| ED > PTSD | 0.13 | 0.13 | .081 (n.s.) | .072 |
| ED > schizophrenia | 0.12 | 0.12 | .009 | .077 |
| recovery > control | 0.37 | 0.38 | < .001 | 3.7e-16 |
| anxiety > recovery | 0.15 | 0.15 | .018 | .0067 |
| depression > recovery | 0.15 | 0.15 | .018 | .011 |
| SI > recovery | 0.50 | 0.52 | < .001 | 1.5e-14 |

**Study 1 effect sizes.** The paper's d is 2t/√df from the mixed model (the formula is in the workbook). The
`mixed_model` stage fits that model, and [`absolutist-mixed.md`](absolutist-mixed.md) compares its results with the
paper's. The `absolutist-study1` pipeline on this page doesn't use it: it reports forum-level Cohen's d, whose reference
point is the same estimator applied to the authors' own LIWC scores.

| contrast | paper d (Table 2) | workbook mixed-model d | forum-level d, authors' scores | forum-level d, ours | forum p ours |
|---|---|---|---|---|---|
| anxiety > control | 3.24 | 3.21 | 4.04 | 4.04 | 1.1e-5 |
| depression > control | 3.14 | 3.06 | 4.70 | 4.61 | 1.1e-5 |
| SI > control | 4.56 | 4.99 | 8.12 | 8.35 | 2.3e-4 |
| SI > anxiety | 1.74 | 2.14 | 3.10 | 3.17 | .0095 |
| SI > depression | 1.71 | 2.13 | 4.42 | 4.20 | .0095 |
| anxiety vs depression | n.s. (p = 1.00) | — | −0.12 | −0.11 | 1.00 |

Member-level d for Study 1 is 0.54 / 0.55 / 0.96 / 0.37 / 0.36. The omnibus F(3, 29) = 71.55 is in
[`absolutist-mixed.md`](absolutist-mixed.md): 71.550 on the authors' data, 73.42 from our text pipeline. Not attempted:
the LIWC emotion/pronoun/certainty comparison dictionaries, Table 2's other rows and the ANCOVA (LIWC2015 is
proprietary).

## Verdict: reproduced (descriptives, Study 2/3 effect sizes, and Study 1's d and F with the mixed model); Study 2/3 significance not reproduced
- **Reproduced.** All 9 group means are within 0.01 percentage points, and Ns are within 2 members. All 8 Study 2/3
  d values are within 0.02. Every hypothesis comes out in the same direction and order: SI > anxiety ≈ depression >
  control; BPD > ED > PTSD ≈ schizophrenia; recovery between control and depression. The result is robust to the
  dictionary details. With the strictly published 19 words plus our exclusions, means move by ≤ 0.03; with no
  false-positive exclusion they rise by 0.04–0.11, but the ordering is unchanged.
- **Study 1 d values: reproduced with the mixed model, in [`absolutist-mixed.md`](absolutist-mixed.md).** The paper's
  d values (3.24, 3.14, 4.56, 1.74, 1.71) come from a WC-weighted mixed model with a log10 outcome. Fitting that model,
  `absolutist-study1-mixed` gives all five to every printed digit on the authors' data, and within 2% from our raw text.
  The workbook's 3.21, 3.06, 4.99, 2.14, 2.13 come from the same model on the untransformed outcome. That page also
  explains why 2t/√df is better read as a test statistic than as an effect size.
- **Study 1 forum-level check (this page).** The forum-level estimator is not the paper's. It gives the same d on our
  scores as on the authors' scores (within 5%), and all five contrasts are significant at the forum level (n = 4–19
  forums).
- **Study 2/3 significance: not reproduced.** The paper's p-values come from a mixed model, which we fit only for Study
  1, and the current pipeline withholds member-level p-values. The earlier run's Mann-Whitney p-values treated members
  as independent, so they aren't a replacement. In that run, ED vs schizophrenia was .077 against the paper's .009.

## Deviations
1. Dictionary wildcards were inferred, not published.
2. The false-positive phrase list was reverse-engineered from the authors' per-member FP index; the paper does not
   publish it.
3. We used our tokenizer's ≥ 100 cut, not LIWC's WC, which moves the control N by 1.
4. Two members are missing from Figshare.
5. In `absolutist-study1`, forum-level Cohen's d stands in for the mixed-model d. `absolutist-study1-mixed` fits the
   mixed model.
6. Building the corpus needs `openpyxl` to read the xlsx. It is in the project's dev dependency group.

## Runtime
Building the corpus takes about 15 s. `absolutist-study1` runs in about 13 s, almost all of it `lexicon_net` over
6,743 members × 63 phrases. `absolutist-study2-3` takes about 13 s (instant when the stage is cached). The slow test
takes about 26 s.
