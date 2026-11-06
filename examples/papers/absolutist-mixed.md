# In an Absolute State: Study 1 with the mixed model

Companion to `absolutist.md`, which covers the data, the dictionary and Studies 2–3. Pipeline:
`examples/absolutist-study1-mixed.yaml`. Stages: `mixed_model`, `anova` (plugin `plugins/ckviz-stats`). Test:
`plugins/ckviz-stats/tests/test_paper_absolutist_mixed.py`.

## Study 1 with the mixed model
The supplement's `.sps` runs SPSS `MIXED` on log10(Absolutist% + 1), with condition as the fixed effect and a
random forum intercept. It uses REML, residuals weighted by word count (`/REGWGT=WC`) and Satterthwaite df, and
computes d = 2t/√df. `mixed_model` fits the same model and reports that number as `d_paper`, next to a
standardized `d` = estimate / √(τ² + σ²). Our pipeline:
- `lexicon_net` (as in Study 1)
- keep members with ≥ 100 words, and the 4 Study 1 groups (35 forums, 3,866 members)
- `mixed_model` with `scale: 100, transform: log10p1, weights: n_tokens, cluster: conversation_id`

The authors' Figshare workbook runs the same model on the **untransformed** %, so we fit that version too.

`d_paper` (= the paper's d, 2t/√df):

| | F (df) | anx>ctl | dep>ctl | SI>ctl | SI>anx | SI>dep |
|---|---|---|---|---|---|---|
| paper (log10) | 71.549 (3, 29) | 3.24 | 3.14 | 4.56 | 1.74 | 1.71 |
| authors' data, our stage, log10 | 71.550 (3, 28.67) | 3.243 | 3.141 | 4.564 | 1.736 | 1.708 |
| our text pipeline, log10 | 73.42 (3, 28.88) | 3.262 | 3.170 | 4.612 | 1.763 | 1.731 |
| workbook SPSS (untransformed) | 80.496 (3, 29.29) | 3.212 | 3.062 | 4.987 | 2.143 | 2.125 |
| authors' data, our stage, untransformed | 80.496 (3, 29.29) | 3.212 | 3.062 | 4.987 | 2.143 | 2.125 |
| our text pipeline, untransformed | 82.51 (3, 29.36) | 3.244 | 3.087 | 5.047 | 2.172 | 2.161 |

"Authors' data" is the workbook's `Hypothesis 1 data` sheet: their 3,867 members and their LIWC scores.

**2t/√df is not Cohen's d.** t grows with the number of members per forum while df stays near the number of forums
(~29), so 2t/√df is several times a standardized mean difference. The stage's `d` divides the estimate by the total
SD (random intercept + residual) instead:

| standardized `d` | anx>ctl | dep>ctl | SI>ctl | SI>anx | SI>dep |
|---|---|---|---|---|---|
| authors' data, log10 | 0.586 | 0.585 | 1.027 | 0.441 | 0.441 |
| our text pipeline, log10 | 0.589 | 0.590 | 1.037 | 0.448 | 0.447 |
| authors' data, untransformed | 0.540 | 0.535 | 1.046 | 0.506 | 0.512 |
| our text pipeline, untransformed | 0.543 | 0.536 | 1.054 | 0.510 | 0.517 |

These are medium-to-large effects, in line with the member-level d's of Studies 2–3 (`absolutist.md`), not the
"d > 3" the paper's Table 2 suggests. We reproduce the paper's 2t/√df exactly, but read it as a test statistic,
not as an effect size.

**Verdict: reproduced.**
- On the authors' data, our stage gives SPSS's output to every printed digit: t, the Satterthwaite df (e.g. 28.515)
  and F. So the method, including how df are computed, is not a source of difference.
- The workbook's numbers differ from the paper's only because the workbook fits the untransformed outcome and the
  paper fits log10. This explains the gap noted in `absolutist.md`.
- From raw text, our `d_paper` values are within 2% of the paper's and F is within 3%. The remaining difference is the data:
  our tokenizer and rebuilt dictionary, and 2 members missing from Figshare.
- All contrasts have p < .001, as in the paper.
- A one-way ANOVA on forum means gives F(3, 31) = 90.7 and η² = .90.

**Deviation.** statsmodels' default optimizer stops short of the optimum on these data, so the stage fits with
cg/powell and keeps the best fit (see `plugins/ckviz-stats/README.md`).
