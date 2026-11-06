# ckviz-stats

Stages built on statsmodels for models the built-in stages lack.

| stage | what it does |
|---|---|
| `anova` | One-way ANOVA of a numeric column across groups, at the utterance, speaker or conversation level. Reports F, df, p and η², plus n, mean and SD per group. `welch: true` runs Welch's ANOVA (`statsmodels.stats.oneway.anova_oneway`). |
| `mixed_model` | `y ~ group + (1 \| cluster)`, fitted by REML. The group is dummy-coded against `reference`. Optional `scale` and `transform: log10p1` fit log10(scale·y + 1). Optional `weights` are residual weights, as in SPSS `/REGWGT`: Var(eᵢ) = σ²/wᵢ. For each contrast (`"a vs b"`; the default is each group vs the reference) it reports the estimate, SE, Satterthwaite df, t, p, `d` = estimate / √(τ² + σ²) (standardized by the total SD, comparable to Cohen's d) and `d_paper` = 2t/√df (what the SPSS-style papers report; with few large clusters it is several times the standardized difference, so don't read it as Cohen's d). It also gives an omnibus F with a Satterthwaite denominator df (Fai & Cornelius, as in SAS and lmerTest). |
| `mixed_logit` | Generic binomial mixed model, `logit P(y) ~ predictor [+ covariates] [+ ns(spline, df)] + (1 \| g1) [+ (1 \| g2)]`: one or two grouping factors with random intercepts, crossed if two, e.g. `outcome ~ x + (1 \| person) + (1 \| item)`. Predictor coded against `reference` (`coding: treatment`) or vs the grand mean (`coding: sum`, all levels reported). Covariates are centred; the spline is patsy's natural cubic `cr(x, df, constraints="center")`. Per fixed effect: posterior mean, SD, 95% interval, odds ratio with interval; random-effect SDs; and an interval chart of the adjusted probability per level (covariates at their mean, random effects 0). |

**mixed_logit is approximate.** Step 1: statsmodels' `BinomialBayesMixedGLM.fit_vb` (mean-field variational
Bayes; priors: fixed effects N(0, 5²), log random-effect SDs N(0, 1)) gives the random-effect SDs. Step 2 (`method:
laplace`, default): with those SDs fixed, penalized IRLS finds the joint mode of fixed and random effects, and the
fixed-effect covariance is the inverse Hessian: lme4's Laplace step, so estimates and Wald intervals are conditional on
the SDs. `method: vb` reports the VB fixed effects as they are. Mean-field VB treats the effects of an item-level predictor as
independent of the item intercepts, although the predictor is constant within an item, so its intervals are too narrow.

Coverage check: 40 seeded simulations (seeds 100 to 139; 40 people × 20 items, crossed person and item intercepts with SD 1
and 0.3; `test_mixed_logit.py`, `CKVIZ_EXAMPLES=1`). The nominal 95% intervals for the three item-level effects covered the
truth 115 of 120 times (96%) with laplace and 94 of 120 (78%) with vb. A one-off script on the same simulations gave the
rest: statsmodels' `fit_map` (the joint posterior mode) covered 109 of 120 (91%), with the item SD collapsing to 0 in every
rep; VB's item SD averaged 0.25 (truth 0.3); and the nominal 95% intervals for the SDs held the truth in 23 of 40 reps for
item and 32 of 40 for person, so they are too narrow.

Use the stage to explore. For reported results, refit in R with `lme4::glmer` (Laplace; with crossed effects use
`confint(method = "profile")` or a parametric bootstrap) or fit a full Bayesian model with `brms`/PyMC. I found no
well-validated pure-Python frequentist binomial GLMM with crossed effects. PyMC/bambi (NUTS) would give
better-calibrated intervals, but they are a heavy dependency and slow for each fit.

**Weights.** `MixedLM` has no weights, so the stage fits the model with every row multiplied by √w, and uses √w as
the random-effect design. That is exactly the weighted model: the REML likelihood changes only by a constant. Weights
are not frequency weights. Multiplying all of them by a constant changes nothing.

**Optimizer.** statsmodels' default optimizer (bfgs) stops short of the optimum on the absolutist data: the REML is
0.04 too low and the SEs are 4% off SPSS. The stage therefore fits with `cg` and with `powell` and keeps the fit with
the higher likelihood.

**SEs and df.** SEs are (X′V⁻¹X)⁻¹, as in SPSS and lme4. statsmodels' own `bse_fe` comes from the full Hessian and
differs from these by about 0.1%. statsmodels has no Satterthwaite df, so the stage computes them from a closed-form
REML likelihood: numerical gradient of the contrast variance, and the observed information of (τ², σ²). On the
authors' data they match SPSS to every printed digit (`tests/test_paper_absolutist_mixed.py`).

Limits:
- One random intercept only.
- If τ² lands on the boundary (0), the df use a one-sided-ish finite-difference step, so treat them with caution.

Install: `uv sync` installs it in this repo. Tests:

```sh
uv run pytest -q plugins/ckviz-stats
CKVIZ_EXAMPLES=1 uv run pytest plugins/ckviz-stats/tests/test_mixed_logit.py -q   # the 40-rep coverage check
CKVIZ_EXAMPLES=1 uv run pytest -q plugins/ckviz-stats/tests/test_paper_absolutist_mixed.py   # needs the absolutist corpus + Figshare xlsx
```
