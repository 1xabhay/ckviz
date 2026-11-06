"""ckviz stages wrapping statsmodels: one-way ANOVA, a random-intercept linear mixed model (SPSS MIXED-style) and an
approximate Bayesian binomial mixed model (mixed_logit)."""
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ckviz.frames import Frames
from ckviz.stages import Level, at, need, sig, stage


def g6(x: float) -> float:
    return float(f"{x:.6g}")


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(need(df, col), errors="coerce")


class Anova(BaseModel):
    level: Level = "utterance"
    column: str = Field(description="numeric column, e.g. meta.absolutist")
    group_by: str = Field(description="column holding the group labels")
    welch: bool = Field(False, description="Welch's ANOVA (no equal-variance assumption)")


@stage(Anova)
def anova(f: Frames, c: Anova):
    """One-way ANOVA of a column across groups: F, df, p, eta squared, plus per-group n / mean / SD."""
    from statsmodels.stats.oneway import anova_oneway

    df = at(f, c.level)
    d = pd.DataFrame({"y": numeric(df, c.column), "g": need(df, c.group_by).astype(str)}).dropna()
    if d.g.nunique() < 2:
        raise ValueError(f"need at least 2 groups in {c.group_by!r}; have {d.g.unique().tolist()}")
    r = anova_oneway(d.y.to_numpy(), d.g.to_numpy(), use_var="unequal" if c.welch else "equal")
    desc = d.groupby("g").y.agg(["count", "mean", "std"])
    eta2 = (desc["count"] * (desc["mean"] - d.y.mean()) ** 2).sum() / ((d.y - d.y.mean()) ** 2).sum()
    df1, df2 = (float(x) for x in r.df)
    s = {"F": g6(r.statistic), "df1": g6(df1), "df2": g6(df2), "p": sig(r.pvalue), "eta2": round(float(eta2), 4), "n": len(d)}
    rows = [[k, int(n), g6(m), g6(sd) if sd == sd else None] for k, (n, m, sd) in desc.iterrows()]
    kind = "Welch's" if c.welch else "one-way"
    return f, {"kind": "table", "columns": ["group", "n", "mean", "sd"], "rows": rows, "total": len(rows), "summary": s,
               "note": f"{kind} ANOVA: F({s['df1']:.4g}, {s['df2']:.4g}) = {s['F']:.4g}, p = {s['p']}, η² = {s['eta2']} ({c.level}s)"}


class MixedModel(BaseModel):
    level: Level = "utterance"
    column: str = Field(description="numeric outcome column")
    group_by: str = Field(description="categorical fixed effect, e.g. meta.group")
    cluster: str = Field("conversation_id", description="random-intercept cluster column, e.g. conversation_id (forum)")
    reference: str = Field("", description="reference group for the dummy coding; default: first in sorted order")
    contrasts: list[str] = Field([], description="'a vs b' pairs to report (a - b); default: each group vs the reference")
    transform: Literal["none", "log10p1"] = Field("none", description="log10p1 fits log10(scale*y + 1)")
    scale: float = Field(1.0, description="multiply y by this first, e.g. 100 turns a share into %")
    weights: str = Field("", description="residual-weight column (SPSS /REGWGT): Var(e_i) = sigma^2 / w_i; e.g. n_tokens")


class Fit:
    """Weighted random-intercept model y = Xb + u_cluster + e, Var(u) = t2, Var(e_i) = s2 / w_i.

    statsmodels' MixedLM has no weights, so it is fitted on rows scaled by sqrt(w) with sqrt(w) as the random-effect
    design: exactly the weighted model (the REML likelihood changes only by a constant). Satterthwaite df, which
    statsmodels lacks, come from the closed-form REML likelihood below (per-cluster Woodbury sums)."""

    def __init__(self, y, X, codes, w):
        import warnings

        import statsmodels.api as sm
        from statsmodels.tools.sm_exceptions import ConvergenceWarning

        w = w / w.mean()
        s = np.sqrt(w)
        model = sm.MixedLM(y * s, X * s[:, None], groups=codes, exog_re=s[:, None])
        # The default (bfgs) stops short on the absolutist data (REML 0.04 below the optimum, SEs 4% off SPSS);
        # cg and powell both reach it. Keep the better of the two.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            m = max((model.fit(reml=True, method=k) for k in ("cg", "powell")), key=lambda r: r.llf)
        self.theta = np.array([float(np.asarray(m.cov_re)[0, 0]), float(m.scale)])
        J = codes.max() + 1
        wx = X * w[:, None]
        self.n, self.p = X.shape
        self.Sxx, self.Sxy, self.Syy, self.slw = wx.T @ X, wx.T @ y, w @ (y * y), np.log(w).sum()
        self.A = np.zeros((J, X.shape[1]))
        np.add.at(self.A, codes, wx)
        self.ay, self.sw = np.bincount(codes, w * y, J), np.bincount(codes, w, J)
        self.b, self.C = self.parts(self.theta)[1:]
        self.H = hessian(lambda th: self.parts(th)[0], self.theta)

    def parts(self, th):
        t2, s2 = th
        c = t2 / (s2 + t2 * self.sw)
        XVX = (self.Sxx - (self.A.T * c) @ self.A) / s2
        XVy = (self.Sxy - (self.A.T * c) @ self.ay) / s2
        yVy = (self.Syy - c @ self.ay ** 2) / s2
        C = np.linalg.inv(XVX)
        b = C @ XVy
        logdetV = self.n * np.log(s2) - self.slw + np.log1p(t2 * self.sw / s2).sum()
        return 0.5 * (logdetV + np.linalg.slogdet(XVX)[1] + yVy - b @ XVy), b, C  # REML -loglik (+ const)

    def df(self, L):
        """Satterthwaite df for L'b, using the observed REML information for the variance parameters."""
        g = lambda th: L @ self.parts(th)[2] @ L
        grad = gradient(g, self.theta)
        return 2 * g(self.theta) ** 2 / (grad @ np.linalg.solve(self.H, grad))

    def ftest(self, Lm):
        """Joint test of the rows of Lm; denominator df by Fai & Cornelius (1996), as SAS/lmerTest."""
        V = Lm @ self.C @ Lm.T
        Lb = Lm @ self.b
        q = len(Lm)
        F = Lb @ np.linalg.solve(V, Lb) / q
        _, P = np.linalg.eigh(V)
        nus = np.array([self.df(P[:, i] @ Lm) for i in range(q)])
        E = (nus / (nus - 2))[nus > 2].sum()
        return F, q, 2 * E / (E - q) if E > q else float("nan")


def steps(th):
    return np.where(th > 0, 1e-4 * th, 1e-8 * max(th.max(), 1e-12))  # ponytail: tau2 on the boundary (0) gives a one-sided-ish step


def gradient(fn, th):
    h = steps(th)
    return np.array([(fn(th + e) - fn(th - e)) / (2 * e[i]) for i, e in enumerate(np.diag(h))])


def hessian(fn, th):
    return np.array([(gradient(fn, th + e) - gradient(fn, th - e)) / (2 * e[i]) for i, e in enumerate(np.diag(steps(th)))])


@stage(MixedModel)
def mixed_model(f: Frames, c: MixedModel):
    """Linear mixed model: y ~ group + (1 | cluster), REML, optional residual weights and log10(x+1) outcome.
    Per contrast: estimate, SE, Satterthwaite df, t, p, d = estimate / sqrt(tau2 + sigma2) (a standardized mean
    difference) and d_paper = 2t/sqrt(df) (what SPSS-style papers report; it grows with cluster size, so it is not
    comparable to Cohen's d); omnibus F with Satterthwaite denominator df."""
    from scipy import stats

    df = at(f, c.level)
    cols = {"y": numeric(df, c.column) * c.scale, "g": need(df, c.group_by), "cl": need(df, c.cluster),
            "w": numeric(df, c.weights) if c.weights else 1.0}
    d = pd.DataFrame(cols).dropna()
    if c.transform == "log10p1":
        d = d[d.y > -1].assign(y=lambda x: np.log10(x.y + 1))
    if (d.w <= 0).any():
        raise ValueError(f"weights in {c.weights!r} must be > 0")
    d["g"] = d.g.astype(str)
    levels = sorted(d.g.unique())
    ref = c.reference or levels[0]
    if ref not in levels:
        raise ValueError(f"reference {ref!r} not in {c.group_by!r}; have {levels}")
    levels = [ref] + [k for k in levels if k != ref]
    pairs = [tuple(s.split(" vs ")) for s in c.contrasts] or [(k, ref) for k in levels[1:]]
    for a, b in pairs:
        if a not in levels or b not in levels:
            raise ValueError(f"contrast {a!r} vs {b!r}: groups must be in {levels}")
    X = np.column_stack([np.ones(len(d))] + [d.g == k for k in levels[1:]]).astype(float)
    codes = pd.factorize(d.cl)[0]
    fit = Fit(d.y.to_numpy(float), X, codes, d.w.to_numpy(float))

    unit = lambda k: np.eye(len(levels))[levels.index(k)] * (k != ref)
    rows = []
    for a, b in pairs:
        L = unit(a) - unit(b)
        est, se, nu = L @ fit.b, np.sqrt(L @ fit.C @ L), fit.df(L)
        t = est / se
        rows.append([f"{a} vs {b}", g6(est), g6(se), g6(nu), g6(t), sig(2 * stats.t.sf(abs(t), nu)), round(float(2 * t / np.sqrt(nu)), 3),
                     round(float(est / np.sqrt(fit.theta.sum())), 3)])
    F, q, df2 = fit.ftest(np.eye(len(levels))[1:])
    s = {"F": g6(F), "df1": q, "df2": g6(df2), "p": sig(stats.f.sf(F, q, df2)), "n": len(d), "clusters": int(codes.max() + 1),
         "tau2": g6(fit.theta[0]), "sigma2": g6(fit.theta[1])}
    y = "log10(y+1)" if c.transform == "log10p1" else "y"
    return f, {"kind": "table", "columns": ["contrast", "estimate", "se", "df", "t", "p", "d_paper", "d"], "rows": rows, "total": len(rows),
               "summary": s, "note": f"{y} ~ {c.group_by} + (1 | {c.cluster}){', weighted by ' + c.weights if c.weights else ''}: "
                                     f"F({q}, {df2:.4g}) = {F:.4g}, p = {s['p']}; n = {len(d)}, {s['clusters']} clusters; "
                                     f"Satterthwaite df; d = estimate / √(τ² + σ²), d_paper = 2t/√df"}


class MixedLogit(BaseModel):
    level: Level = "utterance"
    outcome: str = Field(description="boolean / 0-1 column")
    predictor: str = Field(description="categorical fixed effect")
    coding: Literal["treatment", "sum"] = Field("treatment", description="treatment = each level vs reference; sum = each level vs the grand mean")
    reference: str = Field("", description="treatment coding: reference level; default first in sorted order")
    covariates: list[str] = Field([], description="numeric fixed effects (centred, so effects are per unit at the mean)")
    spline: str = Field("", description="optional numeric column fitted as a natural cubic spline")
    spline_df: int = Field(3, ge=1, le=10)
    groups: list[str] = Field(["speaker"], min_length=1, max_length=2, description="one or two crossed random-intercept columns")
    method: Literal["laplace", "vb"] = Field(
        "laplace", description="laplace = variance components from VB, then fixed effects by Laplace (as glmer); vb = mean-field VB only (too narrow)")


def pirls(model, tau: np.ndarray, fe_sd: float) -> tuple[np.ndarray, np.ndarray]:
    """Fixed effects of a binomial GLMM given the random-effect SDs tau: Newton (penalized IRLS) on the joint mode of
    (b, u), u ~ N(0, tau²), b ~ N(0, fe_sd²); covariance of b from the inverse Hessian. This is lme4's Laplace step
    (b and its Wald SEs conditional on the variance components). Mean-field VB alone drops the posterior correlation
    between a predictor that is constant within a group and that group's intercepts, so its intervals are too narrow."""
    from scipy import sparse
    from scipy.special import expit

    A = sparse.hstack([sparse.csr_matrix(model.exog), sparse.csr_matrix(model.exog_vc)]).tocsr()
    prec = np.r_[np.full(model.k_fep, fe_sd ** -2.0), tau[model.ident] ** -2.0]
    y, b = model.endog, np.zeros(A.shape[1])
    # ponytail: dense (p + q)² Hessian; fine for a few thousand random-effect levels, sparse Cholesky beyond that
    hess = lambda mu: (A.T @ A.multiply((mu * (1 - mu))[:, None])).toarray() + np.diag(prec)
    for _ in range(100):
        mu = expit(A @ b)
        step = np.linalg.solve(hess(mu), A.T @ (y - mu) - prec * b)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    C = np.linalg.inv(hess(expit(A @ b)))
    return b[:model.k_fep], C[:model.k_fep, :model.k_fep]


@stage(MixedLogit)
def mixed_logit(f: Frames, c: MixedLogit):
    """Binomial GLMM: logit P(y) ~ predictor [+ covariates] [+ ns(spline)] + (1|group1) [+ (1|group2)], approximate
    fit: random-effect SDs from statsmodels' BinomialBayesMixedGLM (variational Bayes; priors: fixed effects N(0, 5²),
    log random-effect SDs N(0, 1)), then fixed effects and their covariance by Laplace given those SDs (see pirls).
    Reports posterior mean, SD, 95% interval and odds ratio per fixed effect, random-effect SDs, and per predictor level
    the adjusted probability (covariates at their mean, spline at the mean of its basis, random effects 0)."""
    import warnings

    from scipy.special import expit
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    df = at(f, c.level)
    cols = {"y": numeric(df, c.outcome).astype(float), "g": need(df, c.predictor)}
    cols |= {f"x{i}": numeric(df, k) for i, k in enumerate(c.covariates)}
    cols |= {"s": numeric(df, c.spline)} if c.spline else {}
    cols |= {f"r{i}": need(df, k) for i, k in enumerate(c.groups)}
    d = pd.DataFrame(cols).dropna()
    if not d.y.isin([0, 1]).all():
        raise ValueError(f"{c.outcome} must be boolean or 0/1")
    d["g"] = d.g.astype(str)
    levels = sorted(d.g.unique())
    ref = c.reference or levels[0]
    if ref not in levels:
        raise ValueError(f"reference {ref!r} not in {c.predictor!r}; have {levels}")
    if len(levels) < 2:
        raise ValueError(f"need at least 2 levels in {c.predictor!r}")
    levels = [ref] + [k for k in levels if k != ref]
    d["g"] = pd.Categorical(d.g, categories=levels)  # treatment: first category is the reference; sum: last one is implied
    for i in range(len(c.covariates)):
        d[f"x{i}"] -= d[f"x{i}"].mean()
    rhs = ["C(g, Sum)" if c.coding == "sum" else "C(g)", *(f"x{i}" for i in range(len(c.covariates)))]
    rhs += [f"cr(s, df={c.spline_df}, constraints='center')"] if c.spline else []
    vc = {f"r{i}": f"0 + C(r{i})" for i in range(len(c.groups))}
    model = BinomialBayesMixedGLM.from_formula("y ~ " + " + ".join(rhs), vc, d.assign(**{k: d[k].astype(str) for k in vc}), vcp_p=1, fe_p=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit_vb(rng=0)
    p = model.k_fep
    if c.method == "laplace":
        mean, C = pirls(model, np.exp(res.vcp_mean), fe_sd=5.0)
    else:
        mean, C = np.asarray(res.fe_mean), np.diag(np.asarray(res.fe_sd) ** 2)

    # One contrast vector per reported fixed effect / per-level linear predictor, over the fixed-effect parameters.
    unit = lambda j: np.eye(p)[j]
    names = model.exog_names
    level_L = {}  # predictor level -> its effect (vs reference, or vs grand mean)
    for j, n in enumerate(names):
        if n.startswith("C(g"):
            level_L[n.split(".", 1)[1][:-1]] = unit(j)
    if c.coding == "sum":
        level_L[levels[-1]] = -sum(level_L.values())
    terms = [(f"{c.predictor}={k}", level_L[k]) for k in levels if k in level_L]
    terms += [(c.covariates[int(n[1:])], unit(j)) for j, n in enumerate(names) if n[0] == "x" and n[1:].isdigit()]
    terms += [(f"spline({c.spline})[{n.rsplit('[', 1)[1]}", unit(j)) for j, n in enumerate(names) if n.startswith("cr(")]
    terms.insert(0, ("(intercept)", unit(0)))
    z = 1.959964
    rows = []
    for name, L in terms:
        m, sd = float(L @ mean), float(np.sqrt(L @ C @ L))
        odds = [None] * 3 if name.startswith("spline(") else [g6(np.exp(m)), g6(np.exp(m - z * sd)), g6(np.exp(m + z * sd))]
        rows.append([name, g6(m), g6(sd), g6(m - z * sd), g6(m + z * sd), *odds])
    for i, k in enumerate(c.groups):
        m, sd = float(res.vcp_mean[i]), float(res.vcp_sd[i])  # posterior of log SD
        rows.append([f"sd({k})", g6(np.exp(m)), None, g6(np.exp(m - z * sd)), g6(np.exp(m + z * sd)), None, None, None])

    est, lo, hi = [], [], []
    for k in levels:
        L = unit(0) + level_L.get(k, 0)
        m, sd = float(L @ mean), float(np.sqrt(L @ C @ L))
        est.append(round(float(expit(m)), 4)), lo.append(round(float(expit(m - z * sd)), 4)), hi.append(round(float(expit(m + z * sd)), 4))
    groups = {k: int(d[f"r{i}"].nunique()) for i, k in enumerate(c.groups)}
    s = {"n": len(d), "groups": groups, "method": c.method, "coding": c.coding, "reference": ref if c.coding == "treatment" else None}
    eff = "vs " + ref if c.coding == "treatment" else "vs the grand mean (sum-to-zero)"
    formula = (f"logit P({c.outcome}) ~ {c.predictor}" + "".join(f" + {k}" for k in c.covariates)
               + (f" + ns({c.spline}, df={c.spline_df})" if c.spline else "") + "".join(f" + (1 | {k})" for k in c.groups))
    method = ("random-effect SDs by variational Bayes, fixed effects by Laplace given them" if c.method == "laplace"
              else "mean-field variational Bayes (fixed-effect intervals too narrow)")
    return f, {"kind": "table", "columns": ["term", "mean", "sd", "lo", "hi", "odds_ratio", "or_lo", "or_hi"], "rows": rows,
               "total": len(rows), "summary": s,
               "note": f"{formula}; n = {len(d)}, " + ", ".join(f"{v} {k}" for k, v in groups.items())
                       + f"; {c.predictor} effects {eff}. Approximate Bayesian fit, {method}; 95% intervals are approximate "
                         "(they ignore uncertainty in the random-effect SDs; in a simulation with crossed person and item effects "
                         "they covered 96% with laplace, 78% with vb). "
                         "For confirmatory inference refit with lme4::glmer (adaptive quadrature / profile CIs).",
               "chart": {"kind": "interval", "labels": levels, "x_label": f"adjusted P({c.outcome}) (covariates at mean, random effects 0)",
                         "series": [{"name": c.method, "est": est, "lo": lo, "hi": hi}]}}
