"""Fast tests on synthetic data with known answers (direct scipy / statsmodels / dense-matrix fits)."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy import optimize, stats

import ckviz_stats  # noqa: F401  registers the stages
from ckviz.frames import Frames
from ckviz.stages import apply


def frames(df: pd.DataFrame) -> Frames:
    utts = df.assign(speaker=[f"s{i}" for i in range(len(df))], reply_to=None, timestamp=0, text="")
    utts.index = pd.Index([f"u{i}" for i in range(len(df))], name="id")
    convos = pd.DataFrame(index=pd.Index(sorted(df["conversation_id"].unique()), name="id"))
    return Frames(utts, pd.DataFrame(index=pd.Index(utts["speaker"], name="id")), convos)


def clustered(seed=0, per=(5, 6, 4), sizes=None, tau=0.5, sigma=1.0, effects=(0.0, 1.0, 2.0)):
    """Groups g0..g2 with per[k] clusters each; cluster-level random intercept; member-level noise and word counts."""
    rng = np.random.default_rng(seed)
    rows, j = [], 0
    for k, (n_cl, eff) in enumerate(zip(per, effects)):
        for _ in range(n_cl):
            u, n = rng.normal(0, tau), sizes or int(rng.integers(5, 30))
            for _ in range(n):
                wc = int(rng.integers(100, 600))
                rows.append({"conversation_id": f"c{j}", "meta.g": f"g{k}", "wc": wc,
                             "y": eff + u + rng.normal(0, sigma / np.sqrt(wc / 300))})
            j += 1
    return pd.DataFrame(rows)


def contrast_rows(view):
    cols = view["columns"]
    return {r[0]: dict(zip(cols, r)) for r in view["rows"]}


# ---------- anova ----------

def test_anova_matches_scipy_classic_and_welch():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"conversation_id": "c0", "y": np.r_[rng.normal(0, 1, 20), rng.normal(.5, 2, 15), rng.normal(1, .5, 25)],
                       "meta.g": ["a"] * 20 + ["b"] * 15 + ["c"] * 25})
    groups = [df.y[df["meta.g"] == g] for g in "abc"]
    for welch in (False, True):
        _, v = apply("anova", frames(df), {"column": "y", "group_by": "meta.g", "welch": welch})
        ref = stats.f_oneway(*groups, equal_var=not welch)
        s = v["summary"]
        assert s["F"] == pytest.approx(ref.statistic, rel=1e-6) and s["p"] == pytest.approx(ref.pvalue, rel=5e-3)  # p kept to 3 significant digits
        assert s["df1"] == 2 and (welch or s["df2"] == 57)
    ss_b = sum(len(g) * (g.mean() - df.y.mean()) ** 2 for g in groups)
    assert s["eta2"] == pytest.approx(ss_b / ((df.y - df.y.mean()) ** 2).sum(), abs=1e-4)
    assert [r[0] for r in v["rows"]] == ["a", "b", "c"]


def test_anova_at_conversation_level_and_rejects_one_group():
    f = frames(clustered())
    f = Frames(f.utts, f.speakers, f.convos.assign(**{"meta.x": range(len(f.convos)), "meta.g": "same"}))
    with pytest.raises(ValueError, match="2 groups"):
        apply("anova", f, {"level": "conversation", "column": "meta.x", "group_by": "meta.g"})


# ---------- mixed_model ----------

def test_unweighted_mixed_model_matches_statsmodels_formula_fit():
    df = clustered()
    _, v = apply("mixed_model", frames(df), {"column": "y", "group_by": "meta.g", "reference": "g0"})
    m = smf.mixedlm("y ~ C(g, Treatment('g0'))", df.rename(columns={"meta.g": "g"}), groups="conversation_id").fit(reml=True)
    r = contrast_rows(v)
    for k in ("g1", "g2"):
        name = f"C(g, Treatment('g0'))[T.{k}]"
        assert r[f"{k} vs g0"]["estimate"] == pytest.approx(m.fe_params[name], rel=1e-4)
        # statsmodels' SEs come from the inverse of the full REML Hessian; ours (like SPSS / lme4) are
        # (X'V^-1 X)^-1, checked exactly against a dense fit below; the two differ by ~0.1% here.
        assert r[f"{k} vs g0"]["se"] == pytest.approx(m.bse_fe[name], rel=5e-3)
        # d = estimate / total SD (random intercept + residual); d_paper = 2t/sqrt(df), what SPSS users report
        total_sd = np.sqrt(float(m.cov_re.iloc[0, 0]) + m.scale)
        assert r[f"{k} vs g0"]["d"] == pytest.approx(m.fe_params[name] / total_sd, rel=1e-2)
        assert r[f"{k} vs g0"]["d_paper"] == pytest.approx(2 * r[f"{k} vs g0"]["t"] / np.sqrt(r[f"{k} vs g0"]["df"]), rel=1e-3)


def test_satterthwaite_df_is_exact_in_balanced_between_cluster_design():
    # Equal cluster sizes, no weights, group constant within cluster: the contrast's t is exactly a t on
    # (clusters - groups) df, which Satterthwaite recovers exactly; so does the omnibus F's denominator df.
    df = clustered(seed=3, per=(5, 5, 5), sizes=8)
    _, v = apply("mixed_model", frames(df), {"column": "y", "group_by": "meta.g", "reference": "g0"})
    for row in contrast_rows(v).values():
        assert row["df"] == pytest.approx(12, rel=1e-3)
    assert v["summary"]["df2"] == pytest.approx(12, rel=1e-2) and v["summary"]["df1"] == 2
    # and the omnibus F equals the classic ANOVA on cluster means
    means = df.groupby(["conversation_id", "meta.g"]).y.mean().reset_index()
    ref = stats.f_oneway(*[means.y[means["meta.g"] == g] for g in ("g0", "g1", "g2")])
    assert v["summary"]["F"] == pytest.approx(ref.statistic, rel=1e-3)


def dense_weighted_reml(y, X, cl, w):
    """Independent dense REML for y = Xb + u_cluster + e, Var(e_i) = s2 / w_i (SPSS MIXED /REGWGT)."""
    Z = (cl[:, None] == np.unique(cl)[None, :]).astype(float)

    def parts(th):
        t2, s2 = np.exp(th)
        V = t2 * Z @ Z.T + s2 * np.diag(1 / w)
        Vi = np.linalg.inv(V)
        C = np.linalg.inv(X.T @ Vi @ X)
        b = C @ X.T @ Vi @ y
        r = y - X @ b
        nll = 0.5 * (np.linalg.slogdet(V)[1] - np.linalg.slogdet(C)[1] + r @ Vi @ r)
        return nll, b, C

    th = optimize.minimize(lambda t: parts(t)[0], np.zeros(2), method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-12}).x
    return parts(th)[1:]


def test_weighted_mixed_model_matches_dense_reml_and_ignores_weight_scale():
    df = clustered(seed=5, per=(3, 3, 2))
    X = np.c_[np.ones(len(df)), df["meta.g"] == "g1", df["meta.g"] == "g2"].astype(float)
    b, C = dense_weighted_reml(df.y.to_numpy(), X, df.conversation_id.to_numpy(), df.wc.to_numpy(float))
    cfg = {"column": "y", "group_by": "meta.g", "reference": "g0", "weights": "wc"}
    _, v = apply("mixed_model", frames(df), cfg)
    r = contrast_rows(v)
    for i, k in ((1, "g1"), (2, "g2")):
        assert r[f"{k} vs g0"]["estimate"] == pytest.approx(b[i], rel=1e-3)
        assert r[f"{k} vs g0"]["se"] == pytest.approx(np.sqrt(C[i, i]), rel=1e-3)
    _, v10 = apply("mixed_model", frames(df.assign(wc=df.wc * 10)), cfg)
    np.testing.assert_allclose([x[1:] for x in v10["rows"]], [x[1:] for x in v["rows"]], rtol=1e-3)
    # unweighted differs
    _, vu = apply("mixed_model", frames(df), cfg | {"weights": ""})
    assert contrast_rows(vu)["g1 vs g0"]["estimate"] != pytest.approx(b[1], rel=1e-3)


def test_custom_contrasts_equal_refit_with_other_reference_and_transform():
    df = clustered(seed=7)
    df["y"] = df.y.abs() * 3  # positive, so log10(x/100+1)-style transforms are sensible
    base = {"column": "y", "group_by": "meta.g", "weights": "wc", "transform": "log10p1", "scale": 100}
    _, v = apply("mixed_model", frames(df), base | {"reference": "g0", "contrasts": ["g2 vs g1", "g1 vs g0"]})
    r = contrast_rows(v)
    assert list(r) == ["g2 vs g1", "g1 vs g0"]
    _, v1 = apply("mixed_model", frames(df), base | {"reference": "g1"})
    assert r["g2 vs g1"]["t"] == pytest.approx(contrast_rows(v1)["g2 vs g1"]["t"], rel=1e-3)
    _, vm = apply("mixed_model", frames(df.assign(y=np.log10(df.y * 100 + 1))), base | {"transform": "none", "scale": 1, "reference": "g1"})
    assert contrast_rows(vm)["g2 vs g1"]["t"] == pytest.approx(contrast_rows(v1)["g2 vs g1"]["t"], rel=1e-4)


def test_mixed_model_rejects_unknown_groups():
    with pytest.raises(ValueError, match="nope"):
        apply("mixed_model", frames(clustered()), {"column": "y", "group_by": "meta.g", "reference": "nope"})
    with pytest.raises(ValueError, match="g9"):
        apply("mixed_model", frames(clustered()), {"column": "y", "group_by": "meta.g", "contrasts": ["g9 vs g0"]})
