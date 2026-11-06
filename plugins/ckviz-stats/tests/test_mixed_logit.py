"""mixed_logit on seeded simulations with known level log-odds (fast); a 40-rep coverage check behind CKVIZ_EXAMPLES."""
import os

import numpy as np
import pandas as pd
import pytest

import ckviz_stats  # noqa: F401  registers the stages
from ckviz.frames import Frames
from ckviz.stages import apply

LEVELS = {"a": 0.0, "b": 0.8, "c": -0.6, "d": 0.3}  # log-odds vs level a


def sim_frames(seed, people=40, items=20, sd_person=1.0, sd_item=0.3):
    """One row per (person, item): outcome ~ x(item) + linear trend in item + (1|person) + (1|item). x is item-level."""
    rng = np.random.default_rng(seed)
    x = np.array(list(LEVELS))[np.arange(items) % 4]
    u, v = rng.normal(0, sd_person, people), rng.normal(0, sd_item, items)
    p, i = np.repeat(np.arange(people), items), np.tile(np.arange(items), people)
    eta = 0.5 + np.vectorize(LEVELS.get)(x[i]) - 1.5 * i / items + u[p] + v[i]
    utts = pd.DataFrame({"speaker": [f"p{k}" for k in p], "conversation_id": [f"p{k}" for k in p], "reply_to": None,
                         "timestamp": i, "text": "", "person": [f"p{k}" for k in p], "item": i + 1, "x": x[i],
                         "outcome": rng.random(len(p)) < 1 / (1 + np.exp(-eta))},
                        index=pd.Index([f"u{k}" for k in range(len(p))], name="id"))
    return Frames(utts, pd.DataFrame(index=pd.Index(sorted(set(utts.speaker)), name="id")),
                  pd.DataFrame(index=pd.Index(sorted(set(utts.conversation_id)), name="id")))


CFG = {"outcome": "outcome", "predictor": "x", "reference": "a", "spline": "item", "groups": ["person", "item"]}


def rows(view):
    return {r[0]: dict(zip(view["columns"], r)) for r in view["rows"]}


def test_recovers_level_log_odds_and_reports_adjusted_probabilities():
    _, v = apply("mixed_logit", sim_frames(0), CFG)
    r = rows(v)
    for k in "bcd":
        assert abs(r[f"x={k}"]["mean"] - LEVELS[k]) < 3 * r[f"x={k}"]["sd"]
        assert r[f"x={k}"]["odds_ratio"] == pytest.approx(np.exp(r[f"x={k}"]["mean"]), rel=1e-3)
        assert r[f"x={k}"]["lo"] < r[f"x={k}"]["mean"] < r[f"x={k}"]["hi"]
    assert {"sd(person)", "sd(item)"} <= set(r) and 0.6 < r["sd(person)"]["mean"] < 1.5
    assert sum(k.startswith("spline(item)") for k in r) == 3
    c = v["chart"]
    assert c["kind"] == "interval" and c["labels"] == list("abcd") and c["x_label"].startswith("adjusted P(outcome)")
    s = c["series"][0]
    assert all(lo < e < hi for e, lo, hi in zip(s["est"], s["lo"], s["hi"])) and s["est"][1] > s["est"][0] > s["est"][2]
    assert v["summary"]["n"] == 800 and v["summary"]["groups"] == {"person": 40, "item": 20}
    assert "approximate" in v["note"]


def test_sum_coding_effects_sum_to_zero_and_vb_runs():
    f = sim_frames(1, items=40)  # the 0.3 tolerance needs more than the default 5 items per level
    _, v = apply("mixed_logit", f, {**CFG, "coding": "sum"})
    eff = [rows(v)[f"x={k}"]["mean"] for k in "abcd"]
    assert sum(eff) == pytest.approx(0, abs=1e-3)
    truth = np.array(list(LEVELS.values())) - np.mean(list(LEVELS.values()))
    assert np.abs(np.array(eff) - truth).max() < 0.3
    _, vb = apply("mixed_logit", f, {**CFG, "method": "vb"})
    assert rows(vb)["x=b"]["mean"] == pytest.approx(0.8, abs=0.35)


def test_covariate_and_one_group_and_validation():
    f = sim_frames(2, people=30)
    f.utts["z"] = np.random.default_rng(0).normal(0, 1, len(f.utts))
    _, v = apply("mixed_logit", f, {"outcome": "outcome", "predictor": "x", "covariates": ["z"], "groups": ["person"]})
    assert abs(rows(v)["z"]["mean"]) < 0.3 and "sd(item)" not in rows(v)
    f.utts["n"] = 2
    with pytest.raises(ValueError, match="0/1"):
        apply("mixed_logit", f, {**CFG, "outcome": "n"})
    with pytest.raises(ValueError, match="reference"):
        apply("mixed_logit", f, {**CFG, "reference": "zz"})


def coverage(reps, method):
    hits = 0
    for seed in range(reps):
        r = rows(apply("mixed_logit", sim_frames(100 + seed), {**CFG, "method": method})[1])
        hits += sum(r[f"x={k}"]["lo"] <= LEVELS[k] <= r[f"x={k}"]["hi"] for k in "bcd")
    return hits / (3 * reps)


def test_intervals_cover_truth_in_most_reps():
    assert coverage(5, "laplace") >= 0.7


@pytest.mark.skipif(not os.environ.get("CKVIZ_EXAMPLES"), reason="slow: set CKVIZ_EXAMPLES=1")
def test_coverage_40_reps_laplace_beats_vb():
    lap, vb = coverage(40, "laplace"), coverage(40, "vb")
    print(f"coverage of nominal 95% intervals: laplace {lap:.2f}, vb {vb:.2f}")
    assert lap >= 0.85 and lap > vb + 0.15
