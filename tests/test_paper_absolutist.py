"""Al-Mosaiwi & Johnstone (2018) "In an Absolute State", Clinical Psychological Science 6(4):529-542.

Slow, and needs the Figshare corpus built first (see examples/papers/absolutist.md):
CKVIZ_EXAMPLES=1 uv run pytest tests/test_paper_absolutist.py
Paper numbers are % of words; ckviz rates are shares, hence the *100.
"""
import os
from pathlib import Path

import pytest

from ckviz import datasets as ds
from ckviz.pipeline import parse, run
from ckviz.stages import apply

ROOT = Path(__file__).parent.parent
pytestmark = [pytest.mark.skipif(not os.environ.get("CKVIZ_EXAMPLES"), reason="CKVIZ_EXAMPLES not set"),
              pytest.mark.skipif(not (ROOT / "data/papers/absolutist/corpus").exists(), reason="corpus not built")]

# group: (N members, mean %) from the paper's text (Study 1-3) and the Figshare descriptives
PAPER_MEANS = {"control": (2373, 0.97), "anxiety": (597, 1.45), "depression": (529, 1.45), "suicidal_ideation": (368, 1.80),
               "ptsd": (534, 1.13), "schizophrenia": (591, 1.14), "eating_disorder": (547, 1.25), "bpd": (326, 1.47),
               "recovery": (558, 1.31)}
# Study 2 and 3 d values: member-level Cohen's d (they equal (M1-M2)/pooled SD from the paper's own descriptives)
PAPER_MEMBER_D = {("bpd", "ptsd"): 0.36, ("bpd", "schizophrenia"): 0.35, ("eating_disorder", "ptsd"): 0.13,
                  ("eating_disorder", "schizophrenia"): 0.12, ("recovery", "control"): 0.37, ("anxiety", "recovery"): 0.15,
                  ("depression", "recovery"): 0.15, ("suicidal_ideation", "recovery"): 0.50}
STUDY1 = [("anxiety", "control"), ("depression", "control"), ("suicidal_ideation", "control"),
          ("suicidal_ideation", "anxiety"), ("suicidal_ideation", "depression")]


def tables(name: str):
    p = parse((ROOT / f"examples/{name}.yaml").read_text())
    key, frames = ds.load(p["dataset"])
    steps = run(p["stages"], frames, key)
    assert all(s["status"] == "ok" for s in steps), [s.get("error") for s in steps]
    out = {}
    for st, s in zip(p["stages"], steps):
        (stage, cfg), = st.items()
        if stage == "compare_groups":
            (row,) = s["view"]["rows"]
            n1, n2 = (int(part.split("n=")[1]) for part in s["view"]["note"].split(": ", 1)[1].split(", "))
            out[cfg.get("level", "utterance"), cfg["class1"], cfg["class2"]] = dict(zip(s["view"]["columns"], row), n1=n1, n2=n2)
    return out


@pytest.fixture(scope="module")
def all_tables():
    return tables("absolutist-study1") | tables("absolutist-study2-3")


def test_group_means_and_ns(all_tables):
    seen = {}
    for (level, a, b), r in all_tables.items():
        if level == "utterance":
            seen[a], seen[b] = (r["n1"], r[a] * 100), (r["n2"], r[b] * 100)
    assert seen.keys() == PAPER_MEANS.keys()
    for g, (n, m) in PAPER_MEANS.items():
        # 2 files are missing from Figshare; our >=100-token cut differs from LIWC's WC for 1 control member
        assert abs(seen[g][0] - n) <= 2, g
        # both sides are rounded to 2 dp, plus a few % of members counted 1 word differently from LIWC
        assert seen[g][1] == pytest.approx(m, abs=0.015), g


def test_study2_3_member_level_effect_sizes(all_tables):
    for (a, b), d in PAPER_MEMBER_D.items():
        # the write-up's table is within 0.02 at 2 dp; the extra 0.005 is that rounding
        assert all_tables["utterance", a, b]["d"] == pytest.approx(d, abs=0.025), (a, b)


def test_study1_forum_level_direction_and_significance(all_tables):
    for a, b in STUDY1:
        r = all_tables["conversation", a, b]
        assert r["d"] > 1 and r["p"] < 0.05, (a, b, r)
    assert all_tables["conversation", "anxiety", "depression"]["p"] > 0.05  # paper: n.s. (p = 1.00)


def test_study1_forum_d_matches_same_estimator_on_authors_liwc_scores(all_tables):
    # The paper's Study 1 d (3.24, 3.14, 4.56, 1.74, 1.71) is 2t/sqrt(df) from an SPSS mixed model; that model is checked
    # in plugins/ckviz-stats/tests/test_paper_absolutist_mixed.py. Here, check that our text pipeline gives the same
    # forum-level d as the authors' own per-member LIWC indices (meta.liwc_absolutist, Figshare 'Original' sheet) put
    # through the same stages.
    _, f = ds.load("absolutist-forums")
    f, _ = apply("filter", f, {"column": "meta.liwc_wc", "op": ">=", "value": "100"})
    f, _ = apply("rollup", f, {"column": "meta.liwc_absolutist", "to": "conversation", "agg": "mean", "into": "liwc"})
    for a, b in STUDY1:
        _, v = apply("compare_groups", f, {"level": "conversation", "column": "meta.liwc", "group_by": "meta.group", "class1": a, "class2": b})
        assert all_tables["conversation", a, b]["d"] == pytest.approx(v["rows"][0][4], rel=0.1), (a, b)


def test_member_scores_match_authors_liwc_output():
    p = parse((ROOT / "examples/absolutist-study1.yaml").read_text())
    _, f = ds.load("absolutist-forums")
    f, _ = apply("lexicon_net", f, p["stages"][0]["lexicon_net"])
    diff = (f.utts["absolutist"] * 100 - f.utts["meta.liwc_absolutist"].astype(float)).abs()
    assert (diff < 0.02).mean() > 0.93
    assert ((f.utts["n_tokens"] / f.utts["meta.liwc_wc"].astype(float) - 1).abs() < 0.01).mean() > 0.95  # token counts within 1% of LIWC WC
