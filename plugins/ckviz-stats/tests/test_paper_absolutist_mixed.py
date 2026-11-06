"""Al-Mosaiwi & Johnstone (2018) Study 1 with the mixed model (examples/papers/absolutist-mixed.md).

Slow; needs the corpus and the Figshare workbook (data/papers/absolutist/raw/stats.xlsx, read with openpyxl):
CKVIZ_EXAMPLES=1 uv run --no-project --with-editable . --with-editable plugins/ckviz-stats --with pytest --with openpyxl \
    pytest -q plugins/ckviz-stats/tests/test_paper_absolutist_mixed.py
"""
import os
from pathlib import Path

import pandas as pd
import pytest

import ckviz_stats  # noqa: F401  registers the stages
from ckviz import datasets as ds
from ckviz.frames import Frames
from ckviz.pipeline import parse, run
from ckviz.stages import apply

ROOT = Path(__file__).parents[3]
XLSX = ROOT / "data/papers/absolutist/raw/stats.xlsx"
pytestmark = [pytest.mark.skipif(not os.environ.get("CKVIZ_EXAMPLES"), reason="CKVIZ_EXAMPLES not set"),
              pytest.mark.skipif(not (ROOT / "data/papers/absolutist/corpus").exists(), reason="corpus not built")]

CONTRASTS = ["anxiety vs control", "depression vs control", "suicidal_ideation vs control",
             "suicidal_ideation vs anxiety", "suicidal_ideation vs depression"]
PAPER_D = [3.24, 3.14, 4.56, 1.74, 1.71]  # Table 2, d = 2t/sqrt(df): our d_paper column
PAPER_F = 71.549  # F(3, 29)
# The workbook's SPSS output ('H1 and H2 Results', 'T Values H1'): untransformed %, WC-weighted. (t, Satterthwaite df, d)
WORKBOOK = [(8.576, 28.515, 3.212019), (8.382, 29.979, 3.061745), (13.709, 30.229, 4.986822),
            (5.746871, 28.776379, 2.142611), (5.774689, 29.528333, 2.125392)]
WORKBOOK_F = (80.496, 29.29)
COND = {1: "control", 2: "anxiety", 3: "depression", 4: "suicidal_ideation"}


def fit(f: Frames, **cfg):
    _, v = apply("mixed_model", f, {"group_by": "meta.group", "reference": "control", "contrasts": CONTRASTS} | cfg)
    return {r[0]: dict(zip(v["columns"], r)) for r in v["rows"]}, v["summary"]


@pytest.fixture(scope="module")
def workbook() -> Frames:
    """The exact members the authors' model saw: sheet 'Hypothesis 1 data' (3,867 members with WC >= 100)."""
    pytest.importorskip("openpyxl")
    h = pd.read_excel(XLSX, sheet_name="Hypothesis 1 data", usecols=["Forum", "Conditions", "WC (>100)", "Absolutist"])
    utts = pd.DataFrame({"conversation_id": "forum" + h.Forum.astype(str), "meta.group": h.Conditions.map(COND),
                         "wc": h["WC (>100)"], "absolutist": h.Absolutist})
    return Frames(utts, pd.DataFrame(), pd.DataFrame())


def test_authors_data_untransformed_matches_their_spss_output(workbook):
    # Same data, same model: SPSS prints t and df to 3 dp, so anything past rounding would be a method difference.
    rows, s = fit(workbook, column="absolutist", weights="wc")
    for name, (t, df, d) in zip(CONTRASTS, WORKBOOK):
        assert abs(rows[name]["t"]) == pytest.approx(t, abs=1e-3), name
        assert rows[name]["df"] == pytest.approx(df, abs=1e-3), name  # Satterthwaite, as SPSS
        assert rows[name]["d_paper"] == pytest.approx(d, abs=1e-3), name
    assert s["F"] == pytest.approx(WORKBOOK_F[0], abs=1e-3) and s["df2"] == pytest.approx(WORKBOOK_F[1], abs=5e-3)


def test_authors_data_log10_matches_the_paper(workbook):
    # The paper's numbers come from the log10(x+1) model of its .sps; they match to the paper's printed precision.
    rows, s = fit(workbook, column="absolutist", weights="wc", transform="log10p1")
    assert [rows[c]["d_paper"] for c in CONTRASTS] == pytest.approx(PAPER_D, abs=0.005)
    assert s["F"] == pytest.approx(PAPER_F, abs=0.002) and round(s["df2"]) == 29


@pytest.fixture(scope="module")
def ours():
    p = parse((ROOT / "examples/absolutist-study1-mixed.yaml").read_text())
    key, frames = ds.load(p["dataset"])
    steps = run(p["stages"], frames, key)
    assert all(s["status"] == "ok" for s in steps), [s.get("error") for s in steps]
    views = [s["view"] for st, s in zip(p["stages"], steps) if next(iter(st)) in ("mixed_model", "anova")]
    return [({r[0]: dict(zip(v["columns"], r)) for r in v["rows"]}, v["summary"]) for v in views]


def test_text_pipeline_matches_paper_and_workbook(ours):
    # Our scores come from the raw text (our tokenizer + rebuilt dictionary: 97.6% of members match LIWC to 0.02 points)
    # and miss 2 members, so d moves a little; tolerance 5%.
    (log_rows, log_s), (raw_rows, raw_s), (_, anova) = ours
    assert [log_rows[c]["d_paper"] for c in CONTRASTS] == pytest.approx(PAPER_D, rel=0.05)
    assert log_s["F"] == pytest.approx(PAPER_F, rel=0.05) and log_s["df1"] == 3
    assert [raw_rows[c]["d_paper"] for c in CONTRASTS] == pytest.approx([w[2] for w in WORKBOOK], rel=0.05)
    assert raw_s["F"] == pytest.approx(WORKBOOK_F[0], rel=0.05)
    assert all(r["p"] < 1e-3 for r in log_rows.values())  # paper: all p < .001
    # the standardized d (estimate / total SD) is a member-level effect size, far below 2t/sqrt(df) with 35 forums
    assert [raw_rows[c]["d"] for c in CONTRASTS] == pytest.approx([0.543, 0.536, 1.054, 0.510, 0.517], abs=0.03)
    assert all(r["d"] < r["d_paper"] / 2 for r in [*log_rows.values(), *raw_rows.values()])
    assert anova["df1"] == 3 and anova["df2"] == 31 and anova["p"] < 1e-6  # 35 forums, 4 groups
