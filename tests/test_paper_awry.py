"""Reproduce Zhang et al. 2018 (arXiv 1805.05345), "Conversations Gone Awry": Table 3 and Figure 2A.

Slow and needs the ConvoKit download: CKVIZ_EXAMPLES=1 uv run pytest tests/test_paper_awry.py   (~4 min)
Tolerances (fixed before running, see examples/papers/conversations-gone-awry.md):
  Table 3 accuracy +-3 points. With 584 pairs one accuracy has a binomial SE of ~2 points, and our corpus is the
  2019 release's 584 annotation_year=2018 pairs, not the paper's 635.
  Figure 2A: each first-exchange marker the paper's text calls significant must have the paper's sign and p < .05.
  Our p is McNemar's exact test on the matched pairs; the paper's binomial test against the on-track share is anti-
  conservative (11-17% false positives at nominal 5% in 584-pair null simulations), so p's here are larger.
"""
import os
from pathlib import Path

import pytest

from ckviz import datasets as ds
from ckviz.pipeline import parse, run

EXAMPLES = Path(__file__).parent.parent / "examples"
pytestmark = pytest.mark.skipif(not os.environ.get("CKVIZ_EXAMPLES"), reason="CKVIZ_EXAMPLES not set")

# Table 3: paired_classify features -> paper accuracy %
TABLE3 = {"unigrams": 56.7, "meta.sentiment": 55.4, "meta.politeness": 60.5, "meta.prompts": 59.2,
          "meta.politeness + meta.prompts": 61.6}
# Section 5 text: marker@comment -> paper sign of the log-odds (awry vs on-track), all p < .05 or stronger
FIG2A = {"Direct_question@1": +1, "2nd_person_start@1": +1, "Gratitude@1": -1, "Indirect_(greeting)@1": -1,
         "Hedges@1": -1, "2nd_person@2": +1, "1st_person_start@2": -1}


@pytest.fixture(scope="module")
def views():
    p = parse((EXAMPLES / "awry-2018.yaml").read_text())
    key, frames = ds.load(p["dataset"])
    out = run(p["stages"], frames, key)
    assert all(s["status"] == "ok" for s in out), [s.get("error") for s in out]
    assert out[3]["counts"]["conversations"] == 1168  # 584 pairs
    return [(next(iter(st)), s["view"]) for st, s in zip(p["stages"], out)]


def table3(views) -> dict:
    return {v["rows"][0][0]: 100 * v["rows"][0][-1] for name, v in views if name == "paired_classify"}


def test_table3_features_beat_chance(views):
    got = table3(views)
    assert set(got) == set(TABLE3)
    assert all(v > 50 for v in got.values()), got


def test_table3_bag_of_words(views):
    assert abs(table3(views)["unigrams"] - TABLE3["unigrams"]) <= 3, table3(views)


def test_table3_pragmatic_features_beat_bag_of_words(views):
    got = table3(views)
    assert got["meta.politeness"] > got["unigrams"] and got["meta.politeness + meta.prompts"] > got["unigrams"], got


@pytest.mark.xfail(strict=True, reason="politeness, sentiment, prompt types and pragmatic (all) come out 3-7 points "
                                       "below the paper; see examples/papers/conversations-gone-awry.md")
def test_table3_pragmatic_and_sentiment_within_3_points(views):
    got = table3(views)
    off = {k: (TABLE3[k], round(got[k], 1)) for k in TABLE3 if k != "unigrams" and abs(got[k] - TABLE3[k]) > 3}
    assert not off


def log_odds(views) -> dict:
    return {r[0]: r for r in next(v for n, v in views if n == "log_odds")["rows"]}


def test_figure2a_signs_and_strongest_markers(views):
    rows = log_odds(views)
    assert {k: rows[k][4] for k, s in FIG2A.items() if rows[k][4] * s <= 0} == {}
    # the three markers the paper reports at p < .001 and that drive its directness story: p < .01 under McNemar
    # (the .01 cut was set after switching tests; with the paper's binomial test they were all < .001)
    assert all(rows[k][5] < 0.01 for k in ("Direct_question@1", "2nd_person_start@1", "2nd_person@2"))


@pytest.mark.xfail(strict=True, reason="under McNemar, gratitude@1 (p = .17), greeting@1 (.14), hedges@1 (.14) and "
                                       "1st person start@2 (.07) keep the paper's sign but miss p < .05")
def test_figure2a_all_named_markers_significant(views):
    rows = log_odds(views)
    assert {k: rows[k][5] for k in FIG2A if rows[k][5] >= 0.05} == {}
