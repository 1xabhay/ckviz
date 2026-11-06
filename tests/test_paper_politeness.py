"""Reproduce Danescu-Niculescu-Mizil et al. 2013 (arXiv 1306.6078), Tables 3 and 4.

Slow and needs the ConvoKit downloads: CKVIZ_EXAMPLES=1 uv run pytest tests/test_paper_politeness.py
Tolerances (fixed before running, see examples/papers/politeness-2013.md):
  Table 4 accuracy +-3 points.
  Table 3 mean score +-0.15 (about 0.2 SD of scores, SD = 0.7) and top-quartile share +-10 points; these allow for
  spaCy-vs-Stanford parses and the ConvoKit 5 lexicon, which change *which* requests carry a strategy, while
  still failing on a 25%-vs-75% type swap and on a wrong sign further than 0.15 from the paper's mean.
"""
import os
from pathlib import Path

import pytest

from ckviz import datasets as ds
from ckviz.pipeline import parse, run

EXAMPLES = Path(__file__).parent.parent / "examples"
pytestmark = pytest.mark.skipif(not os.environ.get("CKVIZ_EXAMPLES"), reason="CKVIZ_EXAMPLES not set")

# paper name -> (ConvoKit 5 strategy, mean politeness, % in top quartile)
TABLE3 = {
    "Gratitude": ("Gratitude", 0.87, 78), "Deference": ("Deference", 0.78, 70),
    "Greeting": ("Indirect_(greeting)", 0.43, 45), "Positive lexicon": ("HASPOSITIVE", 0.12, 32),
    "Negative lexicon": ("HASNEGATIVE", -0.13, 22), "Apologizing": ("Apologizing", 0.36, 53),
    "Please": ("Please", 0.49, 57), "Please start": ("Please_start", -0.30, 22),
    "Indirect (btw)": ("Indirect_(btw)", 0.63, 58), "Direct question": ("Direct_question", -0.27, 15),
    "Direct start": ("Direct_start", -0.43, 9), "Counterfactual modal": ("SUBJUNCTIVE", 0.47, 52),
    "Indicative modal": ("INDICATIVE", 0.09, 27), "1st person start": ("1st_person_start", 0.12, 29),
    "1st person pl.": ("1st_person_pl.", 0.08, 27), "1st person": ("1st_person", 0.08, 28),
    "2nd person": ("2nd_person", 0.05, 30), "2nd person start": ("2nd_person_start", -0.30, 17),
    "Hedges": ("Hedges", 0.14, 28), "Factuality": ("Factuality", -0.38, 13),
}
# (features, train, test) -> paper accuracy %
W, S = "wikipedia-politeness-corpus", "stack-exchange-politeness-corpus"
TABLE4 = {("BOW", W, W): 79.84, ("BOW", S, S): 74.47, ("BOW", W, S): 64.23, ("BOW", S, W): 72.17,
          ("Ling.", W, W): 83.79, ("Ling.", S, S): 78.19, ("Ling.", W, S): 67.53, ("Ling.", S, W): 75.43}


def steps(name: str) -> list[tuple[str, dict, dict]]:
    p = parse((EXAMPLES / f"{name}.yaml").read_text())
    key, frames = ds.load(p["dataset"])
    out = run(p["stages"], frames, key)
    assert all(s["status"] == "ok" for s in out), [s.get("error") for s in out]
    return [(*next(iter(st.items())), s["view"]) for st, s in zip(p["stages"], out)]


def test_table3_strategies():
    view = steps("politeness-2013-strategies")[-1][2]
    ours = {r[0]: r for r in view["rows"]}
    misses = []
    for name, (key, mean, pct) in TABLE3.items():
        _, n, m, p, *_ = ours[key]
        if abs(m - mean) > 0.15 or abs(p - pct) > 10:
            misses.append(f"{name}: paper {mean:+.2f}/{pct}% ours {m:+.2f}/{p:.0f}% (n={n})")
    assert not misses, misses


def table4() -> dict:
    got = {}
    for name, cfg, view in steps("politeness-2013-classifiers"):
        if name == "text_classify":
            kind = "Ling." if "meta.politeness_strategies" in cfg["features"] else "BOW"
            got[(kind, cfg["train"], cfg["test"])] = 100 * view["rows"][0][4]
    assert len(got) == 8
    return got


def off(got: dict, test: str) -> dict:
    return {k: (TABLE4[k], round(v, 2)) for k, v in got.items() if k[2] == test and abs(v - TABLE4[k]) > 3}


def test_table4_tested_on_wikipedia():
    # Wiki in-domain and SE -> Wiki, BOW and Ling.: all within 3 points.
    assert not off(table4(), W)


@pytest.mark.xfail(strict=True, reason="SE-tested cells come out 3-15 points below the paper (SE in-domain ~63-64 vs "
                                       "74-78); see examples/papers/politeness-2013.md")
def test_table4_tested_on_stack_exchange():
    assert not off(table4(), S)
