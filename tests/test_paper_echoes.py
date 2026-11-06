"""Reproduce Danescu-Niculescu-Mizil et al. 2012, "Echoes of power" (Fig. 1 and Fig. 4).

Slow (~4 min; wiki-corpus is 391k utterances) and needs the downloads + the supreme subset described in
examples/papers/echoes-of-power.md:  CKVIZ_EXAMPLES=1 uv run pytest tests/test_paper_echoes.py
"""
import os
import re
from pathlib import Path

import pytest

from ckviz import datasets as ds
from ckviz.pipeline import parse, run

EXAMPLES = Path(__file__).parent.parent / "examples"
pytestmark = pytest.mark.skipif(not os.environ.get("CKVIZ_EXAMPLES"), reason="CKVIZ_EXAMPLES not set")


def views(name: str) -> list[dict]:
    p = parse((EXAMPLES / f"{name}.yaml").read_text())
    key, frames = ds.load(p["dataset"])
    steps = run(p["stages"], frames, key)
    assert all(s["status"] == "ok" for s in steps), [s.get("error") for s in steps]
    return [s["view"] for s in steps]


def table(view) -> dict:
    """measure -> {group: score, 'p': p}."""
    cols = view["columns"]
    return {r[0]: dict(zip(cols[1:], r[1:])) for r in view["rows"]}


def counts(view) -> dict:
    """group -> (n with all 8 markers, n with any marker), parsed from the note."""
    return {g: (int(a), int(b)) for g, a, b in re.findall(r"(\w+): (\d+) / (\d+) speakers", view["note"])}


AGGS = ["aggregate 1", "aggregate 2", "aggregate 3"]


@pytest.fixture(scope="module")
def wiki():
    return views("echoes-of-power-wiki")


def test_fig1a_everyone_coordinates_more_toward_admins(wiki):
    v = wiki[0]
    # Same populations as ConvoKit's coordination notebook (legend "Target-admins (total: 966, 1370)").
    assert counts(v) == {"False": (2208, 3548), "True": (966, 1370)}
    t = table(v)
    for a in AGGS:  # P_target: C(U, admins) > C(U, non-admins)
        assert t[a]["True"] > t[a]["False"]
    # Bar heights read off the notebook's saved chart (admins / non-admins, aggregate 1..3).
    assert [t[a]["True"] for a in AGGS] == pytest.approx([0.0190, 0.0187, 0.0174], abs=0.0005)
    assert [t[a]["False"] for a in AGGS] == pytest.approx([0.0172, 0.0164, 0.0144], abs=0.0005)
    # Paper: aggregate 2 ** and 3 *** significant; here p < 0.05 for both (aggregate 1 is not: p ~ 0.16).
    assert t["aggregate 2"]["p"] < 0.05 and t["aggregate 3"]["p"] < 0.05


def test_fig1b_admins_coordinate_more_as_speakers(wiki):
    v = wiki[1]
    assert counts(v) == {"False": (1804, 2775), "True": (871, 1203)}
    t = table(v)
    for a in AGGS:  # contradicts P_speaker, as in the paper
        assert t[a]["True"] > t[a]["False"]
        assert t[a]["p"] < 0.05
    assert [t[a]["True"] for a in AGGS] == pytest.approx([0.0192, 0.0188, 0.0175], abs=0.0005)
    assert [t[a]["False"] for a in AGGS] == pytest.approx([0.0168, 0.0162, 0.0146], abs=0.0005)


@pytest.mark.skipif(not (EXAMPLES.parent / "data/supreme-echoes-2012").exists(), reason="supreme subset not built")
def test_fig4_lawyers_coordinate_more_to_justices():
    v = views("echoes-of-power-supreme")[0]
    t = table(v)
    for a in AGGS:
        assert t[a]["A"] > t[a]["J"]
        assert t[a]["p"] < 0.05  # paper: *** (p < 0.001); here only p < 0.05
    # Paper markers with the largest lawyer > Justice gaps (***) also come out significant here.
    for m in ["auxverb", "ppron", "ipron"]:
        assert t[m]["A"] > t[m]["J"] and t[m]["p"] < 0.01
    n = counts(v)  # paper legend: lawyers (500, 535), Justices (815, 1307)
    assert n["A"][1] == pytest.approx(535, rel=0.05)
    assert n["J"][1] == pytest.approx(1307, rel=0.15)
