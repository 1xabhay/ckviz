"""Reproduce ConvoKit's example notebooks; expected values are copied from their saved outputs.

Slow and needs the ConvoKit downloads: CKVIZ_EXAMPLES=1 uv run pytest tests/test_examples.py
"""
import os
from pathlib import Path

import pytest

from ckviz import datasets as ds
from ckviz.pipeline import parse, run

EXAMPLES = Path(__file__).parent.parent / "examples"
pytestmark = pytest.mark.skipif(not os.environ.get("CKVIZ_EXAMPLES"), reason="CKVIZ_EXAMPLES not set")


def views(name: str) -> dict:
    p = parse((EXAMPLES / f"{name}.yaml").read_text())
    key, frames = ds.load(p["dataset"])
    steps = run(p["stages"], frames, key)
    assert all(s["status"] == "ok" for s in steps), [s.get("error") for s in steps]
    return {next(iter(st)): s["view"] for st, s in zip(p["stages"], steps)}


def top(view, side, n):
    return {r["label"]: r["value"] for r in view[side][:n]}


def test_intro_fighting_words():
    v = views("intro-fighting-words")["fighting_words"]
    assert top(v, "pos", 2) == pytest.approx({"god": 9.664, "sin": 9.204}, abs=0.01)
    assert top(v, "neg", 2) == pytest.approx({"money": -7.957, "religion": -7.769}, abs=0.01)


def test_sigdial_mixed_gender():
    # Expected = the notebook's code run verbatim on today's movie-corpus. The notebook's saved output
    # (the -29.13, sir -25.21; 100,931 vs 203,782 utterances) predates a dataset change: most speaker genders are
    # now lowercase (m/f), so its case-sensitive 'M'/'F' check flags only 4,574 utterances as mixed.
    v = views("sigdial-mixed-gender")["fighting_words"]
    assert top(v, "neg", 3) == pytest.approx({"well": -6.037, "we": -5.58, "gonna": -5.556}, abs=0.01)


def test_sigdial_mixed_gender_casefold():
    # Matching gender case-insensitively recovers the notebook's saved output.
    v = views("sigdial-mixed-gender-casefold")["fighting_words"]
    assert top(v, "neg", 6) == pytest.approx(
        {"the": -29.133, "sir": -25.212, "man": -19.330, "we": -17.650, "mr": -15.859, "shit": -14.219}, abs=0.01)


def test_politeness_classifier():
    v = views("politeness-classifier")["classify_cv"]
    notebook = [0.7912844, 0.72706422, 0.77522936, 0.77011494, 0.75402299]  # unseeded folds in the notebook
    assert sum(v["values"]) / 5 == pytest.approx(sum(notebook) / 5, abs=0.02)


def test_politeness_admin():
    v = views("politeness-admin")["compare_groups"]
    diff = {r[0]: round(r[3], 2) for r in v["rows"]}
    # Expected = the notebook's code run verbatim on convokit 5.0.0 / spaCy 3.8 / en_core_web_sm 3.8.
    # The notebook's *saved* output came from an older stack and differs on 8/21 strategies
    # (e.g. 1st_person 0.18 -> 0.04, HASHEDGE 0.10 -> 0.00, 1st_person_pl. -0.07 -> -0.04): drift in ConvoKit, not ckviz.
    convokit_5 = {"Please": -0.01, "Please_start": -0.01, "HASHEDGE": 0.0, "Indirect_(btw)": -0.0, "Hedges": 0.02,
                  "Factuality": 0.01, "Deference": -0.0, "Gratitude": 0.03, "Apologizing": 0.01, "1st_person_pl.": -0.04,
                  "1st_person": 0.04, "1st_person_start": 0.01, "2nd_person": -0.01, "2nd_person_start": -0.01,
                  "Indirect_(greeting)": 0.02, "Direct_question": 0.02, "Direct_start": 0.02, "HASPOSITIVE": 0.0,
                  "HASNEGATIVE": -0.01, "SUBJUNCTIVE": 0.02, "INDICATIVE": 0.0}
    assert diff == pytest.approx(convokit_5, abs=0.006)


def test_bow_score():
    v = views("bow-score")["bow_classify"]
    assert top(v, "pos", 5) == pytest.approx(
        {"hotels": 1.270, "hbhs": 1.116, "engine": 1.110, "involves": 1.082, "lincoln": 1.071}, abs=0.01)
    assert top(v, "neg", 2) == pytest.approx({"creation": -0.785, "binary": -0.764}, abs=0.01)
