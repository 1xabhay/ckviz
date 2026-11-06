"""Stages for Danescu-Niculescu-Mizil et al. 2013 (politeness): strategy table and text classifier."""
import pandas as pd
import pytest

from ckviz.stages import apply
from helpers import tiny


def requests():
    # 8 requests from two domains: "thanks" marks the polite (top-quartile) ones, "fix" the impolite ones.
    texts = ["thanks could you help"] * 2 + ["fix this now"] * 2
    utts = pd.DataFrame({
        "speaker": "s", "conversation_id": "c", "reply_to": None, "timestamp": 0,
        "text": texts * 2,
        "meta.score": [1.0, 0.5, -1.0, -0.5] * 2,
        "meta.top": [1, 1, -1, -1] * 2,
        "meta.corpus": ["A"] * 4 + ["B"] * 4,
        "meta.strategies": [{"Gratitude": 1, "Direct": 0}] * 2 + [{"Gratitude": 0, "Direct": 1}] * 2
                           + [{"Gratitude": 1, "Direct": 1}, {"Gratitude": 0, "Direct": 0}] * 2,
    }, index=[f"r{i}" for i in range(8)])
    return tiny()._replace(utts=utts)


def test_strategy_table_mean_score_and_share_in_top():
    _, v = apply("strategy_table", requests(), {"features": "meta.strategies", "score": "meta.score",
                                                "label": "meta.top", "value": "1"})
    rows = {r[0]: r for r in v["rows"]}
    assert v["columns"][:4] == ["feature", "n", "mean_score", "pct_top"]
    # Gratitude: r0, r1, r4, r6 -> scores 1, .5, 1, -1 ; top: 3 of 4
    assert rows["Gratitude"][1:4] == [4, pytest.approx(0.375), pytest.approx(75.0)]
    # Direct: r2, r3, r4, r6 -> -1, -.5, 1, -1 ; top: 1 of 4
    assert rows["Direct"][1:4] == [4, pytest.approx(-0.375), pytest.approx(25.0)]


def test_text_classify_cross_domain_unigrams():
    _, v = apply("text_classify", requests(), {"features": ["unigrams"], "label": "meta.top", "value": "1",
                                               "domain": "meta.corpus", "train": "A", "test": "B", "min_count": 1})
    assert v["kind"] == "table"
    assert v["rows"][0][:4] == ["A", "B", 4, 4]
    assert v["rows"][0][4] == pytest.approx(1.0)  # B uses the same words for the same classes


def test_text_classify_in_domain_cv_with_dict_features():
    f = requests()
    f = f._replace(utts=f.utts[f.utts["meta.corpus"] == "A"])
    _, v = apply("text_classify", f, {"features": ["unigrams", "meta.strategies"], "label": "meta.top",
                                      "value": "1", "folds": 2, "min_count": 1, "model": "logreg"})
    assert v["rows"][0][4] == pytest.approx(1.0)


def test_min_count_drops_rare_unigrams():
    with pytest.raises(ValueError, match="no features"):
        apply("text_classify", requests(), {"features": ["unigrams"], "label": "meta.top", "value": "1",
                                            "domain": "meta.corpus", "train": "A", "test": "B", "min_count": 10})
