"""Shared research stages: levels, lexicons, roll-ups, statistics."""
import pandas as pd
import pytest

from ckviz.stages import STAGES, apply, holm
from helpers import tiny


def test_holm_adjusts_and_preserves_order():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_lexicon_rate_with_wildcards():
    t = tiny()
    t.utts.loc["a1", "text"] = "I always feel I never win, totally"
    f, view = apply("lexicon", t, {"words": ["always", "never", "total*"], "into": "absolutist"})
    assert f.utts.loc["a1", "absolutist"] == pytest.approx(3 / 7)
    assert f.utts.loc["a1", "n_tokens"] == 7
    assert f.utts.loc["a0", "absolutist"] == 0
    assert view["kind"] == "table"


def test_lexicon_matches_contractions_and_case():
    t = tiny()
    t.utts.loc["a1", "text"] = "I’m sad. I said I'D go"  # curly apostrophe too
    f, _ = apply("lexicon", t, {"words": ["i", "i'm", "i'd"], "into": "i_words"})
    assert f.utts.loc["a1", "i_words"] == pytest.approx(3 / 6)


def test_rollup_to_speaker_and_conversation():
    t = tiny()
    t.utts["x"] = [1.0, 2.0, 3.0, 4.0, 6.0]
    f, _ = apply("rollup", t, {"column": "x", "to": "speaker", "agg": "mean", "into": "mean_x"})
    assert f.speakers["meta.mean_x"].to_dict() == {"bot": pytest.approx(8 / 3), "u1": 2.0, "u2": 6.0}
    f, view = apply("rollup", f, {"column": "x", "to": "conversation", "agg": "sum", "into": "total_x"})
    assert f.convos["meta.total_x"].to_dict() == {"a0": 6.0, "b0": 10.0}
    assert view["kind"] == "table"


def test_rollup_count_needs_no_numeric_column():
    f, _ = apply("rollup", tiny(), {"column": "text", "to": "speaker", "agg": "count", "into": "n"})
    assert f.speakers["meta.n"].to_dict() == {"bot": 3, "u1": 1, "u2": 1}


def test_first_n_keeps_earliest_utterances_per_conversation():
    f, _ = apply("first_n", tiny(), {"n": 2})
    assert sorted(f.utts.index) == ["a0", "a1", "b0", "b1"]


def two_groups():
    t = tiny()
    utts = pd.DataFrame({
        "speaker": ["s"] * 40, "conversation_id": ["c"] * 40, "reply_to": [None] * 40, "timestamp": range(40),
        "text": ["x"] * 40, "g": ["A"] * 20 + ["B"] * 20,
        "y": [float(i % 5) for i in range(20)] + [float(i % 5) + 2 for i in range(20)],
        "noise": [float(i % 7) for i in range(40)],
    }, index=[f"u{i}" for i in range(40)])
    return t._replace(utts=utts)


def test_compare_groups_reports_effect_and_corrected_p():
    _, view = apply("compare_groups", two_groups(), {"column": "y", "group_by": "g", "class1": "B", "class2": "A"})
    assert view["columns"] == ["feature", "B", "A", "diff", "d", "p", "p_holm"]
    (row,) = view["rows"]
    assert row[1:4] == [4.0, 2.0, 2.0]
    assert row[4] > 1.0 and row[5] < 0.001


def test_compare_groups_on_dict_column_corrects_across_features():
    t = two_groups()
    t.utts["feats"] = [{"y": y, "noise": n} for y, n in zip(t.utts["y"], t.utts["noise"])]
    _, view = apply("compare_groups", t, {"column": "feats", "group_by": "g", "class1": "B", "class2": "A"})
    rows = {r[0]: r for r in view["rows"]}
    assert rows["y"][6] < 0.001 and rows["noise"][6] > 0.05
    assert view["rows"][0][0] == "y"  # sorted by p


def test_compare_groups_at_speaker_level():
    t = tiny()
    t.speakers["meta.score"] = [1.0, 5.0, 7.0]
    _, view = apply("compare_groups", t, {"level": "speaker", "column": "meta.score", "group_by": "meta.role",
                                          "class1": "person", "class2": "ai"})
    assert view["rows"][0][1:4] == [6.0, 1.0, 5.0]


def test_correlate_spearman():
    t = two_groups()
    t.utts["z"] = t.utts["y"] * 2 + 1
    _, view = apply("correlate", t, {"x": "z", "y": "y"})
    assert view["columns"] == ["feature", "r", "p", "n", "p_holm"]
    assert view["rows"][0][1] == pytest.approx(1.0)


def test_stats_stages_offer_levels():
    for name in ("compare_groups", "correlate"):
        assert STAGES[name].config.model_json_schema()["properties"]["level"]["enum"] == ["utterance", "speaker", "conversation"]
