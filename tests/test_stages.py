import pandas as pd
import pytest

from ckviz.frames import load_dir
from ckviz.stages import STAGES, apply
from helpers import SYNTHETIC, tiny


def test_every_stage_has_a_form_schema():
    for name, s in STAGES.items():
        assert s.config.model_json_schema()["type"] == "object", name


def test_join_meta_copies_speaker_meta_onto_utterances():
    f, _ = apply("join_meta", tiny(), {"source": "speaker", "keys": ["phq9"]})
    assert f.utts.loc["a1", "speaker.phq9"] == 20


def test_filter_keeps_matching_utterances_and_prunes_orphans():
    f, view = apply("filter", tiny(), {"column": "meta.role", "op": "==", "value": "person"})
    assert list(f.utts.index) == ["a1", "b1"]
    assert sorted(f.speakers.index) == ["u1", "u2"]
    assert view["kind"] == "table"


def test_filter_coerces_numeric_values():
    f, _ = apply("join_meta", tiny(), {"source": "speaker", "keys": ["phq9"]})
    f, _ = apply("filter", f, {"column": "speaker.phq9", "op": ">=", "value": "10"})
    assert list(f.utts.index) == ["a1"]


def test_filter_rejects_unknown_column():
    with pytest.raises(ValueError, match="nope"):
        apply("filter", tiny(), {"column": "nope", "op": "==", "value": "x"})


def test_split_median_labels_rows():
    f, _ = apply("join_meta", tiny(), {"source": "speaker", "keys": ["phq9"]})
    f, _ = apply("filter", f, {"column": "meta.role", "op": "==", "value": "person"})
    f, _ = apply("split", f, {"column": "speaker.phq9", "into": "group"})
    assert f.utts["group"].to_dict() == {"a1": "high", "b1": "low"}


def test_split_threshold_at_speaker_level():
    f, view = apply("split", tiny(), {"level": "speaker", "column": "meta.phq9", "method": "threshold", "threshold": 10,
                                       "into": "meta.severe", "labels": ["yes", "no"]})
    s = f.speakers["meta.severe"]
    assert (s["u1"], s["u2"]) == ("yes", "no") and pd.isna(s["bot"])
    assert "group" not in f.utts and view["values"] == [1, 1] and "≥ 10" in view["note"]
    f, _ = apply("split", tiny(), {"level": "speaker", "column": "meta.phq9", "method": "threshold", "threshold": 20, "into": "g"})
    assert f.speakers.loc["u1", "g"] == "high"  # >= threshold is labels[0]


def test_split_quantile_top_quartile():
    t = tiny()
    t.utts["x"] = [1, 2, 3, 4, 5]
    f, view = apply("split", t, {"column": "x", "method": "quantile", "quantile": 0.75, "labels": ["top", "rest"]})
    assert f.utts["group"].tolist() == ["rest", "rest", "rest", "top", "top"] and view["values"] == [2, 3]


def test_aggregate_sorts_numeric_labels_numerically():
    t = tiny()
    t.utts["day"] = [1, 10, 2, 1, 2]
    _, view = apply("aggregate", t, {"by": "day"})
    assert view["labels"] == ["1", "2", "10"] and view["values"] == [2, 2, 1]


def test_aggregate_proportion_wilson():
    from statsmodels.stats.proportion import proportion_confint

    t = tiny()
    t.utts["day"] = [1, 10, 2, 1, 2]
    t.utts["replied"] = [True, False, True, False, True]
    _, view = apply("aggregate", t, {"by": "day", "measure": "proportion", "value": "replied"})
    rows = {r[0]: dict(zip(view["columns"], r)) for r in view["rows"]}
    assert list(rows) == ["1", "2", "10"] and (rows["1"]["n"], rows["1"]["k"], rows["1"]["proportion"]) == (2, 1, 0.5)
    for k, r in rows.items():
        lo, hi = proportion_confint(r["k"], r["n"], method="wilson")
        assert (r["lo"], r["hi"]) == (pytest.approx(lo, abs=1e-4), pytest.approx(hi, abs=1e-4))
    c = view["chart"]
    assert c["kind"] == "interval" and c["labels"] == ["1", "2", "10"] and c["x_label"] == "proportion replied"
    assert c["series"][0]["est"] == [0.5, 1.0, 0.0] and c["series"][0]["lo"][0] == pytest.approx(rows["1"]["lo"])


def test_aggregate_proportion_rejects_non_binary():
    with pytest.raises(ValueError, match="0/1"):
        apply("aggregate", tiny(), {"by": "meta.role", "measure": "proportion", "value": "timestamp"})
    with pytest.raises(ValueError, match="value"):
        apply("aggregate", tiny(), {"by": "meta.role", "measure": "proportion"})


def test_aggregate_counts_per_group_as_bar():
    _, view = apply("aggregate", tiny(), {"by": "meta.role", "measure": "utterances"})
    assert view["kind"] == "bar"
    assert dict(zip(view["labels"], view["values"])) == {"ai": 3, "person": 2}


def test_aggregate_mean_words():
    _, view = apply("aggregate", tiny(), {"by": "meta.role", "measure": "mean_words"})
    assert dict(zip(view["labels"], view["values"]))["person"] == 3.0


def test_show_conversation_returns_reply_tree():
    _, view = apply("show_conversation", tiny(), {"conversation_id": "a0"})
    assert view["kind"] == "tree"
    assert [(u["id"], u["reply_to"]) for u in view["utterances"]] == [("a0", None), ("a1", "a0"), ("a2", "a1")]


def test_fighting_words_recovers_the_words_planted_in_the_corpus():
    f = load_dir(SYNTHETIC)
    for name, cfg in [
        ("join_meta", {"source": "speaker", "keys": ["phq9"]}),
        ("filter", {"column": "meta.role", "op": "==", "value": "person"}),
        ("split", {"column": "speaker.phq9", "into": "group", "labels": ["higher", "lower"]}),
    ]:
        f, split_view = apply(name, f, cfg)
    # Scores are i * 3 // 2 for i in 0..18, with 40 or 41 messages each, so the median message sits with the 10th
    # speaker, score 13: 10 speakers (13..27) are higher, 9 (0..12) lower.
    assert "≥ 13" in split_view["note"]
    assert f.utts.groupby("group")["speaker"].nunique().to_dict() == {"higher": 10, "lower": 9}
    _, view = apply("fighting_words", f, {"group_by": "group", "class1": "higher", "class2": "lower", "ngram_max": 2})
    assert view["kind"] == "diverging"
    # make_synthetic.py: every garden clause says "compost", every bicycle clause says "bicycle", clause rates move with
    # the score and nothing else does. Mean P(compost) is about .35 higher vs .14 lower, and mirrored for bicycle.
    assert view["pos"][0]["label"] == "compost" and view["pos"][0]["value"] > 1.96
    assert view["neg"][0]["label"] == "bicycle" and view["neg"][0]["value"] < -1.96


def test_fighting_words_defaults_to_convokit_ngram_range():
    assert STAGES["fighting_words"].config.model_fields["ngram_max"].default == 3


def test_speakers_include_flags_conversations():
    f, view = apply("speakers_include", tiny(), {"key": "gender", "values": ["M", "F"], "into": "mixed"})
    assert f.utts["mixed"].to_dict() == {"a0": True, "a1": True, "a2": True, "b0": False, "b1": False}
    assert view["kind"] == "bar"


def test_speakers_include_can_ignore_case():
    t = tiny()
    t.speakers["meta.gender"] = ["f", "M", "f"]
    f, _ = apply("speakers_include", t, {"key": "gender", "values": ["M", "F"], "into": "mixed", "ignore_case": True})
    assert f.utts.groupby("conversation_id")["mixed"].first().to_dict() == {"a0": True, "b0": False}


def test_text_clean_replaces_text_and_keeps_original():
    f, _ = apply("text_clean", tiny(), {})
    assert f.utts.loc["a0", "text"] == "hi"
    t = tiny()
    t.utts.loc["a1", "text"] = "See HTTPS://example.com NOW"
    f, _ = apply("text_clean", t, {})
    assert f.utts.loc["a1", "text"] == "see <url> now"
    assert f.utts.loc["a1", "meta.original"] == "See HTTPS://example.com NOW"


def test_parse_then_politeness_adds_strategy_dicts():
    t = tiny()
    t.utts.loc["a1", "text"] = "Hello, could you please help me proofread this article?"
    f, _ = apply("parse", t, {})
    assert "meta.parsed" in f.utts.columns
    f, view = apply("politeness", f, {})
    s = f.utts.loc["a1", "meta.politeness_strategies"]
    assert s["feature_politeness_==Please=="] == 1
    assert view["kind"] == "bar" and "Please" in view["labels"]


def test_compare_groups_of_dict_column():
    t = tiny()
    t.utts["meta.feats"] = [{"x": 1, "y": 0}, {"x": 0, "y": 1}, {"x": 1, "y": 0}, {"x": 1, "y": 1}, {"x": 0, "y": 1}]
    _, view = apply("compare_groups", t, {"column": "meta.feats", "group_by": "meta.role", "class1": "ai", "class2": "person"})
    rows = {r[0]: r for r in view["rows"]}
    assert view["columns"][:4] == ["feature", "ai", "person", "diff"]
    assert rows["x"][1:4] == [1.0, 0.0, 1.0]
    assert rows["y"][1:4] == [0.3333, 1.0, -0.6667]


def labelled_synthetic():
    f = load_dir(SYNTHETIC)
    f.utts["meta.feats"] = [{"len": len(t), "q": t.count("?")} for t in f.utts["text"]]
    return f


def test_classify_cv_reports_seeded_fold_accuracies():
    cfg = {"features": "meta.feats", "label": "meta.role", "op": "==", "value": "person", "folds": 5, "seed": 1}
    _, v1 = apply("classify_cv", labelled_synthetic(), cfg)
    _, v2 = apply("classify_cv", labelled_synthetic(), cfg)
    assert v1["kind"] == "bar" and len(v1["values"]) == 5
    assert v1["values"] == v2["values"]
    assert 0.5 < sum(v1["values"]) / 5 <= 1


def test_bow_classify_ranks_terms():
    cfg = {"label": "meta.role", "op": "==", "value": "person", "top_k": 5}
    _, view = apply("bow_classify", load_dir(SYNTHETIC), cfg)
    assert view["kind"] == "diverging" and len(view["pos"]) == 5
    assert view["pos"][0]["value"] > 0 > view["neg"][0]["value"]
