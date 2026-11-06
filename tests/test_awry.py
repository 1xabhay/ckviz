"""Conversations Gone Awry stages (Zhang et al. 2018): exchange features, paired log-odds, paired grouped-CV classifier."""
import numpy as np
import pandas as pd
import pytest

from ckviz.frames import Frames
from ckviz.stages import apply


def pairs(n_pages: int = 6, signal: bool = True) -> Frames:
    """n_pages talk pages x 2 pairs; every conversation has 2 comments. Awry conversations' first comment
    asks a direct question (if signal) and says 'why'; on-track ones say 'thanks'."""
    utts, convos = [], []
    for p in range(n_pages):
        for q in range(2):
            for awry in (True, False):
                cid = f"p{p}q{q}{'a' if awry else 'o'}"
                other = f"p{p}q{q}{'o' if awry else 'a'}"
                convos.append({"id": cid, "meta.page_id": p, "meta.pair_id": other, "meta.awry": awry})
                dq = int(awry) if signal else (p + q) % 2
                utts.append({"id": cid + "0", "conversation_id": cid, "timestamp": 0.0, "speaker": "x", "reply_to": None,
                             "text": "why is this here" if awry else "thanks for this",
                             "meta.ps": {"Direct_question": dq * 2, "Gratitude": int(not awry)}})
                utts.append({"id": cid + "1", "conversation_id": cid, "timestamp": 1.0, "speaker": "y", "reply_to": cid + "0",
                             "text": "ok", "meta.ps": {"Direct_question": 0, "Gratitude": 0}})
    u = pd.DataFrame(utts).set_index("id")
    return Frames(u, pd.DataFrame({"id": ["x", "y"]}).set_index("id"), pd.DataFrame(convos).set_index("id"))


def test_exchange_spreads_features_by_comment_position_and_binarises():
    f, view = apply("exchange", pairs(), {"column": "meta.ps", "into": "ps2"})
    d = f.convos.loc["p0q0a", "meta.ps2"]
    assert d == {"Direct_question@1": 1, "Gratitude@1": 0, "Direct_question@2": 0, "Gratitude@2": 0}
    assert f.convos.loc["p0q0o", "meta.ps2"]["Gratitude@1"] == 1
    assert view["kind"] == "table"


def test_exchange_keeps_counts_and_subsets_keys():
    f, _ = apply("exchange", pairs(), {"column": "meta.ps", "into": "dq", "binarize": False, "keys": ["Direct_question"]})
    assert f.convos.loc["p0q0a", "meta.dq"] == {"Direct_question@1": 2, "Direct_question@2": 0}


def test_exchange_pads_short_conversations_with_zeros():
    t = pairs()
    t = t._replace(utts=t.utts.drop(index="p0q0a1"))
    f, _ = apply("exchange", t, {"column": "meta.ps", "into": "ps2"})
    assert f.convos.loc["p0q0a", "meta.ps2"]["Gratitude@2"] == 0


def test_log_odds_compares_shares_with_mcnemar_test():
    f, _ = apply("exchange", pairs(), {"column": "meta.ps", "into": "ps2"})
    _, view = apply("log_odds", f, {"column": "meta.ps2", "label": "meta.awry", "value": "true", "min_count": 0})
    rows = {r[0]: dict(zip(view["columns"], r)) for r in view["rows"]}
    dq = rows["Direct_question@1"]
    assert (dq["awry_share"], dq["other_share"]) == (1.0, 0.0)
    assert dq["log_odds"] > 0 and rows["Gratitude@1"]["log_odds"] < 0
    assert dq["p"] < 0.001
    assert rows["Gratitude@2"]["log_odds"] == 0 and rows["Gratitude@2"]["p"] == 1.0


CLS = {"label": "meta.awry", "value": "true", "pair": "meta.pair_id", "group": "meta.page_id"}


def test_paired_classify_learns_a_separating_feature_leave_one_page_out():
    f, _ = apply("exchange", pairs(), {"column": "meta.ps", "into": "ps2", "keys": ["Direct_question"]})
    _, view = apply("paired_classify", f, {**CLS, "features": ["meta.ps2"]})
    row = dict(zip(view["columns"], view["rows"][0]))
    assert row["accuracy"] == 1.0
    assert (row["pairs"], row["folds"], row["n_features"]) == (12, 6, 2)


def test_paired_classify_is_chance_without_signal():
    f, _ = apply("exchange", pairs(signal=False), {"column": "meta.ps", "into": "ps2", "keys": ["Direct_question"]})
    _, view = apply("paired_classify", f, {**CLS, "features": ["meta.ps2"]})
    assert view["rows"][0][view["columns"].index("accuracy")] <= 0.5


def test_paired_classify_bag_of_words_over_the_exchange():
    _, view = apply("paired_classify", pairs(), {**CLS, "features": ["unigrams"], "max_words": 100})
    row = dict(zip(view["columns"], view["rows"][0]))
    assert row["accuracy"] == 1.0 and row["n_features"] == 7  # why is this here / thanks for / ok


def test_paired_classify_rejects_unpaired_conversations():
    t = pairs()
    t.convos.loc["p0q0a", "meta.pair_id"] = "nope"
    f, _ = apply("exchange", t, {"column": "meta.ps", "into": "ps2"})
    with pytest.raises(ValueError, match="pair"):
        apply("paired_classify", f, {**CLS, "features": ["meta.ps2"]})


@pytest.mark.parametrize("vectors", ["/etc/x", "../../x", "downloads/../../x"])
def test_prompt_types_vectors_must_stay_under_the_data_dir(tmp_path, monkeypatch, vectors):
    monkeypatch.setattr("ckviz.datasets.DATA", tmp_path)
    with pytest.raises(ValueError, match="data directory"):
        apply("prompt_types", pairs(n_pages=1), {"vectors": vectors})


def test_prompt_types_assigns_nearest_type_or_none(tmp_path, monkeypatch):
    monkeypatch.setattr("ckviz.datasets.DATA", tmp_path)
    t = pairs(n_pages=1)
    ids = list(t.utts.index[:3])
    np.save(tmp_path / "v.npy", np.array([[0.2, 0.9], [0.8, 0.3], [1.0, 1.2]]))
    (tmp_path / "v.keys").write_text("\n".join(ids))
    f, view = apply("prompt_types", t, {"vectors": "v"})
    pt = f.utts["meta.prompt_type"]
    assert pt[ids[0]] == {"type_0": 1, "type_1": 0}
    assert pt[ids[1]] == {"type_0": 0, "type_1": 1}
    assert pt[ids[2]] == {"type_0": 0, "type_1": 0}  # distance >= 1 to every centroid: the paper's null type
    assert pt[t.utts.index[5]] == {"type_0": 0, "type_1": 0}  # no vector
    assert view["kind"] == "bar"


def test_paired_classify_can_tune_C_inside_each_training_fold():
    f, _ = apply("exchange", pairs(), {"column": "meta.ps", "into": "ps2", "keys": ["Direct_question"]})
    _, view = apply("paired_classify", f, {**CLS, "features": ["meta.ps2"], "tune_C": True})
    assert view["rows"][0][view["columns"].index("accuracy")] == 1.0
    assert "nested" in view["note"]
