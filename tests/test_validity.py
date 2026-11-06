"""Seeded null simulations: each stage's p-values / accuracies must not be inflated by nesting, leakage or pairing."""
import numpy as np
import pandas as pd

from ckviz.frames import Frames
from ckviz.stages import apply


def nested(convos=20, per=30, icc=0.1, seed=0) -> Frames:
    """Utterances in conversations; the group label and a speaker score are per conversation, y has a conversation
    effect but no group effect (the null)."""
    rng = np.random.default_rng(seed)
    cid = np.repeat([f"c{i}" for i in range(convos)], per)
    u = rng.normal(0, np.sqrt(icc), convos)[np.repeat(range(convos), per)]
    utts = pd.DataFrame({"conversation_id": cid, "speaker": [f"s{c}" for c in cid], "reply_to": None, "timestamp": 0.0,
                         "text": "x", "meta.g": np.where(np.repeat(range(convos), per) % 2, "A", "B"),
                         "y": u + rng.normal(0, np.sqrt(1 - icc), len(cid)),
                         "meta.score": np.repeat(rng.normal(size=convos), per)},
                        index=[f"u{i}" for i in range(len(cid))])
    return Frames(utts, pd.DataFrame(index=pd.Index(sorted(set(utts.speaker)), name="id")),
                  pd.DataFrame(index=pd.Index(sorted(set(cid)), name="id")))


def test_compare_groups_withholds_p_when_group_is_nested_in_conversations():
    _, v = apply("compare_groups", nested(), {"column": "y", "group_by": "meta.g", "class1": "A", "class2": "B"})
    (row,) = v["rows"]
    r = dict(zip(v["columns"], row))
    assert r["p"] is None and r["p_holm"] is None
    assert r["d"] is not None and abs(r["diff"] - (r["A"] - r["B"])) < 1e-3
    assert "withheld" in v["note"] and "20 clusters" in v["note"]


def test_compare_groups_keeps_p_when_group_varies_within_conversations():
    f = nested()
    f = f._replace(utts=f.utts.assign(**{"meta.g": np.tile(["A", "B"], len(f.utts) // 2)}))
    _, v = apply("compare_groups", f, {"column": "y", "group_by": "meta.g", "class1": "A", "class2": "B"})
    assert v["rows"][0][5] is not None and "withheld" not in v["note"]


def test_correlate_withholds_p_when_y_is_nested_in_speakers():
    _, v = apply("correlate", nested(), {"x": "y", "y": "meta.score"})
    r = dict(zip(v["columns"], v["rows"][0]))
    assert r["p"] is None and r["p_holm"] is None and r["r"] is not None
    assert "withheld" in v["note"]


def leaky(convos=100, per=8, dims=20, seed=0) -> Frames:
    """Conversation-level label independent of the features; features carry a conversation fingerprint."""
    rng = np.random.default_rng(seed)
    g = np.repeat(range(convos), per)
    x = rng.normal(size=(convos, dims))[g] + 0.3 * rng.normal(size=(len(g), dims))
    label = rng.permutation(np.r_[np.zeros(convos // 2), np.ones(convos - convos // 2)])[g]
    utts = pd.DataFrame({"conversation_id": [f"c{i}" for i in g], "speaker": "s", "meta.y": label.astype(int),
                         "meta.x": [{f"f{j}": v for j, v in enumerate(r)} for r in x]}, index=[f"u{i}" for i in range(len(g))])
    return Frames(utts, pd.DataFrame(), pd.DataFrame())


def test_classify_cv_groups_conversations_by_default():
    cfg = {"features": "meta.x", "label": "meta.y", "value": "1"}
    _, grouped = apply("classify_cv", leaky(), cfg)
    _, leak = apply("classify_cv", leaky(), cfg | {"group": "none"})
    assert np.mean(leak["values"]) > 0.6  # the scenario does leak with plain k-fold
    assert np.mean(grouped["values"]) < 0.6
    assert "conversation" in grouped["note"]


def test_log_odds_mcnemar_false_positive_rate_is_nominal():
    """300 matched pairs, 200 null markers whose prevalence is shared within a pair (as on one talk page)."""
    rng = np.random.default_rng(0)
    pairs, k = 300, 200
    prev = rng.beta(1, 3, size=(pairs, k))
    awry, other = rng.random((pairs, k)) < prev, rng.random((pairs, k)) < prev
    ids = [f"a{i}" for i in range(pairs)] + [f"o{i}" for i in range(pairs)]
    x = np.vstack([awry, other]).astype(int)
    convos = pd.DataFrame({"meta.x": [{f"f{j}": v for j, v in enumerate(r)} for r in x], "meta.awry": [True] * pairs + [False] * pairs,
                           "meta.pair_id": ids[pairs:] + ids[:pairs]}, index=pd.Index(ids, name="id"))
    _, v = apply("log_odds", Frames(pd.DataFrame(), pd.DataFrame(), convos),
                 {"column": "meta.x", "label": "meta.awry", "value": "true", "min_count": 0})
    p = np.array([r[v["columns"].index("p")] for r in v["rows"]])
    assert len(p) == k and (p < 0.05).mean() < 0.10
