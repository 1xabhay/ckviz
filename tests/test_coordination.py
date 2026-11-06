"""Linguistic style coordination (Danescu-Niculescu-Mizil et al. 2012), on hand-computable exchanges."""
import pandas as pd
import pytest

from ckviz.frames import Frames
from ckviz.stages import apply


def exchanges(pairs):
    """pairs: (target, target_text, speaker, reply_text); each pair is its own two-utterance conversation."""
    rows = []
    for i, (t, tt, s, st) in enumerate(pairs):
        rows += [(f"c{i}a", t, f"c{i}a", None, 0, tt), (f"c{i}b", s, f"c{i}a", f"c{i}a", 1, st)]
    utts = pd.DataFrame(rows, columns=["id", "speaker", "conversation_id", "reply_to", "timestamp", "text"]).set_index("id")
    speakers = pd.DataFrame({"id": ["boss", "peer", "s1", "s2"], "meta.admin": [True, False, False, False]}).set_index("id")
    convos = pd.DataFrame(index=pd.Index(sorted(set(utts["conversation_id"])), name="id"))
    return Frames(utts, speakers, convos)


def corpus():
    # Toward boss: s1 says "the" exactly when boss does -> article C = P(the|the) - P(the) = 1 - 2/4 = 0.5
    #              s2 says "the" half the time regardless  -> 1/2 - 2/4 = 0
    # Toward peer: s1 always says "the"                    -> 1 - 1 = 0
    #              s2 never says "the"                     -> 0 - 0 = 0
    ex = []
    for s, replies in [("s1", ["the cat", "the cat", "dog", "dog"]), ("s2", ["the cat", "dog", "the cat", "dog"])]:
        ex += [("boss", t, s, r) for t, r in zip(["the cat", "the cat", "dog", "dog"], replies)]
    ex += [("peer", t, "s1", "the cat") for t in ["the cat", "the cat", "dog", "dog"]]
    ex += [("peer", t, "s2", "dog") for t in ["the cat", "the cat", "dog", "dog"]]
    return exchanges(ex)


def rows(view):
    return {r[0]: r[1:] for r in view["rows"]}


def test_coordination_toward_target_groups():
    _, view = apply("coordination", corpus(), {"group_by": "meta.admin", "side": "target", "target_thresh": 1})
    assert view["kind"] == "table"
    assert view["columns"][:3] == ["measure", "False", "True"]
    r = rows(view)
    assert r["article"][:2] == pytest.approx([0.0, 0.25])  # mean of s1 (0.5) and s2 (0) per group
    assert r["aggregate 3"][:2] == pytest.approx([0.0, 0.25])  # only one marker defined, so agg3 = article
    assert "aggregate 1" not in r  # nobody has all 8 markers
    assert "ppron" not in r  # no marker anywhere -> no row


def test_speaker_selector_restricts_who_replies():
    _, view = apply("coordination", corpus(), {"group_by": "meta.admin", "side": "target", "target_thresh": 1,
                                               "speaker_column": "id", "speaker_op": "==", "speaker_value": "s1"})
    assert rows(view)["article"][:2] == pytest.approx([0.0, 0.5])


def test_group_speakers_and_ttest():
    # one group per replying speaker (boss and peer never reply, so they form no group)
    _, view = apply("coordination", corpus(), {"group_by": "id", "side": "speaker", "target_thresh": 1})
    assert view["columns"][:3] == ["measure", "s1", "s2"]
    r = rows(view)
    # s1 over all 8 exchanges: P(the | target the) = 4/4, P(the) = 6/8 -> 0.25; s2: 1/4 - 2/8 = 0
    assert r["article"][:2] == pytest.approx([0.25, 0.0])
    assert r["article"][3] is None  # one speaker per group: no t-test


def test_ttest_between_groups():
    _, view = apply("coordination", corpus(), {"group_by": "meta.admin", "side": "target", "target_thresh": 1})
    assert view["columns"] == ["measure", "False", "True", "diff", "p"]
    # [0.5, 0] vs [0, 0]: pooled t = 1, df = 2
    assert rows(view)["article"][3] == pytest.approx(0.423, abs=1e-3)
    assert "True: 0 / 2 speakers" in view["note"]
