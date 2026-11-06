"""lexicon_net: a word list minus false-positive phrases (Al-Mosaiwi & Johnstone 2018's absolutist index)."""
import pytest

from ckviz.stages import apply
from helpers import tiny


def test_lexicon_net_subtracts_false_positive_phrases():
    t = tiny()
    t.utts.loc["a1", "text"] = "Hello everyone. I always fail, not completely though"  # 8 tokens
    f, view = apply("lexicon_net", t, {"words": ["always", "complete*", "everyone"],
                                       "exclude": ["hello everyone", "not completely"], "into": "absolutist"})
    assert f.utts.loc["a1", "absolutist"] == pytest.approx((3 - 2) / 8)
    assert f.utts.loc["a1", "absolutist_fp"] == pytest.approx(2 / 8)
    assert f.utts.loc["a1", "n_tokens"] == 8
    assert f.utts.loc["a0", "absolutist"] == 0
    assert view["kind"] == "table"


def test_lexicon_net_counts_every_phrase_occurrence_case_insensitively():
    t = tiny()
    t.utts.loc["a1", "text"] = "Not all. NOT ALL! all"
    f, _ = apply("lexicon_net", t, {"words": ["all"], "exclude": ["not all"], "into": "x"})
    assert f.utts.loc["a1", "x"] == pytest.approx((3 - 2) / 5)


def test_lexicon_net_without_exclusions_equals_lexicon():
    t = tiny()
    a, _ = apply("lexicon_net", t, {"words": ["sad", "good"], "into": "x"})
    b, _ = apply("lexicon", t, {"words": ["sad", "good"], "into": "x"})
    assert a.utts["x"].tolist() == b.utts["x"].tolist()
