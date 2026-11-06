import pandas as pd
import pytest

pytest.importorskip("vaderSentiment")
import ckviz_sentiment  # noqa: E402,F401  registers the stage
from ckviz.frames import Frames  # noqa: E402
from ckviz.stages import apply  # noqa: E402


def frames(texts) -> Frames:
    utts = pd.DataFrame({"speaker": "u", "conversation_id": "c", "reply_to": None, "timestamp": 0, "text": texts},
                        index=pd.Index([f"u{i}" for i in range(len(texts))], name="id"))
    return Frames(utts, pd.DataFrame(index=pd.Index(["u"], name="id")), pd.DataFrame(index=pd.Index(["c"], name="id")))


# Reference values: vaderSentiment 3.3.2 SentimentIntensityAnalyzer().polarity_scores(...)
TEXTS = ["I love this so much!", "This is terrible.", "The sky is blue.", "not bad at all :)"]
COMPOUND = [0.6696, -0.4767, 0.0, 0.705]


def test_vader_compound_matches_reference():
    f, view = apply("vader", frames(TEXTS), {})
    assert f.utts["sentiment"].tolist() == COMPOUND
    assert "sentiment_pos" not in f.utts and view["kind"] == "table" and "sentiment" in view["columns"]


def test_vader_components_and_missing_text():
    f, _ = apply("vader", frames(["This is terrible.", None, ""]), {"components": True, "into": "s"})
    assert f.utts.loc["u0", ["s", "s_neg", "s_neu", "s_pos"]].tolist() == [-0.4767, 0.608, 0.392, 0.0]
    assert pd.isna(f.utts.loc["u1", "s"]) and f.utts.loc["u2", "s"] == 0.0
