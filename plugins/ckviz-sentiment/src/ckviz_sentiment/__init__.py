"""ckviz stage wrapping VADER (Hutto & Gilbert 2014), a lexicon + rules sentiment model tuned for short social-media text."""
import pandas as pd
from pydantic import BaseModel, Field

from ckviz.frames import Frames
from ckviz.stages import need, stage, table


class Vader(BaseModel):
    column: str = Field("text", description="utterance text column")
    into: str = Field("sentiment", description="new column: VADER compound score, -1 (negative) … +1 (positive)")
    components: bool = Field(False, description="also write <into>_neg, <into>_neu, <into>_pos (proportions of the text)")


@stage(Vader)
def vader(f: Frames, c: Vader):
    """VADER sentiment per utterance; missing text stays missing (empty text scores 0)."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    score = SentimentIntensityAnalyzer().polarity_scores
    text = need(f.utts, c.column)
    s = pd.DataFrame([score(str(t)) if pd.notna(t) else {} for t in text], index=f.utts.index,
                     columns=["compound", "neg", "neu", "pos"])
    new = {c.into: s["compound"]} | ({f"{c.into}_{k}": s[k] for k in ("neg", "neu", "pos")} if c.components else {})
    utts = f.utts.assign(**new)
    return Frames(utts, f.speakers, f.convos), table(utts[["speaker", c.column, *new]])
