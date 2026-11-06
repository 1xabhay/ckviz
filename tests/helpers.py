from pathlib import Path

import pandas as pd

from ckviz.frames import Frames

SYNTHETIC = Path(__file__).parent.parent / "examples/data/synthetic"


def tiny() -> Frames:
    utts = pd.DataFrame({
        "id": ["a0", "a1", "a2", "b0", "b1"],
        "speaker": ["bot", "u1", "bot", "bot", "u2"],
        "conversation_id": ["a0", "a0", "a0", "b0", "b0"],
        "reply_to": [None, "a0", "a1", None, "b0"],
        "timestamp": [0, 1, 2, 0, 1],
        "text": ["hi", "i feel sad today", "oh no", "hey", "good good"],
        "meta.role": ["ai", "person", "ai", "ai", "person"],
    }).set_index("id")
    speakers = pd.DataFrame({
        "id": ["bot", "u1", "u2"],
        "meta.role": ["ai", "person", "person"],
        "meta.phq9": [None, 20, 3],
        "meta.gender": ["F", "M", "F"],
    }).set_index("id")
    convos = pd.DataFrame({"id": ["a0", "b0"], "meta.source": ["s1", "s2"]}).set_index("id")
    return Frames(utts, speakers, convos)
