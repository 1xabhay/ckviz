"""Look at raw conversations."""
import json


from pydantic import BaseModel, Field

from ckviz.frames import Frames
from ckviz.stages import *  # noqa: F403  helpers + registry
from ckviz.stages import stage


class ShowConversation(BaseModel):
    conversation_id: str = Field("", description="blank = first conversation")


@stage(ShowConversation)
def show_conversation(f: Frames, c: ShowConversation):
    cid = c.conversation_id or f.utts["conversation_id"].iloc[0]
    utts = f.utts[f.utts["conversation_id"] == cid].sort_values("timestamp").reset_index()
    if utts.empty:
        raise ValueError(f"no conversation {cid!r}")
    return f, {"kind": "tree", "conversation_id": cid, "utterances": json.loads(utts.to_json(orient="records"))}
