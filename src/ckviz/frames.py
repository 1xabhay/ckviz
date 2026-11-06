"""A corpus as three ConvoKit-shaped DataFrames (index=id, meta keys as `meta.<key>` columns) plus corpus-level meta."""
import json
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from convokit import Corpus


class Frames(NamedTuple):
    utts: pd.DataFrame
    speakers: pd.DataFrame
    convos: pd.DataFrame
    meta: dict = {}  # corpus.json + any extra *.json shipped with the corpus (e.g. extra.json); never mutated


def from_corpus(c: Corpus) -> Frames:
    drop = lambda df: df.drop(columns="vectors", errors="ignore")
    # Some legacy corpora (e.g. movie-corpus) come back with speaker meta unprefixed; all non-vector
    # speaker/conversation columns are meta, so normalise them to meta.<key>.
    meta = lambda df: drop(df).rename(columns=lambda col: col if col.startswith("meta.") else f"meta.{col}")
    return Frames(drop(c.get_utterances_dataframe()), meta(c.get_speakers_dataframe()), meta(c.get_conversations_dataframe()),
                  dict(c.meta))


def meta_key(col: str) -> str:
    """Corpus meta key for a frame column. from_pandas splits meta names on ".", so derived stage
    columns (e.g. speaker.phq9) become "_"-joined keys (speaker_phq9)."""
    return col.removeprefix("meta.") if col.startswith("meta.") else col.replace(".", "_")


def to_corpus(f: Frames) -> Corpus:
    utts = f.utts.rename(columns=lambda c: c if c in UTT_FIELDS else "meta." + meta_key(c))
    c = Corpus.from_pandas(utts, f.speakers.copy(), f.convos.copy())  # from_pandas mutates its inputs
    for u in c.iter_utterances():
        if u.reply_to is not None and pd.isna(u.reply_to):  # from_pandas turns missing reply_to into NaN
            u.reply_to = None
    return c


STANDARD_FILES = {"utterances.jsonl", "speakers.json", "users.json", "conversations.json", "corpus.json", "index.json"}


def load_dir(path: str | Path, **kw) -> Frames:
    f = from_corpus(Corpus(filename=str(path), **kw))
    extra = {p.stem: json.loads(p.read_text()) for p in sorted(Path(path).glob("*.json")) if p.name not in STANDARD_FILES}
    return f._replace(meta={**f.meta, **extra})


UTT_FIELDS = {"speaker", "conversation_id", "reply_to", "timestamp", "text"}
