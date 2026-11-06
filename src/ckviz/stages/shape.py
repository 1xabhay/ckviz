"""Reshape: join, filter, split, flag, map values."""
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from ckviz.frames import Frames
from ckviz.stages import *  # noqa: F403  helpers + registry
from ckviz.stages import stage
from ckviz.stages.describe import lab


class JoinMeta(BaseModel):
    source: Literal["speaker", "conversation"] = "speaker"
    keys: list[str] = Field(description="meta keys to copy onto each utterance")


@stage(JoinMeta)
def join_meta(f: Frames, c: JoinMeta):
    src, on = (f.speakers, "speaker") if c.source == "speaker" else (f.convos, "conversation_id")
    utts = f.utts.copy()
    for k in c.keys:
        utts[f"{c.source}.{k}"] = utts[on].map(need(src, f"meta.{k}"))
    return Frames(utts, f.speakers, f.convos), table(utts)


class Filter(BaseModel):
    column: str = Field(description="utterance column, e.g. meta.role or speaker.phq9")
    op: Op = "=="
    value: str


@stage(Filter)
def filter(f: Frames, c: Filter):
    utts = f.utts[predicate(f.utts, c.column, c.op, c.value)]
    return prune(f, utts), table(utts)


class SpeakersInclude(BaseModel):
    key: str = Field(description="speaker meta key, e.g. gender")
    values: list[str] = Field(description="flag conversations whose speakers cover all of these")
    into: str = Field("flag", description="name of the new True/False column")
    ignore_case: bool = False


@stage(SpeakersInclude)
def speakers_include(f: Frames, c: SpeakersInclude):
    norm = (lambda v: str(v).casefold()) if c.ignore_case else (lambda v: v)
    per_convo = f.utts["speaker"].map(need(f.speakers, f"meta.{c.key}").map(norm)).groupby(f.utts["conversation_id"]).agg(set)
    flag = per_convo.map({norm(v) for v in c.values}.issubset)
    counts = flag.value_counts()
    return (Frames(f.utts.assign(**{c.into: f.utts["conversation_id"].map(flag)}), f.speakers, f.convos),
            {"kind": "bar", "labels": ["True", "False"], "values": [int(counts.get(True, 0)), int(counts.get(False, 0))],
             "measure": "conversations"})


class Split(BaseModel):
    level: Level = Field("utterance", description="frame holding the column; the label column is written there")
    column: str = Field(description="numeric column to split, e.g. speaker.phq9 or (level speaker) meta.active_days")
    method: Literal["median", "threshold", "quantile"] = Field("median", description="where to cut: rows >= the cut get labels[0]")
    threshold: float = Field(0.0, description="method threshold: the cut value")
    quantile: float = Field(0.75, gt=0, lt=1, description="method quantile: cut at this quantile, e.g. 0.75 = top quartile vs rest")
    into: str = Field("group", description="name of the new label column (use meta.<key> at speaker/conversation level for join_meta)")
    labels: list[str] = Field(["high", "low"], min_length=2, max_length=2, description="labels for >= cut, < cut")


@stage(Split)
def split(f: Frames, c: Split):
    """Two-group split of a numeric column; missing values stay unlabelled."""
    df = at(f, c.level)
    col = pd.to_numeric(need(df, c.column))
    cut = {"median": col.median(), "threshold": c.threshold, "quantile": col.quantile(c.quantile)}[c.method]
    df = df.assign(**{c.into: col.ge(cut).map({True: c.labels[0], False: c.labels[1]}).where(col.notna())})
    f = f._replace(**{{"utterance": "utts", "speaker": "speakers", "conversation": "convos"}[c.level]: df})
    note = {"median": "median", "threshold": "threshold", "quantile": f"quantile {c.quantile:g}"}[c.method]
    return f, {"kind": "bar", "labels": c.labels, "values": [int((df[c.into] == l).sum()) for l in c.labels],
               "measure": f"{c.level}s", "note": f"{c.labels[0]}: {c.column} ≥ {cut:g} ({note})"}


class Rollup(BaseModel):
    column: str = Field(description="utterance column to aggregate")
    to: Literal["speaker", "conversation"] = "speaker"
    agg: Literal["mean", "median", "sum", "max", "min", "count", "nunique"] = "mean"
    into: str = Field(description="new meta key on the speaker / conversation")


@stage(Rollup)
def rollup(f: Frames, c: Rollup):
    """Aggregate an utterance column up to speakers or conversations (written as meta.<into>)."""
    col = need(f.utts, c.column)
    col = col if c.agg in ("count", "nunique") else pd.to_numeric(col)
    s = col.groupby(f.utts["speaker" if c.to == "speaker" else "conversation_id"]).agg(c.agg)
    target = at(f, c.to).assign(**{f"meta.{c.into}": s})
    f = f._replace(speakers=target) if c.to == "speaker" else f._replace(convos=target)
    return f, table(target)


class FirstN(BaseModel):
    n: int = Field(2, ge=1, description="keep this many earliest utterances per conversation")


@stage(FirstN)
def first_n(f: Frames, c: FirstN):
    keep = f.utts.sort_values("timestamp", kind="stable").groupby("conversation_id", sort=False).head(c.n).index
    return prune(f, f.utts[f.utts.index.isin(keep)]), table(f.utts[f.utts.index.isin(keep)])


class MapValues(BaseModel):
    column: str = Field(description="utterance column whose values are mapped")
    mapping: list[str] = Field(description='"key=value" strings, e.g. 1=apple (a codebook)')
    into: str = Field(description="new column; unmapped values become null")


@stage(MapValues)
def map_values(f: Frames, c: MapValues):
    m = dict(s.split("=", 1) for s in c.mapping)
    src = need(f.utts, c.column)
    out = src.map(lambda v: None if pd.isna(v) else m.get(lab(v)))
    miss = src.notna() & out.isna()
    utts = f.utts.assign(**{c.into: out})
    view = table(utts[[c.column, c.into]])
    view["note"] = f"{int(miss.sum())} non-missing values unmapped" + (f", e.g. {sorted(map(lab, src[miss].unique()))[:5]}" if miss.any() else "")
    return Frames(utts, f.speakers, f.convos, f.meta), view
