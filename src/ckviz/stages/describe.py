"""Descriptive stages for longitudinal chat data: time units, reply latency, distributions, counts. No p-values."""
import math
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ckviz.frames import Frames
from ckviz.stages import *  # noqa: F403  helpers + registry
from ckviz.stages import stage

KEY = {"speaker": "speaker", "conversation": "conversation_id"}


class TimeBins(BaseModel):
    column: str = Field("timestamp", description="epoch seconds, or ISO datetimes (naive ones are read as already in tz)")
    unit: Literal["day", "week", "weekday", "hour"] = "day"
    relative_to: Literal["none", "speaker", "conversation"] = Field(
        "none", description="none = calendar date / ISO week; speaker or conversation = day/week 1 is its first message")
    tz: str = Field("UTC", description="time zone for calendar days, e.g. America/New_York")
    into: str = Field("day", description="new column; weekday is 1 = Monday … 7 = Sunday")


@stage(TimeBins)
def time_bins(f: Frames, c: TimeBins):
    if c.relative_to != "none" and c.unit in ("weekday", "hour"):
        raise ValueError(f"relative_to={c.relative_to} only applies to unit day or week, not {c.unit}")
    raw = need(f.utts, c.column)
    num = pd.to_numeric(raw, errors="coerce")
    if num.notna().sum() == raw.notna().sum():
        t = pd.to_datetime(num, unit="s", utc=True).dt.tz_convert(c.tz).dt.tz_localize(None)
    else:
        t = pd.to_datetime(raw, format="ISO8601")
        t = t.dt.tz_convert(c.tz).dt.tz_localize(None) if t.dt.tz is not None else t
    if c.unit == "hour":
        out = t.dt.hour.astype("Int64")
    elif c.unit == "weekday":
        out = (t.dt.weekday + 1).astype("Int64")
    elif c.relative_to == "none":
        out = t.dt.strftime("%Y-%m-%d") if c.unit == "day" else t.dt.strftime("%G-W%V")
        out = out.where(t.notna())
    else:
        d = t.dt.normalize()
        n = ((d - d.groupby(f.utts[KEY[c.relative_to]]).transform("min")).dt.days + 1).astype("Int64")
        out = n if c.unit == "day" else (n - 1) // 7 + 1
    utts = f.utts.assign(**{c.into: out})
    return Frames(utts, f.speakers, f.convos), table(utts[["speaker", "conversation_id", c.column, c.into]])


class ReplyLatency(BaseModel):
    into: str = "latency_s"
    only_cross_speaker: bool = Field(True, description="ignore replies to one's own message")
    time_column: str = Field("timestamp", description="epoch-seconds column, e.g. meta.sent_at (actual send time)")
    fallback_to_timestamp: bool = Field(False, description="use timestamp where time_column is null or absent")


@stage(ReplyLatency)
def reply_latency(f: Frames, c: ReplyLatency):
    """Seconds from the parent utterance (reply_to) to this one. Parents not in the data (e.g. filtered out) give NaN."""
    ts = pd.to_numeric(f.utts["timestamp"])
    if c.time_column in f.utts:
        t = pd.to_numeric(f.utts[c.time_column])
        note = ""
        if c.fallback_to_timestamp:
            note = f"{int(t.isna().sum())} of {len(t)} utterances had no {c.time_column}; used timestamp"
            t = t.fillna(ts)
    elif c.fallback_to_timestamp:
        t, note = ts, f"{c.time_column} absent; used timestamp for all"
    else:
        need(f.utts, c.time_column)
    parent = f.utts["reply_to"]
    lat = t - parent.map(t)
    if c.only_cross_speaker:
        lat = lat.where(f.utts["speaker"] != parent.map(f.utts["speaker"]))
    utts = f.utts.assign(**{c.into: lat})
    view = table(utts[["speaker", "reply_to", c.into]])
    view["note"] = note
    return Frames(utts, f.speakers, f.convos), view


class WordCount(BaseModel):
    column: str = "text"
    into: str = "words"


@stage(WordCount)
def word_count(f: Frames, c: WordCount):
    """Whitespace-separated words per utterance (empty text = 0)."""
    utts = f.utts.assign(**{c.into: need(f.utts, c.column).str.split().str.len()})
    return Frames(utts, f.speakers, f.convos), table(utts[["speaker", c.column, c.into]])


def lab(v) -> str:
    return "(missing)" if pd.isna(v) else f"{v:g}" if isinstance(v, float) else str(v)


def edges(lo: float, hi: float, integer: bool, max_bins: int = 20) -> np.ndarray:
    """Half-open bins [a, b) with a 1/2/5×10^k step, the last one holding hi; integer data never gets steps < 1."""
    step = 10 ** math.floor(math.log10((hi - lo) / max_bins or 1))
    while True:
        for m in (1, 2, 5):
            s = max(step * m, 1.0) if integer else step * m
            start = math.floor(lo / s) * s
            if (n := math.floor((hi - start) / s) + 1) <= max_bins:
                return start + s * np.arange(n + 1)
        step *= 10


def log_edges(lo: float, hi: float) -> np.ndarray:
    """1-2-5 × 10^k edges from the largest one <= lo to the first one >= hi (lo > 0)."""
    cands = [float(f"{m}e{k}") for k in range(math.floor(math.log10(lo)), math.floor(math.log10(hi)) + 2) for m in (1, 2, 5)]
    start = max(i for i, e in enumerate(cands) if e <= lo)
    stop = next(i for i, e in enumerate(cands) if e >= hi and i > start)
    return np.array(cands[start:stop + 1])


def order(vals) -> list:
    """Distinct values sorted (numbers numerically), one missing value last."""
    vals = list(pd.unique(vals))
    known = [v for v in vals if pd.notna(v)]
    try:
        known.sort()
    except TypeError:
        known.sort(key=str)
    return known + [v for v in vals if pd.isna(v)][:1]


class Describe(BaseModel):
    level: Level = "utterance"
    column: str = Field(description="numeric column, or a dict-valued one with key")
    key: str = Field("", description="for dict-valued columns (e.g. meta.usage): the key to describe")
    group_by: str = Field("", description="optional column holding group labels")
    log_bins: Literal["auto", "on", "off"] = Field(
        "auto", description="1-2-5 × 10^k histogram bins (latency-like data); auto = when min > 0 and max/min > 1000")


@stage(Describe)
def describe(f: Frames, c: Describe):
    """n, missing, mean, sd, quartiles per group, plus a histogram (one series per group). Descriptive only."""
    df = at(f, c.level)
    col = need(df, c.column)
    if c.key:
        col = col.map(lambda d: d.get(c.key) if isinstance(d, dict) else None)
    x = pd.to_numeric(col, errors="coerce")
    if (x.isna() & col.notna()).any():
        raise ValueError(f"{c.column} is not numeric (e.g. {col[x.isna() & col.notna()].iloc[0]!r})"
                         + ("; give key for a dict-valued column" if isinstance(col.dropna().iloc[0], dict) else ""))
    g = need(df, c.group_by) if c.group_by else pd.Series("all", index=df.index)
    v = x.dropna()
    integer = bool((v == v.round()).all())
    log = c.log_bins == "on" or (c.log_bins == "auto" and len(v) and v.min() > 0 and v.max() / v.min() > 1000)
    if log and len(v) and v.min() <= 0:
        raise ValueError(f"log_bins needs values > 0; min is {v.min():g}")
    if log and len(v):
        bins = log_edges(v.min(), v.max())
        labels = [f"{a:g}–{b:g}" for a, b in zip(bins, bins[1:])]
    else:
        bins = edges(v.min(), v.max(), integer) if len(v) else np.array([])
        step = bins[1] - bins[0] if len(bins) > 1 else 0
        labels = [f"{a:g}" if integer and step == 1 else f"{a:g}–{a + step:g}" for a in bins[:-1]]
    rows, series = [], []
    for k in order(g):
        xs = x[(g == k) if pd.notna(k) else g.isna()]
        d = xs.dropna()
        q = d.quantile([0.25, 0.5, 0.75]) if len(d) else pd.Series([np.nan] * 3)
        rows.append([lab(k), len(d), int(xs.isna().sum()), *(round(float(s), 3) for s in
                     (d.mean(), d.std(), d.min(), q.iloc[0], q.iloc[1], q.iloc[2], d.max()))])
        series.append({"name": lab(k), "values": np.histogram(d, bins)[0].tolist() if len(bins) else []})
    name = f"{c.column}.{c.key}" if c.key else c.column
    return f, {"kind": "table", "columns": ["group", "n", "missing", "mean", "sd", "min", "p25", "median", "p75", "max"],
               "rows": rows, "total": len(rows), "note": f"{name} at {c.level} level; histogram bins are [low, high)" + (", log-spaced (1-2-5)" if log else ""),
               "chart": {"kind": "grouped", "labels": labels, "measure": f"{c.level}s", "series": series}}


class CountBy(BaseModel):
    rows: str = Field(description="utterance column for the rows")
    cols: str = Field("", description="optional second column → cross-tab")
    measure: Literal["utterances", "speakers", "conversations"] = Field("utterances", description="count rows or distinct ids")
    normalize: bool = Field(False, description="cross-tab cells as row % (utterances only); total stays a count")


@stage(CountBy)
def count_by(f: Frames, c: CountBy):
    """Counts per value of one column (bar), or a cross-tab of two (table + grouped chart). Missing = '(missing)'."""
    df = f.utts
    unit = {"utterances": None, "speakers": "speaker", "conversations": "conversation_id"}[c.measure]
    count = lambda gb: gb.size() if unit is None else gb[unit].nunique()
    r = need(df, c.rows)
    total = count(df.groupby(r, dropna=False))
    if not c.cols:
        return f, {"kind": "bar", "labels": [lab(k) for k in total.index], "values": [int(v) for v in total], "measure": c.measure}
    if c.normalize and unit:
        raise ValueError("normalize (row %) only applies to measure: utterances")
    tab = count(df.groupby([r, need(df, c.cols)], dropna=False)).unstack(fill_value=0)
    if c.normalize:
        tab = (tab.div(total.values, axis=0) * 100).round(1)
    names = [lab(k) for k in tab.columns]
    out = [[lab(k), *(v.item() for v in vals), int(n)] for k, vals, n in zip(tab.index, tab.values, total.values)]
    view = {"kind": "table", "columns": [c.rows, *names, "total"], "rows": out, "total": len(out),
            "note": f"{c.measure} per {c.rows} × {c.cols}" + (" (row %)" if c.normalize else "")}
    if len(names) <= 8:  # the UI has 8 series colours; wider tables stay table-only
        view["chart"] = grouped(out[:20], names, "%" if c.normalize else c.measure)
    return f, view
