"""Daily-ping data: validation flags, ping episodes, the participant × day grid.

In daily-ping data a chatbot messages each participant once or more per day (a "ping") and the participant may reply.
The data is one conversation per participant; each stage reads its field names from form fields with generic defaults.

Episodes live on the ping utterances: ping_episodes writes episode metrics as columns on each ping row (other rows
get NaN), so `filter episode == true` gives the episode table and rollup / describe / compare_groups work unchanged.
"""
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ckviz.frames import Frames
from ckviz.stages import *  # noqa: F403  helpers + registry
from ckviz.stages import stage
from ckviz.stages.describe import lab

NA = "n/a (field absent or empty)"


class Common(BaseModel):
    ping_flag: str = Field("meta.is_ping", description="utterance column that is true on the bot's daily ping")
    day_column: str = Field("meta.day_number", description="day number of each utterance")
    time_column: str = Field("timestamp", description="epoch seconds")
    status_column: str = Field("meta.status", description="delivery status of each utterance; absent = every participant message counts")
    reply_statuses: list[str] = Field([], description="status values of participant messages that count; empty = any status")
    owner_column: str = Field("meta.owner_id", description="conversation column naming the participant; absent = the speaker")
    user_type_column: str = Field("meta.type", description="speaker column; the value 'bot' marks the chatbot")
    bot_prefix: str = Field("", description="speaker ids starting with this are the bot's too, and it is removed to name the participant when no owner_column; empty = no prefix")


def is_bot(f: Frames, c: Common) -> pd.Series:
    sp = f.utts["speaker"].astype(str)
    typ = sp.map(f.speakers[c.user_type_column]) if c.user_type_column in f.speakers else pd.Series(None, index=sp.index)
    return typ.eq("bot") | (sp.str.startswith(c.bot_prefix) if c.bot_prefix else False)


def participant(f: Frames, c: Common) -> pd.Series:
    """The participant each utterance belongs to: the conversation owner (or the speaker, minus any bot_prefix)."""
    own = f.utts["speaker"].astype(str).str.removeprefix(c.bot_prefix)
    if c.owner_column in f.convos:
        return f.utts["conversation_id"].map(f.convos[c.owner_column]).fillna(own)
    return own


def col(f: Frames, name: str, numeric=False) -> pd.Series:
    s = f.utts[name] if name in f.utts else pd.Series(np.nan, index=f.utts.index)
    return pd.to_numeric(s, errors="coerce") if numeric else s


def texts(f: Frames) -> pd.Series:
    return f.utts["text"].fillna("").astype(str).str.strip()


def messages(f: Frames, c: Common) -> pd.Series:
    """Participant messages that count: not the bot, non-empty, status in reply_statuses (any status if that list or the field is absent or empty)."""
    status = col(f, c.status_column)
    ok = status.isin(c.reply_statuses) if c.reply_statuses and status.notna().any() else True
    return ~is_bot(f, c) & texts(f).ne("") & ok


def pings(f: Frames, c: Common) -> tuple[pd.Series, pd.Series]:
    """(episode pings, repeat pings): the first ping per participant × day starts an episode; later same-day ones don't."""
    p = need(f.utts, c.ping_flag).eq(True)
    d = pd.DataFrame({"t": col(f, c.time_column, True), "who": participant(f, c), "day": col(f, c.day_column, True)})[p]
    d = d.sort_values("t", kind="stable")
    repeat = f.utts.index.isin(d.index[d.duplicated(["who", "day"]) & d["day"].notna()])
    return p & ~repeat, pd.Series(repeat, index=f.utts.index)


# ping_validate

class Validate(Common):
    allowlist: list[str] = Field([], description="participant speaker ids; empty = no check")
    day_min: int = Field(1, description="first day")
    day_max: int = Field(description="last day")
    local_time_column: str = Field("meta.local_time", description="participant's local datetime, ISO 8601")
    prompt_id_column: str = Field("meta.prompt_id", description="id of the bot's system prompt, to spot config changes")
    pending_statuses: list[str] = Field([], description="status values to count as still pending; empty = no check")


@stage(Validate)
def ping_validate(f: Frames, c: Validate):
    """Data checks for daily-ping data, where a bot messages each participant every day and they may reply:
    one flag_* column per check plus a table of counts. Flags only; nothing is deleted."""
    u, who, bot = f.utts, participant(f, c), is_bot(f, c)
    has = lambda name: name in u and u[name].notna().any()
    ping, repeat = pings(f, c)
    day, t, text, msg = col(f, c.day_column, True), col(f, c.time_column, True), texts(f), messages(f, c)
    false = pd.Series(False, index=u.index)
    flags, rows = {}, []

    def check(name, flag, n, unit, ok=True):
        flags[flag] = flag_values.get(flag, false) if ok else false
        rows.append([name, n if ok else NA, unit])

    pinged = set(zip(who[ping], day[ping]))
    first = t[ping].groupby(who[ping]).min()
    status = col(f, c.status_column)
    local = col(f, c.local_time_column)
    sp_id = col(f, c.prompt_id_column)
    changed = sp_id[bot].groupby(who[bot]).nunique() > 1
    flag_values = {
        "flag_not_allowlisted": ~who.isin(c.allowlist),
        "flag_repeat_ping": repeat,
        "flag_day_out_of_range": (~bot | ping | repeat) & ~day.between(c.day_min, c.day_max),  # bot replies carry no day
        "flag_reply_without_ping": msg & ~(pd.Series(list(zip(who, day)), index=u.index).isin(pinged) & (t >= who.map(first))),
        "flag_empty_text": text.eq(""),
        "flag_trivial_text": text.ne("") & (text.str.len().le(1) | ~text.str.contains(r"\w")),
        "flag_pending_status": status.isin(c.pending_statuses),
        "flag_no_tz_offset": local.notna() & ~local.astype(str).str.contains(r"(?:Z|[+-]\d\d:?\d\d)$"),
        "flag_config_change": who.map(changed).eq(True),
    }
    n = lambda flag: int(flag_values[flag].sum())
    check("speakers not on allowlist", "flag_not_allowlisted", n("flag_not_allowlisted"), "rows", ok=bool(c.allowlist))
    check("same-day repeat pings", "flag_repeat_ping", n("flag_repeat_ping"), "pings")
    check("day out of range", "flag_day_out_of_range", n("flag_day_out_of_range"), "pings and participant messages", ok=has(c.day_column))
    check("replies without a ping", "flag_reply_without_ping", n("flag_reply_without_ping"), "participant messages", ok=has(c.day_column))
    check("empty text", "flag_empty_text", n("flag_empty_text"), "rows")
    check("text of 1 character or no word characters", "flag_trivial_text", n("flag_trivial_text"), "rows")
    check("pending status", "flag_pending_status", n("flag_pending_status"), "rows", ok=bool(c.pending_statuses) and has(c.status_column))
    check("timezone offset missing", "flag_no_tz_offset", n("flag_no_tz_offset"), "rows", ok=has(c.local_time_column))
    check("participants with more than one prompt id", "flag_config_change", int(changed.sum()), "participants", ok=sp_id[bot].notna().any())
    everyone = set(who[ping]) | set(who[~bot])
    rows.append(["never replied", len(everyone - set(who[msg])), "participants"])
    spread = (first.max() - first.min()) / 86_400 if len(first) else 0
    rows.append(["first ping spread", int(np.floor(spread)) if pd.notna(spread) else NA, "days between first pings"])
    utts = u.assign(**{k: v.astype(bool) for k, v in flags.items()})
    return Frames(utts, f.speakers, f.convos, f.meta), {"kind": "table", "columns": ["check", "n", "unit"],
                                                       "rows": rows, "total": len(rows)}


# ping_episodes

class PingEpisodes(Common):
    reply_no_word_statuses: list[str] = Field([], description="status values whose messages add zero to n_words; reply_statuses alone decides if they count as replies")
    max_hours: float = Field(24, gt=0, description="an episode ends at the next episode's ping (a later day's first ping) or after this many hours; later same-day pings don't end it")


EPISODE_COLS = ["participant", "episode_day", "replied", "latency_first_s", "n_msgs", "n_words", "turns", "depth"]


@stage(PingEpisodes)
def ping_episodes(f: Frames, c: PingEpisodes):
    """Turns daily-ping data (a bot messages each participant every day; they may reply) into one episode per
    participant-day: the day's first ping and what follows, until a later day's first ping or max_hours. Later same-day
    pings are merged in, so latency runs from the first ping. Writes metrics on each episode's ping: replied,
    latency_first_s, n_msgs, n_words (zero for reply_no_word_statuses), turns (participant turns: runs of messages between bot
    messages), depth (n_msgs >= 2), episode_day, participant."""
    u = f.utts
    t = col(f, c.time_column, True)
    ep, repeat = pings(f, c)
    msg = messages(f, c)
    starts = pd.DataFrame({"conv": u["conversation_id"].astype(str), "start": t})[ep].dropna().sort_values("start")
    nxt = starts.groupby("conv")["start"].shift(-1)
    starts["end"] = np.minimum(nxt.fillna(np.inf), starts["start"] + c.max_hours * 3600)
    starts["ping"] = starts.index
    words = texts(f).str.split().str.len().where(~col(f, c.status_column).isin(c.reply_no_word_statuses), 0)
    rest = pd.DataFrame({"conv": u["conversation_id"].astype(str), "t": t, "msg": msg, "bot": is_bot(f, c), "words": words})
    rest = rest[~ep & t.notna() & (msg | rest["bot"])].sort_values("t")
    m = pd.merge_asof(rest.rename_axis("id").reset_index(), starts, left_on="t", right_on="start", by="conv", direction="backward")
    m = m[m["t"] < m["end"]]
    turn = m["msg"] & ~m.groupby("ping")["msg"].shift(fill_value=False).astype(bool)
    um = m[m["msg"]].groupby("ping")
    idx = starts.index
    n_msgs = um.size().reindex(idx, fill_value=0)
    out = pd.DataFrame({
        "participant": participant(f, c)[idx],
        "episode_day": col(f, c.day_column, True)[idx],
        "replied": n_msgs > 0,
        "latency_first_s": (um["t"].min() - starts["start"]).reindex(idx),
        "n_msgs": n_msgs,
        "n_words": um["words"].sum().reindex(idx, fill_value=0),
        "turns": turn.groupby(m["ping"]).sum().reindex(idx, fill_value=0),
        "depth": n_msgs >= 2,
    }, index=idx)
    utts = u.drop(columns=[k for k in [*EPISODE_COLS, "episode"] if k in u]).join(out.astype(object))
    utts["episode"] = utts.index.isin(idx)
    loose = int((msg & ~u.index.isin(m.loc[m["msg"], "id"])).sum())
    rate = f"{out['replied'].mean():.0%}" if len(out) else "n/a"
    view = table(out)
    view["note"] = (f"{len(out)} episodes (first ping per participant-day; {int(repeat.sum())} later same-day pings merged), reply rate {rate}; "
                    f"{loose} participant messages not in any episode")
    return Frames(utts, f.speakers, f.convos, f.meta), view


# engagement_grid

class EngagementGrid(BaseModel):
    day_max: int = Field(ge=1, description="last day; grid columns are 1..day_max")
    segment: str = Field("", description="optional utterance column on the pings (e.g. speaker.group) to group rows by")
    ongoing: bool = Field(False, description="censor participants whose last ping is before day_max")


@stage(EngagementGrid)
def engagement_grid(f: Frames, c: EngagementGrid):
    """Daily-ping engagement, after ping_episodes. Participant × day grid (2 replied, 1 pinged no reply, 0 not pinged), daily share replied with Wilson 95% CIs,
    Kaplan–Meier retention of last_active_day. Writes meta.pings/active_days/reply_rate/last_active_day on speakers."""
    if "episode" not in f.utts:
        raise ValueError("no episode column; run ping_episodes first")
    e = f.utts[f.utts["episode"].eq(True)]
    day = pd.to_numeric(e["episode_day"], errors="coerce")
    e, day = e[day.between(1, c.day_max)], day[day.between(1, c.day_max)].astype(int)
    if e.empty:
        raise ValueError(f"no ping episodes within days 1..{c.day_max}; was a filter run after ping_episodes that dropped the pings?")
    rep, who = e["replied"].eq(True), e["participant"].astype(str)
    days = list(range(1, c.day_max + 1))
    grid = (pd.DataFrame({"p": who, "d": day, "v": np.where(rep, 2, 1)})
            .pivot_table(index="p", columns="d", values="v", aggfunc="max")
            .reindex(columns=days).fillna(0).astype(int))
    per = pd.DataFrame({"pings": who.value_counts(), "active_days": rep.groupby(who).sum(),
                        "last_active_day": day.where(rep).groupby(who).max(), "last_ping_day": day.groupby(who).max()})
    per["reply_rate"] = per["active_days"] / per["pings"]
    seg = need(e, c.segment).groupby(who).first().map(lab) if c.segment else None
    order = (per.sort_index().assign(seg=seg if seg is not None else "", neg=-per["active_days"])
             .sort_values(["seg", "neg"], kind="stable").index)
    grid = grid.reindex(order)

    n, k = (grid > 0).sum(), (grid == 2).sum()
    lo, hi = wilson(k, n)
    share = (k / n.where(n > 0)).tolist()
    daily = [[d, int(n[d]), int(k[d]), share[i], lo[i], hi[i]] for i, d in enumerate(days)]

    T = per["last_active_day"].fillna(0)
    cens = (T >= c.day_max) | (c.ongoing & (per["last_ping_day"] < c.day_max))
    s, km = 1.0, []
    for d in [0, *days]:
        risk, ev = int((T >= d).sum()), int(((T == d) & ~cens).sum())
        s *= 1 - ev / risk if risk else 1
        km.append(s)

    speakers = f.speakers.assign(**{f"meta.{k}": per[k] for k in ["pings", "active_days", "reply_rate", "last_active_day"]})
    heat = {"kind": "heatmap", "rows": list(grid.index), "cols": days, "cells": grid.values.tolist(),
            "legend": {"2": "replied", "1": "pinged, no reply", "0": "not pinged"}}
    if seg is not None:
        heat["row_groups"] = [seg[p] for p in grid.index]
    view = {"kind": "table", "columns": ["day", "pinged", "replied", "share", "lo", "hi"], "rows": daily, "total": len(daily),
            "note": f"{len(per)} participants; median active_days {per['active_days'].median():g} of {c.day_max}; "
                    f"Wilson 95% CIs; retention = Kaplan–Meier of last_active_day"
                    + (" (censored if last ping < day_max)" if c.ongoing else ""),
            "chart": heat,
            "extra_charts": [
                {"kind": "line", "x": days, "series": [{"name": "replied / pinged", "y": share, "lo": list(lo), "hi": list(hi)}],
                 "x_label": "day", "y_label": "share replied", "y_min": 0, "y_max": 1},
                {"kind": "line", "x": [0, *days], "series": [{"name": "still active", "y": km}], "x_label": "day",
                 "y_label": "share still active (Kaplan–Meier)", "y_min": 0, "y_max": 1, "step": True}]}
    return Frames(f.utts, speakers, f.convos, f.meta), view
