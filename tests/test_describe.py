"""Descriptive stages (time_bins, reply_latency, describe, count_by, word_count) on the invented fixture."""
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from ckviz.frames import load_dir
from ckviz.pipeline import json_safe, parse, run
from ckviz.stages import apply
from helpers import tiny
from ping_fixture import DAY, export_zip

EXAMPLE = Path(__file__).parent.parent / "examples/daily-ping-descriptive.yaml"


@pytest.fixture(params=[True, False], ids=["v2", "v1"])
def corpus(request, tmp_path):
    zipfile.ZipFile(io.BytesIO(export_zip(v2=request.param))).extractall(tmp_path)
    return load_dir(tmp_path / "ping_corpus_test")


@pytest.fixture
def v2(tmp_path):
    zipfile.ZipFile(io.BytesIO(export_zip(v2=True))).extractall(tmp_path)
    return load_dir(tmp_path / "ping_corpus_test")


def rows(view):
    return {r[0]: dict(zip(view["columns"], r)) for r in view["rows"]}


# time_bins

def test_day_relative_to_conversation(corpus):
    f, _ = apply("time_bins", corpus, {"unit": "day", "relative_to": "conversation", "into": "day"})
    assert f.utts["day"].to_dict() == {"a1": 1, "a2": 1, "a3": 1, "a4": 2, "a5": 2, "a6": 3, "a7": 3,
                                      "b1": 1, "b2": 1, "b3": 1, "b4": 3, "b5": 3}


def test_week_relative_and_calendar_units():
    t = tiny()
    t.utts["timestamp"] = [0, 6 * DAY, 7 * DAY, 3600 * 5, 13 * DAY + 1]  # 1970-01-01 is a Thursday
    f, _ = apply("time_bins", t, {"unit": "week", "relative_to": "conversation", "into": "w"})
    assert f.utts["w"].tolist() == [1, 1, 2, 1, 2]
    f, _ = apply("time_bins", t, {"unit": "hour", "into": "h"})
    assert f.utts["h"].tolist() == [0, 0, 0, 5, 0]
    f, _ = apply("time_bins", t, {"unit": "weekday", "into": "wd"})
    assert f.utts["wd"].tolist() == [4, 3, 4, 4, 3]
    f, _ = apply("time_bins", t, {"unit": "day", "into": "d"})
    assert f.utts.loc["a0", "d"] == "1970-01-01"
    f, _ = apply("time_bins", t, {"unit": "hour", "tz": "America/New_York", "into": "h"})
    assert f.utts.loc["a0", "h"] == 19


def test_time_bins_from_datetime_column():
    t = tiny()  # naive local times are taken as-is (tz UTC = no shift)
    t.utts["local"] = ["2024-03-01T23:59:00", "2024-03-02T00:01:00", "2024-03-09T08:00:00", None, "2024-03-05T12:00:00"]
    f, _ = apply("time_bins", t, {"column": "local", "unit": "day", "relative_to": "conversation", "into": "d"})
    assert f.utts["d"].tolist()[:3] == [1, 2, 9] and pd.isna(f.utts.loc["b0", "d"]) and f.utts.loc["b1", "d"] == 1


def test_time_bins_rejects_relative_hour():
    with pytest.raises(ValueError, match="relative_to"):
        apply("time_bins", tiny(), {"unit": "hour", "relative_to": "speaker", "into": "h"})


# reply_latency

def test_reply_latency_from_timestamps(corpus):
    f, view = apply("reply_latency", corpus, {"into": "lat"})
    lat = f.utts["lat"]
    assert pd.isna(lat["a1"]) and pd.isna(lat["b1"])
    assert lat[["a2", "a3", "a4", "a5", "a6", "a7"]].tolist() == [30, 5, DAY - 25, 4, DAY + 36, 5]
    assert lat[["b2", "b3", "b4", "b5"]].tolist() == [120, 5, 2 * DAY + 175, 2]


def test_reply_latency_prefers_sent_time_with_fallback(v2):
    f, view = apply("reply_latency", v2, {"into": "lat", "time_column": "meta.sent_at", "fallback_to_timestamp": True})
    lat = f.utts["lat"]
    # user replies measured from when the bot message went out (timestamp + 4 s); bot replies end at their send time
    assert lat[["a2", "a3", "a4", "a5", "a6", "a7"]].tolist() == [26, 9, DAY - 29, 8, DAY + 32, 9]
    assert lat[["b2", "b3", "b4", "b5"]].tolist() == [116, 9, 2 * DAY + 171, 2]  # b5 failed: no sent_at, falls back
    assert "timestamp" in view["note"]


def test_reply_latency_missing_time_column(corpus):
    has = "meta.sent_at" in corpus.utts
    cfg = {"into": "lat", "time_column": "meta.sent_at"}
    if has:
        f, _ = apply("reply_latency", corpus, cfg)
        assert pd.isna(f.utts.loc["b5", "lat"])  # no fallback: null sent time stays NaN
    else:
        with pytest.raises(ValueError, match="sent_at"):
            apply("reply_latency", corpus, cfg)
        f, view = apply("reply_latency", corpus, {**cfg, "fallback_to_timestamp": True})
        assert f.utts.loc["a2", "lat"] == 30 and "absent" in view["note"]


def test_reply_latency_removed_parent_and_self_reply():
    t = tiny()
    t.utts.loc["a2", "speaker"] = "u1"  # a2 now replies to its own speaker
    f, _ = apply("reply_latency", t, {"into": "lat"})
    assert pd.isna(f.utts.loc["a2", "lat"])
    f, _ = apply("reply_latency", t, {"into": "lat", "only_cross_speaker": False})
    assert f.utts.loc["a2", "lat"] == 1
    t = t._replace(utts=t.utts.drop("a0"))
    f, _ = apply("reply_latency", t, {"into": "lat"})
    assert pd.isna(f.utts.loc["a1", "lat"])


# word_count, describe

def test_describe_words_by_speaker_type(corpus):
    f, _ = apply("join_meta", corpus, {"source": "speaker", "keys": ["type"]})
    f, _ = apply("word_count", f, {"into": "words"})
    assert f.utts.loc["a2", "words"] == 6 and f.utts.loc["b5", "words"] == 0
    _, view = apply("describe", f, {"column": "words", "group_by": "speaker.type"})
    r = rows(view)
    assert r["user"]["n"] == 5 and r["user"]["mean"] == pytest.approx(4.4) and r["user"]["median"] == 5
    assert r["bot"]["n"] == 7 and r["bot"]["min"] == 0 and r["bot"]["max"] == 8
    c = view["chart"]
    assert [s["name"] for s in c["series"]] == ["bot", "user"] and len(c["labels"]) <= 20
    assert sum(c["series"][1]["values"]) == 5 and c["labels"][0] == "0"  # integer data: unit-width bins
    assert "p" not in view["columns"]


def test_describe_dict_key_and_level(corpus):
    info = [{"n": 20} if s.startswith("bot:") else None for s in corpus.utts["speaker"]]
    _, view = apply("describe", corpus._replace(utts=corpus.utts.assign(**{"meta.info": info})), {"column": "meta.info", "key": "n"})
    r = rows(view)["all"]
    assert (r["n"], r["missing"], r["mean"], r["sd"]) == (7, 5, 20, 0)
    f, _ = apply("rollup", corpus, {"column": "text", "to": "speaker", "agg": "count", "into": "n"})
    _, view = apply("describe", f, {"level": "speaker", "column": "meta.n"})
    assert rows(view)["all"]["mean"] == 3


def test_describe_empty_and_single_value():
    t = tiny()
    t.utts["x"] = [None] * 5
    view = json_safe(apply("describe", t, {"column": "x"})[1])
    assert rows(view)["all"]["n"] == 0 and rows(view)["all"]["mean"] is None and view["chart"]["labels"] == []
    t.utts["x"] = [3.0] * 5
    view = json_safe(apply("describe", t, {"column": "x", "group_by": "meta.role"})[1])
    assert rows(view)["ai"]["sd"] == 0 and view["chart"]["labels"] == ["3"]


def test_describe_rejects_text():
    with pytest.raises(ValueError, match="numeric"):
        apply("describe", tiny(), {"column": "text"})


# count_by, rollup nunique

def test_count_by_crosstab_and_row_percent(corpus):
    f, _ = apply("join_meta", corpus, {"source": "speaker", "keys": ["type"]})
    _, view = apply("count_by", f, {"rows": "meta.status", "cols": "speaker.type"})
    r = rows(view)
    assert r["sent"]["bot"] == 6 and r["failed"]["bot"] == 1 and r["moderated"]["user"] == 1
    assert r["received"]["user"] == 4 and r["received"]["total"] == 4
    _, view = apply("count_by", f, {"rows": "speaker.type", "cols": "meta.status", "normalize": True})
    assert rows(view)["bot"]["sent"] == pytest.approx(85.7) and rows(view)["bot"]["total"] == 7


def test_count_by_distinct_speakers_sorted_naturally():
    t = tiny()
    t.utts["day"] = [10, 2, 2, 10, 1]
    _, view = apply("count_by", t, {"rows": "day", "measure": "speakers"})
    assert view["kind"] == "bar" and view["labels"] == ["1", "2", "10"] and view["values"] == [1, 2, 1]
    t.utts.loc["a1", "day"] = None
    _, view = apply("count_by", t, {"rows": "day"})
    assert view["labels"][-1] == "(missing)"


def test_rollup_distinct(corpus):
    f, _ = apply("time_bins", corpus, {"relative_to": "conversation", "into": "day"})
    f, _ = apply("rollup", f, {"column": "day", "agg": "nunique", "into": "active_days"})
    assert f.speakers.loc[["u-aaa", "u-bbb"], "meta.active_days"].tolist() == [3, 2]


# the example pipeline, on both export versions

def test_example_runs_on_both_versions(corpus):
    p = parse(EXAMPLE.read_text())
    assert p["dataset"] == "daily-ping"
    steps = run(p["stages"], corpus, f"test-{len(corpus.utts.columns)}")
    assert all(s["status"] == "ok" for s in steps), [(s["stage"], s.get("error")) for s in steps]
    v = {}
    for st, s in zip(p["stages"], steps):
        v.setdefault(next(iter(st)), []).append(s["view"])
    status = rows(v["count_by"][0])
    assert status["moderated"]["total"] == 1 and status["failed"]["total"] == 1
    latency = rows(v["describe"][1])
    assert latency["bot"]["n"] == 4  # the failed reply is excluded
    assert latency["bot"]["mean"] == (8.75 if "meta.sent_at" in corpus.utts else 4.75)
    by_day = v["count_by"][-1]
    assert by_day["labels"] == ["1", "2", "3"] and by_day["values"] == [2, 1, 2]
    assert v["show_conversation"][0]["kind"] == "tree"


def test_describe_log_bins():
    t = tiny()
    t.utts["lat"] = [1.5, 7.0, 40.0, 300.0, 5000.0]
    _, view = apply("describe", t, {"column": "lat"})  # auto: max/min > 1000 and min > 0
    c = view["chart"]
    assert c["labels"][0] == "1–2" and c["labels"][-1] == "2000–5000" and len(c["labels"]) == 11
    assert c["series"][0]["values"] == [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1] and "log" in view["note"]
    _, view = apply("describe", t, {"column": "lat", "log_bins": "off"})
    assert len(view["chart"]["labels"]) <= 20 and view["chart"]["labels"][0] == "0–500"
    t.utts["lat"] = [3.0, 4.0, 4.5, 6.0, 10.0]
    _, view = apply("describe", t, {"column": "lat", "log_bins": "on"})
    assert view["chart"]["labels"] == ["2–5", "5–10"] and view["chart"]["series"][0]["values"] == [3, 2]
    t.utts["lat"] = [0.0, 4.0, 4.5, 6.0, 10.0]
    with pytest.raises(ValueError, match="> 0"):
        apply("describe", t, {"column": "lat", "log_bins": "on"})
