"""Engagement stages (ping_validate, ping_episodes, engagement_grid, map_values) on the invented fixture and simulator.

Fixture by hand (only day-1 pings exist: a1, b1 at T0; max_hours 24 -> episodes end at T0 + 86400):
  A: a2 at +30 s (6 words) is in the episode; a4 (+86410 s, day 2) and a6 (day 3) are replies without a ping.
  B: b2 at +120 s has status "moderated" (3 words, zero if reply_no_word_statuses lists it); b4 (day 3) is a reply without a ping.
"""
import io
import math
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from ckviz.frames import load_dir
from ckviz.pipeline import json_safe, parse, run
from ckviz import stages
from ckviz.stages import STAGES, apply, engagement
from ping_fixture import T0, export_zip
from ping_sim import simulate

EXAMPLE = Path(__file__).parent.parent / "examples/daily-ping-engagement.yaml"


def load(tmp_path, v2):
    zipfile.ZipFile(io.BytesIO(export_zip(v2=v2))).extractall(tmp_path)
    return load_dir(tmp_path / "ping_corpus_test")


@pytest.fixture(params=[True, False], ids=["v2", "v1"])
def corpus(request, tmp_path):
    return load(tmp_path, request.param)


@pytest.fixture
def v2(tmp_path):
    return load(tmp_path, True)


EPISODES = {"reply_statuses": ["received", "moderated"], "reply_no_word_statuses": ["moderated"], "bot_prefix": "bot:"}
V = {"day_max": 3}  # the fixture has days 1-3
PENDING = {**V, "pending_statuses": ["queued"]}


def checks(view):
    return {r[0]: dict(zip(view["columns"], r)) for r in view["rows"]}


# ping_validate

def test_validate_flags_v2(v2):
    f, view = apply("ping_validate", v2, {**V, "allowlist": ["u-aaa"]})
    u, c = f.utts, checks(view)
    assert len(u) == 12  # flags, never deletes
    assert set(u.index[u["flag_not_allowlisted"]]) == {"b1", "b2", "b3", "b4", "b5"}
    assert set(u.index[u["flag_reply_without_ping"]]) == {"a4", "a6", "b4"}
    assert set(u.index[u["flag_empty_text"]]) == {"b5"}
    assert u["flag_no_tz_offset"].all()  # fixture local times are naive
    assert set(u.index[u["flag_config_change"]]) == {"a1", "a2", "a3", "a4", "a5", "a6", "a7"}  # A saw sp1 then sp2
    assert not u[["flag_repeat_ping", "flag_day_out_of_range", "flag_trivial_text", "flag_pending_status"]].any().any()
    assert c["speakers not on allowlist"]["n"] == 5 and c["replies without a ping"]["n"] == 3
    assert c["participants with more than one prompt id"]["n"] == 1 and c["participants with more than one prompt id"]["unit"] == "participants"
    assert c["never replied"]["n"] == 0 and c["first ping spread"]["n"] == 0
    assert view["columns"] == ["check", "n", "unit"]
    assert all(len(r) == 3 for r in view["rows"])


def test_validate_v1_absent_fields(tmp_path):
    f, view = apply("ping_validate", load(tmp_path, False), V)
    c = checks(view)
    assert c["participants with more than one prompt id"]["n"] == "n/a (field absent or empty)"
    assert c["speakers not on allowlist"]["n"].startswith("n/a")  # empty allowlist = no check
    assert not f.utts["flag_config_change"].any() and not f.utts["flag_not_allowlisted"].any()


@pytest.mark.parametrize("column, name, flag", [
    ("meta.prompt_id", "participants with more than one prompt id", "flag_config_change"),
    ("meta.status", "pending status", "flag_pending_status"),
    ("meta.local_time", "timezone offset missing", "flag_no_tz_offset"),
    ("meta.day_number", "day out of range", "flag_day_out_of_range"),
    ("meta.day_number", "replies without a ping", "flag_reply_without_ping"),
])
def test_validate_all_null_column_is_na_not_zero(v2, column, name, flag):
    f, view = apply("ping_validate", v2._replace(utts=v2.utts.assign(**{column: None})), PENDING)
    assert checks(view)[name]["n"] == "n/a (field absent or empty)"
    assert not f.utts[flag].any()


def test_validate_prompt_id_only_on_participant_rows_is_na(v2):
    bots = v2.utts["speaker"].str.startswith("bot:")
    u = v2.utts.assign(**{"meta.prompt_id": v2.utts["meta.prompt_id"].where(~bots)})
    u.loc["a2", "meta.prompt_id"] = "sp1"
    f, view = apply("ping_validate", v2._replace(utts=u), V)
    assert checks(view)["participants with more than one prompt id"]["n"].startswith("n/a") and not f.utts["flag_config_change"].any()


def test_all_null_status_counts_every_participant_message(v2):
    all_null = v2._replace(utts=v2.utts.assign(**{"meta.status": None}))
    _, view = apply("ping_validate", all_null, V)
    assert checks(view)["never replied"]["n"] == 0  # not 2: a null status isn't a rejected one
    f, _ = apply("ping_episodes", all_null, {})
    assert f.utts.loc["a1", "n_msgs"] == 1 and f.utts.loc["b1", "replied"] == True  # noqa: E712


def test_validate_repeats_range_trivial_queued(v2):
    u = v2.utts.copy()
    u.loc["a1x"] = u.loc["a1"]
    u.loc["a1x", "timestamp"] = T0 + 20
    u.loc["a4", "meta.day_number"] = 15
    u.loc["a6", "text"] = "👍"
    u.loc["a2", "meta.status"] = "queued"
    u.loc["a7", "meta.local_time"] = "2024-03-03T09:00:55+02:00"
    f, view = apply("ping_validate", v2._replace(utts=u), PENDING)
    u = f.utts
    assert set(u.index[u["flag_repeat_ping"]]) == {"a1x"}  # the later of two day-1 pings
    assert set(u.index[u["flag_day_out_of_range"]]) == {"a4"}
    assert set(u.index[u["flag_trivial_text"]]) == {"a6"}
    assert set(u.index[u["flag_pending_status"]]) == {"a2"}
    assert checks(view)["pending status"]["n"] == 1
    assert not u.loc["a7", "flag_no_tz_offset"]
    assert checks(view)["same-day repeat pings"]["n"] == 1
    assert not any("duplicate" in r[0] for r in view["rows"])


def test_validate_day_check_skips_bot_replies(v2):
    u = v2.utts.copy()
    u.loc[["a3", "a5", "a7", "b3"], "meta.day_number"] = None  # bot replies carry no day
    u.loc["a4", "meta.day_number"] = None  # a participant message should have one
    u.loc["b1", "meta.day_number"] = None  # so should a ping
    u.loc["a6", "meta.day_number"] = 15
    f, view = apply("ping_validate", v2._replace(utts=u), V)
    assert set(f.utts.index[f.utts["flag_day_out_of_range"]]) == {"a4", "a6", "b1"}
    row = checks(view)["day out of range"]
    assert row["n"] == 3 and row["unit"] == "pings and participant messages"


def test_validate_never_replied_and_stagger(v2):
    u = v2.utts.copy()
    u = u[~u.index.isin(["b2", "b4"])]
    u.loc["b1", "timestamp"] = T0 + 2 * 86_400
    _, view = apply("ping_validate", v2._replace(utts=u), V)
    c = checks(view)
    assert c["never replied"]["n"] == 1 and c["first ping spread"]["n"] == 2


def test_validate_and_episodes_follow_configured_field_names(v2):
    names = {"meta.is_ping": "x.ping", "meta.status": "x.st", "meta.local_time": "x.lt", "meta.prompt_id": "x.pid", "meta.day_number": "x.day"}
    cfg = {**V, "ping_flag": "x.ping", "status_column": "x.st", "local_time_column": "x.lt", "prompt_id_column": "x.pid", "day_column": "x.day",
           "owner_column": "meta.who"}
    moved = v2._replace(utts=v2.utts.rename(columns=names), convos=v2.convos.rename(columns={"meta.owner_id": "meta.who"}))
    f, view = apply("ping_validate", moved, cfg)
    assert view["rows"] == apply("ping_validate", v2, V)[1]["rows"]
    assert set(f.utts.index[f.utts["flag_config_change"]]) == {"a1", "a2", "a3", "a4", "a5", "a6", "a7"}
    f, _ = apply("ping_episodes", moved, {**EPISODES, **{k: cfg[k] for k in ["ping_flag", "status_column", "day_column", "owner_column"]}})
    e = f.utts[f.utts["episode"]]
    assert e["participant"].tolist() == ["u-aaa", "u-bbb"] and e["n_words"].tolist() == [6, 0]


# ping_episodes

def test_episodes_by_hand(corpus):
    f, view = apply("ping_episodes", corpus, EPISODES)
    u = f.utts
    e = u[u["episode"]]
    assert list(e.index) == ["a1", "b1"]
    assert e["participant"].tolist() == ["u-aaa", "u-bbb"]
    assert e["replied"].tolist() == [True, True]  # B's moderated message counts: it is in reply_statuses
    assert e["latency_first_s"].tolist() == [30, 120]
    assert e["n_msgs"].tolist() == [1, 1] and e["n_words"].tolist() == [6, 0]  # moderated is in reply_no_word_statuses
    assert e["turns"].tolist() == [1, 1] and e["depth"].tolist() == [False, False]
    assert e["episode_day"].tolist() == [1, 1]
    assert u.loc[~u["episode"], "replied"].isna().all()
    assert "2 episodes (first ping per participant-day; 0 later same-day pings merged)" in view["note"] and "100%" in view["note"] and "3 participant messages" in view["note"]


def test_episodes_max_hours_and_next_ping(v2):
    f, _ = apply("ping_episodes", v2, {**EPISODES, "max_hours": 30})  # now a4 (day 2) falls in A's day-1 episode
    a = f.utts.loc["a1"]
    assert a["n_msgs"] == 2 and a["n_words"] == 9 and a["turns"] == 2 and a["depth"]
    u = v2.utts.copy()
    u.loc["a5", "meta.is_ping"] = True  # a ping right after a4: a6 belongs to it, a4 still to day 1
    u.loc["a5", "meta.day_number"] = 2
    f, _ = apply("ping_episodes", v2._replace(utts=u), {**EPISODES, "max_hours": 72})
    e = f.utts
    assert e.loc["a1", "n_msgs"] == 2 and e.loc["a5", "n_msgs"] == 1 and e.loc["a5", "latency_first_s"] == 86_400 + 36


def test_episodes_merge_later_same_day_pings(v2):
    u = v2.utts.copy()
    u.loc["a1x"] = u.loc["a1"]
    u.loc["a1x", "timestamp"] = T0 + 20
    f, view = apply("ping_episodes", v2._replace(utts=u), {})
    u = f.utts
    assert not u.loc["a1x", "episode"] and pd.isna(u.loc["a1x", "replied"])
    assert u.loc["a1", "n_msgs"] == 1 and u.loc["a1", "latency_first_s"] == 30  # not cut short by the later ping
    assert "1 later same-day pings merged" in view["note"] and "duplicate" not in view["note"]


def test_episodes_no_replies_and_filtered_rows(v2):
    users = v2.utts["speaker"].str.startswith("u-")
    f, view = apply("ping_episodes", v2._replace(utts=v2.utts[~users]), {})
    e = f.utts[f.utts["episode"]]
    assert e["replied"].tolist() == [False, False] and e["n_msgs"].tolist() == [0, 0] and e["latency_first_s"].isna().all()
    assert e["participant"].tolist() == ["u-aaa", "u-bbb"] and "0%" in view["note"]
    f, _ = apply("filter", v2, {"column": "meta.status", "op": "!=", "value": "moderated"})
    f, _ = apply("ping_episodes", f, {})
    assert f.utts.loc["b1", "replied"] == False  # noqa: E712
    f, _ = apply("ping_episodes", v2, {"reply_statuses": ["received"]})
    assert f.utts.loc["b1", "replied"] == False  # noqa: E712


def test_default_statuses_get_no_special_treatment(v2):
    f, _ = apply("ping_episodes", v2, {})
    assert f.utts.loc["b1", "replied"] == True and f.utts.loc["b1", "n_words"] == 3  # noqa: E712
    u = v2.utts.copy()
    u.loc["a2", "meta.status"] = "anything"
    f, view = apply("ping_validate", v2._replace(utts=u.assign(**{"meta.status": u["meta.status"].replace("received", "queued")})), V)
    assert not f.utts["flag_pending_status"].any() and checks(view)["pending status"]["n"].startswith("n/a")
    assert checks(view)["never replied"]["n"] == 0  # a queued or unknown status still counts as a message
    f, _ = apply("ping_episodes", v2._replace(utts=u), {})
    assert f.utts.loc["a1", "n_msgs"] == 1 and f.utts.loc["a1", "n_words"] == 6


def test_status_lists_are_configurable(v2):
    f, _ = apply("ping_episodes", v2, {"reply_no_word_statuses": ["received"]})
    assert f.utts.loc["a1", "n_words"] == 0 and f.utts.loc["b1", "n_words"] == 3
    f, view = apply("ping_validate", v2, {**V, "pending_statuses": ["sent", "failed"]})
    assert f.utts["flag_pending_status"].sum() == 7 and checks(view)["pending status"]["n"] == 7


def without_speaker_types(f):
    return f._replace(speakers=f.speakers.drop(columns="meta.type"))


def test_bot_prefix_is_empty_by_default_and_marks_bots_when_set(v2):
    bare = without_speaker_types(v2)
    bots = {i for i in v2.utts.index if v2.utts.loc[i, "speaker"].startswith("bot:")}
    assert set(v2.utts.index[engagement.is_bot(v2, engagement.Common())]) == bots  # type column alone
    assert not engagement.is_bot(bare, engagement.Common()).any()  # an empty prefix must not match every id
    assert set(v2.utts.index[engagement.is_bot(bare, engagement.Common(bot_prefix="bot:"))]) == bots
    f, _ = apply("ping_episodes", bare, {"bot_prefix": "bot:"})
    assert f.utts.loc["a1", "n_msgs"] == 1  # bot rows are not counted as replies


def test_bot_prefix_is_stripped_from_the_participant_without_an_owner_column(v2):
    no_owner = v2._replace(convos=v2.convos.drop(columns="meta.owner_id"))
    f, _ = apply("ping_episodes", no_owner, {"bot_prefix": "bot:"})
    assert f.utts[f.utts["episode"]]["participant"].tolist() == ["u-aaa", "u-bbb"]
    f, _ = apply("ping_episodes", no_owner, {})
    assert f.utts[f.utts["episode"]]["participant"].tolist() == ["bot:u-aaa", "bot:u-bbb"]  # nothing to strip by default
    f, _ = apply("ping_episodes", v2, {})
    assert f.utts[f.utts["episode"]]["participant"].tolist() == ["u-aaa", "u-bbb"]  # the owner column needs no prefix


# engagement_grid

def test_grid_fixture(v2):
    f, _ = apply("ping_episodes", v2, {})
    f, view = apply("engagement_grid", f, {"day_max": 3})
    h = view["chart"]
    assert h["kind"] == "heatmap" and h["cols"] == [1, 2, 3] and h["rows"] == ["u-aaa", "u-bbb"]
    assert h["cells"] == [[2, 0, 0], [2, 0, 0]] and h["legend"] == {"2": "replied", "1": "pinged, no reply", "0": "not pinged"}
    view = json_safe(view)
    assert view["rows"][0][:4] == [1, 2, 2, 1.0] and view["rows"][1][1:4] == [0, 0, None]
    sp = f.speakers
    assert sp.loc["u-aaa", "meta.active_days"] == 1 and sp.loc["u-aaa", "meta.reply_rate"] == 1
    assert sp.loc["u-bbb", "meta.last_active_day"] == 1
    line, km = view["extra_charts"]
    assert line["kind"] == "line" and line["x"] == [1, 2, 3] and line["series"][0]["y"][0] == 1
    assert km["step"] and km["x"] == [0, 1, 2, 3] and km["series"][0]["y"] == [1, 0, 0, 0]  # both last active day 1
    _, view = apply("engagement_grid", f, {"day_max": 3, "ongoing": True})
    assert view["extra_charts"][1]["series"][0]["y"] == [1, 1, 1, 1]  # last ping day 1 < 3: censored


@pytest.mark.parametrize("stage", ["ping_validate", "engagement_grid"])
def test_day_max_is_required(v2, stage):
    with pytest.raises(ValueError, match="day_max"):
        apply(stage, v2, {})


def test_grid_needs_episodes(v2):
    with pytest.raises(ValueError, match="episode"):
        apply("engagement_grid", v2, V)


def wilson(k, n, z=1.959963984540054):
    p = k / n
    mid, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (mid - half) / (1 + z * z / n), (mid + half) / (1 + z * z / n)


def test_wilson_value_pinned():
    assert stages.wilson(3, 10) == pytest.approx((0.1078, 0.6032), abs=5e-5)  # hand-pinned before the two copies merged
    assert stages.engagement.wilson is stages.stats.wilson is stages.wilson


def test_grid_simulated():
    profiles = [(1.0, 10), (1.0, 5), (0.0, 10), (0.5, 10), (0.7, 9), (0.3, 10)]
    f, truth = simulate(profiles, seed=1)
    f, _ = apply("ping_episodes", f, {})
    f, view = apply("engagement_grid", f, {"day_max": 10, "segment": "speaker"})
    sp = f.speakers
    for pid, ds in truth.items():
        assert sp.loc[pid, "meta.active_days"] == len(ds)
        assert sp.loc[pid, "meta.reply_rate"] == pytest.approx(len(ds) / 10)
        assert (pd.isna(sp.loc[pid, "meta.last_active_day"]) if not ds else sp.loc[pid, "meta.last_active_day"] == max(ds))
    assert sp.loc["p00", "meta.active_days"] == 10 and sp.loc["p01", "meta.last_active_day"] == 5
    assert "meta.active_days" not in sp or pd.isna(sp.loc["bot:p00", "meta.active_days"])
    h = view["chart"]
    i = h["rows"].index("p00")
    assert h["cells"][i] == [2] * 10 and h["cells"][h["rows"].index("p02")] == [1] * 10
    assert len(h["row_groups"]) == len(h["rows"])
    # per-day share and Wilson CI
    for r in view["rows"]:
        day, n, k, share, lo, hi = r[:6]
        assert n == 6 and k == sum(day in ds for ds in truth.values()) and share == pytest.approx(k / n)
        assert (lo, hi) == pytest.approx(wilson(k, n))
    sm = pytest.importorskip("statsmodels.stats.proportion")
    r = view["rows"][0]
    assert (r[4], r[5]) == pytest.approx(sm.proportion_confint(r[2], r[1], method="wilson"))
    # KM: p02 never replies (drops at 0); p00 is censored at 10; others drop at their last active day
    km = view["extra_charts"][1]["series"][0]["y"]
    last = [max(ds) if ds else 0 for ds in truth.values()]
    s, expect = 1.0, []
    for d in range(11):
        s *= 1 - sum(t == d < 10 for t in last) / sum(t >= d for t in last)
        expect.append(s)
    assert km == pytest.approx(expect)


def test_grid_rows_sorted_by_active_days():
    f, _ = simulate([(0.2, 10), (1.0, 10), (0.6, 10)], seed=3)
    f, _ = apply("ping_episodes", f, {})
    _, view = apply("engagement_grid", f, {"day_max": 10})
    act = [sum(c == 2 for c in row) for row in view["chart"]["cells"]]
    assert act == sorted(act, reverse=True) and view["chart"]["rows"][0] == "p01"


# map_values

def test_map_values():
    f, _ = simulate([(1.0, 10)])
    assert STAGES["map_values"].fn.__module__ == "ckviz.stages.shape"
    f, view = apply("map_values", f, {"column": "meta.day_number", "mapping": ["1=x", "2=y"], "into": "label"})
    u = f.utts
    assert u.loc["c00-d1", "label"] == "x" and u.loc["c00-d2", "label"] == "y"
    assert u.loc[u["meta.day_number"] > 2, "label"].isna().all()
    assert f"{(u['meta.day_number'] > 2).sum()} " in view["note"]


# the example pipeline

def test_engagement_example_runs_on_both_versions(corpus):
    p = parse(EXAMPLE.read_text())
    steps = run(p["stages"], corpus, f"eng-{len(corpus.utts.columns)}")
    assert all(s["status"] == "ok" for s in steps), [(s["stage"], s.get("error")) for s in steps]
    views = {next(iter(st)): s["view"] for st, s in zip(p["stages"], steps)}
    assert views["engagement_grid"]["chart"]["kind"] == "heatmap"
    assert views["count_by"]["rows"][0][0] == "1"


def test_grid_without_episodes_is_a_clear_error(v2):
    f, _ = apply("ping_episodes", v2, {})
    f, _ = apply("filter", f, {"column": "meta.status", "op": "==", "value": "received"})  # drops every ping
    with pytest.raises(ValueError, match="no ping episodes"):
        apply("engagement_grid", f, V)
