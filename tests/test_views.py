"""Group-comparison stages carry a grouped-bar `chart` beside their table."""
from ckviz.stages import apply
from helpers import tiny
from test_coordination import corpus


def test_compare_groups_chart_mirrors_table_capped_at_20():
    t = tiny()
    t.utts["meta.feats"] = [{f"f{k}": (i + k) % 3 for k in range(25)} for i in range(5)]
    _, view = apply("compare_groups", t, {"column": "meta.feats", "group_by": "meta.role", "class1": "ai", "class2": "person"})
    ch = view["chart"]
    assert ch["kind"] == "grouped" and len(view["rows"]) == 25
    assert ch["labels"] == [r[0] for r in view["rows"][:20]]
    assert [s["name"] for s in ch["series"]] == ["ai", "person"]
    assert ch["series"][0]["values"] == [r[1] for r in view["rows"][:20]]
    assert ch["series"][1]["values"] == [r[2] for r in view["rows"][:20]]


def test_coordination_chart_one_series_per_group():
    _, view = apply("coordination", corpus(), {"group_by": "meta.admin", "side": "target", "target_thresh": 1})
    ch = view["chart"]
    assert ch["kind"] == "grouped" and ch["labels"] == [r[0] for r in view["rows"]]
    assert [s["name"] for s in ch["series"]] == ["False", "True"]
    assert ch["series"][1]["values"] == [r[2] for r in view["rows"]]
