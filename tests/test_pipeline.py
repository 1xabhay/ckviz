from ckviz.pipeline import parse, run
from helpers import tiny

YAML = """
name: demo
stages:
  - join_meta: {source: speaker, keys: [phq9]}
  - filter: {column: meta.role, op: "==", value: person}
  - aggregate: {by: speaker.phq9, measure: utterances}
"""


def test_parse_reads_single_key_stage_list():
    p = parse(YAML)
    assert p["name"] == "demo"
    assert [next(iter(s)) for s in p["stages"]] == ["join_meta", "filter", "aggregate"]


def test_run_reports_each_stage():
    steps = run(parse(YAML)["stages"], tiny(), key="tiny@1")
    assert [s["status"] for s in steps] == ["ok"] * 3
    assert steps[1]["counts"] == {"utterances": 2, "speakers": 2, "conversations": 2}
    assert "speaker.phq9" in steps[0]["columns"]


def test_error_stops_pipeline_and_skips_rest():
    stages = [{"filter": {"column": "nope", "op": "==", "value": "x"}}, {"aggregate": {"by": "meta.role"}}]
    steps = run(stages, tiny(), key="tiny@err")
    assert [s["status"] for s in steps] == ["error", "skipped"]
    assert "nope" in steps[0]["error"]


def test_unchanged_prefix_is_cached():
    stages = parse(YAML)["stages"]
    run(stages, tiny(), key="tiny@cache")
    stages[2] = {"aggregate": {"by": "meta.role", "measure": "utterances"}}
    steps = run(stages, tiny(), key="tiny@cache")
    assert [s["cached"] for s in steps] == [True, True, False]
