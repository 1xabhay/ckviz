"""NaN-safe results, literal 'contains', host check, CSRF header, localhost binding."""
import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ckviz.api import app
from ckviz.pipeline import run
from ckviz.stages import apply
from helpers import tiny

H = {"X-CKVIZ": "1"}
ROOT = Path(__file__).parents[1]
repo_only = pytest.mark.skipif(not (ROOT / "Dockerfile").exists(), reason="repo files are not copied into the test image")


def test_nan_in_a_view_becomes_null_and_stays_json_safe():
    stages = [{"compare_groups": {"level": "speaker", "column": "meta.phq9", "group_by": "meta.role",
                                  "class1": "ai", "class2": "person"}}]
    (step,) = run(stages, tiny(), key="tiny@nan")
    assert step["status"] == "ok"
    json.dumps(step, allow_nan=False)  # raises on NaN / inf
    assert None in step["view"]["rows"][0]


def test_contains_is_literal_not_regex():
    t = tiny()
    t.utts.loc["a1", "text"] = "a.b (x"
    f, _ = apply("filter", t, {"column": "text", "op": "contains", "value": "a.b (x"})
    assert list(f.utts.index) == ["a1"]
    f, _ = apply("filter", t, {"column": "text", "op": "contains", "value": "^(\\w+\\s?)*$!"})
    assert f.utts.empty


def test_mutating_routes_require_csrf_header():
    client = TestClient(app, base_url="http://localhost")
    body = {"dataset": "synthetic", "stages": []}
    assert client.post("/api/runs", json=body).status_code == 403
    assert client.put("/api/pipelines/x", json={"stages": []}).status_code == 403
    assert client.post("/api/datasets/synthetic/snapshot").status_code == 403
    assert client.get("/api/stages").status_code == 200  # reads stay open


def test_unknown_host_header_is_rejected():
    assert TestClient(app, base_url="http://attacker.example").get("/api/stages", headers=H).status_code == 400
    assert TestClient(app, base_url="http://localhost").get("/api/stages").status_code == 200


def test_server_binds_localhost_by_default(monkeypatch):
    import uvicorn
    import ckviz

    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw))
    monkeypatch.delenv("CKVIZ_HOST", raising=False)
    ckviz.main()
    assert seen["host"] == "127.0.0.1"


def test_in_operator_matches_any_listed_value_literally():
    f, _ = apply("filter", tiny(), {"column": "speaker", "op": "in", "value": "u1, u2"})
    assert sorted(f.utts.index) == ["a1", "b1"]
    f, _ = apply("filter", tiny(), {"column": "speaker", "op": "in", "value": "u.*"})
    assert f.utts.empty


@repo_only
def test_image_does_not_bind_all_interfaces_and_drops_root():
    docker = (ROOT / "Dockerfile").read_text()
    assert "CKVIZ_HOST" not in docker
    assert docker.rstrip().count("USER ckviz") == 2  # base and dev stages
    assert "CKVIZ_HOST: 0.0.0.0" in (ROOT / "compose.yaml").read_text()


@pytest.mark.skipif(not (ROOT / ".git").exists(), reason="needs the git checkout")
def test_gitignore_keeps_private_files_out_but_not_fixtures():
    ignored = lambda p: subprocess.run(["git", "check-ignore", "-q", "--no-index", p], cwd=ROOT).returncode == 0
    for p in ("pipelines/mine.yaml", ".env", ".env.local", "nb.ipynb", ".DS_Store", ".ruff_cache/x", ".pytest_cache/x",
              "people.csv", "export.zip"):
        assert ignored(p), p
    for p in ("examples/x.csv", "examples/x.zip", "tests/x.csv", "tests/x.zip", "examples/awry-2018.yaml"):
        assert not ignored(p), p
    assert "!" not in (ROOT / ".gitignore").read_text()
    docker = (ROOT / ".dockerignore").read_text().split()
    assert {".git", ".env*", "data/"} <= set(docker)
