import time

from fastapi.testclient import TestClient

from ckviz.api import app

client = TestClient(app, base_url="http://localhost", headers={"X-CKVIZ": "1"})


def finished(run_id: str) -> dict:
    for _ in range(100):
        job = client.get(f"/api/runs/{run_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise TimeoutError(run_id)


def test_stages_endpoint_lists_form_schemas():
    r = client.get("/api/stages").json()
    assert r["filter"]["properties"]["op"]["enum"]


def test_datasets_endpoint_lists_examples():
    assert "synthetic" in client.get("/api/datasets").json()


def test_run_pipeline_on_dataset():
    body = {"dataset": "synthetic", "stages": [{"aggregate": {"by": "meta.role", "measure": "utterances"}}]}
    r = finished(client.post("/api/runs", json=body).json()["id"])
    assert "meta.role" in r["fields"]["utterance"]
    assert "phq9" in r["fields"]["speaker"]
    steps = r["steps"]
    assert steps[0]["status"] == "ok"
    assert sum(steps[0]["view"]["values"]) == 1543


def test_pipeline_files_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("ckviz.api.PIPELINES", tmp_path / "pipelines")
    p = {"name": "x", "stages": [{"filter": {"column": "meta.role", "op": "==", "value": "person"}}]}
    assert client.put("/api/pipelines/x", json=p).status_code == 200
    assert "x" in client.get("/api/pipelines").json()
    assert client.get("/api/pipelines/x").json() == p
    for bad in ("..%2Fevil", "a.b", "%2E%2E"):
        assert client.put(f"/api/pipelines/{bad}", json=p).status_code != 200
    assert sorted(f.name for f in tmp_path.rglob("*.yaml")) == ["x.yaml"]


def test_unknown_dataset_is_404():
    r = client.post("/api/runs", json={"dataset": "nope", "stages": []})
    assert r.status_code == 404 and "nope" in r.json()["detail"]
    assert client.get("/api/runs/missing").status_code == 404


def test_blocking_run_route_is_gone():
    assert client.post("/api/run", json={"dataset": "synthetic", "stages": []}).status_code in (404, 405)


def test_examples_are_listed_and_loadable():
    assert "intro-fighting-words" in client.get("/api/pipelines").json()
    p = client.get("/api/pipelines/intro-fighting-words").json()
    assert p["dataset"] == "reddit-corpus-small" and p["reproduces"].endswith(".ipynb")


def test_background_run_streams_steps():
    body = {"dataset": "synthetic", "stages": [{"aggregate": {"by": "meta.role"}}, {"first_n": {"n": 1}}]}
    job = finished(client.post("/api/runs", json=body).json()["id"])
    assert job["status"] == "done" and job["total"] == 2
    assert [s["status"] for s in job["steps"]] == ["ok", "ok"]
    assert "meta.role" in job["fields"]["utterance"]


def test_malformed_run_and_pipeline_bodies_are_client_errors(tmp_path, monkeypatch):
    monkeypatch.setattr("ckviz.api.PIPELINES", tmp_path)
    for stages in ([{}], [{"a": {}, "b": {}}], [5], 5):
        assert client.post("/api/runs", json={"dataset": "synthetic", "stages": stages}).status_code == 422
    for body in ({"stages": 5}, {"stages": None}, {"stages": [{}]}):
        assert client.put("/api/pipelines/x", json=body).status_code == 400
    assert not list(tmp_path.iterdir())


def test_pipelines_directory_is_created_on_demand(tmp_path, monkeypatch):
    monkeypatch.setattr("ckviz.api.PIPELINES", tmp_path / "absent")
    assert client.get("/api/pipelines").status_code == 200
    assert client.put("/api/pipelines/x", json={"stages": []}).status_code == 200
    assert (tmp_path / "absent" / "x.yaml").exists()


def test_stages_endpoint_includes_descriptions():
    r = client.get("/api/stages").json()
    assert "ConvoKit TextCleaner" in r["text_clean"]["x-description"]
    assert r["filter"]["x-description"] == ""  # no docstring -> empty, never missing
