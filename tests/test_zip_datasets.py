"""Datasets from ConvoKit zip exports: local file or HTTP, sha256-checked, safely extracted, snapshotted."""
import hashlib
import http.server
import io
import json
import os
import threading
import time
import zipfile

import pandas as pd
import pytest

from ckviz import datasets as ds
from ckviz.pipeline import run
from ping_fixture import export_zip


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "CONFIG", tmp_path / "datasets.yaml")
    monkeypatch.setattr(ds, "DATA", tmp_path / "data")
    monkeypatch.setattr(ds, "SNAPSHOTS", tmp_path / "data" / "snapshots")
    ds._load_dir.cache_clear()
    return tmp_path


def serve(body: bytes, sha: str | None = None, token: str = "sekret-token-123", sha_header: str = "X-Export-Sha256"):
    """A local HTTP server mimicking an export endpoint; records requests."""
    seen = []

    class H(http.server.BaseHTTPRequestHandler):
        def _reply(self):
            seen.append({"method": self.command, "path": self.path, "auth": self.headers.get("Authorization")})
            if self.headers.get("Authorization") not in (f"Bearer {token}", "Basic dXNlcjpwYXNz"):
                self.send_response(401), self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            if sha:
                self.send_header(sha_header, sha)
            self.send_header("X-Other", "not-recorded")
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = _reply

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}", seen


def test_local_zip_loads_with_corpus_meta(home):
    (home / "corpus.zip").write_bytes(export_zip(v2=True))
    (home / "datasets.yaml").write_text("pings: {zip: corpus.zip}\n")
    key, f = ds.load("pings")
    assert (len(f.utts), len(f.speakers), len(f.convos)) == (12, 4, 2)
    assert f.meta["export_version"] == 2
    assert f.meta["extra"] == {"label": "demo"}
    assert key.startswith("pings@")


def test_participants_file_joins_on_conversation_owner(home):
    (home / "corpus.zip").write_bytes(export_zip(v2=True))
    (home / "people.csv").write_text("participant_id,group\nu-aaa,a\nu-zzz,b\n")
    (home / "datasets.yaml").write_text("pings: {zip: corpus.zip, participants: people.csv, participants_on: meta.owner_id}\n")
    key, f = ds.load("pings")
    assert len(f.utts) == 12 and len(f.convos) == 2  # no rows gained: u-zzz owns no conversation
    assert f.convos.loc["a1", "meta.group"] == "a" and f.convos["meta.group"].notna().sum() == 1  # u-bbb has no record
    steps = run([{"join_meta": {"source": "conversation", "keys": ["group"]}}], f, key)
    assert steps[0]["status"] == "ok"
    (home / "people.csv").write_text("participant_id,group\nu-aaa,b\n")
    assert ds.load("pings")[0] != key  # a corrected file must not reuse cached results


@pytest.mark.parametrize("config, csv, needle", [
    ("participants: people.csv", "participant_id,group\nu-aaa,x\n", "participants_on"),
    ("participants: people.csv, participants_on: meta.nope", "participant_id,group\nu-aaa,x\n", "meta.nope"),
    ("participants: people.csv, participants_on: meta.owner_id", "who,group\nu-aaa,x\n", "participant_id"),
])
def test_participants_config_errors_are_clear(home, config, csv, needle):
    (home / "corpus.zip").write_bytes(export_zip(v2=True))
    (home / "people.csv").write_text(csv)
    (home / "datasets.yaml").write_text(f"pings: {{zip: corpus.zip, {config}}}\n")
    with pytest.raises(ValueError, match=needle):
        ds.load("pings")


def test_participants_join_on_any_conversation_column(home):
    (home / "corpus.zip").write_bytes(export_zip(v2=True))
    (home / "people.csv").write_text("participant_id,group\nc-aaa,x\n")
    (home / "datasets.yaml").write_text("pings: {zip: corpus.zip, participants: people.csv, participants_on: meta.site}\n")
    key, f = ds.load("pings")
    assert f.convos.loc["a1", "meta.group"] == "x"
    (home / "datasets.yaml").write_text("pings: {zip: corpus.zip, participants: people.csv, participants_on: meta.owner_id}\n")
    assert ds.load("pings")[0] != key  # another join column gives other results


def test_participants_file_rejects_duplicate_ids(home):
    (home / "corpus.zip").write_bytes(export_zip(v2=True))
    (home / "people.csv").write_text("participant_id,group\nu-aaa,x\nu-aaa,y\n")
    (home / "datasets.yaml").write_text("pings: {zip: corpus.zip, participants: people.csv, participants_on: meta.owner_id}\n")
    with pytest.raises(ValueError, match="duplicate"):
        ds.load("pings")


def api_client():
    from fastapi.testclient import TestClient
    from ckviz.api import app

    return TestClient(app, base_url="http://localhost", headers={"X-CKVIZ": "1"})


def run_to_end(client, dataset: str) -> dict:
    run_id = client.post("/api/runs", json={"dataset": dataset, "stages": []}).json()["id"]
    for _ in range(200):
        job = client.get(f"/api/runs/{run_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise TimeoutError(run_id)


def test_bad_participants_config_is_not_reported_as_no_dataset(home):
    (home / "corpus.zip").write_bytes(export_zip(v2=True))
    (home / "people.csv").write_text("wrong_column,group\nu-aaa,x\n")
    (home / "datasets.yaml").write_text("pings: {zip: corpus.zip, participants: people.csv, participants_on: meta.owner_id}\n")
    job = run_to_end(api_client(), "pings")
    assert job["status"] == "error" and "no dataset" not in job["error"]


@pytest.mark.parametrize("config", ["{dir: missing/corpus}", "{zip: missing/corpus.zip}", "{dir: /nowhere/missing/corpus}"])
def test_missing_data_is_a_clear_400_without_absolute_paths(home, config):
    (home / "datasets.yaml").write_text(f"gone: {config}\n")
    with pytest.raises(ValueError, match="not found; see datasets.yaml"):
        ds.load("gone")
    r = api_client().post("/api/runs", json={"dataset": "gone", "stages": []})
    assert r.status_code == 400 and "not found; see datasets.yaml" in r.json()["detail"]
    assert str(home) not in r.text and "/nowhere" not in r.text


def test_missing_participants_file_is_a_clear_400(home):
    (home / "corpus.zip").write_bytes(export_zip())
    (home / "datasets.yaml").write_text("x: {zip: corpus.zip, participants: no.csv, participants_on: meta.y}\n")
    r = api_client().post("/api/runs", json={"dataset": "x", "stages": []})
    assert r.status_code == 400 and "no.csv not found" in r.json()["detail"]


def test_failed_refresh_is_a_400_naming_the_cause(home):
    srv, url, _ = serve(export_zip(), token="right")
    (home / "datasets.yaml").write_text(f'x: {{zip_url: "{url}/x", bearer: wrong}}\n')
    r = api_client().post("/api/datasets/x/snapshot")
    assert r.status_code == 400 and "export download failed: HTTP 401" in r.json()["detail"]
    srv.shutdown()


def test_zip_without_a_corpus_error_has_no_absolute_path(home):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("readme.txt", "x")
    (home / "empty.zip").write_bytes(buf.getvalue())
    (home / "datasets.yaml").write_text("x: {zip: empty.zip}\n")
    job = run_to_end(api_client(), "x")
    assert job["status"] == "error" and "no single ConvoKit corpus" in job["error"] and str(home) not in job["error"]


def test_partial_directories_are_not_snapshots(home):
    (ds.SNAPSHOTS / "x" / "20260101T000000Z").mkdir(parents=True)
    (ds.SNAPSHOTS / "x" / "20260102T000000Z.partial").mkdir()
    assert [p.name for p in ds.snapshots("x")] == ["20260101T000000Z"]


def corpus_on_disk(home):
    """A plain corpus dir, unpacked from the export fixture."""
    with zipfile.ZipFile(io.BytesIO(export_zip(v2=False, name="corpus"))) as z:
        z.extractall(home)
    (home / "datasets.yaml").write_text("x: {dir: corpus}\n")
    return home / "corpus" / "utterances.jsonl"


def test_rewriting_a_corpus_in_place_gives_a_new_key_and_new_data(home):
    path = corpus_on_disk(home)
    key, f = ds.load("x")
    assert f.utts.loc["a2", "text"] != "rewritten"
    path.write_text(path.read_text().replace(f.utts.loc["a2", "text"], "rewritten"))
    os.utime(path, (2_000_000_000, 2_000_000_000))
    key2, f2 = ds.load("x")
    assert key2 != key and f2.utts.loc["a2", "text"] == "rewritten"


def test_frame_cache_is_private_and_written_atomically(home):
    corpus_on_disk(home)
    ds.load("x")
    frames = ds.DATA / "frames"
    (cached,) = frames.iterdir()  # no temp file left behind
    assert cached.suffix == ".pkl"
    assert frames.stat().st_mode & 0o777 == 0o700
    assert cached.stat().st_mode & 0o077 == 0


def test_frame_cache_is_trusted_only_if_private(home, monkeypatch):
    corpus_on_disk(home)
    _, real = ds.load("x")
    (cached,) = (ds.DATA / "frames").iterdir()
    pd.to_pickle(real._replace(utts=real.utts.iloc[:1]), cached)  # a planted pickle
    ds._load_dir.cache_clear()
    assert len(ds.load("x")[1].utts) == 1  # private: read back as is
    cached.chmod(0o666)
    ds._load_dir.cache_clear()
    assert len(ds.load("x")[1].utts) == len(real.utts)  # world-writable: reparsed
    assert cached.stat().st_mode & 0o077 == 0  # and replaced with a private one
    pd.to_pickle(real._replace(utts=real.utts.iloc[:1]), cached)
    monkeypatch.setattr(os, "getuid", lambda: os.geteuid() + 1)  # owned by someone else
    ds._load_dir.cache_clear()
    assert len(ds.load("x")[1].utts) == len(real.utts)


def test_v1_zip_has_no_extra_file_but_loads(home):
    (home / "corpus.zip").write_bytes(export_zip(v2=False))
    (home / "datasets.yaml").write_text("pings: {zip: corpus.zip}\n")
    _, f = ds.load("pings")
    assert len(f.utts) == 12 and "extra" not in f.meta


def test_zip_url_downloads_verifies_and_snapshots(home):
    body = export_zip(v2=True)
    srv, url, seen = serve(body, sha=hashlib.sha256(body).hexdigest())
    (home / "datasets.yaml").write_text(f'pings: {{zip_url: "{url}/x", bearer: sekret-token-123, params: {{a: "1"}}}}\n')
    _, f = ds.load("pings")
    assert len(f.utts) == 12
    assert seen[0]["method"] == "GET" and seen[0]["path"] == "/x?a=1"
    (snap,) = ds.snapshots("pings")
    prov = json.loads((snap / "export.json").read_text())
    assert prov["sha256"] == hashlib.sha256(body).hexdigest() and "sekret-token-123" not in json.dumps(prov)
    ds.load("pings")
    assert len(seen) == 1  # later loads reuse the snapshot; "Refresh data" takes a new one
    srv.shutdown()


def test_redirect_is_refused_and_credentials_stay_home(home):
    srv2, url2, seen2 = serve(export_zip())

    class Redirect(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", url2 + "/elsewhere")
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Redirect)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    (home / "datasets.yaml").write_text(f'x: {{zip_url: "http://127.0.0.1:{srv.server_port}/x", bearer: sekret-token-123}}\n')
    with pytest.raises(ValueError, match="export download failed: HTTP 302"):
        ds.load("x")
    assert seen2 == []
    srv.shutdown(), srv2.shutdown()


def test_lowercase_sha_header_with_a_wrong_hash_fails(home):
    srv, url, _ = serve(export_zip(), sha="0" * 64, sha_header="x-export-sha256")
    (home / "datasets.yaml").write_text(f'x: {{zip_url: "{url}/x", bearer: sekret-token-123, sha256_header: X-Export-Sha256}}\n')
    with pytest.raises(ValueError, match="sha256"):
        ds.load("x")
    srv.shutdown()


def test_zip_url_post_with_basic_auth(home):
    srv, url, seen = serve(export_zip(v2=False))
    (home / "datasets.yaml").write_text(f'pings: {{zip_url: "{url}/x", method: POST, basic_auth: "user:pass"}}\n')
    ds.load("pings")
    assert seen[0]["method"] == "POST"
    srv.shutdown()


def test_sha_mismatch_fails_and_leaves_no_snapshot(home):
    srv, url, _ = serve(export_zip(v2=True), sha="0" * 64)
    (home / "datasets.yaml").write_text(f'pings: {{zip_url: "{url}/x", bearer: sekret-token-123, sha256_header: X-Export-Sha256}}\n')
    with pytest.raises(ValueError, match="sha256"):
        ds.load("pings")
    assert ds.snapshots("pings") == []
    srv.shutdown()


def test_opted_in_sha_header_must_be_present(home):
    srv, url, _ = serve(export_zip())
    (home / "datasets.yaml").write_text(f'pings: {{zip_url: "{url}/x", bearer: sekret-token-123, sha256_header: X-Export-Sha256}}\n')
    with pytest.raises(ValueError, match="sha256"):
        ds.load("pings")
    srv.shutdown()


def test_sha_header_is_opt_in_and_the_only_header_recorded(home):
    body = export_zip()
    srv, url, _ = serve(body, sha="0" * 64)
    (home / "datasets.yaml").write_text(f'plain: {{zip_url: "{url}/x", bearer: sekret-token-123}}\n')
    ds.load("plain")  # header sent but not opted in: not checked, not recorded
    (snap,) = ds.snapshots("plain")
    assert json.loads((snap / "export.json").read_text())["headers"] == {}
    srv.shutdown()
    sha = hashlib.sha256(body).hexdigest()
    srv, url, _ = serve(body, sha=sha)
    (home / "datasets.yaml").write_text(f'opted: {{zip_url: "{url}/x", bearer: sekret-token-123, sha256_header: X-Export-Sha256}}\n')
    ds.load("opted")
    (snap,) = ds.snapshots("opted")
    assert json.loads((snap / "export.json").read_text())["headers"] == {"X-Export-Sha256": sha}
    srv.shutdown()


def test_bad_token_is_a_clear_error(home):
    srv, url, _ = serve(export_zip(), token="right")
    (home / "datasets.yaml").write_text(f'pings: {{zip_url: "{url}/x", bearer: wrong}}\n')
    with pytest.raises(ValueError, match="401"):
        ds.load("pings")
    srv.shutdown()


def test_zip_slip_is_rejected(home):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../../evil.txt", "x")
    (home / "bad.zip").write_bytes(buf.getvalue())
    (home / "datasets.yaml").write_text("bad: {zip: bad.zip}\n")
    with pytest.raises(ValueError, match="unsafe"):
        ds.load("bad")
    assert not (home.parent / "evil.txt").exists()


@pytest.mark.parametrize("member", ["c/x-bin.p", "x-overall-bin.P", "vectors.v.p"])
def test_zip_with_pickle_member_is_refused(home, member):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(member, b"pickle")
    (home / "bad.zip").write_bytes(buf.getvalue())
    (home / "datasets.yaml").write_text("bad: {zip: bad.zip}\n")
    with pytest.raises(ValueError, match="unsafe"):
        ds.load("bad")
    assert ds.snapshots("bad") == []


def test_corpus_meta_survives_every_stage(home):
    (home / "corpus.zip").write_bytes(export_zip(v2=True))
    (home / "datasets.yaml").write_text("pings: {zip: corpus.zip}\n")
    key, f = ds.load("pings")
    from ckviz.stages import STAGES  # noqa: F401  (registry loaded)
    import ckviz.pipeline as pl

    seen = []
    pl._cache.clear()
    orig = pl.apply
    pl.apply = lambda name, fr, cfg: (seen.append(fr.meta.get("export_version")), orig(name, fr, cfg))[1]
    try:
        run([{"filter": {"column": "meta.status", "op": "==", "value": "received"}},
             {"aggregate": {"by": "speaker"}}], f, key)
    finally:
        pl.apply = orig
    assert seen == [2, 2]


def test_refresh_endpoint_takes_a_new_snapshot_for_zip_url(home, monkeypatch):
    from fastapi.testclient import TestClient
    from ckviz.api import app

    body = export_zip(v2=True)
    srv, url, seen = serve(body)
    (home / "datasets.yaml").write_text(f'pings: {{zip_url: "{url}/x", bearer: sekret-token-123}}\n')
    client = TestClient(app, base_url="http://localhost", headers={"X-CKVIZ": "1"})
    assert client.get("/api/datasets").json()["pings"]["kind"] == "zip_url"
    r = client.post("/api/datasets/pings/snapshot")
    assert r.status_code == 200 and len(seen) == 1
    assert client.get("/api/datasets").json()["pings"]["snapshots"] == [r.json()["snapshot"]]
    srv.shutdown()


def test_convokit_source_reads_this_data_dir_not_where_convokit_last_recorded(home, monkeypatch):
    """ConvoKit's download() returns the last path in its global registry, even for another clone's deleted folder."""
    here = home / "data" / "downloads" / "corpus-x"
    here.mkdir(parents=True)
    (here / "utterances.jsonl").write_text("")
    monkeypatch.setattr(ds, "download", lambda n, data_dir: pytest.fail("a local copy must not touch the network"))
    monkeypatch.setattr(ds, "_load_dir", lambda path, end, stamp=None: path)
    (home / "datasets.yaml").write_text("x: {convokit: corpus-x}\n")
    assert ds.load("x")[1] == here


def test_convokit_source_downloads_into_this_data_dir_when_absent(home, monkeypatch):
    here = home / "data" / "downloads" / "corpus-x"
    calls = []

    def fake(n, data_dir):
        calls.append((n, data_dir))
        here.mkdir(parents=True)
        (here / "utterances.jsonl").write_text("")
        return str(home / "elsewhere" / n)

    monkeypatch.setattr(ds, "download", fake)
    monkeypatch.setattr(ds, "_load_dir", lambda path, end, stamp=None: path)
    (home / "datasets.yaml").write_text("x: {convokit: corpus-x}\n")
    assert ds.load("x")[1] == here
    assert calls == [("corpus-x", str(home / "data" / "downloads"))]


def test_convokit_source_whose_folder_is_gone_is_a_clear_error(home, monkeypatch):
    (home / "data" / "downloads").mkdir(parents=True)
    monkeypatch.setattr(ds, "download", lambda n, data_dir: str(home / "gone" / n))
    (home / "datasets.yaml").write_text("x: {convokit: corpus-x}\n")
    with pytest.raises(ValueError, match="folder is gone") as e:
        ds.load("x")
    assert str(home) not in str(e.value)
