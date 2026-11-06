"""Needs a Postgres: `docker compose up -d db` then CKVIZ_TEST_DSN=postgresql://postgres:ckviz@localhost:54329/postgres."""
import os
import uuid

import psycopg
import pytest
from psycopg.conninfo import make_conninfo

from ckviz import datasets as ds
from ckviz.frames import load_dir

DSN = os.environ.get("CKVIZ_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="CKVIZ_TEST_DSN not set")


@pytest.fixture
def db():
    schema = f"t_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(f"""
            CREATE SCHEMA {schema}; SET search_path = {schema};
            CREATE TABLE ck_speakers (id text PRIMARY KEY, meta jsonb);
            CREATE TABLE ck_conversations (id text PRIMARY KEY, meta jsonb);
            CREATE TABLE ck_utterances (id text PRIMARY KEY, speaker text, conversation_id text,
                                        reply_to text, timestamp double precision, text text, meta jsonb);
            INSERT INTO ck_speakers VALUES ('bot', '{{"role": "ai"}}'), ('u1', '{{"role": "person", "phq9": 12}}');
            INSERT INTO ck_conversations VALUES ('c0', '{{"source": "a"}}');
            INSERT INTO ck_utterances VALUES
              ('c0', 'bot', 'c0', NULL, 1, 'hello', '{{"latency_ms": 5}}'),
              ('c1', 'u1', 'c0', 'c0', 2, 'hi there', NULL);
        """)
        yield make_conninfo(DSN, options=f"-c search_path={schema}")
        conn.execute(f"DROP SCHEMA {schema} CASCADE")


def test_dsn_dataset_snapshots_views_to_convokit_dir(db, tmp_path, monkeypatch):
    (tmp_path / "datasets.yaml").write_text(f'proj: {{dsn: "{db}"}}\n')
    monkeypatch.setattr(ds, "CONFIG", tmp_path / "datasets.yaml")
    monkeypatch.setattr(ds, "SNAPSHOTS", tmp_path / "snapshots")

    key, f = ds.load("proj")
    assert len(ds.snapshots("proj")) == 1 and key.startswith("proj@")
    assert f.speakers.loc["u1", "meta.phq9"] == 12
    assert f.utts.loc["c1", "reply_to"] == "c0"
    assert f.convos.loc["c0", "meta.source"] == "a"

    # snapshot is a plain ConvoKit corpus dir, reusable from Jupyter
    assert len(load_dir(ds.snapshots("proj")[0]).utts) == 2
