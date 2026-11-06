import os
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from ckviz.frames import Frames
from ckviz_dlatk import parse_csv, write_db

ROOT = Path(__file__).parents[3]


def tiny() -> Frames:
    utts = pd.DataFrame({
        "id": ["a0", "a1", "b0", "b1"],
        "speaker": ["bot", "u1", "bot", "u2"],
        "conversation_id": ["a", "a", "b", "b"],
        "text": ["hi", "i feel sad", "hey", "good good"],
    }).set_index("id")
    speakers = pd.DataFrame({"id": ["bot", "u1", "u2"], "meta.phq9": [None, 20, 3]}).set_index("id")
    convos = pd.DataFrame({"id": ["a", "b"], "meta.score": [1.5, 2.5]}).set_index("id")
    return Frames(utts, speakers, convos)


def test_write_db_speaker_level(tmp_path):
    db = tmp_path / "c.db"
    write_db(tiny(), "speaker", "meta.phq9", db)
    con = sqlite3.connect(db)
    assert con.execute("select * from msgs order by message_id").fetchall() == [
        ("a0", "bot", "hi"), ("a1", "u1", "i feel sad"), ("b0", "bot", "hey"), ("b1", "u2", "good good")]
    assert con.execute("select * from outcomes order by unit_id").fetchall() == [("u1", 20.0), ("u2", 3.0)]


def test_write_db_conversation_level(tmp_path):
    db = tmp_path / "c.db"
    write_db(tiny(), "conversation", "meta.score", db)
    con = sqlite3.connect(db)
    assert con.execute("select unit_id from msgs order by message_id").fetchall() == [("a",), ("a",), ("b",), ("b",)]
    assert con.execute("select * from outcomes order by unit_id").fetchall() == [("a", 1.5), ("b", 2.5)]


def test_write_db_rejects_missing_outcome(tmp_path):
    with pytest.raises(ValueError, match="no column"):
        write_db(tiny(), "speaker", "meta.nope", tmp_path / "c.db")


CSV = '''Namespace(corpdb='x', outcomefields=['outcome'])


feature,outcome,p,N,CI_l,CI_u,freq
",",-0.1,0.7,12,-0.6,0.4,300
it's,-0.5,0.03,12,-0.8,-0.1,7
kettle,0.6,0.002,12,0.3,0.8,40

SORTED:
rank,outcome,r,p,N,CI_l,CI_u,freq
1,kettle,0.6,0.002,12,0.3,0.8,40
'''


def test_parse_csv_reads_first_block_sorted_by_p():
    df = parse_csv(CSV)
    assert list(df.columns) == ["feature", "r", "p", "n", "freq"]
    assert df["feature"].tolist() == ["kettle", "it's", ","]
    assert df.iloc[0].tolist() == ["kettle", 0.6, 0.002, 12, 40]


@pytest.mark.skipif(not os.environ.get("CKVIZ_EXAMPLES"), reason="CKVIZ_EXAMPLES not set (runs DLATK via uvx)")
def test_dlatk_recovers_the_words_planted_in_the_corpus(monkeypatch):
    # examples/data/make_synthetic.py plants "compost" (rate rises with the score) and "bicycle" (rate falls with it).
    monkeypatch.chdir(ROOT)
    from ckviz import datasets as ds
    from ckviz.pipeline import parse, run

    p = parse((ROOT / "examples/dlatk-phq9.yaml").read_text())
    key, frames = ds.load(p["dataset"])
    steps = run(p["stages"], frames, key)
    assert all(s["status"] == "ok" for s in steps), [s.get("error") for s in steps]
    view = steps[-1]["view"]
    assert view["total"] == 294  # regression value from running DLATK 1.3.15 on this corpus, not derived by hand
    rows = {r[0]: dict(zip(view["columns"], r)) for r in view["rows"]}
    people = frames.utts[frames.utts["meta.role"] == "person"]["text"].str.lower()
    # tests/test_make_synthetic.py predicts |r| near 0.88 for both words; a corpus-level count pins freq independently.
    assert rows["compost"]["r"] > 0.7 and rows["bicycle"]["r"] < -0.7
    for word in ("compost", "bicycle"):
        assert rows[word]["p"] < 0.05 and rows[word]["n"] == 19
        assert rows[word]["freq"] == people.str.count(rf"\b{word}\b").sum()
