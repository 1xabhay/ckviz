from ckviz.frames import load_dir, to_corpus
from helpers import SYNTHETIC, tiny


def test_load_dir_reads_convokit_corpus():
    f = load_dir(SYNTHETIC)
    assert (len(f.utts), len(f.speakers), len(f.convos)) == (1543, 20, 19)
    assert "meta.phq9" in f.speakers.columns
    assert "vectors" not in f.utts.columns


def test_to_corpus_roundtrip_keeps_meta():
    c = to_corpus(tiny())
    assert c.get_speaker("u1").meta["phq9"] == 20
    assert c.get_utterance("a1").reply_to == "a0"


def test_convokit_dataset_downloads_and_truncates(monkeypatch, tmp_path):
    from ckviz import datasets as ds

    (tmp_path / "datasets.yaml").write_text("small: {convokit: synthetic-corpus, utterance_end_index: 9}\n")
    monkeypatch.setattr(ds, "CONFIG", tmp_path / "datasets.yaml")
    monkeypatch.setattr(ds, "download", lambda name, data_dir: str(SYNTHETIC))
    key, f = ds.load("small")
    assert len(f.utts) == 10 and "small@" in key


def test_unprefixed_speaker_meta_is_normalised():
    from convokit import Corpus, Speaker, Utterance
    from ckviz.frames import from_corpus

    c = Corpus(utterances=[Utterance(id="a", speaker=Speaker(id="s", meta={"gender": "F"}), conversation_id="a", text="hi")])
    real = c.get_speakers_dataframe
    c.get_speakers_dataframe = lambda: real().rename(columns={"meta.gender": "gender"})  # legacy corpus shape
    assert from_corpus(c).speakers.loc["s", "meta.gender"] == "F"


def test_to_corpus_does_not_mutate_frames():
    f = tiny()
    before = [list(df.columns) for df in f[:3]]
    to_corpus(f)
    assert [list(df.columns) for df in f[:3]] == before


def test_convokit_list_dataset_merges_and_tags_corpus(monkeypatch, tmp_path):
    from ckviz import datasets as ds

    to_corpus(tiny()).dump("tiny", base_path=str(tmp_path))
    paths = {"one": str(SYNTHETIC), "two": str(tmp_path / "tiny")}
    (tmp_path / "datasets.yaml").write_text("both: {convokit: [one, two]}\n")
    monkeypatch.setattr(ds, "CONFIG", tmp_path / "datasets.yaml")
    monkeypatch.setattr(ds, "DATA", tmp_path)
    monkeypatch.setattr(ds, "download", lambda name, data_dir: paths[name])
    _, f = ds.load("both")
    assert len(f.utts) == 1543 + 5
    assert f.utts["meta.corpus"].value_counts().to_dict() == {"one": 1543, "two": 5}
    assert f.speakers.index.is_unique and f.convos.index.is_unique
    assert f.utts.index[0].startswith("one:")


def test_loaded_frames_are_cached_on_disk(monkeypatch, tmp_path):
    from ckviz import datasets as ds

    (tmp_path / "datasets.yaml").write_text(f"syn: {{dir: {SYNTHETIC}}}\n")
    monkeypatch.setattr(ds, "CONFIG", tmp_path / "datasets.yaml")
    monkeypatch.setattr(ds, "DATA", tmp_path)
    ds._load_dir.cache_clear()
    _, first = ds.load("syn")
    assert list((tmp_path / "frames").glob("*.pkl"))
    ds._load_dir.cache_clear()
    monkeypatch.setattr(ds, "load_dir", lambda *a, **k: (_ for _ in ()).throw(AssertionError("not cached")))
    _, again = ds.load("syn")
    assert again.utts.equals(first.utts)


def test_merge_prefixes_ids_so_colliding_corpora_stay_consistent():
    from ckviz.datasets import merge

    f = merge({"a": tiny(), "b": tiny()})
    assert f.utts.index.is_unique and len(f.utts) == 10
    assert f.utts.loc["b:a1", ["reply_to", "conversation_id", "speaker"]].tolist() == ["b:a0", "b:a0", "b:u1"]
    assert f.utts.loc["a:a0", "reply_to"] is None
    assert len(f.speakers) == 6 and len(f.convos) == 4
    assert to_corpus(f).get_utterance("b:a1").reply_to == "b:a0"


def test_to_corpus_turns_missing_reply_to_into_none():
    import numpy as np

    f = tiny()
    f.utts["reply_to"] = f.utts["reply_to"].astype(object).where(f.utts["reply_to"].notna(), np.nan)
    c = to_corpus(f)
    assert c.get_utterance("a0").reply_to is None
    assert c.get_utterance("a1").reply_to == "a0"
