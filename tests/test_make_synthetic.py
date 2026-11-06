"""examples/data/make_synthetic.py is deterministic, reproduces the committed corpus, and plants what it says it plants."""
import json
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from ckviz.frames import load_dir
from helpers import SYNTHETIC

SCRIPT = SYNTHETIC.parent / "make_synthetic.py"
FILES = ["utterances.jsonl", "speakers.json", "conversations.json", "index.json", "corpus.json"]


def generate(out: Path) -> dict[str, bytes]:
    subprocess.run([sys.executable, str(SCRIPT), str(out)], check=True)
    return {name: (out / name).read_bytes() for name in FILES}


def test_generator_is_deterministic(tmp_path):
    assert generate(tmp_path / "a") == generate(tmp_path / "b")


def test_committed_corpus_matches_generator(tmp_path):
    assert generate(tmp_path) == {name: (SYNTHETIC / name).read_bytes() for name in FILES}


def test_load_dir_reads_generated_corpus(tmp_path):
    generate(tmp_path)
    f = load_dir(tmp_path)
    assert (len(f.utts), len(f.speakers), len(f.convos)) == (1543, 20, 19)
    assert f.meta["synthetic"] is True


def test_score_covers_every_user_and_only_users():
    s = load_dir(SYNTHETIC).speakers
    people = s[s["meta.role"] == "person"]
    assert len(people) == 19 and people["meta.phq9"].notna().all()
    assert s.loc[s["meta.role"] == "ai", "meta.phq9"].isna().all()
    # i * 3 // 2 for i in 0..18: 19 distinct integers spanning 0..27.
    assert sorted(people["meta.phq9"].astype(int)) == [i * 3 // 2 for i in range(19)]


def test_corpus_stays_small_and_users_clear_the_dlatk_word_threshold():
    assert sum(p.stat().st_size for p in SYNTHETIC.iterdir()) < 350_000
    f = load_dir(SYNTHETIC)
    words = f.utts[f.utts["meta.role"] == "person"].groupby("speaker")["text"].apply(lambda t: t.str.split().str.len().sum())
    assert words.min() >= 500  # examples/dlatk-phq9.yaml sets occ_group_freq_thresh: 500


def test_assistant_never_uses_the_planted_words():
    utts = [json.loads(line) for line in (SYNTHETIC / "utterances.jsonl").read_text().splitlines()]
    bot = " ".join(u["text"].lower() for u in utts if u["meta"]["role"] == "ai")
    assert "compost" not in bot and "bicycle" not in bot


@pytest.mark.parametrize("word,sign", [("compost", 1), ("bicycle", -1)])
def test_planted_word_rate_correlates_with_score(word, sign):
    # P(clause per message) moves 0.40 across the score range, so with ~40 messages the signal sd is about 5 uses against
    # binomial noise of about 2.7, which predicts |r| near 5 / sqrt(5^2 + 2.7^2) = 0.88. 0.7 leaves room for sampling.
    f = load_dir(SYNTHETIC)
    people = f.utts[f.utts["meta.role"] == "person"]
    rate = people.groupby("speaker")["text"].apply(lambda t: t.str.lower().str.count(word).sum() / t.str.split().str.len().sum())
    score = f.speakers.loc[rate.index, "meta.phq9"].astype(float)
    assert sign * statistics.correlation(rate.tolist(), score.tolist()) > 0.7
