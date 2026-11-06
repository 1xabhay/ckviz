"""DLATK (WWBP's Differential Language Analysis ToolKit) as ckviz stages.

DLATK pins numpy<=1.25 / pandas<=1.5 (no Python 3.12 wheels), so it can't share ckviz's environment. Each stage
writes the frames into a temp SQLite db in DLATK's msgs/outcomes shape and runs `dlatkInterface.py` under
`uvx --python 3.11` (override the command with CKVIZ_DLATK, e.g. a preinstalled dlatkInterface.py).
"""
import csv
import io
import os
import shlex
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from ckviz.frames import Frames
from ckviz.stages import at, need, sig, stage, table

DLATK = "uvx --python 3.11 --from dlatk==1.3.15 dlatkInterface.py"
GROUP = {"speaker": "speaker", "conversation": "conversation_id"}


def write_db(f: Frames, level: str, outcome: str, db: Path) -> None:
    """msgs(message_id, unit_id, message) from utterances; outcomes(unit_id, outcome) from the level's frame."""
    y = pd.to_numeric(need(at(f, level), outcome)).dropna()
    msgs = pd.DataFrame({"message_id": f.utts.index.astype(str), "unit_id": f.utts[GROUP[level]].astype(str).values,
                         "message": f.utts["text"].fillna("").values})
    with sqlite3.connect(db) as con:
        msgs.to_sql("msgs", con, index=False)
        pd.DataFrame({"unit_id": y.index.astype(str), "outcome": y.astype(float).values}).to_sql("outcomes", con, index=False)


def parse_csv(text: str) -> pd.DataFrame:
    """The first (unsorted) block of DLATK's --csv output, sorted by p."""
    rows = list(csv.reader(io.StringIO(text)))
    start = next(i for i, r in enumerate(rows) if r[:1] == ["feature"]) + 1
    body = rows[start:next((i for i in range(start, len(rows)) if not rows[i]), len(rows))]
    df = pd.DataFrame([[r[0], float(r[1]), float(r[2]), int(r[3]), int(r[6])] for r in body],
                      columns=["feature", "r", "p", "n", "freq"])
    return df.sort_values("p", kind="stable").reset_index(drop=True)


class NgramsCorrelate(BaseModel):
    level: Literal["speaker", "conversation"] = "speaker"
    outcome: str = Field("meta.phq9", description="numeric column on the speaker/conversation frame")
    n: int = Field(1, ge=1, le=3, description="n-gram length (1 = unigrams)")
    p_occ: float = Field(0.05, ge=0, le=1, description="keep n-grams used by at least this share of groups")
    occ_group_freq_thresh: int = Field(500, ge=0, description="occurrence filter counts only groups with at least this many "
                                       "words (500 = DLATK's default for a group column named user_id)")
    group_freq_thresh: int = Field(50, ge=0, description="correlate only groups with at least this many words")
    correction: Literal["none", "BH", "bonferroni", "holm", "BY"] = Field("BH", description="p-value correction (DLATK default BH)")


@stage(NgramsCorrelate)
def dlatk_ngrams_correlate(f: Frames, c: NgramsCorrelate):
    """DLATK: extract n-grams per group, filter rare ones, Pearson-correlate relative frequency with the outcome."""
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "corpus"
        write_db(f, c.level, c.outcome, db.with_suffix(".db"))
        base = [*shlex.split(os.environ.get("CKVIZ_DLATK", DLATK)), "--db_engine", "sqlite", "-d", str(db),
                "-t", "msgs", "-c", "unit_id"]
        feat = f"feat${c.n}gram$msgs$unit_id"
        pocc = str(c.p_occ).replace(".", "_").replace("-", "n")  # DLATK's table-name label
        corr = ["--no_correction"] if c.correction == "none" else ["--p_correction", c.correction]
        for args in (["--add_ngrams", "-n", str(c.n)],
                     ["-f", feat, "--feat_occ_filter", "--set_p_occ", str(c.p_occ), "--group_freq_thresh", str(c.occ_group_freq_thresh)],
                     ["-f", f"{feat}${pocc}", "--outcome_table", "outcomes", "--outcomes", "outcome",
                      "--group_freq_thresh", str(c.group_freq_thresh), "--correlate", "--csv", *corr,
                      "--output_name", str(db)]):
            # ponytail: DLATK always makedirs ~/sqlite_data (unused, since -d is absolute); harmless, left alone.
            r = subprocess.run(base + args, capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(f"dlatk {args[0]} failed: {(r.stderr or r.stdout)[-800:]}")
        out = db.with_suffix(".csv")
        if not out.exists():
            raise RuntimeError(f"dlatk wrote no correlations (no groups left after group_freq_thresh?): {r.stdout[-400:]}")
        df = parse_csv(out.read_text())
    view = table(df.assign(r=df.r.round(4), p=df.p.map(sig)).set_index("feature"), len(df))
    note = "uncorrected" if c.correction == "none" else f"{c.correction}-corrected"
    return f, {**view, "note": f"DLATK {c.n}-grams × {c.outcome} at {c.level} level · Pearson r · p {note} · "
                                f"{int((df.p < 0.05).sum())} of {len(df)} features p < .05"}
