"""Linguistic style coordination (Danescu-Niculescu-Mizil et al. 2012, "Echoes of power"), via ConvoKit's Coordination."""
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from ckviz.frames import Frames, meta_key, to_corpus
from ckviz.stages import Op, predicate, sig, stage
from ckviz.stages.stats import grouped

MARKERS = ["article", "auxverb", "conj", "adverb", "ppron", "ipron", "preps", "quant"]


class CoordinationCfg(BaseModel):
    group_by: str = Field(description="speaker column whose values split one side into groups, e.g. meta.is-admin")
    side: Literal["target", "speaker"] = Field("target", description="group the targets (replied-to) or the speakers (repliers)")
    speaker_column: str = Field("", description="optional speaker filter on repliers, e.g. meta.type (blank = everyone)")
    speaker_op: Op = "=="
    speaker_value: str = ""
    target_column: str = Field("", description="optional speaker filter on targets (blank = everyone)")
    target_op: Op = "=="
    target_value: str = ""
    focus: Literal["speakers", "targets"] = Field("speakers", description="average over repliers or over targets")
    speaker_thresh: int = 0
    target_thresh: int = 3
    split_by: list[str] = Field([], description="utterance columns that split a person into several, e.g. meta.is-admin")


def scores(s) -> pd.DataFrame:
    """CoordinationScore -> one row per person, one column per marker (NaN where undefined)."""
    return pd.DataFrame([{m: v.get(m) for m in MARKERS} for v in s.values()], dtype=float)


def measures(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Per-person values of the paper's three aggregates plus each marker."""
    full = df.notna().all(axis=1)
    return {"aggregate 1": df[full].mean(axis=1), "aggregate 2": df.fillna(df.mean()).mean(axis=1),
            "aggregate 3": df.mean(axis=1), **{m: df[m].dropna() for m in MARKERS}}


@stage(CoordinationCfg)
def coordination(f: Frames, c: CoordinationCfg):
    """Coordination C(B, A) per marker and aggregated, one column per group; independent t-test when two groups."""
    from convokit import Coordination
    from scipy.stats import ttest_ind

    sp = f.speakers.assign(id=f.speakers.index)
    ids = lambda col, op, val: set(sp.index[predicate(sp, col, op, val)]) if col else set(sp.index)
    who, whom = ids(c.speaker_column, c.speaker_op, c.speaker_value), ids(c.target_column, c.target_op, c.target_value)
    corpus = to_corpus(f)
    coord = Coordination(speaker_thresh=c.speaker_thresh, target_thresh=c.target_thresh)
    coord.fit(corpus)
    groups = {}
    for g, members in sp.index.groupby(sp[c.group_by].astype(str)).items():
        a, b = (who, whom & set(members)) if c.side == "target" else (who & set(members), whom)
        s = coord.summarize(corpus, lambda x: x.id in a, lambda x: x.id in b, focus=c.focus,
                            split_by_attribs=[meta_key(k) for k in c.split_by])
        if s:
            groups[g] = measures(scores(s))
    if not groups:
        raise ValueError("no coordination scores: loosen thresholds or filters")
    names = sorted(groups)
    rnd = lambda v: None if pd.isna(v) else round(float(v), 5)
    rows = []
    for m in ["aggregate 1", "aggregate 2", "aggregate 3", *MARKERS]:
        vals = [groups[g][m] for g in names]
        if all(v.empty for v in vals):
            continue
        row = [m, *(rnd(v.mean()) for v in vals)]
        if len(names) == 2:
            p = ttest_ind(*vals).pvalue if min(len(v) for v in vals) > 1 else float("nan")
            row += [rnd(vals[0].mean() - vals[1].mean()), None if pd.isna(p) else sig(p)]
        rows.append(row)
    cols = ["measure", *names] + (["diff", "p"] if len(names) == 2 else [])
    counts = " · ".join(f"{g}: {len(groups[g]['aggregate 1'])} / {len(groups[g]['aggregate 3'])} speakers" for g in names)
    return f, {"kind": "table", "columns": cols, "rows": rows, "total": len(rows),
               "note": f"{c.side}s grouped by {c.group_by}; focus {c.focus}; n (all 8 markers / any marker) {counts}",
               "chart": grouped(rows, names, "coordination")}
