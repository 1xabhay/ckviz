"""Aggregates, group comparisons, classifiers."""
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from ckviz.frames import Frames
from ckviz.stages import *  # noqa: F403  helpers + registry
from ckviz.stages import stage


class Aggregate(BaseModel):
    by: str = Field(description="utterance column to group by")
    measure: Literal["utterances", "speakers", "conversations", "mean_words", "proportion"] = "utterances"
    value: str = Field("", description="measure proportion: boolean / 0-1 column, e.g. replied")


@stage(Aggregate)
def aggregate(f: Frames, c: Aggregate):
    """Per value of `by`: counts or mean words (bar), or the proportion of a 0/1 column with Wilson 95% CIs (table + interval chart).
    Numeric group labels sort numerically."""
    key = need(f.utts, c.by)
    num = pd.to_numeric(key, errors="coerce")
    if num.notna().sum() == key.notna().sum():
        key = num.map(lambda v: f"{v:g}", na_action="ignore")
        sort = lambda idx: sorted(idx, key=float)
    else:
        key, sort = key.astype(str), sorted
    g = f.utts.groupby(key)
    if c.measure == "proportion":
        if not c.value:
            raise ValueError("measure proportion needs value: the boolean / 0-1 column")
        y = pd.to_numeric(need(f.utts, c.value), errors="coerce").astype(float)
        if not y.dropna().isin([0, 1]).all():
            raise ValueError(f"{c.value} must be boolean or 0/1 for measure proportion")
        ok = y.notna() & key.notna()
        s = y[ok].groupby(key[ok]).agg(["size", "sum"])
        rows = []
        for lbl in sort(s.index):
            n, k = int(s.loc[lbl, "size"]), int(s.loc[lbl, "sum"])
            lo, hi = wilson(k, n)
            rows.append([lbl, n, k, round(k / n, 4), round(lo, 4), round(hi, 4)])
        return f, {"kind": "table", "columns": [c.by, "n", "k", "proportion", "lo", "hi"], "rows": rows, "total": len(rows),
                   "note": f"proportion of {c.value} per {c.by} (utterances, missing dropped); 95% Wilson intervals, not adjusted "
                           "for clustering within speakers",
                   "chart": {"kind": "interval", "labels": [r[0] for r in rows], "x_label": f"proportion {c.value}",
                             "series": [{"name": c.value, "est": [r[3] for r in rows], "lo": [r[4] for r in rows], "hi": [r[5] for r in rows]}]}}
    s = {
        "utterances": g.size(),
        "speakers": g["speaker"].nunique(),
        "conversations": g["conversation_id"].nunique(),
        "mean_words": g["text"].apply(lambda t: t.str.split().str.len().mean()).round(2),
    }[c.measure]
    s = s[sort(s.index)]
    return f, {"kind": "bar", "labels": list(s.index), "values": [float(v) if c.measure == "mean_words" else int(v) for v in s], "measure": c.measure}


def nested(df: pd.DataFrame, values: pd.Series, name: str, level: str) -> str:
    """At utterance level: a warning if `values` (named `name`) never varies inside a conversation (or speaker), since then utterances
    are pseudo-replicates and a p-value treating them as independent is too small. Empty string if not nested."""
    if level != "utterance":
        return ""
    d = df[values.notna()].assign(_v=values)
    hits = sorted((n.size, unit) for unit, key in (("conversations", "conversation_id"), ("speakers", "speaker"))
                  if key in d and (n := d.groupby(key)["_v"].nunique()).size < len(d) and (n <= 1).all())
    if not hits:
        return ""
    n, unit = hits[0]  # ponytail: the unit with fewer clusters (the smaller effective n); no crossed-effects check
    # no ": " in here: tests/test_paper_absolutist.py parses the n's after the note's first ": "
    return (f"{name} is constant within {unit} ({n} clusters), so utterances aren't independent and p-values are "
            "withheld; roll up to conversation/speaker level (rollup stage) or use mixed_model · ")


class CompareGroups(BaseModel):
    level: Level = "utterance"
    column: str = Field(description="numeric or dict-valued column, e.g. meta.politeness_strategies")
    group_by: str = Field(description="column holding the group labels")
    class1: str
    class2: str


@stage(CompareGroups)
def compare_groups(f: Frames, c: CompareGroups):
    """Per feature: group means, Cohen's d, Mann-Whitney p and Holm-corrected p (across features), sorted by p.
    At utterance level, if the groups are nested in conversations or speakers, p is withheld and rows are sorted by |d|."""
    from scipy.stats import mannwhitneyu

    df = at(f, c.level)
    x, g = features(need(df, c.column)), need(df, c.group_by).astype(str)
    a, b = x[g == c.class1], x[g == c.class2]
    warn = nested(df, g.where(g.isin([c.class1, c.class2])), c.group_by, c.level)
    if a.empty or b.empty:
        raise ValueError(f"both groups need rows; {c.class1}={len(a)}, {c.class2}={len(b)}")
    rows = []
    for k in x.columns:
        xa, xb = a[k].dropna().astype(float), b[k].dropna().astype(float)
        sd = (((len(xa) - 1) * xa.var() + (len(xb) - 1) * xb.var()) / max(len(xa) + len(xb) - 2, 1)) ** 0.5
        p = mannwhitneyu(xa, xb).pvalue if len(xa) and len(xb) else float("nan")
        rows.append([k, round(xa.mean(), 4), round(xb.mean(), 4), round(xa.mean() - xb.mean(), 4),
                     round((xa.mean() - xb.mean()) / sd, 3) if sd else 0.0, 1.0 if p != p else sig(p)])
    for r, ph in zip(rows, holm([r[5] for r in rows])):
        r.append(sig(ph))
    if warn:
        for r in rows:
            r[5] = r[6] = None
    rows.sort(key=(lambda r: -abs(r[4])) if warn else (lambda r: r[5]))
    return f, {"kind": "table", "columns": ["feature", c.class1, c.class2, "diff", "d", "p", "p_holm"], "rows": rows,
               "total": len(rows), "note": f"{warn}{c.level}s: {c.class1} n={len(a)}, {c.class2} n={len(b)}",
               "chart": grouped(rows[:20], [c.class1, c.class2], "mean")}  # ponytail: top 20 by p (by |d| when p is withheld); table has all


class Correlate(BaseModel):
    level: Level = "utterance"
    x: str = Field(description="numeric or dict-valued column")
    y: str = Field(description="numeric column, e.g. meta.phq9")
    method: Literal["spearman", "pearson"] = "spearman"


@stage(Correlate)
def correlate(f: Frames, c: Correlate):
    """Correlation of each feature in x with y; Holm-corrected across features; sorted by p.
    At utterance level, if y is constant within conversations or speakers, p is withheld (None), sorted by |r|."""
    from scipy.stats import pearsonr, spearmanr

    df = at(f, c.level)
    xs, y = features(need(df, c.x)), pd.to_numeric(need(df, c.y))
    rows = []
    for k in xs.columns:
        pair = pd.concat([xs[k].astype(float), y], axis=1).dropna()
        r, p = (spearmanr if c.method == "spearman" else pearsonr)(pair.iloc[:, 0], pair.iloc[:, 1]) if len(pair) > 2 else (0.0, 1.0)
        rows.append([k, round(float(r), 4), 1.0 if p != p else sig(p), len(pair)])
    for r, ph in zip(rows, holm([r[2] for r in rows])):
        r.append(sig(ph))
    warn = nested(df, y, c.y, c.level)
    if warn:
        for r in rows:
            r[2] = r[4] = None
    rows.sort(key=(lambda r: -abs(r[1])) if warn else (lambda r: r[2]))
    return f, {"kind": "table", "columns": ["feature", "r", "p", "n", "p_holm"], "rows": rows, "total": len(rows),
               "note": f"{warn}{c.method} correlation with {c.y} at {c.level} level"}


class ClassifyCV(Label):
    features: str = Field(description="dict-valued feature column, e.g. meta.politeness_strategies")
    folds: int = Field(5, ge=2, le=20)
    seed: int = 0
    group: Literal["conversation", "speaker", "none"] = Field(
        "conversation", description="keep each conversation (or speaker) in one fold; none = plain k-fold over utterances")


@stage(ClassifyCV)
def classify_cv(f: Frames, c: ClassifyCV):
    """ConvoKit's default classifier (scaled logistic regression), k-fold accuracy with seeded shuffling. Folds are
    grouped by conversation by default: utterances of one conversation in both train and test inflate accuracy."""
    from sklearn.model_selection import GroupKFold, KFold, cross_val_score

    x, y = features(need(f.utts, c.features)).fillna(0), labels(f, c)
    key = {"conversation": "conversation_id", "speaker": "speaker"}.get(c.group)
    cv = GroupKFold(c.folds, shuffle=True, random_state=c.seed) if key else KFold(c.folds, shuffle=True, random_state=c.seed)
    scores = cross_val_score(logreg(), x.values, y.values, cv=cv, groups=need(f.utts, key).values if key else None)
    return f, {"kind": "bar", "labels": [f"fold {i + 1}" for i in range(len(scores))],
               "values": [round(float(v), 4) for v in scores], "measure": "accuracy",
               "note": f"mean accuracy {scores.mean():.4f} · {int(y.sum())} positive of {len(y)} · "
                       + (f"folds grouped by {c.group}" if key else "ungrouped folds")}
