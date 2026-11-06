"""Zhang et al. 2018, "Conversations Gone Awry": first-exchange features, paired log-odds, paired leave-one-page-out
classifier, and prompt types from ConvoKit's precomputed vectors."""
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ckviz.frames import Frames
from ckviz.stages import *  # noqa: F403  helpers + registry
from ckviz.stages import stage


class Exchange(BaseModel):
    column: str = Field(description="utterance feature column (dict-valued or numeric), e.g. meta.politeness_strategies")
    into: str = Field("exchange", description="new conversation meta key: dict of <feature>@<comment position>")
    n: int = Field(2, ge=1, description="comment positions to keep (chronological; filter section headers first)")
    keys: list[str] = Field([], description="only these dict keys (empty = all)")
    binarize: bool = Field(True, description="1 if the feature occurs in the comment, else 0")


@stage(Exchange)
def exchange(f: Frames, c: Exchange):
    """One conversation-level feature per (feature, comment position): the paper's '19 strategies per comment'.
    Conversations with fewer than n comments get zeros for the missing positions."""
    utts = f.utts.sort_values("timestamp", kind="stable")
    x = features(need(utts, c.column)).fillna(0)
    if c.keys:
        x = x[[need(x, k).name for k in c.keys]]
    x = (x > 0).astype(int) if c.binarize else x
    pos = utts.groupby("conversation_id", sort=False).cumcount() + 1
    blocks = [x[pos == i].set_axis(utts.loc[pos == i, "conversation_id"]).reindex(f.convos.index).fillna(0)
              .astype(x.dtypes.iloc[0] if len(x.columns) else int).rename(columns=lambda k, i=i: f"{k}@{i}")
              for i in range(1, c.n + 1)]
    wide = pd.concat(blocks, axis=1)
    convos = f.convos.assign(**{f"meta.{c.into}": pd.Series(wide.to_dict("records"), index=wide.index)})
    return f._replace(convos=convos), table(convos)


class LogOdds(BaseModel):
    column: str = Field(description="conversation dict column, e.g. meta.exchange")
    label: str = Field(description="conversation column with the outcome, e.g. meta.conversation_has_personal_attack")
    op: Op = "=="
    value: str = "true"
    min_count: int = Field(50, ge=0, description="skip features occurring in fewer conversations (paper's Fig. 2: 50)")
    pair: str = Field("meta.pair_id", description="conversation column naming the matched partner")


def matched(f: Frames, label: str, op: str, value: str, pair: str) -> tuple[pd.Index, pd.Index]:
    """(label conversations, their matched non-label partners), aligned."""
    y = predicate(f.convos, label, op, value)
    awry = f.convos.index[y]
    other = need(f.convos, pair).loc[awry]
    bad = [a for a, b in other.items() if b not in f.convos.index or y[b]]
    if bad or other.duplicated().any():
        raise ValueError(f"every label conversation needs a distinct non-label pair; {len(bad)} do not, e.g. {bad[:3]}")
    return awry, pd.Index(other.values)


@stage(LogOdds)
def log_odds(f: Frames, c: LogOdds):
    """Per feature: share of label vs other conversations exhibiting it, log-odds ratio (+0.5 smoothing), and McNemar's
    exact test on the matched pairs (discordant pairs, binomial at 1/2); Holm across features.
    The paper's binomial test against the other group's observed share ignores that share's noise and the pairing."""
    from scipy.stats import binomtest

    awry, other = matched(f, c.label, c.op, c.value, c.pair)
    x = features(need(f.convos, c.column)).fillna(0) > 0
    a, o = x.loc[awry].to_numpy(), x.loc[other].to_numpy()
    rows = []
    for j, k in enumerate(x.columns):
        ka, ko, n = int(a[:, j].sum()), int(o[:, j].sum()), len(a)
        if ka + ko < c.min_count:
            continue
        lo = np.log((ka + .5) / (n - ka + .5)) - np.log((ko + .5) / (n - ko + .5))
        b, d = int((a[:, j] & ~o[:, j]).sum()), int((~a[:, j] & o[:, j]).sum())
        rows.append([k, ka + ko, round(ka / n, 4), round(ko / n, 4), round(float(lo), 3), sig(binomtest(b, b + d).pvalue) if b + d else 1.0])
    for r, ph in zip(rows, holm([r[5] for r in rows])):
        r.append(sig(ph))
    rows.sort(key=lambda r: r[5])
    return f, {"kind": "table", "columns": ["feature", "n", "awry_share", "other_share", "log_odds", "p", "p_holm"],
               "rows": rows, "total": len(rows), "note": f"{len(awry)} matched pairs; McNemar exact p, Holm-corrected"}


class PairedClassify(BaseModel):
    features: list[str] = Field(description='conversation dict / numeric columns, and/or "unigrams" (the conversation\'s text)')
    label: str = Field(description="conversation column with the outcome")
    op: Op = "=="
    value: str = "true"
    pair: str = Field("meta.pair_id", description="conversation column naming the matched partner")
    group: str = Field("meta.page_id", description="hold out one value per fold (leave-one-page-out)")
    max_words: int = Field(5000, ge=1, description="unigram vocabulary size (most frequent in the training fold)")
    C: float = Field(1.0, gt=0)
    tune_C: bool = Field(False, description="pick C from 1e-4..1e2 by 3-fold group CV inside each training fold")


def _matrix(f: Frames, cols: list[str], ids: pd.Index, fit: pd.Index, max_words: int):
    """Feature matrix for ids; the unigram vocabulary is fitted on `fit` only."""
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import CountVectorizer

    blocks = []
    for col in cols:
        if col == "unigrams":
            text = f.utts.sort_values("timestamp", kind="stable").groupby("conversation_id")["text"].agg(" ".join)
            text = text.reindex(f.convos.index).fillna("")
            cv = CountVectorizer(max_features=max_words).fit(text[fit])
            blocks.append(cv.transform(text[ids]))
        else:
            blocks.append(csr_matrix(features(need(f.convos, col)).fillna(0).astype(float).loc[ids].values))
    return hstack(blocks).tocsr()


@stage(PairedClassify)
def paired_classify(f: Frames, c: PairedClassify):
    """The paper's balanced task: given a matched pair, which conversation goes awry? Each pair is the difference of
    its two feature vectors, entered in both orders when training (so the model is order-free); ConvoKit's scaled
    logistic regression; leave-one-group-out CV. Ties (zero difference) count half."""
    from scipy.sparse import vstack
    from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut

    awry, other = matched(f, c.label, c.op, c.value, c.pair)
    groups = need(f.convos, c.group).loc[awry].values
    scores = []
    for tr, te in LeaveOneGroupOut().split(awry, groups=groups):
        fit = awry[tr].append(other[tr])
        d = lambda idx: _matrix(f, c.features, awry[idx], fit, c.max_words) - _matrix(f, c.features, other[idx], fit, c.max_words)
        dtr = d(tr)
        model = logreg().set_params(logreg__C=c.C)
        if c.tune_C:
            model = GridSearchCV(model, {"logreg__C": [10.0 ** i for i in range(-4, 3)]}, cv=GroupKFold(3))
        model.fit(vstack([dtr, -dtr]), np.r_[np.ones(len(tr)), np.zeros(len(tr))],
                  **({"groups": np.r_[groups[tr], groups[tr]]} if c.tune_C else {}))
        s = model.decision_function(d(te))
        scores += list((s > 0) + 0.5 * (s == 0))
    n_feat = _matrix(f, c.features, awry[:1], awry.append(other), c.max_words).shape[1]
    return f, {"kind": "table", "columns": ["features", "pairs", "folds", "n_features", "accuracy"],
               "rows": [[" + ".join(c.features), len(awry), len(set(groups)), n_feat, round(float(np.mean(scores)), 4)]],
               "total": 1, "note": f"leave-one-{c.group}-out; logistic regression "
               + ("C nested-tuned per fold" if c.tune_C else f"C={c.C:g}") + "; chance = 0.5"}


class PromptTypes(BaseModel):
    vectors: str = Field("downloads/conversations-gone-awry-corpus/vect_info.prompt_types__prompt_dists.6",
                         description="ConvoKit vector file stem (.npy + .keys) under the data directory: per-utterance distance to each prompt type")
    into: str = "prompt_type"
    max_distance: float = Field(1.0, description="farther than this from every type = the paper's null type")


@stage(PromptTypes)
def prompt_types(f: Frames, c: PromptTypes):
    """One-hot nearest prompt type per utterance, from the distances ConvoKit's PromptTypeWrapper stored with the
    corpus (the paper's k=6 types; footnote 7's null type when no centroid is within max_distance)."""
    from ckviz import datasets as ds

    stem = (ds.DATA / c.vectors).resolve()
    if not stem.is_relative_to(ds.DATA.resolve()):
        raise ValueError(f"vectors must stay under the data directory: {c.vectors!r}")
    dist = pd.DataFrame(np.load(f"{stem}.npy"), index=Path(f"{stem}.keys").read_text().split())
    k = dist.shape[1]
    dist = dist.reindex(f.utts.index).fillna(np.inf)
    best = dist.idxmin(axis=1).where(dist.min(axis=1) < c.max_distance)
    onehot = best.map(lambda b: {f"type_{i}": int(b == i) for i in range(k)})
    counts = best.value_counts()
    return (f._replace(utts=f.utts.assign(**{f"meta.{c.into}": onehot})),
            {"kind": "bar", "labels": [f"type_{i}" for i in range(k)] + ["none"],
             "values": [int(counts.get(i, 0)) for i in range(k)] + [int(best.isna().sum())], "measure": "utterances"})
