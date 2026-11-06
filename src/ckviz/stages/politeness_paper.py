"""Danescu-Niculescu-Mizil et al. 2013, "A computational approach to politeness": Table 3 and Table 4 stages."""
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field

from ckviz.frames import Frames
from ckviz.stages import *  # noqa: F403  helpers + registry
from ckviz.stages import stage


class StrategyTable(Label):
    features: str = Field(description="dict-valued strategy column, e.g. meta.politeness_strategies")
    score: str = Field(description="numeric score column, e.g. meta.Normalized Score")


@stage(StrategyTable)
def strategy_table(f: Frames, c: StrategyTable):
    """Per strategy: requests exhibiting it, their mean score and % in the label class (e.g. top quartile).
    p_score = Mann-Whitney vs the rest; p_top = binomial test vs the overall label rate."""
    from scipy.stats import binomtest, mannwhitneyu

    x, y, top = features(need(f.utts, c.features)).fillna(0) > 0, pd.to_numeric(need(f.utts, c.score)), labels(f, c)
    rows = []
    for k in x.columns:
        on = x[k]
        if not on.any():
            continue
        p_score = mannwhitneyu(y[on], y[~on]).pvalue if (~on).any() else 1.0
        rows.append([k, int(on.sum()), round(float(y[on].mean()), 4), round(100 * float(top[on].mean()), 2),
                     sig(p_score), sig(binomtest(int(top[on].sum()), int(on.sum()), float(top.mean())).pvalue)])
    return f, {"kind": "table", "columns": ["feature", "n", "mean_score", "pct_top", "p_score", "p_top"],
               "rows": rows, "total": len(rows),
               "note": f"{len(y)} utterances; overall mean score {y.mean():.3f}, {100 * top.mean():.1f}% in label class"}


class TextClassify(Label):
    features: list[str] = Field(["unigrams"], description='"unigrams" and/or dict-valued / numeric columns')
    model: Literal["svm", "logreg"] = "svm"
    C: float = Field(1.0, gt=0, description="inverse regularisation strength (smaller = simpler model)")
    min_count: int = Field(10, ge=1, description="drop unigrams seen fewer times in the training data")
    domain: str = Field("", description="column naming each row's domain; empty = one domain, k-fold CV")
    train: str = Field("", description="domain to train on (CV within it when test is the same or empty)")
    test: str = ""
    folds: int = Field(10, ge=2, description="CV folds when train == test")
    seed: int = 0


def _fit_score(c: TextClassify, utts: pd.DataFrame, y: pd.Series, train: pd.Index, test: pd.Index) -> float:
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.svm import LinearSVC

    blocks_tr, blocks_te = [], []
    for col in c.features:
        if col == "unigrams":
            cv = CountVectorizer()
            m = cv.fit_transform(utts.loc[train, "text"])
            keep = np.asarray(m.sum(axis=0)).ravel() >= c.min_count
            blocks_tr.append(m[:, keep])
            blocks_te.append(cv.transform(utts.loc[test, "text"])[:, keep])
        else:
            x = features(need(utts, col)).fillna(0).astype(float)
            blocks_tr.append(csr_matrix(x.loc[train].values))
            blocks_te.append(csr_matrix(x.loc[test].values))
    xtr, xte = hstack(blocks_tr).tocsr(), hstack(blocks_te).tocsr()
    if xtr.shape[1] == 0:
        raise ValueError(f"no features left (min_count={c.min_count})")
    model = LinearSVC(C=c.C) if c.model == "svm" else logreg().set_params(logreg__C=c.C)
    return float((model.fit(xtr, y[train]).predict(xte) == y[test]).mean())


@stage(TextClassify)
def text_classify(f: Frames, c: TextClassify):
    """Unigram (min_count) + strategy features; linear SVM (as in the paper) or ConvoKit's logistic regression.
    Train on one domain and test on another, or stratified k-fold CV within one domain."""
    from sklearn.model_selection import StratifiedKFold

    utts, y = f.utts, labels(f, c)
    dom = need(utts, c.domain).astype(str) if c.domain else pd.Series("all", index=utts.index)
    train, test = c.train or "all", c.test or c.train or "all"
    tr, te = utts.index[dom == train], utts.index[dom == test]
    if tr.empty or te.empty:
        raise ValueError(f"no rows for train={train!r} ({len(tr)}) or test={test!r} ({len(te)})")
    if train != test:
        acc = _fit_score(c, utts, y, tr, te)
        setting = "cross-domain"
    else:
        folds = StratifiedKFold(c.folds, shuffle=True, random_state=c.seed).split(tr, y[tr])
        acc = float(np.mean([_fit_score(c, utts, y, tr[a], tr[b]) for a, b in folds]))
        setting = f"{c.folds}-fold CV"
    return f, {"kind": "table", "columns": ["train", "test", "n_train", "n_test", "accuracy"],
               "rows": [[train, test, len(tr), len(te), round(acc, 4)]], "total": 1,
               "note": f"{setting}; {c.model} (C={c.C:g}) on {' + '.join(c.features)}; {int(y[te].sum())}/{len(te)} positive in test"}
