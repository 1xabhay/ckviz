"""Stages: pure functions (Frames, config) -> (Frames, view). Config models double as UI form schemas.

Helpers and the registry live here; stages live in sibling modules, which are all imported at the bottom,
so a new module only has to exist to be registered. Installed packages add stages through the
`ckviz.stages` entry-point group (see plugins/README.md).
"""
import importlib
import json
import pkgutil
from typing import Callable, Literal, NamedTuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ckviz.frames import Frames, meta_key, to_corpus

View = dict


class Stage(NamedTuple):
    config: type[BaseModel]
    fn: Callable[[Frames, BaseModel], tuple[Frames, View]]


STAGES: dict[str, Stage] = {}


def stage(config: type[BaseModel]):
    def register(fn):
        old = STAGES.get(fn.__name__)
        if old and old.fn.__module__ != fn.__module__:
            raise ValueError(f"stage {fn.__name__!r} already registered by {old.fn.__module__}")
        STAGES[fn.__name__] = Stage(config, fn)
        return fn
    return register


def apply(name: str, f: Frames, cfg: dict) -> tuple[Frames, View]:
    if name not in STAGES:
        raise ValueError(f"unknown stage {name!r}")
    s = STAGES[name]
    return s.fn(f, s.config.model_validate(cfg))


def table(df: pd.DataFrame, n: int = 50) -> View:
    head = df.head(n).reset_index()
    return {"kind": "table", "columns": list(head.columns), "rows": json.loads(head.to_json(orient="values")), "total": len(df)}


def need(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise ValueError(f"no column {col!r}; have {sorted(map(str, df.columns))}")
    return df[col]


def via_corpus(f: Frames, transform: Callable) -> Frames:
    """Run a ConvoKit transformer and copy its utterance text + meta back onto the frames."""
    c = to_corpus(f)
    transform(c)
    out = c.get_utterances_dataframe().reindex(f.utts.index)
    derived = {"meta." + meta_key(col) for col in f.utts.columns if not col.startswith("meta.")}
    new = {col: out[col] for col in out.columns if col.startswith("meta.") and col not in derived}
    return Frames(f.utts.assign(text=out["text"], **new), f.speakers, f.convos)


def features(col: pd.Series) -> pd.DataFrame:
    """A dict-valued column (e.g. politeness_strategies) as one numeric column per key."""
    if col.map(lambda v: isinstance(v, dict)).all():
        return pd.DataFrame(col.tolist(), index=col.index).rename(columns=short)
    return pd.to_numeric(col).to_frame(col.name)


def short(name: str) -> str:
    return name.removeprefix("feature_politeness_==").removesuffix("==")


def diverging(s: pd.Series, pos_name: str, neg_name: str, k: int, note: str = "") -> View:
    s = s.sort_values()
    rows = lambda x: [{"label": str(i), "value": round(float(v), 3)} for i, v in x.items()]
    return {"kind": "diverging", "pos_name": pos_name, "neg_name": neg_name,
            "pos": rows(s[::-1].head(k)), "neg": rows(s.head(k)), "note": note}


def prune(f: Frames, utts: pd.DataFrame) -> Frames:
    return Frames(utts, f.speakers[f.speakers.index.isin(utts["speaker"])],
                  f.convos[f.convos.index.isin(utts["conversation_id"])])


OPS = {"==": "eq", "!=": "ne", ">": "gt", ">=": "ge", "<": "lt", "<=": "le"}
Op = Literal["==", "!=", ">", ">=", "<", "<=", "contains", "in"]


def coerce(v: str):
    try:
        return json.loads(v)
    except ValueError:
        return v


def predicate(df: pd.DataFrame, column: str, op: str, value: str) -> pd.Series:
    col = need(df, column)
    if op == "in":  # comma-separated list of values, matched as text
        m = col.astype(str).isin([v.strip() for v in value.split(",")])
    elif op == "contains":
        m = col.astype(str).str.contains(value, case=False, regex=False)
    else:
        m = getattr(col, OPS[op])(coerce(value))
    return m.fillna(False).astype(bool)

class Label(BaseModel):
    label: str = Field(description="utterance column the label comes from")
    op: Op = "=="
    value: str


def labels(f: Frames, c: Label) -> pd.Series:
    return predicate(f.utts, c.label, c.op, c.value)


def logreg():
    # ConvoKit's default classifier model. Built here directly because convokit 5.0.0's Classifier(pred_feats=...)
    # and VectorClassifier are broken (missing attribute / bad kwarg).
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("standardScaler", StandardScaler(with_mean=False)), ("logreg", LogisticRegression(solver="liblinear"))])


class NoConfig(BaseModel):
    pass


Level = Literal["utterance", "speaker", "conversation"]


def at(f: Frames, level: str) -> pd.DataFrame:
    return {"utterance": f.utts, "speaker": f.speakers, "conversation": f.convos}[level]


def holm(ps: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in input order."""
    order, adj, running = sorted(range(len(ps)), key=lambda i: ps[i]), [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def sig(p: float) -> float:
    return float(f"{p:.3g}")


def wilson(k, n, z: float = 1.959963984540054):
    """Wilson score 95% interval for k successes of n (Wilson 1927); works on scalars and arrays, n = 0 gives NaN."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p, den = k / n, 1 + z * z / n
        mid, half = (p + z * z / (2 * n)) / den, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def grouped(rows, names, measure):
    """Grouped-bar chart payload: categories = first column, one series per following column in `names` order."""
    return {"kind": "grouped", "labels": [r[0] for r in rows], "measure": measure,
            "series": [{"name": n, "values": [r[i + 1] for r in rows]} for i, n in enumerate(names)]}


for _m in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_m.name}")


PLUGINS: dict[str, dict] = {}


def load_plugins(eps=None) -> None:
    """Import every `ckviz.stages` entry point; a plugin that fails is recorded in PLUGINS, never fatal."""
    from importlib.metadata import entry_points

    for ep in entry_points(group="ckviz.stages") if eps is None else eps:
        before = set(STAGES)
        try:
            ep.load()
            PLUGINS[ep.name] = {"module": ep.value, "status": "ok", "stages": sorted(set(STAGES) - before)}
        except Exception as e:
            PLUGINS[ep.name] = {"module": ep.value, "status": "error", "error": f"{type(e).__name__}: {e}"}


load_plugins()
