"""ConvoKit text transformers and lexical comparisons."""
import re

import pandas as pd
from pydantic import BaseModel, Field

from ckviz.frames import Frames, to_corpus
from ckviz.stages import *  # noqa: F403  helpers + registry
from ckviz.stages import stage


@stage(NoConfig)
def text_clean(f: Frames, c: NoConfig):
    """ConvoKit TextCleaner: lowercases, normalises URLs/emails/numbers; original kept in meta.original."""
    from convokit import TextCleaner

    f = via_corpus(f, lambda corpus: TextCleaner(verbosity=10**9).transform(corpus))
    return f, table(f.utts)


@stage(NoConfig)
def parse(f: Frames, c: NoConfig):
    """spaCy dependency parse (ConvoKit TextParser) into meta.parsed; needed by politeness."""
    from convokit import TextParser

    f = via_corpus(f, lambda corpus: TextParser(verbosity=10**9).transform(corpus))
    return f, table(f.utts.drop(columns="meta.parsed"))


@stage(NoConfig)
def politeness(f: Frames, c: NoConfig):
    """ConvoKit PolitenessStrategies into meta.politeness_strategies; view = share of utterances per strategy."""
    from convokit import PolitenessStrategies

    need(f.utts, "meta.parsed")
    f = via_corpus(f, lambda corpus: PolitenessStrategies().transform(corpus))
    means = features(f.utts["meta.politeness_strategies"]).mean().round(4)
    return f, {"kind": "bar", "labels": list(means.index), "values": [float(v) for v in means], "measure": "mean"}


class BowClassify(Label):
    top_k: int = Field(15, ge=1, le=100)


@stage(BowClassify)
def bow_classify(f: Frames, c: BowClassify):
    """ConvoKit BoWTransformer + its default VectorClassifier model; view = most positive / negative unigram coefficients."""
    from convokit import BoWTransformer

    corpus = to_corpus(f)
    BoWTransformer(obj_type="utterance").fit_transform(corpus)
    m = corpus.get_vector_matrix("bow_vector")
    model = logreg().fit(m.matrix, labels(f, c).reindex(m.ids).values)
    coefs = pd.Series(model.named_steps["logreg"].coef_[0], index=m.columns)
    return f, diverging(coefs, f"{c.label} {c.op} {c.value}", "otherwise", c.top_k, "logistic regression coefficient")


class FightingWordsCfg(BaseModel):
    group_by: str = Field(description="utterance column holding the two groups")
    class1: str
    class2: str
    ngram_max: int = Field(3, ge=1, le=3)
    top_k: int = Field(15, ge=1, le=100)


@stage(FightingWordsCfg)
def fighting_words(f: Frames, c: FightingWordsCfg):
    from convokit import FightingWords

    groups = need(f.utts, c.group_by).astype(str)
    ids1, ids2 = set(groups.index[groups == c.class1]), set(groups.index[groups == c.class2])
    if not ids1 or not ids2:
        raise ValueError(f"both groups need utterances; {c.class1}={len(ids1)}, {c.class2}={len(ids2)}")
    fw = FightingWords(ngram_range=(1, c.ngram_max))
    fw.fit(to_corpus(f), class1_func=lambda u: u.id in ids1, class2_func=lambda u: u.id in ids2)
    z = fw.get_ngram_zscores(c.class1, c.class2)["z-score"]
    return f, diverging(z, c.class1, c.class2, c.top_k, "weighted log-odds z-score; |z| > 1.96 ≈ p < .05")


TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokens(col: pd.Series) -> pd.Series:
    """Lowercased word tokens per row; contractions kept whole (i'm, don't)."""
    return col.fillna("").astype(str).str.lower().str.replace("’", "'").str.findall(TOKEN)


class Lexicon(BaseModel):
    words: list[str] = Field(description="word list; a trailing * matches any ending (LIWC style), e.g. absolut*")
    into: str = Field("lexicon", description="new column: share of tokens matching the list")
    column: str = "text"


@stage(Lexicon)
def lexicon(f: Frames, c: Lexicon):
    """Rate of word-list tokens per utterance (+ n_tokens). Lowercased; contractions kept whole (i'm, don't)."""
    words = [w.lower() for w in c.words]
    exact, prefixes = {w for w in words if not w.endswith("*")}, tuple(w[:-1] for w in words if w.endswith("*"))
    toks = tokens(need(f.utts, c.column))
    n = toks.str.len()
    hits = toks.map(lambda ts: sum(t in exact or bool(prefixes and t.startswith(prefixes)) for t in ts))
    utts = f.utts.assign(**{c.into: (hits / n.where(n > 0)).fillna(0.0), "n_tokens": n})
    return Frames(utts, f.speakers, f.convos), table(utts[["speaker", "text", c.into, "n_tokens"]])
