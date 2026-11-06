"""Word-list rates net of false-positive phrases (LIWC users' 'second phrase dictionary' trick)."""
from pydantic import Field

from ckviz.frames import Frames
from ckviz.stages import need, stage, table
from ckviz.stages.text import TOKEN, Lexicon, lexicon, tokens


class LexiconNet(Lexicon):
    exclude: list[str] = Field([], description="false-positive phrases (e.g. not completely, hello everyone); "
                                               "each occurrence is subtracted from the word count")


@stage(LexiconNet)
def lexicon_net(f: Frames, c: LexiconNet):
    """lexicon rate minus the rate of excluded phrases (+ <into>_fp). Same tokens as lexicon."""
    f, _ = lexicon(f, Lexicon(words=c.words, into=c.into, column=c.column))
    phrases = [tuple(TOKEN.findall(p.lower())) for p in c.exclude]
    toks = tokens(need(f.utts, c.column))
    fp = toks.map(lambda ts: sum(tuple(ts[i:i + len(p)]) == p for p in phrases for i in range(len(ts) - len(p) + 1)))
    n = f.utts["n_tokens"]
    fp_rate = (fp / n.where(n > 0)).fillna(0.0)
    utts = f.utts.assign(**{c.into: f.utts[c.into] - fp_rate, f"{c.into}_fp": fp_rate})
    return Frames(utts, f.speakers, f.convos), table(utts[["speaker", "text", c.into, f"{c.into}_fp", "n_tokens"]])
