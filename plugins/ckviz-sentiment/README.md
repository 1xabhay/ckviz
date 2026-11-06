# ckviz-sentiment

| stage | what it does |
|---|---|
| `vader` | VADER sentiment (`vaderSentiment`, MIT) per utterance: the compound score (−1 … +1) in `into` (default `sentiment`), optionally `<into>_neg/_neu/_pos`. Missing text stays missing; empty text scores 0. |

VADER is a lexicon-and-rules model built for short English social-media text (Hutto & Gilbert, 2014). It handles
negation, intensifiers, capitals and emoticons, but not sarcasm or domain words; check it against hand-coded
messages before reading small differences. Common cut-offs: compound ≥ 0.05 positive, ≤ −0.05 negative.

Tests:

```sh
uv run --no-project --with-editable . --with-editable plugins/ckviz-sentiment --with pytest --with httpx pytest -q plugins/ckviz-sentiment/tests -p no:cacheprovider
```
