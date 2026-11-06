# ckviz

ckviz is a local web app for analysing conversation data in [ConvoKit](https://convokit.cornell.edu/) format, made for
researchers who would rather not write code. You build an analysis as a list of **stages** (load, filter, transform,
compare, chart), set each stage up through a form, and see its output as the data runs through. Pipelines save as YAML,
so you can re-run them on another dataset. The repo also includes pipelines that reproduce ConvoKit example notebooks
and published papers.

## Quickstart

Install from a clone with [uv](https://docs.astral.sh/uv/getting-started/installation/). ckviz is not on PyPI and
`pip install` won't work, because its spaCy model (`en-core-web-sm`) isn't on PyPI either. The install is about 1.2 GB.

```sh
git clone https://github.com/1xabhay/ckviz
cd ckviz
uv sync
uv run ckviz
```

Open <http://127.0.0.1:8000>. The app opens on the `phq9-fighting-words` example, which uses the synthetic data in this
repo, and starts running it. You can also choose it from the pipeline list and press **Run**. The first run takes
from a few seconds to over a minute, depending on whether models have to load for the first time (measured at 30-70 s).
A repeat run with warm caches takes seconds at most.

## Docker

```sh
docker compose up
```

This builds the image with all three plugins and starts the app at <http://127.0.0.1:8765>, plus a local Postgres for
tests on port 54329. Set `CKVIZ_PORT` and `CKVIZ_DB_PORT` to use other ports. The app runs as uid 1000; on Linux, create
`./data` and `./pipelines` first and make them writable by that uid.

The image binds to 127.0.0.1 unless `CKVIZ_HOST` is set. Compose sets it and publishes the port on localhost only. A
bare `docker run -p` doesn't set it, so the app is unreachable from the host.

## Data

`datasets.yaml` names the datasets you can pick in the app. It ships with public corpora and commented examples of
each kind.

| kind | example | notes |
|---|---|---|
| `dir` | `my-data: {dir: path/to/corpus}` | a ConvoKit corpus folder |
| `convokit` | `movie-corpus: {convokit: movie-corpus}` | downloaded to `data/downloads`. A list merges corpora, adding `meta.corpus` and prefixing ids with `<corpus>:`. `utterance_end_index: N` stops at index N, so it reads N+1 utterances |
| `dsn` | `my-db: {dsn: "${MY_DATABASE_URL}"}` | a read-only Postgres connection to `ck_utterances`, `ck_speakers` and `ck_conversations` views, saved as a snapshot under `data/snapshots/` |
| `zip` | `my-zip: {zip: path/to/export.zip}` | a zipped ConvoKit corpus |
| `zip_url` | `my-url: {zip_url: "https://example.org/corpus.zip", bearer: "${MY_EXPORT_TOKEN}", sha256_header: X-Export-Sha256}` | downloaded and saved as a snapshot. Optional: `method`, `params`, `bearer` or `basic_auth`, and `sha256_header`, a response header that must hold the download's SHA-256 |

Paths are relative to `datasets.yaml`, and `${VAR}` is read from the environment. Keep credentials in the environment.

Any kind can add participant records: `participants: path.csv` with `participants_on: <conversation metadata column>`.
The CSV needs a `participant_id` column, and its other columns become conversation metadata.

Loaded corpora are cached as pickles in `data/frames`. Delete that folder after upgrading ckviz or pandas.

### Settings

| variable | default | what it sets |
|---|---|---|
| `CKVIZ_DATASETS` | `datasets.yaml` in the repo | the dataset list |
| `CKVIZ_DATA` | `data/` beside the dataset list | downloads, snapshots and caches |
| `CKVIZ_PIPELINES` | `pipelines/` beside the dataset list | where your saved pipelines go |
| `CKVIZ_LISTEN_PORT` | `8000` | the port for `uv run ckviz` |
| `CKVIZ_PORT` | `8765` | the host port for `docker compose` only |
| `CKVIZ_HOST` | `127.0.0.1` | the address to bind |
| `CKVIZ_ALLOWED_HOSTS` | `localhost,127.0.0.1,[::1]` | accepted `Host` headers |

## Plugins

Plugins add stages. `uv sync` installs all three for development; [`plugins/README.md`](plugins/README.md) shows how to
write one.

- [`ckviz-dlatk`](plugins/ckviz-dlatk/README.md): DLATK differential language analysis, run in its own Python via `uvx`.
- [`ckviz-stats`](plugins/ckviz-stats/README.md): ANOVA and mixed models, using statsmodels.
- [`ckviz-sentiment`](plugins/ckviz-sentiment/README.md): VADER sentiment.

## Reproductions

[`examples/README.md`](examples/README.md) is the ledger: ConvoKit notebooks and four papers (Al-Mosaiwi & Johnstone
2018; Danescu-Niculescu-Mizil et al. 2012 and 2013; Zhang et al. 2018), each with a paper-vs-ours write-up in
`examples/papers/`. Two corpora (`absolutist-forums`, `supreme-echoes-2012`) aren't downloaded automatically: the
write-ups give the build steps, and `datasets.yaml` has commented entries to uncomment once they're built.

## Known limitations

These qualify the published reproductions. Read them with the ledger.

- **Politeness Table 3:** the test allows 0.15 on the mean score, so on the 8 strategies whose paper score is smaller
  than that, a score could change sign and still pass. The signs in the current table all match the paper, but the test
  doesn't enforce it.
- **Politeness Table 4:** the classifier's C=0.1 was chosen after looking at in-domain results, so the matches are
  post hoc rather than independent confirmation.
- **Conversations Gone Awry, Table 3:** the "beats BOW" comparisons are within noise on 584 pairs, so don't read them as
  an established ordering.
- **Echoes of Power, Supreme Court:** the p-values treat each count of a Justice as independent, but the group is 11
  people counted many times. Don't rely on these p-values; the result is the direction of the effect.
- **Absolutist words:** the member-level Mann-Whitney p-values in `examples/papers/absolutist.md` come from an earlier
  run; the current pipeline withholds them. The Study 1 mixed-model results are in
  [`absolutist-mixed.md`](examples/papers/absolutist-mixed.md).

Engineering and deployment issues are listed in [`CHANGELOG.md`](CHANGELOG.md).

## Statistics: read before trusting a p-value

Utterances within a conversation or speaker aren't independent. At utterance level, `compare_groups` and `correlate`
detect when the grouping is nested and **withhold p-values**. Roll up to speaker or conversation level (`rollup`,
`level:`) or use `mixed_model`. `classify_cv` keeps whole conversations in the same fold by default.

## Tests

```sh
uv run pytest -q                                  # fast suite, about 20 s
CKVIZ_EXAMPLES=1 uv run pytest -q                 # also the reproductions: about 20-25 min and about 4 GB of downloads
```

Some slow tests need data that isn't downloaded automatically. They skip until you build it. The Postgres tests need a
database; see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Security

- **Network:** ckviz binds to localhost by default. The `Host` check stops DNS rebinding; it isn't access control.
- **No login:** anyone who can reach the port can use the app, so don't expose it to a network.
- **Trusted input:** treat `dir` and `convokit` sources as code you would run, because ConvoKit can unpickle files in a
  corpus. Only load corpora from sources you trust.
- **Participant data:** never commit it. `data/` holds participant text (downloads, snapshots, pickled caches) and is
  ignored by git. Keep it writable only by you.

See [`SECURITY.md`](SECURITY.md) for how to report a problem.

## Licence

MIT. See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md); the second covers material from
others and GPL dependencies that matter if you redistribute a bundle. To cite ckviz, see
[`CITATION.cff`](CITATION.cff).
