# Contributing

## Setup

```sh
uv sync                 # Python 3.12; includes the plugins and dev tools
uv run pytest -q        # fast suite, about 20 s
uv run ruff check
```

CI runs the same commands, after `uv sync --locked`. If you change dependencies, run `uv lock` and commit `uv.lock`.

The reproductions are slow: `CKVIZ_EXAMPLES=1 uv run pytest -q` takes about 20-25 min and about 4 GB of downloads. Run
the relevant `tests/test_paper_*.py` file if you touch something a reproduction depends on.

The Postgres tests need a database:

```sh
docker compose up -d db
CKVIZ_TEST_DSN=postgresql://postgres:ckviz@localhost:54329/postgres uv run pytest -q tests/test_sql.py
```

The tests connect to whatever `CKVIZ_TEST_DSN` names, so if another Postgres already uses port 54329, they could reach
it instead. Pick a free port with `CKVIZ_DB_PORT` and use the same port in the DSN:

```sh
CKVIZ_DB_PORT=54330 docker compose up -d db
CKVIZ_TEST_DSN=postgresql://postgres:ckviz@localhost:54330/postgres uv run pytest -q tests/test_sql.py
```

## Making a change

Write the test first, and watch it fail before you write the code.

- **New stage:** add a module under `src/ckviz/stages/`. Any `@stage(Config)` function is picked up automatically.
- **New library:** write a plugin under `plugins/`, following [`plugins/README.md`](plugins/README.md).
- **Reproduction claims:** say what didn't match. Update the [ledger](examples/README.md) and the write-up with the
  change. Fix tolerances before running, as the existing write-ups describe.

## Data

Never commit participant data, including in test fixtures, screenshots and saved pipeline output. Use
`examples/data/synthetic` or generate fake data in the test.
