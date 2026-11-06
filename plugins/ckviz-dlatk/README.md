# ckviz-dlatk

[DLATK](https://github.com/dlatk/dlatk) (WWBP's Differential Language Analysis ToolKit) as a ckviz stage.

`dlatk_ngrams_correlate` runs DLATK's DLA workflow on the current frames: `--add_ngrams -n N`, then
`--feat_occ_filter --set_p_occ P`, then `--correlate` of each n-gram's relative frequency against a numeric
speaker or conversation meta column (Pearson r, DLATK's p correction or `none`). It returns a table
`feature, r, p, n, freq` sorted by p. Messages are all utterances in the frames, so filter first
(e.g. `meta.role == person`). See `examples/dlatk-phq9.yaml`, which runs it on the repo's invented corpus and
recovers the two words planted there: "compost" (r = 0.92) and "bicycle" (r = -0.96), both p < .001.

## License boundary

DLATK is GPL-3.0. This plugin never imports or vendors it: DLATK runs as a separate process via `uvx`, and the two
exchange data only through a temp SQLite db, command-line arguments and a CSV file.

## Install

From the repo root: `uv sync --extra dlatk`. The plugin itself only needs ckviz; DLATK runs through `uvx`,
which needs `uv` on PATH and network the first time (it fetches Python 3.11 + `dlatk==1.3.15` and caches them).
Set `CKVIZ_DLATK` to use a different command, e.g. `CKVIZ_DLATK=/opt/dlatk/bin/dlatkInterface.py`.

## Why DLATK runs in its own environment

DLATK 1.3.15 (latest) pins `numpy<=1.25.2`, `pandas<=1.5.3`, `scikit-learn<=1.1.3`, `scipy<=1.11.4`. None of these
have Python 3.12 wheels and ckviz needs pandas 3, so DLATK can't be imported in-process. Following the
plugins README pattern, the stage writes a temp SQLite db (`msgs(message_id, unit_id, message)`,
`outcomes(unit_id, outcome)`), runs `uvx --python 3.11 --from dlatk==1.3.15 dlatkInterface.py --db_engine sqlite`,
and parses the `--csv` output. The 1.3.15 wheel doesn't require `mysqlclient`, so no MySQL headers are needed.

Notes:
- DLATK chooses the occurrence filter's group-word threshold from the group column's *name* (500 for
  `user_id`). `occ_group_freq_thresh` defaults to that 500 and is always passed explicitly, because the stage's
  group column has a different name.
- DLATK creates an (unused) `~/sqlite_data/` directory on every run.

## Tests

```
uv run --no-project --with-editable . --with-editable plugins/ckviz-dlatk --with pytest --with httpx \
  pytest -q plugins/ckviz-dlatk/tests -p no:cacheprovider          # add CKVIZ_EXAMPLES=1 to run DLATK itself
```
