# Plugins

A plugin is a Python package that adds stages. At startup ckviz imports every installed package registered
under the `ckviz.stages` entry point. Import failures show up at `/api/plugins` and never stop the app.
A plugin can't reuse a built-in stage name.

## Layout

```
plugins/ckviz-<name>/
  pyproject.toml
  src/ckviz_<name>/__init__.py     # @stage functions (same API as src/ckviz/stages/*)
  tests/test_<name>.py             # fast tests; guard slow ones with CKVIZ_EXAMPLES
```

`pyproject.toml`:

```toml
[project]
name = "ckviz-<name>"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["ckviz", "<the library>"]

[project.entry-points."ckviz.stages"]
<name> = "ckviz_<name>"

[tool.uv.sources]
ckviz = { workspace = true }

[build-system]
requires = ["uv_build>=0.11.7,<0.12.0"]
build-backend = "uv_build"
```

Add the plugin to the root `pyproject.toml` as an extra, then run `uv sync --extra <name>`:

```toml
[project.optional-dependencies]
<name> = ["ckviz-<name>"]

[tool.uv.sources]
ckviz-<name> = { workspace = true }
```

For containers: `docker compose build --build-arg UV_SYNC_ARGS="--extra <name>"`.

## Writing stages

```python
from pydantic import BaseModel, Field
from ckviz.frames import Frames
from ckviz.stages import stage, need, table   # also: at, features, holm, sig, via_corpus, predicate, ...

class Cfg(BaseModel):               # becomes the UI form: str / int / float / bool / Literal / list[str]
    column: str = Field(description="shown as a hint in the form")

@stage(Cfg)
def my_stage(f: Frames, c: Cfg):    # pure: return new frames + a view
    return f, table(f.utts)
```

Import heavy libraries inside the stage function. The app then starts fast, and a missing optional
dependency only breaks that one stage.

## Libraries that can't share ckviz's environment

Some tools pin incompatible versions or need a different Python. For those, keep the plugin package light
and have the stage run the tool in its own environment, e.g.
`subprocess.run(["uvx", "--from", "<pkg>", "<cli>", ...])`. Pass data through files (CSV / SQLite) in a temp dir.
