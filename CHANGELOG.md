# Changelog

## 0.1.0

First public release.

- Local web app: pipelines of stages over ConvoKit-format data, set up through forms and saved as YAML.
- Dataset kinds: `dir`, `convokit`, `dsn`, `zip`, `zip_url`, with optional participant records.
- Plugins: `ckviz-dlatk`, `ckviz-stats`, `ckviz-sentiment`.
- Reproductions of ConvoKit example notebooks and four published papers, under `examples/`.

## Known issues

Limitations of the published reproductions are in the README, under "Known limitations".

**Engineering**

- A `reply_to` that points at a filtered-out parent isn't turned into None in `to_corpus`.
- `predicate` doesn't match id-like strings such as "123".
- Stage-cache eviction isn't LRU.
- A `zip_url` download with no corpus in it leaves a snapshot that fails later loads until you delete it.
- `dsn` snapshots aren't written via a `.partial` folder.
- Job errors can contain raw exception text from ConvoKit.
- On Linux, the `./data` and `./pipelines` folders that `docker compose` creates may be owned by root, while the
  container runs as uid 1000.

**Reproducibility**

- The Supreme Court subset script isn't committed; `examples/papers/echoes-of-power.md` describes the steps.
- ConvoKit downloads have no recorded checksums.
- DLATK's dependencies aren't pinned.
- The frame-cache key doesn't include code and library versions.

**Before any shared host**

- Add a sign-in proxy.
- Return generic errors to clients and log details on the server.
- Use a SELECT-only database role with TLS.
- Set retention for `data/`.
- Cap jobs and caches.

**Interface**

- Some field labels need plainer language (for example "Class1" could be "Group A").
