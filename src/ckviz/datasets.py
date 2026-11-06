"""Datasets come from datasets.yaml: `{dir: path}` (a ConvoKit corpus dir), `{convokit: name | [names]}` (ConvoKit
downloads, merged and tagged with meta.corpus; optional utterance_end_index), `{dsn: postgres-url}`, `{zip: path}`
(a ConvoKit zip export) or `{zip_url: url, method?, bearer? | basic_auth?, params?, sha256_header?}` (the same over
HTTP; each download is frozen as a snapshot, and checked against sha256_header when named). Any of these may add
`participants: path.csv, participants_on: <conversation meta column>` (a participant_id CSV joined onto conversations).

A dsn dataset is read through the ck_utterances / ck_speakers / ck_conversations views and frozen
into a ConvoKit dir under data/snapshots/<name>/<utc-timestamp>; runs always read a snapshot.
"""
import base64
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener
from functools import lru_cache
from pathlib import Path

import pandas as pd
import psycopg
import yaml
from convokit import download
from psycopg.rows import dict_row

from ckviz.frames import Frames, load_dir, to_corpus

CONFIG = Path(os.environ.get("CKVIZ_DATASETS", Path(__file__).parents[2] / "datasets.yaml"))
DATA = Path(os.environ.get("CKVIZ_DATA", CONFIG.parent / "data"))
SNAPSHOTS = DATA / "snapshots"  # read via DATA at call time so tests can redirect it


def datasets() -> dict[str, dict]:
    return yaml.safe_load(os.path.expandvars(CONFIG.read_text())) or {}


def snapshots(name: str) -> list[Path]:
    return sorted(p for p in (SNAPSHOTS / name).glob("*/") if not p.name.endswith(".partial"))


def check_paths(name: str, ds: dict) -> None:
    """Raises if a configured local path is missing. The message may reach API clients, so no absolute path in it."""
    for path in (ds[k] for k in ("dir", "zip", "participants") if k in ds):
        if not (CONFIG.parent / path).exists():
            shown = Path(path).name if Path(path).is_absolute() else path
            raise ValueError(f"dataset {name!r}: {shown} not found; see datasets.yaml")


def convokit_path(name: str) -> Path:
    """A local copy is used without touching the network. Otherwise ConvoKit downloads it; ConvoKit returns the last path
    in its global registry, which can be another clone's folder, so this data directory's copy still comes first."""
    here = DATA / "downloads" / name
    if (here / "utterances.jsonl").exists():
        return here
    returned = Path(download(name, data_dir=str(DATA / "downloads")))
    if (here / "utterances.jsonl").exists():
        return here
    if returned.exists():
        return returned
    raise ValueError(f"dataset {name!r}: ConvoKit records it as downloaded but the folder is gone; "
                     "remove its lines from downloads/downloaded.txt in ConvoKit's data directory to download it again")


def load(name: str) -> tuple[str, Frames]:
    """Returns (cache key, frames). dsn datasets use their latest snapshot, taking one if none exists."""
    ds = datasets().get(name)
    if ds is None:
        raise KeyError(name)
    check_paths(name, ds)
    end = ds.get("utterance_end_index")
    if "dir" in ds:
        sources = {name: (CONFIG.parent / ds["dir"]).resolve()}
    elif "zip" in ds:
        sources = {name: unpack_file(name, CONFIG.parent / ds["zip"])}
    elif "convokit" in ds:
        (DATA / "downloads").mkdir(parents=True, exist_ok=True)
        names = ds["convokit"] if isinstance(ds["convokit"], list) else [ds["convokit"]]
        sources = {n: convokit_path(n) for n in names}
    else:
        sources = {name: corpus_dir((snapshots(name) or [snapshot(name)])[-1])}
    stamps = {n: max(x.stat().st_mtime for x in Path(p).iterdir()) for n, p in sources.items()}
    frames = {n: _load_dir(p, end, stamps[n]) for n, p in sources.items()}
    key = f"{name}@{','.join(f'{p}:{stamps[n]}' for n, p in sources.items())}:{end}"
    f = next(iter(frames.values())) if len(frames) == 1 else merge(frames)
    if "participants" in ds:
        body = (CONFIG.parent / ds["participants"]).read_bytes()
        if "participants_on" not in ds:
            raise ValueError(f"dataset {name!r}: participants needs participants_on: <conversation meta column>")
        key += f":{ds['participants_on']}:" + hashlib.sha256(body).hexdigest()[:12]
        f = with_participants(f, body, ds["participants_on"])
    return key, f


def with_participants(f: Frames, csv: bytes, on: str) -> Frames:
    """Adds a participant_id CSV (e.g. a demographics table) as meta.* on each conversation whose `on` column names that
    participant, so join_meta reaches bot rows and people with no speaker row. Accounts with no record get NaN; records
    with no account are ignored."""
    if on not in f.convos:
        raise ValueError(f"participants_on: no conversation column {on!r}")
    p = pd.read_csv(io.BytesIO(csv), dtype=str)
    if "participant_id" not in p:
        raise ValueError("participants file has no participant_id column")
    p = p.set_index("participant_id").add_prefix("meta.")
    if p.index.has_duplicates:
        raise ValueError(f"participants file has {int(p.index.duplicated().sum())} duplicate participant_id rows")
    return f._replace(convos=f.convos.join(p, on=on))


def merge(frames: dict[str, Frames]) -> Frames:
    """Concatenate corpora, tagging utterances with meta.corpus and prefixing every id with "<corpus>:"
    (corpora can share ids, e.g. the wiki and stack-exchange politeness corpora share 39)."""
    def tag(n: str, f: Frames) -> Frames:
        # built as an object Series: Series.map would turn None (a root's reply_to) into NaN
        pre = lambda s: pd.Series([None if pd.isna(v) else f"{n}:{v}" for v in s], index=s.index, dtype=object)
        utts = f.utts.assign(**{"meta.corpus": n, "speaker": pre(f.utts["speaker"]), "reply_to": pre(f.utts["reply_to"]),
                                "conversation_id": pre(f.utts["conversation_id"])})
        return Frames(*(df.set_axis(n + ":" + df.index.astype(str)) for df in (utts, f.speakers, f.convos)), meta=f.meta)

    tagged = [tag(n, f) for n, f in frames.items()]
    return Frames(*(pd.concat([t[i] for t in tagged]) for i in range(3)), meta={n: t.meta for n, t in zip(frames, tagged)})


def _read_cache(cache: Path) -> Frames | None:
    """The cached frames, or None if absent or writable by anyone else: unpickling a planted file would run its code."""
    try:
        with open(cache, "rb") as fh:
            st = os.fstat(fh.fileno())
            return pd.read_pickle(fh) if st.st_uid == os.getuid() and not st.st_mode & 0o022 else None
    except FileNotFoundError:
        return None


@lru_cache(maxsize=4)
def _load_dir(path: Path, end: int | None, stamp: float) -> Frames:
    """Frames for a corpus dir, pickled under data/frames so big corpora parse once (keyed by path, end, mtime)."""
    cache = DATA / "frames" / (hashlib.sha1(f"{path}:{end}:{stamp}".encode()).hexdigest() + ".pkl")
    f = _read_cache(cache)
    if f is None:
        f = load_dir(path, utterance_end_index=end)
        cache.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.NamedTemporaryFile(dir=cache.parent, delete_on_close=False) as t:
            pd.to_pickle(f, t)
            t.close()
            os.replace(t.name, cache)
    return f


def snapshot(name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ds = datasets()[name]
    if "zip_url" in ds:
        body, prov = fetch_zip(ds)
        unpack(body, SNAPSHOTS / name / stamp, prov)
        return SNAPSHOTS / name / stamp
    (SNAPSHOTS / name).mkdir(parents=True, exist_ok=True)
    to_corpus(from_sql(ds["dsn"])).dump(stamp, base_path=str(SNAPSHOTS / name))
    return SNAPSHOTS / name / stamp


class NoRedirect(HTTPRedirectHandler):
    """urllib copies Authorization onto redirects, even cross-host; refuse, so a 3xx surfaces as an HTTPError."""

    def redirect_request(self, *a, **k):
        return None


def fetch_zip(ds: dict) -> tuple[bytes, dict]:
    """Download a zip export. Credentials go in headers and are never recorded."""
    url, method, params = ds["zip_url"], ds.get("method", "GET").upper(), ds.get("params") or {}
    data = urlencode(params).encode() if method != "GET" else None
    req = Request(url + ("?" + urlencode(params) if params and method == "GET" else ""), data=data or b"" if method != "GET" else None,
                  method=method)
    if "bearer" in ds:
        req.add_header("Authorization", f"Bearer {ds['bearer']}")
    if "basic_auth" in ds:
        req.add_header("Authorization", "Basic " + base64.b64encode(ds["basic_auth"].encode()).decode())
    try:
        with build_opener(NoRedirect).open(req, timeout=int(ds.get("timeout", 900))) as r:
            body, headers = r.read(), r.headers
    except HTTPError as e:
        raise ValueError(f"export download failed: HTTP {e.code} {e.reason}") from None
    except URLError as e:
        raise ValueError(f"export download failed: {e.reason}") from None
    sha = hashlib.sha256(body).hexdigest()
    recorded = {}
    if name := ds.get("sha256_header"):  # opt-in; once named, a missing header fails like a wrong one
        want = headers.get(name, "")
        if want.lower() != sha:
            raise ValueError(f"export sha256 mismatch: header {name}={want!r}, downloaded {sha}")
        recorded[name] = want
    return body, {"source": url, "params": params, "sha256": sha, "bytes": len(body),
                  "downloaded_at": datetime.now(timezone.utc).isoformat(), "headers": recorded}


def unpack_file(name: str, path: Path) -> Path:
    """A local zip export, unpacked once per file content into data/snapshots/<name>/<sha12>."""
    body = path.read_bytes()
    sha = hashlib.sha256(body).hexdigest()
    dest = SNAPSHOTS / name / sha[:12]
    return corpus_dir(dest) if dest.exists() else unpack(body, dest, {"source": path.name, "sha256": sha})


def unpack(body: bytes, dest: Path, provenance: dict) -> Path:
    """Extract a zip into dest (atomically, refusing absolute or ../ paths and .p pickles, which ConvoKit's Corpus
    loader would unpickle); returns the corpus folder inside. `dir` and `convokit` sources skip this: they are trusted."""
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        for m in z.infolist():
            if Path(m.filename).is_absolute() or ".." in Path(m.filename).parts or m.filename.lower().endswith(".p"):
                raise ValueError(f"unsafe path in zip: {m.filename!r}")
        # ponytail: 20 GB uncompressed cap as a zip-bomb guard; raise if real exports outgrow it.
        if sum(m.file_size for m in z.infolist()) > 20e9:
            raise ValueError("zip expands beyond 20 GB")
        tmp = dest.with_name(dest.name + ".partial")
        shutil.rmtree(tmp, ignore_errors=True)
        z.extractall(tmp)
    (tmp / "export.json").write_text(json.dumps(provenance, indent=2))
    tmp.rename(dest)
    return corpus_dir(dest)


def corpus_dir(path: Path) -> Path:
    """The folder holding utterances.jsonl: path itself or its single subfolder (some exports nest one level)."""
    if (path / "utterances.jsonl").exists():
        return path
    found = [p.parent for p in path.glob("*/utterances.jsonl")]
    if len(found) != 1:
        raise ValueError(f"no single ConvoKit corpus inside {path.name}")
    return found[0]


def from_sql(dsn: str) -> Frames:
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        conn.read_only = True

        def frame(sql: str, cols: list[str]) -> pd.DataFrame:
            rows = conn.execute(sql).fetchall()
            df = pd.DataFrame(rows, columns=cols + ["meta"]).set_index("id")
            meta = pd.json_normalize(df.pop("meta").fillna({}).tolist(), max_level=0).add_prefix("meta.")
            return df.join(meta.set_index(df.index))

        return Frames(
            frame("SELECT id, speaker, conversation_id, reply_to, timestamp, text, meta FROM ck_utterances",
                  ["id", "speaker", "conversation_id", "reply_to", "timestamp", "text"]),
            frame("SELECT id, meta FROM ck_speakers", ["id"]),
            frame("SELECT id, meta FROM ck_conversations", ["id"]),
        )
