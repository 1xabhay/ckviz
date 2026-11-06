"""A pipeline is a list of single-key dicts: [{stage_name: config}, ...]."""
import hashlib
import json
import time

import yaml

from ckviz.frames import Frames
from ckviz.stages import apply

# ponytail: in-process cache of the last 64 stage outputs; move to disk/Redis when running >1 worker.
_cache: dict[str, tuple] = {}


def check_stages(stages: list) -> list:
    if not isinstance(stages, list):
        raise ValueError(f"stages must be a list, got {stages!r}")
    for s in stages:
        if not (isinstance(s, dict) and len(s) == 1):
            raise ValueError(f"each stage must be a single-key mapping, got {s!r}")
    return stages


def parse(text: str) -> dict:
    p = yaml.safe_load(text) or {}
    if not isinstance(p, dict):
        raise ValueError("a pipeline must be a mapping")
    check_stages(p.get("stages", []))
    return p


def json_safe(view: dict) -> dict:
    """NaN/inf -> null and numpy scalars -> Python, so every view serialises as strict JSON."""
    text = json.dumps(view, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return json.loads(text, parse_constant=lambda _: None)


def counts(f: Frames) -> dict:
    return {"utterances": len(f.utts), "speakers": len(f.speakers), "conversations": len(f.convos)}


def run(stages: list[dict], f: Frames, key: str) -> list[dict]:
    return list(iter_run(stages, f, key))


def iter_run(stages: list[dict], f: Frames, key: str):
    """Yields one step report per stage as it finishes."""
    failed = False
    for s in stages:
        (name, cfg), = s.items()
        key = hashlib.sha256(json.dumps([key, name, cfg], sort_keys=True).encode()).hexdigest()
        if failed:
            yield {"stage": name, "status": "skipped"}
            continue
        cached, t = key in _cache, time.perf_counter()
        try:
            if not cached:
                out, view = apply(name, f, cfg or {})
                out = out if out.meta or not f.meta else out._replace(meta=f.meta)  # stages needn't carry corpus meta
                _cache[key] = out, json_safe(view)
                if len(_cache) > 64:
                    _cache.pop(next(iter(_cache)))
            f, view = _cache[key]
            yield ({"stage": name, "status": "ok", "cached": cached, "counts": counts(f), "columns": list(f.utts.columns), "view": view,
                   "ms": round((time.perf_counter() - t) * 1000)})
        except Exception as e:
            failed = True
            yield {"stage": name, "status": "error", "cached": False, "error": f"{type(e).__name__}: {e}"}
