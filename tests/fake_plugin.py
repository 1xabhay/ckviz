"""A stand-in third-party plugin (registered via a fake entry point in test_plugins)."""
from pydantic import BaseModel

from ckviz.stages import stage


class Echo(BaseModel):
    note: str = "hi"


@stage(Echo)
def fake_echo(f, c):
    return f, {"kind": "bar", "labels": [c.note], "values": [len(f.utts)]}
