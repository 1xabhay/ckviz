from importlib.metadata import EntryPoint

from fastapi.testclient import TestClient

from ckviz.api import app
from ckviz.stages import PLUGINS, STAGES, apply, load_plugins
from helpers import tiny

ep = lambda name, module: EntryPoint(name, module, "ckviz.stages")


def test_plugin_entry_point_registers_its_stages():
    load_plugins([ep("fake", "fake_plugin")])
    assert PLUGINS["fake"] == {"module": "fake_plugin", "status": "ok", "stages": ["fake_echo"]}
    _, view = apply("fake_echo", tiny(), {"note": "n"})
    assert view["values"] == [5]


def test_broken_plugin_is_reported_not_fatal():
    load_plugins([ep("broken", "no_such_module_xyz")])
    assert PLUGINS["broken"]["status"] == "error"
    assert "no_such_module_xyz" in PLUGINS["broken"]["error"]


def test_plugin_cannot_shadow_a_builtin_stage():
    load_plugins([ep("shadow", "shadow_plugin")])
    assert PLUGINS["shadow"]["status"] == "error" and "already registered" in PLUGINS["shadow"]["error"]
    assert STAGES["filter"].fn.__module__ == "ckviz.stages.shape"


def test_plugins_endpoint_lists_plugins_and_stage_sources():
    load_plugins([ep("fake", "fake_plugin")])
    client = TestClient(app, base_url="http://localhost")
    assert client.get("/api/plugins").json()["fake"]["status"] == "ok"
    assert client.get("/api/stages").json()["fake_echo"]["x-module"] == "fake_plugin"
