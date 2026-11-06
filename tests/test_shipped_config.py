"""What a stranger gets from a clean clone: the shipped datasets.yaml and the app's first screen."""
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
TEXT = (ROOT / "datasets.yaml").read_text()


def test_local_entries_exist():
    for name, ds in yaml.safe_load(TEXT).items():
        for kind in ("dir", "zip"):
            if kind in ds:
                assert (ROOT / ds[kind]).exists(), f"{name}: {ds[kind]} is not in the repo"


def test_first_screen_opens_the_example_that_works_from_a_clone():
    from fastapi.testclient import TestClient

    from ckviz.api import app

    # A string check on the served page, not a browser run: the startup itself isn't exercised here.
    client = TestClient(app, base_url="http://localhost")
    assert 'pipes.includes("phq9-fighting-words")' in client.get("/").text
    assert "phq9-fighting-words" in client.get("/api/pipelines").json()
    dataset = yaml.safe_load((ROOT / "examples/phq9-fighting-words.yaml").read_text())["dataset"]
    assert (ROOT / yaml.safe_load(TEXT)[dataset]["dir"]).exists()


def test_a_pipeline_whose_dataset_is_not_registered_does_not_run_on_another():
    page = (ROOT / "src/ckviz/static/index.html").read_text()
    load = page[page.index("async function loadPipeline"):page.index("async function fillDatasets")]
    assert "not in datasets.yaml" in load and load.index("not in datasets.yaml") < load.index("run()")


def test_docker_build_has_the_files_pyproject_names():
    docker = (ROOT / "Dockerfile").read_text()
    assert "COPY pyproject.toml uv.lock README.md LICENSE ./" in docker
