"""Static checks on the single-page UI, plus its number formatter run under node (skipped without node)."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HTML = (Path(__file__).parents[1] / "src/ckviz/static/index.html").read_text()


def test_no_blocking_dialogs_or_two_decimal_formatting():
    assert "alert(" not in HTML
    assert "toFixed(2)" not in HTML


def test_accessibility_hooks_present():
    assert 'id="live" class="sr-only" aria-live="polite"' in HTML  # one persistent status region
    assert '"aria-describedby"' in HTML  # field help is linked, not title-only
    assert 'role: "alert"' not in HTML  # no per-poll alert nodes
    assert '"x-ckviz": "1"' in HTML  # CSRF header on every api() call


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_fmt():
    src = re.search(r"// --- number formatting \(fmt-begin\) ---(.*?)// \(fmt-end\)", HTML, re.S).group(1)
    cases = [[0.0145, ""], [0.0097, ""], [3.8e-31, "p"], [0.0123, "p_holm"], [None, "p"], [12345, ""], [0.5, "d"], [-0.0, "diff"]]
    js = src + f"console.log(JSON.stringify({json.dumps(cases)}.map(([v, c]) => fmt(v, c))))"
    out = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    assert out[:5] == ["0.0145", "0.0097", "3.8e-31", "0.0123", "—"]
    assert out[5] in ("12,345", "12.345", "12 345", "12 345") and out[6] in ("0.5", "0,5")
    assert out[7] == "0"


def test_chart_kinds_and_safe_dom():
    assert "innerHTML" not in HTML  # labels are untrusted: textContent / createTextNode only
    charts = re.search(r"const CHARTS = \{(.*?)\n\};", HTML, re.S).group(1)
    for kind in ("grouped", "heatmap", "line", "interval"):
        assert re.search(rf"^\s*{kind}[(:]", charts, re.M), kind
    assert "extra_charts" in HTML
    for i in range(1, 5):  # ordinal ramp, both modes
        assert HTML.count(f"--ord-{i}:") == 2


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_line_geometry():
    src = re.search(r"// \(geom-begin\)(.*?)// \(geom-end\)", HTML, re.S).group(1)
    js = src + """console.log(JSON.stringify([
      runs([1, null, 2, 3, null]),
      stepPts([[0, 1], [1, 0.8], [2, 0.5]]),
      logTicks(1, 300),
    ]))"""
    runs, step, lt = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    assert runs == [[0], [2, 3]]  # null y breaks the line
    assert step == [[0, 1], [1, 1], [1, 0.8], [2, 0.8], [2, 0.5]]  # step-after (Kaplan-Meier)
    assert lt[0] == 1 and lt[-1] == 1000 and 10 in lt and 100 in lt
