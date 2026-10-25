"""Prose numbers stay true across every recorded hosted run (Kurt, 2026-10-25: point to output + ranges).

Recorded hosted runs: Colab T4 `7cdec4b1` 112.2 s (2026-10-09), Colab T4 `e82d281c` 96.3 s (2026-10-25),
Kaggle CPU of an earlier notebook 215.2 s (2026-09-14). Timings are approximate with the runtime named.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "t5_small_text2text_colab.ipynb"
RV = ROOT / "docs" / "release-verification.md"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_timings_are_approximate_and_name_the_runtime():
    md = _markdown()
    for exact in ("112.2 s", "96.3 s", "215.2 s"):
        assert exact not in md, exact
    assert md.count("about 2 minutes") >= 2 and "on a Colab T4" in md
    assert "`e82d281c` on 2026-10-25" in md


def test_release_record_states_the_declared_spec():
    text = RV.read_text(encoding="utf-8")
    assert "the notebook declares 2.2 since" not in text
    assert "Specification 2.5, which the notebook declares" in text
    nb = json.loads(NB.read_text(encoding="utf-8"))
    assert nb["metadata"]["dimer"]["notebook_spec"] == "2.5"
