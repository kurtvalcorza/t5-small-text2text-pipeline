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
    # Historical wording (2026-10-26): exact recorded times are allowed again, each labelled with its run.
    assert "215.2 s" not in md
    assert md.count("112.2 s") >= 2 and "on a Colab T4" in md
    assert "96.3 s (`e82d281c`, 2026-10-25)" in md


def test_historical_wording_records_every_hosted_run():
    # Historical wording (2026-10-26): timings and quoted scores name every recorded hosted run; no run-status
    # sentence. Fails on PR #8 head cb54454.
    md = _markdown()
    for gone in ("about 2 minutes", "has not been hosted-run", "not re-run here", "the hosted run printed"):
        assert gone not in md, gone
    for quoted in ("107.9 s (`622b9ef3`, 2026-10-25)", "`622b9ef3` (commit `524353b`, both 2026-10-25)",
                   "your run may take a different time"):
        assert quoted in md, quoted


def test_release_record_states_the_declared_spec():
    text = RV.read_text(encoding="utf-8")
    assert "the notebook declares 2.2 since" not in text
    assert "Specification 2.5, which the notebook declares" in text
    nb = json.loads(NB.read_text(encoding="utf-8"))
    assert nb["metadata"]["dimer"]["notebook_spec"] == "2.5"
