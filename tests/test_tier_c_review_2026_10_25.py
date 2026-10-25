"""Tier C review findings T90-1..T90-3 (2026-10-25)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "t5_small_text2text_colab.ipynb"


def _nb():
    return json.loads(NB.read_text(encoding="utf-8"))


def _markdown():
    md = "\n".join("".join(c["source"]) for c in _nb()["cells"] if c["cell_type"] == "markdown")
    return " ".join(md.split())


def test_t90_1_dat9_and_domain_claim():
    md = _markdown()
    assert "so this sample is out of its training domain" not in md
    assert "**Overlap with the training data:**" in md and "inferred, not verified" in md


def test_t90_2_no_stale_no_run_claims():
    md = _markdown()
    for stale in ("no run of this revision is recorded yet", "No run of this revision has been recorded yet",
                  "has not been timed yet"):
        assert stale not in md, stale
    assert "`7cdec4b1`" in md and "112.2 s" in md


def test_t90_3_spec_items():
    nb = _nb()
    dimer = nb["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.5" and dimer["notebook_kind"] == "TUTORIAL"
    first_code = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    head = "\n".join("".join(c["source"]) for c in nb["cells"][:first_code])
    assert "Notice.** For research, training and evaluation only." in head
    md = _markdown()
    assert "[1] C. Raffel" in md and "`DIMER_ISOLATED_ENV`" in md
