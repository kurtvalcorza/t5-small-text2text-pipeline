"""Regression tests for the 2026-10-05 notebook review findings (T5S-M1..M3, T5S-m1, T5S-m2).

They need only CI's dependencies: the carried module's evaluation and BYOD helpers run on real text, and the generated
notebook's own learner cells are executed with a stand-in pipeline (no weights, no torch call). None of this is model
evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

import t5_small_text2text_pipeline.pipeline as module
from t5_small_text2text_pipeline import (
    baseline_output,
    evaluation_report,
    fetch_reference_sample,
    known_prefix,
    read_byod_inputs,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"t5s_fix_{name}", ROOT / "tools" / f"{name}.py")
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


build = _load("build_notebook")
TEMPLATE = _load("notebook_template").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(notebook: dict) -> list[str]:
    return [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]


def _markdown(notebook: dict) -> str:
    return "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")


def _cell(notebook: dict, startswith: str) -> str:
    return next(s for s in _code(notebook) if s.startswith(startswith))


class _StandInPipeline:
    """Echoes the first sentence of the input: enough to drive the learner cells without weights."""

    device = "cpu"
    source = "stand-in"

    def generate(self, text, max_new_tokens=64, num_beams=1):
        output, _kind = baseline_output(text)
        return {
            "text": output,
            "generated_tokens": len(output.split()),
            "input_tokens": len(text.split()) + 1,
            "stopped_by": "eos",
            "known_prefix": known_prefix(text),
            "generation": {"max_new_tokens": max_new_tokens, "num_beams": num_beams, "do_sample": False, "decision_rule": module.DECISION_RULE},
        }


def _namespace(tmp_path: Path, monkeypatch) -> dict:
    monkeypatch.chdir(tmp_path)
    import os
    import time

    namespace = {k: v for k, v in vars(module).items() if not k.startswith("__")}
    namespace.update(os=os, time=time, json=json, pipe=_StandInPipeline())
    return namespace


# ---- T5S-M1: the default path measures; references are scored ----------------------------------------------------


def _jsonl(records: list[dict]) -> bytes:
    return "\n".join(json.dumps(r) for r in records).encode("utf-8")


def _records(n: int) -> list[dict]:
    body = "We study a problem. " + "It matters for many reasons and has a long history. " * 6
    return [{"paper_id": f"p{i}", "source": [f"Sentence {i} opens the abstract.", body], "target": [f"Paper {i} studies a problem."]} for i in range(n)]


def test_t5s_M1_reference_sample_is_digest_pinned_and_parsed(tmp_path: Path, monkeypatch) -> None:
    data = _jsonl(_records(4) + [{"paper_id": "short", "source": ["Too short."], "target": ["x"]}])
    monkeypatch.setattr(module, "REFERENCE_SAMPLE_BYTES", len(data))
    monkeypatch.setattr(module, "REFERENCE_SAMPLE_SHA256", hashlib.sha256(data).hexdigest())
    items = fetch_reference_sample(3, cache_dir=tmp_path, fetcher=lambda _url: data)
    assert [i["id"] for i in items] == ["p0", "p1", "p2"]
    assert all(i["input"].startswith("summarize: ") and i["references"] for i in items)
    # the cached copy is reused without a fetch
    assert fetch_reference_sample(3, cache_dir=tmp_path, fetcher=lambda _url: pytest.fail("refetched")) == items
    # an altered download is refused, never parsed
    with pytest.raises(ValueError, match="refusing it"):
        fetch_reference_sample(3, cache_dir=tmp_path / "other", fetcher=lambda _url: data + b" ")


def test_t5s_M1_evaluation_report_scores_beside_lead1_and_is_not_measurable_without_references() -> None:
    source = "summarize: The first sentence names the topic. The second gives details nobody needs."
    result = {"text": "The topic is named in the first sentence.", "input": source, "generated_tokens": 9, "stopped_by": "max_new_tokens"}
    report = evaluation_report([result, {**result, "stopped_by": "eos"}], [["The first sentence names the topic."], ["The first sentence names the topic."]])
    assert report["verdict"] == "sample-sanity" and [m["metric"] for m in report["metrics"]] == ["rouge1", "rouge2", "rougeL"]
    assert report["baselines"][0]["rouge1"] == 100.0  # Lead-1 equals the reference here
    assert report["outputs_stopped_by_max_new_tokens"] == [0]
    assert evaluation_report([result])["verdict"] == "not-measurable"


def test_t5s_M1_lead1_baseline_on_the_pinned_sample_matches_the_prose_when_the_file_is_present() -> None:
    cached = ROOT / "weights" / "scitldr" / "test.jsonl"
    if not cached.is_file():
        pytest.skip("the pinned SciTLDR-A file is not cached in this checkout (it is downloaded by the notebook)")
    items = fetch_reference_sample(cache_dir=cached.parent)
    results = [{"input": i["input"], "text": baseline_output(i["input"])[0]} for i in items]
    report = evaluation_report(results, [i["references"] for i in items])
    assert [m["value"] for m in report["metrics"]] == [27.22, 11.12, 22.22]


def test_t5s_M1_notebook_measures_on_the_referenced_sample(notebook: dict, tmp_path: Path, monkeypatch) -> None:
    md = _markdown(notebook)
    assert "**ROUGE-1 27.22, ROUGE-2 11.12, ROUGE-L 22.22**" in md
    assert "no metric helper" not in md and "the verdict is always `not-measurable`" not in md
    data = _jsonl(_records(20))
    monkeypatch.setattr(module, "REFERENCE_SAMPLE_BYTES", len(data))
    monkeypatch.setattr(module, "REFERENCE_SAMPLE_SHA256", hashlib.sha256(data).hexdigest())
    ns = _namespace(tmp_path, monkeypatch)
    (tmp_path / "weights" / "scitldr").mkdir(parents=True)
    (tmp_path / "weights" / "scitldr" / "test.jsonl").write_bytes(data)
    for prefix in ("import hashlib", "import json", "import time", "reference_items = fetch_reference_sample()"):
        exec(_cell(notebook, prefix), ns)  # noqa: S102 - the notebook's own learner cells
    report = json.loads((tmp_path / "outputs" / f"{TEMPLATE['stem']}_evaluation_report.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "sample-sanity" and report["n_items"] == 20 and report["sample"]["sha256"] == module.REFERENCE_SAMPLE_SHA256
    inputs = json.loads((tmp_path / "outputs" / f"{TEMPLATE['stem']}_inputs_evaluation_report.json").read_text(encoding="utf-8"))
    assert inputs["verdict"] == "not-measurable"  # the authored inputs have no references


# ---- T5S-M2: isolated runtime, no restart --------------------------------------------------------------------------


def test_t5s_M2_no_kernel_install_and_no_restart_instruction(notebook: dict) -> None:
    assert "Restart the runtime" not in NOTEBOOK.read_text(encoding="utf-8")
    sources = _code(notebook)
    assert not any("[sys.executable, '-m', 'pip'" in s for s in sources)
    kernel = [s for s in sources if "# dimer: kernel cell" in s]
    assert len(kernel) == 2
    install = next(s for s in kernel if "LOCK_TEXT = r'''" in s)
    for needed in ('"--managed-python"', '"--require-hashes"', '"--only-binary"', "UV_SHA256", "LOCK_SHA256"):
        assert needed in install


def test_t5s_M2_carried_lock_is_the_committed_hash_lock_of_the_pins() -> None:
    lock = (ROOT / TEMPLATE["lock"]).read_text(encoding="utf-8")
    build.check_lock(build._pins(ROOT, TEMPLATE), lock)
    assert "--only-binary :all:" in lock.splitlines()[1] and "x86_64-manylinux" in lock.splitlines()[1]


# ---- T5S-M3: the guided layer --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "marker",
    ["## How to use this notebook", "**Who this notebook is for.**", "## The task: Input → Model/System → Output", "## Roadmap", "<strong>Glossary</strong>", "**Predict before running:**", "**What to notice", "<summary>Check your reasoning</summary>", "## 9. Activity: change one thing — beam search", "## Troubleshooting", "## Conclusion (your notes)"],
)
def test_t5s_M3_guided_layer_marker_is_present(notebook: dict, marker: str) -> None:
    assert marker in _markdown(notebook)


def test_t5s_M3_infrastructure_cells_are_collapsed(notebook: dict) -> None:
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    learner_start = next(i for i, c in enumerate(code) if c["source"].startswith("import hashlib\nfrom pathlib import Path"))
    assert learner_start >= 4
    for cell in code[:learner_start]:
        assert cell["metadata"].get("cellView") == "form", cell["source"][:60]
    assert _markdown(notebook).count("**Predict before running:**") >= 3


# ---- T5S-m1: BYOD by path, BOM, encodings, upload count ------------------------------------------------------------


def test_t5s_m1_bom_keeps_the_first_prefix_and_latin1_is_refused_by_name() -> None:
    texts, refs = read_byod_inputs("notes.txt", "﻿translate English to German: The house is wonderful.\r\nsummarize: A text.\r\n".encode())
    assert known_prefix(texts[0]) == "translate English to German: " and refs is None
    with pytest.raises(ValueError, match=r"latin\.txt: not UTF-8 text"):
        read_byod_inputs("latin.txt", "translate English to French: café".encode("latin-1"))


def test_t5s_m1_csv_with_references_is_read_and_incomplete_rows_are_refused() -> None:
    texts, refs = read_byod_inputs("pairs.csv", b"input,reference\ntranslate English to German: Hello.,Hallo.\n")
    assert texts == ["translate English to German: Hello."] and refs == ["Hallo."]
    with pytest.raises(ValueError, match="row 2 has an empty `reference`"):
        read_byod_inputs("pairs.csv", b"input,reference\nsummarize: x,\n")
    with pytest.raises(ValueError, match="needs a header with an `input` column"):
        read_byod_inputs("pairs.csv", b"text\nsummarize: x\n")


def test_t5s_m1_path_based_byod_runs_in_jupyter_and_scores_references(notebook: dict, tmp_path: Path, monkeypatch) -> None:
    ns = _namespace(tmp_path, monkeypatch)
    csv_path = tmp_path / "mine.csv"
    csv_path.write_bytes("﻿input,reference\nsummarize: The cat sat. It was warm.,The cat sat.\n".encode())
    source = _cell(notebook, "import hashlib").replace("USE_BYOD = False", "USE_BYOD = True").replace("BYOD_PATH = ''", f"BYOD_PATH = {str(csv_path)!r}")
    exec(source, ns)  # noqa: S102
    assert ns["references"] == ["The cat sat."] and ns["sample_kind"] == "BYOD file with references"
    for prefix in ("import json", "import time"):
        exec(_cell(notebook, prefix), ns)  # noqa: S102
    report = evaluation_report(ns["results"], ns["references"], sample_kind=ns["sample_kind"])
    assert report["verdict"] == "sample-sanity"


@pytest.mark.parametrize("uploads", [{}, {"a.txt": b"summarize: x", "b.txt": b"summarize: y"}])
def test_t5s_m1_cancelled_or_multi_file_upload_stops_with_the_rule(notebook: dict, tmp_path: Path, monkeypatch, uploads: dict) -> None:
    files = types.ModuleType("google.colab.files")
    files.upload = lambda: uploads
    colab = types.ModuleType("google.colab")
    colab.files = files
    google = types.ModuleType("google")
    google.colab = colab
    for name, mod in (("google", google), ("google.colab", colab), ("google.colab.files", files)):
        monkeypatch.setitem(sys.modules, name, mod)
    ns = _namespace(tmp_path, monkeypatch)
    source = _cell(notebook, "import hashlib").replace("USE_BYOD = False", "USE_BYOD = True")
    with pytest.raises(RuntimeError, match=rf"Upload exactly one \.txt or \.csv file \(received {len(uploads)}\)"):
        exec(source, ns)  # noqa: S102


# ---- T5S-m2: the release record no longer contradicts its run row ---------------------------------------------------


def test_t5s_m2_release_record_is_consistent() -> None:
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "never run end-to-end" not in record and "will be the first real execution" not in record
    assert "Not recorded: restart status, runtime versions" in record
    assert "Notebook Specification 2.2" in (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
