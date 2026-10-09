"""Text-to-text generation with the pinned ``google-t5/t5-small`` checkpoint.

The class loads weights only from a digest-verified local snapshot (``weights/t5-small/``) or, when
explicitly allowed, from the Hugging Face Hub at the pinned revision. The caller supplies the task
prefix (``summarize: ``, ``translate English to German: `` ...); this module never adds one.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "google-t5/t5-small"
MODEL_REVISION = "df1b051c49625cf57a3d0d8d3863ed4d13564fe4"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "t5-small"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

MAX_INPUT_TOKENS = 512  # ``n_positions`` in the snapshot config.json; longer inputs are rejected, not cut
MAX_NEW_TOKENS = 512  # ceiling on decoder steps per call
DEFAULT_MAX_NEW_TOKENS = 64
MAX_TEXT_CHARS = 20_000  # pre-tokenisation guard on the input string
MAX_NUM_BEAMS = 8
DECISION_RULE = "greedy argmax per decoding step (num_beams=1); beam search when num_beams > 1; no sampling"
# The four prefixes the upstream checkpoint was trained on, read from ``task_specific_params`` in the
# snapshot config.json. Reported back as ``known_prefix``; the pipeline does not prepend any of them.
TASK_PREFIXES = (
    "summarize: ",
    "translate English to German: ",
    "translate English to French: ",
    "translate English to Romanian: ",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        return json.load(fh)


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest.get("files", []):
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one non-empty str that already carries its task prefix; the pipeline prepends none",
    "text_chars": [1, MAX_TEXT_CHARS],
    "input_tokens": [1, MAX_INPUT_TOKENS],
    "max_new_tokens": [1, MAX_NEW_TOKENS],
    "num_beams": [1, MAX_NUM_BEAMS],
    "task_prefixes": list(TASK_PREFIXES),
    "decision_rule": DECISION_RULE,
    "preprocessing": (
        "SentencePiece encoding with no prefix added and no truncation: an input over "
        "MAX_INPUT_TOKENS is rejected with a ValueError naming the count, never cut"
    ),
}


def _check_inputs(text: Any, max_new_tokens: Any, num_beams: Any) -> str:
    """Raise TypeError/ValueError naming the first violated ceiling; return the text.

    The encoder-token ceiling is not checked here because it needs the loaded tokenizer;
    ``_check_input_tokens`` applies it inside the pipeline once the count is known.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    if not text.strip():
        raise ValueError("text is empty")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text has {len(text)} chars; ceiling is MAX_TEXT_CHARS={MAX_TEXT_CHARS}")
    for name, value, ceiling in (
        ("max_new_tokens", max_new_tokens, MAX_NEW_TOKENS),
        ("num_beams", num_beams, MAX_NUM_BEAMS),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int")
        if not 1 <= value <= ceiling:
            raise ValueError(f"{name} must be between 1 and {ceiling}, got {value}")
    return text


def _check_input_tokens(n_input: int) -> int:
    """The encoder-token ceiling, applied once the tokenizer has counted."""
    if n_input > MAX_INPUT_TOKENS:
        raise ValueError(f"input is {n_input} tokens; ceiling is MAX_INPUT_TOKENS={MAX_INPUT_TOKENS}")
    return n_input


def known_prefix(text: str) -> str | None:
    """Which trained task prefix ``text`` starts with, or ``None``; nothing is prepended."""
    return next((prefix for prefix in TASK_PREFIXES if text.startswith(prefix)), None)


def validate_inputs(
    texts: Sequence[str],
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    num_beams: int = 1,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-input observations, verdict).

    Rejection is reported by raising exactly as ``generate`` would: both route through
    ``_check_inputs``. ``generate`` takes one text per call, so ``texts`` is the batch the notebook
    will loop over and every entry is validated with the same settings. An input that starts with
    no trained prefix is **not** rejected — the pipeline does not refuse it either — but the
    manifest records ``known_prefix: null`` so the caller can see it. The encoder-token ceiling
    (``MAX_INPUT_TOKENS``) needs the loaded tokenizer and is enforced inside ``generate``.
    """
    if isinstance(texts, str | bytes) or not isinstance(texts, Sequence):
        raise TypeError("texts must be a sequence of str, not a single string")
    if not texts:
        raise ValueError("texts must hold at least one item")
    checked = [_check_inputs(text, max_new_tokens, num_beams) for text in texts]
    if names is not None and len(names) != len(checked):
        raise ValueError("names must have one entry per text")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {
                "id": names[i] if names else f"input{i:02d}",
                "chars": len(text),
                "known_prefix": known_prefix(text),
            }
            for i, text in enumerate(checked)
        ],
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


# ---------------------------------------------------------------------------------------------------------
# Reference-based evaluation (T5S-M1): ROUGE-1/2/L F1 (own implementation, rouge-score recipe without
# stemming), a trivial baseline, a pinned referenced sample (SciTLDR-A test) and a BYOD CSV reader.
# ---------------------------------------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[^\W_]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
METRIC_DEFINITIONS = {
    "rouge1": "unigram overlap F1 with the best-matching reference, averaged over items; percent",
    "rouge2": "bigram overlap F1 with the best-matching reference, averaged over items; percent",
    "rougeL": "longest-common-subsequence F1 with the best-matching reference, averaged over items; percent",
    "tokenisation": (
        "lower-cased runs of letters and digits (Unicode, so German, French and Romanian words stay whole); "
        "no stemming; on ASCII text the same tokens as rouge-score without its stemmer"
    ),
}


def rouge_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _f1(overlap: int, n_hyp: int, n_ref: int) -> float:
    if overlap == 0 or n_hyp == 0 or n_ref == 0:
        return 0.0
    precision, recall = overlap / n_hyp, overlap / n_ref
    return 2 * precision * recall / (precision + recall)


def rouge_n(hypothesis: str, reference: str, n: int) -> float:
    hyp, ref = rouge_tokens(hypothesis), rouge_tokens(reference)
    hyp_grams = Counter(tuple(hyp[i : i + n]) for i in range(len(hyp) - n + 1))
    ref_grams = Counter(tuple(ref[i : i + n]) for i in range(len(ref) - n + 1))
    return _f1(sum((hyp_grams & ref_grams).values()), sum(hyp_grams.values()), sum(ref_grams.values()))


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token in a:
        current = [0]
        for j, other in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if token == other else max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def rouge_l(hypothesis: str, reference: str) -> float:
    hyp, ref = rouge_tokens(hypothesis), rouge_tokens(reference)
    return _f1(_lcs_length(hyp, ref), len(hyp), len(ref))


def text_metrics(hypotheses: Sequence[str], references: Sequence[Sequence[str]]) -> dict[str, Any]:
    """Mean ROUGE-1/2/L F1 in percent over parallel outputs and reference lists (best reference per item)."""
    if len(hypotheses) != len(references):
        raise ValueError(f"{len(hypotheses)} outputs but {len(references)} reference lists")
    if not hypotheses:
        raise ValueError("no outputs to score")
    per_item = []
    for hypothesis, refs in zip(hypotheses, references, strict=True):
        if not refs:
            raise ValueError("every item needs at least one reference")
        per_item.append(
            {
                "rouge1": max(rouge_n(hypothesis, r, 1) for r in refs),
                "rouge2": max(rouge_n(hypothesis, r, 2) for r in refs),
                "rougeL": max(rouge_l(hypothesis, r) for r in refs),
            }
        )
    return {
        "n": len(per_item),
        **{
            key: 100.0 * sum(d[key] for d in per_item) / len(per_item)
            for key in ("rouge1", "rouge2", "rougeL")
        },
        "per_item": per_item,
    }


def baseline_output(source: str) -> tuple[str, str]:
    """The trivial output for one prefixed input: Lead-1 (the first sentence after the prefix) for
    ``summarize: ``, the untranslated source for a ``translate`` prefix (copy-source)."""
    prefix = known_prefix(source) or ""
    body = source[len(prefix) :].strip()
    if prefix.startswith("translate"):
        return body, "copy-source (the untranslated input as the output)"
    return _SENTENCE_RE.split(body)[0], "lead-1 (the first sentence of the input as the output)"


def evaluation_report(
    result: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    references: Sequence[str] | Sequence[Sequence[str]] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report; it **measures** whenever references exist.

    ``result`` is one ``generate`` result or a list of them (each carrying its prefixed ``input``).
    Without ``references`` the verdict is ``not-measurable``. With references — one reference string or a
    list of references per result — the outputs are scored with ROUGE-1/2/L F1 (best reference per item)
    beside the trivial baseline of ``baseline_output`` (Lead-1 for summaries, copy-source for
    translations), and the verdict is ``sample-sanity``: tutorial evidence on the given items, not a
    benchmark; with fewer than two items no dispersion can be stated.
    """
    results = [result] if isinstance(result, Mapping) else list(result)
    if not results:
        raise ValueError("at least one result is required")
    base = {
        "task": "caller-prefixed text-to-text generation (summarisation, translation)",
        "score_semantics": (
            "the pipeline emits no probability, confidence or score: generated_tokens, input_tokens "
            "and stopped_by are counts and flags, and "
            f"{results[0].get('generation', {}).get('decision_rule', DECISION_RULE)} produces some token at "
            "every step with no minimum-probability cut-off and no shipped acceptance threshold"
        ),
        "sample_kind": sample_kind,
        "n_items": len(results),
        "n_generated_tokens": sum(int(r.get("generated_tokens", 0)) for r in results),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if references is None:
        return {
            **base,
            "metrics": [],
            "baselines": [],
            "verdict": "not-measurable",
            "reason": "the evaluated sample has no reference outputs",
            "needs": (
                "reference outputs from the deployment domain — a reference summary per document, a "
                "reference translation per sentence — over enough items to state a dispersion; pass them "
                "as `references` (BYOD: a CSV with `input,reference` columns) and the report scores "
                "ROUGE-1/2/L beside the trivial baseline, excluding or re-running outputs whose "
                "stopped_by is max_new_tokens"
            ),
        }
    refs = [list(r) if isinstance(r, list | tuple) else [r] for r in references]
    if isinstance(result, Mapping) and len(refs) > 1 and all(isinstance(r, str) for r in references):
        refs = [list(references)]  # several references for the one result
    if len(refs) != len(results):
        raise ValueError(f"{len(results)} results but {len(refs)} reference entries")
    sources = [str(r.get("input", "")) for r in results]
    if not all(sources):
        raise ValueError("every result must carry its prefixed `input` so the baseline can be formed")
    model = text_metrics([str(r["text"]) for r in results], refs)
    baseline_texts, baseline_kinds = zip(*(baseline_output(s) for s in sources), strict=True)
    baseline = text_metrics(list(baseline_texts), refs)
    truncated = [i for i, r in enumerate(results) if r.get("stopped_by") == "max_new_tokens"]
    return {
        **base,
        "metrics": [
            {
                "metric": key,
                "value": round(model[key], 2),
                "units": "percent F1",
                "definition": METRIC_DEFINITIONS[key],
            }
            for key in ("rouge1", "rouge2", "rougeL")
        ],
        "per_item": model["per_item"],
        "baselines": [
            {
                "id": "trivial",
                "kinds": sorted(set(baseline_kinds)),
                **{key: round(baseline[key], 2) for key in ("rouge1", "rouge2", "rougeL")},
            }
        ],
        "outputs_stopped_by_max_new_tokens": truncated,
        "verdict": "sample-sanity",
        "reason": (
            f"ROUGE on {len(results)} referenced item(s); tutorial evidence on these items, not a benchmark"
            + ("; one item cannot state a dispersion" if len(results) < 2 else "")
        ),
        "needs": (
            "a larger, domain-representative referenced set and repeated or bootstrapped scoring for any "
            "generalisable claim; ROUGE measures word overlap, not faithfulness"
        ),
    }


REFERENCE_SAMPLE_NAME = "SciTLDR-A test (abstract -> TLDR), first items of 200..1600 characters"
REFERENCE_SAMPLE_URL = (
    "https://raw.githubusercontent.com/allenai/scitldr/5ccad9c00a60ad75c9e04abf7f27d0f53f983b20/"
    "SciTLDR-Data/SciTLDR-A/test.jsonl"
)
REFERENCE_SAMPLE_BYTES = 1_204_107
REFERENCE_SAMPLE_SHA256 = "fb42dd6cd4f4a1928ae8a01a189456fbfe994a07e938bd49f68653933f6503c9"
REFERENCE_SAMPLE_LICENSE = "Apache-2.0 (Cachola et al. 2020; allenai/scitldr)"
REFERENCE_SAMPLE_SIZE = 20


def fetch_reference_sample(
    n_items: int = REFERENCE_SAMPLE_SIZE,
    *,
    cache_dir: str | Path = Path("weights") / "scitldr",
    fetcher: Any = None,
) -> list[dict[str, Any]]:
    """The default referenced sample: the pinned SciTLDR-A test file (refused unless its size and SHA-256
    match), its first ``n_items`` papers whose abstract has 200..1600 characters, each as
    ``{id, input: 'summarize: ' + abstract, references: [TLDR, ...]}``. No split is invented."""
    cache = Path(cache_dir)
    target = cache / "test.jsonl"
    if (
        target.is_file()
        and target.stat().st_size == REFERENCE_SAMPLE_BYTES
        and _sha256(target) == REFERENCE_SAMPLE_SHA256
    ):
        data = target.read_bytes()
    else:
        if fetcher is None:
            with urllib.request.urlopen(REFERENCE_SAMPLE_URL, timeout=60) as response:
                data = response.read(REFERENCE_SAMPLE_BYTES + 1)
        else:
            data = fetcher(REFERENCE_SAMPLE_URL)
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != REFERENCE_SAMPLE_BYTES or digest != REFERENCE_SAMPLE_SHA256:
            raise ValueError(
                f"reference sample: {len(data)} bytes / sha256 {digest[:16]}... != pinned "
                f"{REFERENCE_SAMPLE_BYTES} / {REFERENCE_SAMPLE_SHA256[:16]}...; refusing it"
            )
        cache.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    items = []
    for line in data.decode("utf-8").splitlines():
        record = json.loads(line)
        abstract = " ".join(sentence.strip() for sentence in record["source"]).strip()
        if 200 <= len(abstract) <= 1600 and record.get("target"):
            items.append(
                {
                    "id": record["paper_id"],
                    "input": "summarize: " + abstract,
                    "references": list(record["target"]),
                }
            )
        if len(items) == n_items:
            break
    return items


def read_byod_inputs(name: str, data: bytes) -> tuple[list[str], list[str] | None]:
    """BYOD reader: ``(inputs, references or None)`` from one uploaded file.

    A ``.csv`` needs an ``input`` column and may have a ``reference`` column (then every row needs one); any
    other file is one prefixed input per non-empty line. Text is decoded as UTF-8 with an optional BOM
    (``utf-8-sig``); other encodings are refused with a message naming the file."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{name}: not UTF-8 text (byte {exc.start}); save the file as UTF-8 and upload it again"
        ) from exc
    if name.lower().endswith(".csv"):
        rows = list(csv.DictReader(io.StringIO(text)))
        if not rows or "input" not in rows[0]:
            raise ValueError(
                f"{name}: a CSV needs a header with an `input` column (and optionally `reference`)"
            )
        inputs = [row["input"].strip() for row in rows]
        if any(not value for value in inputs):
            raise ValueError(f"{name}: row {inputs.index('') + 2} has an empty `input`")
        if "reference" in rows[0]:
            references = [(row.get("reference") or "").strip() for row in rows]
            if any(not value for value in references):
                raise ValueError(
                    f"{name}: row {references.index('') + 2} has an empty `reference`; every row needs one"
                )
            return inputs, references
        return inputs, None
    inputs = [line.strip() for line in text.splitlines() if line.strip()]
    if not inputs:
        raise ValueError(f"{name}: expected at least one non-empty line, each starting with its task prefix")
    return inputs, None


@dataclass
class T5SmallText2TextPipeline:
    """``_runner(text, max_new_tokens, num_beams)`` -> ``(generated_text, generated_tokens, stopped_by)``;
    ``_count_tokens(text)`` -> encoder token count incl. EOS. Both injectable so tests run offline."""

    _runner: Callable[[str, int, int], tuple[str, int, str]]
    _count_tokens: Callable[[str], int]
    device: str = "cpu"
    source: str = "injected"

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> T5SmallText2TextPipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            location, kwargs, source = str(root), dict(local_files_only=True), "local-snapshot"
        elif allow_download:
            location, kwargs, source = MODEL_ID, dict(revision=MODEL_REVISION), "hf-hub"
        else:
            raise FileNotFoundError(f"no verified snapshot at {root} and allow_download=False")
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import T5ForConditionalGeneration, T5TokenizerFast

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        tokenizer = T5TokenizerFast.from_pretrained(location, trust_remote_code=False, **kwargs)
        model = T5ForConditionalGeneration.from_pretrained(
            location, dtype=torch.float32, trust_remote_code=False, **kwargs
        )
        model = model.to(resolved_device).eval()
        eos_id, pad_id = model.config.eos_token_id, model.config.pad_token_id

        def count_tokens(text: str) -> int:
            return len(tokenizer(text, truncation=False)["input_ids"])

        def runner(text: str, max_new_tokens: int, num_beams: int) -> tuple[str, int, str]:
            enc = tokenizer(text, return_tensors="pt", truncation=False).to(resolved_device)
            with torch.inference_mode():
                out = model.generate(
                    **enc, max_new_tokens=max_new_tokens, num_beams=num_beams, do_sample=False
                )
            ids = out[0].tolist()
            content = [t for t in ids if t not in (eos_id, pad_id)]
            stopped_by = "eos" if eos_id in ids else "max_new_tokens"
            return tokenizer.decode(content, skip_special_tokens=True), len(content), stopped_by

        return cls(runner, count_tokens, resolved_device, source)

    def _validate(self, text: Any, max_new_tokens: Any, num_beams: Any) -> int:
        text = _check_inputs(text, max_new_tokens, num_beams)
        return _check_input_tokens(self._count_tokens(text))

    def generate(
        self, text: str, *, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, num_beams: int = 1
    ) -> dict[str, Any]:
        """Run one prefixed input through encoder-decoder generation; the caller owns the task prefix."""
        n_input = self._validate(text, max_new_tokens, num_beams)
        generated, n_generated, stopped_by = self._runner(text, max_new_tokens, num_beams)
        if not isinstance(generated, str) or not isinstance(n_generated, int):
            raise RuntimeError("runner must return (str, int, str)")
        prefix = known_prefix(text)
        return {
            "text": generated,
            "generated_tokens": n_generated,
            "input_tokens": n_input,
            "stopped_by": stopped_by,
            "known_prefix": prefix,
            "generation": {
                "max_new_tokens": max_new_tokens,
                "num_beams": num_beams,
                "do_sample": False,
                "decision_rule": DECISION_RULE,
            },
            "device": self.device,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
