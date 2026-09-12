# T5-Small Text2Text Pipeline

DIMER-oriented inference wrapper for **google-t5/t5-small** (the original 60 M-parameter T5, not FLAN), pinned to an immutable Hugging Face revision. The repository exposes prefixed text-to-text generation — the caller supplies `summarize: `, `translate English to German: `, `translate English to French: ` or `translate English to Romanian: ` — with deterministic greedy decoding by default, a supply-chain check of the local weight snapshot, and machine-readable provenance.

## Upstream alignment

- Model: `google-t5/t5-small`
- Revision: `df1b051c49625cf57a3d0d8d3863ed4d13564fe4`
- Upstream weight license: Apache-2.0
- Upstream task: text-to-text generation (summarisation and En→De/Fr/Ro translation via task prefixes; pre-trained on C4 plus a supervised multi-task mixture)
- Repository adaptation: **none**; inference only

## Quick start

```python
from t5_small_text2text_pipeline import T5SmallText2TextPipeline

pipe = T5SmallText2TextPipeline.from_pretrained()          # verifies weights/t5-small first
result = pipe.generate("translate English to German: The house is wonderful.", max_new_tokens=32)
print(result["text"])            # 'Das Haus ist wunderbar.'  (CPU smoke, greedy)
print(result["generated_tokens"], result["stopped_by"], result["known_prefix"])
```

`generate(text, *, max_new_tokens=64, num_beams=1)` takes one non-empty string of at most 20,000 characters (`MAX_TEXT_CHARS`) that tokenises to at most 512 SentencePiece tokens (`MAX_INPUT_TOKENS`; longer inputs are rejected, not truncated), `max_new_tokens` in 1..512 (`MAX_NEW_TOKENS`) and `num_beams` in 1..8 (`MAX_NUM_BEAMS`). The pipeline never adds a prefix; `TASK_PREFIXES` lists the four the checkpoint was trained on and each result reports which one the input started with as `known_prefix` (or `None`). Upstream `task_specific_params` (min_length, length_penalty, no_repeat_ngram_size) are not applied. Every result carries `text`, `generated_tokens`, `input_tokens`, `stopped_by`, `generation` settings, `device`, `source`, `model_id` and `model_revision`. No metric helper ships: ROUGE/BLEU need reference outputs the caller must supply.

## Weights layout

```
weights/t5-small/
  config.json  generation_config.json  model.safetensors  spiece.model  tokenizer.json
  tokenizer_config.json  README.md  dimer-base-manifest.json
```

`from_pretrained()` calls `stage_missing_files()` (fetches absent manifest entries at the pinned revision, only with `allow_download=True`) then `verify_snapshot()` (size + SHA-256 of every entry), and loads `T5ForConditionalGeneration` + `T5TokenizerFast` with `local_files_only=True` and `trust_remote_code=False`. Without a manifest it raises unless `allow_download=True`. See `docs/WEIGHTS.md`.

## Tests

```
pip install -e . --no-deps
pytest -q -o addopts= tests
```

Tests are offline: they use an injected fake runner and token counter plus temporary manifests, never the weights.

## Tutorial

None yet. This card pass ships the pipeline package, tests and `MODEL_CARD.md`; a `NOTEBOOK_SPEC` 1.0 tutorial notebook is a later pass.

## Release status

**Candidate / source-complete.** Unit tests, one executed CPU smoke run from the pinned snapshot, and the static card gate exist; no notebook and no clean-runtime notebook evidence. See `STATUS.md`.

## Licensing

This repository's code is Apache-2.0 (`LICENSE`). The packaged upstream weights are Apache-2.0; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.
