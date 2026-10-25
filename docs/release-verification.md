# Release verification

`tutorials/t5_small_text2text_colab.ipynb` (`TASK-INFERENCE`) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests,
JSON validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary
checks but are **not** runtime evidence under DIMER Notebook Specification 2.5, which the notebook declares (revisions from the 2026-10-05 review fixes until the tier C review fixes declared 2.2; earlier ones declared 2.0). This file is
the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the primary
  path; exactly one cell tagged `embedded_module` equal to `src/t5_small_text2text_pipeline/pipeline.py` after the generator's documented
  rewrites; the inline `MANIFEST` equal to the committed 7-entry snapshot manifest and the inline `PINS` equal to
  the `pyproject.toml` runtime pins; the notebook byte-identical to `tools/build_notebook.py` output; exactly two kernel cells (the isolated `uv` install — pinned `uv` wheel by digest, managed CPython, the carried hash lock `tutorials/requirements-colab.lock.txt` with `--require-hashes --only-binary :all:` — and the router), no kernel `pip` and no restart instruction; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision is
  a 40-hex immutable commit, and the same identity string appears in `README.md`,
  `MODEL_CARD.md`, and `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `T5SmallText2TextPipeline.from_pretrained(weights_dir=...)`, `validate_inputs`, `generate(text, max_new_tokens=..., num_beams=...)`, `evaluation_report`), the ceiling print
  (`MAX_TEXT_CHARS`, `MAX_INPUT_TOKENS`, `MAX_NEW_TOKENS`, `MAX_NUM_BEAMS`, `DEFAULT_MAX_NEW_TOKENS`, `DECISION_RULE`, `TASK_PREFIXES`), the exports, the learner-facing statements (original T5 not FLAN, the pipeline invents no prefix, greedy argmax as the default decision rule, no probability or score emitted, ROUGE-1/2/L beside the Lead-1 baseline on the pinned referenced sample, the guided-layer markers, inputs above the token ceiling rejected not truncated, named exclusions) and the
  gated-off BYOD default listed in the validator; forbidden patterns (credential-in-URL, any `git clone` /
  `github.com` / repository import on the primary path, a mutable `revision='main'`, direct
  `from transformers import` / `T5ForConditionalGeneration` / `T5TokenizerFast` / `from huggingface_hub import` /
  `model.generate(` use **outside the carried module cell**,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, required heading order, and
  immutable provenance.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `tokenizers`, `sentencepiece`, `huggingface-hub`, `safetensors` and `numpy`, runs `ruff`, `tools/build_notebook.py --check`, and the
offline unit suite (`tests/test_pipeline.py`, `tests/test_role_helpers.py`, `tests/test_notebook_parity.py`; injected runner and token counter, no weights). These are
source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle CPU kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (no repository checkout is needed — the notebook is standalone) |
| Local Windows-venv harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, `CUDA_VISIBLE_DEVICES=-1` | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU (or CUDA) runtime (Colab, or the Kaggle
   executor above) with **no repository checkout** and a clean model cache;
3. run the notebook top-to-bottom once, with **no runtime restart**, without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`, `BYOD_PATH = ''`, `GEN_MAX_NEW_TOKENS = 64`, `NUM_BEAMS = 1`), and record `restarted: false`;
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`);
5. verify every default-path stage completes:
   - the isolated environment built from the carried hash lock (Section 1 prints the isolated Python 3.12.12 and the kernel's), with no GitHub access;
   - the carried module cell executing (defining `T5SmallText2TextPipeline`, `validate_inputs`, `evaluation_report` and the ceilings) with no import of the repository package;
   - the two synthetic prefixed inputs authored in code with their text SHA-256 printed, and the ceilings (`MAX_TEXT_CHARS` 20000, `MAX_INPUT_TOKENS` 512, `MAX_NEW_TOKENS` 512, `MAX_NUM_BEAMS` 8, `DEFAULT_MAX_NEW_TOKENS` 64), `DECISION_RULE` and the four `TASK_PREFIXES` surfaced, and `validate_inputs` writing `outputs/t5_small_text2text_input_manifest.json` (verdict `accepted`, both inputs recognised by a known prefix, one recorded rejection finding from the `num_beams` ceiling probe);
   - pinned `google-t5/t5-small` acquisition at the immutable revision through the carried module, the inline `MANIFEST` asserted against the module identity and written to `weights/t5-small/`:
     `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports `['model.safetensors']` on a
     clean runtime, `verify_snapshot` returns its dict (7 files), and `from_pretrained(weights_dir=WEIGHTS_DIR)`
     loads from the verified directory with `source` `local-snapshot`;
   - `generate` returning, per input, `text`, `input_tokens`, `generated_tokens`, `stopped_by`, `known_prefix` and the echoed `generation` settings, with every sanity check `True`; record the two outputs, token counts and `stopped_by` values (the card-pass CPU smoke returned `Das Haus ist wunderbar.` for the translation, 11 → 5 tokens stopped by `eos`; a different output on another runtime is a finding to record, not a failure by itself, because no metric is asserted);
   - `fetch_reference_sample()` accepting the pinned SciTLDR-A file (1,204,107 bytes, SHA-256 `fb42dd6c…`) and returning 20 items; `evaluation_report` writing `outputs/t5_small_text2text_evaluation_report.json` with verdict `sample-sanity`, three ROUGE metrics and the Lead-1 baseline (27.22 / 11.12 / 22.22); record the three model scores and the count of outputs stopped by `max_new_tokens`;
   - `outputs/t5_small_text2text_inputs_evaluation_report.json` with verdict `not-measurable` for the authored inputs;
   - `outputs/t5_small_text2text_result.json`, `outputs/t5_small_text2text_generations.csv` and `outputs/t5_small_text2text_reference_generations.csv` written with `NOTEBOOK_SOURCE`, model revision,
     model licence, runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device),
   model identifier and immutable revision, whether the model cache was clean, outcome, produced
   outputs, and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/t5_small_text2text_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/t5_small_text2text_colab.ipynb`). Wall times, when recorded,
are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `e77fa79` / `caf0327ec98f` | Kaggle CPU (`kurtvalcorza/dimer-nb2-t5-small-text2text` v1) | Default sample path | 215.2 s | **PASSED** — 8/8 ok code cells executed cleanly, 16 files, 244 MB staged. Not recorded: restart status, runtime versions, the generated outputs. This was the earlier notebook (kernel `pip` install with a restart guard, no referenced sample, spec 2.0). |
| 2026-10-09 | `77f629f` / `7cdec4b16f48` | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 (session `suite-t5-77f629f-daab`) — not a browser Run all | Default sample path (`USE_BYOD = False`, `GEN_MAX_NEW_TOKENS = 64`, `NUM_BEAMS = 1`) | 112.2 s | **PASSED** — 10/10 code cells in order, 0 error outputs, `restarted: false` (no output asks for a restart; the carried-module cell is silent by design). Isolated env: 48 locked packages, CPython 3.12.12 (kernel 3.13.15), setup 65 s; torch 2.14.0+cu130, transformers 4.57.6, `cuda:0`, model `google-t5/t5-small` @ `df1b051c4962` staged clean (7 files, 244,236,215 bytes, `source` `local-snapshot`). Authored inputs: En→De `Das Haus ist wunderbar.` (11 → 5 tokens, `eos`); summary 127 → 43 tokens, `eos`; input manifest `accepted` with the `num_beams` probe rejected. SciTLDR-A 20 items: ROUGE-1/2/L 28.62 / 10.72 / 22.51 vs Lead-1 27.22 / 11.12 / 22.22, verdict `sample-sanity`, 0 stopped by `max_new_tokens`; authored inputs `not-measurable`; six exports written. Evidence: `docs/execution-evidence/2026-10-09-77f629f/` (executed notebook SHA-256 `766adec5050d…`). Not exercised: REL12 BYOD, the `NUM_BEAMS = 4` activity. |
| 2026-10-25 (07:34:08 UTC start) | `a353e77` (PR #8 head) / `e82d281c6e85` (tier C review fixes) | Colab CLI sequential execution (`colab exec -f`, not a browser Run all), fresh Colab Tesla T4 (session `suite-t5-a353e77-daab`) | Default sample path (`USE_BYOD = False`, `GEN_MAX_NEW_TOKENS = 64`, `NUM_BEAMS = 1`) | 96.3 s | **PASSED** — 10/10 code cells in order in `exec.log`, one pass, no restart, 0 error outputs, input notebook unchanged; isolated env 48 locked packages, CPython 3.12.12 (kernel 3.13.16), setup 50 s; `cuda:0`, `source` `local-snapshot`. Authored inputs: En→De `Das Haus ist wunderbar.` (11 → 5 tokens, `eos`); summary 127 → 43 tokens, `eos`; input manifest `accepted` with the `num_beams` probe rejected. SciTLDR-A 20 items: ROUGE-1/2/L 28.62 / 10.72 / 22.51 vs Lead-1 27.22 / 11.12 / 22.22, verdict `sample-sanity`, 0 stopped by `max_new_tokens`; authored inputs `not-measurable`. Every metric equals the 2026-10-09 run and the numbers the notebook quotes. Not exercised: REL12 BYOD, the `NUM_BEAMS = 4` activity. Evidence: `docs/execution-evidence/2026-10-25-a353e77/` — SHA-256 captured → committed: `t5_small_text2text_colab_a353e77_colab-cli-t4_default.ipynb` `bba5329becdb39cd1b62bc47a337e3ac501fd6b669ef0f6fead8cf2f80086afb` → unchanged (no local path); `exec.log` `5324d9b34856ad8e09a92b5d3c8498df5702dac8fbb2970f7ee42fc5b3cc8868` → `8b52354641a1cfb0a37e0832ee1e2d68e3c705107fc8940189690e2b0bf5f493` (2 local-path replacements); `run_summary.json` `17584f95bbecd582e02ac302d2f853a08a2cce5eab82ec6f13f371c345bc9807` → unchanged (no local path). |
| 2026-10-25 (12:19:28 UTC start) | `524353b` (PR #8 head) / `622b9ef3675c` (prose kept true across runs) | Colab CLI sequential execution (`colab exec -f`, not a browser Run all), fresh Colab Tesla T4 | Default sample path (`USE_BYOD = False`, `GEN_MAX_NEW_TOKENS = 64`, `NUM_BEAMS = 1`) | 107.9 s | **PASSED** — 10/10 code cells in order in `exec.log`, one pass, no restart, 0 error outputs, input notebook unchanged (git blob `622b9ef3675c14c59319e5a93245ba8f314ab52d` at full SHA `524353b93388890d0c726d0a78af3ad4de9f2c8a`; captured executed-notebook SHA-256 equals `run_summary.json` `executed_sha256`); isolated env 48 locked packages, CPython 3.12.12 (kernel 3.13.16), setup 52 s; `cuda:0`, `source` `local-snapshot`. Authored inputs: En→De `Das Haus ist wunderbar.` (11 → 5 tokens, `eos`); summary 127 → 43 tokens, `eos`; input manifest `accepted` with the `num_beams` probe rejected. SciTLDR-A 20 items: ROUGE-1/2/L 28.62 / 10.72 / 22.51 vs Lead-1 27.22 / 11.12 / 22.22, verdict `sample-sanity`, 0 stopped by `max_new_tokens`; authored inputs `not-measurable`. Every metric equals the 2026-10-09 and the earlier 2026-10-25 runs, and every run-derived number the notebook prose quotes (Lead-1 27.22 / 11.12 / 22.22, ROUGE-1 28.62, about 2 minutes on a Colab T4) holds for this run. Not exercised: REL12 BYOD, the `NUM_BEAMS = 4` activity. Evidence: `docs/execution-evidence/2026-10-25-524353b/` — SHA-256 captured → committed: `t5_small_text2text_colab_524353b_colab-cli-t4_default.ipynb` `3a8ca2351e13dcc1d211ad4c6355d5181bdec5892f43ecabe33aacbafe17777d` → unchanged (no local path); `exec.log` `cc22c21167dc104cb6b4ba6be8568c12bb212e6a911d08f7e6c4518649b3a367` → `00731537cfb11bebf86f0bf637b722a81a80983522a1872e675698f4111c985b` (2 local-path replacements); `run_summary.json` `3c7f7852fb62e77cfc6cfef761b1ab54d520f7b541b0647390d5f416facba34e` → unchanged (no local path). |

## Current status

**Candidate (2026-10-26 record).** The current blob is `tutorials/t5_small_text2text_colab.ipynb` `622b9ef3675c14c59319e5a93245ba8f314ab52d` (commit `524353b`, regenerated at `59c6f81`; markdown only: the wall times are now approximate, about 2 minutes on a Colab T4). That exact blob passed a Colab CLI sequential execution on a fresh Colab Tesla T4 on 2026-10-25 (10/10 code cells, one pass, no restart, 0 errors, 107.9 s; ROUGE-1/2/L 28.62 / 10.72 / 22.51 vs Lead-1 27.22 / 11.12 / 22.22, verdict `sample-sanity`; see the 12:19:28 UTC row under "Recorded executions"; evidence in `docs/execution-evidence/2026-10-25-524353b/`). This covers the **default path** only. Remaining gates: REL12 (a BYOD run, one input accepted and one refused) and REL2 (a browser run of the interactive journeys). The runs of earlier blobs, including the 2026-10-25 run of `e82d281c` described next, are history (REL14).

**Previous blob (history, REL14).** `tutorials/t5_small_text2text_colab.ipynb` blob `e82d281c6e8576eff4e46ec5ed8bbb1858da9474` (PR #8 head `a353e77`, tier C review fixes: markdown, the use notice, `metadata.dimer` and the `NOTEBOOK_SOURCE` record only) passed a Colab CLI sequential execution on a fresh Colab Tesla T4 at `a353e77` on 2026-10-25 (10/10 code cells, one pass, no restart, 0 errors, 96.3 s; see the 2026-10-25 row under "Recorded executions"; evidence in `docs/execution-evidence/2026-10-25-a353e77/`). This covers the **default path** only. Remaining gates: REL12 (a BYOD run, one input accepted and one refused) and REL2 (a browser run of the interactive journeys). Under REL14 the runs of earlier blobs recorded here are history for those blobs.

One clean-room run is recorded (the row above): the earlier notebook blob `caf0327ec98f` ran its default path on a Kaggle CPU kernel, which executed the standalone carrier — the carried module cell, the real `stage_missing_files` fetch of `model.safetensors` from the Hub, `verify_snapshot` and CPU generation — end to end without the repository. That run did not record its restart status, runtime versions or outputs.

The 2026-10-05 review fixes (`docs/reviews/2026-10-05-notebook-review/t5_small_text2text_colab_Fixes.md`) changed the notebook: an isolated, hash-locked `uv` environment replaces the in-kernel install, the default path now measures on a pinned referenced sample, BYOD accepts a path and references, and the guided layer was added. The regenerated notebook was then run on a hosted runtime: the review-fix blob `7cdec4b16f48` (commit `77f629f`; isolated `uv` environment, referenced SciTLDR-A sample, guided layer, T5S-M1..M3 / T5S-m1..m2 fixes) completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-09 (Colab CLI 0.7.4 sequential execution, 10/10 code cells, 112.2 s; isolated Python 3.12.12, torch 2.14.0+cu130, transformers 4.57.6, `cuda:0`; ROUGE-1/2/L 28.62 / 10.72 / 22.51 beside Lead-1 27.22 / 11.12 / 22.22 on the 20-item sample, verdict `sample-sanity`, 0 outputs stopped by `max_new_tokens`). That run used the CUDA path; the CPU path of the regenerated notebook has not been hosted-run. REL12 BYOD exercise (path-based and upload, `.txt` and `.csv` with references, a refused Latin-1 file) and the optional `NUM_BEAMS = 4` activity are pending on a hosted runtime, so the status stays Candidate (REL14). Promotion is not performed by the builder. `STATUS.md` now cites Notebook Specification 2.2.
