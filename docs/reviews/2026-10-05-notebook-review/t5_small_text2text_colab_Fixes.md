# T5-Small text-to-text notebook — review fixes

**Review:** `t5_small_text2text_colab_Review.md` (5 October 2026, T5S-M1..M3, T5S-m1..m2).
**Fixed in:** the generator (`tools/build_notebook.py`, now the isolated-runtime generator `build_notebook.py/2.1-swc`; `tools/notebook_template.py`), the carried module (`pipeline.py`, `__init__.py`), the new hash lock `tutorials/requirements-colab.lock.txt`, the validator, `tutorials/README.md`, `README.md`, `MODEL_CARD.md`, `docs/release-verification.md` and the tests. The notebook was regenerated and `--check` passes.
**Readiness:** **Verification pending** until the hosted gates below are recorded. `STATUS.md` and every release label are unchanged.

## Findings

| ID | Status | Change | Cells / files | Evidence |
|---|---|---|---|---|
| T5S-M1 | Fixed (summarisation); translation metric not added | **Scoring.** `evaluation_report` scores whenever references are given, one or several per output: ROUGE-1/2/L F1 (best reference per item, in percent) beside a trivial baseline formed from each input — Lead-1 for `summarize: `, copy-source for `translate …`. The verdict is `sample-sanity` and the report lists the outputs that stopped at `max_new_tokens`. Without references it stays `not-measurable`. The scorer is the sibling t5-base recipe without stemming; tokens are Unicode letter/digit runs, so German, French and Romanian words stay whole.<br>**Referenced sample.** `fetch_reference_sample()`: the SciTLDR-A test file at a pinned upstream commit (1,204,107 bytes, SHA-256 `fb42dd6c…`, refused on mismatch, cached), first 20 abstracts of 200–1600 characters behind `summarize: `.<br>**Section 7** now generates all 20 and writes `evaluation_report.json` with verdict `sample-sanity` beside Lead-1 (27.22 / 11.12 / 22.22, computed offline). The authored inputs get `inputs_evaluation_report.json` (`not-measurable`). The prose explains what ROUGE on 20 out-of-domain items can and cannot show. The learning objectives, exclusions, Prerequisites, interpretation, README files and model card were rewritten to match.<br>**Not done:** BLEU/chrF and a referenced translation sample. Translations are scored with ROUGE only, when BYOD references are given. | `pipeline.py`; Sections 4, 7, 8; header; closing; validator markers | `test_t5s_M1_*` (4 tests: digest refusal and parsing, scoring beside Lead-1, the Section 7 cells run on a 20-item stand-in sample, and the Lead-1 figures checked against the real file when it is cached locally — run here, passed) |
| T5S-M2 | Fixed | The in-kernel `pip` install and restart guard are replaced by the generator's isolated runtime: a pinned `uv` 0.12.15 wheel (checked by size and SHA-256), managed CPython 3.12.12, and the carried hash lock (48 packages, `--require-hashes --only-binary :all:`, compiled from the `pyproject.toml` pins). A router sends every later cell to one persistent worker in that environment. The notebook declares spec 2.2. The validator now requires exactly two kernel cells and forbids kernel `pip` and any restart instruction. | Sections 1–3 (generator); validator; `tutorials/README.md` | `test_t5s_M2_*`; `grep "Restart the runtime"` on the notebook returns nothing |
| T5S-M3 | Fixed | The notebook now has:<ul><li>How to use, with the audience statement, cell kinds, form controls, section tags and the predict/check convention</li><li>an Input → Model/System → Output table and a roadmap</li><li>a glossary covering prefix, SentencePiece, greedy/beam, `stopped_by`, references, ROUGE, Lead-1, copy-source and the verdicts</li><li>Predict before running prompts in Sections 5, 6 and 7, each with What to notice and a collapsed Check your reasoning</li><li>a Predict → Change → Run → Observe → Explain activity using `NUM_BEAMS = 4`</li><li>a troubleshooting table and a conclusion template</li></ul>Sections 1–3 are labelled Infrastructure and collapsed (`cellView: form`). | `guided_opening`, stage markdown, closing | `test_t5s_M3_*` (12 tests) |
| T5S-m1 | Fixed | The Section 4 cell gains `BYOD_PATH`, so BYOD works by path in any Jupyter runtime; the upload dialog must return exactly one file. `read_byod_inputs` decodes `utf-8-sig`, so a BOM no longer hides the first prefix, and refuses other encodings with a message naming the file. A `.csv` may carry `input,reference` columns and its outputs are then scored. Troubleshooting rows cover each case. | `read_byod_inputs`; Section 4; Troubleshooting | `test_t5s_m1_*` (5 tests: BOM/Latin-1, CSV, the notebook's own Section 4–6 cells run by path, cancelled or two-file upload) |
| T5S-m2 | Fixed (record); `STATUS.md` left | `docs/release-verification.md`: *Current status* now describes the recorded Kaggle run, which executed the standalone carrier end to end, and lists what that run did not record (restart status, versions, outputs). It says that no run of the regenerated notebook exists. The run row names those gaps and the earlier blob's install method; the procedure asks for `restarted: false` and the new outputs. The spec is aligned to 2.2 in the record, `tutorials/README.md` and `README.md`. `STATUS.md` still cites 1.1 because status files are out of scope for this cycle. | `docs/release-verification.md`, `README.md`, `tutorials/README.md` | `test_t5s_m2_release_record_is_consistent` |

The other suggestions are not taken in this cycle. S2 was partly applied: the Prerequisites now quote the 215.2 s Kaggle figure and label it as coming from the earlier notebook. S3 was applied: spec 2.2 is declared.

## User-visible changes

- Section 1 builds an isolated environment instead of installing into the kernel. It needs PyPI and the managed CPython download, and Linux x86_64 only. The lock carries the Linux (CUDA) build of `torch`, which is a larger download than before. Every later cell runs in the isolated worker.
- Section 4 has a new form field, `BYOD_PATH`. BYOD accepts `.csv` with references, and an upload of zero or several files stops with a message.
- Section 7 downloads 1.2 MB from `raw.githubusercontent.com` (pinned commit) and runs 20 more generations.
- New outputs: `outputs/t5_small_text2text_inputs_evaluation_report.json` and `outputs/t5_small_text2text_reference_generations.csv`.
- `evaluation_report.json` now holds the referenced-sample report (`sample-sanity`, metrics, Lead-1 baseline, per-item scores) instead of an always-`not-measurable` report on the first output.
- `result.json` gains `inputs_evaluation_report` and `reference_generations_file`; `sample` gains `references`.
- The `evaluation_report(result, references)` signature also accepts a list of results. Supplying references now produces a score rather than a `not-measurable` reason.

## Verification (offline, not clean-runtime evidence)

- `python tools/build_notebook.py --check`: OK. `python tools/validate_release_assets.py`: PASS. `ruff check src tests tools`: clean.
- `pytest` with CI's lightweight dependencies and **no torch**: 31 passed before; **54 passed / 1 skipped** after. The skip is the real-file Lead-1 check, which needs the downloaded sample in `weights/scitldr/`; it was run here with the file in place and passed. CI installs torch; nothing in the new tests needs it.
- **Real data:** the pinned SciTLDR-A file was downloaded and accepted by size and digest. The Lead-1 baseline on the 20 items is ROUGE-1 27.22, ROUGE-2 11.12, ROUGE-L 22.22, the figures quoted in the notebook.
- **Notebook cells:** the Section 4–7 learner cells were executed from the generated notebook with a stand-in pipeline (`tests/test_review_fixes.py`). They prove the plumbing, not model behaviour.
- **Not run:**
  - no T5-small generation: the Hugging Face Hub is unreachable from this container and torch is not installed, so no model score is quoted;
  - the isolated install cell was not executed here for this repository; it is the same generator code verified in the swin-classification fix.

## Remaining gates

1. A hosted one-pass **Run all** of the regenerated notebook on a fresh runtime, with no restart and `restarted: false` recorded, plus the runtime versions, the three model ROUGE scores and the count of outputs stopped by `max_new_tokens`. Then re-run the export cell.
2. The REL12 BYOD journey on a hosted runtime: a `.txt` and a `.csv` with references, one by path and one through the upload dialog, and one refused file (Latin-1).
3. Maintainer decisions: align `STATUS.md` (spec citation and status text); decide whether a referenced translation sample with chrF/BLEU should be added.
