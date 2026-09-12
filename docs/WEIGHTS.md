# Weight provenance and DIMER hosting

- Upstream: `google-t5/t5-small`
- Immutable revision: `df1b051c49625cf57a3d0d8d3863ed4d13564fe4`
- Weight format: SafeTensors (`model.safetensors`, 242,043,056 bytes, float32)
- Upstream weight license: Apache-2.0 (`license: apache-2.0` in the pinned upstream README front matter)
- Local layout: `weights/t5-small/` holds the 7 files listed in `dimer-base-manifest.json` (`config.json`, `generation_config.json`, `model.safetensors`, `spiece.model`, `tokenizer.json`, `tokenizer_config.json`, upstream `README.md`; `totalBytes` 244,236,215) with byte size and SHA-256 for each. `verify_snapshot()` in `src/t5_small_text2text_pipeline/pipeline.py` checks all of them before any load; `stage_missing_files(allow_download=True)` fetches only absent entries at the pinned revision into that directory. `.safetensors` files are git-ignored; the Git repository does not vendor the checkpoint.
- DIMER hosting: Apache-2.0 permits use, modification, redistribution and commercial use subject to preservation of the license and notices. DIMER may mirror the pinned checkpoint in its model store under those terms.
- Loader trust boundary: Transformers `T5ForConditionalGeneration` + `T5TokenizerFast` with `trust_remote_code=False`; the loader reads only the verified local directory (`local_files_only=True`) and falls back to the Hub at the pinned revision only when `allow_download=True` is passed explicitly.
