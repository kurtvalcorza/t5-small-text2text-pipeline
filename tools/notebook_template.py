"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

REPO = "t5-small-text2text-pipeline"

TEMPLATE = {
    "package": "t5_small_text2text_pipeline",
    "repo_name": REPO,
    "stem": "t5_small_text2text",
    "notebook_name": "t5_small_text2text_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    # NOTEBOOK_SPEC 2.2 §5 (T5S-M2): the pins are installed into an isolated uv environment and every later cell runs in
    # a persistent worker there, so a hosted runtime's preloaded packages never force a restart. The lock is compiled with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "isolated_runtime": True,
    "infrastructure_labels": True,
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime (CPU is enough) builds an isolated Python environment from the "
        "hash-locked pins (the kernel's own packages are left alone, so no restart is needed; a second Run all reuses it), "
        "stages and digest-verifies the pinned snapshot, authors the two-input synthetic sample, validates it into an input "
        "manifest before the model runs, generates through the carried pipeline, then **measures** the model on a pinned "
        "referenced sample — 20 SciTLDR-A abstracts with their human-written one-sentence summaries — with ROUGE-1/2/L beside "
        "the Lead-1 baseline, writes the evaluation reports, and exports machine-readable outputs with provenance. No "
        "repository clone, DIMER worker or service, credential, upload dialog, configuration edit or runtime restart is "
        "required (NOTEBOOK_SPEC 2.2 §5). The one recorded run (Kaggle CPU, 14 September 2026, 215.2 s) was of the earlier "
        "notebook, which installed into the kernel and had no referenced sample; no run of this revision is recorded yet."
    ),
    "byod": (
        "Optional and off by default: set `USE_BYOD = True` in Section 4 and either give a file path in `BYOD_PATH` or "
        "upload exactly one file. A `.txt` file holds one already-prefixed input per line; a `.csv` file has an `input` "
        "column and may have a `reference` column, and then Section 7 scores your outputs against your references with "
        "the same ROUGE and baseline as the default sample. Files must be UTF-8 (a byte-order mark is accepted). The upload "
        "stays inside this runtime."
    ),
    "pipeline_class": "T5SmallText2TextPipeline",
    "weights_key": "t5-small",
    "runtime_imports": ["torch", "transformers"],
    "title": "T5-Small — DIMER text-to-text generation tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            f"https://github.com/kurtvalcorza/{REPO}",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            f"https://colab.research.google.com/github/kurtvalcorza/{REPO}/blob/main/tutorials/t5_small_text2text_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google--t5%2Ft5--small-ffcc4d?style=flat",
            "https://huggingface.co/google-t5/t5-small",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-google--research%2Ftext--to--text--transfer--transformer-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/google-research/text-to-text-transfer-transformer",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-1910.10683-b31b1b.svg", "https://arxiv.org/abs/1910.10683"),
    ],
    "capability": "caller-prefixed text-to-text generation (summarisation and English→German/French/Romanian translation) using the pinned `google-t5/t5-small` weights",
    "intro": (
        "`google-t5/t5-small` is the original 60 M-parameter T5 of Raffel et al. (2020) — **not FLAN-T5**, so it follows "
        "only the task prefixes it was trained on, not free-form instructions. At inference the encoder reads the whole "
        "prefixed input once and the decoder emits one SentencePiece token per step until `</s>` or a step ceiling; "
        "**greedy decoding** (per-step argmax, `num_beams=1`, `do_sample=False`) is the default decision rule and beam "
        "search is available on request — there is no sampling and no seed. The caller writes the task prefix — "
        "`summarize: `, `translate English to German: `, `translate English to French: ` or `translate English to "
        "Romanian: ` — in front of the text: **the pipeline invents no prefix** and only reports which known one an "
        "input started with (`known_prefix`, or `None`). **No adaptation occurs:** no training, fine-tuning, in-context "
        "conditioning, or preprocessing fitting — the pinned checkpoint is used as published. What the upstream "
        "checkpoint supplies is the model, the SentencePiece tokenizer and the prefix convention; what the carried "
        "pipeline module adds is manifest verification, input validation with named ceilings, a fixed output contract, "
        "the list of trained prefixes as `TASK_PREFIXES`, the `validate_inputs` and `evaluation_report` stage helpers, "
        "a ROUGE-1/2/L scorer with a trivial baseline (Lead-1 for summaries, copy-source for translations), a "
        "digest-pinned referenced sample (`fetch_reference_sample`) and a BYOD reader that accepts references "
        "(`read_byod_inputs`)."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, author two already-prefixed "
        "inputs (or bring your own, with or without references), stage and digest-verify the immutable upstream snapshot, "
        "surface the pipeline's ceilings and the decision rule and validate the inputs into an input manifest before any "
        "model work, generate through the public API with explicit `max_new_tokens`/`num_beams`, read `stopped_by`, "
        "`known_prefix` and the token counts correctly, **measure** summaries on a pinned referenced sample with "
        "ROUGE-1/2/L against a Lead-1 baseline and say what that number can and cannot show, read why the authored inputs "
        "are `not-measurable`, and export every generation with its identifier plus provenance."
    ),
    "exclusions": (
        "instruction following or chat, sampling-based decoding, batching, classification or embedding (the GLUE tasks "
        "in the training mixture are not exposed), source languages other than English or target languages other than "
        "German, French and Romanian, the upstream `task_specific_params` decoding settings (`min_length`, "
        "`length_penalty`, `no_repeat_ngram_size` are not applied by the pipeline), a benchmark-scale evaluation, a "
        "translation metric such as BLEU or chrF, or a referenced translation sample (translations are scored only when "
        "you bring references). The repository exposes none of these."
    ),
    "guided_opening": [
        (
            "## How to use this notebook\n\n"
            "**Who this notebook is for.** Learners who can open a hosted notebook (Google Colab, Kaggle or Jupyter), run cells in "
            "order and read short Python, and who want to see how a pretrained text-to-text model is driven safely and how its "
            "output is measured honestly. No prior experience with T5 is assumed: each term is explained where it is first needed "
            "and again in the glossary below. A CPU runtime is enough. The **Prerequisites** give the details.\n\n"
            "**Running it.** Choose *Runtime → Run all*. The default path needs no edit, no upload, no account, no token and no "
            "runtime restart. Section 1 builds the isolated environment (the pinned `torch` is the largest download, so it is the "
            "slowest step); a second Run all in the same runtime reuses it. You can also run the notebook one cell at a time with "
            "*Shift + Enter*.\n\n"
            "**Where the code runs.** The first two code cells run in the notebook kernel: they build the environment and start "
            "one Python process inside it. Every later cell is sent to that process, so the pinned `torch` and `transformers` are "
            "used without replacing anything the hosted runtime had already loaded. Printed output and errors come back to the "
            "notebook as usual, and variables persist from cell to cell.\n\n"
            "**Two kinds of cell.** *Infrastructure cells* (Sections 1–3: the isolated install and router, the carried pipeline "
            "module and the pinned-model staging) are collapsed and labelled **Infrastructure**; you may run them without "
            "studying their implementation. *Learner cells* (Sections 4–8) are the workflow.\n\n"
            "**Form controls.** The Section 4 cell starts with fields Colab renders as a form: `USE_BYOD`, `BYOD_PATH`, "
            "`GEN_MAX_NEW_TOKENS` and `NUM_BEAMS`. Leave them at their defaults for the first run: the notes and sample answers "
            "describe the default path.\n\n"
            "**Section tags.** Each learner heading carries one tag. **[Concept]** — what the model does and why. **[Evaluation "
            "practice]** — how the evidence is produced and how to read it. **[Engineering]** — reproducibility, provenance and "
            "packaging.\n\n"
            "**Predict, then check.** Before each principal result a **Predict before running** prompt asks you to commit to an "
            "expectation; after it, **What to notice** describes normal output and a collapsed **Check your reasoning** block "
            "gives a worked answer. Write your own answer first, then open it. No run of this revision has been recorded yet, so "
            "the answers describe the shape of a normal result; the one score they quote — the Lead-1 baseline — was computed "
            "offline on the pinned referenced sample, which needs no model."
        ),
        (
            "## The task: Input → Model/System → Output\n\n"
            "| Stage | Input | Model / system | Output |\n"
            "|---|---|---|---|\n"
            "| **Validate** | prefixed texts and the generation settings | the carried `validate_inputs` | an input manifest: ids, lengths, `known_prefix`, ceilings, verdict |\n"
            "| **Generate** | one prefixed text, e.g. `summarize: …` | T5-small encoder-decoder, greedy (or beam) decoding | `text`, `generated_tokens`, `input_tokens`, `stopped_by`, `known_prefix` |\n"
            "| **Measure** | 20 abstracts with human one-sentence summaries (SciTLDR-A, digest-pinned) | ROUGE-1/2/L against the best reference, beside Lead-1 | an evaluation report with verdict `sample-sanity` |\n"
            "| **Export** | everything above | — | CSV + JSON with provenance |\n\n"
            "## Roadmap\n\n"
            "| Section | Tag | What happens | What you read |\n"
            "|---|---|---|---|\n"
            "| 1. Install the pinned runtime | [Engineering] | isolated environment built; later cells routed to it | versions, CUDA |\n"
            "| 2. Pipeline code | [Engineering] | the repository's module, carried verbatim | nothing to run by hand |\n"
            "| 3. Pin, stage and verify the model | [Engineering] | snapshot downloaded and digest-checked | the verified files |\n"
            "| 4. Author the sample or BYOD | [Concept] | two prefixed inputs (or your file) | the inputs and their digest |\n"
            "| 5. Validate | [Evaluation practice] | input manifest and a rejection probe | `known_prefix`, findings |\n"
            "| 6. Generate | [Concept] | one call per input | outputs, token counts, `stopped_by` |\n"
            "| 7. Measure | [Evaluation practice] | 20 referenced summaries scored beside Lead-1 | ROUGE-1/2/L and the baseline |\n"
            "| 8. Export | [Engineering] | CSV and JSON outputs with provenance | the file list |\n"
            "| 9. Activity (optional) | [Concept] | change `NUM_BEAMS` and compare | your comparison |\n"
            "| Interpretation, Troubleshooting, Conclusion | — | limits, recovery, your notes | when needed |\n\n"
            "**Fast path.** Short on time? Run all, then read Sections 6 and 7 and the conclusion."
        ),
        (
            "<details>\n<summary><strong>Glossary</strong> — open when a term is unfamiliar</summary>\n\n"
            "| Term | Meaning in this notebook |\n"
            "|---|---|\n"
            "| **Task prefix** | The words in front of the text that tell T5 which trained task to perform, e.g. `summarize: ` or `translate English to German: `. The pipeline never adds one. |\n"
            "| **Encoder-decoder (seq2seq)** | A model that reads the whole input once (encoder) and then writes the output one token at a time (decoder). |\n"
            "| **SentencePiece token** | The sub-word unit T5 reads and writes; a word can be one or several tokens. Ceilings are counted in tokens. |\n"
            "| **Greedy decoding** | At every step, emit the single most likely token (`num_beams=1`). Fast and deterministic. |\n"
            "| **Beam search** | Keep the `num_beams` best partial outputs at every step and return the best complete one; can differ from greedy. |\n"
            "| **`stopped_by`** | `eos` when the model ended the output itself, `max_new_tokens` when the step ceiling cut it off mid-thought. |\n"
            "| **`known_prefix`** | Which trained prefix the input started with, or `None`; an unknown prefix still produces text. |\n"
            "| **Reference output** | A human-written answer for an input (here a one-sentence summary); several references per input are allowed. |\n"
            "| **ROUGE-1 / ROUGE-2 / ROUGE-L** | Word-overlap scores between an output and a reference: shared single words, shared word pairs, and the longest shared in-order word sequence; reported as F1 in percent. They measure overlap, not truth. |\n"
            "| **Lead-1 baseline** | Use the first sentence of the input as the summary. A trivial rule any summariser should be compared with. |\n"
            "| **Copy-source baseline** | Return the untranslated input as the \"translation\": the floor for a translation score. |\n"
            "| **`sample-sanity` / `not-measurable`** | The report's verdicts: scored on a small referenced sample (tutorial evidence, not a benchmark) / no reference exists, so nothing is scored. |\n"
            "| **Isolated environment** | A separate Python built from hash-locked pins, in which every learner cell runs. |\n"
            "| **BYOD** | Bring Your Own Data: the optional switch that runs the same cells on your file. |\n\n"
            "</details>"
        ),
    ],
    "prerequisites": [
        "- **Runtime:** a fresh **Linux x86_64** runtime (Google Colab, Kaggle or Linux Jupyter); a CPU is enough and CUDA is used automatically when present (float32 either way). The kernel's own Python version does not matter: Section 1 builds a separate environment with CPython 3.12.12 from the hash-locked pins and every later cell runs there. The lock carries the Linux build of `torch==2.14.0` with its CUDA libraries, which is the largest download of the run (several GB of disk; not measured for this notebook). Timing: the earlier, kernel-install version of this notebook ran on a Kaggle CPU in 215.2 s end to end (14 September 2026); this version adds the isolated-environment build and 20 more generations, and has not been timed yet. For scale, the repository's model card records, for a Windows-venv smoke on an Intel Core Ultra 9 275HX, 3.89 s to load and digest-verify the 244 MB snapshot, 0.16 s for the 11-token translation and 0.31 s for a 94-token `summarize: ` input that generated 48 tokens (a development machine, not a hosted runtime).",
        "- **Knowledge:** basic Python; what an encoder-decoder (seq2seq) model is; what greedy decoding and beam search do; why a generated sentence can be fluent and wrong. The glossary above covers the rest.",
        "- **Data:** two inputs **authored in code** (one translation sentence, one four-sentence passage to summarise), which have no reference outputs, and a **referenced sample** for measurement: the first 20 abstracts of 200–1600 characters from the SciTLDR-A test file (Cachola et al. 2020, Apache-2.0), downloaded once (1.2 MB) from a pinned commit and refused unless its size and SHA-256 match; each abstract has two to four human-written one-sentence summaries. Optional BYOD is off by default; a `.txt` file holds one prefixed input per line, a `.csv` file has `input` and optionally `reference` columns. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded text remains in the notebook runtime; this pipeline does not send it to a third-party inference API.",
        "- **Credentials:** none. The pinned model and the referenced sample are public.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Author the synthetic sample or optional BYOD · [Concept]\n\n"
                "The default sample is **synthetic**: two inputs authored in this cell, each already carrying its task "
                "prefix because the pipeline invents no prefix. The first is the translation sentence the repository's "
                "card-pass smoke used (`translate English to German: The house is wonderful.`); the second is a "
                "four-sentence English passage about the C4 corpus, written here after the T5 paper's description, behind "
                "`summarize: `. Neither has a reference output, so these two are shown, not scored; Section 7 measures the "
                "model on a separate referenced sample. The sample identity and a SHA-256 of its text are printed so an "
                "export can be tied to exactly these inputs.\n\n"
                "Two form parameters fix the generation settings for every call: `GEN_MAX_NEW_TOKENS` (default 64, "
                "the package's `DEFAULT_MAX_NEW_TOKENS`) and `NUM_BEAMS` (default 1 = greedy). They are checked against "
                "the carried module's ceilings in the next section and apply to Section 7 too.\n\n"
                "BYOD is optional and disabled by default. Set `USE_BYOD = True` and either put a file path in `BYOD_PATH` "
                "(any Jupyter runtime) or leave it empty and upload **exactly one** file in Colab. `read_byod_inputs` "
                "reads it: a `.txt` file holds one input per non-empty line; a `.csv` file has a header with an `input` "
                "column and optionally a `reference` column (then every row needs a reference, and Section 7 scores your "
                "outputs against them). Every input must already start with its task prefix, be at most `MAX_TEXT_CHARS` "
                "characters and tokenise to at most `MAX_INPUT_TOKENS` SentencePiece pieces, which the pipeline enforces by "
                "rejecting, not by truncating. The file must be UTF-8; a byte-order mark (Windows Notepad's \"UTF-8 with "
                "BOM\") is removed, and any other encoding is refused with a message naming the file. The file stays inside "
                "this runtime."
            ),
            "code": (
                "import hashlib\n"
                "from pathlib import Path\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "GEN_MAX_NEW_TOKENS = 64  # @param {{type:\"integer\"}}\n"
                "NUM_BEAMS = 1  # @param {{type:\"integer\"}}\n\n"
                "references = None\n"
                "if USE_BYOD:\n"
                "    if BYOD_PATH:\n"
                "        sample_name, sample_bytes = Path(BYOD_PATH).name, Path(BYOD_PATH).read_bytes()\n"
                "    else:\n"
                "        from google.colab import files\n"
                "        uploaded = files.upload()\n"
                "        if len(uploaded) != 1:\n"
                "            raise RuntimeError(f'Upload exactly one .txt or .csv file (received {{len(uploaded)}}); or set BYOD_PATH and run this cell again.')\n"
                "        ((sample_name, sample_bytes),) = uploaded.items()\n"
                "    texts, references = read_byod_inputs(sample_name, sample_bytes)\n"
                "    sample_kind = 'BYOD file' + (' with references' if references else '')\n"
                "else:\n"
                "    texts = [\n"
                "        'translate English to German: The house is wonderful.',\n"
                "        'summarize: The Colossal Clean Crawled Corpus, or C4, is a cleaned version of the April 2019 Common Crawl web snapshot. '\n"
                "        'The corpus was prepared by keeping only lines that end in terminal punctuation, dropping pages with fewer than five sentences, '\n"
                "        'and removing pages that contain words from a block list, placeholder text or source code. '\n"
                "        'Duplicate three-sentence spans were also removed so that boilerplate does not dominate the training signal. '\n"
                "        'It was used to pre-train the T5 family of models with a span-corruption denoising objective.',\n"
                "    ]\n"
                "    sample_name = 'synthetic_prefixed_inputs'\n"
                "    sample_kind = 'synthetic (authored in this cell)'\n"
                "item_ids = [f'input{{index:02d}}' for index in range(len(texts))]\n"
                "sample_sha256 = hashlib.sha256('\\n'.join(texts).encode('utf-8')).hexdigest()\n"
                "print({{'sample': sample_name, 'sample_kind': sample_kind, 'inputs': len(texts), 'references': references is not None, 'text_sha256': sample_sha256, 'max_new_tokens': GEN_MAX_NEW_TOKENS, 'num_beams': NUM_BEAMS}})\n"
                "for item_id, text in zip(item_ids, texts, strict=True):\n"
                "    print(f'{{item_id}}: {{text[:110]}}' + ('...' if len(text) > 110 else ''))"
            ),
        },
        {
            "md": (
                "## 5. Validate the inputs → input manifest · [Evaluation practice]\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `generate` "
                "applies — both route through the same private `_check_inputs` — so the text type, non-emptiness, the "
                "character ceiling `MAX_TEXT_CHARS`, `max_new_tokens` in 1..`MAX_NEW_TOKENS` and `num_beams` in "
                "1..`MAX_NUM_BEAMS` are enforced identically. `generate` takes one text per call, so the helper "
                "validates the whole batch the notebook will loop over with the same settings and returns one **input "
                "manifest** naming the schema and ceilings, each input's identifier, character count and "
                "`known_prefix`, the settings in force, and the verdict; it is written to "
                "`outputs/{stem}_input_manifest.json`. An input that starts with no trained prefix is **not** rejected — "
                "the pipeline does not refuse it either — but `known_prefix` is `null` so a reader can see it, and the "
                "model then returns something plausible-looking rather than an error. `MAX_INPUT_TOKENS` (encoder tokens "
                "including `</s>`; the upstream `n_positions`) needs the real tokenizer and is therefore enforced inside "
                "`generate`, which **rejects with a `ValueError` naming the count, never silently cuts**; every result "
                "reports `input_tokens`. `DECISION_RULE` states the decoding rule in force (greedy per-step argmax, beam "
                "search when `num_beams > 1`, no sampling) and `TASK_PREFIXES` lists the four prefixes the checkpoint "
                "was trained on. To show what rejection looks like, the cell also validates an out-of-range `num_beams` "
                "and records the pipeline's own error message as a finding. Nothing here trims or alters the texts.\n\n"
                "**Predict before running:** which `known_prefix` will each default input get, and will the out-of-range "
                "`num_beams` probe be rejected, or silently clamped to the ceiling?"
            ),
            "code": (
                "import json\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "ceilings = {{'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'MAX_INPUT_TOKENS': MAX_INPUT_TOKENS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'MAX_NUM_BEAMS': MAX_NUM_BEAMS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS}}\n"
                "print(ceilings)\n"
                "print({{'decision_rule': DECISION_RULE}})\n"
                "print({{'task_prefixes': list(TASK_PREFIXES)}})\n"
                "input_manifest = validate_inputs(texts, max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=NUM_BEAMS, names=item_ids)\n"
                "# Demonstrate rejection on a setting that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(texts, max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=MAX_NUM_BEAMS + 1)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'num-beams-ceiling-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))\n"
                "unknown = [entry['id'] for entry in input_manifest['inputs'] if entry['known_prefix'] is None]\n"
                "if unknown:\n"
                "    print({{'warning': f'{{unknown}} start with no trained prefix; the model will still return text, but not a summary or translation'}})\n"
                "print({{'token_ceiling': 'enforced by generate() with the real tokenizer; reported as input_tokens'}})"
            ),
        },
        {
            "md": (
                "**What to notice (Section 5):** the verdict is `accepted`, `input00` has `known_prefix` "
                "`translate English to German: ` and `input01` has `summarize: `, and `findings` holds one entry, the "
                "`num-beams-ceiling-probe`, with the pipeline's own message.\n\n"
                "<details><summary>Check your reasoning</summary>Both default inputs start with a trained prefix, so both are "
                "recognised. The probe is **rejected**: the pipeline refuses a setting outside its ceilings with a "
                "`ValueError` instead of clamping it, so a caller always gets the settings they asked for or an error — never "
                "a silent change. An input without a prefix would be accepted with `known_prefix: null`, because the model "
                "can still produce text for it.</details>\n\n"
                "## 6. Generate and read the outputs correctly · [Concept]\n\n"
                "`generate(text, max_new_tokens=..., num_beams=...)` runs one prefixed input through the encoder-decoder "
                "and returns a dict: `text` (the decoded generation with special tokens removed), `generated_tokens` "
                "(decoder tokens emitted, excluding `</s>`/pad), `input_tokens` (encoder tokens including `</s>`, the "
                "number checked against `MAX_INPUT_TOKENS`), `stopped_by` (`eos` when the model ended the sequence "
                "itself, `max_new_tokens` when it hit the ceiling — such an output is cut mid-thought and should be "
                "re-run with a larger `GEN_MAX_NEW_TOKENS` before anyone reads it as a finished summary), `known_prefix` "
                "(which trained prefix the input started with, or `None`), the `generation` settings actually used "
                "(`max_new_tokens`, `num_beams`, `do_sample=False`, `decision_rule`), `device`, `source` and the model "
                "identity. **Score semantics:** the pipeline emits **no probability, confidence or score of any kind** — "
                "the counts above are counts, not scores; greedy decoding is an implicit per-step argmax with no "
                "minimum-probability cut-off, so some token is always produced, and the pipeline ships no acceptance "
                "threshold on output quality. Whoever deploys it owns any acceptance rule, judged on their own "
                "references. The run is deterministic for a given input, settings, weights, device and library versions "
                "(no sampling, `model.eval()`, no seed needed); float32 kernel differences between CPU and CUDA can flip "
                "a near-tied token and change the rest of the sequence from that point, and beam search can differ from "
                "greedy. The checks below are falsifiable plumbing checks — one result per input, every count within its "
                "ceiling, every default input recognised by its prefix — plus a per-call wall time measured on the "
                "runtime identified in Section 1 (the first call includes warm-up).\n\n"
                "**Predict before running:** for each input, will the output stop by `eos` or by `max_new_tokens`? And "
                "will the summary of the four-sentence passage be shorter than its first sentence, or longer?"
            ),
            "code": (
                "import time\n\n"
                "results = []\n"
                "for item_id, text in zip(item_ids, texts, strict=True):\n"
                "    started = time.perf_counter()\n"
                "    result = pipe.generate(text, max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=NUM_BEAMS)\n"
                "    elapsed = time.perf_counter() - started\n"
                "    results.append({{'id': item_id, 'input': text, 'seconds': round(elapsed, 3), **result}})\n"
                "    print(f\"{{item_id}} [{{result['known_prefix'] or 'no known prefix'}}] {{result['input_tokens']}} -> {{result['generated_tokens']}} tokens, stopped_by={{result['stopped_by']}}, {{elapsed:.2f}} s\")\n"
                "    print(f\"    {{result['text']}}\")\n"
                "checks = {{\n"
                "    'one_result_per_input': len(results) == len(texts),\n"
                "    'generated_within_ceiling': all(r['generated_tokens'] <= GEN_MAX_NEW_TOKENS for r in results),\n"
                "    'input_within_ceiling': all(r['input_tokens'] <= MAX_INPUT_TOKENS for r in results),\n"
                "    'settings_echoed': all(r['generation']['max_new_tokens'] == GEN_MAX_NEW_TOKENS and r['generation']['num_beams'] == NUM_BEAMS and r['generation']['do_sample'] is False for r in results),\n"
                "}}\n"
                "if not USE_BYOD:\n"
                "    checks['every_default_input_has_known_prefix'] = all(r['known_prefix'] is not None for r in results)\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'generate output failed a sanity check: {{checks}}')\n"
                "print({{'checks': checks, 'decision_rule': results[0]['generation']['decision_rule'], 'hit_token_ceiling': [r['id'] for r in results if r['stopped_by'] == 'max_new_tokens']}})"
            ),
        },
        {
            "md": (
                "**What to notice (Section 6):** a short German sentence for `input00` and an English condensation for "
                "`input01`, every check `True`, and `hit_token_ceiling` naming any output that was cut off.\n\n"
                "<details><summary>Check your reasoning</summary>The translation is short and ends by `eos` well below 64 "
                "tokens (the card-pass smoke on one machine returned `Das Haus ist wunderbar.`, 5 tokens). The summary is "
                "most likely to end by `eos` too, but if it lists `max_new_tokens` it was cut mid-sentence: raise "
                "`GEN_MAX_NEW_TOKENS` before reading it. T5-small often summarises by **copying** one or two sentences of "
                "the input nearly verbatim, so the summary may well be longer than the first sentence. Whether either "
                "output is *good* is exactly what these checks cannot tell you: they test the contract, not the quality. "
                "That needs references, which Section 7 brings.</details>\n\n"
                "## 7. Measure on a referenced sample → evaluation report · [Evaluation practice]\n\n"
                "Summarisation and translation are measurable tasks: given human-written **reference outputs**, a "
                "generated output can be scored by its overlap with them. `evaluation_report` is the pipeline's public "
                "evaluation stage. Without references it returns the verdict `not-measurable`; with references it scores "
                "the outputs with **ROUGE-1/2/L F1** (best-matching reference per item, averaged, in percent) and, beside "
                "them, the same scores for a **trivial baseline** formed from each input — **Lead-1** (the first sentence) "
                "for `summarize: `, **copy-source** for `translate …` — and returns the verdict `sample-sanity`.\n\n"
                "The default measured sample is `fetch_reference_sample()`: the SciTLDR-A test file (scientific-paper "
                "abstracts, each with two to four human one-sentence summaries) at a pinned upstream commit, refused unless "
                "its size and SHA-256 match, from which the first 20 abstracts of 200–1600 characters are taken, each "
                "behind `summarize: `. All 20 are generated with the Section 4 settings and scored; the report lands at "
                "`outputs/{stem}_evaluation_report.json`, with every output in `outputs/{stem}_reference_generations.csv`. "
                "The Lead-1 baseline on these 20 items, computed offline with the carried scorer (no model needed), is "
                "**ROUGE-1 27.22, ROUGE-2 11.12, ROUGE-L 22.22**. The Section 4 inputs get their own report, "
                "`outputs/{stem}_inputs_evaluation_report.json`: `not-measurable` for the authored inputs (they have no "
                "references), scored the same way for a BYOD `.csv` with a `reference` column.\n\n"
                "**What the number can and cannot show.** ROUGE counts shared words with a human summary; a fluent summary "
                "that states a wrong fact can score well, and a correct paraphrase can score poorly. Twenty abstracts give "
                "tutorial evidence about this model on this kind of text, not a benchmark result and not a dispersion "
                "estimate. T5-small was trained to summarise news articles, not scientific abstracts, so this sample is "
                "out of its training domain. The scorer follows `rouge-score` without stemming (lower-cased runs of letters "
                "and digits), so its numbers are close to, not identical with, published ROUGE figures; the upstream "
                "per-task figures in the paper's Table 14 are upstream claims, not measured here.\n\n"
                "**Predict before running:** will T5-small's ROUGE-1 on the 20 abstracts be above or below the Lead-1 "
                "baseline's 27.22? Write down a number before you run the cell."
            ),
            "code": (
                "reference_items = fetch_reference_sample()\n"
                "reference_results = []\n"
                "for item in reference_items:\n"
                "    started = time.perf_counter()\n"
                "    result = pipe.generate(item['input'], max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=NUM_BEAMS)\n"
                "    reference_results.append({{'id': item['id'], 'input': item['input'], 'seconds': round(time.perf_counter() - started, 3), **result}})\n"
                "report = evaluation_report(reference_results, [item['references'] for item in reference_items], sample_kind=REFERENCE_SAMPLE_NAME)\n"
                "report['sample'] = {{'name': REFERENCE_SAMPLE_NAME, 'url': REFERENCE_SAMPLE_URL, 'sha256': REFERENCE_SAMPLE_SHA256, 'license': REFERENCE_SAMPLE_LICENSE, 'items': len(reference_items)}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "inputs_report = evaluation_report(results, references, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_inputs_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(inputs_report, handle, indent=2, ensure_ascii=False)\n"
                "model_scores = {{m['metric']: m['value'] for m in report['metrics']}}\n"
                "lead = report['baselines'][0]\n"
                "print({{'verdict': report['verdict'], 'items': report['n_items'], 't5_small': model_scores, 'lead_1': {{k: lead[k] for k in ('rouge1', 'rouge2', 'rougeL')}}, 'stopped_by_max_new_tokens': len(report['outputs_stopped_by_max_new_tokens'])}})\n"
                "for item, scored in list(zip(reference_results, report['per_item'], strict=True))[:3]:\n"
                "    print(f\"{{item['id']}} ROUGE-L {{100 * scored['rougeL']:.1f}}: {{item['text'][:160]}}\")\n"
                "print({{'inputs_verdict': inputs_report['verdict'], 'inputs_reason': inputs_report['reason'], 'inputs_metrics': inputs_report['metrics']}})"
            ),
        },
        {
            "md": (
                "**What to notice (Section 7):** the verdict `sample-sanity`, three model scores beside the three Lead-1 "
                "scores, how many outputs stopped by `max_new_tokens`, three example outputs with their ROUGE-L, and, for "
                "the authored inputs, `not-measurable` with the reason \"the evaluated sample has no reference outputs\".\n\n"
                "<details><summary>Check your reasoning</summary>Either outcome is informative, and neither is a benchmark. "
                "Lead-1 is a strong baseline on abstracts because the first sentence often names the topic, and the "
                "references are single sentences, so a long, copied T5-small output pays a precision penalty. If T5-small "
                "lands below Lead-1, the model adds nothing over a one-line rule on this text, which is plausible for a "
                "small model summarising outside its training domain; if it lands above, the margin on 20 items is still "
                "too small to generalise from. Read the three examples: a high ROUGE-L with a wrong claim, or a low one "
                "with a fair paraphrase, shows what overlap scores cannot see. Outputs listed under "
                "`outputs_stopped_by_max_new_tokens` were cut off and score lower than they would finished.</details>\n\n"
                "## 8. Export the generations and provenance · [Engineering]\n\n"
                "Three further files are written under `outputs/` beside the input manifest and the two evaluation reports: "
                "`outputs/{stem}_generations.csv` — one row per Section 4 input with its identifier, the known prefix, the "
                "input text, the generated text, both token counts, `stopped_by` and the wall time, so every generation maps "
                "back to its input; `outputs/{stem}_reference_generations.csv` — the same for the 20 referenced items, plus "
                "the first reference and the item's ROUGE-L; and `outputs/{stem}_result.json`, which carries the items plus "
                "the generation settings in force, the ceilings, the sanity checks, the input manifest, both evaluation "
                "reports, the sample identities and digests, the notebook's source (repository, revision, embedded module "
                "digest, generator), the model identifier, the immutable model revision, the model licence, the verified "
                "snapshot summary, and the runtime identity (Python, `torch`, `transformers`, device, dtype). No "
                "credentials are involved in any step, so none can reach the export."
            ),
            "code": (
                "import csv\n\n"
                "items = [\n"
                "    {{'id': r['id'], 'known_prefix': r['known_prefix'], 'input': r['input'], 'output': r['text'], 'input_tokens': r['input_tokens'], 'generated_tokens': r['generated_tokens'], 'stopped_by': r['stopped_by'], 'seconds': r['seconds']}}\n"
                "    for r in results\n"
                "]\n"
                "with open('outputs/{stem}_generations.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.DictWriter(handle, fieldnames=list(items[0]))\n"
                "    writer.writeheader()\n"
                "    writer.writerows(items)\n"
                "reference_rows = [\n"
                "    {{'id': r['id'], 'output': r['text'], 'reference_1': item['references'][0], 'rougeL': round(100 * scored['rougeL'], 2), 'generated_tokens': r['generated_tokens'], 'stopped_by': r['stopped_by'], 'seconds': r['seconds']}}\n"
                "    for r, item, scored in zip(reference_results, reference_items, report['per_item'], strict=True)\n"
                "]\n"
                "with open('outputs/{stem}_reference_generations.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.DictWriter(handle, fieldnames=list(reference_rows[0]))\n"
                "    writer.writeheader()\n"
                "    writer.writerows(reference_rows)\n"
                "payload = {{\n"
                "    'items': items,\n"
                "    'generation': results[0]['generation'],\n"
                "    'ceilings': ceilings,\n"
                "    'sanity_checks': checks,\n"
                "    'generations_file': 'outputs/{stem}_generations.csv',\n"
                "    'reference_generations_file': 'outputs/{stem}_reference_generations.csv',\n"
                "    'input_manifest': input_manifest,\n"
                "    'evaluation_report': report,\n"
                "    'inputs_evaluation_report': inputs_report,\n"
                "    'sample': {{'name': sample_name, 'kind': sample_kind, 'inputs': len(texts), 'references': references is not None, 'text_sha256': sample_sha256}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': len(snapshot['files']), 'total_bytes': snapshot.get('totalBytes'), 'fetched_this_run': fetched}},\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "        'dtype': 'float32',\n"
                "        'source': pipe.source,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## 9. Activity: change one thing — beam search · [Concept]\n\n"
        "Optional; **Predict → Change one thing → Run → Observe → Explain**. It changes nothing unless you do it.\n\n"
        "1. **Predict:** with `NUM_BEAMS = 4` instead of 1, will the translation change? Will the ROUGE scores in Section 7 go "
        "up, down, or stay within a point?\n"
        "2. **Change:** in Section 4 set `NUM_BEAMS = 4`. Change nothing else. (Note the Section 6 outputs and the Section 7 "
        "scores of your first run before you continue: the next step overwrites `outputs/`.)\n"
        "3. **Run:** select the Section 4 cell and choose *Runtime → Run after*. The referenced sample is already cached and is "
        "not downloaded again.\n"
        "4. **Observe:** compare the outputs, `generated_tokens` and wall times in Section 6, and the three ROUGE scores in "
        "Section 7, with your notes.\n"
        "5. **Explain:** in one sentence, why can beam search change an output that greedy decoding produced, and why is a "
        "change of one ROUGE point on 20 items not evidence that one setting is better?\n\n"
        "<details><summary>Check your reasoning</summary>Greedy decoding commits to the single best token at each step; beam "
        "search keeps four partial outputs and can prefer a sequence whose first token was not the greedy choice, so outputs "
        "can differ — the card-pass smoke found the short translation unchanged, and longer summaries are more likely to "
        "move. Beam search costs more time per call. On 20 items one or two outputs changing can move a mean by a point "
        "either way; without a dispersion estimate or a larger sample you cannot separate that from noise.</details>\n\n"
        "## Interpretation and limits\n\n"
        "The generated strings are the model's continuation of *your* prefixed input under greedy (or beam) decoding: "
        "fluent text that can add, drop or invert a fact, and the pipeline attaches no probability, confidence or "
        "quality score to it — `generated_tokens`, `input_tokens` and `stopped_by` are counts and flags, not evidence of "
        "correctness. The measured result is ROUGE on 20 scientific abstracts beside a Lead-1 baseline: tutorial "
        "`sample-sanity` evidence about overlap with human summaries on one small, out-of-domain sample, with no "
        "dispersion estimate; it says nothing about factual accuracy, about translation quality (no referenced translation "
        "sample is carried; bring one through BYOD), or about your domain. The authored inputs are `not-measurable` because "
        "they have no references. An unknown prefix produces plausible-looking output rather than an error; inputs above "
        "`MAX_INPUT_TOKENS` are refused rather than cut; outputs that stop at `max_new_tokens` are truncated mid-thought "
        "and are listed in the report. The pipeline exposes no instruction following, sampling, batching, or the upstream "
        "`task_specific_params` decoding settings. Decoding is deterministic on a fixed device and dtype, but CPU and CUDA "
        "float32 kernels can diverge on a near-tied token.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, "
        "can acquire and digest-verify the pinned model snapshot and the pinned referenced sample, validate the "
        "demonstrated inputs against the enforced ceilings, execute the public pipeline path in an isolated, hash-locked "
        "environment, score its outputs against references beside a trivial baseline, and emit the shown machine-readable "
        "outputs in the tested runtime — without the repository being reachable. It does **not** establish benchmark "
        "superiority, summarisation or translation quality on any domain, factual faithfulness, a usable acceptance "
        "threshold, safety for high-consequence decisions, or production fitness on an unseen domain.\n\n"
        "**Next experiments.** Translate the same sentence with the French and Romanian prefixes; bring a `.csv` of your "
        "own inputs with references through BYOD (a few dozen rows from your domain) and compare the model with the "
        "baseline there; raise `GEN_MAX_NEW_TOKENS` until no referenced output stops by `max_new_tokens` and see whether "
        "the scores move; run the same notebook on a CUDA runtime and diff the outputs against the CPU run. None of these "
        "turns the sample result into evidence of production fitness.\n\n"
        "## Troubleshooting\n\n"
        "| Symptom | Likely cause | What to do |\n"
        "|---|---|---|\n"
        "| Section 1 stops with `This notebook needs a Linux x86_64 runtime` | a local Windows or macOS kernel, or an ARM machine | Use Google Colab, Kaggle or a Linux x86_64 Jupyter; the lock holds manylinux x86_64 wheels. |\n"
        "| Section 1 fails while downloading, or `The pinned uv wheel failed its size/SHA-256 check` | a network failure, or an altered download | Run the Section 1 install cell again; a repeated mismatch means the download is being altered — never edit the digest. |\n"
        "| `holds Python …, not 3.12.12` in Section 1 | an older `dimer_isolated_env/` folder from another notebook version | Delete that folder (or start a fresh runtime) and run the install cell again. |\n"
        "| `The isolated environment's Python process exited` | the worker ran out of memory | Restart the session and choose *Runtime → Run all* again. |\n"
        "| `FileNotFoundError: snapshot file missing` or a `sha256`/`size` `ValueError` in Section 3 | a staged file is incomplete or altered | Delete it from `weights/{MODEL_KEY}/` and run Section 3 again; never edit the manifest. |\n"
        "| `RuntimeError: Upload exactly one .txt or .csv file` in Section 4 | the upload dialog was cancelled, or several files were chosen | Run the cell again and choose one file, or set `BYOD_PATH`. |\n"
        "| `…: not UTF-8 text` in Section 4 | the file was saved in another encoding (for example Windows-1252) | Save it as UTF-8 (with or without BOM) and upload it again. |\n"
        "| `a CSV needs a header with an input column` or `has an empty reference` | the CSV header or a row is incomplete | Add the `input` (and, for scoring, `reference`) header; give every row a reference or drop the column. |\n"
        "| `ValueError: input is N tokens; ceiling is MAX_INPUT_TOKENS=512` in Section 6 | a BYOD input is too long | Split or shorten that input and run again from Section 4. |\n"
        "| `stopped_by` equal to `max_new_tokens` | the output hit the step ceiling | Raise `GEN_MAX_NEW_TOKENS` (ceiling `MAX_NEW_TOKENS`) in Section 4 and run again from there. |\n"
        "| `reference sample: … != pinned …; refusing it` in Section 7 | the download was cut off or altered | Run Section 7 again; a repeated mismatch means the file is being altered — never edit the digest. |\n"
        "| Output that copies the input | the input has no trained prefix, or the passage is too short to summarise | Check `known_prefix` in Section 5. |\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional. Fill in from your own run, one sentence each:\n\n"
        "1. The translation of `The house is wonderful.` was ___, stopped by ___.\n"
        "2. On the 20 referenced abstracts T5-small scored ROUGE-1/2/L ___ / ___ / ___, against Lead-1's 27.22 / 11.12 / 22.22; ___ outputs stopped by `max_new_tokens`.\n"
        "3. What this comparison can show about T5-small as a summariser, and what it cannot: ___.\n"
        "4. With `NUM_BEAMS = 4` (Section 9), the outputs ___ and the scores ___.\n"
        "5. What I would need before trusting this model's summaries or translations on my own text: ___.\n\n"
        "## References\n\n"
        f"- Repository README: https://github.com/kurtvalcorza/{REPO}/blob/main/README.md\n"
        f"- Repository model card: https://github.com/kurtvalcorza/{REPO}/blob/main/MODEL_CARD.md\n"
        f"- Weight provenance: https://github.com/kurtvalcorza/{REPO}/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/google-research/text-to-text-transfer-transformer\n"
        "- Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (Raffel et al., JMLR 2020): https://arxiv.org/abs/1910.10683\n"
        "- TLDR: Extreme Summarization of Scientific Documents (Cachola et al., Findings of EMNLP 2020), the SciTLDR data: https://arxiv.org/abs/2004.15011\n"
        "- ROUGE: A Package for Automatic Evaluation of Summaries (Lin, 2004): https://aclanthology.org/W04-1013/"
    ),
}
