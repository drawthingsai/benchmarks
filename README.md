# Reproducible GGUF Comparisons

Compare GGUF artifacts built by this project with established community releases on GPQA Diamond, AIME, IFEval, and other versioned benchmark profiles. Every published comparison records the exact file hashes, stated provenance, evaluation settings, coverage and failure information, and raw evaluator outputs behind the scores.

Runs use the pinned EvalScope `1.11.0` evaluation engine. The installed `gguf-bench` command runs individual artifacts, rebuilds reports, and produces cross-GGUF Markdown comparisons. The implementation remains small: `benchmark.py` handles execution and `report.py` normalizes results.

## What a published comparison contains

| Layer | What is fixed or recorded |
|---|---|
| GGUF artifact | Exact filename, byte size, SHA-256, source URL, source revision, and stated quantization method |
| Evaluation contract | Versioned profile, dataset splits, sample counts, seed, generation settings, and primary metrics |
| Run health | Completed responses, empty outputs, parse failures, token-limit stops, missing samples, and duplicates |
| Evidence | Raw evaluator artifacts, run manifest, normalized results, and generated Markdown reports |

Scores are compared only when runs have the same profile SHA-256. Missing or incomplete results remain visible, and unrelated benchmarks are never collapsed into a composite score.

## Install

Python 3.10 or newer is required. Local GGUF evaluation also requires a recent `llama-server` executable on `PATH`.

```bash
git clone https://github.com/drawthingsai/benchmarks.git
cd benchmarks
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The package pins `evalscope==1.11.0`, and the runner refuses any other installed version.

## Run one GGUF

The default profile runs a five-sample GSM8K pipeline check. It validates the setup but is not a publishable benchmark result.

```bash
gguf-bench run --gguf /path/to/model.gguf
```

For a run intended for publication, select a versioned profile and record artifact provenance at execution time:

```bash
gguf-bench run \
  --profile profiles/qwen3.8-thinking.json \
  --gguf /path/to/model.gguf \
  --model project-q2-k \
  --run-id project-q2-k \
  --ctx-size 98304 \
  --source-url https://huggingface.co/owner/repository \
  --source-revision COMMIT_OR_TAG \
  --quantization "standard llama.cpp Q2_K"
```

The smoke defaults use one server slot, single-request concurrency, 32K total context, and request GPU offload for all model layers.

Use `--dry-run` to validate the profile and inspect the execution plan without loading a model, downloading datasets, or sending API requests:

```bash
gguf-bench run --gguf /path/to/model.gguf --dry-run
```

## Compare project and community GGUFs

Run each artifact with the same versioned profile, a unique `--model` label, and a unique `--run-id`. Record the canonical upstream details for every community artifact.

```bash
gguf-bench run \
  --profile profiles/qwen3.8-thinking.json \
  --gguf /path/to/community-a.gguf \
  --model community-a-q2-k \
  --run-id community-a-q2-k \
  --ctx-size 98304 \
  --source-url https://huggingface.co/owner/community-repository \
  --source-revision COMMIT_OR_TAG \
  --quantization "UPSTREAM_QUANTIZATION_METHOD"
```

After all runs finish, pass their output directories to the comparison command:

```bash
gguf-bench compare \
  runs/project-q2-k \
  runs/community-a-q2-k \
  runs/community-b-q2-k \
  --title "Qwen3.8 27B GGUF comparison" \
  --output results/qwen3.8-27b.md
```

The command rejects profile hash mismatches and writes a Markdown score matrix followed by artifact provenance and the reproduction contract.

### Comparison report structure

| Model / GGUF | Run health | GPQA Diamond | AIME 2024 | AIME 2025 | IFEval |
|---|---:|---:|---:|---:|---:|
| Project GGUF | — | — | — | — | — |
| Community baseline A | — | — | — | — | — |
| Community baseline B | — | — | — | — | — |

The generated report replaces the placeholders with scores, preserves missing values as em dashes, and adds a provenance table containing each run ID, filename, byte size, SHA-256, source revision, and quantization method. No HTML or composite score is generated.

## Run an OpenAI-compatible endpoint

Use the same profiles with a local or remote OpenAI-compatible API:

```bash
gguf-bench run --url http://127.0.0.1:8000/v1 --model model-id
```

API keys are not accepted as command-line arguments. The runner reads a named environment variable and injects the credential through a loopback proxy, so the key is absent from evaluator process arguments, environment, manifests, and logs:

```bash
export MODEL_API_KEY='...'
gguf-bench run \
  --url https://provider.example/v1 \
  --model model-id \
  --api-key-env MODEL_API_KEY
```

By default, `doctor` performs a non-generative `/models` connectivity check. Add `--generate` to send a minimal generation request; remote providers may charge for it.

```bash
gguf-bench doctor --url https://provider.example/v1 \
  --model model-id --api-key-env MODEL_API_KEY
```

## Output layout

```text
runs/<run-id>/
├── manifest.json
├── profile.json
├── status.json
├── cases/<case-id>/attempt-0001/
│   ├── attempt.json
│   ├── evaluator.log
│   └── evalscope/          # raw evaluator output
├── summary.json
└── report.md
```

All human-readable benchmark result artifacts are Markdown. Rebuild a single report without rerunning inference with:

```bash
gguf-bench report runs/<run-id>
```

## Evaluation profiles

- `profiles/smoke.json` is a minimal pipeline check.
- `profiles/qwen3.8-thinking.json` covers GPQA Diamond, AIME 2024, AIME 2025, and IFEval with fixed dataset splits, sample counts, metrics, sampling parameters, and reasoning mode.

Profiles are strict JSON. Unknown fields, duplicate keys, non-finite numbers, unsafe or duplicate case IDs, and missing sample counts are rejected. `expected_samples` is the total after applying any `limit`; it is multiplied by `repeats` to calculate planned requests.

Do not modify a profile after publishing results. Add a versioned profile so prior comparisons remain reproducible.
