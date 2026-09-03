# Reproducible GGUF Comparisons

Reproduce comparisons between project-built and community GGUF files on GPQA Diamond, AIME, IFEval, and other fixed benchmark profiles. EvalScope 1.11.0 is the evaluation engine; this repository provides the runner and clean Markdown result tables.

## Install

Python 3.10+ is required. Local GGUF evaluation also needs `llama-server` on `PATH`.

```bash
git clone https://github.com/drawthingsai/benchmarks.git
cd benchmarks
python3 -m pip install evalscope==1.11.0
```

## Run a GGUF

The default command runs a five-sample GSM8K smoke test:

```bash
python3 benchmark.py run --gguf /path/to/model.gguf
```

Use the public comparison profile for a real run:

```bash
python3 benchmark.py run \
  --profile profiles/qwen3.8-thinking.json \
  --gguf /path/to/model.gguf \
  --model project-q2-k \
  --run-id project-q2-k \
  --ctx-size 98304 \
  --source-url https://huggingface.co/owner/repository \
  --source-revision COMMIT_OR_TAG \
  --quantization "standard llama.cpp Q2_K"
```

Repeat the same command for each community GGUF, changing only its path, label, run ID, and provenance. Use `--dry-run` to inspect a run without loading the model.

## Compare results

```bash
python3 benchmark.py compare \
  runs/project-q2-k \
  runs/community-a-q2-k \
  runs/community-b-q2-k \
  --title "Qwen3.8 27B GGUF comparison" \
  --output results/qwen3.8-27b.md
```

The generated Markdown keeps every benchmark separate, preserves missing results, and includes run health plus GGUF filename, size, SHA-256, source revision, and quantization method.

| Model / GGUF | Run health | GPQA Diamond | AIME 2024 | AIME 2025 | IFEval |
|---|---:|---:|---:|---:|---:|
| Project GGUF | pass | ... | ... | ... | ... |
| Community GGUF | pass | ... | ... | ... | ... |

Runs can be compared only when their profile SHA-256 values match. No HTML or composite score is generated.

## OpenAI-compatible API

```bash
export MODEL_API_KEY='...'
python3 benchmark.py run \
  --url https://provider.example/v1 \
  --model model-id \
  --api-key-env MODEL_API_KEY
```

The API key is optional and is not written to process arguments, manifests, or logs.

Results are stored under `runs/<run-id>/`. Rebuild a report with `python3 benchmark.py report runs/<run-id>`.
