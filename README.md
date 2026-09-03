# Reproducible GGUF Comparisons

Reproduce comparisons between project-built and community GGUF files on GPQA Diamond, AIME, IFEval, and BFCL. EvalScope 1.11.0 is the evaluation engine; this repository provides the runner and clean Markdown result tables.

## Install

Python 3.10+ is required. Local GGUF evaluation also needs `llama-server` on `PATH`.

```bash
git clone https://github.com/drawthingsai/benchmarks.git
cd benchmarks

conda create -n benchmarks python=3.12
conda activate benchmarks

python3 -m pip install 'evalscope[bfcl]==1.11.0'
```

For local GGUF runs, use llama.cpp commit `0df974d777c904dda1da3b00faa7769c6310ae74` (`llama.cpp 0.3.0-dev`, build `474`). Build `llama-server` from that exact revision and place `llama.cpp/build/bin` on `PATH`:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
git -C llama.cpp checkout 0df974d777c904dda1da3b00faa7769c6310ae74
cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build llama.cpp/build --target llama-server -j
export PATH="$PWD/llama.cpp/build/bin:$PATH"
```

## Datasets

| Profile | Dataset | EvalScope ID | Dataset rows | Evaluated rows | Primary metric |
|---|---|---|---:|---:|---|
| Qwen3.8 thinking | GPQA Diamond | `gpqa_diamond` | 198 | 198 | `mean_acc` |
| Qwen3.8 thinking | AIME 2026 | `aime26` | 30 | 30 | `mean_acc` |
| Qwen3.8 thinking | IFEval | `ifeval` | 541 | 541 | `mean_prompt_level_strict` |
| Qwen3.8 thinking | BFCL v4 Quick | `bfcl_v4` | 5,106 | 200 | `acc` |

BFCL v4 Quick evaluates 10 fixed categories with 20 examples each. It excludes memory and Web Search tasks, so no SerpAPI key is needed.

Datasets are downloaded from Hugging Face. GPQA is gated: accept the [official dataset terms](https://huggingface.co/datasets/Idavidrein/gpqa) and run `hf auth login` before the full profile.

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
  --model-name project-q2-k \
  --run-id project-q2-k
```

`--model-name` is optional for a local file; its full GGUF filename is used by default. Repeat the command for each community GGUF. Use `--dry-run` to inspect a run without loading the model.

Qwen3.8 provides a native 262,144-token context window, and the profile uses the same value as its generation ceiling. No fixed prompt allowance is configured: llama-server uses the actual prompt length and stops generation when the remaining context is exhausted. A prompt that alone exceeds the model context is reported as an error.

The command above starts llama-server with these effective arguments:

```text
--model MODEL.gguf
--alias MODEL
--host 127.0.0.1
--port AUTO_ASSIGNED
--ctx-size PROFILE_LIMIT_TIMES_SLOTS
--parallel AUTO_SLOTS
--n-gpu-layers 999
--cache-type-k f16
--cache-type-v f16
--jinja
--no-context-shift
--no-webui
```

For local GGUF runs, EvalScope concurrency always matches the resolved llama-server slot count: `--parallel 4` automatically uses `eval_batch_size=4`. Total context grows by the same factor so every slot retains the full profile context limit. The KV cache uses F16 for both K and V. The loopback address and automatically selected port keep the service private and avoid port collisions. `--no-context-shift` prevents generation from discarding the beginning of a benchmark prompt. Other settings use the defaults from the pinned llama.cpp revision.

## Compare results

```bash
python3 benchmark.py compare \
  runs/project-q2-k \
  runs/community-a-q2-k \
  runs/community-b-q2-k \
  --title "Qwen3.8 27B GGUF comparison" \
  --output results/qwen3.8-27b.md
```

The generated Markdown contains one comparison table. GGUF size and MTP-free size are detected from the file automatically.

| Model / GGUF | GGUF size | MTP | Size without MTP | GPQA Diamond | AIME 2026 | IFEval | BFCL v4 Quick |
|---|---:|:---:|---:|---:|---:|---:|---:|
| Project GGUF | ... | Yes | ... | ... | ... | ... | ... |
| Community GGUF | ... | No | ... | ... | ... | ... | ... |

Runs can be compared only when their profile SHA-256 values match. No HTML or composite score is generated.

## OpenAI-compatible API

```bash
export MODEL_API_KEY='...'
python3 benchmark.py run \
  --url https://provider.example/v1 \
  --model-name model-id \
  --api-key-env MODEL_API_KEY
```

The URL is the OpenAI-compatible API endpoint. `--model-name` is the API model identifier and the name shown in the report. The API key is optional and is not written to process arguments, manifests, or logs.

The bundled GPQA Diamond, AIME 2026, IFEval, BFCL, and GSM8K profiles do not require Docker. A future code-execution benchmark must run with an isolated sandbox.

Results are stored under `runs/<run-id>/`. Rebuild a report with `python3 benchmark.py report runs/<run-id>`.

## Reproduce the community baselines

The commands below download the pinned community GGUF revisions used by this comparison and evaluate every model with the same profile. Models are stored under `$HOME/models`; change that path if needed. The runner selects a safe concurrency level from the available GPU memory, or you can override it with `--parallel`.

```bash
# Unsloth
hf download unsloth/Qwen3.8-27B-GGUF \
    Qwen3.8-27B-UD-Q2_K_XL.gguf Qwen3.8-27B-UD-IQ2_S.gguf \
    --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502 \
    --local-dir "$HOME/models/unsloth/Qwen3.8-27B-GGUF" \
    --max-workers 8

python3 benchmark.py run \
  --profile profiles/qwen3.8-thinking.json \
  --gguf "$HOME/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q2_K_XL.gguf" \
  --model-name UD-Q2_K_XL \
  --run-id UD-Q2_K_XL

python3 benchmark.py run \
  --profile profiles/qwen3.8-thinking.json \
  --gguf "$HOME/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ2_S.gguf" \
  --model-name UD-IQ2_S \
  --run-id UD-IQ2_S

python3 benchmark.py run \
  --profile profiles/qwen3.8-thinking.json \
  --gguf "$HOME/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf" \
  --model-name UD-Q4_K_M \
  --run-id UD-Q4_K_M

python3 benchmark.py run \
  --profile profiles/qwen3.8-thinking.json \
  --gguf "$HOME/models/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf" \
  --model-name Q4_K_M \
  --run-id Q4_K_M

# AtomicChat
hf download AtomicChat/Qwen3.8-27B-GGUF \
    Qwen3.8-27B-AD-IQ2_XS.gguf Qwen3.8-27B-AD-IQ1_M.gguf \
    --revision ca10ebceb1887be9d33b838770a36b39d75a8a4c \
    --local-dir "$HOME/models/AtomicChat/Qwen3.8-27B-GGUF" \
    --max-workers 8

python3 benchmark.py run \
  --profile profiles/qwen3.8-thinking.json \
  --gguf "$HOME/models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ2_XS.gguf" \
  --model-name AD-IQ2_XS \
  --run-id AD-IQ2_XS

python3 benchmark.py run \
  --profile profiles/qwen3.8-thinking.json \
  --gguf "$HOME/models/AtomicChat/Qwen3.8-27B-GGUF/Qwen3.8-27B-AD-IQ1_M.gguf" \
  --model-name AD-IQ1_M \
  --run-id AD-IQ1_M
```
