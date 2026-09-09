# Reproducible GGUF Comparisons

Reproduce comparisons between project-built and community GGUF files on GPQA Diamond, AIME, IFEval, and BFCL. EvalScope 1.11.0 is the evaluation engine; this repository provides the runner and clean Markdown result tables. A 200-sample vision profile also covers RealWorldQA and OCRBench.

## Install

Python 3.10+ is required. Local GGUF evaluation also needs `llama-server` on `PATH`.

```bash
git clone https://github.com/drawthingsai/benchmarks.git
cd benchmarks

conda create -n benchmarks python=3.12
conda activate benchmarks

python3 -m pip install 'evalscope[bfcl,ifeval]==1.11.0' 'soundfile==0.13.1'
```

For local GGUF runs, use a llama.cpp build that supports your model and the server arguments below. The runner records `llama-server --version` when available; a different or unknown version does not block the run. Use the same build across runs for reproducible comparisons. Example CUDA build:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
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
| Qwen3.8 thinking (default comparison profile) | BFCL v4 1K | `bfcl_v4` | 5,106 | 1,002 | `acc` |
| Qwen3.8 thinking (non-Web full profile) | BFCL v4 — All non-Web-Search tasks | `bfcl_v4` | 5,106 | 4,906 | `acc` |

Choose a profile explicitly for a comparison run:

| Profile file | Evaluations |
|---|---|
| `profiles/qwen3.8-thinking.json` | AIME 2026, GPQA Diamond, IFEval, and BFCL v4 1K (1,002 rows) |
| `profiles/qwen3.8-thinking-bfcl-non-web.json` | AIME 2026, GPQA Diamond, IFEval, and all 4,906 BFCL v4 non-Web-Search rows |

The Qwen3.8 comparison profile now uses BFCL v4 1K, replacing its previous 200-row Quick sample; the separate `qwen3.8-thinking-bfcl-1k.json` file has been removed. The 1K configuration takes up to the first 56 examples from each of 20 selected non-Web-Search categories with shuffling disabled, including multi-turn and memory tasks. Both bundled BFCL configurations exclude Web Search, so no SerpAPI key is needed; their scores are not the official BFCL v4 Overall score, which includes Web Search.

Datasets are downloaded from Hugging Face. GPQA is gated: accept the [official dataset terms](https://huggingface.co/datasets/Idavidrein/gpqa) and run `hf auth login` before the full profile.

## Quick vision comparison (200 samples)

Use `prepare_vision_quick.py` to select **100 RealWorldQA** examples (seed 42)
and **100 OCRBench** examples (10 per category, seed 42). These are small subset
scores, not full benchmark scores. RealWorldQA measures visual understanding;
OCRBench covers text recognition and document questions. Scoring uses EvalScope's
existing adapters, with no external judge. The profile disables thinking and uses
temperature 0, a 1,024-token output limit, and four concurrent requests.

Download the pinned source data once (or use an existing local copy):

```bash
vision_data="$PWD/.workspace/cache/vision-data"
hf download xai-org/RealworldQA --repo-type dataset \
  --revision 17e7f75e092e47169732462ea3cdfebe911105dd --include 'data/*.parquet' \
  --local-dir "$vision_data/datasets/vision/xai-org/RealworldQA/17e7f75e092e47169732462ea3cdfebe911105dd/raw"
hf download echo840/OCRBench --repo-type dataset \
  --revision 92a54bd1384387c178d5a07140a2d85e0a3d12e1 --include 'data/*.parquet' \
  --local-dir "$vision_data/datasets/vision/echo840/OCRBench/92a54bd1384387c178d5a07140a2d85e0a3d12e1/raw"
python3 prepare_vision_quick.py --data-root "$vision_data" \
  --output-dir .workspace/vision-quick-data
```

The generated `samples.json` records source revisions, selected row indices,
image hashes and prepared-data hashes. WebP images are transported as lossless PNG
for llama.cpp compatibility, preserving decoded pixels and dimensions. The runner
verifies prepared-data fingerprints before evaluation. Reuse the generated profile
for every vision encoder and keep the language-model GGUF and server build fixed:

```bash
model_dir=/path/to/Qwen3.8-27B-GGUF
for format in Q6_K Q5_K Q4_K BF16 F16 Q8_0; do
  python3 benchmark.py run \
    --profile .workspace/vision-quick-data/profile.json \
    --gguf "$model_dir/Qwen3.8-27B-DT-IQ3_XXS.gguf" \
    --mmproj "$model_dir/mmproj-Qwen3.8-27B-DT-$format.gguf" \
    --image-min-tokens 64 --image-max-tokens 4096 \
    --parallel 4 --ctx-size 32768 \
    --model-name "IQ3-vision-$format" --run-id "IQ3-vision-$format" \
    --output-dir .workspace/vision-runs
done
python3 benchmark.py compare \
  .workspace/vision-runs/IQ3-vision-{Q6_K,Q5_K,Q4_K,BF16,F16,Q8_0} \
  --output .workspace/vision-comparison.md
```

BF16 is the vision baseline; the language model remains quantized in all six
runs. Image token limits bound server preprocessing while preserving aspect ratio.
Manifests also record language-model and vision GGUF hashes. Result tables include
the vision filename and size in MiB. With only 100 examples per dataset, one answer
changes its score by one percentage point; small differences need larger follow-up
runs before drawing conclusions.

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
--ctx-size PROFILE_LIMIT_TIMES_SLOTS_PER_SERVER
--parallel SLOTS_PER_SERVER
--device ONE_ACCELERATOR_PER_SERVER
--split-mode none
--n-gpu-layers 999
--cache-type-k f16
--cache-type-v f16
--cache-prompt
--jinja
--no-context-shift
--no-webui
```

For local GGUF runs, `--parallel N` sets total EvalScope concurrency. The runner starts one complete model copy per visible accelerator, capped at `N`, and distributes new conversations round-robin. Later turns with the same system and first conversation message return to the same server so llama.cpp can reuse its prompt cache. Each server is pinned to one accelerator with `--split-mode none`; a model is never split across accelerators. Restrict the devices with `CUDA_VISIBLE_DEVICES`. Context is sized per server so every slot retains the full profile limit. The KV cache uses F16 for both K and V.

Add `--mtp` to enable MTP speculative decoding for a local GGUF containing MTP weights. This passes `--spec-type draft-mtp --spec-draft-n-max 3` to each server. Use `--mtp-draft-tokens N` to change the maximum draft length. Without `--mtp`, the runner explicitly passes `--spec-type none`.

For an MTP comparison, use the same GGUF, profile, and concurrency with distinct run IDs and model names:

```bash
python3 benchmark.py run --gguf /path/to/model.gguf \
  --profile profiles/qwen3.8-thinking.json --parallel 1 \
  --run-id model-no-mtp --model-name model-no-mtp
python3 benchmark.py run --gguf /path/to/model.gguf \
  --profile profiles/qwen3.8-thinking.json --parallel 1 \
  --mtp --mtp-draft-tokens 3 --run-id model-mtp --model-name model-mtp
python3 benchmark.py compare model-no-mtp model-mtp --output results/mtp-comparison.md
```

MTP settings appear in `--dry-run` output and are recorded in the run manifest. Resume requires the same MTP settings; existing manifests without MTP settings are treated as disabled. These options apply only to `--gguf`; configure remote services separately. The report table's `MTP` column describes whether the GGUF contains MTP tensors, not whether speculative decoding was enabled.

## Compare results

```bash
python3 benchmark.py compare \
  runs/project-q2-k \
  runs/community-a-q2-k \
  runs/community-b-q2-k \
  --title "Qwen3.8 27B GGUF comparison" \
  --output results/qwen3.8-27b.md
```

Pass one run directory to summarize it, or multiple run directories to compare them. The generated Markdown contains a score table, an output-health table, and per-benchmark output-token distributions for correct and incorrect samples (mean, minimum, P5, P95, and maximum). Multi-turn token counts are summed across model calls, while samples missing API usage metadata are reported but excluded from the distribution. GGUF size and MTP-free size are detected from the file automatically.

Report and compare commands also accept bare run names under `./runs/`, for example `python3 benchmark.py report project-q2-k` or `python3 benchmark.py compare project-q2-k community-a-q2-k --output results/comparison.md`. Existing paths take precedence. The same applies to `report.py run` and `report.py compare`.

| Model / GGUF | GGUF size | MTP | Size without MTP | AIME 2026 | GPQA Diamond | IFEval | BFCL v4 1K |
|---|---:|:---:|---:|---:|---:|---:|---:|
| Project GGUF | ... | Yes | ... | ... | ... | ... | ... |
| Community GGUF | ... | No | ... | ... | ... | ... | ... |

Runs must contain the same ordered benchmark cases. Model-specific generation settings may differ and remain recorded in each run. No HTML or composite score is generated.

Existing Qwen3.8 Quick runs cannot be compared directly with runs using the updated 1K profile. Use a new run ID for the updated profile; `--resume` requires the original profile configuration.

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

Results are stored under `runs/<run-id>/`. After an interrupted run, repeat the same command with `--resume`; completed benchmarks and cached samples are reused. Rebuild a report with `python3 benchmark.py report runs/<run-id>`.

## Reproduce the community baselines

The commands below download the pinned community GGUF revisions used by this comparison and evaluate every model with the same profile. Models are stored under `$HOME/models`; change that path if needed. The profile supplies the default concurrency, or you can override it with `--parallel`.

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
