# Published comparisons

This directory contains reviewable Markdown comparisons between project-built GGUF artifacts and community GGUF baselines.

Generate a comparison from completed runs with `python3 benchmark.py compare`. Before publication, verify that every GGUF row includes its canonical source URL, revision, exact filename, byte size, SHA-256, and quantization method.

Do not hand-edit generated score tables. Preserve the matching run manifests and raw evaluator artifacts so every value can be audited and reproduced.
