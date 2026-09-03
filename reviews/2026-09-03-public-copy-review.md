# Public copy tripartite review

Date: 2026-09-03

## Subsequent user revision

After this review, the maintainer explicitly simplified the public interface. The package entry point and virtual-environment instructions were removed in favor of installing EvalScope directly and running `./run.sh`. The README was shortened accordingly. This direct maintainer decision supersedes the earlier CLI packaging decision recorded below.

## Review brief

The reviewed copy described a public repository for reproducible comparisons between project-built GGUF artifacts and established community releases. The required product emphasis was reproducibility, simple execution, exact local artifact hashes, stated upstream provenance, fixed benchmark profiles, visible coverage and failures, and Markdown comparison tables. EvalScope 1.11.0 was specified as the pinned evaluation engine, not the identity of the repository.

The reviewers received a self-contained dossier containing the product intent, package description, full draft README, the author's independent assessment, and six questions covering positioning, natural English, credibility, ambiguity, exact revisions, and release readiness. They were instructed not to inspect files, browse, or use tools.

## Author's independent assessment

The draft correctly made reproducible GGUF comparison the subject and positioned EvalScope as the pinned evaluator. It front-loaded the publication contract before installation details, separated individual execution from multi-GGUF comparison, and explained why profile identity and artifact provenance matter.

The author identified three remaining risks: the opening was abstract, the repository did not yet have a real published score table, and the README spent too much time on safeguards before showing the community comparison workflow.

## AGY independent review

Reviewer: AGY, `gemini-3.7-flash-high`

Verdict: **Approve with edits**.

AGY found the title, package description, and core positioning clear. It agreed that the repository was about reproducible GGUF comparison rather than EvalScope. Its material findings were:

1. The sentence saying that EvalScope was "not the project itself" sounded defensive. The dependency should be described affirmatively as the pinned evaluation engine.
2. The main reader journey should be run a GGUF, compare GGUFs, and inspect results. The comparison section should therefore appear before the secondary OpenAI-compatible endpoint workflow.
3. The draft mixed `gguf-bench` commands with direct `python report.py` commands. The public interface should be consistent.
4. A prose section describing the intended comparison format should be replaced with a rendered Markdown table that uses unmistakable placeholders rather than invented benchmark results.
5. Security and strict-parser implementation details should be shortened where they distract from the public workflow.
6. The profile contract and absence of a composite score were credible and useful, but the README should show the central deliverable sooner.

AGY proposed a compact quickstart, an example score matrix plus provenance table, a later "Additional capabilities" section for APIs, and simpler wording around the version pin and credential handling.

## Codex independent review

Reviewer: Codex, high reasoning effort

Verdict: **Approve with edits**.

Codex independently agreed that the purpose was clear and that EvalScope was correctly treated as infrastructure. It found no release-blocking copy problem. Its material findings were:

1. The opening should say directly that the project compares project-built and established community GGUF artifacts with fixed profiles.
2. `Reproducible GGUF Comparisons` was a sharper title than `Reproducible GGUF Benchmarks` if comparison was the primary identity.
3. The comparison workflow should move earlier, ahead of endpoint details.
4. Computed identity and declared provenance must be distinguished. Filename, byte size, and SHA-256 are computed locally; source URL, revision, and quantization description are recorded claims rather than independently verified origin.
5. Phrases such as "complete artifact identity", "primary deliverable", "publication contract", and "evaluated bits" were too abstract or stronger than the evidence supported.
6. The output section should render a placeholder comparison matrix and explicitly retain missing values instead of showing fabricated scores.
7. The CLI should use one installed command consistently, and the relationship between run IDs and comparison input directories should be explicit.

Codex recommended keeping the package description `Reproducible GGUF benchmarking and community model comparisons` unchanged.

## Adjudication

| Issue | Decision | Result |
|---|---|---|
| Repository identity | Accept both reviewers | Title changed to `Reproducible GGUF Comparisons`; opening begins with project and community GGUF comparison. |
| EvalScope wording | Accept both reviewers | EvalScope is described positively as the pinned evaluation engine in the second paragraph. |
| Section order | Accept both reviewers | Single-GGUF execution flows directly into project/community comparison; API evaluation follows later. |
| CLI consistency | Accept both reviewers | Public examples use the installed `gguf-bench` command for run, doctor, report, and compare. |
| Run-directory clarity | Accept Codex | Examples set explicit run IDs and pass the resulting `runs/<run-id>` directories to compare. |
| Artifact claims | Accept Codex | Copy now says exact file hashes and **stated** provenance, avoiding any claim that user-supplied source metadata is independently verified. |
| Example output | Accept both reviewers with one constraint | Added a rendered score matrix containing em-dash placeholders only; no synthetic scores are presented as results. |
| Clone command | Strengthen beyond the draft | Replaced the placeholder with the real public repository URL. |
| Security explanation | Partially accept AGY | Kept the user-relevant guarantee that credentials are absent from evaluator arguments, environment, manifests, and logs; removed the incorrect shell-history claim. |
| Strict profile validation | Retain, condensed | The accepted schema boundaries remain documented because they are part of the reproducibility contract. |
| HTML and composite metrics | Retain | Human-readable result artifacts remain Markdown, and no composite score is calculated. |

## Final decision

Approved after edits. The final copy leads with reproducible project-versus-community GGUF comparisons, presents EvalScope only as the pinned engine, shows the core workflow and output shape early, and limits provenance claims to what the tool can actually compute or record.
