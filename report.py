#!/usr/bin/env python3
"""Build trustworthy Markdown reports and cross-GGUF comparison tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


class BenchError(RuntimeError):
    """An expected error suitable for direct display in the CLI."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BenchError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                           parse_constant=lambda value: (_ for _ in ()).throw(
                               BenchError(f"Non-finite JSON number: {value}")))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchError(f"The top-level JSON value must be an object: {path}")
    _validate_finite(value, str(path))
    return value


def _validate_finite(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise BenchError(f"Non-finite JSON number in {label}")
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite(item, f"{label}[{index}]")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _positive_int(value: Any, label: str, *, zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise BenchError(f"{label} must be {'a non-negative' if zero else 'a positive'} integer")
    return value


def load_profile(path: Path) -> dict[str, Any]:
    profile = load_json(path)
    allowed = {"schema_version", "suite", "generation", "evalscope", "cases"}
    unknown = set(profile) - allowed
    if unknown:
        raise BenchError(f"Unknown profile field(s): {', '.join(sorted(unknown))}")
    if profile.get("schema_version") != 1:
        raise BenchError("Only profile.schema_version=1 is supported")
    suite = profile.get("suite")
    if not isinstance(suite, dict) or not isinstance(suite.get("id"), str):
        raise BenchError("profile.suite.id is required")
    generation = profile.get("generation", {})
    settings = profile.get("evalscope", {})
    cases = profile.get("cases")
    if not isinstance(generation, dict) or not isinstance(settings, dict):
        raise BenchError("profile.generation and profile.evalscope must be objects")
    if not isinstance(cases, list) or not cases:
        raise BenchError("profile.cases must be a non-empty array")
    seen: set[str] = set()
    for index, case in enumerate(cases):
        label = f"profile.cases[{index}]"
        if not isinstance(case, dict):
            raise BenchError(f"{label} must be an object")
        allowed_case = {"id", "dataset", "primary_metric", "expected_samples", "limit",
                        "repeats", "dataset_args", "generation", "sandbox_required"}
        unknown_case = set(case) - allowed_case
        if unknown_case:
            raise BenchError(f"Unknown field(s) in {label}: {', '.join(sorted(unknown_case))}")
        case_id = case.get("id")
        safe_id = isinstance(case_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", case_id)
        if not safe_id or case_id in {".", ".."} or case_id in seen:
            raise BenchError(f"{label}.id must be unique and file-system safe")
        seen.add(case_id)
        if not isinstance(case.get("dataset"), str):
            raise BenchError(f"{label}.dataset is required")
        metric = case.get("primary_metric")
        if not isinstance(metric, dict) or not isinstance(metric.get("name"), str):
            raise BenchError(f"{label}.primary_metric.name is required")
        _positive_int(case.get("expected_samples"), f"{label}.expected_samples")
        if case.get("limit") is not None:
            _positive_int(case["limit"], f"{label}.limit")
        _positive_int(case.get("repeats", 1), f"{label}.repeats")
        if not isinstance(case.get("generation", {}), dict) \
                or not isinstance(case.get("dataset_args", {}), dict):
            raise BenchError(f"{label}.generation and {label}.dataset_args must be objects")
        resolved = dict(generation)
        resolved.update(case.get("generation", {}))
        _positive_int(resolved.get("max_tokens"), f"{label}.generation.max_tokens")
        for key, value in resolved.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise BenchError(f"{label}.generation.{key} must be finite")
        case["resolved_generation"] = resolved
        case["planned_samples"] = case["expected_samples"] * case.get("repeats", 1)
        case.setdefault("dataset_args", {})
    return profile


def _jsonl(paths: Iterable[Path]) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise BenchError(f"Invalid JSONL in {path}:{line_no}: {exc}") from exc
                if isinstance(value, dict):
                    yield path, value


def _category_name(value: Any) -> str:
    return "/".join(str(item) for item in value) if isinstance(value, list) else str(value)


def _metric_names(metric: dict[str, Any]) -> set[str]:
    """Return EvalScope v1 and v2 metric names plus legacy profile aliases."""
    names = {value for value in (metric.get("name"), metric.get("legacy_name"))
             if isinstance(value, str)}
    identity = metric.get("identity")
    if isinstance(identity, dict) and isinstance(identity.get("name"), str):
        names.add(identity["name"])
    aliases = set(names)
    for name in names:
        aliases.add(name.removeprefix("mean_"))
        aliases.add(f"mean_{name}")
        if name in {"acc", "accuracy"}:
            aliases.update({"acc", "accuracy", "mean_acc", "mean_accuracy"})
    return aliases


def select_primary(reports: list[dict[str, Any]], selector: dict[str, Any],
                   default_dataset: str) -> dict[str, Any]:
    dataset = selector.get("dataset", default_dataset)
    matches: list[dict[str, Any]] = []
    for report in reports:
        if report.get("dataset_name") != dataset:
            continue
        for metric in report.get("metrics", []):
            if not isinstance(metric, dict) or selector["name"] not in _metric_names(metric):
                continue
            identity = metric.get("identity")
            metric_name = metric.get("name")
            if not isinstance(metric_name, str) and isinstance(identity, dict):
                metric_name = identity.get("name")
            if not isinstance(metric_name, str):
                metric_name = selector["name"]
            if selector.get("category") is None and selector.get("subset") is None:
                matches.append({"dataset": dataset, "name": metric_name, "category": None,
                                "subset": None, "score": metric.get("score"), "num": metric.get("num"),
                                "direction": selector.get("direction"), "scale": selector.get("scale")})
                continue
            for category in metric.get("categories", []):
                name = _category_name(category.get("name"))
                if selector.get("category") is not None and name != selector["category"]:
                    continue
                if selector.get("subset") is None:
                    matches.append({"dataset": dataset, "name": metric_name, "category": name,
                                    "subset": None, "score": category.get("score"),
                                    "num": category.get("num"), "direction": selector.get("direction"),
                                    "scale": selector.get("scale")})
                else:
                    for subset in category.get("subsets", []):
                        if subset.get("name") == selector["subset"]:
                            matches.append({"dataset": dataset, "name": metric_name,
                                            "category": name, "subset": selector["subset"],
                                            "score": subset.get("score"), "num": subset.get("num"),
                                            "direction": selector.get("direction"),
                                            "scale": selector.get("scale")})
    if len(matches) != 1:
        raise BenchError(f"Primary metric selector matched {len(matches)} values; expected exactly one")
    score = matches[0].get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise BenchError("Primary metric has no numeric score")
    return matches[0]


def _content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(str(item.get("text", "")) for item in value
                       if isinstance(item, dict) and item.get("type") in {"text", "output_text"})
    return ""


def prediction_health(root: Path, max_tokens: int) -> dict[str, int]:
    counts = {"prediction_rows": 0, "responded": 0, "nonempty": 0, "parsed": 0,
              "errors": 0, "capped": 0, "unique": 0, "duplicates": 0}
    identities: list[str] = []
    for path, row in _jsonl(sorted(root.glob("**/predictions/**/*.jsonl"))):
        counts["prediction_rows"] += 1
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        row_id = row.get("index", metadata.get("id", counts["prediction_rows"]))
        identities.append(f"{path.relative_to(root)}:{row_id}")
        output = row.get("model_output") or {}
        if not isinstance(output, dict):
            counts["errors"] += 1
            continue
        choices = output.get("choices") or []
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            counts["errors"] += 1
            continue
        counts["responded"] += 1
        choice = choices[0]
        message = choice.get("message") or {}
        if not isinstance(message, dict):
            counts["errors"] += 1
            continue
        text = _content(message.get("content"))
        if text.strip():
            counts["nonempty"] += 1
        error = bool(output.get("error"))
        if error:
            counts["errors"] += 1
        elif text.strip() or message.get("tool_calls"):
            counts["parsed"] += 1
        usage = output.get("usage") or {}
        tokens = usage.get("output_tokens", usage.get("completion_tokens"))
        if choice.get("stop_reason") == "length" or choice.get("finish_reason") == "length" \
                or isinstance(tokens, int) and tokens >= max_tokens:
            counts["capped"] += 1
    counts["unique"] = len(set(identities))
    counts["duplicates"] = len(identities) - counts["unique"]
    return counts


def summarize_run(run_dir: Path) -> dict[str, Any]:
    profile_path = run_dir / "profile.json"
    profile = load_profile(profile_path)
    manifest = load_json(run_dir / "manifest.json")
    recorded_hash = (manifest.get("profile") or {}).get("sha256")
    if recorded_hash and recorded_hash != sha256_file(profile_path):
        raise BenchError(f"Profile hash does not match the run manifest: {run_dir}")
    results = []
    for case in profile["cases"]:
        attempt_dir = run_dir / "cases" / case["id"] / "attempt-0001"
        alerts: list[str] = []
        attempt = load_json(attempt_dir / "attempt.json") if attempt_dir.exists() else {}
        root = attempt_dir / "evalscope"
        reports = [load_json(path) for path in sorted(root.glob("**/reports/**/*.json"))]
        try:
            primary = select_primary(reports, case["primary_metric"], case["dataset"])
        except BenchError as exc:
            primary = None
            alerts.append(str(exc))
        health = prediction_health(root, case["resolved_generation"]["max_tokens"])
        planned = case["planned_samples"]
        coverage = {"planned": planned, **health,
                    "scored": primary.get("num") if primary else None,
                    "missing": max(0, planned - health["unique"])}
        if attempt.get("return_code") != 0:
            status = "failed"
            alerts.append(f"EvalScope exited with code {attempt.get('return_code', 'missing')}")
        elif primary is None:
            status = "failed"
        elif coverage["missing"] or isinstance(coverage["scored"], bool) \
                or not isinstance(coverage["scored"], int) \
                or coverage["scored"] < planned:
            status = "incomplete"
            alerts.append("Coverage is below the planned sample count")
        elif health["errors"] or health["capped"] or health["duplicates"]:
            status = "warning"
            alerts.append("Response health checks found errors, token caps, or duplicate samples")
        else:
            status = "pass"
        results.append({"case_id": case["id"], "dataset": case["dataset"], "status": status,
                        "primary_metric": primary, "coverage": coverage, "alerts": alerts})
    rank = {"pass": 0, "warning": 1, "incomplete": 2, "failed": 3}
    overall = max((item["status"] for item in results), key=lambda value: rank[value])
    run_id = manifest.get("run_id") if isinstance(manifest.get("run_id"), str) else run_dir.name
    return {"schema_version": 1, "generated_at": utc_now(), "run_id": run_id,
            "suite": profile["suite"], "status": overall, "manifest": manifest, "cases": results}


def _score(metric: dict[str, Any] | None) -> str:
    value = metric.get("score") if metric else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    return f"{value * 100:.2f}%" if metric.get("scale") == "fraction" else f"{value:.4f}"


def _size(value: Any) -> str:
    return f"{value / 2**30:.2f} GiB" if isinstance(value, int) else "—"


def _result_table(summaries: list[dict[str, Any]]) -> list[str]:
    cases = summaries[0]["cases"]
    benchmark_headers = []
    for index, case in enumerate(cases):
        metric = next((summary["cases"][index].get("primary_metric") for summary in summaries
                       if summary["cases"][index].get("primary_metric")), None)
        benchmark_headers.append(
            f"{case['case_id']} ({metric['name'] if metric else 'score'})")
    headers = ["Model / GGUF", "GGUF size", "MTP", "Size without MTP", *benchmark_headers]
    lines = ["| " + " | ".join(_cell(value) for value in headers) + " |",
             "|---|---:|:---:|---:|" + "---:|" * len(benchmark_headers)]
    for summary in summaries:
        backend = summary["manifest"].get("backend", {})
        artifact = backend.get("gguf", {})
        has_mtp = artifact.get("has_mtp")
        mtp = "Yes" if has_mtp is True else "No" if has_mtp is False else "—"
        values = [backend.get("model"), _size(artifact.get("size_bytes")), mtp,
                  _size(artifact.get("size_without_mtp_bytes")),
                  *(_score(case.get("primary_metric")) for case in summary["cases"])]
        lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    return lines


def markdown(summary: dict[str, Any]) -> str:
    lines = [f"# {summary['suite'].get('title', summary['suite']['id'])}", "",
             *_result_table([summary]), ""]
    return "\n".join(lines)


def write_reports(run_dir: Path, summary: dict[str, Any]) -> None:
    write_json(run_dir / "summary.json", summary)
    (run_dir / "report.md").write_text(markdown(summary), encoding="utf-8")


def _cell(value: Any) -> str:
    return str(value if value not in {None, ""} else "—").replace("|", "\\|").replace("\n", " ")


def comparison(run_dirs: list[Path], title: str | None = None) -> str:
    if not run_dirs:
        raise BenchError("At least one run directory is required")
    summaries = []
    for run_dir in run_dirs:
        summaries.append(summarize_run(run_dir))
    profile_hashes = {item["manifest"]["profile"]["sha256"] for item in summaries}
    if len(profile_hashes) != 1:
        raise BenchError("Compared runs must use byte-identical profiles")
    case_ids = [[case["case_id"] for case in item["cases"]] for item in summaries]
    if any(ids != case_ids[0] for ids in case_ids[1:]):
        raise BenchError("Compared runs do not contain the same ordered benchmark cases")
    labels = [item["manifest"]["backend"]["model"] for item in summaries]
    if len(set(labels)) != len(labels):
        raise BenchError("Compared runs must have unique model names; set --model-name when running")

    resolved_title = title or (
        "GGUF benchmark results" if len(summaries) == 1 else "GGUF benchmark comparison")
    lines = [f"# {resolved_title}", "", *_result_table(summaries), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Markdown GGUF benchmark reports")
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser("run", help="rebuild one run report")
    single.add_argument("run_dir", type=Path)
    compare = commands.add_parser(
        "compare", help="summarize or compare one or more completed runs")
    compare.add_argument("run_dirs", nargs="+", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--title")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            run_dir = args.run_dir.expanduser().resolve()
            summary = summarize_run(run_dir)
            write_reports(run_dir, summary)
            print(f"Reports written to {run_dir}")
        else:
            run_dirs = [path.expanduser().resolve() for path in args.run_dirs]
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(comparison(run_dirs, args.title), encoding="utf-8")
            label = "Report" if len(run_dirs) == 1 else "Comparison"
            print(f"{label} written to {output}")
        return 0
    except BenchError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
