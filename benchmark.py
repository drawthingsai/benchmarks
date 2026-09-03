#!/usr/bin/env python3
"""Run reproducible GGUF benchmarks with EvalScope 1.11 as the evaluator."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import importlib.metadata
import json
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import AbstractContextManager, nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import report

ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE = ROOT / "profiles" / "smoke.json"
LLAMA_CPP_COMMIT = "0df974d777c904dda1da3b00faa7769c6310ae74"
HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
               "te", "trailer", "transfer-encoding", "upgrade"}
GGUF_SCALARS = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
    7: "?", 10: "Q", 11: "q", 12: "d",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _gguf_read(handle: Any, size: int) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise report.BenchError("The GGUF header is truncated")
    return value


def _gguf_string(handle: Any) -> str:
    length = struct.unpack("<Q", _gguf_read(handle, 8))[0]
    return _gguf_read(handle, length).decode("utf-8", errors="replace")


def _gguf_skip_value(handle: Any, value_type: int) -> None:
    if value_type in GGUF_SCALARS:
        handle.seek(struct.calcsize("<" + GGUF_SCALARS[value_type]), os.SEEK_CUR)
    elif value_type == 8:
        handle.seek(struct.unpack("<Q", _gguf_read(handle, 8))[0], os.SEEK_CUR)
    elif value_type == 9:
        item_type, count = struct.unpack("<IQ", _gguf_read(handle, 12))
        if item_type in GGUF_SCALARS:
            handle.seek(struct.calcsize("<" + GGUF_SCALARS[item_type]) * count, os.SEEK_CUR)
        else:
            for _ in range(count):
                _gguf_skip_value(handle, item_type)
    else:
        raise report.BenchError(f"Unsupported GGUF value type: {value_type}")


def gguf_inventory(path: Path) -> dict[str, Any]:
    """Read file size and MTP payload size without loading tensor data."""
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        if _gguf_read(handle, 4) != b"GGUF":
            raise report.BenchError(f"Not a GGUF file: {path}")
        version = struct.unpack("<I", _gguf_read(handle, 4))[0]
        if version not in {2, 3}:
            raise report.BenchError(f"Unsupported GGUF version {version}: {path}")
        tensor_count, field_count = struct.unpack("<QQ", _gguf_read(handle, 16))
        alignment = 32
        for _ in range(field_count):
            key = _gguf_string(handle)
            value_type = struct.unpack("<I", _gguf_read(handle, 4))[0]
            if key == "general.alignment" and value_type == 4:
                alignment = struct.unpack("<I", _gguf_read(handle, 4))[0]
            else:
                _gguf_skip_value(handle, value_type)
        tensors: list[tuple[str, int]] = []
        for _ in range(tensor_count):
            name = _gguf_string(handle)
            dimensions = struct.unpack("<I", _gguf_read(handle, 4))[0]
            handle.seek(8 * dimensions + 4, os.SEEK_CUR)
            offset = struct.unpack("<Q", _gguf_read(handle, 8))[0]
            tensors.append((name, offset))
        data_offset = (handle.tell() + alignment - 1) // alignment * alignment

    mtp_layers = {
        int(match.group(1))
        for name, _ in tensors
        if (match := re.match(r"^blk\.(\d+)\..*nextn", name))
    }
    ordered = sorted(tensors, key=lambda item: item[1])
    mtp_bytes = 0
    for index, (name, offset) in enumerate(ordered):
        next_offset = ordered[index + 1][1] if index + 1 < len(ordered) else file_size - data_offset
        match = re.match(r"^blk\.(\d+)\.", name)
        if match and int(match.group(1)) in mtp_layers:
            mtp_bytes += max(0, next_offset - offset)
    return {
        "path": str(path.resolve()),
        "size_bytes": file_size,
        "has_mtp": bool(mtp_layers),
        "mtp_bytes": mtp_bytes,
        "size_without_mtp_bytes": file_size - mtp_bytes,
    }


def llama_server_identity(binary: str) -> str:
    try:
        result = subprocess.run([binary, "--version"], capture_output=True, text=True,
                                timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise report.BenchError(f"Could not read llama-server version: {exc}") from exc
    identity = " ".join(f"{result.stdout}\n{result.stderr}".split())
    if result.returncode or LLAMA_CPP_COMMIT[:9] not in identity:
        raise report.BenchError(
            f"llama-server must be built from llama.cpp {LLAMA_CPP_COMMIT}; found: "
            f"{identity or 'unknown version'}")
    return identity


def accelerator_count(binary: str) -> int:
    """Count accelerators visible to llama.cpp, falling back to CPU-only execution."""
    try:
        result = subprocess.run([binary, "--list-devices"], capture_output=True, text=True,
                                timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return 1
    if result.returncode:
        return 1
    devices = re.findall(r"^\s+\S+\d+:\s", f"{result.stdout}\n{result.stderr}", re.MULTILINE)
    return max(1, len(devices))


def terminate_process_group(process: subprocess.Popen[Any], grace_seconds: float = 10) -> None:
    """Stop a child session and every process it spawned."""
    pgid = process.pid
    previous_sigint: Any = None
    protect_cleanup = threading.current_thread() is threading.main_thread()
    if protect_cleanup:
        previous_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            process.poll()
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    finally:
        if protect_cleanup:
            signal.signal(signal.SIGINT, previous_sigint)


def safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    return normalized[:80] or "run"


def validate_url(value: str) -> urllib.parse.SplitResult:
    try:
        parsed = urllib.parse.urlsplit(value)
        _ = parsed.port
    except ValueError as exc:
        raise report.BenchError("--url must be a valid HTTP(S) OpenAI-compatible base URL") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise report.BenchError("--url must be a valid HTTP(S) OpenAI-compatible base URL")
    if parsed.username or parsed.password:
        raise report.BenchError("Credentials are not allowed in --url; use --api-key-env")
    return parsed


def redact_url(value: str) -> str:
    parsed = validate_url(value)
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    query = urllib.parse.urlencode(
        [(key, "<redacted>") for key, _ in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)])
    return urllib.parse.urlunsplit((parsed.scheme, host, parsed.path.rstrip("/"), query, ""))


def resolve_key(name: str | None) -> str | None:
    if not name:
        return None
    value = os.environ.get(name)
    if not value:
        raise report.BenchError(f"Environment variable {name} is unset or empty")
    return value


def request(base_url: str, path: str, *, key: str | None = None,
            key_header: str = "Authorization", key_prefix: str = "Bearer ",
            method: str = "GET", body: dict[str, Any] | None = None,
            timeout: float = 10) -> tuple[int, Any]:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    headers, raw = {"Accept": "application/json"}, None
    if key:
        headers[key_header] = f"{key_prefix}{key}"
    if body is not None:
        raw = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=raw, headers=headers, method=method),
                                    timeout=timeout) as response:
            payload = response.read()
            try:
                decoded = json.loads(payload)
            except json.JSONDecodeError:
                decoded = payload.decode(errors="replace")
            return response.status, decoded
    except urllib.error.HTTPError as exc:
        raise report.BenchError(f"Endpoint returned HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        reason = getattr(exc, "reason", exc)
        raise report.BenchError(f"Cannot reach endpoint: {type(reason).__name__}") from None


def probe(base_url: str, *, key: str | None = None, key_header: str = "Authorization",
          key_prefix: str = "Bearer ", model: str | None = None,
          generate: bool = False) -> dict[str, Any]:
    status, payload = request(base_url, "models", key=key, key_header=key_header,
                              key_prefix=key_prefix)
    models = [str(item.get("id")) for item in payload.get("data", [])
              if isinstance(item, dict) and item.get("id")] if isinstance(payload, dict) else []
    result: dict[str, Any] = {"http_status": status, "models": models}
    if model and models and model not in models:
        result["warning"] = f"The requested model {model!r} was not listed by /models"
    if generate:
        if not model:
            raise report.BenchError("A generation probe requires --model-name")
        result["generation_http_status"], _ = request(
            base_url, "chat/completions", key=key, key_header=key_header, key_prefix=key_prefix,
            method="POST", timeout=60,
            body={"model": model, "messages": [{"role": "user", "content": "Reply with OK."}],
                  "temperature": 0, "max_tokens": 8})
    return result


class AuthProxy(AbstractContextManager["AuthProxy"]):
    """Forward requests while keeping the real credential out of child processes."""

    def __init__(self, target_url: str, key: str | None, header: str, prefix: str) -> None:
        self.target = validate_url(target_url)
        self.key, self.header, self.prefix = key, header, prefix
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        if not self.server:
            raise RuntimeError("Authentication proxy has not started")
        return f"http://127.0.0.1:{self.server.server_address[1]}{self.target.path.rstrip('/')}"

    def __enter__(self) -> "AuthProxy":
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, _format: str, *_args: object) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802
                self.forward()

            def do_POST(self) -> None:  # noqa: N802
                self.forward()

            def forward(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length else None
                excluded = HOP_HEADERS | {"host", "content-length", "authorization",
                                          "proxy-authorization", owner.header.lower()}
                headers = {key: value for key, value in self.headers.items()
                           if key.lower() not in excluded}
                if owner.key:
                    headers[owner.header] = f"{owner.prefix}{owner.key}"
                incoming = urllib.parse.urlsplit(self.path)
                base = owner.target.path.rstrip("/")
                suffix = incoming.path[len(base):] if base and incoming.path.startswith(base) else incoming.path
                query = urllib.parse.urlencode(
                    urllib.parse.parse_qsl(owner.target.query, keep_blank_values=True)
                    + urllib.parse.parse_qsl(incoming.query, keep_blank_values=True))
                path = f"{base}/{suffix.lstrip('/')}" + (f"?{query}" if query else "")
                cls = http.client.HTTPSConnection if owner.target.scheme == "https" else http.client.HTTPConnection
                connection = cls(owner.target.hostname, owner.target.port, timeout=3600)
                try:
                    connection.request(self.command, path, body=body, headers=headers)
                    upstream = connection.getresponse()
                    self.send_response(upstream.status, upstream.reason)
                    for key, value in upstream.getheaders():
                        if key.lower() not in HOP_HEADERS | {"content-length"}:
                            self.send_header(key, value)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    while chunk := upstream.read(64 * 1024):
                        self.wfile.write(chunk)
                finally:
                    connection.close()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.thread:
            self.thread.join(timeout=3)


class LlamaServer(AbstractContextManager["LlamaServer"]):
    """Manage one loopback-only llama.cpp server for a complete suite."""

    def __init__(self, args: argparse.Namespace, model: str, log: Path) -> None:
        self.args, self.model, self.log = args, model, log
        self.process: subprocess.Popen[bytes] | None = None
        self.handle = None
        self.port = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self) -> "LlamaServer":
        binary = shutil.which(self.args.llama_server)
        if not binary:
            raise report.BenchError(f"llama-server was not found: {self.args.llama_server}")
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.log.open("ab")
        try:
            error = "startup timeout"
            for attempt in range(1, 4):
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    self.port = int(sock.getsockname()[1])
                argv = [binary, "--model", str(self.args.gguf.resolve()), "--alias", self.model,
                        "--host", "127.0.0.1", "--port", str(self.port),
                        "--ctx-size", str(self.args.ctx_size), "--parallel", str(self.args.parallel),
                        "--n-gpu-layers", str(self.args.gpu_layers),
                        "--cache-type-k", "f16", "--cache-type-v", "f16",
                        "--jinja", "--no-context-shift", "--no-webui"]
                self.process = subprocess.Popen(argv, stdout=self.handle, stderr=subprocess.STDOUT,
                                                start_new_session=True)
                deadline = time.monotonic() + self.args.startup_timeout
                while time.monotonic() < deadline:
                    if self.process.poll() is not None:
                        error = f"exit code {self.process.returncode} on attempt {attempt}/3"
                        break
                    try:
                        probe(self.url, model=self.model)
                        return self
                    except report.BenchError as exc:
                        error = str(exc)
                        time.sleep(1)
                if self.process.poll() is None:
                    break
            raise report.BenchError(f"llama-server failed to start ({error}); see {self.log}")
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *_args: object) -> None:
        if self.process:
            terminate_process_group(self.process, grace_seconds=15)
        if self.handle:
            self.handle.close()


def evalscope_command(profile: dict[str, Any], case: dict[str, Any], model: str,
                      api_url: str, attempt: Path) -> list[str]:
    executable = shutil.which("evalscope") or "evalscope"
    settings = profile.get("evalscope", {})
    argv = [executable, "eval", "--model", model, "--api-url", api_url,
            "--eval-type", "openai_api", "--eval-backend", "Native",
            "--datasets", case["dataset"],
            "--dataset-hub", str(settings.get("dataset_hub", "modelscope")),
            "--dataset-args", json.dumps({case["dataset"]: case["dataset_args"]}, separators=(",", ":")),
            "--seed", str(settings.get("seed", 42)),
            "--eval-batch-size", str(settings.get("eval_batch_size", 1)),
            "--generation-config", json.dumps(case["resolved_generation"], separators=(",", ":")),
            "--work-dir", str(attempt / "evalscope"), "--enable-progress-tracker"]
    if settings.get("dataset_dir"):
        argv.extend(["--dataset-dir", str(Path(settings["dataset_dir"]).expanduser())])
    if case.get("limit"):
        argv.extend(["--limit", str(case["limit"])])
    if case.get("repeats", 1) != 1:
        argv.extend(["--repeats", str(case["repeats"])])
    if case.get("sandbox_required"):
        argv.append("--use-sandbox")
    argv.append("--collect-perf" if settings.get("collect_perf", True) else "--no-collect-perf")
    return argv


def run_case(profile: dict[str, Any], case: dict[str, Any], model: str, api_url: str,
             attempt: Path, key_env: str | None) -> int:
    attempt.mkdir(parents=True)
    argv = evalscope_command(profile, case, model, api_url, attempt)
    record = {"schema_version": 1, "case_id": case["id"], "attempt": 1,
              "started_at": report.utc_now(), "command": argv}
    report.write_json(attempt / "attempt.json", record)
    environment = os.environ.copy()
    if key_env:
        environment.pop(key_env, None)
    with (attempt / "evaluator.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(argv, cwd="/tmp", env=environment, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(f"[{case['id']}] {line}", end="")
            code = process.wait()
        except BaseException as exc:
            terminate_process_group(process)
            record.update({"finished_at": report.utc_now(), "return_code": process.returncode,
                           "interrupted_by": type(exc).__name__})
            report.write_json(attempt / "attempt.json", record)
            raise
        finally:
            if process.stdout:
                process.stdout.close()
    record.update({"finished_at": report.utc_now(), "return_code": code})
    report.write_json(attempt / "attempt.json", record)
    return code


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Reproducible GGUF benchmark runner powered by EvalScope 1.11")
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a benchmark profile")
    run.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    backend = run.add_mutually_exclusive_group(required=True)
    backend.add_argument("--gguf", type=Path)
    backend.add_argument("--url")
    run.add_argument("--model-name", help="display name and remote API model name")
    run.add_argument("--api-key-env", help="environment variable containing the API key")
    run.add_argument("--api-key-header", default="Authorization")
    run.add_argument("--api-key-prefix", default="Bearer ")
    run.add_argument("--output-dir", type=Path, default=Path("runs"))
    run.add_argument("--run-id")
    run.add_argument("--llama-server", default="llama-server")
    run.add_argument("--ctx-size", type=int,
                     help="total server context; defaults to profile limit times resolved slots")
    run.add_argument("--parallel", type=int,
                     help="server slots; defaults to evaluator concurrency and visible accelerators, capped at four")
    run.add_argument("--gpu-layers", type=int, default=999)
    run.add_argument("--startup-timeout", type=int, default=300)
    run.add_argument("--dry-run", action="store_true")
    doctor = commands.add_parser("doctor", help="probe a remote endpoint")
    doctor.add_argument("--url", required=True)
    doctor.add_argument("--model-name")
    doctor.add_argument("--api-key-env")
    doctor.add_argument("--api-key-header", default="Authorization")
    doctor.add_argument("--api-key-prefix", default="Bearer ")
    doctor.add_argument("--generate", action="store_true",
                        help="send one minimal generation request; this may incur API charges")
    rebuild = commands.add_parser("report", help="rebuild one completed run report")
    rebuild.add_argument("run_dir", type=Path)
    compare = commands.add_parser("compare", help="compare two or more completed runs")
    compare.add_argument("run_dirs", nargs="+", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--title", default="GGUF benchmark comparison")
    return root


def run(args: argparse.Namespace) -> int:
    profile = report.load_profile(args.profile.resolve())
    key = resolve_key(args.api_key_env)
    if args.gguf:
        if not args.gguf.is_file():
            raise report.BenchError(f"GGUF file does not exist: {args.gguf}")
        binary = shutil.which(args.llama_server)
        if not binary:
            raise report.BenchError(f"llama-server was not found: {args.llama_server}")
        model = args.model_name or args.gguf.name
        required = max(case["resolved_generation"]["max_tokens"] for case in profile["cases"])
        if args.parallel is None:
            batch_size = profile.get("evalscope", {}).get("eval_batch_size", 1)
            if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
                raise report.BenchError("profile.evalscope.eval_batch_size must be a positive integer")
            args.parallel = min(4, batch_size, accelerator_count(binary))
        if args.parallel <= 0:
            raise report.BenchError("--parallel must be positive")
        if args.ctx_size is None:
            args.ctx_size = required * args.parallel
        if args.ctx_size <= 0 or args.parallel <= 0 or args.ctx_size % args.parallel:
            raise report.BenchError("--ctx-size must be positive and divisible by --parallel")
        if args.ctx_size // args.parallel < required:
            minimum_total = required * args.parallel
            raise report.BenchError(
                f"Per-slot context is {args.ctx_size // args.parallel:,}; the profile requires "
                f"at least {required:,}. Remove --ctx-size and --parallel to use GPU-aware "
                f"defaults, or set --ctx-size to at least {minimum_total:,} for "
                f"--parallel {args.parallel}.")
        public_endpoint = "managed local llama.cpp endpoint"
    else:
        if not args.model_name:
            raise report.BenchError("--model-name is required with --url")
        validate_url(args.url)
        model, public_endpoint = args.model_name, redact_url(args.url)
    plan = {"profile": str(args.profile.resolve()), "suite": profile["suite"],
            "backend": "gguf" if args.gguf else "url", "model": model,
            "endpoint": public_endpoint,
            "cases": [{"id": case["id"], "planned_samples": case["planned_samples"],
                       "generation": case["resolved_generation"]} for case in profile["cases"]]}
    if args.gguf:
        plan["server"] = {"parallel": args.parallel, "ctx_size": args.ctx_size,
                          "slot_context": args.ctx_size // args.parallel}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    try:
        version = importlib.metadata.version("evalscope")
    except importlib.metadata.PackageNotFoundError:
        raise report.BenchError(
            "EvalScope is not installed; run: python3 -m pip install evalscope==1.11.0"
        ) from None
    if version != "1.11.0":
        raise report.BenchError(f"EvalScope 1.11.0 is required, found {version}")
    stamp = report.utc_now().replace(":", "").replace("-", "")
    run_id = safe_name(args.run_id or f"{stamp}-{profile['suite']['id']}-{model}")
    run_dir = args.output_dir.expanduser().resolve() / run_id
    if run_dir.exists():
        raise report.BenchError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "profile.json").write_text(args.profile.read_text(encoding="utf-8"), encoding="utf-8")
    backend: dict[str, Any]
    if args.gguf:
        server_identity = llama_server_identity(binary)
        backend = {"type": "gguf", "model": model,
                   "gguf": gguf_inventory(args.gguf),
                   "llama_server": {"path": binary, "version": server_identity,
                                    "commit": LLAMA_CPP_COMMIT},
                   "ctx_size": args.ctx_size, "parallel": args.parallel,
                   "slot_context": args.ctx_size // args.parallel, "gpu_layers": args.gpu_layers}
    else:
        backend = {"type": "openai-compatible", "model": model, "url": public_endpoint,
                   "api_key_env": args.api_key_env, "api_key_header": args.api_key_header}
    manifest = {"schema_version": 1, "run_id": run_id, "created_at": report.utc_now(),
                "evalscope_version": version,
                "profile": {"file": "profile.json", "source": str(args.profile.resolve()),
                            "sha256": sha256_file(args.profile)}, "backend": backend}
    report.write_json(run_dir / "manifest.json", manifest)
    report.write_json(run_dir / "status.json", {"state": "running", "updated_at": report.utc_now()})
    service = LlamaServer(args, model, run_dir / "llama-server.log") if args.gguf else nullcontext()
    try:
        with service as local:
            endpoint = local.url if args.gguf else args.url
            proxy_context = nullcontext() if args.gguf else AuthProxy(
                endpoint, key, args.api_key_header, args.api_key_prefix)
            with proxy_context as proxy:
                child_url = endpoint if args.gguf else proxy.url
                for case in profile["cases"]:
                    run_case(profile, case, model, child_url,
                             run_dir / "cases" / case["id"] / "attempt-0001",
                             args.api_key_env)
        summary = report.summarize_run(run_dir)
        report.write_reports(run_dir, summary)
        report.write_json(run_dir / "status.json", {"state": "finished", "result": summary["status"],
                                                     "updated_at": report.utc_now()})
        print(f"Markdown report: {run_dir / 'report.md'}")
        return 0 if summary["status"] in {"pass", "warning"} else 1
    except KeyboardInterrupt:
        report.write_json(run_dir / "status.json", {"state": "interrupted",
                                                       "updated_at": report.utc_now()})
        raise
    except BaseException:
        report.write_json(run_dir / "status.json", {"state": "failed", "updated_at": report.utc_now()})
        raise


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            return run(args)
        if args.command == "doctor":
            key = resolve_key(args.api_key_env)
            result = probe(args.url, key=key, key_header=args.api_key_header,
                           key_prefix=args.api_key_prefix, model=args.model_name,
                           generate=args.generate)
            result["endpoint"] = redact_url(args.url)
            print(json.dumps(result, indent=2))
        elif args.command == "report":
            run_dir = args.run_dir.expanduser().resolve()
            summary = report.summarize_run(run_dir)
            report.write_reports(run_dir, summary)
            print(f"Markdown report: {run_dir / 'report.md'}")
        else:
            run_dirs = [path.expanduser().resolve() for path in args.run_dirs]
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(report.comparison(run_dirs, args.title), encoding="utf-8")
            print(f"Markdown comparison: {output}")
        return 0
    except report.BenchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted; evaluator and model server stopped.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
