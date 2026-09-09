#!/usr/bin/env python3
"""Run reproducible GGUF benchmarks with EvalScope 1.11 as the evaluator."""

from __future__ import annotations

import argparse
import fcntl
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
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import report

ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE = ROOT / "profiles" / "smoke.json"
BFCL_EVAL_VERSION = "2025.10.27.1"
HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
               "te", "trailer", "transfer-encoding", "upgrade"}
BFCL_STEP_LOG = re.compile(r"^ID: .+, Turn: \d+, Step: \d+$")
LEGACY_RUN_ACTIVITY_GRACE_SECONDS = 300
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
    """Record version information when available without restricting the build."""
    try:
        result = subprocess.run([binary, "--version"], capture_output=True, text=True,
                                timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return "unknown version"
    if result.returncode:
        return "unknown version"
    return " ".join(f"{result.stdout}\n{result.stderr}".split()) or "unknown version"


def accelerator_devices(binary: str) -> list[str]:
    """Return accelerator names reported by llama.cpp."""
    try:
        result = subprocess.run([binary, "--list-devices"], capture_output=True, text=True,
                                timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode:
        return []
    return re.findall(r"^\s+(\S+):\s", f"{result.stdout}\n{result.stderr}", re.MULTILINE)


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


def acquire_run_lock(run_dir: Path) -> Any:
    """Prevent two current runners from writing the same run directory."""
    path = run_dir / ".run.lock"
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise report.BenchError(
            f"Run {run_dir.name!r} is already active in another benchmark.py process"
        ) from None
    return handle


def process_is_alive(pid: Any) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def latest_run_activity(run_dir: Path) -> float:
    """Return the newest run artifact timestamp, excluding the lock itself."""
    latest = 0.0
    for path in run_dir.rglob("*"):
        if path.name == ".run.lock" or not path.is_file():
            continue
        try:
            latest = max(latest, path.stat().st_mtime)
        except FileNotFoundError:
            pass
    return latest


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


def validate_runtime_dependencies(profile: dict[str, Any]) -> None:
    """Fail before inference when an optional benchmark dependency is unusable."""
    if not any(case.get("dataset") == "bfcl_v4" for case in profile["cases"]):
        return

    install = (
        "python3 -m pip install 'evalscope[bfcl,ifeval]==1.11.0' "
        "'soundfile==0.13.1'"
    )
    try:
        version = importlib.metadata.version("bfcl-eval")
    except importlib.metadata.PackageNotFoundError:
        raise report.BenchError(
            f"BFCL dependencies are missing; run: {install}"
        ) from None

    if version != BFCL_EVAL_VERSION:
        raise report.BenchError(
            f"BFCL requires bfcl-eval {BFCL_EVAL_VERSION}, found {version}; "
            f"run: {install}"
        )

    try:
        from bfcl_eval.eval_checker.eval_runner import (  # noqa: F401
            _evaluate_single_agentic_entry,
        )
    except ImportError as exc:
        raise report.BenchError(
            f"BFCL dependencies are unusable ({exc}); run: {install}"
        ) from None


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


class EndpointProxy(AbstractContextManager["EndpointProxy"]):
    """Forward requests with round-robin and multi-turn conversation affinity."""

    def __init__(self, target_urls: str | list[str], key: str | None,
                 header: str, prefix: str) -> None:
        urls = [target_urls] if isinstance(target_urls, str) else target_urls
        self.targets = [validate_url(url) for url in urls]
        self.key, self.header, self.prefix = key, header, prefix
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.target_index = 0
        self.target_lock = threading.Lock()
        self.prefix_targets: dict[str, int] = {}

    @property
    def url(self) -> str:
        if not self.server:
            raise RuntimeError("Authentication proxy has not started")
        path = self.targets[0].path.rstrip("/")
        return f"http://127.0.0.1:{self.server.server_address[1]}{path}"

    @staticmethod
    def routing_prefix(body: bytes | None) -> str | None:
        """Hash the stable beginning of an OpenAI chat conversation."""
        if not body:
            return None
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
            return None

        prefix = []
        has_conversation_message = False
        for message in payload["messages"]:
            if not isinstance(message, dict):
                return None
            prefix.append(message)
            if message.get("role") not in {"system", "developer"}:
                has_conversation_message = True
                break
        if not has_conversation_message:
            return None

        identity = {
            "model": payload.get("model"),
            "tools": payload.get("tools"),
            "tool_choice": payload.get("tool_choice"),
            "messages": prefix,
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False).encode()
        return hashlib.sha256(canonical).hexdigest()

    def next_target(self, body: bytes | None = None) -> urllib.parse.SplitResult:
        prefix = self.routing_prefix(body)
        with self.target_lock:
            index = self.prefix_targets.get(prefix) if prefix else None
            if index is None:
                index = self.target_index
                self.target_index = (self.target_index + 1) % len(self.targets)
                if prefix:
                    self.prefix_targets[prefix] = index
            return self.targets[index]

    def __enter__(self) -> "EndpointProxy":
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
                target = owner.next_target(body)
                excluded = HOP_HEADERS | {"host", "content-length", "authorization",
                                          "proxy-authorization", owner.header.lower()}
                headers = {key: value for key, value in self.headers.items()
                           if key.lower() not in excluded}
                if owner.key:
                    headers[owner.header] = f"{owner.prefix}{owner.key}"
                incoming = urllib.parse.urlsplit(self.path)
                base = target.path.rstrip("/")
                suffix = incoming.path[len(base):] if base and incoming.path.startswith(base) else incoming.path
                query = urllib.parse.urlencode(
                    urllib.parse.parse_qsl(target.query, keep_blank_values=True)
                    + urllib.parse.parse_qsl(incoming.query, keep_blank_values=True))
                path = f"{base}/{suffix.lstrip('/')}" + (f"?{query}" if query else "")
                cls = http.client.HTTPSConnection if target.scheme == "https" else http.client.HTTPConnection
                connection = cls(target.hostname, target.port, timeout=3600)
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
    """Manage one loopback-only llama.cpp server."""

    def __init__(self, args: argparse.Namespace, model: str, log: Path,
                 device: str | None = None) -> None:
        self.args, self.model, self.log = args, model, log
        self.device = device
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
                        "--ctx-size", str(self.args.ctx_size),
                        "--parallel", str(self.args.slots_per_server),
                        "--n-gpu-layers", str(self.args.gpu_layers),
                        "--split-mode", "none",
                        "--cache-type-k", "f16", "--cache-type-v", "f16",
                        "--cache-prompt", "--jinja", "--no-context-shift", "--no-webui"]
                argv.extend(["--spec-type", "draft-mtp" if self.args.mtp else "none"])
                if self.args.mmproj:
                    argv.extend(["--mmproj", str(self.args.mmproj.resolve()),
                                 "--image-min-tokens", str(self.args.image_min_tokens),
                                 "--image-max-tokens", str(self.args.image_max_tokens)])
                if self.args.mtp:
                    argv.extend(["--spec-draft-n-max", str(self.args.mtp_draft_tokens)])
                if self.device:
                    argv.extend(["--device", self.device])
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


@contextmanager
def llama_servers(args: argparse.Namespace, model: str, run_dir: Path) -> Iterator[str]:
    """Start one complete model copy on each selected accelerator."""
    with ExitStack() as stack:
        urls = []
        for device in args.server_devices:
            log_name = ("llama-server.log" if len(args.server_devices) == 1
                        else f"llama-server-{device}.log")
            server = stack.enter_context(LlamaServer(args, model, run_dir / log_name, device))
            urls.append(server.url)
        if len(urls) == 1:
            yield urls[0]
        else:
            proxy = stack.enter_context(EndpointProxy(urls, None, "Authorization", "Bearer "))
            yield proxy.url


def evalscope_command(profile: dict[str, Any], case: dict[str, Any], model: str,
                      api_url: str, attempt: Path, eval_batch_size: int,
                      resume_cache: Path | None = None) -> list[str]:
    executable = shutil.which("evalscope") or "evalscope"
    settings = profile.get("evalscope", {})
    argv = [executable, "eval", "--model", model, "--api-url", api_url,
            "--eval-type", "openai_api", "--eval-backend", "Native",
            "--datasets", case["dataset"],
            "--dataset-hub", str(settings.get("dataset_hub", "modelscope")),
            "--dataset-args", json.dumps({case["dataset"]: case["dataset_args"]}, separators=(",", ":")),
            "--seed", str(settings.get("seed", 42)),
            "--eval-batch-size", str(eval_batch_size),
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
    if resume_cache:
        argv.extend(["--use-cache", str(resume_cache), "--rerun-review"])
    argv.append("--collect-perf" if settings.get("collect_perf", True) else "--no-collect-perf")
    return argv


def forward_evalscope_stdout(stream: Any) -> None:
    """Forward evaluator output while hiding BFCL's per-step debug prints."""
    for line in stream:
        content = line.rstrip("\r\n")
        if content == "-" * 100 or BFCL_STEP_LOG.fullmatch(content):
            continue
        sys.stdout.write(line)
        sys.stdout.flush()


def run_case(profile: dict[str, Any], case: dict[str, Any], model: str, api_url: str,
             attempt: Path, key_env: str | None, eval_batch_size: int,
             resume_cache: Path | None = None) -> int:
    # Resume reuses the existing case directory. The first run is still protected
    # by the run-directory existence check before cases are created.
    attempt.mkdir(parents=True, exist_ok=True)
    argv = evalscope_command(profile, case, model, api_url, attempt, eval_batch_size, resume_cache)
    record_path = attempt / "attempt.json"
    attempt_number = 1
    history: list[dict[str, Any]] = []
    if record_path.is_file():
        previous = report.load_json(record_path)
        previous_number = previous.get("attempt", 1)
        if isinstance(previous_number, int) and not isinstance(previous_number, bool):
            attempt_number = previous_number + 1
        previous_history = previous.get("history")
        if isinstance(previous_history, list):
            history.extend(item for item in previous_history if isinstance(item, dict))
        history.append({key: previous[key] for key in (
            "attempt", "started_at", "finished_at", "return_code", "interrupted_by", "command"
        ) if key in previous})
    record = {"schema_version": 1, "case_id": case["id"], "attempt": attempt_number,
              "started_at": report.utc_now(), "command": argv}
    if history:
        record["history"] = history
    if resume_cache:
        record["resumed_from"] = str(resume_cache)
    report.write_json(record_path, record)
    environment = os.environ.copy()
    if key_env:
        environment.pop(key_env, None)
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        argv,
        cwd="/tmp",
        env=environment,
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    try:
        assert process.stdout is not None
        forward_evalscope_stdout(process.stdout)
        code = process.wait()
    except BaseException as exc:
        terminate_process_group(process)
        record.update({"finished_at": report.utc_now(), "return_code": process.returncode,
                       "interrupted_by": type(exc).__name__})
        report.write_json(record_path, record)
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()
    record.update({"finished_at": report.utc_now(), "return_code": code})
    report.write_json(record_path, record)
    return code


def latest_evalscope_cache(attempt: Path) -> Path | None:
    configs = list((attempt / "evalscope").glob("*/configs/task_config.yaml"))
    if not configs:
        return None
    return max(configs, key=lambda path: path.stat().st_mtime).parent.parent


def case_completed(attempt: Path) -> bool:
    record_path = attempt / "attempt.json"
    return record_path.is_file() and report.load_json(record_path).get("return_code") == 0


def resumable_profile(path: Path) -> dict[str, Any]:
    profile = report.load_json(path)
    settings = dict(profile.get("evalscope", {}))
    settings.pop("eval_batch_size", None)
    profile["evalscope"] = settings
    return profile


def resumable_case(case: dict[str, Any]) -> dict[str, Any]:
    """Return only fields that can change evaluator inputs or execution."""
    contract = dict(case)
    contract.pop("primary_metric", None)
    contract.pop("expected_samples", None)
    return contract


def profile_transition(original_path: Path, requested_path: Path) -> bool:
    """Allow a profile to add or replace cases without invalidating reusable cases."""
    original = resumable_profile(original_path)
    requested = resumable_profile(requested_path)
    if original == requested:
        return False

    original_cases = {case["id"]: case for case in original.get("cases", [])}
    requested_cases = {case["id"]: case for case in requested.get("cases", [])}
    shared_ids = original_cases.keys() & requested_cases.keys()
    original_settings = {key: value for key, value in original.items()
                         if key not in {"suite", "cases"}}
    requested_settings = {key: value for key, value in requested.items()
                          if key not in {"suite", "cases"}}
    if shared_ids and original_settings != requested_settings:
        raise report.BenchError(
            "Cannot resume: shared cases use different generation or EvalScope settings")
    changed = sorted(
        case_id for case_id in shared_ids
        if resumable_case(original_cases[case_id]) != resumable_case(requested_cases[case_id])
    )
    if changed:
        raise report.BenchError(
            f"Cannot resume: shared case definitions changed: {', '.join(changed)}")
    return True


def adopt_resume_profile(run_dir: Path, profile_path: Path) -> None:
    """Store the requested profile as the run's current reproducibility contract."""
    stored_profile = run_dir / "profile.json"
    report.write_json(stored_profile, report.load_json(profile_path))
    manifest = report.load_json(run_dir / "manifest.json")
    manifest["profile"] = {
        "file": "profile.json",
        "source": str(profile_path),
        "sha256": sha256_file(stored_profile),
    }
    report.write_json(run_dir / "manifest.json", manifest)


def validate_resume(run_dir: Path, args: argparse.Namespace, model: str,
                    public_endpoint: str, profile_path: Path, version: str) -> bool:
    manifest = report.load_json(run_dir / "manifest.json")
    recorded_profile = manifest.get("profile") if isinstance(manifest.get("profile"), dict) else {}
    recorded_hash = recorded_profile.get("sha256")
    original_profile = run_dir / "profile.json"
    if recorded_hash != sha256_file(original_profile):
        raise report.BenchError("Cannot resume: the recorded profile has been modified")
    profile_changed = profile_transition(original_profile, profile_path)
    if manifest.get("evalscope_version") != version:
        raise report.BenchError("Cannot resume: the EvalScope version does not match the original run")

    backend = manifest.get("backend") if isinstance(manifest.get("backend"), dict) else {}
    expected_type = "gguf" if args.gguf else "openai-compatible"
    if backend.get("type") != expected_type or backend.get("model") != model:
        raise report.BenchError("Cannot resume: the model or backend does not match the original run")
    if args.gguf:
        artifact = backend.get("gguf") if isinstance(backend.get("gguf"), dict) else {}
        if artifact.get("path") != str(args.gguf.resolve()):
            raise report.BenchError("Cannot resume: the GGUF path does not match the original run")
        current_vision = vision_identity(args)
        if backend.get("vision") != current_vision:
            raise report.BenchError("Cannot resume: vision files or image limits have changed")
        if current_vision and artifact.get("sha256") != sha256_file(args.gguf):
            raise report.BenchError("Cannot resume: the language model weights have changed")
        if backend.get("mtp", False) != args.mtp or (
                args.mtp and backend.get("mtp_draft_tokens") != args.mtp_draft_tokens):
            raise report.BenchError("Cannot resume: MTP settings do not match the original run")
    elif backend.get("url") != public_endpoint or backend.get("api_key_header") != args.api_key_header:
        raise report.BenchError("Cannot resume: the API endpoint does not match the original run")

    status = report.load_json(run_dir / "status.json")
    if status.get("state") == "running":
        owner_host = status.get("hostname")
        owner_pid = status.get("pid")
        local_host = socket.gethostname()
        if owner_host in {None, local_host} and process_is_alive(owner_pid):
            raise report.BenchError(
                f"Cannot resume: run {run_dir.name!r} is still active as PID {owner_pid}"
            )
        if owner_host is None:
            idle_seconds = time.time() - latest_run_activity(run_dir)
            if idle_seconds < LEGACY_RUN_ACTIVITY_GRACE_SECONDS:
                raise report.BenchError(
                    "Cannot resume: this legacy run has recent activity but no host identity; "
                    "wait five minutes after it stops"
                )
    requested = report.load_json(profile_path)
    has_pending_case = any(
        not case_completed(run_dir / "cases" / case["id"] / "attempt-0001")
        for case in requested.get("cases", [])
    )
    if status.get("state") == "finished" and status.get("result") != "failed" \
            and not has_pending_case:
        raise report.BenchError("Cannot resume a run that has already completed successfully")
    return profile_changed


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Reproducible GGUF benchmark runner powered by EvalScope 1.11")
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a benchmark profile")
    run.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    backend = run.add_mutually_exclusive_group(required=True)
    backend.add_argument("--gguf", type=Path)
    backend.add_argument("--url")
    run.add_argument("--mmproj", type=Path, help="vision encoder GGUF for a local multimodal run")
    run.add_argument("--image-min-tokens", type=int, default=64)
    run.add_argument("--image-max-tokens", type=int, default=4096)
    run.add_argument("--model-name", help="display name and remote API model name")
    run.add_argument("--api-key-env", help="environment variable containing the API key")
    run.add_argument("--api-key-header", default="Authorization")
    run.add_argument("--api-key-prefix", default="Bearer ")
    run.add_argument("--output-dir", type=Path, default=Path("runs"))
    run.add_argument("--run-id")
    run.add_argument("--llama-server", default="llama-server")
    run.add_argument("--mtp", action="store_true",
                     help="enable MTP speculative decoding for a local GGUF with MTP weights")
    run.add_argument("--mtp-draft-tokens", type=int,
                     help="maximum tokens to draft with --mtp (default: 3)")
    run.add_argument("--ctx-size", type=int,
                     help="context per server; defaults to profile limit times its slots")
    run.add_argument(
        "--parallel",
        type=int,
        help=("total evaluator concurrency, distributed across local servers; defaults to the "
              "profile setting"),
    )
    run.add_argument("--gpu-layers", type=int, default=999)
    run.add_argument("--startup-timeout", type=int, default=300)
    run.add_argument("--resume", action="store_true",
                     help="resume an interrupted run with the same run ID")
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
    rebuild.add_argument("run_dir", type=Path, help="run directory or name under runs/")
    compare = commands.add_parser(
        "compare", help="summarize or compare one or more completed runs")
    compare.add_argument("run_dirs", nargs="+", type=Path,
                         help="run directories or names under runs/")
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--title")
    return root


def vision_identity(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.mmproj:
        return None
    return {"path": str(args.mmproj.resolve()), "sha256": sha256_file(args.mmproj),
            "size_bytes": args.mmproj.stat().st_size,
            "image_min_tokens": args.image_min_tokens,
            "image_max_tokens": args.image_max_tokens}


def validate_vision_samples(profile: dict, profile_path: Path) -> None:
    expected = profile["suite"].get("samples_sha256")
    if not expected:
        return
    manifest = profile_path.parent / "samples.json"
    if not manifest.is_file() or sha256_file(manifest) != expected:
        raise report.BenchError("Vision sample manifest fingerprint changed; prepare a new profile")
    records = report.load_json(manifest)
    for case in profile["cases"]:
        artifact = Path(case["dataset_args"]["dataset_id"]) / "data/test-00000-of-00001.parquet"
        if not artifact.is_file() or sha256_file(artifact) != records[case["dataset"]]["parquet_sha256"]:
            raise report.BenchError("Vision input data fingerprint changed; prepare a new profile")


def run(args: argparse.Namespace) -> int:
    profile_path = args.profile.expanduser().resolve()
    profile = report.load_profile(profile_path)
    if profile["suite"].get("modality") == "vision" and args.gguf and not args.mmproj:
        raise report.BenchError("A local vision profile requires --mmproj")
    if any(str(c.get("dataset_args", {}).get("dataset_id", "")).startswith("PREPARE_")
           for c in profile["cases"]):
        raise report.BenchError("Run prepare_vision_quick.py and use its generated profile.json")
    validate_vision_samples(profile, profile_path)
    key = resolve_key(args.api_key_env)
    if args.mmproj and (not args.gguf or not args.mmproj.is_file()):
        raise report.BenchError("--mmproj requires --gguf and an existing vision GGUF file")
    if not 0 < args.image_min_tokens <= args.image_max_tokens:
        raise report.BenchError("Image token limits must be positive and min <= max")
    if args.resume and not args.run_id:
        raise report.BenchError("--resume requires --run-id")
    profile_batch_size = profile.get("evalscope", {}).get("eval_batch_size", 1)
    if (isinstance(profile_batch_size, bool) or not isinstance(profile_batch_size, int)
            or profile_batch_size <= 0):
        raise report.BenchError("profile.evalscope.eval_batch_size must be a positive integer")
    if args.parallel is not None and args.parallel <= 0:
        raise report.BenchError("--parallel must be positive")
    if args.mtp and not args.gguf:
        raise report.BenchError("--mtp requires --gguf")
    if args.mtp_draft_tokens is not None:
        if not args.mtp:
            raise report.BenchError("--mtp-draft-tokens requires --mtp")
        if args.mtp_draft_tokens <= 0:
            raise report.BenchError("--mtp-draft-tokens must be positive")
    elif args.mtp:
        args.mtp_draft_tokens = 3
    if args.gguf:
        if not args.gguf.is_file():
            raise report.BenchError(f"GGUF file does not exist: {args.gguf}")
        binary = shutil.which(args.llama_server)
        if not binary:
            raise report.BenchError(f"llama-server was not found: {args.llama_server}")
        model = args.model_name or args.gguf.name
        required = max(case["resolved_generation"]["max_tokens"] for case in profile["cases"])
        devices = accelerator_devices(binary)
        if args.parallel is None:
            args.parallel = min(4, profile_batch_size, max(1, len(devices)))
        args.server_devices = devices[:args.parallel] or [None]
        args.slots_per_server = (
            args.parallel + len(args.server_devices) - 1) // len(args.server_devices)
        if args.ctx_size is None:
            args.ctx_size = required * args.slots_per_server
        if args.ctx_size <= 0 or args.ctx_size % args.slots_per_server:
            raise report.BenchError(
                "--ctx-size must be positive and divisible by the slots per server")
        if args.ctx_size // args.slots_per_server < required:
            minimum_total = required * args.slots_per_server
            raise report.BenchError(
                f"Per-slot context is {args.ctx_size // args.slots_per_server:,}; the profile "
                f"requires at least {required:,}. Remove --ctx-size to use the resolved default, "
                f"or set --ctx-size to at least {minimum_total:,} for "
                f"{args.slots_per_server} slots per server.")
        eval_batch_size = args.parallel
        public_endpoint = "managed local llama.cpp endpoint"
    else:
        if not args.model_name:
            raise report.BenchError("--model-name is required with --url")
        validate_url(args.url)
        model, public_endpoint = args.model_name, redact_url(args.url)
        eval_batch_size = args.parallel if args.parallel is not None else profile_batch_size
    plan = {"profile": str(profile_path), "suite": profile["suite"], "resume": args.resume,
            "backend": "gguf" if args.gguf else "url", "model": model,
            "endpoint": public_endpoint, "eval_batch_size": eval_batch_size,
            "cases": [{"id": case["id"], "planned_samples": case["planned_samples"],
                       "generation": case["resolved_generation"]} for case in profile["cases"]]}
    if args.gguf:
        plan["server"] = {"total_parallel": args.parallel,
                          "devices": args.server_devices,
                          "slots_per_server": args.slots_per_server,
                          "ctx_size_per_server": args.ctx_size,
                          "slot_context": args.ctx_size // args.slots_per_server,
                          "mtp": args.mtp, "mtp_draft_tokens": args.mtp_draft_tokens}
        if args.mmproj:
            plan["server"].update({"mmproj": str(args.mmproj.resolve()),
                                   "image_min_tokens": args.image_min_tokens,
                                   "image_max_tokens": args.image_max_tokens})
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
    validate_runtime_dependencies(profile)
    stamp = report.utc_now().replace(":", "").replace("-", "")
    run_id = safe_name(args.run_id or f"{stamp}-{profile['suite']['id']}-{model}")
    run_dir = args.output_dir.expanduser().resolve() / run_id
    if args.resume:
        if not run_dir.is_dir():
            raise report.BenchError(f"Cannot resume: run directory does not exist: {run_dir}")
    else:
        if run_dir.exists():
            raise report.BenchError(f"Run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True)
    # Keep this handle alive for the entire run. The OS releases the lock on exit,
    # including SIGINT, SIGTERM, crashes, and SSH disconnects.
    run_lock = acquire_run_lock(run_dir)
    if args.resume:
        if validate_resume(run_dir, args, model, public_endpoint, profile_path, version):
            adopt_resume_profile(run_dir, profile_path)
    else:
        (run_dir / "profile.json").write_text(profile_path.read_text(encoding="utf-8"), encoding="utf-8")
        backend: dict[str, Any]
        if args.gguf:
            server_identity = llama_server_identity(binary)
            backend = {"type": "gguf", "model": model,
                       "gguf": gguf_inventory(args.gguf),
                       "llama_server": {"path": binary, "version": server_identity},
                       "ctx_size_per_server": args.ctx_size, "parallel": args.parallel,
                       "devices": args.server_devices,
                       "slots_per_server": args.slots_per_server,
                       "slot_context": args.ctx_size // args.slots_per_server,
                       "gpu_layers": args.gpu_layers,
                       "mtp": args.mtp, "mtp_draft_tokens": args.mtp_draft_tokens}
            if args.mmproj:
                backend["vision"] = vision_identity(args)
                backend["gguf"]["sha256"] = sha256_file(args.gguf)
        else:
            backend = {"type": "openai-compatible", "model": model, "url": public_endpoint,
                       "api_key_env": args.api_key_env, "api_key_header": args.api_key_header,
                       "eval_batch_size": eval_batch_size}
        manifest = {"schema_version": 1, "run_id": run_id, "created_at": report.utc_now(),
                    "evalscope_version": version,
                    "profile": {"file": "profile.json", "source": str(profile_path),
                                "sha256": sha256_file(profile_path)}, "backend": backend}
        report.write_json(run_dir / "manifest.json", manifest)
    report.write_json(run_dir / "status.json", {
        "state": "running", "resumed": args.resume, "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "updated_at": report.utc_now(),
    })
    service = llama_servers(args, model, run_dir) if args.gguf else nullcontext(args.url)
    try:
        with service as endpoint:
            proxy_context = nullcontext() if args.gguf else EndpointProxy(
                endpoint, key, args.api_key_header, args.api_key_prefix)
            with proxy_context as proxy:
                child_url = endpoint if args.gguf else proxy.url
                for case in profile["cases"]:
                    attempt = run_dir / "cases" / case["id"] / "attempt-0001"
                    if args.resume and case_completed(attempt):
                        continue
                    resume_cache = latest_evalscope_cache(attempt) if args.resume else None
                    run_case(profile, case, model, child_url,
                             attempt, args.api_key_env, eval_batch_size, resume_cache)
        summary = report.summarize_run(run_dir)
        report.write_reports(run_dir, summary)
        report.write_json(run_dir / "status.json", {"state": "finished", "result": summary["status"],
                                                     "updated_at": report.utc_now()})
        print(f"Markdown report: {run_dir / 'report.md'}")
        run_lock.close()
        return 0 if summary["status"] in {"pass", "warning"} else 1
    except KeyboardInterrupt:
        report.write_json(run_dir / "status.json", {"state": "interrupted",
                                                       "updated_at": report.utc_now()})
        run_lock.close()
        raise
    except BaseException:
        report.write_json(run_dir / "status.json", {"state": "failed", "updated_at": report.utc_now()})
        run_lock.close()
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
            run_dir = report.resolve_run_dir(args.run_dir)
            summary = report.summarize_run(run_dir)
            report.write_reports(run_dir, summary)
            print(f"Markdown report: {run_dir / 'report.md'}")
        else:
            run_dirs = [report.resolve_run_dir(path) for path in args.run_dirs]
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(report.comparison(run_dirs, args.title), encoding="utf-8")
            label = "report" if len(run_dirs) == 1 else "comparison"
            print(f"Markdown {label}: {output}")
        return 0
    except report.BenchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted; evaluator and model server stopped.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
