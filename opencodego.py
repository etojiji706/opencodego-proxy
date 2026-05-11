#!/usr/bin/env python3
"""
OpenCode Go x LiteLLM integration proxy.

This is a single-file, standard-library-only implementation for running a
small OpenAI-compatible failover proxy in front of multiple OpenCode Go-style
workspaces, and for generating LiteLLM model configuration from upstream model
listings.

The exact OpenCode Go public API shape is not assumed. Each workspace is
configured as a generic OpenAI-compatible upstream with a base URL and one API
key. Requests sent to this proxy under /v1/* are forwarded to the current
workspace. On rate limits, token limits, request limits, API errors, timeouts,
or provider unavailable responses, the proxy switches to the next workspace in
a circular route:

    Workspace 1 -> Workspace 2 -> Workspace 3 -> Workspace 4 -> Workspace 1

Non-streaming requests keep same-request retry/failover behavior. Streaming
requests are detected primarily from JSON bodies with "stream": true and also
from SSE-oriented Accept/upstream Content-Type headers. Streaming/SSE responses
are relayed chunk-by-chunk as they arrive, with a per-stream session id,
workspace/model/request context lock, bounded in-memory token/chunk buffer, and
stream lifecycle logs. Once a stream response has started, the workspace/API is
never switched inside that stream; stream errors terminate the session and
advance the circular route for the next request.

Streaming caveats: the proxy intentionally does not resume a failed stream,
does not retry transparently after any stream bytes have been relayed, and does
not log request bodies, API keys, Authorization headers, or chunk contents.

Usage:

    python opencodego.py init-config ./opencodego.config.json
    cp .env.example .env
    # Fill .env with OPENCODEGO_PROXY_API_KEY and OPENCODE_GO_API_KEY_1...
    python opencodego.py serve --config ./opencodego.config.json
    python opencodego.py sync-models --config ./opencodego.config.json
    python opencodego.py print-litellm-config --config ./opencodego.config.json
    python opencodego.py self-test

Minimal config example:

    {
      "listen_host": "127.0.0.1",
      "listen_port": 8088,
      "request_timeout_seconds": 60,
      "model_cache_path": "./opencodego.models.cache.json",
      "litellm_generated_config_path": "./opencodego.litellm.generated.yaml",
      "litellm_proxy_api_base": "http://127.0.0.1:8088/v1",
      "litellm_proxy_api_key_env": "OPENCODEGO_PROXY_API_KEY",
      "workspaces": [
        {
          "name": "workspace-1",
          "base_url": "https://workspace-1.example.com/v1",
          "api_key_env": "OPENCODE_GO_API_KEY_1"
        },
        {
          "name": "workspace-2",
          "base_url": "https://workspace-2.example.com/v1",
          "api_key_env": "OPENCODE_GO_API_KEY_2"
        }
      ]
    }

Public safety notes:

* Do not commit real API keys. Inline api_key config is rejected; use api_key_env.
* Set OPENCODEGO_PROXY_API_KEY before exposing the proxy outside localhost.
* Logs never print configured API keys or client Authorization headers.
* The generated LiteLLM config contains model entries only. It does not set or
  alter defaults, priority, cost, routing weight, classification, or
  temperature presets.
* Model sync failure is logged as a warning and does not stop the proxy.
"""

from __future__ import annotations

import argparse
import codecs
import contextlib
import dataclasses
import datetime as _dt
import hmac
import http.client
import http.server
import json
import logging
import os
import re
import socket
import socketserver
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


APP_NAME = "opencodego"
CONFIG_ENV = "OPENCODEGO_CONFIG"
DEFAULT_ENV_FILE = ".env"
DEFAULT_TTL_SECONDS = 24 * 60 * 60
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_REQUEST_BODY_BYTES = 32 * 1024 * 1024
STREAM_READ_SIZE = 1024
STREAM_BUFFER_MAX_CHUNKS = 256
STREAM_BUFFER_MAX_BYTES = 1024 * 1024
STREAM_RECENT_SESSION_LIMIT = 50

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
LIMIT_STATUS_CODES = {408, 409, 425, 429}
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "authorization",
}
STRIPPED_FORWARD_HEADERS = HOP_BY_HOP_HEADERS | {"accept-encoding"}
ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_time(value: object) -> Optional[_dt.datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = _dt.datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    return parsed.astimezone(_dt.timezone.utc)


def read_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a JSON object: {path}")
    return data


def atomic_write_text(path: str, text: str) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".txt", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp_path, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(temp_path)
        raise


def atomic_write_json(path: str, data: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def default_cache_path() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, APP_NAME, "models.cache.json")


class ConfigError(ValueError):
    pass


def _strip_unquoted_env_comment(value: str) -> str:
    for index, char in enumerate(value):
        if char == "#" and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value.rstrip()


def _parse_double_quoted_env_value(value: str, path: str, line_number: int) -> str:
    output: List[str] = []
    escaped = False
    for index, char in enumerate(value[1:], start=1):
        if escaped:
            output.append(
                {
                    "n": "\n",
                    "r": "\r",
                    "t": "\t",
                    "\\": "\\",
                    '"': '"',
                    "#": "#",
                    "$": "$",
                }.get(char, char)
            )
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            trailing = value[index + 1 :].strip()
            if trailing and not trailing.startswith("#"):
                raise ConfigError(f"invalid .env syntax at {path}:{line_number}")
            return "".join(output)
        output.append(char)
    raise ConfigError(f"unterminated quoted value in .env at {path}:{line_number}")


def _parse_single_quoted_env_value(value: str, path: str, line_number: int) -> str:
    end_index = value.find("'", 1)
    if end_index < 0:
        raise ConfigError(f"unterminated quoted value in .env at {path}:{line_number}")
    trailing = value[end_index + 1 :].strip()
    if trailing and not trailing.startswith("#"):
        raise ConfigError(f"invalid .env syntax at {path}:{line_number}")
    return value[1:end_index]


def parse_env_line(line: str, path: str, line_number: int) -> Optional[Tuple[str, str]]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].lstrip()
    if "=" not in stripped:
        raise ConfigError(f"invalid .env line at {path}:{line_number}; expected KEY=VALUE")
    key, raw_value = stripped.split("=", 1)
    key = key.strip()
    if not ENV_KEY_RE.fullmatch(key):
        raise ConfigError(f"invalid env var name in .env at {path}:{line_number}")
    value = raw_value.strip()
    if value.startswith('"'):
        return key, _parse_double_quoted_env_value(value, path, line_number)
    if value.startswith("'"):
        return key, _parse_single_quoted_env_value(value, path, line_number)
    return key, _strip_unquoted_env_comment(value)


def load_env_file(path: str, required: bool = False) -> int:
    if not os.path.exists(path):
        if required:
            raise ConfigError(f"env file not found: {path}")
        return 0
    loaded = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parsed = parse_env_line(line, path, line_number)
            if parsed is None:
                continue
            key, value = parsed
            if key not in os.environ:
                os.environ[key] = value
                loaded += 1
    logging.getLogger("opencodego.config").info("loaded env file path=%s variables=%s", path, loaded)
    return loaded


def resolve_env_file(
    config_data: Mapping[str, Any],
    cli_env_file: Optional[str] = None,
    load_env: bool = True,
) -> Tuple[Optional[str], bool]:
    if not load_env:
        return None, False
    if cli_env_file:
        return cli_env_file, True
    if "env_file" not in config_data:
        return DEFAULT_ENV_FILE, False
    configured = config_data.get("env_file")
    if configured is None or configured is False or configured == "":
        return None, False
    if not isinstance(configured, str):
        raise ConfigError("env_file must be a string path, null, false, or empty string")
    return configured, True


@dataclasses.dataclass(frozen=True)
class WorkspaceConfig:
    name: str
    base_url: str
    api_key_env: Optional[str] = None
    models_path: str = "/v1/models"

    @classmethod
    def from_dict(cls, index: int, data: Mapping[str, Any]) -> "WorkspaceConfig":
        if not isinstance(data, Mapping):
            raise ConfigError(f"workspaces[{index}] must be an object")
        name = str(data.get("name") or f"workspace-{index + 1}")
        base_url = str(data.get("base_url") or "").strip()
        if not base_url:
            raise ConfigError(f"{name}: base_url is required")
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ConfigError(f"{name}: base_url must be an absolute http(s) URL")
        if "api_key" in data:
            raise ConfigError(f"{name}: plaintext api_key is not allowed; use api_key_env")
        api_key_env = data.get("api_key_env")
        if not isinstance(api_key_env, str) or not api_key_env.strip():
            raise ConfigError(f"{name}: api_key_env is required")
        api_key_env = api_key_env.strip()
        models_path = str(data.get("models_path") or "/v1/models")
        if not models_path.startswith("/"):
            models_path = "/" + models_path
        return cls(
            name=name,
            base_url=base_url.rstrip("/"),
            api_key_env=api_key_env,
            models_path=models_path,
        )

    def resolved_api_key(self) -> str:
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return ""


@dataclasses.dataclass(frozen=True)
class AppConfig:
    workspaces: Tuple[WorkspaceConfig, ...]
    listen_host: str = "127.0.0.1"
    listen_port: int = 8088
    request_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES
    model_cache_path: str = dataclasses.field(default_factory=default_cache_path)
    model_cache_ttl_seconds: int = DEFAULT_TTL_SECONDS
    litellm_generated_config_path: Optional[str] = None
    litellm_proxy_api_base: Optional[str] = None
    litellm_proxy_api_key_env: str = "OPENCODEGO_PROXY_API_KEY"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AppConfig":
        for forbidden_key in ("litellm_proxy_api_key", "proxy_api_key", "client_proxy_api_key"):
            if forbidden_key in data and data.get(forbidden_key) is not None:
                raise ConfigError(f"{forbidden_key} is not allowed; use litellm_proxy_api_key_env")
        raw_workspaces = data.get("workspaces")
        if not isinstance(raw_workspaces, list) or not raw_workspaces:
            raise ConfigError("config requires a non-empty workspaces array")
        parsed_workspaces = tuple(WorkspaceConfig.from_dict(i, item) for i, item in enumerate(raw_workspaces))
        enabled_workspaces: List[WorkspaceConfig] = []
        config_log = logging.getLogger("opencodego.config")
        for workspace in parsed_workspaces:
            env_name = workspace.api_key_env or ""
            if os.environ.get(env_name, "").strip():
                enabled_workspaces.append(workspace)
                continue
            config_log.warning("workspace disabled name=%s api_key_env=%s", workspace.name, env_name)
        if not enabled_workspaces:
            raise ConfigError("no enabled workspaces; configure at least one workspace api_key_env in the environment or .env")
        workspaces = tuple(enabled_workspaces)
        listen_host = str(data.get("listen_host") or "127.0.0.1")
        listen_port = int(data.get("listen_port") or 8088)
        timeout = float(data.get("request_timeout_seconds") or DEFAULT_TIMEOUT_SECONDS)
        max_body_raw = data.get("max_request_body_bytes")
        max_body = int(max_body_raw) if max_body_raw is not None else DEFAULT_MAX_REQUEST_BODY_BYTES
        cache_path = str(data.get("model_cache_path") or default_cache_path())
        ttl = int(data.get("model_cache_ttl_seconds") or DEFAULT_TTL_SECONDS)
        generated_path_raw = data.get("litellm_generated_config_path")
        generated_path = str(generated_path_raw) if generated_path_raw else None
        api_base_raw = data.get("litellm_proxy_api_base")
        api_base = str(api_base_raw) if api_base_raw else None
        api_key_env = str(data.get("litellm_proxy_api_key_env") or "OPENCODEGO_PROXY_API_KEY")
        if listen_port <= 0 or listen_port > 65535:
            raise ConfigError("listen_port must be in 1..65535")
        if timeout <= 0:
            raise ConfigError("request_timeout_seconds must be positive")
        if max_body <= 0:
            raise ConfigError("max_request_body_bytes must be positive")
        if ttl <= 0:
            raise ConfigError("model_cache_ttl_seconds must be positive")
        return cls(
            workspaces=workspaces,
            listen_host=listen_host,
            listen_port=listen_port,
            request_timeout_seconds=timeout,
            max_request_body_bytes=max_body,
            model_cache_path=cache_path,
            model_cache_ttl_seconds=ttl,
            litellm_generated_config_path=generated_path,
            litellm_proxy_api_base=api_base,
            litellm_proxy_api_key_env=api_key_env,
        )

    def proxy_api_base(self) -> str:
        return self.litellm_proxy_api_base or f"http://{self.listen_host}:{self.listen_port}/v1"


def load_config(
    path: Optional[str],
    env_file: Optional[str] = None,
    load_env: bool = True,
) -> AppConfig:
    config_path = path or os.environ.get(CONFIG_ENV)
    if not config_path:
        raise ConfigError(f"provide --config or set {CONFIG_ENV}")
    data = read_json_file(config_path)
    selected_env_file, env_file_required = resolve_env_file(
        data,
        cli_env_file=env_file,
        load_env=load_env,
    )
    if selected_env_file:
        load_env_file(selected_env_file, required=env_file_required)
    return AppConfig.from_dict(data)


def sample_config() -> Dict[str, Any]:
    return {
        "listen_host": "127.0.0.1",
        "listen_port": 8088,
        "request_timeout_seconds": 60,
        "max_request_body_bytes": DEFAULT_MAX_REQUEST_BODY_BYTES,
        "model_cache_ttl_seconds": DEFAULT_TTL_SECONDS,
        "model_cache_path": "./opencodego.models.cache.json",
        "litellm_generated_config_path": "./opencodego.litellm.generated.yaml",
        "litellm_proxy_api_base": "http://127.0.0.1:8088/v1",
        "litellm_proxy_api_key_env": "OPENCODEGO_PROXY_API_KEY",
        "workspaces": [
            {
                "name": f"workspace-{i}",
                "base_url": f"https://workspace-{i}.example.com/v1",
                "api_key_env": f"OPENCODE_GO_API_KEY_{i}",
            }
            for i in range(1, 5)
        ],
    }


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def join_upstream_url(base_url: str, request_target: str) -> str:
    """Join an upstream base URL with a client request path and query."""
    split_target = urllib.parse.urlsplit(request_target)
    request_path = split_target.path or "/"
    request_query = split_target.query
    split_base = urllib.parse.urlsplit(base_url.rstrip("/"))
    base_path = split_base.path.rstrip("/")

    forwarded_path = request_path
    if base_path.endswith("/v1") and request_path == "/v1":
        forwarded_path = ""
    elif base_path.endswith("/v1") and request_path.startswith("/v1/"):
        forwarded_path = request_path[len("/v1") :]

    if not forwarded_path.startswith("/"):
        forwarded_path = "/" + forwarded_path
    final_path = (base_path + forwarded_path) if forwarded_path != "/" else (base_path or "/")
    if not final_path.startswith("/"):
        final_path = "/" + final_path
    return urllib.parse.urlunsplit(
        (split_base.scheme, split_base.netloc, final_path, request_query, "")
    )


def sanitize_forward_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    output: Dict[str, str] = {}
    for key, value in headers.items():
        lowered = key.lower()
        if lowered in STRIPPED_FORWARD_HEADERS:
            continue
        output[key] = value
    return output


def classify_retryable(status: int, body: bytes, exc: Optional[BaseException] = None) -> Tuple[bool, str]:
    if exc is not None:
        if isinstance(exc, socket.timeout):
            return True, "timeout"
        if isinstance(exc, TimeoutError):
            return True, "timeout"
        return True, "api_error"

    body_text = body[:4096].decode("utf-8", errors="ignore").lower()
    limit_needles = (
        "rate limit",
        "rate_limit",
        "ratelimit",
        "token limit",
        "token_limit",
        "request limit",
        "request_limit",
        "quota",
        "insufficient_quota",
        "too many requests",
    )
    unavailable_needles = (
        "provider unavailable",
        "service unavailable",
        "temporarily unavailable",
        "upstream unavailable",
        "timeout",
        "timed out",
    )

    if status in LIMIT_STATUS_CODES or any(needle in body_text for needle in limit_needles):
        return True, "limit"
    if status in RETRYABLE_STATUS_CODES or any(needle in body_text for needle in unavailable_needles):
        return True, "provider_unavailable"
    if 500 <= status <= 599:
        return True, "api_error"
    return False, "non_retryable"


def is_sse_content_type(value: Optional[str]) -> bool:
    return "text/event-stream" in (value or "").lower()


def header_get(headers: Mapping[str, str], name: str, default: str = "") -> str:
    lowered_name = name.lower()
    for key, value in headers.items():
        if key.lower() == lowered_name:
            return value
    return default


def content_encoding_is_identity(value: str) -> bool:
    codings = [part.strip().lower() for part in value.split(",") if part.strip()]
    return all(coding == "identity" for coding in codings)


def request_accepts_sse(headers: Mapping[str, str]) -> bool:
    for key, value in headers.items():
        if key.lower() == "accept" and "text/event-stream" in value.lower():
            return True
    return False


def extract_request_json(body: bytes) -> Optional[Mapping[str, Any]]:
    if not body:
        return None
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def request_wants_stream(body: bytes, headers: Mapping[str, str]) -> bool:
    payload = extract_request_json(body)
    if payload is not None and payload.get("stream") is True:
        return True
    return request_accepts_sse(headers)


def extract_model_from_body(body: bytes) -> str:
    payload = extract_request_json(body)
    if payload is None:
        return ""
    model = payload.get("model")
    return str(model) if isinstance(model, str) else ""


def classify_stream_exception(exc: BaseException) -> str:
    if isinstance(exc, (BrokenPipeError, ConnectionAbortedError)):
        return "client_disconnect"
    if isinstance(exc, ConnectionResetError):
        return "connection_reset"
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return "timeout"
    if isinstance(exc, http.client.IncompleteRead):
        return "connection_reset"
    if isinstance(exc, (http.client.HTTPException, urllib.error.URLError, OSError)):
        return "provider_error"
    if isinstance(exc, UnicodeDecodeError):
        return "malformed_sse"
    if isinstance(exc, ValueError):
        return "invalid_chunk"
    return "unknown_stream_error"


@dataclasses.dataclass
class UpstreamResponse:
    status: int
    headers: Dict[str, str]
    body: bytes
    workspace: str
    attempts: int


@dataclasses.dataclass
class UpstreamStream:
    status: int
    headers: Dict[str, str]
    response: Any
    workspace: str
    attempts: int
    stream_intended: bool


@dataclasses.dataclass
class StreamSession:
    session_id: str
    workspace: str
    model: str
    method: str
    path: str
    started_at: str
    status: str = "active"
    error_type: str = ""
    first_chunk_at: str = ""
    finished_at: str = ""
    chunk_count: int = 0
    bytes_received: int = 0
    chunks: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    buffered_bytes: int = 0

    def add_chunk(self, chunk: bytes) -> None:
        now = iso_now()
        if not self.first_chunk_at:
            self.first_chunk_at = now
        self.chunk_count += 1
        self.bytes_received += len(chunk)
        if not chunk:
            return
        record = {"received_at": now, "size": len(chunk), "chunk": chunk}
        self.chunks.append(record)
        self.buffered_bytes += len(chunk)
        while len(self.chunks) > STREAM_BUFFER_MAX_CHUNKS or self.buffered_bytes > STREAM_BUFFER_MAX_BYTES:
            removed = self.chunks.pop(0)
            self.buffered_bytes -= int(removed.get("size") or 0)

    def summary(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "workspace": self.workspace,
            "model": self.model,
            "method": self.method,
            "path": self.path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "error_type": self.error_type,
            "chunk_count": self.chunk_count,
            "bytes_received": self.bytes_received,
            "buffered_chunks": len(self.chunks),
            "buffered_bytes": self.buffered_bytes,
            "first_chunk_at": self.first_chunk_at,
        }


class StreamingSessionManager:
    def __init__(self) -> None:
        self._active: Dict[str, StreamSession] = {}
        self._recent: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self.log = logging.getLogger("opencodego.streaming")

    def start(self, workspace: str, model: str, method: str, path: str) -> StreamSession:
        session = StreamSession(
            session_id=uuid.uuid4().hex,
            workspace=workspace,
            model=model,
            method=method,
            path=path,
            started_at=iso_now(),
        )
        with self._lock:
            self._active[session.session_id] = session
        self.log.info(
            "stream start session_id=%s workspace=%s model=%s method=%s path=%s",
            session.session_id,
            workspace,
            model or "<unknown>",
            method,
            path,
        )
        self.log.info("workspace lock session_id=%s workspace=%s", session.session_id, workspace)
        return session

    def record_chunk(self, session: StreamSession, chunk: bytes) -> None:
        with self._lock:
            session.add_chunk(chunk)
            if session.chunk_count == 1 or session.chunk_count % 25 == 0:
                self.log.info(
                    "token buffer session_id=%s workspace=%s chunks=%s bytes=%s buffered_chunks=%s buffered_bytes=%s",
                    session.session_id,
                    session.workspace,
                    session.chunk_count,
                    session.bytes_received,
                    len(session.chunks),
                    session.buffered_bytes,
                )

    def finish(self, session: StreamSession, status: str, error_type: str = "") -> None:
        with self._lock:
            session.status = status
            session.error_type = error_type
            session.finished_at = iso_now()
            summary = session.summary()
            self._active.pop(session.session_id, None)
            self._recent.append(summary)
            if len(self._recent) > STREAM_RECENT_SESSION_LIMIT:
                self._recent = self._recent[-STREAM_RECENT_SESSION_LIMIT:]
        if status == "completed":
            self.log.info(
                "stream complete session_id=%s workspace=%s chunks=%s bytes=%s",
                session.session_id,
                session.workspace,
                session.chunk_count,
                session.bytes_received,
            )
        else:
            self.log.warning(
                "stream error session_id=%s workspace=%s status=%s error_type=%s chunks=%s bytes=%s",
                session.session_id,
                session.workspace,
                status,
                error_type or "unknown_stream_error",
                session.chunk_count,
                session.bytes_received,
            )
            self.log.warning(
                "streaming session terminated session_id=%s workspace=%s status=%s",
                session.session_id,
                session.workspace,
                status,
            )
        self.log.info("workspace unlock session_id=%s workspace=%s", session.session_id, session.workspace)

    def recent_summaries(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._recent)


class WorkspaceRouter:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._index = 0
        self._lock = threading.Lock()
        self.log = logging.getLogger("opencodego.router")

    def current_workspace(self) -> WorkspaceConfig:
        with self._lock:
            return self.config.workspaces[self._index]

    def current_index(self) -> int:
        with self._lock:
            return self._index

    def switch_next(self, reason: str) -> WorkspaceConfig:
        with self._lock:
            old = self.config.workspaces[self._index]
            self._index = (self._index + 1) % len(self.config.workspaces)
            new = self.config.workspaces[self._index]
        self.log.warning("workspace switch: %s -> %s reason=%s", old.name, new.name, reason)
        return new

    def switch_after_workspace(self, workspace_name: str, reason: str) -> WorkspaceConfig:
        with self._lock:
            old = self.config.workspaces[self._index]
            failed_index = None
            for i, candidate in enumerate(self.config.workspaces):
                if candidate.name == workspace_name:
                    failed_index = i
                    break
            if failed_index is None:
                self._index = (self._index + 1) % len(self.config.workspaces)
                new = self.config.workspaces[self._index]
                anchor_found = False
            else:
                self._index = (failed_index + 1) % len(self.config.workspaces)
                new = self.config.workspaces[self._index]
                anchor_found = True
        self.log.warning(
            "workspace switch: %s -> %s reason=%s failed_workspace=%s anchor_found=%s",
            old.name,
            new.name,
            reason,
            workspace_name or "<unknown>",
            anchor_found,
        )
        return new

    def _workspace_for_attempt(self, attempt: int) -> WorkspaceConfig:
        with self._lock:
            index = (self._index + attempt) % len(self.config.workspaces)
            return self.config.workspaces[index]

    def _request_start_index(self) -> int:
        with self._lock:
            return self._index

    def _commit_success_workspace(self, workspace: WorkspaceConfig) -> None:
        with self._lock:
            for i, candidate in enumerate(self.config.workspaces):
                if candidate.name == workspace.name:
                    self._index = i
                    return

    def _build_upstream_request(
        self,
        workspace: WorkspaceConfig,
        method: str,
        request_target: str,
        headers: Mapping[str, str],
        body: bytes,
    ) -> urllib.request.Request:
        url = join_upstream_url(workspace.base_url, request_target)
        request_headers = sanitize_forward_headers(headers)
        api_key = workspace.resolved_api_key()
        if api_key:
            request_headers["Authorization"] = f"Bearer {api_key}"
        request_headers["Accept-Encoding"] = "identity"
        request_headers.setdefault("User-Agent", f"{APP_NAME}/1.0")
        self.log.debug(
            "forward workspace=%s method=%s url=%s api_key_env=%s",
            workspace.name,
            method,
            url,
            workspace.api_key_env or "<unset>",
        )
        return urllib.request.Request(
            url,
            data=body if method.upper() not in {"GET", "HEAD"} else None,
            headers=request_headers,
            method=method.upper(),
        )

    def forward_or_open_stream(
        self,
        method: str,
        request_target: str,
        headers: Mapping[str, str],
        body: bytes,
        stream_intended: bool,
    ) -> Tuple[Optional[UpstreamResponse], Optional[UpstreamStream]]:
        attempts_allowed = 1 if stream_intended else len(self.config.workspaces)
        start_index = self._request_start_index()
        last_response: Optional[UpstreamResponse] = None
        last_exception: Optional[BaseException] = None

        for attempt in range(attempts_allowed):
            workspace = self.config.workspaces[(start_index + attempt) % len(self.config.workspaces)]
            request = self._build_upstream_request(workspace, method, request_target, headers, body)
            try:
                response = urllib.request.urlopen(request, timeout=self.config.request_timeout_seconds)
                response_headers = dict(response.headers.items())
                if stream_intended or is_sse_content_type(header_get(response_headers, "Content-Type")):
                    self._commit_success_workspace(workspace)
                    return None, UpstreamStream(
                        status=response.getcode(),
                        headers=response_headers,
                        response=response,
                        workspace=workspace.name,
                        attempts=attempt + 1,
                        stream_intended=stream_intended,
                    )
                with response:
                    response_body = response.read()
                result = UpstreamResponse(
                    status=response.getcode(),
                    headers=response_headers,
                    body=response_body,
                    workspace=workspace.name,
                    attempts=attempt + 1,
                )
                self._commit_success_workspace(workspace)
                if attempt:
                    self.log.info("retry succeeded workspace=%s attempts=%s", workspace.name, attempt + 1)
                return result, None
            except urllib.error.HTTPError as exc:
                response_body = exc.read()
                retryable, reason = classify_retryable(exc.code, response_body)
                last_response = UpstreamResponse(
                    status=exc.code,
                    headers=dict(exc.headers.items()) if exc.headers else {},
                    body=response_body,
                    workspace=workspace.name,
                    attempts=attempt + 1,
                )
                if retryable:
                    self.log.warning(
                        "upstream retryable response workspace=%s status=%s reason=%s stream_intended=%s",
                        workspace.name,
                        exc.code,
                        reason,
                        stream_intended,
                    )
                    if reason == "limit":
                        self.log.warning("api limit reached workspace=%s status=%s", workspace.name, exc.code)
                    if stream_intended:
                        stream_reason = "rate_limit" if reason == "limit" else reason
                        self.switch_after_workspace(workspace.name, stream_reason)
                        self.log.warning("next request failover reason=%s after_stream_setup_error=true", stream_reason)
                        return last_response, None
                    if attempt + 1 < attempts_allowed:
                        self.switch_next(reason)
                        self.log.info("retrying request attempt=%s", attempt + 2)
                        continue
                    self.switch_next(reason)
                    self.log.warning(
                        "upstream retryable response exhausted workspaces status=%s reason=%s",
                        exc.code,
                        reason,
                    )
                    return last_response, None
                self.log.warning(
                    "upstream returned non-retryable response workspace=%s status=%s",
                    workspace.name,
                    exc.code,
                )
                return last_response, None
            except (urllib.error.URLError, socket.timeout, TimeoutError, http.client.HTTPException) as exc:
                last_exception = exc
                retryable, reason = classify_retryable(0, b"", exc)
                self.log.warning(
                    "api error workspace=%s reason=%s error=%s stream_intended=%s",
                    workspace.name,
                    reason if retryable else "non_retryable",
                    exc.__class__.__name__,
                    stream_intended,
                )
                if stream_intended:
                    stream_reason = "timeout" if reason == "timeout" else "provider_error"
                    self.switch_after_workspace(workspace.name, stream_reason)
                    self.log.warning("next request failover reason=%s after_stream_setup_error=true", stream_reason)
                    break
                if attempt + 1 < attempts_allowed:
                    self.switch_next(reason)
                    self.log.info("retrying request attempt=%s", attempt + 2)
                    continue
                self.switch_next(reason)
                break

        if last_response is not None:
            return last_response, None
        message = {
            "error": {
                "message": "all upstream workspaces failed",
                "type": "opencodego_upstream_error",
                "detail": last_exception.__class__.__name__ if last_exception else "unknown",
            }
        }
        return UpstreamResponse(
            status=503,
            headers={"Content-Type": "application/json"},
            body=json.dumps(message).encode("utf-8"),
            workspace=self.current_workspace().name,
            attempts=attempts_allowed,
        ), None

    def forward(
        self,
        method: str,
        request_target: str,
        headers: Mapping[str, str],
        body: bytes,
        max_attempts: Optional[int] = None,
    ) -> UpstreamResponse:
        attempts_allowed = max_attempts or len(self.config.workspaces)
        attempts_allowed = max(1, min(attempts_allowed, len(self.config.workspaces)))
        start_index = self._request_start_index()
        last_response: Optional[UpstreamResponse] = None
        last_exception: Optional[BaseException] = None

        for attempt in range(attempts_allowed):
            workspace = self.config.workspaces[(start_index + attempt) % len(self.config.workspaces)]
            url = join_upstream_url(workspace.base_url, request_target)
            request_headers = sanitize_forward_headers(headers)
            api_key = workspace.resolved_api_key()
            if api_key:
                request_headers["Authorization"] = f"Bearer {api_key}"
            request_headers["Accept-Encoding"] = "identity"
            request_headers.setdefault("User-Agent", f"{APP_NAME}/1.0")

            self.log.debug(
                "forward attempt=%s workspace=%s method=%s url=%s api_key_env=%s",
                attempt + 1,
                workspace.name,
                method,
                url,
                workspace.api_key_env or "<unset>",
            )

            request = urllib.request.Request(
                url,
                data=body if method.upper() not in {"GET", "HEAD"} else None,
                headers=request_headers,
                method=method.upper(),
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.request_timeout_seconds
                ) as response:
                    response_body = response.read()
                    result = UpstreamResponse(
                        status=response.getcode(),
                        headers=dict(response.headers.items()),
                        body=response_body,
                        workspace=workspace.name,
                        attempts=attempt + 1,
                    )
                    self._commit_success_workspace(workspace)
                    if attempt:
                        self.log.info(
                            "retry succeeded workspace=%s attempts=%s",
                            workspace.name,
                            attempt + 1,
                        )
                    return result
            except urllib.error.HTTPError as exc:
                response_body = exc.read()
                retryable, reason = classify_retryable(exc.code, response_body)
                last_response = UpstreamResponse(
                    status=exc.code,
                    headers=dict(exc.headers.items()) if exc.headers else {},
                    body=response_body,
                    workspace=workspace.name,
                    attempts=attempt + 1,
                )
                if retryable:
                    self.log.warning(
                        "upstream retryable response workspace=%s status=%s reason=%s",
                        workspace.name,
                        exc.code,
                        reason,
                    )
                    if reason == "limit":
                        self.log.warning("api limit reached workspace=%s status=%s", workspace.name, exc.code)
                    if attempt + 1 < attempts_allowed:
                        self.switch_next(reason)
                        self.log.info("retrying request attempt=%s", attempt + 2)
                        continue
                    self.switch_next(reason)
                    self.log.warning(
                        "upstream retryable response exhausted workspaces status=%s reason=%s",
                        exc.code,
                        reason,
                    )
                    return last_response
                self.log.warning(
                    "upstream returned non-retryable response workspace=%s status=%s",
                    workspace.name,
                    exc.code,
                )
                return last_response
            except (urllib.error.URLError, socket.timeout, TimeoutError, http.client.HTTPException) as exc:
                last_exception = exc
                retryable, reason = classify_retryable(0, b"", exc)
                self.log.warning(
                    "api error workspace=%s reason=%s error=%s",
                    workspace.name,
                    reason if retryable else "non_retryable",
                    exc.__class__.__name__,
                )
                if attempt + 1 < attempts_allowed:
                    self.switch_next(reason)
                    self.log.info("retrying request attempt=%s", attempt + 2)
                    continue
                self.switch_next(reason)
                break

        if last_response is not None:
            return last_response
        message = {
            "error": {
                "message": "all upstream workspaces failed",
                "type": "opencodego_upstream_error",
                "detail": last_exception.__class__.__name__ if last_exception else "unknown",
            }
        }
        return UpstreamResponse(
            status=503,
            headers={"Content-Type": "application/json"},
            body=json.dumps(message).encode("utf-8"),
            workspace=self.current_workspace().name,
            attempts=attempts_allowed,
        )


def extract_models(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            raw_items = payload["data"]
        elif isinstance(payload.get("models"), list):
            raw_items = payload["models"]
        else:
            raw_items = []
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raw_items = []

    output: List[Dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, str):
            output.append({"id": item})
        elif isinstance(item, Mapping):
            model_id = item.get("id") or item.get("name") or item.get("model")
            if isinstance(model_id, str) and model_id:
                copied = dict(item)
                copied["id"] = model_id
                output.append(copied)
    return output


def model_is_active(model: Mapping[str, Any]) -> bool:
    for key in ("active", "enabled", "available"):
        if key in model and model[key] is False:
            return False
    status = str(
        model.get("status")
        or model.get("state")
        or model.get("availability")
        or ""
    ).strip().lower()
    if status in {"inactive", "disabled", "unavailable", "deprecated", "deleted", "error"}:
        return False
    if status and status not in {"active", "available", "enabled", "ready", "ok", "public"}:
        return False
    return True


def normalize_cached_models(raw: Any) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []
    seen = set()
    if not isinstance(raw, list):
        return output
    for item in raw:
        if isinstance(item, str):
            model_id = item
            source = ""
            first_seen = ""
        elif isinstance(item, Mapping):
            model_id = str(item.get("id") or "")
            source = str(item.get("source_workspace") or "")
            first_seen = str(item.get("first_seen_at") or "")
        else:
            continue
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        output.append({"id": model_id, "source_workspace": source, "first_seen_at": first_seen})
    return output


class ModelRegistry:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.log = logging.getLogger("opencodego.models")
        self._lock = threading.Lock()

    def load_cache(self) -> Dict[str, Any]:
        path = self.config.model_cache_path
        if not os.path.exists(path):
            return {"version": 1, "last_sync_at": None, "models": []}
        try:
            data = read_json_file(path)
        except Exception as exc:
            self.log.warning("model cache read failed path=%s error=%s", path, exc)
            return {"version": 1, "last_sync_at": None, "models": []}
        data["models"] = normalize_cached_models(data.get("models"))
        return data

    def save_cache(self, cache: Mapping[str, Any]) -> None:
        atomic_write_json(self.config.model_cache_path, cache)

    def cache_is_fresh(self, cache: Mapping[str, Any]) -> bool:
        last_sync = parse_time(cache.get("last_sync_at"))
        if last_sync is None:
            return False
        age = (utc_now() - last_sync).total_seconds()
        return age < self.config.model_cache_ttl_seconds

    def fetch_workspace_models(self, workspace: WorkspaceConfig) -> List[Dict[str, Any]]:
        url = join_upstream_url(workspace.base_url, workspace.models_path)
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "User-Agent": f"{APP_NAME}/1.0",
        }
        api_key = workspace.resolved_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(request, timeout=self.config.request_timeout_seconds) as response:
            body = response.read()
        payload = json.loads(body.decode("utf-8"))
        return [model for model in extract_models(payload) if model_is_active(model)]

    def sync(self, force: bool = False) -> List[Dict[str, str]]:
        with self._lock:
            cache = self.load_cache()
            if not force and self.cache_is_fresh(cache):
                self.log.info("model sync skipped cache fresh ttl_seconds=%s", self.config.model_cache_ttl_seconds)
                return normalize_cached_models(cache.get("models"))

            self.log.info("model sync start workspaces=%s", len(self.config.workspaces))
            existing = normalize_cached_models(cache.get("models"))
            existing_by_id = {item["id"]: item for item in existing}
            active_by_id: Dict[str, str] = {}
            successful_workspaces = set()
            added = 0
            removed = 0
            had_success = False

            for workspace in self.config.workspaces:
                try:
                    models = self.fetch_workspace_models(workspace)
                    had_success = True
                    successful_workspaces.add(workspace.name)
                except Exception as exc:
                    self.log.warning(
                        "model sync failed workspace=%s error=%s",
                        workspace.name,
                        exc.__class__.__name__,
                    )
                    continue

                for model in models:
                    model_id = str(model.get("id") or "")
                    if not model_id or model_id in active_by_id:
                        continue
                    active_by_id[model_id] = workspace.name

            if not had_success:
                self.log.warning("model sync failed all workspaces; continuing with existing cache")
                return existing

            all_workspaces_succeeded = len(successful_workspaces) == len(self.config.workspaces)
            merged: List[Dict[str, str]] = []
            merged_ids = set()

            for item in existing:
                model_id = item["id"]
                source_workspace = item.get("source_workspace") or ""
                if model_id in active_by_id:
                    merged.append(item)
                    merged_ids.add(model_id)
                    continue
                if source_workspace and source_workspace not in successful_workspaces:
                    merged.append(item)
                    merged_ids.add(model_id)
                    continue
                if not source_workspace and not all_workspaces_succeeded:
                    merged.append(item)
                    merged_ids.add(model_id)
                    continue
                removed += 1
                self.log.info("model sync remove stale model=%s source_workspace=%s", model_id, source_workspace)

            for model_id, workspace_name in active_by_id.items():
                if model_id in merged_ids:
                    continue
                added += 1
                item = existing_by_id.get(model_id) or {
                    "id": model_id,
                    "source_workspace": workspace_name,
                    "first_seen_at": iso_now(),
                }
                merged.append(item)
                merged_ids.add(model_id)
                self.log.info("model sync add model=%s source_workspace=%s", model_id, workspace_name)

            new_cache = {
                "version": 1,
                "last_sync_at": iso_now(),
                "ttl_seconds": self.config.model_cache_ttl_seconds,
                "models": merged,
            }
            self.save_cache(new_cache)
            self.log.info("model sync complete models=%s added=%s removed=%s", len(merged), added, removed)
            if self.config.litellm_generated_config_path:
                text = render_litellm_config(merged, self.config)
                atomic_write_text(self.config.litellm_generated_config_path, text)
            return merged

    def models_for_health(self) -> Dict[str, Any]:
        cache = self.load_cache()
        return {
            "cache_path": self.config.model_cache_path,
            "last_sync_at": cache.get("last_sync_at"),
            "fresh": self.cache_is_fresh(cache),
            "model_count": len(normalize_cached_models(cache.get("models"))),
        }


def yaml_scalar(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:@+-]+", value):
        return value
    return json.dumps(value, ensure_ascii=False)


def render_litellm_config(models: Sequence[Mapping[str, str]], config: AppConfig) -> str:
    lines = [
        "# Generated by opencodego.py.",
        "# This file only adds active, available model entries discovered upstream.",
        "# It intentionally does not set defaults, priority, costs, routing weights,",
        "# classifications, or temperature presets.",
        "model_list:",
    ]
    api_base = config.proxy_api_base()
    api_key_ref = f"os.environ/{config.litellm_proxy_api_key_env}"
    for item in sorted(models, key=lambda model: model.get("id", "")):
        model_id = str(item.get("id") or "")
        if not model_id:
            continue
        lines.extend(
            [
                f"  - model_name: {yaml_scalar(model_id)}",
                "    litellm_params:",
                f"      model: {yaml_scalar('openai/' + model_id)}",
                f"      api_base: {yaml_scalar(api_base)}",
                f"      api_key: {yaml_scalar(api_key_ref)}",
            ]
        )
    return "\n".join(lines) + "\n"


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    server_version = "OpenCodeGoProxy/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        logging.getLogger("opencodego.http").info(
            "%s - %s",
            self.address_string(),
            fmt % args,
        )

    @property
    def router(self) -> WorkspaceRouter:
        return self.server.router  # type: ignore[attr-defined]

    @property
    def registry(self) -> ModelRegistry:
        return self.server.registry  # type: ignore[attr-defined]

    @property
    def stream_manager(self) -> StreamingSessionManager:
        return self.server.stream_manager  # type: ignore[attr-defined]

    def _send_json(
        self,
        status: int,
        payload: Mapping[str, Any],
        write_body: bool = True,
        extra_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if write_body:
            self.wfile.write(body)

    def _send_proxy_response(self, response: UpstreamResponse, write_body: bool = True) -> None:
        content_encoding = header_get(response.headers, "Content-Encoding").strip()
        if content_encoding and not content_encoding_is_identity(content_encoding):
            self._send_json(
                502,
                {
                    "error": {
                        "message": "unsupported upstream content encoding",
                        "type": "opencodego_upstream_encoding_error",
                    }
                },
                write_body=write_body,
                extra_headers={
                    "X-OpenCodeGo-Workspace": response.workspace,
                    "X-OpenCodeGo-Attempts": str(response.attempts),
                },
            )
            return
        self.send_response(response.status)
        for key, value in response.headers.items():
            lowered = key.lower()
            if lowered in HOP_BY_HOP_HEADERS or lowered in {"content-length", "content-encoding"}:
                continue
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(response.body)))
        self.send_header("X-OpenCodeGo-Workspace", response.workspace)
        self.send_header("X-OpenCodeGo-Attempts", str(response.attempts))
        self.end_headers()
        if write_body:
            self.wfile.write(response.body)

    def _send_stream_headers(self, stream: UpstreamStream, session: StreamSession) -> None:
        self.send_response(stream.status)
        for key, value in stream.headers.items():
            lowered = key.lower()
            if lowered in HOP_BY_HOP_HEADERS or lowered in {"content-length", "content-encoding"}:
                continue
            self.send_header(key, value)
        self.send_header("X-OpenCodeGo-Workspace", stream.workspace)
        self.send_header("X-OpenCodeGo-Attempts", str(stream.attempts))
        self.send_header("X-OpenCodeGo-Stream-Session", session.session_id)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _relay_stream(self, stream: UpstreamStream, method: str, body: bytes) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        model = extract_model_from_body(body)
        session = self.stream_manager.start(stream.workspace, model, method, parsed.path)
        content_encoding = header_get(stream.headers, "Content-Encoding").strip().lower()
        if content_encoding and not content_encoding_is_identity(content_encoding):
            with contextlib.suppress(Exception):
                stream.response.close()
            self.stream_manager.finish(session, "error", "provider_error")
            if stream.stream_intended:
                self.router.switch_after_workspace(stream.workspace, "provider_error")
                logging.getLogger("opencodego.streaming").warning(
                    "next request failover reason=provider_error unsupported_content_encoding=true session_id=%s",
                    session.session_id,
                )
            self._send_json(
                502,
                {
                    "error": {
                        "message": "unsupported upstream streaming content encoding",
                        "type": "opencodego_streaming_error",
                    }
                },
            )
            return

        is_sse = is_sse_content_type(header_get(stream.headers, "Content-Type"))
        decoder = codecs.getincrementaldecoder("utf-8")("strict") if is_sse else None
        error_type = ""
        sent_headers = False
        try:
            self._send_stream_headers(stream, session)
            sent_headers = True
            read_chunk = getattr(stream.response, "read1", None)
            if not callable(read_chunk):
                read_chunk = stream.response.read
            while True:
                chunk = read_chunk(STREAM_READ_SIZE)
                if not chunk:
                    break
                if decoder is not None:
                    decoder.decode(chunk, final=False)
                self.stream_manager.record_chunk(session, chunk)
                self.wfile.write(chunk)
                self.wfile.flush()
            if decoder is not None:
                decoder.decode(b"", final=True)
            self.stream_manager.finish(session, "completed")
        except Exception as exc:
            error_type = classify_stream_exception(exc)
            status = "partial" if session.chunk_count else "error"
            self.stream_manager.finish(session, status, error_type)
            self.router.switch_after_workspace(stream.workspace, error_type)
            logging.getLogger("opencodego.streaming").warning(
                "next request failover reason=%s session_id=%s workspace=%s sent_headers=%s",
                error_type,
                session.session_id,
                stream.workspace,
                sent_headers,
            )
            if not sent_headers:
                self._send_json(
                    502,
                    {
                        "error": {
                            "message": "streaming upstream failed before response started",
                            "type": "opencodego_streaming_error",
                            "stream_error_type": error_type,
                        }
                    },
                )
        finally:
            with contextlib.suppress(Exception):
                stream.response.close()

    def _proxy_api_key(self) -> str:
        env_name = self.router.config.litellm_proxy_api_key_env or "OPENCODEGO_PROXY_API_KEY"
        configured_key = os.environ.get(env_name, "")
        if configured_key or env_name == "OPENCODEGO_PROXY_API_KEY":
            return configured_key
        return os.environ.get("OPENCODEGO_PROXY_API_KEY", "")

    def _authorized_for_v1(self) -> bool:
        expected = self._proxy_api_key()
        if not expected:
            return True
        authorization = self.headers.get("Authorization") or ""
        parts = authorization.split(None, 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return False
        return hmac.compare_digest(parts[1].strip(), expected)

    def _require_v1_auth(self, write_body: bool = True) -> bool:
        if self._authorized_for_v1():
            return True
        self._send_json(
            401,
            {"error": {"message": "unauthorized", "type": "authentication_error"}},
            write_body=write_body,
            extra_headers={"WWW-Authenticate": "Bearer"},
        )
        return False

    def _read_request_body(self) -> Optional[bytes]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            return b""
        try:
            content_length = int(raw_length)
        except ValueError:
            self._send_json(
                400,
                {"error": {"message": "invalid Content-Length", "type": "invalid_request_error"}},
            )
            self.close_connection = True
            return None
        if content_length < 0:
            self._send_json(
                400,
                {"error": {"message": "invalid Content-Length", "type": "invalid_request_error"}},
            )
            self.close_connection = True
            return None
        max_bytes = self.router.config.max_request_body_bytes
        if content_length > max_bytes:
            self._send_json(
                413,
                {
                    "error": {
                        "message": f"request body too large; max {max_bytes} bytes",
                        "type": "request_too_large",
                    }
                },
            )
            self.close_connection = True
            return None
        return self.rfile.read(content_length) if content_length else b""

    def _handle_v1_forward(self, method: str, read_body: bool = False) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        if not parsed.path.startswith("/v1/"):
            self._send_json(
                404,
                {"error": {"message": "not found", "type": "not_found"}},
                write_body=method.upper() != "HEAD",
            )
            return
        if not self._require_v1_auth(write_body=method.upper() != "HEAD"):
            return
        body = self._read_request_body() if read_body else b""
        if body is None:
            return
        if method.upper() == "HEAD":
            response = self.router.forward(method, self.path, dict(self.headers), body)
            self._send_proxy_response(response, write_body=False)
            return
        headers = dict(self.headers)
        response, stream = self.router.forward_or_open_stream(
            method,
            self.path,
            headers,
            body,
            stream_intended=request_wants_stream(body, headers),
        )
        if stream is not None and method.upper() != "HEAD":
            self._relay_stream(stream, method, body)
            return
        if response is not None:
            self._send_proxy_response(response, write_body=method.upper() != "HEAD")

    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/health":
            workspace = self.router.current_workspace()
            self._send_json(
                200,
                {
                    "status": "ok",
                    "current_workspace": workspace.name,
                    "workspace_count": len(self.router.config.workspaces),
                    "models": self.registry.models_for_health(),
                },
            )
            return
        if parsed.path == "/v1/models":
            if not self._require_v1_auth():
                return
            try:
                models = self.registry.sync(force=False)
                payload = {
                    "object": "list",
                    "data": [
                        {"id": item["id"], "object": "model", "owned_by": APP_NAME}
                        for item in models
                    ],
                }
                self._send_json(200, payload)
                return
            except Exception as exc:
                logging.getLogger("opencodego.models").warning(
                    "model sync failed during /v1/models; falling back to upstream proxy error=%s",
                    exc.__class__.__name__,
                )
        if parsed.path.startswith("/v1/"):
            if not self._require_v1_auth():
                return
            headers = dict(self.headers)
            response, stream = self.router.forward_or_open_stream(
                "GET",
                self.path,
                headers,
                b"",
                stream_intended=request_wants_stream(b"", headers),
            )
            if stream is not None:
                self._relay_stream(stream, "GET", b"")
                return
            if response is not None:
                self._send_proxy_response(response)
            return
        self._send_json(404, {"error": {"message": "not found", "type": "not_found"}})

    def do_POST(self) -> None:
        self._handle_v1_forward("POST", read_body=True)

    def do_PUT(self) -> None:
        self._handle_v1_forward("PUT", read_body=True)

    def do_PATCH(self) -> None:
        self._handle_v1_forward("PATCH", read_body=True)

    def do_DELETE(self) -> None:
        self._handle_v1_forward("DELETE", read_body=True)

    def do_OPTIONS(self) -> None:
        self._handle_v1_forward("OPTIONS", read_body=True)

    def do_HEAD(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/health":
            workspace = self.router.current_workspace()
            self._send_json(
                200,
                {
                    "status": "ok",
                    "current_workspace": workspace.name,
                    "workspace_count": len(self.router.config.workspaces),
                    "models": self.registry.models_for_health(),
                },
                write_body=False,
            )
            return
        self._handle_v1_forward("HEAD", read_body=False)


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_server(config: AppConfig) -> ThreadingHTTPServer:
    router = WorkspaceRouter(config)
    registry = ModelRegistry(config)
    stream_manager = StreamingSessionManager()
    server = ThreadingHTTPServer((config.listen_host, config.listen_port), ProxyHandler)
    server.router = router  # type: ignore[attr-defined]
    server.registry = registry  # type: ignore[attr-defined]
    server.stream_manager = stream_manager  # type: ignore[attr-defined]
    return server


def command_init_config(args: argparse.Namespace) -> int:
    path = args.path
    if os.path.exists(path) and not args.force:
        print(f"config already exists: {path}", file=sys.stderr)
        print("use --force to overwrite", file=sys.stderr)
        return 2
    atomic_write_json(path, sample_config())
    print(f"wrote sample config: {path}")
    return 0


def command_sync_models(args: argparse.Namespace) -> int:
    config = load_config(args.config, env_file=args.env_file, load_env=not args.no_env_file)
    registry = ModelRegistry(config)
    models = registry.sync(force=args.force)
    print(f"models available in cache: {len(models)}")
    if config.litellm_generated_config_path:
        print(f"wrote LiteLLM generated config: {config.litellm_generated_config_path}")
    return 0


def command_print_litellm_config(args: argparse.Namespace) -> int:
    config = load_config(args.config, env_file=args.env_file, load_env=not args.no_env_file)
    registry = ModelRegistry(config)
    models = registry.sync(force=args.refresh)
    print(render_litellm_config(models, config), end="")
    return 0


def command_serve(args: argparse.Namespace) -> int:
    config = load_config(args.config, env_file=args.env_file, load_env=not args.no_env_file)
    if args.host:
        config = dataclasses.replace(config, listen_host=args.host)
    if args.port:
        config = dataclasses.replace(config, listen_port=args.port)

    registry = ModelRegistry(config)
    try:
        registry.sync(force=False)
    except Exception as exc:
        logging.getLogger("opencodego.models").warning(
            "model sync failed at startup; proxy will continue error=%s",
            exc.__class__.__name__,
        )

    server = make_server(config)
    logging.getLogger("opencodego.http").info(
        "serving on http://%s:%s workspaces=%s",
        config.listen_host,
        config.listen_port,
        len(config.workspaces),
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.getLogger("opencodego.http").info("shutdown requested")
    finally:
        server.server_close()
    return 0


class FakeUpstreamHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse_ok(self) -> None:
        state = self.server.state  # type: ignore[attr-defined]
        delay = float(state.get("stream_delay_seconds") or 0.05)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for chunk in (b"data: one\n\n", b"data: two\n\n", b"data: [DONE]\n\n"):
            self.wfile.write(chunk)
            self.wfile.flush()
            time.sleep(delay)
        self.close_connection = True

    def _send_sse_error_after_first(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        chunk = b"data: one\n\n"
        self.wfile.write(("%X\r\n" % len(chunk)).encode("ascii") + chunk + b"\r\n")
        self.wfile.flush()
        time.sleep(0.05)
        self.wfile.write(b"not-a-chunk-size\r\n")
        self.wfile.flush()
        self.close_connection = True

    def do_GET(self) -> None:
        state = self.server.state  # type: ignore[attr-defined]
        if urllib.parse.urlsplit(self.path).path.endswith("/models"):
            state["model_requests"] += 1
            self._send_json(200, {"object": "list", "data": state["models"]})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        state = self.server.state  # type: ignore[attr-defined]
        content_length = int(self.headers.get("Content-Length") or "0")
        request_body = b""
        if content_length:
            request_body = self.rfile.read(content_length)
        state["post_requests"] += 1
        state["last_accept_encoding"] = self.headers.get("Accept-Encoding", "")
        if state["fail_posts"]:
            self._send_json(429, {"error": {"message": "rate limit reached", "type": "rate_limit"}})
            return
        auth = self.headers.get("Authorization", "")
        if auth != f"Bearer {state['expected_key']}":
            self._send_json(401, {"error": {"message": "bad auth"}})
            return
        payload = extract_request_json(request_body) or {}
        if payload.get("stream") is True:
            behavior = state.get("stream_behavior") or "ok"
            state["stream_requests"] += 1
            if behavior == "error_after_first":
                self._send_sse_error_after_first()
            else:
                self._send_sse_ok()
            return
        if state.get("non_stream_content_encoding"):
            body = b"encoded-but-not-decoded"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Encoding", str(state["non_stream_content_encoding"]))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._send_json(200, {"ok": True, "workspace": state["name"]})


def start_fake_upstream(
    name: str,
    expected_key: str,
    models: Sequence[Mapping[str, Any]],
    fail_posts: bool,
) -> Tuple[ThreadingHTTPServer, str, Dict[str, Any]]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstreamHandler)
    state: Dict[str, Any] = {
        "name": name,
        "expected_key": expected_key,
        "models": list(models),
        "fail_posts": fail_posts,
        "model_requests": 0,
        "post_requests": 0,
        "stream_requests": 0,
        "last_accept_encoding": "",
        "stream_behavior": "ok",
        "stream_delay_seconds": 0.05,
        "non_stream_content_encoding": "",
    }
    server.state = state  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}/v1", state


def command_self_test(args: argparse.Namespace) -> int:
    setup_logging(verbose=args.verbose)
    servers: List[ThreadingHTTPServer] = []
    managed_env_keys = [
        "OPENCODEGO_TEST_KEY_ONE",
        "OPENCODEGO_TEST_KEY_TWO",
        "OPENCODEGO_TEST_PROXY_KEY",
        "OPENCODE_GO_TEST_API_KEY_1",
        "OPENCODE_GO_TEST_API_KEY_2",
        "OPENCODE_GO_TEST_API_KEY_3",
        "OPENCODE_GO_TEST_API_KEY_4",
        "OPENCODE_GO_TEST_REQUIRED_KEY",
    ]
    old_env = {key: os.environ.get(key) for key in managed_env_keys}
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="opencodego-test-") as temp_dir:
        for key in managed_env_keys:
            os.environ.pop(key, None)

        env_config_path = os.path.join(temp_dir, "env-config.json")
        atomic_write_json(
            env_config_path,
            {
                "listen_host": "127.0.0.1",
                "listen_port": 8088,
                "workspaces": [
                    {
                        "name": "env-workspace-1",
                        "base_url": "https://workspace-1.example.com/v1",
                        "api_key_env": "OPENCODE_GO_TEST_API_KEY_1",
                    },
                    {
                        "name": "env-workspace-2",
                        "base_url": "https://workspace-2.example.com/v1",
                        "api_key_env": "OPENCODE_GO_TEST_API_KEY_2",
                    },
                ],
            },
        )
        atomic_write_text(
            os.path.join(temp_dir, DEFAULT_ENV_FILE),
            "OPENCODE_GO_TEST_API_KEY_1=env-loaded-key\n",
        )
        try:
            os.chdir(temp_dir)
            loaded_env_config = load_config(env_config_path)
        finally:
            os.chdir(old_cwd)
        assert os.environ.get("OPENCODE_GO_TEST_API_KEY_1") == "env-loaded-key"
        assert [workspace.name for workspace in loaded_env_config.workspaces] == ["env-workspace-1"]

        plaintext_config_path = os.path.join(temp_dir, "plaintext-config.json")
        atomic_write_json(
            plaintext_config_path,
            {
                "workspaces": [
                    {
                        "name": "plaintext-workspace",
                        "base_url": "https://workspace.example.com/v1",
                        "api_key": "must-not-appear-in-error",
                        "api_key_env": "OPENCODE_GO_TEST_API_KEY_1",
                    }
                ]
            },
        )
        try:
            load_config(plaintext_config_path, load_env=False)
            raise AssertionError("expected plaintext api_key to be rejected")
        except ConfigError as exc:
            message = str(exc)
            assert "api_key_env" in message, message
            assert "must-not-appear-in-error" not in message, message

        missing_api_key_env_config_path = os.path.join(temp_dir, "missing-api-key-env-config.json")
        atomic_write_json(
            missing_api_key_env_config_path,
            {
                "workspaces": [
                    {
                        "name": "missing-api-key-env-workspace",
                        "base_url": "https://workspace.example.com/v1",
                    }
                ]
            },
        )
        try:
            load_config(missing_api_key_env_config_path, load_env=False)
            raise AssertionError("expected missing api_key_env to be rejected")
        except ConfigError as exc:
            assert "api_key_env is required" in str(exc), str(exc)

        all_missing_config_path = os.path.join(temp_dir, "all-missing-config.json")
        atomic_write_json(
            all_missing_config_path,
            {
                "workspaces": [
                    {
                        "name": "all-missing-workspace",
                        "base_url": "https://workspace.example.com/v1",
                        "api_key_env": "OPENCODE_GO_TEST_API_KEY_4",
                    }
                ]
            },
        )
        try:
            load_config(all_missing_config_path, load_env=False)
            raise AssertionError("expected all missing env keys to fail")
        except ConfigError as exc:
            assert "no enabled workspaces" in str(exc), str(exc)

        server1, url1, state1 = start_fake_upstream(
            "workspace-1",
            "key-one",
            [{"id": "alpha", "active": True}, {"id": "old-disabled", "active": False}],
            fail_posts=True,
        )
        server2, url2, state2 = start_fake_upstream(
            "workspace-2",
            "key-two",
            [{"id": "beta", "status": "available"}, {"id": "hidden", "status": "disabled"}],
            fail_posts=False,
        )
        servers.extend([server1, server2])
        os.environ["OPENCODEGO_TEST_KEY_ONE"] = "key-one"
        os.environ["OPENCODEGO_TEST_KEY_TWO"] = "key-two"
        os.environ["OPENCODEGO_TEST_PROXY_KEY"] = "proxy-key"
        config = AppConfig(
            workspaces=(
                WorkspaceConfig("workspace-1", url1, api_key_env="OPENCODEGO_TEST_KEY_ONE"),
                WorkspaceConfig("workspace-2", url2, api_key_env="OPENCODEGO_TEST_KEY_TWO"),
            ),
            listen_port=0,
            request_timeout_seconds=5,
            model_cache_path=os.path.join(temp_dir, "models.json"),
            litellm_generated_config_path=os.path.join(temp_dir, "litellm.yaml"),
            litellm_proxy_api_base="http://127.0.0.1:9999/v1",
            litellm_proxy_api_key_env="OPENCODEGO_TEST_PROXY_KEY",
        )

        anchor_config = dataclasses.replace(
            config,
            workspaces=(
                WorkspaceConfig("anchor-1", "http://127.0.0.1:9/v1"),
                WorkspaceConfig("anchor-2", "http://127.0.0.1:9/v1"),
                WorkspaceConfig("anchor-3", "http://127.0.0.1:9/v1"),
            ),
        )
        anchor_router = WorkspaceRouter(anchor_config)
        anchor_router._commit_success_workspace(anchor_config.workspaces[2])
        switched = anchor_router.switch_after_workspace("anchor-1", "provider_error")
        assert switched.name == "anchor-2", switched.name
        assert anchor_router.current_workspace().name == "anchor-2"
        switched = anchor_router.switch_after_workspace("anchor-3", "timeout")
        assert switched.name == "anchor-1", switched.name
        assert anchor_router.current_workspace().name == "anchor-1"

        registry = ModelRegistry(config)
        models = registry.sync(force=True)
        model_ids = {item["id"] for item in models}
        assert model_ids == {"alpha", "beta"}, model_ids
        assert os.path.exists(config.litellm_generated_config_path or "")

        state2["models"] = []
        pruned_models = registry.sync(force=True)
        assert {item["id"] for item in pruned_models} == {"alpha"}, pruned_models
        state2["models"] = [{"id": "beta", "status": "available"}]

        state1_before = state1["model_requests"]
        state2_before = state2["model_requests"]
        cached_models = registry.sync(force=False)
        assert {item["id"] for item in cached_models} == {"alpha"}
        assert state1["model_requests"] == state1_before
        assert state2["model_requests"] == state2_before

        router = WorkspaceRouter(config)
        response = router.forward(
            "POST",
            "/v1/chat/completions",
            {"Content-Type": "application/json", "Authorization": "Bearer client-key"},
            b'{"model":"alpha","messages":[]}',
        )
        assert response.status == 200, (response.status, response.body)
        payload = json.loads(response.body.decode("utf-8"))
        assert payload["workspace"] == "workspace-2", payload
        assert state1["post_requests"] == 1
        assert state2["post_requests"] == 1
        assert state1["last_accept_encoding"] == "identity"
        assert state2["last_accept_encoding"] == "identity"
        assert router.current_workspace().name == "workspace-2"

        state2["fail_posts"] = True
        exhausted_router = WorkspaceRouter(config)
        exhausted_response = exhausted_router.forward(
            "POST",
            "/v1/chat/completions",
            {"Content-Type": "application/json"},
            b'{"model":"alpha","messages":[]}',
        )
        assert exhausted_response.status == 429, exhausted_response.status
        assert exhausted_router.current_workspace().name == "workspace-1"
        state2["fail_posts"] = False

        proxy_server = make_server(config)
        servers.append(proxy_server)
        proxy_thread = threading.Thread(target=proxy_server.serve_forever, daemon=True)
        proxy_thread.start()
        proxy_host, proxy_port = proxy_server.server_address[:2]
        proxy_base = f"http://{proxy_host}:{proxy_port}"

        try:
            urllib.request.urlopen(f"{proxy_base}/v1/models", timeout=5)
            raise AssertionError("expected missing proxy token to fail")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401, exc.code
            auth_error = json.loads(exc.read().decode("utf-8"))
            assert auth_error["error"]["type"] == "authentication_error", auth_error

        bad_auth_request = urllib.request.Request(
            f"{proxy_base}/v1/models",
            headers={"Authorization": "Bearer wrong-token"},
            method="GET",
        )
        try:
            urllib.request.urlopen(bad_auth_request, timeout=5)
            raise AssertionError("expected bad proxy token to fail")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401, exc.code

        good_auth_request = urllib.request.Request(
            f"{proxy_base}/v1/models",
            headers={"Authorization": "Bearer proxy-key"},
            method="GET",
        )
        with urllib.request.urlopen(good_auth_request, timeout=5) as auth_response:
            auth_payload = json.loads(auth_response.read().decode("utf-8"))
        assert auth_response.getcode() == 200
        assert {item["id"] for item in auth_payload["data"]} == {"alpha"}, auth_payload

        stream_body = b'{"model":"alpha","messages":[],"stream":true}'
        unauthorized_stream = urllib.request.Request(
            f"{proxy_base}/v1/chat/completions",
            data=stream_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(unauthorized_stream, timeout=5)
            raise AssertionError("expected missing proxy token on stream to fail")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401, exc.code

        state1["fail_posts"] = False
        state1["stream_behavior"] = "ok"
        state1["stream_delay_seconds"] = 0.1
        state2_posts_before_stream = state2["post_requests"]
        conn = http.client.HTTPConnection(proxy_host, proxy_port, timeout=5)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=stream_body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer proxy-key",
                "Content-Length": str(len(stream_body)),
            },
        )
        stream_response = conn.getresponse()
        first_stream_chunk = stream_response.read(len(b"data: one\n\n"))
        remaining_read_started = time.monotonic()
        remaining_stream = stream_response.read()
        remaining_read_elapsed = time.monotonic() - remaining_read_started
        conn.close()
        assert stream_response.status == 200, stream_response.status
        assert first_stream_chunk == b"data: one\n\n", first_stream_chunk
        assert b"data: two\n\n" in remaining_stream, remaining_stream
        assert b"data: [DONE]\n\n" in remaining_stream, remaining_stream
        assert remaining_read_elapsed >= 0.15, remaining_read_elapsed
        assert state2["post_requests"] == state2_posts_before_stream
        recent_streams = proxy_server.stream_manager.recent_summaries()  # type: ignore[attr-defined]
        assert recent_streams[-1]["status"] == "completed", recent_streams[-1]
        assert recent_streams[-1]["workspace"] == "workspace-1", recent_streams[-1]

        state1["stream_behavior"] = "error_after_first"
        state1["stream_delay_seconds"] = 0.05
        state2_posts_before_error_stream = state2["post_requests"]
        conn = http.client.HTTPConnection(proxy_host, proxy_port, timeout=5)
        conn.request(
            "POST",
            "/v1/chat/completions",
            body=stream_body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer proxy-key",
                "Content-Length": str(len(stream_body)),
            },
        )
        error_stream_response = conn.getresponse()
        first_error_chunk = error_stream_response.read(len(b"data: one\n\n"))
        with contextlib.suppress(http.client.HTTPException, OSError):
            error_stream_response.read()
        conn.close()
        assert error_stream_response.status == 200, error_stream_response.status
        assert first_error_chunk == b"data: one\n\n", first_error_chunk
        assert state2["post_requests"] == state2_posts_before_error_stream
        assert proxy_server.router.current_workspace().name == "workspace-2"  # type: ignore[attr-defined]
        error_summary = proxy_server.stream_manager.recent_summaries()[-1]  # type: ignore[attr-defined]
        assert error_summary["status"] == "partial", error_summary
        assert error_summary["error_type"] in {"provider_error", "connection_reset", "invalid_chunk"}, error_summary

        next_request = urllib.request.Request(
            f"{proxy_base}/v1/chat/completions",
            data=b'{"model":"alpha","messages":[]}',
            headers={"Content-Type": "application/json", "Authorization": "Bearer proxy-key"},
            method="POST",
        )
        with urllib.request.urlopen(next_request, timeout=5) as next_response:
            next_payload = json.loads(next_response.read().decode("utf-8"))
        assert next_payload["workspace"] == "workspace-2", next_payload
        assert state2["last_accept_encoding"] == "identity"

        state2["non_stream_content_encoding"] = "gzip"
        encoded_request = urllib.request.Request(
            f"{proxy_base}/v1/chat/completions",
            data=b'{"model":"alpha","messages":[]}',
            headers={"Content-Type": "application/json", "Authorization": "Bearer proxy-key"},
            method="POST",
        )
        try:
            urllib.request.urlopen(encoded_request, timeout=5)
            raise AssertionError("expected encoded upstream body to be rejected")
        except urllib.error.HTTPError as exc:
            assert exc.code == 502, exc.code
            assert exc.headers.get("Content-Encoding") is None, dict(exc.headers.items())
            encoded_error = json.loads(exc.read().decode("utf-8"))
            assert encoded_error["error"]["type"] == "opencodego_upstream_encoding_error", encoded_error
        finally:
            state2["non_stream_content_encoding"] = ""

        text = render_litellm_config(pruned_models, config)
        assert "model_list:" in text
        assert "openai/alpha" in text
        assert "openai/beta" not in text
        assert "old-disabled" not in text
        assert "key-one" not in text and "key-two" not in text

    for server in servers:
        server.shutdown()
        server.server_close()
    os.chdir(old_cwd)
    for key, value in old_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    print("self-test passed")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opencodego.py",
        description="OpenCode Go x LiteLLM circular failover proxy and model sync helper.",
    )
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    env_group = parser.add_mutually_exclusive_group()
    env_group.add_argument(
        "--env-file",
        help="load KEY=VALUE pairs from PATH before config validation; defaults to .env",
    )
    env_group.add_argument("--no-env-file", action="store_true", help="do not load .env")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_config = subparsers.add_parser("init-config", help="write a safe sample JSON config")
    init_config.add_argument("path", nargs="?", default="opencodego.config.json")
    init_config.add_argument("--force", action="store_true", help="overwrite an existing config")
    init_config.set_defaults(func=command_init_config)

    serve = subparsers.add_parser("serve", help="start the HTTP proxy")
    serve.add_argument("--config", help=f"config JSON path; also supports {CONFIG_ENV}")
    serve.add_argument("--host", help="override listen_host")
    serve.add_argument("--port", type=int, help="override listen_port")
    serve.set_defaults(func=command_serve)

    sync = subparsers.add_parser("sync-models", help="refresh the model cache and generated LiteLLM config")
    sync.add_argument("--config", help=f"config JSON path; also supports {CONFIG_ENV}")
    sync.add_argument("--force", action="store_true", help="ignore the 24h cache TTL")
    sync.set_defaults(func=command_sync_models)

    print_config = subparsers.add_parser("print-litellm-config", help="print generated LiteLLM YAML-like config")
    print_config.add_argument("--config", help=f"config JSON path; also supports {CONFIG_ENV}")
    print_config.add_argument("--refresh", action="store_true", help="refresh cache before printing")
    print_config.set_defaults(func=command_print_litellm_config)

    self_test = subparsers.add_parser("self-test", help="run standard-library fake-upstream checks")
    self_test.set_defaults(func=command_self_test)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command != "self-test":
        setup_logging(verbose=args.verbose)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        logging.getLogger("opencodego").error("config error: %s", exc)
        return 2
    except Exception as exc:
        logging.getLogger("opencodego").error("fatal error: %s", exc)
        if getattr(args, "verbose", False):
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
