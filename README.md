# OpenCodeGo

Single-file OpenAI-compatible failover proxy and LiteLLM model-sync helper for multiple
OpenCode Go-style workspaces.

`opencodego.py` is designed as a public, dependency-free reference implementation. It runs in front
of several OpenAI-compatible upstream workspaces, routes requests through them, synchronizes active
models into a LiteLLM config fragment, and supports both normal and streaming/SSE responses.

## What This Solves

Many AI providers and workspaces have request, rate, or token limits. OpenCodeGo lets you define a
pool of workspaces:

```text
Workspace 1 / API 1
Workspace 2 / API 2
Workspace 3 / API 3
Workspace 4 / API 4
```

For non-streaming requests, if one workspace hits a retryable failure, the proxy can retry the same
request on the next workspace. For streaming requests, it follows a stricter rule: once a stream has
started, it never switches workspace inside that same stream. If the stream fails, the current stream
ends and the next request starts from the next workspace.

## Features

- Single Python file: `opencodego.py`
- Standard library only
- OpenAI-compatible `/v1/*` proxy
- Circular workspace failover
- Non-streaming same-request retry/failover
- Streaming/SSE chunk relay
- Streaming session ids, lock/unlock logs, and bounded in-memory token buffer
- Next-request failover after streaming errors
- `/health` endpoint
- `/v1/models` endpoint backed by model sync cache
- 24-hour model cache by default
- Stale model removal when upstreams are successfully checked
- LiteLLM YAML-like model config generation
- Proxy Bearer auth for `/v1/*`
- API keys and Authorization headers are not logged
- Built-in self-test using fake local upstreams

## Repository Layout

```text
.
├── README.md
└── opencodego.py
```

Do not commit generated configs, model caches, logs, or real secrets.

## Requirements

- Python 3.10+ recommended
- No pip install required
- Upstream workspaces should expose OpenAI-compatible endpoints such as:
  - `GET /v1/models`
  - `POST /v1/chat/completions`
  - optional streaming via `stream: true` and `text/event-stream`

The exact OpenCode Go API may differ. If your OpenCode Go endpoint is not OpenAI-compatible, adapt
the workspace `base_url`, `models_path`, or add a small adapter while preserving the routing rules
below.

## Quick Start

1. Create a config file:

```bash
python3 opencodego.py init-config ./opencodego.config.json
```

2. Edit `opencodego.config.json` and replace the sample upstream URLs with your workspace URLs.

3. Export secrets in the shell, not in Git:

```bash
export OPENCODEGO_PROXY_API_KEY="change-this-client-facing-token"
export OPENCODEGO_WS1_API_KEY="workspace-1-secret"
export OPENCODEGO_WS2_API_KEY="workspace-2-secret"
export OPENCODEGO_WS3_API_KEY="workspace-3-secret"
export OPENCODEGO_WS4_API_KEY="workspace-4-secret"
```

4. Run the built-in verification:

```bash
python3 -B opencodego.py --verbose self-test
```

5. Start the proxy:

```bash
python3 -B opencodego.py serve --config ./opencodego.config.json
```

6. Check health:

```bash
curl http://127.0.0.1:8088/health
```

7. Query models through the proxy:

```bash
curl http://127.0.0.1:8088/v1/models \
  -H "Authorization: Bearer $OPENCODEGO_PROXY_API_KEY"
```

## Minimal Config

```json
{
  "listen_host": "127.0.0.1",
  "listen_port": 8088,
  "request_timeout_seconds": 60,
  "max_request_body_bytes": 10485760,
  "model_cache_ttl_seconds": 86400,
  "model_cache_path": "./opencodego.models.cache.json",
  "litellm_generated_config_path": "./opencodego.litellm.generated.yaml",
  "litellm_proxy_api_base": "http://127.0.0.1:8088/v1",
  "litellm_proxy_api_key_env": "OPENCODEGO_PROXY_API_KEY",
  "workspaces": [
    {
      "name": "workspace-1",
      "base_url": "https://workspace-1.example.com/v1",
      "api_key_env": "OPENCODEGO_WS1_API_KEY"
    },
    {
      "name": "workspace-2",
      "base_url": "https://workspace-2.example.com/v1",
      "api_key_env": "OPENCODEGO_WS2_API_KEY"
    }
  ]
}
```

### Config Fields

- `listen_host`: proxy bind address. Keep `127.0.0.1` unless you understand the exposure risk.
- `listen_port`: proxy port.
- `request_timeout_seconds`: upstream request timeout.
- `max_request_body_bytes`: maximum request body accepted by the proxy.
- `model_cache_ttl_seconds`: default `86400` seconds.
- `model_cache_path`: local model cache file.
- `litellm_generated_config_path`: optional generated LiteLLM model config path.
- `litellm_proxy_api_base`: the URL LiteLLM should use to call this proxy.
- `litellm_proxy_api_key_env`: env var that stores the client-facing proxy token.
- `workspaces[].name`: stable workspace id used in logs.
- `workspaces[].base_url`: upstream OpenAI-compatible base URL, usually ending in `/v1`.
- `workspaces[].api_key_env`: env var that stores that workspace API key.
- `workspaces[].models_path`: optional override, default `/v1/models`.

Prefer `api_key_env`. Do not put real `api_key` values in a public repository.

## LiteLLM Setup

Generate or refresh model cache and LiteLLM config:

```bash
python3 opencodego.py sync-models --config ./opencodego.config.json --force
```

Print a config fragment:

```bash
python3 opencodego.py print-litellm-config --config ./opencodego.config.json
```

The generated fragment looks like:

```yaml
model_list:
  - model_name: example-model
    litellm_params:
      model: openai/example-model
      api_base: http://127.0.0.1:8088/v1
      api_key: os.environ/OPENCODEGO_PROXY_API_KEY
```

Include that fragment in your LiteLLM config or copy the generated entries into your main LiteLLM
configuration. The proxy intentionally does not set defaults, priorities, costs, routing weights,
model categories, or temperature presets.

## Using The Proxy Directly

Non-streaming:

```bash
curl http://127.0.0.1:8088/v1/chat/completions \
  -H "Authorization: Bearer $OPENCODEGO_PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "example-model",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

Streaming:

```bash
curl -N http://127.0.0.1:8088/v1/chat/completions \
  -H "Authorization: Bearer $OPENCODEGO_PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "example-model",
    "messages": [{"role": "user", "content": "Stream this"}],
    "stream": true
  }'
```

## Routing Rules

### Non-Streaming Requests

Non-streaming requests may retry within the same request:

```text
Request A
WS1 fails with rate limit
same request retries on WS2
WS2 succeeds
Request A returns WS2 response
```

Retryable conditions include rate limit, token limit, request limit, timeout, provider unavailable,
and 5xx-style API errors.

### Streaming Requests

Streaming requests follow v2.1 rules:

```text
Request B starts stream on WS1
WS1 is locked for that stream
chunks are relayed to the client as they arrive
if the stream fails, Request B ends as partial/error
the next request starts from WS2
```

The proxy does not:

- switch workspace in the middle of a stream
- retry a stream after bytes have been sent
- resume from the middle of a failed stream
- merge chunks from different providers

This avoids duplicate text, broken tool calls, malformed SSE, and provider-specific chunk mismatch.

## Streaming Internals

For each stream, the proxy creates a session with:

- `stream_session_id`
- workspace name
- method and path
- model id if present in JSON body
- start and finish timestamps
- status: `completed`, `partial`, or `error`
- error type if any
- chunk count and byte count
- bounded in-memory chunk buffer

The buffer is for diagnostics and state tracking only. It is not persisted, and chunk contents are
not written to logs.

## Headers And Encoding

The proxy requests `Accept-Encoding: identity` from upstreams.

If an upstream returns non-identity `Content-Encoding`, the proxy rejects the response with `502`
instead of stripping the header and corrupting the body. This is intentional and safer for a
reference implementation.

Hop-by-hop headers and client `Authorization` headers are not forwarded directly. The proxy replaces
upstream `Authorization` with the selected workspace API key.

## Security Notes

- Keep `listen_host` as `127.0.0.1` by default.
- Set `OPENCODEGO_PROXY_API_KEY` before exposing the proxy to anything beyond localhost.
- Never commit real API keys.
- Prefer env vars over inline config secrets.
- Do not publish generated cache/config files if they contain private model names or internal URLs.
- Logs avoid request bodies, API keys, Authorization headers, and chunk contents.

Suggested `.gitignore`:

```gitignore
opencodego.config.json
opencodego.models.cache.json
opencodego.litellm.generated.yaml
*.log
__pycache__/
*.pyc
```

## Commands

```bash
python3 opencodego.py --help
python3 opencodego.py init-config ./opencodego.config.json
python3 opencodego.py serve --config ./opencodego.config.json
python3 opencodego.py sync-models --config ./opencodego.config.json
python3 opencodego.py print-litellm-config --config ./opencodego.config.json
python3 opencodego.py self-test
```

You may also set:

```bash
export OPENCODEGO_CONFIG="./opencodego.config.json"
```

Then omit `--config` for commands that load config.

## Testing Before Release

Run:

```bash
python3 -B -c "import py_compile; py_compile.compile('opencodego.py', cfile='/tmp/opencodego.pyc', doraise=True); print('compile ok')"
python3 -B opencodego.py --verbose self-test
python3 -B opencodego.py --help
```

The self-test starts local fake upstream HTTP servers. In sandboxed environments it may need
permission to bind `127.0.0.1`.

## Production Checklist

- Confirm every upstream `base_url` is correct.
- Confirm every `api_key_env` exists in the runtime environment.
- Set `OPENCODEGO_PROXY_API_KEY`.
- Keep the proxy behind localhost, a private network, or an authenticated reverse proxy.
- Run `self-test`.
- Run `sync-models --force`.
- Inspect generated LiteLLM config before enabling it.
- Test one non-streaming request through LiteLLM.
- Test one streaming request through LiteLLM.
- Watch logs for workspace switches, stream errors, and model sync failures.

## AI Agent Implementation Guide

If you are an AI agent applying this repository to a user's environment, follow this exact sequence:

1. Inspect `opencodego.py` and this README.
2. Do not install dependencies; none are required.
3. Do not invent or expose secrets. Ask the user to provide env vars or use already approved secret
   handling.
4. Run `python3 -B opencodego.py --help`.
5. Run `python3 -B opencodego.py --verbose self-test`.
6. Create config with `init-config`.
7. Replace sample `base_url` values with the user's OpenCode Go/OpenAI-compatible workspace URLs.
8. Ensure every workspace uses `api_key_env`, not inline `api_key`.
9. Export or document required env vars:
   - `OPENCODEGO_PROXY_API_KEY`
   - `OPENCODEGO_WS1_API_KEY`
   - `OPENCODEGO_WS2_API_KEY`
   - additional workspace keys as needed
10. Start the proxy locally.
11. Verify `/health`.
12. Verify `/v1/models` with Bearer auth.
13. Run `sync-models --force`.
14. Generate LiteLLM config with `print-litellm-config` or inspect the generated file.
15. Configure LiteLLM to use this proxy's `/v1` API base and `OPENCODEGO_PROXY_API_KEY`.
16. Test non-streaming chat completion.
17. Test streaming chat completion.
18. Only after local verification, help the user decide whether to expose the proxy beyond
    localhost.

Do not change these behavioral contracts unless the user explicitly asks:

- Non-streaming requests may retry/failover inside the same request.
- Streaming requests must not switch workspace after stream start.
- Streaming failures must terminate the current stream and move the next request to the next
  workspace.
- API keys and request/chunk contents must not be logged.
- Model sync may add/remove model entries, but must not alter defaults, costs, priorities, routing
  weights, categories, or temperature presets.

## Known Limits

- No Redis or distributed state.
- No dashboard.
- No cost-based routing.
- No per-workspace cooldown scoring yet.
- Token buffers are in-memory and bounded.
- Streaming resume is intentionally not implemented.
- Exact OpenCode Go API differences may require adapter work if the upstream is not
  OpenAI-compatible.

## License

Add a license before publishing if this repository is public. MIT is a simple default for a small
reference implementation, but choose the license that matches your project policy.
