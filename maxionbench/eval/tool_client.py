"""Non-streaming chat client for tool calling (OpenAI-compatible: Gemini, vLLM, llama.cpp).

Returns a `CompletionResult` whose `text` is the assistant message as JSON, kept verbatim so it can
be replayed in the next turn (Gemini 3 requires its thought signatures to come back unchanged).
"""

from __future__ import annotations

import http.client
import json
import socket
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from maxionbench.harness.secrets import Secret
from maxionbench.rag.llm_client import CompletionResult


def chat_tools(
    base_url: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    max_tokens: int,
    timeout_s: float,
    tools: Sequence[Mapping[str, Any]] = (),
    temperature: float = 0.0,
    extra_body: Mapping[str, Any] | None = None,
    headers: Mapping[str, Any] | None = None,
    chat_path: str = "/v1/chat/completions",
) -> CompletionResult:
    """Send one request; never raises for transport or server failures."""
    parts = urlsplit(base_url)
    body: dict[str, Any] = {"messages": list(messages), "max_tokens": max_tokens, "temperature": temperature}
    if tools:
        body["tools"] = list(tools)
    body.update(extra_body or {})
    send_headers = {"content-type": "application/json"}
    for name, value in (headers or {}).items():
        send_headers[name] = value.reveal() if isinstance(value, Secret) else str(value)
    started = time.perf_counter()
    conn_cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(parts.hostname or "localhost", parts.port, timeout=timeout_s)
    try:
        conn.request("POST", parts.path.rstrip("/") + chat_path, body=json.dumps(body), headers=send_headers)
        response = conn.getresponse()
        raw = response.read()
        if response.status != 200:
            return _failure("error", started, f"http {response.status}: {raw[:300].decode('utf-8', 'replace')}")
        payload = json.loads(raw)
        message = payload["choices"][0]["message"]
    except (socket.timeout, TimeoutError):
        return _failure("timeout", started, "socket timeout")
    except (OSError, http.client.HTTPException, json.JSONDecodeError, KeyError, IndexError) as exc:
        return _failure("error", started, f"{type(exc).__name__}: {exc}")
    finally:
        conn.close()
    usage = payload.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return CompletionResult(
        text=json.dumps(message),
        status="ok",
        ttft_s=None,  # not streamed
        e2e_s=time.perf_counter() - started,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        cached_tokens=int(details.get("cached_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
    )


def _failure(status: str, started: float, error: str) -> CompletionResult:
    return CompletionResult("", status, None, time.perf_counter() - started, 0, 0, 0, error)
