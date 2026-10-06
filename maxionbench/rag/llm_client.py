"""Streaming client for OpenAI-compatible chat endpoints (llama.cpp, vLLM, llm-d gateways).

Uses only the standard library so timing is not distorted by client-side frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass
import http.client
import json
import socket
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from maxionbench.harness.secrets import Secret


@dataclass(frozen=True)
class CompletionResult:
    text: str
    status: str  # "ok" | "error" | "timeout"
    ttft_s: float | None
    e2e_s: float
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    error: str | None = None


def chat_completion(
    base_url: str,
    messages: Sequence[Mapping[str, str]],
    *,
    max_tokens: int,
    timeout_s: float,
    temperature: float = 0.0,
    extra_body: Mapping[str, Any] | None = None,
    headers: Mapping[str, Any] | None = None,
    chat_path: str = "/v1/chat/completions",
) -> CompletionResult:
    """Send one streaming chat request; never raises for transport or server failures.

    Header values may be `Secret` objects; they are revealed only when the request is sent.
    """
    parts = urlsplit(base_url)
    body: dict[str, Any] = {
        "messages": list(messages),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body.update(extra_body or {})
    started = time.perf_counter()
    ttft: float | None = None
    chunks: list[str] = []
    usage: Mapping[str, Any] = {}
    send_headers = {"content-type": "application/json"}
    for name, value in (headers or {}).items():
        send_headers[name] = value.reveal() if isinstance(value, Secret) else str(value)
    conn_cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(parts.hostname or "localhost", parts.port, timeout=timeout_s)
    try:
        conn.request("POST", parts.path.rstrip("/") + chat_path, body=json.dumps(body), headers=send_headers)
        response = conn.getresponse()
        if response.status != 200:
            detail = response.read(300).decode("utf-8", "replace")
            return _failure("error", started, f"http {response.status}: {detail}")
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            for choice in event.get("choices") or []:
                content = (choice.get("delta") or {}).get("content")
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - started
                    chunks.append(content)
            if event.get("usage"):
                usage = event["usage"]
            if time.perf_counter() - started > timeout_s:
                return _failure("timeout", started, "deadline exceeded while streaming")
    except (socket.timeout, TimeoutError):
        return _failure("timeout", started, "socket timeout")
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        return _failure("error", started, f"{type(exc).__name__}: {exc}")
    finally:
        conn.close()
    details = usage.get("prompt_tokens_details") or {}
    return CompletionResult(
        text="".join(chunks),
        status="ok",
        ttft_s=ttft,
        e2e_s=time.perf_counter() - started,
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        cached_tokens=int(details.get("cached_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
    )


def _failure(status: str, started: float, error: str) -> CompletionResult:
    return CompletionResult(
        text="",
        status=status,
        ttft_s=None,
        e2e_s=time.perf_counter() - started,
        prompt_tokens=0,
        cached_tokens=0,
        completion_tokens=0,
        error=error,
    )
