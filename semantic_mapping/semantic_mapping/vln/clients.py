"""Provider-agnostic VLM/LLM clients for scene-graph grounding (Sec. IV-D).

The grounding loop only needs "text prompt in, text answer out", so the
interface is a single :meth:`VLMClient.complete`. Backends are thin
standard-library HTTP wrappers with no SDK dependencies, selected by name via
:func:`build_vlm_client`:

* ``keyword``           -- deterministic stand-in (no network): picks the
  instance whose label appears in the instruction. For demos and tests only.
* ``scripted``          -- canned responses, for tests.
* ``openai_compatible`` -- any ``/chat/completions`` endpoint (OpenAI, Gemini's
  OpenAI-compatible endpoint, vLLM, Ollama, LM Studio, ...).
* ``anthropic``         -- the Anthropic Messages API.

API keys are read from an environment variable (``api_key_env``) unless passed
explicitly, so no credential ever needs to live in a config file.
"""
from __future__ import annotations

import http.client
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Callable


class VLMError(RuntimeError):
    """A backend failed to produce an answer (transport error, bad status, refusal)."""


class VLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Return the model's free-text response to ``prompt``."""
        raise NotImplementedError


class ScriptedVLMClient(VLMClient):
    """Returns canned responses (a list consumed in order, or a callable)."""

    def __init__(self, responses: list[str] | Callable[[str], str]) -> None:
        self._responses = responses
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if callable(self._responses):
            return self._responses(prompt)
        if not self._responses:
            raise VLMError("ScriptedVLMClient has no responses left")
        return self._responses.pop(0)


_NODE_LINE = re.compile(r"Instance (\d+) \(([^)]+)\) at \[([-\d.]+), ([-\d.]+), ([-\d.]+)\]")


class KeywordVLMClient(VLMClient):
    """Deterministic stand-in for a real model: no network, no reasoning.

    Parses the serialized nodes back out of the prompt and answers with the
    instance(s) whose label occurs in the instruction, preferring the one
    nearest the origin when several share a label. Exists so the example and
    the ROS node run end-to-end without credentials; it cannot resolve
    relational or temporal instructions -- switch to a real backend for those.
    """

    def complete(self, prompt: str) -> str:
        instruction = prompt.rsplit("Instruction:", 1)[-1].strip().lower()
        candidates = []
        for match in _NODE_LINE.finditer(prompt):
            instance_id, label = int(match.group(1)), match.group(2).lower()
            if label and label in instruction:
                distance = sum(float(match.group(k)) ** 2 for k in (3, 4, 5))
                candidates.append((label, distance, instance_id))
        if not candidates:
            return "I could not find a matching object in the scene graph. <answer></answer>"
        # One target per mentioned label, in the order the labels appear in the instruction.
        best_per_label: dict[str, tuple[float, int]] = {}
        for label, distance, instance_id in candidates:
            if label not in best_per_label or distance < best_per_label[label][0]:
                best_per_label[label] = (distance, instance_id)
        ordered = sorted(best_per_label.items(), key=lambda kv: instruction.find(kv[0]))
        ids = ", ".join(str(instance_id) for _label, (_d, instance_id) in ordered)
        return f"Keyword match on {[label for label, _ in ordered]}. <answer>{ids}</answer>"


_RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
"""Transient statuses: timeouts, conflicts, rate limits, server errors and 529 (overloaded)."""


def _retry_after_seconds(headers) -> float | None:
    """``Retry-After`` in seconds (delta-seconds form only), or ``None``."""
    if headers is None:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _post_json(url: str, body: dict, headers: dict[str, str], timeout: float, *,
               max_retries: int = 2, backoff_s: float = 1.0, max_backoff_s: float = 30.0,
               sleep: Callable[[float], None] | None = None) -> dict:
    """POST ``body`` and return the decoded JSON object.

    Every transport, HTTP, decode and shape failure surfaces as :class:`VLMError`
    so callers only need one ``except``. Transient failures (connection errors,
    timeouts, 408/409/429/5xx/529) are retried up to ``max_retries`` times with
    exponential backoff, honouring ``Retry-After`` (capped at ``max_backoff_s``).
    """
    data = json.dumps(body).encode("utf-8")
    attempt = 0
    while True:
        request = urllib.request.Request(
            url, data=data, method="POST", headers={"Content-Type": "application/json", **headers},
        )
        retry_after = None
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
            try:
                decoded = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise VLMError(f"non-JSON response from {url}") from exc
            if not isinstance(decoded, dict):
                raise VLMError(f"response from {url} is not a JSON object")
            return decoded
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:  # noqa: BLE001 - the body is diagnostic only
                detail = ""
            error = VLMError(f"HTTP {exc.code} from {url}: {detail}")
            if exc.code not in _RETRYABLE_STATUS:
                raise error from exc
            retry_after = _retry_after_seconds(exc.headers)
            last = exc
        except urllib.error.URLError as exc:
            error = VLMError(f"could not reach {url}: {exc.reason}")
            last = exc
        except (TimeoutError, ConnectionError, http.client.HTTPException, OSError) as exc:
            # Read timeouts and dropped connections arrive after urlopen() returned.
            error = VLMError(f"transport error talking to {url}: {exc!r}")
            last = exc
        if attempt >= max_retries:
            raise error from last
        delay = min(max_backoff_s, backoff_s * (2 ** attempt))
        if retry_after is not None:
            delay = min(max_backoff_s, max(delay, retry_after))
        (sleep or time.sleep)(delay)
        attempt += 1


def sanitize_prompt_text(text: str, max_length: int = 2000) -> str:
    """Make untrusted text safe to embed in a single prompt line.

    Control characters (including newlines) collapse to spaces and angle
    brackets are replaced, so a label or instruction cannot open an
    ``<answer>`` tag or start a new, fake graph line.
    """
    text = "".join(" " if (ord(c) < 32 or ord(c) == 127 or c in "\u2028\u2029") else c for c in str(text))
    text = text.replace("<", "(").replace(">", ")")
    text = " ".join(text.split())
    return text[:max_length]


def _resolve_api_key(api_key: str | None, api_key_env: str | None) -> str | None:
    if api_key:
        return api_key
    if api_key_env:
        return os.environ.get(api_key_env) or None
    return None


class OpenAICompatibleClient(VLMClient):
    """Chat-completions client for any OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        api_key_env: str | None = "OPENAI_API_KEY",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        max_retries: int = 2,
        retry_backoff_s: float = 1.0,
    ) -> None:
        if not model:
            raise ValueError("OpenAICompatibleClient needs a model id")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = _resolve_api_key(api_key, api_key_env)
        self.temperature = temperature
        self.max_tokens = _positive_int("max_tokens", max_tokens)
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_s = float(retry_backoff_s)

    def complete(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = _post_json(f"{self.base_url}/chat/completions", body, headers, self.timeout,
                          max_retries=self.max_retries, backoff_s=self.retry_backoff_s)
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"] or ""
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise VLMError(f"unexpected chat-completions response shape: {str(data)[:300]}") from exc
        if not isinstance(content, str):
            raise VLMError("chat-completions content is not text")
        if finish_reason == "length" and "</answer>" not in content:
            raise VLMError(f"response truncated at max_tokens={self.max_tokens} before an answer; raise max_tokens")
        return content


class AnthropicMessagesClient(VLMClient):
    """Anthropic Messages API client (raw HTTP, no SDK dependency).

    ``max_tokens`` bounds thinking *and* answer text together: Claude Opus 5
    thinks adaptively by default, so the default leaves room for reasoning over
    a large scene graph (16000 keeps a non-streaming request well inside the
    HTTP timeout). A response cut off by the cap is an explicit error rather
    than a silently empty answer.
    """

    DEFAULT_MODEL = "claude-opus-5"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = "https://api.anthropic.com",
        api_key: str | None = None,
        api_key_env: str | None = "ANTHROPIC_API_KEY",
        max_tokens: int = 16000,
        timeout: float = 600.0,
        max_retries: int = 2,
        retry_backoff_s: float = 1.0,
    ) -> None:
        self.model = model or self.DEFAULT_MODEL
        self.base_url = base_url.rstrip("/")
        self.api_key = _resolve_api_key(api_key, api_key_env)
        self.max_tokens = _positive_int("max_tokens", max_tokens)
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_s = float(retry_backoff_s)

    def complete(self, prompt: str) -> str:
        if not self.api_key:
            raise VLMError("AnthropicMessagesClient: no API key (set ANTHROPIC_API_KEY or pass api_key)")
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        data = _post_json(f"{self.base_url}/v1/messages", body, headers, self.timeout,
                          max_retries=self.max_retries, backoff_s=self.retry_backoff_s)
        stop_reason = data.get("stop_reason")
        if stop_reason == "refusal":
            details = data.get("stop_details")
            details = details if isinstance(details, dict) else {}
            raise VLMError(f"model refused the request ({details.get('category')}): {details.get('explanation')}")
        blocks = data.get("content")
        if not isinstance(blocks, list):
            raise VLMError(f"unexpected messages response shape: {str(data)[:300]}")
        text = "".join(block.get("text", "") for block in blocks
                       if isinstance(block, dict) and block.get("type") == "text"
                       and isinstance(block.get("text", ""), str))
        if stop_reason == "max_tokens":
            raise VLMError(f"response hit max_tokens={self.max_tokens} (thinking counts toward it) "
                           "before finishing; raise max_tokens")
        if not text and not blocks:
            raise VLMError(f"unexpected messages response shape: {str(data)[:300]}")
        return text


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def build_vlm_client(name: str, **kwargs) -> VLMClient:
    """Factory used by the ROS node and the offline example (config-driven)."""
    name = name.lower()
    if name == "keyword":
        return KeywordVLMClient()
    if name == "scripted":
        return ScriptedVLMClient(kwargs.get("responses", []))
    if name in ("openai_compatible", "openai"):
        return OpenAICompatibleClient(**kwargs)
    if name == "anthropic":
        return AnthropicMessagesClient(**kwargs)
    raise ValueError(f"Unknown VLM client: {name!r} (expected keyword|scripted|openai_compatible|anthropic)")
