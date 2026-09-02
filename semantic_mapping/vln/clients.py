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

import json
import os
import re
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


def _post_json(url: str, body: dict, headers: dict[str, str], timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", **headers},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise VLMError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise VLMError(f"could not reach {url}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise VLMError(f"non-JSON response from {url}") from exc


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
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = _resolve_api_key(api_key, api_key_env)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = _post_json(f"{self.base_url}/chat/completions", body, headers, self.timeout)
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise VLMError(f"unexpected chat-completions response shape: {str(data)[:300]}") from exc


class AnthropicMessagesClient(VLMClient):
    """Anthropic Messages API client (raw HTTP, no SDK dependency)."""

    def __init__(
        self,
        model: str = "claude-opus-5",
        base_url: str = "https://api.anthropic.com",
        api_key: str | None = None,
        api_key_env: str | None = "ANTHROPIC_API_KEY",
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = _resolve_api_key(api_key, api_key_env)
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        if not self.api_key:
            raise VLMError("AnthropicMessagesClient: no API key (set ANTHROPIC_API_KEY or pass api_key)")
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        data = _post_json(f"{self.base_url}/v1/messages", body, headers, self.timeout)
        if data.get("stop_reason") == "refusal":
            details = data.get("stop_details") or {}
            raise VLMError(f"model refused the request ({details.get('category')}): {details.get('explanation')}")
        blocks = data.get("content") or []
        text = "".join(block.get("text", "") for block in blocks if block.get("type") == "text")
        if not text and not blocks:
            raise VLMError(f"unexpected messages response shape: {str(data)[:300]}")
        return text


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
