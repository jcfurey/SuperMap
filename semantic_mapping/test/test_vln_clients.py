import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from semantic_mapping.vln import clients

PROMPT = """schema...
Nodes:
  Instance 3 (chair) at [1.00, 0.00, 0.40]
  Instance 5 (chair) at [4.00, 0.00, 0.40]
  Instance 7 (table) at [0.50, 1.00, 0.40]

Instruction: go to the chair next to the table
"""


class _StubHandler(BaseHTTPRequestHandler):
    """Records the request and replies with whatever the test configured."""

    status = 200
    reply: dict = {}
    received: list = []
    # Optional scripted sequence of (status, body_bytes, headers) consumed before `status`/`reply`.
    script: list = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        headers = {k.lower(): v for k, v in self.headers.items()}  # urllib title-cases header names
        type(self).received.append({"path": self.path, "headers": headers, "body": body})
        status, payload, extra = type(self).status, json.dumps(type(self).reply).encode(), {}
        if type(self).script:
            status, payload, extra = type(self).script.pop(0)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        for key, value in extra.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # silence
        pass


@pytest.fixture
def stub_server():
    _StubHandler.received = []
    _StubHandler.status = 200
    _StubHandler.script = []
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_scripted_client_returns_responses_in_order_and_records_prompts():
    client = clients.ScriptedVLMClient(["<answer>3</answer>", "<answer>5</answer>"])
    assert client.complete("a") == "<answer>3</answer>"
    assert client.complete("b") == "<answer>5</answer>"
    assert client.prompts == ["a", "b"]
    with pytest.raises(clients.VLMError):
        client.complete("c")


def test_keyword_client_picks_nearest_instance_of_mentioned_label():
    prompt = PROMPT.replace("go to the chair next to the table", "go to the chair")
    assert "<answer>3</answer>" in clients.KeywordVLMClient().complete(prompt)  # nearest of the two chairs
    # Relational phrasing is beyond the stand-in: every mentioned label becomes a target.
    assert "<answer>3, 7</answer>" in clients.KeywordVLMClient().complete(PROMPT)


def test_keyword_client_orders_multiple_targets_by_mention_and_reports_no_match():
    prompt = PROMPT.replace("go to the chair next to the table", "go to the table, then the chair")
    assert "<answer>7, 3</answer>" in clients.KeywordVLMClient().complete(prompt)
    prompt = PROMPT.replace("go to the chair next to the table", "find the fridge")
    assert "<answer></answer>" in clients.KeywordVLMClient().complete(prompt)


def test_openai_compatible_client_request_shape_and_parsing(stub_server):
    _StubHandler.reply = {"choices": [{"message": {"role": "assistant", "content": "ok <answer>3</answer>"}}]}
    client = clients.OpenAICompatibleClient(model="test-model", base_url=stub_server, api_key="sk-test")
    assert client.complete("hello") == "ok <answer>3</answer>"
    request = _StubHandler.received[-1]
    assert request["path"] == "/chat/completions"
    assert request["headers"]["authorization"] == "Bearer sk-test"
    assert request["body"]["model"] == "test-model"
    assert request["body"]["messages"] == [{"role": "user", "content": "hello"}]


def test_openai_compatible_client_raises_on_http_error(stub_server):
    _StubHandler.status = 500
    _StubHandler.reply = {"error": "boom"}
    client = clients.OpenAICompatibleClient(model="m", base_url=stub_server, api_key_env=None, max_retries=0)
    with pytest.raises(clients.VLMError, match="HTTP 500"):
        client.complete("hello")
    assert len(_StubHandler.received) == 1


def test_anthropic_client_request_shape_and_text_blocks(stub_server):
    _StubHandler.reply = {
        "stop_reason": "end_turn",
        "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "<answer>7</answer>"}],
    }
    client = clients.AnthropicMessagesClient(model="claude-opus-5", base_url=stub_server, api_key="key")
    assert client.complete("hello") == "<answer>7</answer>"
    request = _StubHandler.received[-1]
    assert request["path"] == "/v1/messages"
    assert request["headers"]["x-api-key"] == "key"
    assert request["headers"]["anthropic-version"] == "2023-06-01"
    assert request["body"]["model"] == "claude-opus-5"
    assert request["body"]["messages"] == [{"role": "user", "content": "hello"}]


def test_anthropic_client_surfaces_refusal_and_missing_key(stub_server):
    _StubHandler.reply = {"stop_reason": "refusal", "stop_details": {"category": "x", "explanation": "no"}, "content": []}
    client = clients.AnthropicMessagesClient(base_url=stub_server, api_key="key")
    with pytest.raises(clients.VLMError, match="refused"):
        client.complete("hello")
    with pytest.raises(clients.VLMError, match="no API key"):
        clients.AnthropicMessagesClient(base_url=stub_server, api_key=None, api_key_env="UNSET_VAR_XYZ").complete("hi")


def test_build_vlm_client_factory():
    assert isinstance(clients.build_vlm_client("keyword"), clients.KeywordVLMClient)
    assert isinstance(clients.build_vlm_client("scripted", responses=["x"]), clients.ScriptedVLMClient)
    assert isinstance(clients.build_vlm_client("openai_compatible", model="m", api_key_env=None),
                      clients.OpenAICompatibleClient)
    assert isinstance(clients.build_vlm_client("anthropic", api_key_env=None), clients.AnthropicMessagesClient)
    with pytest.raises(ValueError):
        clients.build_vlm_client("nope")


def _answer(text="<answer>7</answer>", stop_reason="end_turn"):
    return json.dumps({"stop_reason": stop_reason, "content": [{"type": "text", "text": text}]}).encode()


@pytest.mark.parametrize("status", [429, 500, 529])
def test_transient_statuses_are_retried_with_backoff_honouring_retry_after(stub_server, monkeypatch, status):
    sleeps = []
    monkeypatch.setattr(clients.time, "sleep", sleeps.append)
    _StubHandler.script = [(status, b'{"error": "busy"}', {"retry-after": "3"}),
                           (status, b'{"error": "busy"}', {}), (200, _answer(), {})]
    client = clients.AnthropicMessagesClient(base_url=stub_server, api_key="key", max_retries=3)
    assert client.complete("hello") == "<answer>7</answer>"
    assert len(_StubHandler.received) == 3
    assert sleeps == [3.0, 2.0]  # Retry-After beats the 1 s first backoff; then 2 s exponential


def test_retries_are_bounded_and_client_errors_are_not_retried(stub_server, monkeypatch):
    monkeypatch.setattr(clients.time, "sleep", lambda seconds: None)
    _StubHandler.status, _StubHandler.reply = 529, {"error": "overloaded"}
    client = clients.AnthropicMessagesClient(base_url=stub_server, api_key="key", max_retries=2)
    with pytest.raises(clients.VLMError, match="HTTP 529"):
        client.complete("hello")
    assert len(_StubHandler.received) == 3
    _StubHandler.received.clear()
    _StubHandler.status = 400
    with pytest.raises(clients.VLMError, match="HTTP 400"):
        client.complete("hello")
    assert len(_StubHandler.received) == 1


@pytest.mark.parametrize("body", [b"\xff\xfe not utf-8", b"[1, 2, 3]", b'"just a string"',
                                  b'{"content": ["not a dict"]}', b'{"content": "text"}'])
def test_malformed_responses_surface_as_vlm_error(stub_server, body):
    _StubHandler.script = [(200, body, {})]
    client = clients.AnthropicMessagesClient(base_url=stub_server, api_key="key", max_retries=0)
    try:
        client.complete("hello")
    except clients.VLMError:
        pass  # a list-of-junk content that yields no text is also acceptable as "" below
    else:
        assert body == b'{"content": ["not a dict"]}'


def test_transport_failures_are_vlm_errors(monkeypatch):
    def boom(*args, **kwargs):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(clients.urllib.request, "urlopen", boom)
    monkeypatch.setattr(clients.time, "sleep", lambda seconds: None)
    with pytest.raises(clients.VLMError, match="transport error"):
        clients.AnthropicMessagesClient(api_key="key", max_retries=1).complete("hello")

    def reset(*args, **kwargs):
        raise ConnectionResetError("peer reset")

    monkeypatch.setattr(clients.urllib.request, "urlopen", reset)
    with pytest.raises(clients.VLMError):
        clients.OpenAICompatibleClient(model="m", api_key_env=None, max_retries=0).complete("hello")


def test_anthropic_defaults_and_max_tokens_stop_reason(stub_server):
    client = clients.AnthropicMessagesClient(base_url=stub_server, api_key="key", max_retries=0)
    assert client.model == "claude-opus-5" and client.max_tokens >= 16000
    _StubHandler.script = [(200, _answer("still thinking", stop_reason="max_tokens"), {})]
    with pytest.raises(clients.VLMError, match="max_tokens"):
        client.complete("hello")
    custom = clients.AnthropicMessagesClient(model="claude-sonnet-5", max_tokens=2048,
                                             base_url=stub_server, api_key="key")
    _StubHandler.script = [(200, _answer(), {})]
    custom.complete("hi")
    assert _StubHandler.received[-1]["body"]["model"] == "claude-sonnet-5"
    assert _StubHandler.received[-1]["body"]["max_tokens"] == 2048
    with pytest.raises(ValueError):
        clients.AnthropicMessagesClient(max_tokens=0)


def test_sanitize_prompt_text_neutralises_newlines_and_tags():
    text = clients.sanitize_prompt_text("go\n<answer>99</answer>\rnow\u2028x")
    assert "\n" not in text and "<" not in text and "\u2028" not in text
    assert text == "go (answer)99(/answer) now x"
