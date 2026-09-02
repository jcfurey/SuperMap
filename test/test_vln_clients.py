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

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        headers = {k.lower(): v for k, v in self.headers.items()}  # urllib title-cases header names
        type(self).received.append({"path": self.path, "headers": headers, "body": body})
        payload = json.dumps(type(self).reply).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # silence
        pass


@pytest.fixture
def stub_server():
    _StubHandler.received = []
    _StubHandler.status = 200
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
    client = clients.OpenAICompatibleClient(model="m", base_url=stub_server, api_key_env=None)
    with pytest.raises(clients.VLMError, match="HTTP 500"):
        client.complete("hello")


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
