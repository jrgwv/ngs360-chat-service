"""The chat UI's own protocol: POST /api/v1/chat/stream frames and the thread stubs.

No network: Bedrock is replaced with the scripted fake used in test_bedrock.
"""

import asyncio
import json
import uuid

import pytest
from fastapi.testclient import TestClient

import services
from tests.test_bedrock import FakeBedrockRuntime, _text_turn, _tool_turn


def _install_fake(monkeypatch, turns):
    fake = FakeBedrockRuntime(turns)
    real = services.build_llm

    def build():
        llm = real()
        llm.client = fake
        return llm

    monkeypatch.setattr(services, "build_llm", build)
    return fake


def _frames(thread_id=None):
    async def collect():
        out = []
        msgs = [{"id": "m", "role": "user", "parts": [{"type": "text", "text": "hi"}]}]
        async for line in services.stream_frames(msgs, "tok", None, thread_id):
            out.append(json.loads(line.removeprefix("data: ")))
        return out

    return asyncio.run(collect())


def test_frames_text_reply(monkeypatch):
    _install_fake(monkeypatch, [_text_turn(["Hel", "lo"])])
    frames = _frames()
    assert frames[0]["type"] == "thread"
    uuid.UUID(frames[0]["thread_id"])  # the UI validates this as a uuid
    assert "".join(f["delta"] for f in frames if f["type"] == "text") == "Hello"
    assert frames[-1] == {"type": "done"}
    assert {f["type"] for f in frames} <= {"thread", "status", "text", "done"}


def test_frames_echo_the_supplied_thread_id(monkeypatch):
    _install_fake(monkeypatch, [_text_turn(["x"])])
    tid = str(uuid.uuid4())
    assert _frames(tid)[0] == {"type": "thread", "thread_id": tid}


def test_frames_announce_tool_calls(monkeypatch):
    _install_fake(monkeypatch, [_tool_turn(), _text_turn(["Done."])])
    frames = _frames()
    assert {"type": "status", "tool": "navigate"} in frames
    assert frames[-1] == {"type": "done"}


def test_frames_report_failures_in_band_without_internals(monkeypatch):
    def boom(token):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(services, "build_agent", boom)
    frames = _frames()
    assert frames[-1]["type"] == "error"
    assert "secret" not in json.dumps(frames)
    assert frames[0]["type"] == "thread"


@pytest.fixture
def client():
    from main import app

    return TestClient(app)


def _auth():
    from datetime import datetime, timedelta, timezone

    import jwt

    from tests.conftest import TEST_SECRET

    token = jwt.encode(
        {"sub": str(uuid.uuid4()), "exp": datetime.now(timezone.utc) + timedelta(minutes=5)},
        TEST_SECRET, algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}


def test_stream_endpoint_accepts_the_ui_request_shape(monkeypatch, client):
    _install_fake(monkeypatch, [_text_turn(["ok"])])
    body = {
        "id": "chat-1", "trigger": "submit-message", "thread_id": None,
        "messages": [{"id": "m", "role": "user", "parts": [{"type": "text", "text": "hi"}]}],
        "context": {"page": {"type": "project", "id": "P-1", "project_id": "P-1"}, "references": []},
    }
    r = client.post("/api/v1/chat/stream", json=body, headers=_auth())
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    types = [json.loads(line[6:])["type"] for line in r.text.splitlines() if line.startswith("data: ")]
    assert types[0] == "thread" and types[-1] == "done"


def test_stream_and_threads_require_authentication(client):
    assert client.post("/api/v1/chat/stream", json={"messages": []}).status_code == 401
    assert client.get("/api/v1/chat/threads").status_code == 401
    assert client.delete("/api/v1/chat/threads").status_code == 401


def test_thread_stubs_return_an_empty_history(client):
    h = _auth()
    r = client.get("/api/v1/chat/threads?skip=30&limit=30", headers=h)
    assert r.status_code == 200
    assert r.json() == {"data": [], "total_items": 0, "total_pages": 0, "current_page": 2,
                        "per_page": 30, "has_next": False, "has_prev": False}
    tid = str(uuid.uuid4())
    assert client.get(f"/api/v1/chat/threads/{tid}/messages", headers=h).json() == {
        "thread_id": tid, "messages": []}
    assert client.delete("/api/v1/chat/threads", headers=h).status_code == 204
    assert client.delete(f"/api/v1/chat/threads/{tid}", headers=h).status_code == 204
