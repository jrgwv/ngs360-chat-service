"""Bedrock model factory and SSE streaming over Converse chunks (Q4).

No network: the Bedrock runtime client is replaced with a fake that replays
Converse ``converse_stream`` events, so the real ``ChatBedrockConverse``
parsing, ``create_agent`` tool loop and our SSE translation all run.
"""

import asyncio
import json
from typing import Any

from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessageChunk

import services


def test_factory_builds_converse_with_configured_model(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "us.anthropic.claude-sonnet-4-6")
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    llm = services.build_llm()
    assert isinstance(llm, ChatBedrockConverse)
    assert llm.model_id == "us.anthropic.claude-sonnet-4-6"
    assert llm.region_name == "us-west-2"
    assert llm.max_tokens == 8000


def test_factory_default_model(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("ENV_SECRETS", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    llm = services.build_llm()
    assert llm.model_id == "us.anthropic.claude-sonnet-4-6"
    assert llm.region_name == "us-east-1"


def test_delta_text_shapes():
    assert services._delta_text(AIMessageChunk(content="hi")) == "hi"
    assert (
        services._delta_text(AIMessageChunk(content=[{"type": "text", "text": "a", "index": 0}]))
        == "a"
    )
    # Older langchain-aws text deltas omit "type".
    assert services._delta_text(AIMessageChunk(content=[{"text": "b", "index": 0}])) == "b"
    # Tool-use and reasoning blocks are not user-facing text.
    assert (
        services._delta_text(
            AIMessageChunk(
                content=[
                    {"type": "tool_use", "id": "t", "name": "navigate", "input": {}, "index": 1},
                    {"type": "reasoning_content", "reasoning_content": {"text": "x"}, "index": 0},
                ]
            )
        )
        == ""
    )


def _text_turn(text_parts: list[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [{"messageStart": {"role": "assistant"}}]
    for part in text_parts:
        events.append({"contentBlockDelta": {"delta": {"text": part}, "contentBlockIndex": 0}})
    events += [
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}}},
    ]
    return events


def _tool_turn() -> list[dict[str, Any]]:
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": "Opening it."}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {
            "contentBlockStart": {
                "start": {"toolUse": {"toolUseId": "tu_1", "name": "navigate"}},
                "contentBlockIndex": 1,
            }
        },
        {
            "contentBlockDelta": {
                "delta": {"toolUse": {"input": '{"destination": "run", '}},
                "contentBlockIndex": 1,
            }
        },
        {
            "contentBlockDelta": {
                "delta": {"toolUse": {"input": '"id": "R-2024-01"}'}},
                "contentBlockIndex": 1,
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "tool_use"}},
        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}}},
    ]


class FakeBedrockRuntime:
    """Replays one scripted Converse stream per call."""

    def __init__(self, turns: list[list[dict[str, Any]]]) -> None:
        self.turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    def converse_stream(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"stream": iter(self.turns.pop(0))}

    def converse(self, **kwargs: Any) -> dict[str, Any]:  # pragma: no cover
        raise AssertionError("agent should stream, not call converse")


def _run_stream(monkeypatch, turns):
    fake = FakeBedrockRuntime(turns)
    real_build_llm = services.build_llm

    def build_llm_with_fake():
        llm = real_build_llm()
        llm.client = fake
        return llm

    monkeypatch.setattr(services, "build_llm", build_llm_with_fake)

    async def collect() -> list[dict[str, Any]]:
        out = []
        msgs = [{"role": "user", "parts": [{"type": "text", "text": "Take me to run R-2024-01"}]}]
        async for line in services.stream_reply(msgs, "tok", None):
            payload = line.removeprefix("data: ").strip()
            out.append({"type": "DONE"} if payload == "[DONE]" else json.loads(payload))
        return out

    return asyncio.run(collect()), fake


def test_stream_text_only(monkeypatch):
    chunks, fake = _run_stream(monkeypatch, [_text_turn(["Hel", "lo", "!"])])
    deltas = "".join(c["delta"] for c in chunks if c["type"] == "text-delta")
    assert deltas == "Hello!"
    assert [c["type"] for c in chunks][:2] == ["start", "text-start"]
    assert [c["type"] for c in chunks][-3:] == ["text-end", "finish", "DONE"]
    assert fake.calls[0]["modelId"] == "us.anthropic.claude-sonnet-4-6"


def test_stream_tool_call_emits_navigate(monkeypatch):
    chunks, fake = _run_stream(monkeypatch, [_tool_turn(), _text_turn(["Done."])])
    nav = [c for c in chunks if c["type"] == "data-navigate"]
    assert nav == [
        {
            "type": "data-navigate",
            "data": {"destination": "run", "id": "R-2024-01"},
            "transient": True,
        }
    ]
    deltas = "".join(c["delta"] for c in chunks if c["type"] == "text-delta")
    assert deltas == "Opening it.Done."
    # Second model call carries the tool result back to Bedrock.
    assert len(fake.calls) == 2
    second_msgs = fake.calls[1]["messages"]
    assert any(
        "toolResult" in block for m in second_msgs for block in m.get("content", [])
    )
    # Tools were advertised to Converse.
    names = {t["toolSpec"]["name"] for t in fake.calls[0]["toolConfig"]["tools"]}
    assert names == {
        "navigate", "lookup_project", "search_projects", "search_runs", "get_run",
        "list_project_samples", "get_qc",
    }
