"""Bedrock guardrail wiring, intervention logging, and scrubbing of credentials before anything is saved."""

import asyncio
import json
import logging

import pytest

import services
import threads
from tests.test_bedrock import FakeBedrockRuntime, _text_turn

KEY = "AKIAIOSFODNN7EXAMPLE"  # the AWS documentation example key, not a real credential


def _env(monkeypatch, on=True):
    for name in ("GUARDRAIL_ID", "GUARDRAIL_VERSION"):
        monkeypatch.delenv(name, raising=False)
    if on:
        monkeypatch.setenv("GUARDRAIL_ID", "gr-abc123")
        monkeypatch.setenv("GUARDRAIL_VERSION", "3")


def _user(text):
    return {"id": "m", "role": "user", "parts": [{"type": "text", "text": text}]}


def test_no_guardrail_unless_both_settings_are_present(monkeypatch):
    _env(monkeypatch, on=False)
    llm = services.build_llm()
    assert llm.guardrail_config is None and llm.guard_last_turn_only is False
    monkeypatch.setenv("GUARDRAIL_ID", "gr-abc123")  # an id without a version is not enough
    assert services.build_llm().guardrail_config is None


def test_guardrail_is_attached_and_checks_only_the_newest_turn(monkeypatch):
    _env(monkeypatch)
    llm = services.build_llm()
    assert llm.guard_last_turn_only is True
    assert llm.guardrail_config == {
        "guardrailIdentifier": "gr-abc123", "guardrailVersion": "3", "trace": "enabled", "streamProcessingMode": "sync",
    }


def test_requests_carry_the_guardrail_and_wrap_only_the_last_user_text(monkeypatch):
    _env(monkeypatch)
    fake = FakeBedrockRuntime([_text_turn(["ok"])])
    real = services.build_llm

    def build():
        llm = real()
        llm.client = fake
        return llm

    monkeypatch.setattr(services, "build_llm", build)

    async def run():
        history = [_user("first question"), {"role": "assistant", "parts": [{"type": "text", "text": "first answer"}]}, _user("second")]
        return [line async for line in services.stream_frames(history, "tok", None, None)]

    asyncio.run(run())
    call = fake.calls[0]
    assert call["guardrailConfig"]["guardrailIdentifier"] == "gr-abc123"
    assert call["guardrailConfig"]["guardrailVersion"] == "3"
    users = [m for m in call["messages"] if m["role"] == "user"]
    assert "guardContent" in users[-1]["content"][0] and "text" in users[0]["content"][0]  # earlier turns not guarded


TRACE = {
    "guardrail": {
        "inputAssessment": {
            "gr-abc123": {
                "contentPolicy": {"filters": [
                    {"type": "PROMPT_ATTACK", "confidence": "HIGH", "action": "BLOCKED"},
                    {"type": "HATE", "confidence": "NONE", "action": "NONE"},
                ]},
                "sensitiveInformationPolicy": {"piiEntities": [{"type": "AWS_ACCESS_KEY", "match": KEY, "action": "BLOCKED"}]},
            }
        },
        "outputAssessments": {"gr-abc123": [{"contentPolicy": {"filters": [{"type": "VIOLENCE", "confidence": "HIGH", "action": "BLOCKED"}]}}]},
    }
}


def test_policy_names_are_read_without_the_matched_text():
    found = services.guardrail_policies(TRACE)
    assert found == ["content:PROMPT_ATTACK", "content:VIOLENCE", "sensitive:AWS_ACCESS_KEY"]
    assert KEY not in json.dumps(found)
    assert services.guardrail_policies(None) == [] and services.guardrail_policies({}) == []
    assert services.guardrail_policies({"guardrail": {"inputAssessment": "garbage"}}) == []


def _blocked_turn(message):
    return [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockDelta": {"delta": {"text": message}, "contentBlockIndex": 0}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "guardrail_intervened"}},
        {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "trace": TRACE}},
    ]


def _frames(monkeypatch, turns, text, store=None, owner=None):
    fake = FakeBedrockRuntime(turns)
    real = services.build_llm

    def build():
        llm = real()
        llm.client = fake
        return llm

    monkeypatch.setattr(services, "build_llm", build)

    async def run():
        return [json.loads(line.removeprefix("data: ")) async for line in
                services.stream_frames([_user(text)], "tok", None, None, owner=owner, store=store)]

    return asyncio.run(run())


def test_an_intervention_is_logged_by_policy_and_the_user_sees_the_blocked_message(monkeypatch, caplog):
    _env(monkeypatch)
    blocked = "I can't help with that request."
    with caplog.at_level(logging.WARNING):
        frames = _frames(monkeypatch, [_blocked_turn(blocked)], f"here is my key {KEY}", owner="alice")
    assert "".join(f.get("delta", "") for f in frames) == blocked and frames[-1] == {"type": "done"}
    log = caplog.text
    assert "guardrail intervened" in log and "owner=alice" in log
    assert "content:PROMPT_ATTACK" in log and "sensitive:AWS_ACCESS_KEY" in log
    assert KEY not in log and "here is my key" not in log  # neither the question nor the match is logged


def test_a_normal_answer_logs_nothing(monkeypatch, caplog):
    _env(monkeypatch)
    with caplog.at_level(logging.WARNING):
        _frames(monkeypatch, [_text_turn(["fine"])], "hello")
    assert "guardrail" not in caplog.text


# ---- scrubbing before saving -------------------------------------------------------------------------------------


@pytest.mark.parametrize("secret", [
    KEY,
    "ASIAIOSFODNN7EXAMPLE",
    "password: hunter2hunter2",
    "aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123456789",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA PRIVATE KEY-----",
])
def test_credentials_are_scrubbed(secret):
    out = threads.redact_secrets(f"please use {secret} for the login")
    assert "[redacted]" in out and "please use" in out and "for the login" in out
    for fragment in ("AKIA", "ASIA", "hunter2", "wJalr", "abcdefghijklmnop", "eyJhbGci", "MIIEow"):
        assert fragment not in out
    assert threads.redact_secrets(out) == out  # idempotent


@pytest.mark.parametrize("text", [
    "Which samples failed QC in project P-20261030-0001?",
    "The token expired yesterday",
    "260929_A01888_0101_ARXEDGDRXX is Ready",
    "sample S-4002 has mean_coverage 11.2",
    "password reset is disabled",
    "Run 4911686 failed with CANNOT_START_CONTAINER_ERROR",
])
def test_ordinary_text_is_left_alone(text):
    assert threads.redact_secrets(text) == text


def test_saved_history_never_holds_a_pasted_key(monkeypatch):
    _env(monkeypatch)
    store = threads.InMemoryThreadStore()
    frames = _frames(monkeypatch, [_blocked_turn("I can't help with that request.")], f"my key is {KEY} please store it",
                     store=store, owner="alice")
    tid = frames[0]["thread_id"]
    stored = json.dumps(store.messages("alice", tid))
    assert KEY not in stored and "[redacted]" in stored
    title = store.list("alice", 0, 10)[0][0]["title"]
    assert KEY not in title
