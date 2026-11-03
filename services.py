"""
Chat streaming service.

Runs a LangChain agent (``create_agent``) backed by Claude on Amazon Bedrock
(Converse API via ``langchain-aws``) and translates its event stream into the Vercel AI SDK UI Message
Stream Protocol (SSE) the frontend's useChat consumes:
https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol
"""

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncGenerator

from botocore.config import Config
from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import get_settings
from models import ChatContext
from ngs360_client import NGS360Client
from tools import build_tools

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are the NGS360 assistant. You help users work with their genomics "
    "projects, runs, samples, and jobs. Use the navigate tool when the user "
    "asks to open or go to a specific entity, and use the lookup and search "
    "tools (search_projects, search_runs, get_run, list_project_samples, get_qc, "
    "list_workflow_runs, get_workflow_run, explain_workflow_failure, "
    "lookup_project) to ground your answers in real data rather than guessing. "
    "Sequencing runs are instrument runs (search_runs, get_run); analysis runs are "
    "pipeline executions (list_workflow_runs, get_workflow_run). When asked why an "
    "analysis run failed, call explain_workflow_failure and report its reason; do not "
    "guess. These tools are read-only; you cannot change data. When a search reports "
    "that only part of the records were searched, say so. Be concise and direct."
)


def sse_chunk(payload: dict[str, Any]) -> str:
    """Frame one protocol chunk as an SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"


def message_text(message: dict[str, Any]) -> str:
    """Join the text parts of a single UIMessage."""
    return " ".join(
        part.get("text", "")
        for part in message.get("parts", [])
        if part.get("type") == "text"
    ).strip()


def context_note(context: ChatContext | None) -> str:
    """Render the page/entity context the user attached into a system note."""
    if context is None:
        return ""
    parts: list[str] = []
    if context.page is not None:
        parts.append(f"The user is currently viewing {context.page.type} {context.page.id}.")
    if context.references:
        refs = ", ".join(f"{ref.type} {ref.id}" for ref in context.references)
        parts.append(f"The user referenced: {refs}.")
    return " ".join(parts)


def to_lc_messages(
    messages: list[dict[str, Any]], context: ChatContext | None
) -> list[Any]:
    """Convert the UIMessage history into LangChain messages, with a system prompt."""
    system = SYSTEM_PROMPT
    note = context_note(context)
    if note:
        system = f"{system}\n\n{note}"

    lc_messages: list[Any] = [SystemMessage(system)]
    for message in messages:
        text = message_text(message)
        if not text:
            continue
        role = message.get("role")
        if role == "user":
            lc_messages.append(HumanMessage(text))
        elif role == "assistant":
            lc_messages.append(AIMessage(text))
    return lc_messages


def build_llm() -> ChatBedrockConverse:
    """Bedrock Converse chat model.

    AWS access resolves through the default boto3 chain (ECS task role in
    deployed environments); no API key is configured. Construction makes no
    AWS call; the first request does.
    """
    settings = get_settings()
    options: dict[str, Any] = {}
    if settings.GUARDRAIL_ID and settings.GUARDRAIL_VERSION:
        options = {
            # sync: Bedrock checks each part of the answer before it is streamed on, so nothing blocked leaks out.
            "guardrail_config": {
                "guardrailIdentifier": settings.GUARDRAIL_ID,
                "guardrailVersion": settings.GUARDRAIL_VERSION,
                "trace": "enabled",
                "streamProcessingMode": "sync",
            },
            # The chat UI resends the whole conversation every turn. Without this, one blocked message would be
            # checked again, and block every later turn. Only the newest question and the answer are checked.
            "guard_last_turn_only": True,
        }
    return ChatBedrockConverse(
        model=settings.LLM_MODEL,
        region_name=settings.AWS_REGION,
        max_tokens=8000,
        config=Config(
            read_timeout=300,
            connect_timeout=10,
            retries={"max_attempts": 3, "mode": "adaptive"},
        ),
        **options,
    )


def guardrail_policies(trace: Any) -> list[str]:
    """Names of the guardrail policies that acted, from a Converse trace, e.g. ``content:PROMPT_ATTACK``.

    Only policy names are returned. The trace also carries the text that matched (for example the pasted key), and
    that must never reach a log, so no value other than a policy type or name is read.
    """
    found: set[str] = set()
    guard = (trace or {}).get("guardrail") or {}
    assessments = []
    if isinstance(guard.get("inputAssessment"), dict):
        assessments += list(guard["inputAssessment"].values())
    if isinstance(guard.get("outputAssessments"), dict):
        for group in guard["outputAssessments"].values():
            assessments += list(group or [])
    for a in assessments:
        if not isinstance(a, dict):
            continue
        for f in (a.get("contentPolicy") or {}).get("filters") or []:
            if f.get("action") == "BLOCKED":
                found.add(f"content:{f.get('type')}")
        for e in (a.get("sensitiveInformationPolicy") or {}).get("piiEntities") or []:
            if e.get("action") in ("BLOCKED", "ANONYMIZED"):
                found.add(f"sensitive:{e.get('type')}")
        for t in (a.get("topicPolicy") or {}).get("topics") or []:
            if t.get("action") == "BLOCKED":
                found.add(f"topic:{t.get('name')}")
    return sorted(found)


def build_agent(token: str):
    """Construct a ReAct agent whose tools call the REST API with ``token``."""
    return create_agent(build_llm(), build_tools(NGS360Client(token)))


def _delta_text(chunk: Any) -> str:
    """Extract streamable answer text from an AIMessageChunk.

    Bedrock Converse streams ``content`` as a list of blocks, e.g.
    ``[{"type": "text", "text": "Hi", "index": 0}]``; some versions omit
    ``type`` on text deltas. Tool-use and reasoning blocks are skipped so only
    user-facing text reaches the SSE stream.
    """
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif (
                isinstance(block, dict)
                and block.get("type", "text") == "text"
                and isinstance(block.get("text"), str)
            ):
                parts.append(block["text"])
        return "".join(parts)
    return ""


async def stream_reply(
    messages: list[dict[str, Any]], token: str, context: ChatContext | None
) -> AsyncGenerator[str, None]:
    """Run the agent and stream its reply as UI Message Stream chunks."""
    agent = build_agent(token)
    lc_messages = to_lc_messages(messages, context)

    text_id = "t1"
    yield sse_chunk({"type": "start", "messageId": f"msg_{uuid.uuid4().hex}"})
    yield sse_chunk({"type": "text-start", "id": text_id})

    async for event in agent.astream_events({"messages": lc_messages}, version="v2"):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            delta = _delta_text(event["data"]["chunk"])
            if delta:
                yield sse_chunk({"type": "text-delta", "id": text_id, "delta": delta})

        elif kind == "on_tool_end" and event.get("name") == "navigate":
            tool_input = event["data"].get("input", {})
            destination = tool_input.get("destination")
            entity_id = tool_input.get("id")
            if destination and entity_id:
                yield sse_chunk(
                    {
                        "type": "data-navigate",
                        "data": {"destination": destination, "id": entity_id},
                        "transient": True,
                    }
                )

    yield sse_chunk({"type": "text-end", "id": text_id})
    yield sse_chunk({"type": "finish"})
    yield "data: [DONE]\n\n"


def frame(payload: dict[str, Any]) -> str:
    """One frame of the chat UI's stream (thread / status / text / done / error)."""
    return sse_chunk(payload)


def _log_guardrail(output: Any, owner: str | None) -> None:
    """Record that the guardrail acted. Policy names only; the question and the answer are never logged."""
    meta = getattr(output, "response_metadata", None) or {}
    policies = guardrail_policies(meta.get("trace"))
    if meta.get("stopReason") == "guardrail_intervened" or policies:
        logger.warning("guardrail intervened: owner=%s policies=%s", owner, ",".join(policies) or "unknown")


async def stream_frames(
    messages: list[dict[str, Any]],
    token: str,
    context: ChatContext | None,
    thread_id: str | None,
    *,
    owner: str | None = None,
    store: Any = None,
) -> AsyncGenerator[str, None]:
    """Run the agent and stream its reply as the chat UI's own frames.

    This is the protocol of the REST API's ``POST /api/v1/chat/stream``, which the
    frontend's chat transport maps onto the AI SDK: a ``thread`` frame first, a
    ``status`` frame when a tool is called, ``text`` frames for the answer, then
    ``done`` (or ``error``).

    With a thread ``store`` and an ``owner`` the conversation is saved: the thread is created (or updated) before
    the agent runs, so its id and the user's message survive a failed run, and the answer is added once it has
    streamed. Saving never blocks or fails an answer: a storage error is logged and the reply goes on. Without a
    store the service is stateless and the thread id is only echoed (or minted) for the client's bookkeeping.
    """
    saving = store is not None and owner is not None
    if saving:
        try:
            if thread_id is None:
                thread_id = await asyncio.to_thread(store.create, owner, messages)
            else:
                await asyncio.to_thread(store.save, owner, thread_id, messages)
        except Exception:  # noqa: BLE001 - history is best effort; the answer matters more
            logger.exception("could not save the conversation")
            saving = False
    yield frame({"type": "thread", "thread_id": thread_id or str(uuid.uuid4())})
    reply: list[str] = []
    try:
        agent = build_agent(token)
        lc_messages = to_lc_messages(messages, context)
        async for event in agent.astream_events({"messages": lc_messages}, version="v2"):
            kind = event["event"]
            if kind == "on_tool_start" and event.get("name"):
                yield frame({"type": "status", "tool": event["name"]})
            elif kind == "on_chat_model_end":
                _log_guardrail(event["data"].get("output"), owner)
            elif kind == "on_chat_model_stream":
                delta = _delta_text(event["data"]["chunk"])
                if delta:
                    reply.append(delta)
                    yield frame({"type": "text", "delta": delta})
    except Exception:  # noqa: BLE001 - reported in-band; never leak internals to the client
        logger.exception("chat stream failed")
        yield frame({"type": "error", "message": "The assistant ran into a problem. Please try again."})
        return
    if saving:
        try:
            await asyncio.to_thread(store.save, owner, thread_id, messages, "".join(reply))
        except Exception:  # noqa: BLE001
            logger.exception("could not save the answer")
    yield frame({"type": "done"})
