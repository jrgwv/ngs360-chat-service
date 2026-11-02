"""
Local tester for the chat service.

Drives `services.stream_reply` directly (bypassing HTTP/auth), parses the SSE
chunks, and renders streamed text and navigate events. Requires an AWS identity
with Bedrock access (LLM_MODEL / AWS_REGION in .env).

A token is only needed for the data-lookup tools (which call the REST API); set
NGS360_TOKEN to a real bearer token to exercise those, otherwise a placeholder
is used and lookups will fail gracefully while chat/navigate still work.

Usage:
    python scripts/test_chat.py                 # interactive REPL
    python scripts/test_chat.py "your prompt"   # one-shot
"""

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import services  # noqa: E402

TOKEN = os.getenv("NGS360_TOKEN", "local-dev-token")


def user_message(text: str) -> dict[str, Any]:
    return {"id": f"msg_{uuid.uuid4().hex}", "role": "user",
            "parts": [{"type": "text", "text": text}]}


def assistant_message(text: str) -> dict[str, Any]:
    return {"id": f"msg_{uuid.uuid4().hex}", "role": "assistant",
            "parts": [{"type": "text", "text": text}]}


async def send(messages: list[dict[str, Any]]) -> str:
    reply = ""
    print("assistant> ", end="", flush=True)
    async for line in services.stream_reply(messages, TOKEN, None):
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):].strip()
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        kind = chunk.get("type")
        if kind == "text-delta":
            delta = chunk.get("delta", "")
            reply += delta
            print(delta, end="", flush=True)
        elif kind == "data-navigate":
            data = chunk.get("data", {})
            print(f"\n  [navigate → {data.get('destination')} {data.get('id')}]", end="", flush=True)
    print()
    return reply


async def one_shot(prompt: str) -> None:
    await send([user_message(prompt)])


async def repl() -> None:
    print("NGS360 chat tester — /reset to clear, /quit to exit.\n")
    messages: list[dict[str, Any]] = []
    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            print()
            break
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        if text == "/reset":
            messages = []
            print("(conversation reset)\n")
            continue
        messages.append(user_message(text))
        messages.append(assistant_message(await send(messages)))
        print()


def main() -> None:
    if len(sys.argv) > 1:
        asyncio.run(one_shot(" ".join(sys.argv[1:])))
    else:
        asyncio.run(repl())


if __name__ == "__main__":
    main()
