"""
Request/response models for the chat API (mirrors the shape the frontend's
useChat hook sends; kept in sync with the REST API's former api/chat/models.py).
"""

from typing import Any

from pydantic import BaseModel


class ChatContextEntity(BaseModel):
    """An entity attached to a chat message: type ("project", "run", "sample",
    ...) and its id."""

    type: str
    id: str


class ChatContext(BaseModel):
    """Context the user attached: the page they're on and any entities they
    referenced via "@/#", so the assistant can scope its answer."""

    page: ChatContextEntity | None = None
    references: list[ChatContextEntity] = []


class ChatRequest(BaseModel):
    """The full UIMessage history sent by the frontend, plus optional context.

    Messages stay loosely typed — the UIMessage shape is owned by the Vercel AI
    SDK protocol; the agent consumes only the parts it understands.
    """

    messages: list[dict[str, Any]]
    context: ChatContext | None = None


class ChatStreamRequest(ChatRequest):
    """Request body of the frontend's ``POST /api/v1/chat/stream``.

    The chat UI sends the whole conversation each time (``messages``) together
    with the thread id it was given; the service keeps no per-thread state, so the
    history in the request is all the agent sees. Unknown fields (``id``,
    ``trigger``) are ignored.
    """

    thread_id: str | None = None
