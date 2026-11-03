"""
NGS360 Chat Service — FastAPI entrypoint.

Exposes POST /api/v1/chat, streaming replies in the Vercel AI SDK UI Message
Stream Protocol. Split out of the REST API so the LangChain dependency tree and
the long-lived streaming workload stay isolated.
"""

import asyncio
import math

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import services
from auth import AuthDep
from config import get_settings, validate_security_settings
from models import ChatRequest, ChatStreamRequest
from threads import get_thread_store, is_thread_id, to_ui_messages

app = FastAPI(title="NGS360 Chat Service")


def cors_origins(client_origin: str | None) -> list[str]:
    """Explicit CORS allowlist from CLIENT_ORIGIN.

    Unset/blank -> [] (no CORS middleware, so no cross-origin access). A
    wildcard is refused: combined with allow_credentials it would let any site
    make credentialed requests.
    """
    origin = (client_origin or "").strip()
    if not origin:
        return []
    if origin == "*":
        raise ValueError("CLIENT_ORIGIN must be an explicit origin, not '*'")
    return [origin]


settings = get_settings()
# Fail fast at import (gunicorn worker boot) on a placeholder/weak JWT secret.
validate_security_settings(settings)
_origins = cors_origins(settings.CLIENT_ORIGIN)
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/chat")
async def chat(request: ChatRequest, auth: AuthDep) -> StreamingResponse:
    """Stream an assistant reply for the given message history."""
    return StreamingResponse(
        services.stream_reply(request.messages, auth.token, request.context),
        media_type="text/event-stream",
        headers={
            "x-vercel-ai-ui-message-stream": "v1",  # required by the AI SDK protocol
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # prevent proxy buffering of the stream
        },
    )


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",  # prevent proxy buffering of the stream
}


@app.post("/api/v1/chat/stream")
async def chat_stream(request: ChatStreamRequest, auth: AuthDep) -> StreamingResponse:
    """Stream a reply as the chat UI's frames (thread, status, text, done, error)."""
    store = get_thread_store()
    if store is not None and request.thread_id is not None:
        # A thread id is only ever one this service minted for this user; anything else is gone.
        if not is_thread_id(request.thread_id) or not await asyncio.to_thread(
            store.exists, auth.username, request.thread_id
        ):
            raise HTTPException(status_code=404, detail="Conversation not found")
    return StreamingResponse(
        services.stream_frames(
            request.messages, auth.token, request.context, request.thread_id,
            owner=auth.username, store=store,
        ),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# The chat UI lists, reopens and deletes conversation threads. With CHAT_THREADS_TABLE set they are saved per user
# (see threads.py). Without it the service keeps no history and answers with an empty one, as it always did, so
# the panel stays quiet instead of showing errors.
@app.get("/api/v1/chat/threads")
def list_threads(auth: AuthDep, skip: int = 0, limit: int = 20) -> dict[str, object]:
    limit = max(1, min(limit, 100))
    skip = max(0, skip)
    store = get_thread_store()
    items, total = ([], 0) if store is None else store.list(auth.username, skip, limit)
    return {
        "data": [
            {"id": i["thread_id"], "title": i["title"], "created_at": i["created_at"], "updated_at": i["updated_at"]}
            for i in items
        ],
        "total_items": total,
        "total_pages": math.ceil(total / limit),
        "current_page": skip // limit + 1,
        "per_page": limit,
        "has_next": skip + limit < total,
        "has_prev": total > 0 and skip > 0,
    }


@app.get("/api/v1/chat/threads/{thread_id}/messages")
def thread_messages(thread_id: str, auth: AuthDep) -> dict[str, object]:
    store = get_thread_store()
    if store is None:
        return {"thread_id": thread_id, "messages": []}
    stored = store.messages(auth.username, thread_id) if is_thread_id(thread_id) else None
    if stored is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"thread_id": thread_id, "messages": to_ui_messages(stored)}


@app.delete("/api/v1/chat/threads", status_code=204)
def delete_all_threads(auth: AuthDep) -> Response:
    store = get_thread_store()
    if store is not None:
        store.delete_all(auth.username)
    return Response(status_code=204)


@app.delete("/api/v1/chat/threads/{thread_id}", status_code=204)
def delete_thread(thread_id: str, auth: AuthDep) -> Response:
    store = get_thread_store()
    if store is not None and is_thread_id(thread_id):
        store.delete(auth.username, thread_id)
    return Response(status_code=204)
