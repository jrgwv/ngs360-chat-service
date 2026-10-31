"""
NGS360 Chat Service — FastAPI entrypoint.

Exposes POST /api/v1/chat, streaming replies in the Vercel AI SDK UI Message
Stream Protocol. Split out of the REST API so the LangChain dependency tree and
the long-lived streaming workload stay isolated.
"""

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import services
from auth import AuthDep
from config import get_settings, validate_security_settings
from models import ChatRequest, ChatStreamRequest

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
    return StreamingResponse(
        services.stream_frames(request.messages, auth.token, request.context, request.thread_id),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


# The chat UI also lists and deletes conversation threads. This service keeps no
# conversation state (the UI resends the history), so there is nothing to list:
# answer with an empty history instead of a 404 so the panel stays quiet.
@app.get("/api/v1/chat/threads")
def list_threads(auth: AuthDep, skip: int = 0, limit: int = 20) -> dict[str, object]:
    limit = max(1, min(limit, 100))
    return {
        "data": [],
        "total_items": 0,
        "total_pages": 0,
        "current_page": max(0, skip) // limit + 1,
        "per_page": limit,
        "has_next": False,
        "has_prev": False,
    }


@app.get("/api/v1/chat/threads/{thread_id}/messages")
def thread_messages(thread_id: str, auth: AuthDep) -> dict[str, object]:
    return {"thread_id": thread_id, "messages": []}


@app.delete("/api/v1/chat/threads", status_code=204)
def delete_all_threads(auth: AuthDep) -> Response:
    return Response(status_code=204)


@app.delete("/api/v1/chat/threads/{thread_id}", status_code=204)
def delete_thread(thread_id: str, auth: AuthDep) -> Response:
    return Response(status_code=204)
