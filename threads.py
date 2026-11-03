"""
Saved conversations for the chat service.

The chat UI sends the whole conversation with every message, so the model never needs the saved copy. What this
module adds is the history list and the transcripts the UI shows when a user reopens a conversation. It is off
unless CHAT_THREADS_TABLE names a DynamoDB table; without it the service stays stateless, as before.

What is stored, per user: the text of each message the user saw (no tool calls, no tool output, no tokens), a title
taken from the first message, timestamps, and an expiry. Every key starts with the signed-in user's id, so a user
can only list, read or delete their own threads, and a thread id from another user simply does not exist.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from functools import lru_cache
from typing import Any

import boto3
from botocore.exceptions import ClientError

from config import get_settings

logger = logging.getLogger(__name__)

TITLE_CHARS = 60
MAX_MESSAGE_CHARS = 20_000
# DynamoDB items are limited to 400 KB; stay well below it.
MAX_TRANSCRIPT_BYTES = 300_000
LIST_SCAN_LIMIT = 1000
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def is_thread_id(value: str | None) -> bool:
    """Thread ids are uuids minted here; anything else cannot exist."""
    return bool(value) and bool(_UUID_RE.match(value))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="milliseconds")


def message_text(message: dict[str, Any]) -> str:
    """The visible text of a UIMessage: its text parts joined, nothing else."""
    parts = message.get("parts") or []
    return "".join(
        str(p.get("text") or "") for p in parts if isinstance(p, dict) and p.get("type") == "text"
    ).strip()


def project_messages(messages: list[dict[str, Any]], reply: str | None = None) -> list[dict[str, Any]]:
    """Reduce a conversation to what the user saw: user and assistant text, each bounded, newest kept.

    ``reply`` is the assistant's answer to the last message when it has just finished streaming.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        text = message_text(m)
        if role in ("user", "assistant") and text:
            out.append({"role": role, "text": text[:MAX_MESSAGE_CHARS]})
    if reply and reply.strip():
        out.append({"role": "assistant", "text": reply.strip()[:MAX_MESSAGE_CHARS]})
    # Drop the oldest messages until the transcript fits one item.
    while out and len(json.dumps(out)) > MAX_TRANSCRIPT_BYTES:
        out.pop(0)
    return out


def to_ui_messages(stored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The stored transcript in the shape the chat UI replays (AI SDK UIMessages)."""
    return [
        {"id": f"m{i}", "role": m["role"], "parts": [{"type": "text", "text": m["text"]}]}
        for i, m in enumerate(stored)
    ]


def make_title(stored: list[dict[str, Any]]) -> str:
    first = next((m["text"] for m in stored if m["role"] == "user"), "")
    title = " ".join(first.split())
    if not title:
        return "New conversation"
    return title if len(title) <= TITLE_CHARS else title[: TITLE_CHARS - 1].rstrip() + "…"


class InMemoryThreadStore:
    """Same behaviour as the DynamoDB store, in memory. For tests and local development only."""

    def __init__(self, ttl_days: int = 90, max_threads: int = 200) -> None:
        self.ttl_days, self.max_threads = ttl_days, max_threads
        self._lock = threading.Lock()
        self._items: dict[tuple[str, str], dict[str, Any]] = {}

    def _expires(self) -> float:
        return time.time() + self.ttl_days * 86400

    def _live(self, owner: str, thread_id: str) -> dict[str, Any] | None:
        item = self._items.get((owner, thread_id))
        if item is None or item["expires_at"] < time.time():
            return None
        return item

    def _owned(self, owner: str) -> list[dict[str, Any]]:
        items = [v for (o, _), v in self._items.items() if o == owner and v["expires_at"] >= time.time()]
        return sorted(items, key=lambda i: i["updated_at"], reverse=True)

    def create(self, owner: str, messages: list[dict[str, Any]]) -> str:
        stored = project_messages(messages)
        with self._lock:
            owned = self._owned(owner)
            for old in owned[self.max_threads - 1 :]:
                self._items.pop((owner, old["thread_id"]), None)
            thread_id, now = str(uuid.uuid4()), _iso(_now())
            self._items[(owner, thread_id)] = {
                "thread_id": thread_id, "title": make_title(stored), "created_at": now, "updated_at": now,
                "messages": stored, "expires_at": self._expires(),
            }
        return thread_id

    def exists(self, owner: str, thread_id: str) -> bool:
        with self._lock:
            return self._live(owner, thread_id) is not None

    def save(self, owner: str, thread_id: str, messages: list[dict[str, Any]], reply: str | None = None) -> bool:
        stored = project_messages(messages, reply)
        with self._lock:
            item = self._live(owner, thread_id)
            if item is None:
                return False
            item.update(messages=stored, updated_at=_iso(_now()), expires_at=self._expires())
            return True

    def list(self, owner: str, skip: int, limit: int) -> tuple[list[dict[str, Any]], int]:
        with self._lock:
            owned = self._owned(owner)
        page = owned[skip : skip + limit]
        return [{k: i[k] for k in ("thread_id", "title", "created_at", "updated_at")} for i in page], len(owned)

    def messages(self, owner: str, thread_id: str) -> list[dict[str, Any]] | None:
        with self._lock:
            item = self._live(owner, thread_id)
            return None if item is None else list(item["messages"])

    def delete(self, owner: str, thread_id: str) -> None:
        with self._lock:
            self._items.pop((owner, thread_id), None)

    def delete_all(self, owner: str) -> None:
        with self._lock:
            for key in [k for k in self._items if k[0] == owner]:
                del self._items[key]


class DynamoThreadStore:
    """Threads in one DynamoDB table.

    Key: owner (hash) + thread_id (range). A local secondary index on updated_at lists a user's threads newest first
    without reading their transcripts. DynamoDB deletes expired items on its own within about two days, so reads also
    ignore anything past ``expires_at``.
    """

    INDEX = "by_updated"

    def __init__(self, table_name: str, region: str, ttl_days: int = 90, max_threads: int = 200) -> None:
        self.table = boto3.resource("dynamodb", region_name=region).Table(table_name)
        self.ttl_days, self.max_threads = ttl_days, max_threads

    def _expires(self) -> int:
        return int(time.time() + self.ttl_days * 86400)

    def _owned(self, owner: str) -> list[dict[str, Any]]:
        """All of the owner's live threads, newest first, without transcripts."""
        from boto3.dynamodb.conditions import Attr, Key

        items: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {
            "IndexName": self.INDEX,
            "KeyConditionExpression": Key("owner").eq(owner),
            "FilterExpression": Attr("expires_at").gt(int(time.time())),
            "ScanIndexForward": False,
            "ConsistentRead": True,
        }
        while len(items) < LIST_SCAN_LIMIT:
            resp = self.table.query(**kwargs)
            items += resp.get("Items", [])
            if "LastEvaluatedKey" not in resp:
                break
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        return items

    def create(self, owner: str, messages: list[dict[str, Any]]) -> str:
        stored = project_messages(messages)
        owned = self._owned(owner)
        for old in owned[self.max_threads - 1 :]:  # make room: the oldest threads go first
            self.delete(owner, old["thread_id"])
        thread_id, now = str(uuid.uuid4()), _iso(_now())
        self.table.put_item(
            Item={
                "owner": owner, "thread_id": thread_id, "title": make_title(stored),
                "created_at": now, "updated_at": now, "messages": json.dumps(stored),
                "expires_at": self._expires(),
            },
            ConditionExpression="attribute_not_exists(thread_id)",
        )
        return thread_id

    def exists(self, owner: str, thread_id: str) -> bool:
        resp = self.table.get_item(
            Key={"owner": owner, "thread_id": thread_id}, ProjectionExpression="thread_id, expires_at",
            ConsistentRead=True,
        )
        item = resp.get("Item")
        return bool(item) and int(item.get("expires_at", 0)) > time.time()

    def save(self, owner: str, thread_id: str, messages: list[dict[str, Any]], reply: str | None = None) -> bool:
        stored = project_messages(messages, reply)
        try:
            self.table.update_item(
                Key={"owner": owner, "thread_id": thread_id},
                UpdateExpression="SET messages = :m, updated_at = :u, expires_at = :e",
                # Never bring back a thread the user deleted while the answer was streaming.
                ConditionExpression="attribute_exists(thread_id)",
                ExpressionAttributeValues={":m": json.dumps(stored), ":u": _iso(_now()), ":e": self._expires()},
            )
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                return False
            raise
        return True

    def list(self, owner: str, skip: int, limit: int) -> tuple[list[dict[str, Any]], int]:
        owned = self._owned(owner)
        page = owned[skip : skip + limit]
        return [{k: i[k] for k in ("thread_id", "title", "created_at", "updated_at")} for i in page], len(owned)

    def messages(self, owner: str, thread_id: str) -> list[dict[str, Any]] | None:
        resp = self.table.get_item(Key={"owner": owner, "thread_id": thread_id}, ConsistentRead=True)
        item = resp.get("Item")
        if not item or int(item.get("expires_at", Decimal(0))) <= time.time():
            return None
        return json.loads(item["messages"])

    def delete(self, owner: str, thread_id: str) -> None:
        self.table.delete_item(Key={"owner": owner, "thread_id": thread_id})

    def delete_all(self, owner: str) -> None:
        from boto3.dynamodb.conditions import Key

        kwargs: dict[str, Any] = {
            "KeyConditionExpression": Key("owner").eq(owner),
            "ProjectionExpression": "thread_id",
            "ConsistentRead": True,
        }
        while True:
            resp = self.table.query(**kwargs)
            with self.table.batch_writer() as batch:
                for item in resp.get("Items", []):
                    batch.delete_item(Key={"owner": owner, "thread_id": item["thread_id"]})
            if "LastEvaluatedKey" not in resp:
                break
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]


@lru_cache
def get_thread_store() -> InMemoryThreadStore | DynamoThreadStore | None:
    """The configured store, or None when saved conversations are switched off."""
    settings = get_settings()
    table = settings.CHAT_THREADS_TABLE
    if not table:
        return None
    return DynamoThreadStore(table, settings.AWS_REGION, settings.CHAT_THREAD_TTL_DAYS, settings.CHAT_MAX_THREADS)
