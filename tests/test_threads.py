"""Saved conversations: the transcript projection, both stores, and the HTTP behaviour the chat UI relies on."""

import json
import time
import uuid
from datetime import datetime, timedelta, timezone

import boto3
import jwt
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws

import threads
from tests.conftest import TEST_SECRET
from tests.test_frames import _install_fake
from tests.test_bedrock import _text_turn


def user_msg(text):
    return {"id": "m", "role": "user", "parts": [{"type": "text", "text": text}]}


def asst_msg(text):
    return {"id": "a", "role": "assistant", "parts": [{"type": "text", "text": text}]}


# ---- projection ------------------------------------------------------------------------------------------------


def test_projection_keeps_only_visible_text_and_adds_the_reply():
    messages = [
        user_msg("  Which runs failed? "),
        {"role": "assistant", "parts": [{"type": "tool-search", "text": "secret tool output"}, {"type": "text", "text": "Two."}]},
        {"role": "system", "parts": [{"type": "text", "text": "ignored"}]},
        {"role": "user", "parts": []},
    ]
    out = threads.project_messages(messages, reply=" Because. ")
    assert out == [
        {"role": "user", "text": "Which runs failed?"},
        {"role": "assistant", "text": "Two."},
        {"role": "assistant", "text": "Because."},
    ]
    assert "secret" not in json.dumps(out)


def test_projection_bounds_each_message_and_the_whole_transcript():
    big = "x" * (threads.MAX_MESSAGE_CHARS * 2)
    one = threads.project_messages([user_msg(big)])
    assert len(one[0]["text"]) == threads.MAX_MESSAGE_CHARS
    many = threads.project_messages([user_msg(f"{i} " + big) for i in range(40)])
    assert len(json.dumps(many)) <= threads.MAX_TRANSCRIPT_BYTES
    assert many[-1]["text"].startswith("39 ")  # the newest messages are the ones kept


def test_titles_come_from_the_first_user_message():
    assert threads.make_title([{"role": "user", "text": "  Which\n runs failed?  "}]) == "Which runs failed?"
    long = threads.make_title([{"role": "user", "text": "a" * 200}])
    assert len(long) == threads.TITLE_CHARS and long.endswith("…")
    assert threads.make_title([]) == "New conversation"


def test_thread_ids_must_be_uuids():
    assert threads.is_thread_id(str(uuid.uuid4()))
    for bad in (None, "", "abc", "../../x", str(uuid.uuid4()).upper() + "x"):
        assert not threads.is_thread_id(bad)


# ---- both stores behave the same --------------------------------------------------------------------------------


@pytest.fixture(params=["memory", "dynamodb"])
def make_store(request):
    if request.param == "memory":
        yield lambda **kw: threads.InMemoryThreadStore(**kw)
        return
    with mock_aws():
        boto3.client("dynamodb", region_name="us-east-1").create_table(
            TableName="threads",
            BillingMode="PAY_PER_REQUEST",
            AttributeDefinitions=[
                {"AttributeName": "owner", "AttributeType": "S"},
                {"AttributeName": "thread_id", "AttributeType": "S"},
                {"AttributeName": "updated_at", "AttributeType": "S"},
            ],
            KeySchema=[{"AttributeName": "owner", "KeyType": "HASH"}, {"AttributeName": "thread_id", "KeyType": "RANGE"}],
            LocalSecondaryIndexes=[{
                "IndexName": "by_updated",
                "KeySchema": [{"AttributeName": "owner", "KeyType": "HASH"}, {"AttributeName": "updated_at", "KeyType": "RANGE"}],
                "Projection": {"ProjectionType": "INCLUDE", "NonKeyAttributes": ["title", "created_at", "expires_at"]},
            }],
        )
        yield lambda **kw: threads.DynamoThreadStore("threads", "us-east-1", **kw)


def test_create_list_read_in_newest_first_order(make_store):
    s = make_store()
    first = s.create("alice", [user_msg("first question")])
    time.sleep(0.01)
    second = s.create("alice", [user_msg("second question")])
    items, total = s.list("alice", 0, 20)
    assert total == 2 and [i["thread_id"] for i in items] == [second, first]
    assert items[0]["title"] == "second question" and items[0]["created_at"] and items[0]["updated_at"]
    assert s.messages("alice", first) == [{"role": "user", "text": "first question"}]


def test_saving_updates_the_transcript_and_moves_the_thread_to_the_top(make_store):
    s = make_store()
    a = s.create("alice", [user_msg("one")])
    time.sleep(0.01)
    b = s.create("alice", [user_msg("two")])
    time.sleep(0.01)
    assert s.save("alice", a, [user_msg("one")], reply="answer one")
    assert [i["thread_id"] for i in s.list("alice", 0, 20)[0]] == [a, b]
    assert [m["text"] for m in s.messages("alice", a)] == ["one", "answer one"]


def test_users_only_ever_see_their_own_threads(make_store):
    s = make_store()
    mine = s.create("alice", [user_msg("mine")])
    s.create("bob", [user_msg("bobs")])
    assert s.list("alice", 0, 20)[1] == 1 and s.list("bob", 0, 20)[1] == 1
    assert not s.exists("bob", mine)
    assert s.messages("bob", mine) is None
    assert s.save("bob", mine, [user_msg("hijack")]) is False
    s.delete("bob", mine)  # someone else's id: nothing happens
    assert s.exists("alice", mine) and s.messages("alice", mine)[0]["text"] == "mine"


def test_paging_delete_and_delete_all(make_store):
    s = make_store()
    ids = []
    for i in range(5):
        ids.append(s.create("alice", [user_msg(f"q{i}")]))
        time.sleep(0.005)
    page, total = s.list("alice", 2, 2)
    assert total == 5 and [p["thread_id"] for p in page] == [ids[2], ids[1]]
    s.delete("alice", ids[4])
    assert not s.exists("alice", ids[4]) and s.list("alice", 0, 20)[1] == 4
    s.create("bob", [user_msg("keep")])
    s.delete_all("alice")
    assert s.list("alice", 0, 20)[1] == 0 and s.list("bob", 0, 20)[1] == 1


def test_a_deleted_thread_is_not_brought_back_by_a_late_save(make_store):
    s = make_store()
    tid = s.create("alice", [user_msg("q")])
    s.delete("alice", tid)
    assert s.save("alice", tid, [user_msg("q")], reply="late answer") is False
    assert not s.exists("alice", tid)


def test_the_oldest_thread_makes_room_at_the_cap(make_store):
    s = make_store(max_threads=3)
    ids = []
    for i in range(4):
        ids.append(s.create("alice", [user_msg(f"q{i}")]))
        time.sleep(0.005)
    items, total = s.list("alice", 0, 20)
    assert total == 3 and ids[0] not in [i["thread_id"] for i in items]


def test_expired_threads_are_invisible_even_before_dynamodb_deletes_them(make_store):
    s = make_store(ttl_days=1)
    tid = s.create("alice", [user_msg("q")])
    if isinstance(s, threads.InMemoryThreadStore):
        s._items[("alice", tid)]["expires_at"] = time.time() - 1
    else:
        s.table.update_item(
            Key={"owner": "alice", "thread_id": tid},
            UpdateExpression="SET expires_at = :e", ExpressionAttributeValues={":e": int(time.time()) - 1},
        )
    assert not s.exists("alice", tid)
    assert s.messages("alice", tid) is None
    assert s.list("alice", 0, 20) == ([], 0)


# ---- HTTP behaviour --------------------------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    import main

    store = threads.InMemoryThreadStore()
    monkeypatch.setattr(main, "get_thread_store", lambda: store)
    c = TestClient(main.app)
    c.store = store
    return c


def auth(sub):
    token = jwt.encode({"sub": sub, "exp": datetime.now(timezone.utc) + timedelta(minutes=5)}, TEST_SECRET, algorithm="HS256")
    return {"Authorization": f"Bearer {token}"}


def ask(client, sub, text, thread_id=None, history=()):
    body = {"messages": [*history, user_msg(text)], "thread_id": thread_id}
    r = client.post("/api/v1/chat/stream", json=body, headers=auth(sub))
    frames = [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")] if r.status_code == 200 else []
    return r, frames


def test_a_conversation_is_saved_listed_and_replayed(monkeypatch, client):
    _install_fake(monkeypatch, [_text_turn(["Hel", "lo"]), _text_turn(["Fine."])])
    r, frames = ask(client, "alice", "Hi there")
    assert r.status_code == 200 and frames[0]["type"] == "thread" and frames[-1] == {"type": "done"}
    tid = frames[0]["thread_id"]

    listing = client.get("/api/v1/chat/threads", headers=auth("alice")).json()
    assert listing["total_items"] == 1 and listing["data"][0]["id"] == tid and listing["data"][0]["title"] == "Hi there"

    # The second turn continues the same thread; the UI resends the history.
    r2, frames2 = ask(client, "alice", "How are you?", tid, history=[user_msg("Hi there"), asst_msg("Hello")])
    assert frames2[0] == {"type": "thread", "thread_id": tid}
    transcript = client.get(f"/api/v1/chat/threads/{tid}/messages", headers=auth("alice")).json()
    assert transcript["thread_id"] == tid
    assert [(m["role"], m["parts"][0]["text"]) for m in transcript["messages"]] == [
        ("user", "Hi there"), ("assistant", "Hello"), ("user", "How are you?"), ("assistant", "Fine."),
    ]
    assert client.get("/api/v1/chat/threads", headers=auth("alice")).json()["total_items"] == 1


def test_other_users_cannot_read_continue_or_delete_a_thread(monkeypatch, client):
    _install_fake(monkeypatch, [_text_turn(["ok"]), _text_turn(["no"])])
    _, frames = ask(client, "alice", "private question")
    tid = frames[0]["thread_id"]
    assert client.get(f"/api/v1/chat/threads/{tid}/messages", headers=auth("bob")).status_code == 404
    assert ask(client, "bob", "hello", tid)[0].status_code == 404
    assert client.delete(f"/api/v1/chat/threads/{tid}", headers=auth("bob")).status_code == 204
    assert client.get("/api/v1/chat/threads", headers=auth("bob")).json()["total_items"] == 0
    assert client.get(f"/api/v1/chat/threads/{tid}/messages", headers=auth("alice")).status_code == 200


def test_unknown_or_malformed_thread_ids_are_404(client):
    assert ask(client, "alice", "hi", str(uuid.uuid4()))[0].status_code == 404
    assert ask(client, "alice", "hi", "not-a-uuid")[0].status_code == 404
    assert client.get("/api/v1/chat/threads/not-a-uuid/messages", headers=auth("alice")).status_code == 404
    assert client.delete("/api/v1/chat/threads/not-a-uuid", headers=auth("alice")).status_code == 204


def test_deleting_one_and_all(monkeypatch, client):
    _install_fake(monkeypatch, [_text_turn(["a"]), _text_turn(["b"])])
    t1 = ask(client, "alice", "one")[1][0]["thread_id"]
    ask(client, "alice", "two")
    assert client.delete(f"/api/v1/chat/threads/{t1}", headers=auth("alice")).status_code == 204
    assert client.get(f"/api/v1/chat/threads/{t1}/messages", headers=auth("alice")).status_code == 404
    assert client.get("/api/v1/chat/threads", headers=auth("alice")).json()["total_items"] == 1
    assert client.delete("/api/v1/chat/threads", headers=auth("alice")).status_code == 204
    assert client.get("/api/v1/chat/threads", headers=auth("alice")).json()["total_items"] == 0


def test_a_failed_answer_still_keeps_the_thread_and_the_question(monkeypatch, client):
    import services

    monkeypatch.setattr(services, "build_agent", lambda token: (_ for _ in ()).throw(RuntimeError("boom")))
    _, frames = ask(client, "alice", "will fail")
    assert frames[0]["type"] == "thread" and frames[-1]["type"] == "error"
    tid = frames[0]["thread_id"]
    saved = client.get(f"/api/v1/chat/threads/{tid}/messages", headers=auth("alice")).json()["messages"]
    assert [m["parts"][0]["text"] for m in saved] == ["will fail"]


def test_a_storage_failure_never_breaks_the_answer(monkeypatch, client):
    _install_fake(monkeypatch, [_text_turn(["still answered"])])

    def broken(*args, **kwargs):
        raise RuntimeError("table unavailable")

    monkeypatch.setattr(client.store, "create", broken)
    r, frames = ask(client, "alice", "hi")
    assert r.status_code == 200
    assert "".join(f.get("delta", "") for f in frames) == "still answered" and frames[-1] == {"type": "done"}
    uuid.UUID(frames[0]["thread_id"])


def test_paging_matches_the_list_contract(monkeypatch, client):
    for i in range(3):
        client.store.create("alice", [user_msg(f"q{i}")])
    page = client.get("/api/v1/chat/threads?skip=2&limit=2", headers=auth("alice")).json()
    assert (page["total_items"], page["total_pages"], page["current_page"], page["per_page"]) == (3, 2, 2, 2)
    assert (page["has_next"], page["has_prev"], len(page["data"])) == (False, True, 1)
    assert set(page["data"][0]) == {"id", "title", "created_at", "updated_at"}


def test_without_a_table_the_service_stays_stateless(monkeypatch):
    import main

    monkeypatch.setattr(main, "get_thread_store", lambda: None)
    c = TestClient(main.app)
    _install_fake(monkeypatch, [_text_turn(["ok"])])
    tid = str(uuid.uuid4())
    r, frames = ask(c, "alice", "hi", tid)
    assert r.status_code == 200 and frames[0] == {"type": "thread", "thread_id": tid}
    assert c.get(f"/api/v1/chat/threads/{tid}/messages", headers=auth("alice")).json() == {"thread_id": tid, "messages": []}


def test_the_store_is_off_unless_a_table_is_configured(monkeypatch):
    threads.get_thread_store.cache_clear()
    monkeypatch.delenv("CHAT_THREADS_TABLE", raising=False)
    assert threads.get_thread_store() is None
    threads.get_thread_store.cache_clear()
    with mock_aws():
        monkeypatch.setenv("CHAT_THREADS_TABLE", "threads")
        monkeypatch.setenv("CHAT_THREAD_TTL_DAYS", "7")
        monkeypatch.setenv("CHAT_MAX_THREADS", "5")
        store = threads.get_thread_store()
        assert isinstance(store, threads.DynamoThreadStore) and (store.ttl_days, store.max_threads) == (7, 5)
    threads.get_thread_store.cache_clear()
