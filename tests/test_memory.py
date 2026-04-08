"""Tests for the memory subsystem."""

import pytest

from clawscope.memory import InMemoryMemory
from clawscope.memory.session import Session, SessionManager, SessionMemory
from clawscope.message import Msg


# ---------------------------------------------------------------------------
# InMemoryMemory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inmemory_add_and_get() -> None:
    mem = InMemoryMemory()
    msg = Msg(name="user", content="hello", role="user")
    await mem.add([msg])
    messages = await mem.get()
    assert len(messages) == 1
    assert messages[0].content == "hello"


@pytest.mark.asyncio
async def test_inmemory_size() -> None:
    mem = InMemoryMemory()
    assert await mem.size() == 0
    await mem.add([Msg(name="user", content="a", role="user")])
    assert await mem.size() == 1


@pytest.mark.asyncio
async def test_inmemory_clear() -> None:
    mem = InMemoryMemory()
    await mem.add([Msg(name="user", content="test", role="user")])
    await mem.clear()
    assert await mem.size() == 0


@pytest.mark.asyncio
async def test_inmemory_max_messages_trims_oldest() -> None:
    mem = InMemoryMemory(max_messages=3)
    for i in range(5):
        await mem.add([Msg(name="user", content=f"msg{i}", role="user")])
    messages = await mem.get()
    assert len(messages) == 3
    # Only the three newest should remain
    texts = [m.content for m in messages]
    assert "msg2" in texts
    assert "msg3" in texts
    assert "msg4" in texts
    assert "msg0" not in texts


@pytest.mark.asyncio
async def test_inmemory_marks() -> None:
    mem = InMemoryMemory()
    await mem.add([Msg(name="user", content="before", role="user")])
    await mem.add([Msg(name="user", content="at mark", role="user")], mark="cp")
    await mem.add([Msg(name="user", content="after", role="user")])

    # get() without mark returns all 3
    assert await mem.size() == 3

    # get() with mark returns from mark onwards
    since = await mem.get(mark="cp")
    assert len(since) == 2
    assert since[0].content == "at mark"
    assert since[1].content == "after"


@pytest.mark.asyncio
async def test_inmemory_limit() -> None:
    mem = InMemoryMemory()
    for i in range(10):
        await mem.add([Msg(name="user", content=f"m{i}", role="user")])
    messages = await mem.get(limit=3)
    assert len(messages) == 3


@pytest.mark.asyncio
async def test_inmemory_get_since_mark_unknown() -> None:
    mem = InMemoryMemory()
    await mem.add([Msg(name="user", content="x", role="user")])
    result = await mem.get_since_mark("nonexistent")
    assert result == []


@pytest.mark.asyncio
async def test_inmemory_multiple_add_calls() -> None:
    mem = InMemoryMemory()
    await mem.add([
        Msg(name="user", content="a", role="user"),
        Msg(name="assistant", content="b", role="assistant"),
    ])
    assert await mem.size() == 2


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def test_session_add_message() -> None:
    session = Session(key="test:123")
    session.add_message(role="user", content="hello", name="Alice")
    assert len(session.messages) == 1
    assert session.messages[0]["role"] == "user"
    assert session.messages[0]["content"] == "hello"


def test_session_to_and_from_dict() -> None:
    session = Session(key="chan:456")
    session.add_message(role="user", content="ping")
    d = session.to_dict()
    restored = Session.from_dict(d)
    assert restored.key == "chan:456"
    assert len(restored.messages) == 1


# ---------------------------------------------------------------------------
# SessionMemory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_memory_add_get() -> None:
    session = Session(key="s:1")
    mem = SessionMemory(session)
    await mem.add([Msg(name="user", content="hi", role="user")])
    messages = await mem.get()
    assert len(messages) == 1
    assert messages[0].content == "hi"


@pytest.mark.asyncio
async def test_session_memory_clear() -> None:
    session = Session(key="s:2")
    mem = SessionMemory(session)
    await mem.add([Msg(name="user", content="x", role="user")])
    await mem.clear()
    assert await mem.size() == 0


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_manager_get_or_create(tmp_path) -> None:
    manager = SessionManager(workspace=tmp_path)
    session = await manager.get_or_create("telegram:100")
    assert session.key == "telegram:100"


@pytest.mark.asyncio
async def test_session_manager_save_and_reload(tmp_path) -> None:
    manager = SessionManager(workspace=tmp_path)
    session = await manager.get_or_create("slack:200")
    session.add_message(role="user", content="remembered")
    await manager.save(session)

    # Fresh manager should load from disk
    manager2 = SessionManager(workspace=tmp_path)
    reloaded = await manager2.get_or_create("slack:200")
    assert len(reloaded.messages) == 1
    assert reloaded.messages[0]["content"] == "remembered"


@pytest.mark.asyncio
async def test_session_manager_delete(tmp_path) -> None:
    manager = SessionManager(workspace=tmp_path)
    session = await manager.get_or_create("x:1")
    session.add_message(role="user", content="bye")
    await manager.save(session)
    await manager.delete("x:1")
    # After deletion a new empty session is returned
    fresh = await manager.get_or_create("x:1")
    assert len(fresh.messages) == 0


@pytest.mark.asyncio
async def test_session_manager_list_sessions(tmp_path) -> None:
    manager = SessionManager(workspace=tmp_path)
    for key in ("a:1", "b:2", "c:3"):
        s = await manager.get_or_create(key)
        s.add_message(role="user", content="hi")
        await manager.save(s)
    keys = manager.list_sessions()
    assert len(keys) == 3
