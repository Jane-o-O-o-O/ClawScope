"""Tests for the message system."""

from clawscope.message import Msg
from clawscope.message.base import (
    AudioBlock,
    ImageBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)


# ---------------------------------------------------------------------------
# Msg construction
# ---------------------------------------------------------------------------


def test_msg_string_content() -> None:
    msg = Msg(name="user", content="hello", role="user")
    assert msg.get_text_content() == "hello"
    assert msg.name == "user"
    assert msg.role == "user"


def test_msg_has_auto_id() -> None:
    msg = Msg(name="user", content="test", role="user")
    assert msg.id
    assert len(msg.id) > 0


def test_msg_has_timestamp() -> None:
    msg = Msg(name="user", content="test", role="user")
    assert msg.timestamp is not None


def test_msg_with_metadata() -> None:
    msg = Msg(name="user", content="test", role="user", metadata={"key": "value"})
    assert msg.metadata["key"] == "value"


def test_two_msgs_have_different_ids() -> None:
    a = Msg(name="user", content="a", role="user")
    b = Msg(name="user", content="b", role="user")
    assert a.id != b.id


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


def test_msg_text_block_content() -> None:
    msg = Msg(name="user", content=[TextBlock(text="hello")], role="user")
    assert msg.get_text_content() == "hello"


def test_msg_multiple_text_blocks_joined() -> None:
    msg = Msg(
        name="user",
        content=[TextBlock(text="hello"), TextBlock(text=" world")],
        role="user",
    )
    text = msg.get_text_content()
    assert "hello" in text
    assert "world" in text


def test_text_block_to_dict() -> None:
    block = TextBlock(text="test")
    d = block.to_dict()
    assert d["type"] == "text"
    assert d["text"] == "test"


def test_image_block_to_dict() -> None:
    block = ImageBlock(source="https://example.com/img.png", source_type="url")
    d = block.to_dict()
    assert d["type"] == "image"


def test_tool_use_block_to_dict() -> None:
    block = ToolUseBlock(id="abc", name="read_file", input={"path": "/tmp/f"})
    d = block.to_dict()
    assert d["type"] == "tool_use"
    assert d["name"] == "read_file"
    assert d["id"] == "abc"


def test_tool_result_block_to_dict() -> None:
    block = ToolResultBlock(tool_use_id="abc", content="file contents", is_error=False)
    d = block.to_dict()
    assert d["type"] == "tool_result"
    assert d["content"] == "file contents"
    assert d["is_error"] is False


def test_tool_result_block_error() -> None:
    block = ToolResultBlock(tool_use_id="abc", content="oops", is_error=True)
    d = block.to_dict()
    assert d["is_error"] is True


# ---------------------------------------------------------------------------
# get_content_blocks
# ---------------------------------------------------------------------------


def test_get_content_blocks_from_string() -> None:
    msg = Msg(name="user", content="hello", role="user")
    blocks = msg.get_content_blocks()
    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].text == "hello"


def test_get_content_blocks_from_list() -> None:
    blocks_in = [TextBlock(text="a"), ToolUseBlock(id="1", name="fn", input={})]
    msg = Msg(name="user", content=blocks_in, role="user")
    blocks_out = msg.get_content_blocks()
    assert len(blocks_out) == 2
