from clawscope.conversation_context import (
    RUNTIME_CONTEXT_TAG,
    attach_runtime_context,
    strip_runtime_context,
)
from clawscope.message import Msg, TextBlock


def test_attach_runtime_context_to_text_message() -> None:
    msg = Msg(name="user", content="hello", role="user")

    wrapped = attach_runtime_context(
        msg,
        channel="cli",
        chat_id="direct",
        session_key="cli:direct",
        sender_id="user",
    )

    assert wrapped.content.startswith(RUNTIME_CONTEXT_TAG)
    assert "hello" in wrapped.content
    assert wrapped.metadata["_runtime_context"] is True


def test_strip_runtime_context_from_text_content() -> None:
    content = (
        f"{RUNTIME_CONTEXT_TAG}\n"
        "Channel: cli\n"
        "Chat ID: direct\n\n"
        "actual user message"
    )

    assert strip_runtime_context(content) == "actual user message"


def test_strip_runtime_context_from_multimodal_content() -> None:
    content = [
        TextBlock(text=f"{RUNTIME_CONTEXT_TAG}\nChannel: cli"),
        TextBlock(text="hello"),
        {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}},
    ]

    stripped = strip_runtime_context(content)

    assert len(stripped) == 2
    assert isinstance(stripped[0], TextBlock)
    assert stripped[0].text == "hello"
    assert stripped[1]["type"] == "image"
