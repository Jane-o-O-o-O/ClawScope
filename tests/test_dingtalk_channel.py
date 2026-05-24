import asyncio
import sys
from types import ModuleType, SimpleNamespace

from clawscope.bus import MessageBus, OutboundMessage
from clawscope.channels.dingtalk import DingTalkChannel
from clawscope.config import DingTalkConfig


class _FakeAckMessage:
    STATUS_OK = 200

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message


class _FakeChatbotMessage:
    TOPIC = "chatbot"


class _FakeChatbotHandler:
    async def process(self, callback):  # pragma: no cover - overridden in tests
        raise NotImplementedError


class _FakeStreamClient:
    def __init__(self, credential: dict[str, str]) -> None:
        self.credential = credential
        self.handlers: dict[str, object] = {}

    def register_callback_handler(self, topic: str, handler: object) -> None:
        self.handlers[topic] = handler

    async def start_forever(self) -> None:
        await asyncio.sleep(0)


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, int]:
        return {"errcode": 0}


class _FakeHttpClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def post(self, webhook: str, json: dict[str, object]) -> _FakeResponse:
        self.calls.append((webhook, json))
        return _FakeResponse()


def test_dingtalk_channel_processes_inbound_callbacks() -> None:
    async def run_test() -> None:
        fake_module = ModuleType("dingtalk_stream")
        fake_module.AIODingTalkStreamClient = _FakeStreamClient
        fake_module.ChatbotHandler = _FakeChatbotHandler
        fake_module.ChatbotMessage = _FakeChatbotMessage
        fake_module.AckMessage = _FakeAckMessage

        original_module = sys.modules.get("dingtalk_stream")
        sys.modules["dingtalk_stream"] = fake_module
        try:
            bus = MessageBus()
            channel = DingTalkChannel(
                bus=bus,
                config=DingTalkConfig(app_key="key", app_secret="secret"),
            )

            await channel.start()

            handler = channel._handler
            assert handler is not None

            ack = await handler.process(
                SimpleNamespace(
                    data={
                        "senderStaffId": "user-1",
                        "conversationId": "chat-1",
                        "text": {"content": "hello ding"},
                        "sessionWebhook": "https://example.test/dingtalk",
                        "msgtype": "text",
                        "msgId": "msg-1",
                        "senderNick": "Jane",
                    }
                )
            )

            inbound = await bus.consume_inbound()
            assert inbound.channel == "dingtalk"
            assert inbound.sender_id == "user-1"
            assert inbound.chat_id == "chat-1"
            assert inbound.content == "hello ding"
            assert inbound.metadata["session_webhook"] == "https://example.test/dingtalk"
            assert channel._session_webhooks["chat-1"] == "https://example.test/dingtalk"
            assert isinstance(ack, _FakeAckMessage)
            assert ack.status == _FakeAckMessage.STATUS_OK

            await channel.stop()
        finally:
            if original_module is None:
                sys.modules.pop("dingtalk_stream", None)
            else:
                sys.modules["dingtalk_stream"] = original_module

    asyncio.run(run_test())


def test_dingtalk_channel_send_uses_session_webhook() -> None:
    async def run_test() -> None:
        bus = MessageBus()
        channel = DingTalkChannel(
            bus=bus,
            config=DingTalkConfig(app_key="key", app_secret="secret"),
        )
        channel._client = object()
        channel._running = True
        channel._session_webhooks["chat-1"] = "https://example.test/dingtalk"
        channel._http_client = _FakeHttpClient()  # type: ignore[assignment]

        await channel.send(
            OutboundMessage(
                channel="dingtalk",
                chat_id="chat-1",
                content="reply text",
                media=["https://example.test/image.png"],
            )
        )

        assert channel._http_client.calls == [  # type: ignore[union-attr]
            (
                "https://example.test/dingtalk",
                {"msgtype": "text", "text": {"content": "reply text"}},
            ),
            (
                "https://example.test/dingtalk",
                {
                    "msgtype": "text",
                    "text": {"content": "https://example.test/image.png"},
                },
            ),
        ]

    asyncio.run(run_test())
