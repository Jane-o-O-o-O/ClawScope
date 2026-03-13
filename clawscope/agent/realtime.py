"""Realtime voice agent for ClawScope."""

from __future__ import annotations

import asyncio
import base64
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable

from loguru import logger

if TYPE_CHECKING:
    pass


class AudioFormat(str, Enum):
    """Supported audio formats."""

    PCM_16 = "pcm16"
    PCM_24 = "pcm24"
    OPUS = "opus"
    MP3 = "mp3"
    MULAW = "g711_ulaw"
    ALAW = "g711_alaw"


@dataclass
class AudioConfig:
    """Audio configuration."""

    sample_rate: int = 24000
    channels: int = 1
    format: AudioFormat = AudioFormat.PCM_16
    chunk_size: int = 4096

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "format": self.format.value,
        }


@dataclass
class RealtimeEvent:
    """Event in realtime session."""

    type: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


class AudioProvider(ABC):
    """Abstract audio input/output provider."""

    @abstractmethod
    async def start(self) -> None:
        """Start audio capture/playback."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop audio capture/playback."""
        pass

    @abstractmethod
    async def read_audio(self) -> AsyncGenerator[bytes, None]:
        """Read audio chunks."""
        pass

    @abstractmethod
    async def write_audio(self, data: bytes) -> None:
        """Write audio data for playback."""
        pass


class MicrophoneProvider(AudioProvider):
    """Microphone audio provider using sounddevice."""

    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()
        self._stream = None
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._running = False

    async def start(self) -> None:
        """Start microphone capture."""
        try:
            import sounddevice as sd
            import numpy as np

            def callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Audio status: {status}")
                if self._running:
                    # Convert to bytes
                    audio_bytes = (indata * 32767).astype(np.int16).tobytes()
                    try:
                        self._queue.put_nowait(audio_bytes)
                    except asyncio.QueueFull:
                        pass

            self._running = True
            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype="float32",
                blocksize=self.config.chunk_size,
                callback=callback,
            )
            self._stream.start()
            logger.info("Microphone started")

        except ImportError:
            raise ImportError(
                "sounddevice required for microphone. "
                "Install with: pip install sounddevice"
            )

    async def stop(self) -> None:
        """Stop microphone capture."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def read_audio(self) -> AsyncGenerator[bytes, None]:
        """Read audio from microphone."""
        while self._running:
            try:
                data = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=0.1,
                )
                yield data
            except asyncio.TimeoutError:
                continue

    async def write_audio(self, data: bytes) -> None:
        """Write audio for playback."""
        try:
            import sounddevice as sd
            import numpy as np

            # Convert bytes to numpy array
            audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767

            # Play audio
            sd.play(audio_array, self.config.sample_rate)
            await asyncio.sleep(len(audio_array) / self.config.sample_rate)

        except ImportError:
            pass


class RealtimeConnection(ABC):
    """Abstract realtime connection to LLM."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection."""
        pass

    @abstractmethod
    async def send_audio(self, data: bytes) -> None:
        """Send audio data."""
        pass

    @abstractmethod
    async def send_text(self, text: str) -> None:
        """Send text input."""
        pass

    @abstractmethod
    async def receive(self) -> AsyncGenerator[RealtimeEvent, None]:
        """Receive events."""
        pass


class OpenAIRealtimeConnection(RealtimeConnection):
    """OpenAI Realtime API connection."""

    WS_URL = "wss://api.openai.com/v1/realtime"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-realtime-preview-2024-10-01",
        voice: str = "alloy",
        instructions: str | None = None,
    ):
        """
        Initialize OpenAI realtime connection.

        Args:
            api_key: OpenAI API key
            model: Model to use
            voice: Voice for speech synthesis
            instructions: System instructions
        """
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.instructions = instructions or "You are a helpful assistant."
        self._ws = None
        self._connected = False

    async def connect(self) -> None:
        """Connect to OpenAI Realtime API."""
        import os

        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets required for realtime. "
                "Install with: pip install websockets"
            )

        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required")

        url = f"{self.WS_URL}?model={self.model}"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Beta": "realtime=v1",
        }

        self._ws = await websockets.connect(url, extra_headers=headers)
        self._connected = True

        # Send session configuration
        await self._send_event({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": self.instructions,
                "voice": self.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
            },
        })

        logger.info("Connected to OpenAI Realtime API")

    async def disconnect(self) -> None:
        """Disconnect from API."""
        self._connected = False
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def send_audio(self, data: bytes) -> None:
        """Send audio data."""
        if not self._connected or not self._ws:
            return

        # Encode as base64
        audio_b64 = base64.b64encode(data).decode()

        await self._send_event({
            "type": "input_audio_buffer.append",
            "audio": audio_b64,
        })

    async def send_text(self, text: str) -> None:
        """Send text input."""
        if not self._connected or not self._ws:
            return

        await self._send_event({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": text,
                    }
                ],
            },
        })

        await self._send_event({"type": "response.create"})

    async def receive(self) -> AsyncGenerator[RealtimeEvent, None]:
        """Receive events from API."""
        while self._connected and self._ws:
            try:
                message = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=0.1,
                )
                data = json.loads(message)
                yield RealtimeEvent(
                    type=data.get("type", "unknown"),
                    data=data,
                )
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if self._connected:
                    logger.error(f"Receive error: {e}")
                break

    async def _send_event(self, event: dict[str, Any]) -> None:
        """Send event to API."""
        if self._ws:
            await self._ws.send(json.dumps(event))


class RealtimeAgent:
    """
    Realtime voice agent for speech-to-speech interactions.

    Features:
    - Real-time audio streaming
    - Voice activity detection
    - Interrupt handling
    - Function calling during conversation
    """

    def __init__(
        self,
        name: str = "RealtimeAgent",
        connection: RealtimeConnection | None = None,
        audio_provider: AudioProvider | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_transcript: Callable[[str, str], None] | None = None,
        on_audio: Callable[[bytes], None] | None = None,
    ):
        """
        Initialize realtime agent.

        Args:
            name: Agent name
            connection: Realtime API connection
            audio_provider: Audio input/output provider
            tools: Tools for function calling
            on_transcript: Callback for transcripts (role, text)
            on_audio: Callback for audio output
        """
        self.name = name
        self.connection = connection or OpenAIRealtimeConnection()
        self.audio_provider = audio_provider
        self.tools = tools or []
        self.on_transcript = on_transcript
        self.on_audio = on_audio

        self._running = False
        self._transcript_buffer = ""

    async def start(self) -> None:
        """Start the realtime session."""
        logger.info(f"Starting {self.name}")

        # Connect to API
        await self.connection.connect()

        # Start audio provider if available
        if self.audio_provider:
            await self.audio_provider.start()

        self._running = True

    async def stop(self) -> None:
        """Stop the realtime session."""
        self._running = False

        if self.audio_provider:
            await self.audio_provider.stop()

        await self.connection.disconnect()
        logger.info(f"Stopped {self.name}")

    async def run(self) -> None:
        """Run the realtime conversation loop."""
        await self.start()

        try:
            # Start audio streaming and event processing concurrently
            await asyncio.gather(
                self._stream_audio(),
                self._process_events(),
            )
        finally:
            await self.stop()

    async def send_text(self, text: str) -> None:
        """Send text input to the conversation."""
        await self.connection.send_text(text)

        if self.on_transcript:
            self.on_transcript("user", text)

    async def _stream_audio(self) -> None:
        """Stream audio from provider to API."""
        if not self.audio_provider:
            return

        async for audio_chunk in self.audio_provider.read_audio():
            if not self._running:
                break
            await self.connection.send_audio(audio_chunk)

    async def _process_events(self) -> None:
        """Process events from API."""
        async for event in self.connection.receive():
            if not self._running:
                break

            await self._handle_event(event)

    async def _handle_event(self, event: RealtimeEvent) -> None:
        """Handle a realtime event."""
        event_type = event.type

        if event_type == "response.audio.delta":
            # Audio output
            audio_b64 = event.data.get("delta", "")
            if audio_b64:
                audio_data = base64.b64decode(audio_b64)

                if self.on_audio:
                    self.on_audio(audio_data)

                if self.audio_provider:
                    await self.audio_provider.write_audio(audio_data)

        elif event_type == "response.audio_transcript.delta":
            # Assistant transcript
            delta = event.data.get("delta", "")
            self._transcript_buffer += delta

        elif event_type == "response.audio_transcript.done":
            # Transcript complete
            if self.on_transcript and self._transcript_buffer:
                self.on_transcript("assistant", self._transcript_buffer)
            self._transcript_buffer = ""

        elif event_type == "conversation.item.input_audio_transcription.completed":
            # User transcript
            transcript = event.data.get("transcript", "")
            if self.on_transcript and transcript:
                self.on_transcript("user", transcript)

        elif event_type == "response.function_call_arguments.done":
            # Function call
            await self._handle_function_call(event.data)

        elif event_type == "error":
            error = event.data.get("error", {})
            logger.error(f"Realtime error: {error}")

    async def _handle_function_call(self, data: dict[str, Any]) -> None:
        """Handle function call from model."""
        call_id = data.get("call_id")
        name = data.get("name")
        arguments = data.get("arguments", "{}")

        logger.info(f"Function call: {name}")

        try:
            args = json.loads(arguments)

            # Find and execute tool
            result = await self._execute_tool(name, args)

            # Send result back
            if hasattr(self.connection, "_send_event"):
                await self.connection._send_event({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps(result),
                    },
                })
                await self.connection._send_event({"type": "response.create"})

        except Exception as e:
            logger.error(f"Function call error: {e}")

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Execute a tool by name."""
        for tool in self.tools:
            if tool.get("name") == name:
                func = tool.get("function")
                if callable(func):
                    if asyncio.iscoroutinefunction(func):
                        return await func(**args)
                    else:
                        return func(**args)
        return {"error": f"Tool not found: {name}"}


__all__ = [
    "RealtimeAgent",
    "RealtimeConnection",
    "OpenAIRealtimeConnection",
    "AudioProvider",
    "MicrophoneProvider",
    "AudioConfig",
    "AudioFormat",
    "RealtimeEvent",
]
