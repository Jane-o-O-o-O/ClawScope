"""OpenTelemetry tracing module for ClawScope."""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Generator

from loguru import logger

if TYPE_CHECKING:
    pass

# Try to import OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None


class TracingConfig:
    """Tracing configuration."""

    def __init__(
        self,
        service_name: str = "clawscope",
        enabled: bool = True,
        exporter: str = "otlp",
        endpoint: str | None = None,
        console_export: bool = False,
        sample_rate: float = 1.0,
    ):
        """
        Initialize tracing config.

        Args:
            service_name: Service name for traces
            enabled: Enable tracing
            exporter: Exporter type (otlp, jaeger, zipkin)
            endpoint: Exporter endpoint
            console_export: Also export to console
            sample_rate: Sampling rate (0.0 to 1.0)
        """
        self.service_name = service_name
        self.enabled = enabled
        self.exporter = exporter
        self.endpoint = endpoint
        self.console_export = console_export
        self.sample_rate = sample_rate


class Tracer:
    """
    OpenTelemetry tracer wrapper for ClawScope.

    Provides simplified tracing API with fallback for
    when OpenTelemetry is not available.
    """

    def __init__(self, config: TracingConfig | None = None):
        """
        Initialize tracer.

        Args:
            config: Tracing configuration
        """
        self.config = config or TracingConfig()
        self._tracer = None
        self._provider = None
        self._initialized = False

    def initialize(self) -> bool:
        """
        Initialize the tracer.

        Returns:
            True if initialized successfully
        """
        if not OTEL_AVAILABLE:
            logger.warning("OpenTelemetry not available, tracing disabled")
            return False

        if not self.config.enabled:
            return False

        try:
            # Create resource
            resource = Resource.create({
                ResourceAttributes.SERVICE_NAME: self.config.service_name,
                ResourceAttributes.SERVICE_VERSION: "0.1.0",
            })

            # Create provider
            self._provider = TracerProvider(resource=resource)

            # Add exporter
            self._add_exporter()

            # Set as global provider
            trace.set_tracer_provider(self._provider)

            # Get tracer
            self._tracer = trace.get_tracer(self.config.service_name)

            self._initialized = True
            logger.info(f"Tracing initialized: {self.config.service_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize tracing: {e}")
            return False

    def _add_exporter(self) -> None:
        """Add span exporter based on configuration."""
        if self.config.console_export:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            self._provider.add_span_processor(
                BatchSpanProcessor(ConsoleSpanExporter())
            )

        if self.config.exporter == "otlp":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )

                exporter = OTLPSpanExporter(
                    endpoint=self.config.endpoint or "http://localhost:4317",
                )
                self._provider.add_span_processor(BatchSpanProcessor(exporter))

            except ImportError:
                logger.warning("OTLP exporter not available")

        elif self.config.exporter == "jaeger":
            try:
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter

                exporter = JaegerExporter(
                    agent_host_name=self.config.endpoint or "localhost",
                    agent_port=6831,
                )
                self._provider.add_span_processor(BatchSpanProcessor(exporter))

            except ImportError:
                logger.warning("Jaeger exporter not available")

    def shutdown(self) -> None:
        """Shutdown the tracer."""
        if self._provider:
            self._provider.shutdown()
            self._initialized = False

    @contextmanager
    def span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        kind: str = "internal",
    ) -> Generator[Any, None, None]:
        """
        Create a trace span.

        Args:
            name: Span name
            attributes: Span attributes
            kind: Span kind (internal, server, client, producer, consumer)

        Yields:
            Span object
        """
        if not self._initialized or not self._tracer:
            # Fallback: yield a dummy span
            yield DummySpan(name)
            return

        # Map kind string to SpanKind
        kind_map = {
            "internal": trace.SpanKind.INTERNAL,
            "server": trace.SpanKind.SERVER,
            "client": trace.SpanKind.CLIENT,
            "producer": trace.SpanKind.PRODUCER,
            "consumer": trace.SpanKind.CONSUMER,
        }
        span_kind = kind_map.get(kind, trace.SpanKind.INTERNAL)

        with self._tracer.start_as_current_span(
            name,
            kind=span_kind,
            attributes=attributes,
        ) as span:
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def trace(
        self,
        name: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Callable:
        """
        Decorator to trace a function.

        Args:
            name: Span name (defaults to function name)
            attributes: Additional attributes

        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            span_name = name or func.__name__

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.span(span_name, attributes) as span:
                    span.set_attribute("function", func.__name__)
                    return await func(*args, **kwargs)

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.span(span_name, attributes) as span:
                    span.set_attribute("function", func.__name__)
                    return func(*args, **kwargs)

            if asyncio_iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper

        return decorator


class DummySpan:
    """Dummy span for when tracing is disabled."""

    def __init__(self, name: str):
        self.name = name
        self._attributes: dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute."""
        self._attributes[key] = value

    def set_status(self, status: Any) -> None:
        """Set status (no-op)."""
        pass

    def record_exception(self, exception: Exception) -> None:
        """Record exception (no-op)."""
        pass

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add event (no-op)."""
        pass


def asyncio_iscoroutinefunction(func: Callable) -> bool:
    """Check if function is async."""
    import asyncio
    return asyncio.iscoroutinefunction(func)


class AgentTracer:
    """
    Specialized tracer for agent operations.

    Provides semantic tracing for agent activities.
    """

    def __init__(self, tracer: Tracer, agent_name: str):
        """
        Initialize agent tracer.

        Args:
            tracer: Base tracer
            agent_name: Agent name
        """
        self.tracer = tracer
        self.agent_name = agent_name

    @contextmanager
    def trace_call(
        self,
        input_msg: Any,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[Any, None, None]:
        """Trace an agent call."""
        with self.tracer.span(
            f"agent.{self.agent_name}.call",
            attributes={
                "agent.name": self.agent_name,
                "agent.input.type": type(input_msg).__name__,
                **(metadata or {}),
            },
        ) as span:
            yield span

    @contextmanager
    def trace_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Generator[Any, None, None]:
        """Trace a tool call."""
        with self.tracer.span(
            f"tool.{tool_name}",
            attributes={
                "tool.name": tool_name,
                "agent.name": self.agent_name,
            },
        ) as span:
            if arguments:
                for key, value in arguments.items():
                    span.set_attribute(f"tool.arg.{key}", str(value)[:100])
            yield span

    @contextmanager
    def trace_llm_call(
        self,
        model: str,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
    ) -> Generator[Any, None, None]:
        """Trace an LLM call."""
        start_time = time.time()

        with self.tracer.span(
            "llm.call",
            attributes={
                "llm.model": model,
                "agent.name": self.agent_name,
            },
            kind="client",
        ) as span:
            yield span

            duration = time.time() - start_time
            span.set_attribute("llm.duration_ms", duration * 1000)

            if tokens_in:
                span.set_attribute("llm.tokens.input", tokens_in)
            if tokens_out:
                span.set_attribute("llm.tokens.output", tokens_out)

    @contextmanager
    def trace_iteration(
        self,
        iteration: int,
        max_iterations: int,
    ) -> Generator[Any, None, None]:
        """Trace a ReAct iteration."""
        with self.tracer.span(
            f"agent.{self.agent_name}.iteration",
            attributes={
                "agent.name": self.agent_name,
                "iteration.current": iteration,
                "iteration.max": max_iterations,
            },
        ) as span:
            yield span


class MetricsCollector:
    """
    Metrics collector using OpenTelemetry metrics.

    Collects agent performance metrics.
    """

    def __init__(self, service_name: str = "clawscope"):
        """Initialize metrics collector."""
        self.service_name = service_name
        self._meters: dict[str, Any] = {}
        self._counters: dict[str, Any] = {}
        self._histograms: dict[str, Any] = {}
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize metrics collection."""
        try:
            from opentelemetry import metrics
            from opentelemetry.sdk.metrics import MeterProvider

            provider = MeterProvider()
            metrics.set_meter_provider(provider)

            self._meter = metrics.get_meter(self.service_name)

            # Create standard metrics
            self._counters["agent_calls"] = self._meter.create_counter(
                "agent_calls_total",
                description="Total agent calls",
            )

            self._counters["tool_calls"] = self._meter.create_counter(
                "tool_calls_total",
                description="Total tool calls",
            )

            self._counters["llm_calls"] = self._meter.create_counter(
                "llm_calls_total",
                description="Total LLM API calls",
            )

            self._counters["tokens_used"] = self._meter.create_counter(
                "tokens_used_total",
                description="Total tokens used",
            )

            self._histograms["agent_latency"] = self._meter.create_histogram(
                "agent_latency_seconds",
                description="Agent call latency",
            )

            self._histograms["llm_latency"] = self._meter.create_histogram(
                "llm_latency_seconds",
                description="LLM call latency",
            )

            self._initialized = True
            return True

        except ImportError:
            logger.warning("OpenTelemetry metrics not available")
            return False

    def record_agent_call(
        self,
        agent_name: str,
        duration: float,
        success: bool = True,
    ) -> None:
        """Record an agent call."""
        if not self._initialized:
            return

        self._counters["agent_calls"].add(
            1,
            {"agent.name": agent_name, "success": str(success)},
        )

        self._histograms["agent_latency"].record(
            duration,
            {"agent.name": agent_name},
        )

    def record_tool_call(
        self,
        tool_name: str,
        agent_name: str,
        success: bool = True,
    ) -> None:
        """Record a tool call."""
        if not self._initialized:
            return

        self._counters["tool_calls"].add(
            1,
            {"tool.name": tool_name, "agent.name": agent_name, "success": str(success)},
        )

    def record_llm_call(
        self,
        model: str,
        duration: float,
        tokens_in: int,
        tokens_out: int,
    ) -> None:
        """Record an LLM call."""
        if not self._initialized:
            return

        self._counters["llm_calls"].add(1, {"model": model})

        self._counters["tokens_used"].add(
            tokens_in + tokens_out,
            {"model": model, "type": "total"},
        )

        self._histograms["llm_latency"].record(duration, {"model": model})


# Global tracer instance
_global_tracer: Tracer | None = None
_global_metrics: MetricsCollector | None = None


def get_tracer() -> Tracer:
    """Get the global tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer()
    return _global_tracer


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics


def configure_tracing(config: TracingConfig) -> Tracer:
    """Configure global tracing."""
    global _global_tracer
    _global_tracer = Tracer(config)
    _global_tracer.initialize()
    return _global_tracer


__all__ = [
    "Tracer",
    "TracingConfig",
    "AgentTracer",
    "MetricsCollector",
    "DummySpan",
    "get_tracer",
    "get_metrics",
    "configure_tracing",
    "OTEL_AVAILABLE",
]
