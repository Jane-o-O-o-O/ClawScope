"""Model provider registry for ClawScope."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Type
import os

from loguru import logger

from clawscope.model.base import ChatModelBase
from clawscope.config import ModelConfig
from clawscope.exception import ModelNotFoundError


@dataclass
class ProviderSpec:
    """Specification for a model provider."""

    name: str
    display_name: str
    provider_type: Literal["agentscope", "litellm", "direct"]

    # Model class (for agentscope/direct types)
    model_class: Type[ChatModelBase] | None = None

    # LiteLLM configuration
    litellm_prefix: str = ""

    # Environment and detection
    env_key: str = ""
    keywords: tuple[str, ...] = ()
    detect_by_key_prefix: str = ""
    detect_by_base_keyword: str = ""

    # Capabilities
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_vision: bool = False
    supports_realtime: bool = False
    supports_thinking: bool = False

    # Default models
    default_model: str = ""
    models: list[str] = field(default_factory=list)


# Provider specifications registry
PROVIDERS: list[ProviderSpec] = [
    # OpenAI
    ProviderSpec(
        name="openai",
        display_name="OpenAI",
        provider_type="direct",
        env_key="OPENAI_API_KEY",
        keywords=("gpt", "openai", "o1", "o3"),
        supports_vision=True,
        supports_realtime=True,
        default_model="gpt-4",
        models=["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo", "o1", "o1-mini", "o3-mini"],
    ),

    # Anthropic
    ProviderSpec(
        name="anthropic",
        display_name="Anthropic/Claude",
        provider_type="direct",
        env_key="ANTHROPIC_API_KEY",
        keywords=("claude", "anthropic"),
        supports_vision=True,
        supports_thinking=True,
        default_model="claude-sonnet-4-20250514",
        models=["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
    ),

    # DashScope (Alibaba Qwen)
    ProviderSpec(
        name="dashscope",
        display_name="DashScope/Qwen",
        provider_type="agentscope",
        env_key="DASHSCOPE_API_KEY",
        keywords=("qwen", "dashscope", "tongyi"),
        supports_vision=True,
        supports_realtime=True,
        default_model="qwen-max",
        models=["qwen-max", "qwen-plus", "qwen-turbo", "qwen-vl-max"],
    ),

    # Google Gemini
    ProviderSpec(
        name="gemini",
        display_name="Google Gemini",
        provider_type="agentscope",
        env_key="GOOGLE_API_KEY",
        keywords=("gemini", "google"),
        supports_vision=True,
        supports_realtime=True,
        default_model="gemini-pro",
        models=["gemini-pro", "gemini-pro-vision", "gemini-ultra"],
    ),

    # Ollama (Local)
    ProviderSpec(
        name="ollama",
        display_name="Ollama",
        provider_type="agentscope",
        detect_by_base_keyword="localhost:11434",
        keywords=("ollama", "llama", "mistral", "codellama"),
        default_model="llama3",
        models=["llama3", "llama2", "mistral", "codellama", "phi"],
    ),

    # OpenRouter (Gateway)
    ProviderSpec(
        name="openrouter",
        display_name="OpenRouter",
        provider_type="litellm",
        litellm_prefix="openrouter",
        env_key="OPENROUTER_API_KEY",
        keywords=("openrouter",),
        detect_by_key_prefix="sk-or-",
        supports_vision=True,
        default_model="openai/gpt-4",
    ),

    # DeepSeek
    ProviderSpec(
        name="deepseek",
        display_name="DeepSeek",
        provider_type="litellm",
        litellm_prefix="deepseek",
        env_key="DEEPSEEK_API_KEY",
        keywords=("deepseek",),
        supports_thinking=True,
        default_model="deepseek-chat",
        models=["deepseek-chat", "deepseek-coder", "deepseek-reasoner"],
    ),

    # Groq
    ProviderSpec(
        name="groq",
        display_name="Groq",
        provider_type="litellm",
        litellm_prefix="groq",
        env_key="GROQ_API_KEY",
        keywords=("groq",),
        default_model="llama-3.1-70b-versatile",
        models=["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    ),

    # SiliconFlow
    ProviderSpec(
        name="siliconflow",
        display_name="SiliconFlow",
        provider_type="litellm",
        litellm_prefix="siliconflow",
        env_key="SILICONFLOW_API_KEY",
        keywords=("siliconflow", "silicon"),
        default_model="Qwen/Qwen2.5-72B-Instruct",
    ),

    # Moonshot/Kimi
    ProviderSpec(
        name="moonshot",
        display_name="Moonshot/Kimi",
        provider_type="litellm",
        litellm_prefix="moonshot",
        env_key="MOONSHOT_API_KEY",
        keywords=("moonshot", "kimi"),
        supports_thinking=True,
        default_model="moonshot-v1-8k",
        models=["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
    ),

    # Zhipu GLM
    ProviderSpec(
        name="zhipu",
        display_name="Zhipu/GLM",
        provider_type="litellm",
        litellm_prefix="zhipu",
        env_key="ZHIPU_API_KEY",
        keywords=("zhipu", "glm", "chatglm"),
        default_model="glm-4",
        models=["glm-4", "glm-4-air", "glm-4-flash"],
    ),
]


class ModelRegistry:
    """
    Registry for model providers.

    Manages provider detection, initialization, and model creation.
    """

    def __init__(self, config: ModelConfig | None = None):
        """
        Initialize model registry.

        Args:
            config: Optional model configuration
        """
        self.config = config or ModelConfig()
        self._providers: dict[str, ProviderSpec] = {p.name: p for p in PROVIDERS}
        self._instances: dict[str, ChatModelBase] = {}

    def register_provider(self, spec: ProviderSpec) -> None:
        """Register a new provider specification."""
        self._providers[spec.name] = spec
        logger.debug(f"Registered provider: {spec.name}")

    def get_provider(self, name: str) -> ProviderSpec:
        """Get provider specification by name."""
        if name not in self._providers:
            raise ModelNotFoundError(f"Provider not found: {name}")
        return self._providers[name]

    def detect_provider(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
    ) -> ProviderSpec | None:
        """
        Auto-detect provider from configuration.

        Args:
            api_key: API key (may contain provider hint)
            api_base: API base URL
            model: Model name

        Returns:
            Detected ProviderSpec or None
        """
        # Check key prefix
        if api_key:
            for provider in self._providers.values():
                if provider.detect_by_key_prefix and api_key.startswith(provider.detect_by_key_prefix):
                    return provider

        # Check base URL
        if api_base:
            for provider in self._providers.values():
                if provider.detect_by_base_keyword and provider.detect_by_base_keyword in api_base:
                    return provider

        # Check model name keywords
        if model:
            model_lower = model.lower()
            for provider in self._providers.values():
                for keyword in provider.keywords:
                    if keyword in model_lower:
                        return provider

        # Check environment variables
        for provider in self._providers.values():
            if provider.env_key and os.environ.get(provider.env_key):
                return provider

        return None

    def get_model(
        self,
        provider: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> ChatModelBase:
        """
        Get or create a model instance.

        Args:
            provider: Provider name (auto-detected if not specified)
            model: Model name
            **kwargs: Additional model options

        Returns:
            ChatModelBase instance
        """
        # Use config values as defaults
        api_key = kwargs.pop("api_key", self.config.api_key)
        api_base = kwargs.pop("api_base", self.config.api_base)
        model = model or self.config.default_model

        # Detect provider if not specified
        if not provider:
            spec = self.detect_provider(api_key, api_base, model)
            if spec:
                provider = spec.name
            else:
                provider = "openai"  # Default fallback

        spec = self.get_provider(provider)

        # Create cache key
        cache_key = f"{provider}:{model}"
        if cache_key in self._instances:
            return self._instances[cache_key]

        # Create model instance
        instance = self._create_model(spec, model, api_key, api_base, **kwargs)
        self._instances[cache_key] = instance
        return instance

    def _create_model(
        self,
        spec: ProviderSpec,
        model: str,
        api_key: str | None,
        api_base: str | None,
        **kwargs: Any,
    ) -> ChatModelBase:
        """Create model instance based on provider type."""
        # Get API key from environment if not provided
        if not api_key and spec.env_key:
            api_key = os.environ.get(spec.env_key)

        if spec.provider_type == "direct":
            # Use direct implementation
            return self._create_direct_model(spec, model, api_key, api_base, **kwargs)
        elif spec.provider_type == "litellm":
            # Use LiteLLM adapter
            return self._create_litellm_model(spec, model, api_key, api_base, **kwargs)
        elif spec.provider_type == "agentscope":
            # Use AgentScope adapter
            return self._create_agentscope_model(spec, model, api_key, api_base, **kwargs)
        else:
            raise ModelNotFoundError(f"Unknown provider type: {spec.provider_type}")

    def _create_direct_model(
        self,
        spec: ProviderSpec,
        model: str,
        api_key: str | None,
        api_base: str | None,
        **kwargs: Any,
    ) -> ChatModelBase:
        """Create direct API model."""
        if spec.name == "openai":
            from clawscope.model.providers.openai import OpenAIChatModel
            return OpenAIChatModel(
                model_name=model,
                api_key=api_key,
                api_base=api_base,
                stream=self.config.stream,
                timeout=self.config.timeout,
                **kwargs,
            )
        elif spec.name == "anthropic":
            from clawscope.model.providers.anthropic import AnthropicChatModel
            return AnthropicChatModel(
                model_name=model,
                api_key=api_key,
                api_base=api_base,
                stream=self.config.stream,
                timeout=self.config.timeout,
                **kwargs,
            )
        else:
            raise ModelNotFoundError(f"No direct implementation for: {spec.name}")

    def _create_litellm_model(
        self,
        spec: ProviderSpec,
        model: str,
        api_key: str | None,
        api_base: str | None,
        **kwargs: Any,
    ) -> ChatModelBase:
        """Create LiteLLM-based model."""
        from clawscope.model.providers.litellm_adapter import LiteLLMChatModel

        # Add provider prefix if not present
        if spec.litellm_prefix and not model.startswith(spec.litellm_prefix):
            model = f"{spec.litellm_prefix}/{model}"

        return LiteLLMChatModel(
            model_name=model,
            api_key=api_key,
            api_base=api_base,
            stream=self.config.stream,
            timeout=self.config.timeout,
            **kwargs,
        )

    def _create_agentscope_model(
        self,
        spec: ProviderSpec,
        model: str,
        api_key: str | None,
        api_base: str | None,
        **kwargs: Any,
    ) -> ChatModelBase:
        """Create AgentScope-style model (placeholder)."""
        # For now, fall back to LiteLLM
        return self._create_litellm_model(spec, model, api_key, api_base, **kwargs)

    def list_providers(self) -> list[str]:
        """List all registered providers."""
        return list(self._providers.keys())

    def list_models(self, provider: str) -> list[str]:
        """List models for a provider."""
        spec = self.get_provider(provider)
        return spec.models


__all__ = [
    "ModelRegistry",
    "ProviderSpec",
    "PROVIDERS",
]
