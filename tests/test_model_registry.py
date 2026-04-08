"""Tests for ModelRegistry and ProviderSpec."""

import pytest

from clawscope.config import ModelConfig
from clawscope.model.registry import PROVIDERS, ModelRegistry, ProviderSpec


# ---------------------------------------------------------------------------
# PROVIDERS constant
# ---------------------------------------------------------------------------


def test_providers_list_is_nonempty() -> None:
    assert len(PROVIDERS) >= 10


def test_providers_have_required_fields() -> None:
    for spec in PROVIDERS:
        assert spec.name, f"Provider missing name: {spec!r}"
        assert spec.provider_type in ("direct", "litellm", "agentscope")


def test_openai_provider_in_list() -> None:
    names = [p.name for p in PROVIDERS]
    assert "openai" in names
    assert "anthropic" in names
    assert "deepseek" in names


# ---------------------------------------------------------------------------
# ModelRegistry basics
# ---------------------------------------------------------------------------


def test_list_providers_returns_all() -> None:
    registry = ModelRegistry()
    providers = registry.list_providers()
    assert "openai" in providers
    assert "anthropic" in providers
    assert len(providers) == len(PROVIDERS)


def test_get_provider_by_name() -> None:
    registry = ModelRegistry()
    spec = registry.get_provider("openai")
    assert spec.name == "openai"
    assert spec.provider_type == "direct"


def test_get_unknown_provider_raises() -> None:
    from clawscope.exception import ModelNotFoundError

    registry = ModelRegistry()
    with pytest.raises(ModelNotFoundError):
        registry.get_provider("unicorn")


def test_list_models_for_openai() -> None:
    registry = ModelRegistry()
    models = registry.list_models("openai")
    assert "gpt-4" in models
    assert len(models) >= 3


# ---------------------------------------------------------------------------
# detect_provider
# ---------------------------------------------------------------------------


def test_detect_provider_openai_by_model_keyword() -> None:
    registry = ModelRegistry()
    spec = registry.detect_provider(model="gpt-4-turbo")
    assert spec is not None
    assert spec.name == "openai"


def test_detect_provider_anthropic_by_model_keyword() -> None:
    registry = ModelRegistry()
    spec = registry.detect_provider(model="claude-sonnet-4-20250514")
    assert spec is not None
    assert spec.name == "anthropic"


def test_detect_provider_deepseek_by_model_keyword() -> None:
    registry = ModelRegistry()
    spec = registry.detect_provider(model="deepseek-chat")
    assert spec is not None
    assert spec.name == "deepseek"


def test_detect_provider_groq_by_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # Groq keywords overlap with Ollama ("llama"), so detect via env key instead
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-key")
    registry = ModelRegistry()
    spec = registry.detect_provider()
    assert spec is not None
    assert spec.name == "groq"


def test_detect_provider_openrouter_by_key_prefix() -> None:
    registry = ModelRegistry()
    spec = registry.detect_provider(api_key="sk-or-test-key")
    assert spec is not None
    assert spec.name == "openrouter"


def test_detect_provider_ollama_by_base_url() -> None:
    registry = ModelRegistry()
    spec = registry.detect_provider(api_base="http://localhost:11434/v1")
    assert spec is not None
    assert spec.name == "ollama"


def test_detect_provider_returns_none_for_unknown() -> None:
    registry = ModelRegistry()
    spec = registry.detect_provider(model="totally-unknown-xyz")
    assert spec is None


def test_detect_provider_by_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    registry = ModelRegistry()
    spec = registry.detect_provider()
    # Should detect from environment (anthropic key is set)
    assert spec is not None


# ---------------------------------------------------------------------------
# ProviderSpec capabilities
# ---------------------------------------------------------------------------


def test_anthropic_supports_thinking() -> None:
    registry = ModelRegistry()
    spec = registry.get_provider("anthropic")
    assert spec.supports_thinking is True


def test_anthropic_supports_vision() -> None:
    registry = ModelRegistry()
    spec = registry.get_provider("anthropic")
    assert spec.supports_vision is True


def test_openai_supports_realtime() -> None:
    registry = ModelRegistry()
    spec = registry.get_provider("openai")
    assert spec.supports_realtime is True


def test_groq_does_not_support_thinking() -> None:
    registry = ModelRegistry()
    spec = registry.get_provider("groq")
    assert spec.supports_thinking is False


# ---------------------------------------------------------------------------
# Custom config
# ---------------------------------------------------------------------------


def test_registry_uses_config_defaults() -> None:
    cfg = ModelConfig(provider="anthropic", default_model="claude-sonnet-4-20250514")
    registry = ModelRegistry(config=cfg)
    # detect_provider with no args and no env keys should fall back to None,
    # but the config should still be accessible
    assert registry.config.provider == "anthropic"
    assert registry.config.default_model == "claude-sonnet-4-20250514"


def test_register_custom_provider() -> None:
    registry = ModelRegistry()
    custom = ProviderSpec(
        name="custom_llm",
        display_name="Custom LLM",
        provider_type="litellm",
        litellm_prefix="custom",
        keywords=("customllm",),
    )
    registry.register_provider(custom)
    assert "custom_llm" in registry.list_providers()
    spec = registry.detect_provider(model="customllm-v1")
    assert spec is not None
    assert spec.name == "custom_llm"
