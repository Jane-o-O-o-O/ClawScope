"""Tests for the configuration system."""

import json
from pathlib import Path

import pytest

from clawscope.config import (
    AgentConfig,
    Config,
    MemoryConfig,
    ModelConfig,
    ServicesConfig,
    ToolsConfig,
    TracingConfig,
)


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def test_model_config_defaults() -> None:
    cfg = ModelConfig()
    assert cfg.provider == "openai"
    assert cfg.default_model == "gpt-4"
    assert cfg.max_retries == 3


def test_agent_config_defaults() -> None:
    cfg = AgentConfig()
    assert cfg.type == "react"
    assert cfg.kernel == "native"
    assert cfg.max_iterations == 40


def test_memory_config_defaults() -> None:
    cfg = MemoryConfig()
    assert cfg.working == "in_memory"
    assert cfg.session == "jsonl"


def test_services_config_defaults() -> None:
    cfg = ServicesConfig()
    assert cfg.cron_enabled is True
    assert cfg.heartbeat_enabled is True
    assert cfg.heartbeat_interval == 1800


def test_tracing_config_defaults() -> None:
    cfg = TracingConfig()
    assert cfg.enabled is False


def test_tools_config_defaults_include_builtins() -> None:
    cfg = ToolsConfig()
    assert "read_file" in cfg.enabled
    assert "execute_shell" in cfg.enabled


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = Config()
    assert cfg.project == "ClawScope"
    assert cfg.model.provider == "openai"
    assert cfg.agent.type == "react"


# ---------------------------------------------------------------------------
# Loading from YAML file
# ---------------------------------------------------------------------------


def test_config_from_yaml_file(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """\
model:
  provider: anthropic
  default_model: claude-sonnet-4-20250514
agent:
  type: react
  max_iterations: 20
""",
        encoding="utf-8",
    )
    cfg = Config.from_file(config_file)
    assert cfg.model.provider == "anthropic"
    assert cfg.model.default_model == "claude-sonnet-4-20250514"
    assert cfg.agent.max_iterations == 20


def test_config_from_json_file(tmp_path: Path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({"model": {"provider": "deepseek", "default_model": "deepseek-chat"}}),
        encoding="utf-8",
    )
    cfg = Config.from_file(config_file)
    assert cfg.model.provider == "deepseek"


def test_config_from_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Config.from_file(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# Environment variable expansion
# ---------------------------------------------------------------------------


def test_config_env_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "sk-test-123")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """\
model:
  provider: openai
  api_key: ${TEST_API_KEY}
""",
        encoding="utf-8",
    )
    cfg = Config.from_file(config_file)
    assert cfg.model.api_key == "sk-test-123"


def test_config_env_expansion_missing_var(tmp_path: Path) -> None:
    """Missing env var should expand to empty string (not raise)."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """\
model:
  api_key: ${TOTALLY_MISSING_VAR_XYZ}
""",
        encoding="utf-8",
    )
    cfg = Config.from_file(config_file)
    assert cfg.model.api_key == ""


def test_config_env_expansion_nested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TG_TOKEN", "bot-abc")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """\
channels:
  telegram:
    enabled: true
    bot_token: ${TG_TOKEN}
""",
        encoding="utf-8",
    )
    cfg = Config.from_file(config_file)
    assert cfg.channels.telegram.bot_token == "bot-abc"
    assert cfg.channels.telegram.enabled is True


# ---------------------------------------------------------------------------
# ensure_workspace
# ---------------------------------------------------------------------------


def test_ensure_workspace_creates_dirs(tmp_path: Path) -> None:
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_workspace()
    assert (cfg.workspace / "sessions").is_dir()
    assert (cfg.workspace / "memory").is_dir()
    assert (cfg.workspace / "skills").is_dir()
    assert (cfg.workspace / "logs").is_dir()


def test_ensure_workspace_creates_bootstrap_files(tmp_path: Path) -> None:
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_workspace()
    for fname in ("AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"):
        assert (cfg.workspace / fname).exists()
