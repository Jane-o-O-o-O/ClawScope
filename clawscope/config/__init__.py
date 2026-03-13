"""ClawScope configuration system."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """Model provider configuration."""

    provider: str = "openai"
    api_key: str | None = None
    api_base: str | None = None
    default_model: str = "gpt-4"
    stream: bool = True
    timeout: int = 120
    max_retries: int = 3


class AgentConfig(BaseModel):
    """Agent configuration."""

    type: Literal["react", "user", "realtime", "a2a"] = "react"
    name: str = "ClawScope"
    sys_prompt: str = "You are a helpful AI assistant."
    max_iterations: int = 40
    max_tokens: int = 4096


class MemoryConfig(BaseModel):
    """Memory system configuration."""

    working: Literal["in_memory", "redis", "sqlalchemy"] = "in_memory"
    session: Literal["jsonl", "redis", "sqlite"] = "jsonl"
    long_term: Literal["memory_md", "mem0", "reme"] | None = "memory_md"
    redis_url: str | None = None
    database_url: str | None = None


class ChannelBaseConfig(BaseModel):
    """Base channel configuration."""

    enabled: bool = False
    allow_from: list[str] = Field(default_factory=lambda: ["*"])


class TelegramConfig(ChannelBaseConfig):
    """Telegram channel configuration."""

    bot_token: str | None = None
    proxy: str | None = None


class DiscordConfig(ChannelBaseConfig):
    """Discord channel configuration."""

    bot_token: str | None = None
    guild_ids: list[int] = Field(default_factory=list)


class SlackConfig(ChannelBaseConfig):
    """Slack channel configuration."""

    bot_token: str | None = None
    app_token: str | None = None


class FeishuConfig(ChannelBaseConfig):
    """Feishu (Lark) channel configuration."""

    app_id: str | None = None
    app_secret: str | None = None


class DingTalkConfig(ChannelBaseConfig):
    """DingTalk channel configuration."""

    app_key: str | None = None
    app_secret: str | None = None


class ChannelsConfig(BaseModel):
    """All channels configuration."""

    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    slack: SlackConfig = Field(default_factory=SlackConfig)
    feishu: FeishuConfig = Field(default_factory=FeishuConfig)
    dingtalk: DingTalkConfig = Field(default_factory=DingTalkConfig)


class ToolsConfig(BaseModel):
    """Tools configuration."""

    enabled: list[str] = Field(
        default_factory=lambda: [
            "read_file",
            "write_file",
            "execute_shell",
            "web_search",
            "web_fetch",
        ]
    )
    shell_timeout: int = 60
    max_output_length: int = 16000
    workspace_path: str | None = None


class ServicesConfig(BaseModel):
    """Background services configuration."""

    cron_enabled: bool = True
    heartbeat_enabled: bool = True
    heartbeat_interval: int = 1800  # 30 minutes


class TracingConfig(BaseModel):
    """Tracing and observability configuration."""

    enabled: bool = False
    endpoint: str | None = None
    service_name: str = "clawscope"
    log_level: str = "INFO"
    log_file: str | None = None


class Config(BaseSettings):
    """Main ClawScope configuration."""

    model_config = SettingsConfigDict(
        env_prefix="CLAWSCOPE_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Core settings
    project: str = "ClawScope"
    workspace: Path = Field(default_factory=lambda: Path.home() / ".clawscope" / "workspace")

    # Component configs
    model: ModelConfig = Field(default_factory=ModelConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    services: ServicesConfig = Field(default_factory=ServicesConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        """Load configuration from YAML or JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            if path.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            else:
                import json
                data = json.load(f)

        # Expand environment variables
        data = cls._expand_env_vars(data)
        return cls(**data)

    @classmethod
    def _expand_env_vars(cls, data: Any) -> Any:
        """Recursively expand environment variables in config values."""
        if isinstance(data, str):
            if data.startswith("${") and data.endswith("}"):
                env_var = data[2:-1]
                return os.environ.get(env_var, "")
            return data
        elif isinstance(data, dict):
            return {k: cls._expand_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls._expand_env_vars(item) for item in data]
        return data

    def ensure_workspace(self) -> None:
        """Ensure workspace directory exists."""
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.workspace / "sessions").mkdir(exist_ok=True)
        (self.workspace / "memory").mkdir(exist_ok=True)
        (self.workspace / "skills").mkdir(exist_ok=True)
        (self.workspace / "logs").mkdir(exist_ok=True)


__all__ = [
    "Config",
    "ModelConfig",
    "AgentConfig",
    "MemoryConfig",
    "ChannelsConfig",
    "ToolsConfig",
    "ServicesConfig",
    "TracingConfig",
]
