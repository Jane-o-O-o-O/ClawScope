"""Skill base classes for ClawScope."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Awaitable

if TYPE_CHECKING:
    from clawscope.agent import AgentBase
    from clawscope.message import Msg


class SkillCategory(str, Enum):
    """Skill categories."""

    UTILITY = "utility"
    PRODUCTIVITY = "productivity"
    COMMUNICATION = "communication"
    DEVELOPMENT = "development"
    DATA = "data"
    INTEGRATION = "integration"
    CREATIVE = "creative"
    OTHER = "other"


@dataclass
class SkillConfig:
    """Skill configuration."""

    # Identity
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    category: SkillCategory = SkillCategory.OTHER

    # Requirements
    requires_model: bool = False
    requires_tools: list[str] = field(default_factory=list)
    requires_permissions: list[str] = field(default_factory=list)

    # Triggers
    triggers: list[str] = field(default_factory=list)
    priority: int = 5  # 1-10

    # Metadata
    tags: list[str] = field(default_factory=list)
    homepage: str | None = None
    repository: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "category": self.category.value,
            "requires_model": self.requires_model,
            "requires_tools": self.requires_tools,
            "requires_permissions": self.requires_permissions,
            "triggers": self.triggers,
            "priority": self.priority,
            "tags": self.tags,
            "homepage": self.homepage,
            "repository": self.repository,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillConfig":
        """Create from dictionary."""
        category = data.get("category", "other")
        if isinstance(category, str):
            category = SkillCategory(category)

        return cls(
            name=data.get("name", ""),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            category=category,
            requires_model=data.get("requires_model", False),
            requires_tools=data.get("requires_tools", []),
            requires_permissions=data.get("requires_permissions", []),
            triggers=data.get("triggers", []),
            priority=data.get("priority", 5),
            tags=data.get("tags", []),
            homepage=data.get("homepage"),
            repository=data.get("repository"),
        )


@dataclass
class SkillContext:
    """Context passed to skill execution."""

    message: "Msg"
    agent: "AgentBase | None" = None
    session_id: str | None = None
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Skill(ABC):
    """
    Abstract base class for skills.

    Skills are reusable capabilities that can be added to agents.
    """

    def __init__(self, config: SkillConfig | None = None):
        """
        Initialize skill.

        Args:
            config: Skill configuration
        """
        self.config = config or SkillConfig(name=self.__class__.__name__)
        self._enabled = True
        self._hooks: dict[str, list[Callable]] = {}

    @property
    def name(self) -> str:
        """Get skill name."""
        return self.config.name

    @property
    def enabled(self) -> bool:
        """Check if skill is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable the skill."""
        self._enabled = True

    def disable(self) -> None:
        """Disable the skill."""
        self._enabled = False

    def matches(self, message: "Msg") -> bool:
        """
        Check if skill should handle this message.

        Args:
            message: Input message

        Returns:
            True if skill should handle
        """
        if not self._enabled:
            return False

        content = message.get_text_content().lower()

        # Check triggers
        for trigger in self.config.triggers:
            if trigger.lower() in content:
                return True

        return False

    @abstractmethod
    async def execute(self, context: SkillContext) -> "Msg":
        """
        Execute the skill.

        Args:
            context: Execution context

        Returns:
            Response message
        """
        pass

    async def on_load(self) -> None:
        """Called when skill is loaded."""
        pass

    async def on_unload(self) -> None:
        """Called when skill is unloaded."""
        pass

    def add_hook(self, event: str, callback: Callable) -> None:
        """Add an event hook."""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)

    async def _trigger_hook(self, event: str, *args, **kwargs) -> None:
        """Trigger hooks for an event."""
        for callback in self._hooks.get(event, []):
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)


class FunctionSkill(Skill):
    """
    Skill that wraps a simple function.

    For quick skill creation without subclassing.
    """

    def __init__(
        self,
        func: Callable[[SkillContext], Awaitable["Msg"] | "Msg"],
        config: SkillConfig | None = None,
    ):
        """
        Initialize function skill.

        Args:
            func: Function to execute
            config: Skill configuration
        """
        super().__init__(config)
        self._func = func

    async def execute(self, context: SkillContext) -> "Msg":
        """Execute the wrapped function."""
        import asyncio

        if asyncio.iscoroutinefunction(self._func):
            return await self._func(context)
        else:
            return self._func(context)


def skill(
    name: str | None = None,
    triggers: list[str] | None = None,
    **kwargs: Any,
) -> Callable:
    """
    Decorator to create a skill from a function.

    Args:
        name: Skill name
        triggers: Trigger phrases
        **kwargs: Additional config options

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> FunctionSkill:
        config = SkillConfig(
            name=name or func.__name__,
            triggers=triggers or [],
            description=func.__doc__ or "",
            **kwargs,
        )
        return FunctionSkill(func, config)

    return decorator


# Import asyncio for iscoroutinefunction check
import asyncio

__all__ = [
    "Skill",
    "SkillConfig",
    "SkillContext",
    "SkillCategory",
    "FunctionSkill",
    "skill",
]
