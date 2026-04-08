"""
MCPSkill – a ClawScope Skill backed by an MCP server tool.

Allows any MCP server tool to be used as a first-class ClawScope Skill,
complete with trigger matching, priority, and the full hook system.

Two helper factories are provided:

``MCPSkill.from_client(client, tool_name, ...)``
    Wraps a single tool from an already-connected MCPClient.

``MCPSkillBundle.from_client(client, ...)``
    Discovers ALL tools on the server and returns a bundle of MCPSkills
    that can be bulk-registered into a SkillRegistry.

Example::

    from clawscope.mcp import MCPClient, StdioServerConfig, MCPSkillBundle

    client = MCPClient(StdioServerConfig(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
        name="filesystem",
    ))
    await client.connect()

    bundle = await MCPSkillBundle.from_client(client)
    for skill in bundle.skills:
        skill_registry.register(skill)

    # Or use the convenience method directly:
    await bundle.register_all(skill_registry)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from loguru import logger

from clawscope.skills.base import Skill, SkillCategory, SkillConfig, SkillContext
from clawscope.message import Msg

if TYPE_CHECKING:
    from clawscope.mcp.client import MCPClient, MCPToolInfo
    from clawscope.skills import SkillRegistry


class MCPSkill(Skill):
    """
    A ClawScope Skill that delegates execution to an MCP server tool.

    The skill extracts plain-text content from the incoming message and
    passes it (along with any extra kwargs) to the MCP tool as an
    ``input`` argument.  A more structured argument mapping can be
    provided via the *argument_mapper* callable.

    Args:
        client: Connected MCPClient instance.
        tool_name: Name of the tool on the MCP server.
        config: SkillConfig (name, description, triggers …).
        argument_mapper: Optional callable ``(SkillContext) → dict``
            that maps a SkillContext to the argument dict for the tool.
            Defaults to ``{"input": message_text}``.
    """

    def __init__(
        self,
        client: "MCPClient",
        tool_name: str,
        config: SkillConfig,
        argument_mapper: Any | None = None,
    ):
        super().__init__(config)
        self._client = client
        self._tool_name = tool_name
        self._argument_mapper = argument_mapper

    # ------------------------------------------------------------------
    # Skill protocol
    # ------------------------------------------------------------------

    async def execute(self, context: SkillContext) -> Msg | None:
        """Call the MCP tool and return the result as a Msg."""
        if not self._client.is_connected:
            logger.warning(
                f"MCPSkill '{self.config.name}': client not connected, skipping"
            )
            return None

        arguments = self._build_arguments(context)

        try:
            result = await self._client.call_tool(self._tool_name, arguments)
            return Msg(
                name=self.config.name,
                content=result,
                role="assistant",
            )
        except Exception as exc:
            logger.error(
                f"MCPSkill '{self.config.name}' → tool '{self._tool_name}' "
                f"error: {exc}"
            )
            return Msg(
                name=self.config.name,
                content=f"Error calling {self._tool_name}: {exc}",
                role="assistant",
                metadata={"is_error": True},
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_arguments(self, context: SkillContext) -> dict[str, Any]:
        if self._argument_mapper is not None:
            return self._argument_mapper(context)
        # Default: pass the message text as "input"
        text = (
            context.message.get_text_content()
            if context.message
            else ""
        )
        return {"input": text}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_client(
        cls,
        client: "MCPClient",
        tool_name: str,
        *,
        skill_name: str | None = None,
        description: str = "",
        triggers: list[str] | None = None,
        category: SkillCategory = SkillCategory.INTEGRATION,
        priority: int = 50,
        argument_mapper: Any | None = None,
    ) -> "MCPSkill":
        """
        Convenience factory.

        Args:
            client: Connected MCPClient.
            tool_name: Tool name on the MCP server.
            skill_name: ClawScope skill name (defaults to
                ``"{client.name}_{tool_name}"``).
            description: Human-readable description.
            triggers: List of trigger strings.
            category: SkillCategory.
            priority: Dispatch priority (higher = earlier).
            argument_mapper: Optional context → arguments callable.
        """
        name = skill_name or f"{client.name}_{tool_name}"
        config = SkillConfig(
            name=name,
            description=description or f"MCP tool: {tool_name}",
            category=category,
            triggers=triggers or [],
            priority=priority,
            tags=[f"mcp:{client.name}", f"tool:{tool_name}"],
        )
        return cls(client, tool_name, config, argument_mapper)

    @classmethod
    async def from_tool_info(
        cls,
        client: "MCPClient",
        tool_info: "MCPToolInfo",
        **kwargs: Any,
    ) -> "MCPSkill":
        """Create from an MCPToolInfo descriptor (returned by list_tools)."""
        return cls.from_client(
            client,
            tool_info.name,
            description=tool_info.description,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# MCPSkillBundle
# ---------------------------------------------------------------------------


@dataclass
class MCPSkillBundle:
    """
    A collection of MCPSkills created from a single MCP server.

    Attributes:
        client: The MCPClient used to create the skills.
        skills: List of MCPSkill instances, one per server tool.
    """

    client: "MCPClient"
    skills: list[MCPSkill] = field(default_factory=list)

    @classmethod
    async def from_client(
        cls,
        client: "MCPClient",
        *,
        prefix: str | None = None,
        category: SkillCategory = SkillCategory.INTEGRATION,
        base_priority: int = 50,
        argument_mapper: Any | None = None,
    ) -> "MCPSkillBundle":
        """
        Discover all tools on the server and wrap them as MCPSkills.

        Args:
            client: Connected MCPClient.
            prefix: Skill name prefix (default: ``"{client.name}_"``).
            category: Category assigned to all created skills.
            base_priority: Priority for all created skills.
            argument_mapper: Shared argument mapper for all skills.

        Returns:
            MCPSkillBundle containing one skill per server tool.
        """
        _prefix = prefix if prefix is not None else f"{client.name}_"
        tool_infos = await client.list_tools()
        skills = []

        for tool_info in tool_infos:
            skill_name = f"{_prefix}{tool_info.name}"
            skill = MCPSkill.from_client(
                client,
                tool_info.name,
                skill_name=skill_name,
                description=tool_info.description,
                category=category,
                priority=base_priority,
                argument_mapper=argument_mapper,
            )
            skills.append(skill)
            logger.debug(f"MCPSkillBundle: created skill '{skill_name}'")

        logger.info(
            f"MCPSkillBundle from '{client.name}': {len(skills)} skills created"
        )
        return cls(client=client, skills=skills)

    async def register_all(self, skill_registry: "SkillRegistry") -> int:
        """
        Register all skills in this bundle into *skill_registry*.

        Returns:
            Number of skills registered.
        """
        for skill in self.skills:
            skill_registry.register(skill)
        logger.info(
            f"MCPSkillBundle: registered {len(self.skills)} skills "
            f"from '{self.client.name}'"
        )
        return len(self.skills)

    def __len__(self) -> int:
        return len(self.skills)

    def __iter__(self):
        return iter(self.skills)


__all__ = ["MCPSkill", "MCPSkillBundle"]
