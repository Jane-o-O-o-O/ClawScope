"""Skill registry for ClawScope."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from clawscope.skills.base import Skill, SkillConfig, SkillContext, SkillCategory

if TYPE_CHECKING:
    from clawscope.message import Msg


class SkillRegistry:
    """
    Registry for managing skills.

    Features:
    - Skill registration and discovery
    - Trigger-based matching
    - Priority-based execution
    """

    def __init__(self):
        """Initialize skill registry."""
        self._skills: dict[str, Skill] = {}
        self._categories: dict[SkillCategory, list[str]] = {}
        self._triggers: dict[str, list[str]] = {}  # trigger -> skill names

    def register(self, skill: Skill) -> None:
        """
        Register a skill.

        Args:
            skill: Skill to register
        """
        name = skill.config.name

        if name in self._skills:
            logger.warning(f"Skill '{name}' already registered, replacing")

        self._skills[name] = skill

        # Index by category
        category = skill.config.category
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)

        # Index by triggers
        for trigger in skill.config.triggers:
            trigger_lower = trigger.lower()
            if trigger_lower not in self._triggers:
                self._triggers[trigger_lower] = []
            if name not in self._triggers[trigger_lower]:
                self._triggers[trigger_lower].append(name)

        logger.debug(f"Registered skill: {name}")

    def unregister(self, name: str) -> bool:
        """
        Unregister a skill.

        Args:
            name: Skill name

        Returns:
            True if unregistered
        """
        if name not in self._skills:
            return False

        skill = self._skills.pop(name)

        # Remove from category index
        category = skill.config.category
        if category in self._categories:
            self._categories[category] = [
                n for n in self._categories[category] if n != name
            ]

        # Remove from trigger index
        for trigger in skill.config.triggers:
            trigger_lower = trigger.lower()
            if trigger_lower in self._triggers:
                self._triggers[trigger_lower] = [
                    n for n in self._triggers[trigger_lower] if n != name
                ]

        logger.debug(f"Unregistered skill: {name}")
        return True

    def get(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def list_skills(
        self,
        category: SkillCategory | None = None,
        enabled_only: bool = False,
    ) -> list[Skill]:
        """
        List registered skills.

        Args:
            category: Filter by category
            enabled_only: Only return enabled skills

        Returns:
            List of skills
        """
        skills = []

        if category:
            names = self._categories.get(category, [])
            skills = [self._skills[n] for n in names if n in self._skills]
        else:
            skills = list(self._skills.values())

        if enabled_only:
            skills = [s for s in skills if s.enabled]

        # Sort by priority (higher first)
        skills.sort(key=lambda s: s.config.priority, reverse=True)

        return skills

    def find_matching(self, message: "Msg") -> list[Skill]:
        """
        Find skills that match a message.

        Args:
            message: Input message

        Returns:
            List of matching skills
        """
        content = message.get_text_content().lower()
        matching = []

        # Check trigger-based matching first
        for trigger, names in self._triggers.items():
            if trigger in content:
                for name in names:
                    skill = self._skills.get(name)
                    if skill and skill.enabled and skill not in matching:
                        matching.append(skill)

        # Check general matching
        for skill in self._skills.values():
            if skill not in matching and skill.matches(message):
                matching.append(skill)

        # Sort by priority
        matching.sort(key=lambda s: s.config.priority, reverse=True)

        return matching

    async def execute(
        self,
        message: "Msg",
        context_extras: dict[str, Any] | None = None,
    ) -> "Msg | None":
        """
        Find and execute matching skill.

        Args:
            message: Input message
            context_extras: Additional context data

        Returns:
            Response message or None
        """
        matching = self.find_matching(message)

        if not matching:
            return None

        # Execute highest priority skill
        skill = matching[0]
        logger.info(f"Executing skill: {skill.name}")

        context = SkillContext(
            message=message,
            metadata=context_extras or {},
        )

        try:
            return await skill.execute(context)
        except Exception as e:
            logger.error(f"Skill '{skill.name}' error: {e}")
            return None

    async def execute_all(
        self,
        message: "Msg",
        context_extras: dict[str, Any] | None = None,
    ) -> list["Msg"]:
        """
        Execute all matching skills.

        Args:
            message: Input message
            context_extras: Additional context data

        Returns:
            List of response messages
        """
        matching = self.find_matching(message)
        responses = []

        for skill in matching:
            context = SkillContext(
                message=message,
                metadata=context_extras or {},
            )

            try:
                response = await skill.execute(context)
                if response:
                    responses.append(response)
            except Exception as e:
                logger.error(f"Skill '{skill.name}' error: {e}")

        return responses

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        return {
            "total_skills": len(self._skills),
            "enabled_skills": len([s for s in self._skills.values() if s.enabled]),
            "categories": {
                cat.value: len(names)
                for cat, names in self._categories.items()
            },
            "triggers": len(self._triggers),
        }


__all__ = ["SkillRegistry"]
