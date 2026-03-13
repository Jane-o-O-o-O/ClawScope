"""Skill loader for ClawScope."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from loguru import logger

from clawscope.skills.base import Skill, SkillConfig, FunctionSkill
from clawscope.skills.registry import SkillRegistry

if TYPE_CHECKING:
    pass


class SkillLoader:
    """
    Loader for skills from various sources.

    Supports:
    - Python modules
    - YAML/JSON definitions
    - Directory scanning
    - Remote URLs
    """

    def __init__(self, registry: SkillRegistry | None = None):
        """
        Initialize skill loader.

        Args:
            registry: Skill registry to load into
        """
        self.registry = registry or SkillRegistry()
        self._loaded_paths: set[Path] = set()

    async def load_module(self, module_path: str) -> list[Skill]:
        """
        Load skills from a Python module.

        Args:
            module_path: Module path (e.g., 'clawscope.skills.builtin')

        Returns:
            List of loaded skills
        """
        try:
            module = importlib.import_module(module_path)
            skills = []

            # Find Skill subclasses
            for name in dir(module):
                obj = getattr(module, name)

                if isinstance(obj, type) and issubclass(obj, Skill) and obj is not Skill:
                    skill = obj()
                    await skill.on_load()
                    self.registry.register(skill)
                    skills.append(skill)
                    logger.debug(f"Loaded skill from module: {skill.name}")

                elif isinstance(obj, Skill):
                    await obj.on_load()
                    self.registry.register(obj)
                    skills.append(obj)
                    logger.debug(f"Loaded skill instance: {obj.name}")

            return skills

        except ImportError as e:
            logger.error(f"Failed to import module '{module_path}': {e}")
            return []

    async def load_file(self, path: str | Path) -> list[Skill]:
        """
        Load skills from a file.

        Supports:
        - .py files (Python modules)
        - .yaml/.yml files (YAML definitions)
        - .json files (JSON definitions)

        Args:
            path: File path

        Returns:
            List of loaded skills
        """
        path = Path(path)

        if not path.exists():
            logger.error(f"Skill file not found: {path}")
            return []

        if path in self._loaded_paths:
            logger.debug(f"Skill file already loaded: {path}")
            return []

        self._loaded_paths.add(path)

        if path.suffix == ".py":
            return await self._load_python_file(path)
        elif path.suffix in (".yaml", ".yml"):
            return await self._load_yaml_file(path)
        elif path.suffix == ".json":
            return await self._load_json_file(path)
        else:
            logger.warning(f"Unsupported skill file format: {path.suffix}")
            return []

    async def load_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
    ) -> list[Skill]:
        """
        Load all skills from a directory.

        Args:
            directory: Directory path
            recursive: Search recursively

        Returns:
            List of loaded skills
        """
        directory = Path(directory)

        if not directory.is_dir():
            logger.error(f"Not a directory: {directory}")
            return []

        skills = []
        patterns = ["*.py", "*.yaml", "*.yml", "*.json"]

        for pattern in patterns:
            if recursive:
                files = list(directory.rglob(pattern))
            else:
                files = list(directory.glob(pattern))

            for file_path in files:
                # Skip __pycache__ and hidden files
                if "__pycache__" in str(file_path) or file_path.name.startswith("."):
                    continue

                loaded = await self.load_file(file_path)
                skills.extend(loaded)

        logger.info(f"Loaded {len(skills)} skills from {directory}")
        return skills

    async def _load_python_file(self, path: Path) -> list[Skill]:
        """Load skills from Python file."""
        try:
            # Load module from file
            spec = importlib.util.spec_from_file_location(
                f"skill_{path.stem}",
                path,
            )
            if spec is None or spec.loader is None:
                return []

            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            skills = []

            # Find skills in module
            for name in dir(module):
                obj = getattr(module, name)

                if isinstance(obj, type) and issubclass(obj, Skill) and obj is not Skill:
                    skill = obj()
                    await skill.on_load()
                    self.registry.register(skill)
                    skills.append(skill)

                elif isinstance(obj, Skill):
                    await obj.on_load()
                    self.registry.register(obj)
                    skills.append(obj)

                elif isinstance(obj, FunctionSkill):
                    await obj.on_load()
                    self.registry.register(obj)
                    skills.append(obj)

            return skills

        except Exception as e:
            logger.error(f"Failed to load Python skill file {path}: {e}")
            return []

    async def _load_yaml_file(self, path: Path) -> list[Skill]:
        """Load skills from YAML file."""
        try:
            content = path.read_text(encoding="utf-8")
            data = yaml.safe_load(content)

            return await self._load_definition(data, path)

        except Exception as e:
            logger.error(f"Failed to load YAML skill file {path}: {e}")
            return []

    async def _load_json_file(self, path: Path) -> list[Skill]:
        """Load skills from JSON file."""
        try:
            content = path.read_text(encoding="utf-8")
            data = json.loads(content)

            return await self._load_definition(data, path)

        except Exception as e:
            logger.error(f"Failed to load JSON skill file {path}: {e}")
            return []

    async def _load_definition(
        self,
        data: dict[str, Any] | list[dict[str, Any]],
        source: Path,
    ) -> list[Skill]:
        """Load skills from definition data."""
        skills = []

        # Handle single skill or list
        if isinstance(data, dict):
            definitions = [data]
        else:
            definitions = data

        for definition in definitions:
            skill = self._create_skill_from_definition(definition, source)
            if skill:
                await skill.on_load()
                self.registry.register(skill)
                skills.append(skill)

        return skills

    def _create_skill_from_definition(
        self,
        definition: dict[str, Any],
        source: Path,
    ) -> Skill | None:
        """Create skill from definition dict."""
        try:
            config = SkillConfig.from_dict(definition)

            # Check for implementation
            if "implementation" in definition:
                impl = definition["implementation"]

                if "module" in impl:
                    # Load from module
                    module = importlib.import_module(impl["module"])
                    cls = getattr(module, impl.get("class", "Skill"))
                    return cls(config)

                elif "code" in impl:
                    # Inline code (careful with security!)
                    code = impl["code"]
                    namespace = {}
                    exec(code, namespace)

                    if "execute" in namespace:
                        return FunctionSkill(namespace["execute"], config)

            # Create template-based skill
            if "template" in definition:
                return TemplateSkill(config, definition["template"])

            logger.warning(f"No implementation found for skill: {config.name}")
            return None

        except Exception as e:
            logger.error(f"Failed to create skill from definition: {e}")
            return None


class TemplateSkill(Skill):
    """Skill that uses a template response."""

    def __init__(self, config: SkillConfig, template: str):
        """
        Initialize template skill.

        Args:
            config: Skill configuration
            template: Response template
        """
        super().__init__(config)
        self.template = template

    async def execute(self, context):
        """Execute template skill."""
        from clawscope.message import Msg

        # Simple variable substitution
        response = self.template.format(
            user=context.message.name,
            content=context.message.get_text_content(),
            **context.metadata,
        )

        return Msg(
            name=self.name,
            content=response,
            role="assistant",
        )


__all__ = ["SkillLoader", "TemplateSkill"]
