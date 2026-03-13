"""Skills marketplace for ClawScope."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from clawscope.skills.base import SkillConfig, SkillCategory
from clawscope.skills.loader import SkillLoader
from clawscope.skills.registry import SkillRegistry


@dataclass
class MarketplaceSkill:
    """Skill listing from marketplace."""

    name: str
    version: str
    description: str
    author: str
    category: SkillCategory
    downloads: int = 0
    rating: float = 0.0
    tags: list[str] = field(default_factory=list)
    homepage: str | None = None
    repository: str | None = None
    download_url: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "category": self.category.value,
            "downloads": self.downloads,
            "rating": self.rating,
            "tags": self.tags,
            "homepage": self.homepage,
            "repository": self.repository,
            "download_url": self.download_url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarketplaceSkill":
        """Create from dictionary."""
        category = data.get("category", "other")
        if isinstance(category, str):
            category = SkillCategory(category)

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        return cls(
            name=data.get("name", ""),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            category=category,
            downloads=data.get("downloads", 0),
            rating=data.get("rating", 0.0),
            tags=data.get("tags", []),
            homepage=data.get("homepage"),
            repository=data.get("repository"),
            download_url=data.get("download_url"),
            created_at=created_at,
            updated_at=updated_at,
        )


class SkillMarketplace:
    """
    Marketplace for discovering and installing skills.

    Features:
    - Browse available skills
    - Search by category, tags, keywords
    - Install skills from remote sources
    - Version management
    - Rating and reviews
    """

    DEFAULT_REGISTRY_URL = "https://skills.clawscope.io/api/v1"

    def __init__(
        self,
        registry: SkillRegistry | None = None,
        loader: SkillLoader | None = None,
        skills_dir: Path | None = None,
        registry_url: str | None = None,
    ):
        """
        Initialize marketplace.

        Args:
            registry: Local skill registry
            loader: Skill loader
            skills_dir: Directory for installed skills
            registry_url: Remote registry URL
        """
        self.registry = registry or SkillRegistry()
        self.loader = loader or SkillLoader(self.registry)
        self.skills_dir = skills_dir or Path.home() / ".clawscope" / "skills"
        self.registry_url = registry_url or self.DEFAULT_REGISTRY_URL

        # Ensure skills directory exists
        self.skills_dir.mkdir(parents=True, exist_ok=True)

        # Installed skills manifest
        self._manifest_path = self.skills_dir / "manifest.json"
        self._installed: dict[str, dict[str, Any]] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load installed skills manifest."""
        if self._manifest_path.exists():
            try:
                self._installed = json.loads(
                    self._manifest_path.read_text(encoding="utf-8")
                )
            except Exception as e:
                logger.error(f"Failed to load manifest: {e}")
                self._installed = {}

    def _save_manifest(self) -> None:
        """Save installed skills manifest."""
        try:
            self._manifest_path.write_text(
                json.dumps(self._installed, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")

    async def search(
        self,
        query: str | None = None,
        category: SkillCategory | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[MarketplaceSkill]:
        """
        Search for skills in the marketplace.

        Args:
            query: Search query
            category: Filter by category
            tags: Filter by tags
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of matching skills
        """
        params = {
            "limit": limit,
            "offset": offset,
        }

        if query:
            params["q"] = query
        if category:
            params["category"] = category.value
        if tags:
            params["tags"] = ",".join(tags)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.registry_url}/skills",
                    params=params,
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()

                return [
                    MarketplaceSkill.from_dict(item)
                    for item in data.get("skills", [])
                ]

        except httpx.HTTPError as e:
            logger.error(f"Marketplace search failed: {e}")
            return []

    async def get_skill_info(self, name: str) -> MarketplaceSkill | None:
        """
        Get detailed information about a skill.

        Args:
            name: Skill name

        Returns:
            Skill information or None
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.registry_url}/skills/{name}",
                    timeout=30.0,
                )
                response.raise_for_status()
                return MarketplaceSkill.from_dict(response.json())

        except httpx.HTTPError as e:
            logger.error(f"Failed to get skill info: {e}")
            return None

    async def install(
        self,
        name: str,
        version: str | None = None,
        force: bool = False,
    ) -> bool:
        """
        Install a skill from the marketplace.

        Args:
            name: Skill name
            version: Specific version (latest if None)
            force: Force reinstall

        Returns:
            True if installed successfully
        """
        # Check if already installed
        if name in self._installed and not force:
            installed_version = self._installed[name].get("version")
            if version is None or version == installed_version:
                logger.info(f"Skill '{name}' already installed")
                return True

        # Get skill info
        skill_info = await self.get_skill_info(name)
        if not skill_info:
            logger.error(f"Skill '{name}' not found in marketplace")
            return False

        if not skill_info.download_url:
            logger.error(f"Skill '{name}' has no download URL")
            return False

        try:
            # Download skill
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    skill_info.download_url,
                    timeout=60.0,
                    follow_redirects=True,
                )
                response.raise_for_status()
                content = response.content

            # Save to skills directory
            skill_dir = self.skills_dir / name
            skill_dir.mkdir(parents=True, exist_ok=True)

            # Detect file type and save
            if skill_info.download_url.endswith(".py"):
                skill_file = skill_dir / "__init__.py"
                skill_file.write_bytes(content)
            elif skill_info.download_url.endswith((".zip", ".tar.gz")):
                # Extract archive
                await self._extract_archive(content, skill_dir)
            else:
                skill_file = skill_dir / "skill.yaml"
                skill_file.write_bytes(content)

            # Load the skill
            await self.loader.load_directory(skill_dir)

            # Update manifest
            self._installed[name] = {
                "version": skill_info.version,
                "installed_at": datetime.now().isoformat(),
                "path": str(skill_dir),
            }
            self._save_manifest()

            logger.info(f"Installed skill: {name} v{skill_info.version}")
            return True

        except Exception as e:
            logger.error(f"Failed to install skill '{name}': {e}")
            return False

    async def uninstall(self, name: str) -> bool:
        """
        Uninstall a skill.

        Args:
            name: Skill name

        Returns:
            True if uninstalled successfully
        """
        if name not in self._installed:
            logger.warning(f"Skill '{name}' is not installed")
            return False

        try:
            # Unregister from registry
            self.registry.unregister(name)

            # Remove files
            skill_path = Path(self._installed[name].get("path", ""))
            if skill_path.exists():
                import shutil
                shutil.rmtree(skill_path)

            # Update manifest
            del self._installed[name]
            self._save_manifest()

            logger.info(f"Uninstalled skill: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to uninstall skill '{name}': {e}")
            return False

    async def update(self, name: str | None = None) -> dict[str, bool]:
        """
        Update installed skills.

        Args:
            name: Specific skill to update (all if None)

        Returns:
            Dict of skill names to update success
        """
        results = {}

        skills_to_update = [name] if name else list(self._installed.keys())

        for skill_name in skills_to_update:
            if skill_name not in self._installed:
                continue

            current_version = self._installed[skill_name].get("version")
            skill_info = await self.get_skill_info(skill_name)

            if skill_info and skill_info.version != current_version:
                success = await self.install(skill_name, force=True)
                results[skill_name] = success
            else:
                results[skill_name] = True  # Already up to date

        return results

    def list_installed(self) -> list[dict[str, Any]]:
        """List installed skills."""
        return [
            {
                "name": name,
                **info,
            }
            for name, info in self._installed.items()
        ]

    async def load_installed(self) -> int:
        """
        Load all installed skills.

        Returns:
            Number of skills loaded
        """
        count = 0

        for name, info in self._installed.items():
            path = Path(info.get("path", ""))
            if path.exists():
                skills = await self.loader.load_directory(path)
                count += len(skills)

        logger.info(f"Loaded {count} installed skills")
        return count

    async def _extract_archive(self, content: bytes, target: Path) -> None:
        """Extract archive content to target directory."""
        import tarfile
        import zipfile
        import io

        # Try ZIP first
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                zf.extractall(target)
                return
        except zipfile.BadZipFile:
            pass

        # Try tar.gz
        try:
            with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tf:
                tf.extractall(target)
                return
        except tarfile.TarError:
            pass

        raise ValueError("Unknown archive format")

    async def publish(
        self,
        skill_path: Path,
        api_key: str,
    ) -> bool:
        """
        Publish a skill to the marketplace.

        Args:
            skill_path: Path to skill directory or file
            api_key: Marketplace API key

        Returns:
            True if published successfully
        """
        try:
            # Load skill to get metadata
            skills = await self.loader.load_file(skill_path)
            if not skills:
                logger.error("No skills found in path")
                return False

            skill = skills[0]

            # Prepare upload
            files = {}
            if skill_path.is_dir():
                import shutil
                with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                    shutil.make_archive(tmp.name[:-4], "zip", skill_path)
                    tmp_path = Path(tmp.name)
                files["file"] = ("skill.zip", tmp_path.read_bytes())
            else:
                files["file"] = (skill_path.name, skill_path.read_bytes())

            # Upload to marketplace
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.registry_url}/skills",
                    files=files,
                    data=skill.config.to_dict(),
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=120.0,
                )
                response.raise_for_status()

            logger.info(f"Published skill: {skill.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish skill: {e}")
            return False


__all__ = ["SkillMarketplace", "MarketplaceSkill"]
