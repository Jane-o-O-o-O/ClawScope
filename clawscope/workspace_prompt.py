"""Workspace-aware system prompt builder."""

from __future__ import annotations

from pathlib import Path


class WorkspacePromptBuilder:
    """Build an OpenClaw-style system prompt from workspace assets."""

    BOOTSTRAP_FILES = ("AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md")

    def __init__(self, workspace: Path):
        self.workspace = workspace

    def build(self, base_prompt: str) -> str:
        """Build the final system prompt."""
        parts = [
            self._build_runtime_section(),
            base_prompt.strip(),
        ]

        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        skills = self._build_skills_summary()
        if skills:
            parts.append(skills)

        return "\n\n---\n\n".join(part for part in parts if part.strip())

    def _build_runtime_section(self) -> str:
        """Build stable workspace instructions."""
        workspace_path = str(self.workspace.expanduser().resolve())
        return f"""# Workspace

Your workspace is: {workspace_path}

- Bootstrap instructions live in workspace files such as `AGENTS.md`, `SOUL.md`, `USER.md`, and `TOOLS.md`.
- Local skills live under `{workspace_path}/skills/`.
- Read workspace files before making assumptions about project-specific behavior.
- When a local skill looks relevant, read its `SKILL.md` file before using it.
"""

    def _load_bootstrap_files(self) -> str:
        """Load bootstrap markdown files from the workspace root."""
        parts: list[str] = []

        for filename in self.BOOTSTRAP_FILES:
            path = self.workspace / filename
            if not path.exists() or not path.is_file():
                continue

            try:
                content = path.read_text(encoding="utf-8").strip()
            except OSError:
                continue

            if content:
                parts.append(f"## {filename}\n\n{content}")

        return "\n\n".join(parts)

    def _build_skills_summary(self) -> str:
        """Summarize local skills available in the workspace."""
        skills_dir = self.workspace / "skills"
        if not skills_dir.exists() or not skills_dir.is_dir():
            return ""

        entries: list[str] = []
        for child in sorted(skills_dir.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue

            skill_md = child / "SKILL.md"
            if not skill_md.exists():
                continue

            summary = self._read_skill_summary(skill_md)
            if summary:
                entries.append(f"- `{child.name}`: {summary}")
            else:
                entries.append(f"- `{child.name}`: Read `{child.name}/SKILL.md` for usage details.")

        if not entries:
            return ""

        return (
            "# Local Skills\n\n"
            "The following workspace skills are available. Read the relevant `SKILL.md` before using one.\n\n"
            + "\n".join(entries)
        )

    def _read_skill_summary(self, path: Path) -> str:
        """Read a short summary from a skill file."""
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return ""

        for raw_line in lines:
            line = raw_line.strip()
            if not line or line == "---":
                continue
            if line.startswith("#"):
                continue
            if ":" in line and len(line.split(":", 1)[0]) < 20:
                continue
            return line[:180]

        return ""


__all__ = ["WorkspacePromptBuilder"]
