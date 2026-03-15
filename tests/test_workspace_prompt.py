from pathlib import Path

from clawscope.workspace_prompt import WorkspacePromptBuilder


def test_workspace_prompt_includes_bootstrap_and_skills(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "skills").mkdir()

    (workspace / "AGENTS.md").write_text("Repository-specific rules.", encoding="utf-8")

    skill_dir = workspace / "skills" / "weather"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "# Weather\n\nChecks current weather for a location.",
        encoding="utf-8",
    )

    prompt = WorkspacePromptBuilder(workspace).build("Base prompt.")

    assert "Base prompt." in prompt
    assert "Repository-specific rules." in prompt
    assert "`weather`" in prompt
    assert "Checks current weather for a location." in prompt
    assert str(workspace.resolve()) in prompt
