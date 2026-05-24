import asyncio
import sys
from pathlib import Path

from clawscope.sandbox import SandboxConfig
from clawscope.sandbox.manager import SandboxManager


def test_sandbox_manager_direct_execution_preserves_env_and_workspace_cwd(
    tmp_path: Path,
) -> None:
    async def run_test() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        nested = workspace / "nested"
        nested.mkdir()

        manager = SandboxManager(
            config=SandboxConfig(
                enabled=False,
                workspace_path=workspace,
            ),
        )
        command = (
            f'"{sys.executable}" -c "import os, pathlib; '
            "print(os.environ['CLAWSCOPE_SANDBOX_TEST']); "
            'print(pathlib.Path.cwd())"'
        )

        result = await manager.execute(
            command=command,
            timeout=10,
            env={"CLAWSCOPE_SANDBOX_TEST": "direct-ok"},
            cwd="/workspace/nested",
        )

        assert result.success
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        assert lines == ["direct-ok", str(nested.resolve())]

    asyncio.run(run_test())
