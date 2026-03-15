"""ClawScope CLI."""

import asyncio
import sys
from pathlib import Path

import typer
from loguru import logger

from clawscope._version import __version__

app = typer.Typer(
    name="clawscope",
    help="ClawScope - Unified AI Agent Platform",
    add_completion=False,
)


@app.command()
def version():
    """Show version information."""
    typer.echo(f"ClawScope v{__version__}")


@app.command()
def init(
    path: Path = typer.Option(
        Path.home() / ".clawscope",
        "--path", "-p",
        help="Configuration directory",
    ),
):
    """Initialize ClawScope configuration."""
    config_dir = path
    config_dir.mkdir(parents=True, exist_ok=True)

    workspace = config_dir / "workspace"
    workspace.mkdir(exist_ok=True)
    (workspace / "sessions").mkdir(exist_ok=True)
    (workspace / "memory").mkdir(exist_ok=True)
    (workspace / "skills").mkdir(exist_ok=True)

    # Create default config
    config_file = config_dir / "config.yaml"
    if not config_file.exists():
        default_config = """# ClawScope Configuration
project: ClawScope
workspace: {workspace}

model:
  provider: openai
  default_model: gpt-4

agent:
  type: react
  name: Assistant
  max_iterations: 40

channels:
  telegram:
    enabled: false
  discord:
    enabled: false

services:
  cron_enabled: true
  heartbeat_enabled: true
  heartbeat_interval: 1800
""".format(workspace=workspace)

        config_file.write_text(default_config)
        typer.echo(f"Created config: {config_file}")

    typer.echo(f"Initialized ClawScope at: {config_dir}")


@app.command()
def chat(
    config: Path = typer.Option(
        Path.home() / ".clawscope" / "config.yaml",
        "--config", "-c",
        help="Configuration file",
    ),
    message: str = typer.Option(
        None,
        "--message", "-m",
        help="Single message to send",
    ),
):
    """Interactive chat with ClawScope agent."""
    asyncio.run(_chat(config, message))


async def _chat(config_path: Path, message: str | None):
    """Run interactive chat."""
    from clawscope.conversation_context import attach_runtime_context
    from clawscope.config import Config
    from clawscope.kernel import build_kernel
    from clawscope.model import ModelRegistry
    from clawscope.memory import InMemoryMemory
    from clawscope.tool import ToolRegistry
    from clawscope.message import Msg

    # Load config
    if config_path.exists():
        config = Config.from_file(config_path)
    else:
        config = Config()

    # Initialize components
    model_registry = ModelRegistry(config.model)
    tool_registry = ToolRegistry(config.tools)
    await tool_registry.load_builtin_tools()

    kernel = build_kernel(
        agent_config=config.agent,
        model_config=config.model,
        model_registry=model_registry,
        tool_registry=tool_registry,
        workspace=config.workspace,
    )
    agent = kernel.create_agent(
        memory=InMemoryMemory(),
    )

    # Single message mode
    if message:
        msg = attach_runtime_context(
            Msg(name="User", content=message, role="user"),
            channel="cli",
            chat_id="interactive",
            session_key="cli:interactive",
            sender_id="user",
        )
        response = await agent(msg)
        typer.echo(f"\n{agent.name}: {response.get_text_content()}\n")
        return

    # Interactive mode
    typer.echo(f"\nClawScope v{__version__} - Interactive Chat")
    typer.echo("Type 'exit' to quit\n")

    while True:
        try:
            user_input = typer.prompt("You")
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("exit", "quit", "bye"):
            break

        msg = attach_runtime_context(
            Msg(name="User", content=user_input, role="user"),
            channel="cli",
            chat_id="interactive",
            session_key="cli:interactive",
            sender_id="user",
        )
        response = await agent(msg)
        typer.echo(f"\n{agent.name}: {response.get_text_content()}\n")

    typer.echo("Goodbye!")


@app.command()
def serve(
    config: Path = typer.Option(
        Path.home() / ".clawscope" / "config.yaml",
        "--config", "-c",
        help="Configuration file",
    ),
    port: int = typer.Option(
        8080,
        "--port", "-p",
        help="API server port",
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host", "-h",
        help="Host to bind",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable auto-reload",
    ),
):
    """Start ClawScope API server."""
    try:
        from clawscope.server import run_server

        typer.echo(f"Starting ClawScope API server on {host}:{port}...")

        config_path = str(config) if config.exists() else None
        run_server(
            host=host,
            port=port,
            config_path=config_path,
            reload=reload,
        )

    except ImportError:
        typer.echo("Error: FastAPI not installed. Run: pip install clawscope[api]")
        raise typer.Exit(1)


@app.command()
def status(
    config: Path = typer.Option(
        Path.home() / ".clawscope" / "config.yaml",
        "--config", "-c",
        help="Configuration file",
    ),
):
    """Show ClawScope status."""
    from clawscope.config import Config

    typer.echo(f"ClawScope v{__version__}\n")

    if not config.exists():
        typer.echo("Status: Not initialized")
        typer.echo(f"Run 'clawscope init' to create configuration")
        return

    cfg = Config.from_file(config)

    typer.echo(f"Config: {config}")
    typer.echo(f"Workspace: {cfg.workspace}")
    typer.echo(f"Model Provider: {cfg.model.provider}")
    typer.echo(f"Default Model: {cfg.model.default_model}")

    # Check API keys
    import os
    api_key = os.environ.get("OPENAI_API_KEY") or cfg.model.api_key
    if api_key:
        typer.echo("API Key: Configured")
    else:
        typer.echo("API Key: NOT SET")


def main():
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
