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
    from clawscope.config import Config
    from clawscope.model import ModelRegistry
    from clawscope.agent import ReActAgent
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

    model = model_registry.get_model()
    memory = InMemoryMemory()

    agent = ReActAgent(
        name=config.agent.name,
        sys_prompt=config.agent.sys_prompt,
        model=model,
        memory=memory,
        tools=tool_registry,
        max_iterations=config.agent.max_iterations,
    )

    # Single message mode
    if message:
        msg = Msg(name="User", content=message, role="user")
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

        msg = Msg(name="User", content=user_input, role="user")
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
):
    """Start ClawScope gateway server."""
    asyncio.run(_serve(config, port))


async def _serve(config_path: Path, port: int):
    """Run the gateway server."""
    from clawscope import ClawScope

    typer.echo(f"Starting ClawScope gateway on port {port}...")

    if not config_path.exists():
        typer.echo(f"Config not found: {config_path}")
        typer.echo("Run 'clawscope init' first")
        raise typer.Exit(1)

    app = ClawScope.from_config(config_path)

    try:
        await app.run_forever()
    except KeyboardInterrupt:
        typer.echo("\nShutting down...")
        await app.stop()


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
