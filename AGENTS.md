# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `clawscope/`. Key areas include `agent/` for agent implementations, `channels/` for chat integrations, `model/` for provider adapters, `orchestration/` for multi-agent flows, `rag/` for retrieval components, `sandbox/` for execution isolation, and `cli/` plus `server.py` for local and API entry points. Root-level operational files include `pyproject.toml`, `Dockerfile`, `docker-compose.yml`, `deploy.sh`, and `DEPLOYMENT.md`. `pytest` is configured to discover tests from `tests/`; add new test modules there even though the directory does not exist yet.

## Build, Test, and Development Commands
Use Python 3.10+.

- `pip install -e .[dev,api]` installs the package with development and FastAPI dependencies.
- `clawscope init` creates `~/.clawscope/` with a starter config and workspace.
- `clawscope chat -m "hello"` runs a quick CLI smoke test against the configured model.
- `clawscope serve --reload --port 8080` starts the API server for local development.
- `pytest` runs the test suite.
- `pytest --cov=clawscope` checks coverage for the package.
- `ruff check .`, `black .`, and `mypy clawscope` cover linting, formatting, and static typing.
- `docker compose up --build` starts the containerized stack.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and keep lines within Ruff's 100-character limit. Prefer explicit type hints: `mypy` is configured with `disallow_untyped_defs = true`, so new functions should be typed. Use `snake_case` for modules, functions, and variables, `PascalCase` for classes, and keep Typer and FastAPI command names descriptive and verb-based.

## Testing Guidelines
Write tests with `pytest` and `pytest-asyncio`. Name files `test_*.py` and mirror the package layout where practical, for example `tests/agent/test_react.py` for `clawscope/agent/react.py`. Cover async flows, API endpoints, and provider or channel edge cases with mocks rather than live credentials.

## Commit & Pull Request Guidelines
Recent history uses Conventional Commit prefixes such as `feat:` and `chore:`; keep that pattern and write imperative summaries, for example `feat: add Slack channel retry handling`. Pull requests should explain the behavior change, note config or deployment impact, link related issues, and include request and response samples or screenshots for API or channel-facing changes.

## Security & Configuration Tips
Do not commit API keys or channel tokens. Store runtime secrets in environment variables or the user config under `~/.clawscope/`. Treat sandbox, channel, and model settings as environment-specific and document any required extras such as `.[sandbox]` or `.[telegram]` in the PR.
