"""File system tools for ClawScope."""

from __future__ import annotations

import os
from pathlib import Path

import aiofiles


async def read_file(path: str) -> str:
    """
    Read the contents of a file.

    Args:
        path: Path to the file to read

    Returns:
        File contents as string
    """
    try:
        file_path = Path(path).expanduser().resolve()

        if not file_path.exists():
            return f"Error: File not found: {path}"

        if not file_path.is_file():
            return f"Error: Not a file: {path}"

        # Check file size
        size = file_path.stat().st_size
        if size > 1_000_000:  # 1MB limit
            return f"Error: File too large ({size} bytes). Maximum is 1MB."

        async with aiofiles.open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = await f.read()

        return content

    except Exception as e:
        return f"Error reading file: {str(e)}"


async def write_file(path: str, content: str) -> str:
    """
    Write content to a file.

    Args:
        path: Path to the file to write
        content: Content to write

    Returns:
        Success message or error
    """
    try:
        file_path = Path(path).expanduser().resolve()

        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(content)

        return f"Successfully wrote {len(content)} characters to {path}"

    except Exception as e:
        return f"Error writing file: {str(e)}"


async def list_dir(path: str) -> str:
    """
    List contents of a directory.

    Args:
        path: Path to the directory

    Returns:
        Directory listing
    """
    try:
        dir_path = Path(path).expanduser().resolve()

        if not dir_path.exists():
            return f"Error: Directory not found: {path}"

        if not dir_path.is_dir():
            return f"Error: Not a directory: {path}"

        entries = []
        for entry in sorted(dir_path.iterdir()):
            if entry.is_dir():
                entries.append(f"[DIR]  {entry.name}/")
            else:
                size = entry.stat().st_size
                entries.append(f"[FILE] {entry.name} ({size} bytes)")

        if not entries:
            return "Directory is empty"

        return "\n".join(entries)

    except Exception as e:
        return f"Error listing directory: {str(e)}"


__all__ = ["read_file", "write_file", "list_dir"]
