"""Built-in tools for ClawScope."""

from clawscope.tool.builtin.filesystem import read_file, write_file, list_dir
from clawscope.tool.builtin.shell import execute_shell
from clawscope.tool.builtin.web import web_search, web_fetch


def get_builtin_tools(config):
    """Get all built-in tools."""
    from clawscope.tool.registry import Tool, ToolParameter

    tools = []

    # File system tools
    tools.append(Tool(
        name="read_file",
        description="Read the contents of a file",
        parameters=[
            ToolParameter(name="path", type="string", description="Path to the file to read"),
        ],
        func=read_file,
    ))

    tools.append(Tool(
        name="write_file",
        description="Write content to a file",
        parameters=[
            ToolParameter(name="path", type="string", description="Path to the file to write"),
            ToolParameter(name="content", type="string", description="Content to write"),
        ],
        func=write_file,
    ))

    tools.append(Tool(
        name="list_dir",
        description="List contents of a directory",
        parameters=[
            ToolParameter(name="path", type="string", description="Path to the directory"),
        ],
        func=list_dir,
    ))

    # Shell tool
    tools.append(Tool(
        name="execute_shell",
        description="Execute a shell command",
        parameters=[
            ToolParameter(name="command", type="string", description="Shell command to execute"),
            ToolParameter(name="timeout", type="integer", description="Timeout in seconds", required=False, default=60),
        ],
        func=lambda command, timeout=60: execute_shell(command, timeout, config),
    ))

    # Web tools
    tools.append(Tool(
        name="web_search",
        description="Search the web for information",
        parameters=[
            ToolParameter(name="query", type="string", description="Search query"),
        ],
        func=web_search,
    ))

    tools.append(Tool(
        name="web_fetch",
        description="Fetch content from a URL",
        parameters=[
            ToolParameter(name="url", type="string", description="URL to fetch"),
        ],
        func=web_fetch,
    ))

    return tools


__all__ = ["get_builtin_tools"]
