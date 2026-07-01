#!/usr/bin/env python3
"""
research-agent MCP server.

Exposes the research-agent engine as MCP-standard tools. Machine OS spoke backing
the `~~research` connector. Privacy-first by default: with no search API keys set,
it runs in raw DuckDuckGo mode (anonymous, no query logging tied to you), so it
gives value cold and keeps outbound research sovereign.

Transport: stdio (works with Claude Code, Cline, Claude Desktop). Console-script
target (`research-agent`), launchable via:
  uvx --from <path-or-git> research-agent

Mirrors the agent-extractor spoke: static TOOLS with MCP behaviour-hint annotations,
JSON text results, errors surfaced as tool errors (never crash the server). The
research() call is async, so dispatch is async and awaited in call_tool.

Tools:
  research   - research a topic via concurrent multi-source search + LLM synthesis
  info       - read-only spoke introspection (name, version, tool names)

Security note: this is the ONE outbound spoke. When run behind the MCP gateway,
its calls are egress-checked + sanitized (see SOVEREIGN-AGENT-SECURITY-TRUTH.md).
"""
import asyncio
import json
from dataclasses import asdict
from typing import Any, Dict, List

VERSION = "0.1.0"  # keep in sync with pyproject.toml [project].version

TOOLS = [
    {
        "name": "research",
        "description": (
            "Research a topic using concurrent multi-source web search and LLM "
            "synthesis. Returns an executive summary, key findings, and cited "
            "sources. Works with zero API keys (anonymous DuckDuckGo raw mode); "
            "richer with SERPAPI/TAVILY/BRAVE keys in env."
        ),
        # Reads the web (outbound), no local writes, non-deterministic -> open world.
        "annotations": {
            "title": "Research a topic",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "the subject to research"},
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "optional search keywords; derived from topic if omitted",
                },
                "depth": {
                    "type": "string",
                    "enum": ["quick", "standard", "deep"],
                    "description": "search breadth (default: standard)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "max sources to gather (default: 8)",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "info",
        "description": (
            "Return this spoke's name, version, and available tool names. "
            "Read-only introspection; takes no arguments."
        ),
        "annotations": {
            "title": "Spoke info",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
]


async def _dispatch(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Async dispatch - separated from transport so it is unit-testable offline."""
    if name == "info":
        return {
            "name": "research-agent",
            "version": VERSION,
            "tools": [t["name"] for t in TOOLS],
        }

    if name == "research":
        # Imported lazily so `info` stays dependency-light and the server boots
        # even if the search/LLM stack is not fully configured.
        from .agents.research_agent import ResearchAgent, ResearchTask

        task = ResearchTask(
            topic=args["topic"],
            keywords=args.get("keywords") or args["topic"].split(),
            max_results=int(args.get("max_results", 8)),
            depth=args.get("depth", "standard"),
        )
        report = await ResearchAgent().research(task)
        return asdict(report)

    raise ValueError(f"Unknown tool: {name}")


def main() -> None:
    """Console-script entry point: run the stdio MCP server."""
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    import mcp.types as types

    server = Server("research-agent")

    @server.list_tools()
    async def list_tools() -> List[types.Tool]:
        return [
            types.Tool(
                name=t["name"],
                description=t["description"],
                inputSchema=t["inputSchema"],
                annotations=types.ToolAnnotations(**t["annotations"]),
            )
            for t in TOOLS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
        try:
            result = await _dispatch(name, arguments or {})
            text = json.dumps(result, indent=2, default=str)
        except Exception as err:  # surface errors as tool errors, never crash the server
            text = f"Error: {err}"
        return [types.TextContent(type="text", text=text)]

    async def _run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(_run())


if __name__ == "__main__":
    main()
