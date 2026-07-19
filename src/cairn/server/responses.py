"""Shared MCP response builders for all handler modules."""


def mcp_response(text: str) -> dict:
    """Build a successful MCP response."""
    return {"content": [{"type": "text", "text": str(text)}]}


def mcp_error(text: str) -> dict:
    """Build an error MCP response."""
    return {"content": [{"type": "text", "text": f"Error: {text}"}], "isError": True}
