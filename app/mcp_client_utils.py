import os
import asyncio
from fastmcp import Client

MCP_SERVER_PATH = os.path.join(os.path.dirname(__file__), "..", "mcp", "mcp_server.py")

async def _get_tools_async():
    try:
        async with Client(MCP_SERVER_PATH) as client:
            tools = await client.list_tools()
            docs = []
            for t in tools:
                docs.append(f"- Tool Name: [{t.name}]\n  Description: {t.description.strip()}")
            return "\n\n".join(docs)
    except Exception as e:
        return f"Error fetching tools: {e}"

def get_mcp_tools_list_sync():
    """Returns a formatted string of all available MCP tools and their descriptions."""
    return asyncio.run(_get_tools_async())
