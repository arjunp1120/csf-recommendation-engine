"""MCP resource - MCP server management and tools."""

from typing import Any, Dict, List, Optional

from .._base import AsyncResource, SyncResource
from ..models import MCPServerConfig


class MCP(SyncResource):
    """Synchronous MCP resource."""

    def connect(self, server_url: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Connect to an MCP server and list available tools."""
        data = {"server_url": server_url}
        if api_key:
            data["api_key"] = api_key
        return self._post("/api/mcp/connect", json_data=data)

    def list_servers(self) -> List[MCPServerConfig]:
        """List all saved MCP servers."""
        response = self._get("/api/mcp/servers")
        result = []
        for s in response:
            try:
                result.append(MCPServerConfig.model_validate(s))
            except Exception:
                # Fallback - create with available fields
                result.append(
                    MCPServerConfig(
                        id=s.get("id"),
                        name=s.get("name"),
                        url=s.get("url"),
                        api_key=s.get("api_key"),
                        tools=s.get("tools", []),
                        enabled=s.get("enabled", True),
                    )
                )
        return result

    def get_server(self, server_id: str) -> MCPServerConfig:
        """Get MCP server by ID."""
        response = self._get(f"/api/mcp/servers/{server_id}")
        return MCPServerConfig.model_validate(response)

    def create_server(
        self, url: str, name: Optional[str] = None, api_key: Optional[str] = None
    ) -> MCPServerConfig:
        """Add a new MCP server."""
        data = {"url": url}
        if name:
            data["name"] = name
        if api_key:
            data["api_key"] = api_key
        response = self._post("/api/mcp/servers", json_data=data)
        return MCPServerConfig.model_validate(response)

    def update_server(self, server_id: str, **kwargs) -> MCPServerConfig:
        """Update MCP server configuration."""
        response = self._put(f"/api/mcp/servers/{server_id}", json_data=kwargs)
        return MCPServerConfig.model_validate(response)

    def delete_server(self, server_id: str) -> Dict[str, str]:
        """Delete an MCP server."""
        return self._delete(f"/api/mcp/servers/{server_id}")

    def refresh_server(self, server_id: str) -> Dict[str, Any]:
        """Refresh tools list from MCP server."""
        return self._post(f"/api/mcp/servers/{server_id}/refresh")


class AsyncMCP(AsyncResource):
    """Asynchronous MCP resource."""

    async def connect(self, server_url: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Connect to an MCP server and list available tools."""
        data = {"server_url": server_url}
        if api_key:
            data["api_key"] = api_key
        return await self._post("/api/mcp/connect", json_data=data)

    async def list_servers(self) -> List[MCPServerConfig]:
        """List all saved MCP servers."""
        response = await self._get("/api/mcp/servers")
        result = []
        for s in response:
            try:
                result.append(MCPServerConfig.model_validate(s))
            except Exception:
                result.append(
                    MCPServerConfig(
                        id=s.get("id"),
                        name=s.get("name"),
                        url=s.get("url"),
                        api_key=s.get("api_key"),
                        tools=s.get("tools", []),
                        enabled=s.get("enabled", True),
                    )
                )
        return result

    async def get_server(self, server_id: str) -> MCPServerConfig:
        """Get MCP server by ID."""
        response = await self._get(f"/api/mcp/servers/{server_id}")
        return MCPServerConfig.model_validate(response)

    async def create_server(
        self, url: str, name: Optional[str] = None, api_key: Optional[str] = None
    ) -> MCPServerConfig:
        """Add a new MCP server."""
        data = {"url": url}
        if name:
            data["name"] = name
        if api_key:
            data["api_key"] = api_key
        response = await self._post("/api/mcp/servers", json_data=data)
        return MCPServerConfig.model_validate(response)

    async def update_server(self, server_id: str, **kwargs) -> MCPServerConfig:
        """Update MCP server configuration."""
        response = await self._put(f"/api/mcp/servers/{server_id}", json_data=kwargs)
        return MCPServerConfig.model_validate(response)

    async def delete_server(self, server_id: str) -> Dict[str, str]:
        """Delete an MCP server."""
        return await self._delete(f"/api/mcp/servers/{server_id}")

    async def refresh_server(self, server_id: str) -> Dict[str, Any]:
        """Refresh tools list from MCP server."""
        return await self._post(f"/api/mcp/servers/{server_id}/refresh")
