"""Tools resource - Tool listing, custom tools, execution."""

from typing import TYPE_CHECKING, Any, Dict, List

from .._base import AsyncResource, SyncResource
from ..models import CustomTool, ToolDefinition

if TYPE_CHECKING:
    from ..decorators import ToolFunction


class Tools(SyncResource):
    """Synchronous tools resource."""

    def list(self) -> List[ToolDefinition]:
        """Get all available tools including custom tools."""
        response = self._get("/api/tools")
        return [ToolDefinition.model_validate(t) for t in response]

    def list_custom(self) -> List[CustomTool]:
        """List all custom tools."""
        response = self._get("/api/tools/custom")
        return [CustomTool.model_validate(t) for t in response]

    def get_custom(self, name: str) -> CustomTool:
        """Get a specific custom tool by name."""
        response = self._get(f"/api/tools/custom/{name}")
        return CustomTool.model_validate(response)

    def create_custom(
        self, name: str, description: str, parameters: Dict, code: str, **kwargs
    ) -> CustomTool:
        """Create a custom tool."""
        data = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "code": code,
            **kwargs,
        }
        response = self._post("/api/tools/custom", json_data=data)
        return CustomTool.model_validate(response)

    def update_custom(self, name: str, **kwargs) -> CustomTool:
        """Update a custom tool."""
        response = self._put(f"/api/tools/custom/{name}", json_data=kwargs)
        return CustomTool.model_validate(response)

    def delete_custom(self, name: str) -> Dict[str, str]:
        """Delete a custom tool."""
        return self._delete(f"/api/tools/custom/{name}")

    def execute(self, tool_name: str, arguments: Dict, **kwargs) -> Dict[str, Any]:
        """Execute a tool directly."""
        data = {"arguments": arguments, **kwargs}
        return self._post(f"/api/tools/{tool_name}/execute", json_data=data)

    def register(self, tool_func: "ToolFunction") -> CustomTool:
        """
        Register a @custom_tool decorated function with the backend.

        Args:
            tool_func: Function decorated with @custom_tool

        Returns:
            Created CustomTool

        Example:
            @custom_tool
            def my_tool(arg: str) -> str:
                '''My tool description.'''
                return arg

            registered = client.tools.register(my_tool)
        """
        data = tool_func.to_dict()
        response = self._post("/api/tools/custom", json_data=data)
        return CustomTool.model_validate(response)

    def register_all(self, tool_funcs: List["ToolFunction"]) -> List[CustomTool]:
        """
        Register multiple @custom_tool decorated functions.

        Args:
            tool_funcs: List of functions decorated with @custom_tool

        Returns:
            List of created CustomTools
        """
        results = []
        for tool_func in tool_funcs:
            result = self.register(tool_func)
            results.append(result)
        return results


class AsyncTools(AsyncResource):
    """Asynchronous tools resource."""

    async def list(self) -> List[ToolDefinition]:
        """Get all available tools including custom tools."""
        response = await self._get("/api/tools")
        return [ToolDefinition.model_validate(t) for t in response]

    async def list_custom(self) -> List[CustomTool]:
        """List all custom tools."""
        response = await self._get("/api/tools/custom")
        return [CustomTool.model_validate(t) for t in response]

    async def get_custom(self, name: str) -> CustomTool:
        """Get a specific custom tool by name."""
        response = await self._get(f"/api/tools/custom/{name}")
        return CustomTool.model_validate(response)

    async def create_custom(
        self, name: str, description: str, parameters: Dict, code: str, **kwargs
    ) -> CustomTool:
        """Create a custom tool."""
        data = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "code": code,
            **kwargs,
        }
        response = await self._post("/api/tools/custom", json_data=data)
        return CustomTool.model_validate(response)

    async def update_custom(self, name: str, **kwargs) -> CustomTool:
        """Update a custom tool."""
        response = await self._put(f"/api/tools/custom/{name}", json_data=kwargs)
        return CustomTool.model_validate(response)

    async def delete_custom(self, name: str) -> Dict[str, str]:
        """Delete a custom tool."""
        return await self._delete(f"/api/tools/custom/{name}")

    async def execute(self, tool_name: str, arguments: Dict, **kwargs) -> Dict[str, Any]:
        """Execute a tool directly."""
        data = {"arguments": arguments, **kwargs}
        return await self._post(f"/api/tools/{tool_name}/execute", json_data=data)

    async def register(self, tool_func: "ToolFunction") -> CustomTool:
        """Register a @custom_tool decorated function with the backend."""
        data = tool_func.to_dict()
        response = await self._post("/api/tools/custom", json_data=data)
        return CustomTool.model_validate(response)

    async def register_all(self, tool_funcs: List["ToolFunction"]) -> List[CustomTool]:
        """Register multiple @custom_tool decorated functions."""
        results = []
        for tool_func in tool_funcs:
            result = await self.register(tool_func)
            results.append(result)
        return results
