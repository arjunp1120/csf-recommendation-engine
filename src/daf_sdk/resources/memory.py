"""Memory resource - Shared memory management."""

from typing import Any, Dict, List, Optional

from .._base import AsyncResource, SyncResource, parse_list_response, parse_response
from ..models import MemoryBlock


class SharedMemory(SyncResource):
    """Synchronous shared memory resource."""

    def list(self, skip: int = 0, limit: int = 100) -> List[MemoryBlock]:
        """List shared memory blocks."""
        response = self._get("/api/memory/shared", params={"skip": skip, "limit": limit})
        return parse_list_response(response, MemoryBlock)

    def get(self, label: str) -> MemoryBlock:
        """Get a specific shared memory block by label."""
        response = self._get(f"/api/memory/shared/{label}")
        return parse_response(response, MemoryBlock)

    def create(
        self, label: str, value: str, description: Optional[str] = None, **kwargs
    ) -> MemoryBlock:
        """
        Create a shared memory block.

        Args:
            label: Unique label for the memory block
            value: Value to store
            description: Optional description

        Returns:
            Created MemoryBlock
        """
        data = {"label": label, "value": value}
        if description:
            data["description"] = description
        response = self._post("/api/memory/shared", json_data=data)
        return parse_response(response, MemoryBlock)

    def update(
        self, label: str, value: Optional[str] = None, description: Optional[str] = None, **kwargs
    ) -> MemoryBlock:
        """Update a shared memory block."""
        data = {}
        if value is not None:
            data["value"] = value
        if description is not None:
            data["description"] = description
        response = self._patch(f"/api/memory/shared/{label}", json_data=data)
        return parse_response(response, MemoryBlock)

    def delete(self, label: str) -> Dict[str, Any]:
        """Delete a shared memory block."""
        return self._delete(f"/api/memory/shared/{label}")


class AsyncSharedMemory(AsyncResource):
    """Asynchronous shared memory resource."""

    async def list(self, skip: int = 0, limit: int = 100) -> List[MemoryBlock]:
        """List shared memory blocks."""
        response = await self._get("/api/memory/shared", params={"skip": skip, "limit": limit})
        return parse_list_response(response, MemoryBlock)

    async def get(self, label: str) -> MemoryBlock:
        """Get a specific shared memory block by label."""
        response = await self._get(f"/api/memory/shared/{label}")
        return parse_response(response, MemoryBlock)

    async def create(
        self, label: str, value: str, description: Optional[str] = None, **kwargs
    ) -> MemoryBlock:
        """Create a shared memory block."""
        data = {"label": label, "value": value}
        if description:
            data["description"] = description
        response = await self._post("/api/memory/shared", json_data=data)
        return parse_response(response, MemoryBlock)

    async def update(
        self, label: str, value: Optional[str] = None, description: Optional[str] = None, **kwargs
    ) -> MemoryBlock:
        """Update a shared memory block."""
        data = {}
        if value is not None:
            data["value"] = value
        if description is not None:
            data["description"] = description
        response = await self._patch(f"/api/memory/shared/{label}", json_data=data)
        return parse_response(response, MemoryBlock)

    async def delete(self, label: str) -> Dict[str, Any]:
        """Delete a shared memory block."""
        return await self._delete(f"/api/memory/shared/{label}")


class Memory(SyncResource):
    """Memory resource - provides access to shared memory."""

    def __init__(self, client):
        super().__init__(client)
        self.shared = SharedMemory(client)


class AsyncMemory(AsyncResource):
    """Async Memory resource - provides access to shared memory."""

    def __init__(self, client):
        super().__init__(client)
        self.shared = AsyncSharedMemory(client)
