"""Triggers resource - Execution triggers, webhooks, events."""

from typing import Any, Dict, List, Optional

from .._base import AsyncResource, SyncResource
from ..models import Trigger


class Triggers(SyncResource):
    """Synchronous triggers resource."""

    def create(
        self, name: str, target_type: str, target_id: str, trigger_type: str, **kwargs
    ) -> Trigger:
        """Create a new execution trigger."""
        data = {
            "name": name,
            "target_type": target_type,
            "target_id": target_id,
            "trigger_type": trigger_type,
            **kwargs,
        }
        response = self._post("/api/triggers", json_data=data)
        return Trigger.model_validate(response)

    def list(self, skip: int = 0, limit: int = 100, **filters) -> List[Trigger]:
        """List triggers with optional filtering."""
        params = {"skip": skip, "limit": limit, **filters}
        response = self._get("/api/triggers", params=params)
        return [Trigger.model_validate(t) for t in response]

    def get(self, trigger_id: str) -> Trigger:
        """Get a specific trigger by ID."""
        response = self._get(f"/api/triggers/{trigger_id}")
        return Trigger.model_validate(response)

    def update(self, trigger_id: str, **kwargs) -> Trigger:
        """Update a trigger."""
        response = self._patch(f"/api/triggers/{trigger_id}", json_data=kwargs)
        return Trigger.model_validate(response)

    def delete(self, trigger_id: str) -> Dict[str, str]:
        """Delete a trigger."""
        return self._delete(f"/api/triggers/{trigger_id}")

    def fire(
        self, trigger_id: str, message: Optional[str] = None, data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Manually fire a trigger."""
        payload: Dict[str, Any] = {}
        if message:
            payload["message"] = message
        if data:
            payload["data"] = data
        return self._post(f"/api/triggers/{trigger_id}/fire", json_data=payload)

    def test(self, trigger_id: str, **kwargs) -> Dict[str, Any]:
        """Test a trigger without actually executing."""
        return self._post(f"/api/triggers/{trigger_id}/test", json_data=kwargs)

    def enable(self, trigger_id: str) -> Dict[str, Any]:
        """Enable a trigger."""
        response = self._patch(f"/api/triggers/{trigger_id}", json_data={"enabled": True})
        return response if isinstance(response, dict) else {"enabled": True}

    def disable(self, trigger_id: str) -> Dict[str, Any]:
        """Disable a trigger."""
        response = self._patch(f"/api/triggers/{trigger_id}", json_data={"enabled": False})
        return response if isinstance(response, dict) else {"enabled": False}

    def execute(self, trigger_id: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Manually execute a trigger."""
        return self._post(f"/api/triggers/{trigger_id}/execute", json_data=data or {})

    def executions(self, trigger_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """List executions for a trigger."""
        return self._get(f"/api/triggers/{trigger_id}/executions", params={"limit": limit})


class AsyncTriggers(AsyncResource):
    """Asynchronous triggers resource."""

    async def create(
        self, name: str, target_type: str, target_id: str, trigger_type: str, **kwargs
    ) -> Trigger:
        """Create a new execution trigger."""
        data = {
            "name": name,
            "target_type": target_type,
            "target_id": target_id,
            "trigger_type": trigger_type,
            **kwargs,
        }
        response = await self._post("/api/triggers", json_data=data)
        return Trigger.model_validate(response)

    async def list(self, skip: int = 0, limit: int = 100, **filters) -> List[Trigger]:
        """List triggers with optional filtering."""
        params = {"skip": skip, "limit": limit, **filters}
        response = await self._get("/api/triggers", params=params)
        return [Trigger.model_validate(t) for t in response]

    async def get(self, trigger_id: str) -> Trigger:
        """Get a specific trigger by ID."""
        response = await self._get(f"/api/triggers/{trigger_id}")
        return Trigger.model_validate(response)

    async def update(self, trigger_id: str, **kwargs) -> Trigger:
        """Update a trigger."""
        response = await self._patch(f"/api/triggers/{trigger_id}", json_data=kwargs)
        return Trigger.model_validate(response)

    async def delete(self, trigger_id: str) -> Dict[str, str]:
        """Delete a trigger."""
        return await self._delete(f"/api/triggers/{trigger_id}")

    async def fire(
        self, trigger_id: str, message: Optional[str] = None, data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Manually fire a trigger."""
        payload: Dict[str, Any] = {}
        if message:
            payload["message"] = message
        if data:
            payload["data"] = data
        return await self._post(f"/api/triggers/{trigger_id}/fire", json_data=payload)

    async def test(self, trigger_id: str, **kwargs) -> Dict[str, Any]:
        """Test a trigger without actually executing."""
        return await self._post(f"/api/triggers/{trigger_id}/test", json_data=kwargs)

    async def enable(self, trigger_id: str) -> Dict[str, Any]:
        """Enable a trigger."""
        response = await self._patch(f"/api/triggers/{trigger_id}", json_data={"enabled": True})
        return response if isinstance(response, dict) else {"enabled": True}

    async def disable(self, trigger_id: str) -> Dict[str, Any]:
        """Disable a trigger."""
        response = await self._patch(f"/api/triggers/{trigger_id}", json_data={"enabled": False})
        return response if isinstance(response, dict) else {"enabled": False}

    async def execute(self, trigger_id: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Manually execute a trigger."""
        return await self._post(f"/api/triggers/{trigger_id}/execute", json_data=data or {})

    async def executions(self, trigger_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """List executions for a trigger."""
        return await self._get(f"/api/triggers/{trigger_id}/executions", params={"limit": limit})
