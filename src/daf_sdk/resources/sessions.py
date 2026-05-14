"""Sessions resource - Chat session management."""

from typing import Dict, List, Optional

from .._base import AsyncResource, SyncResource
from ..models import ChatSession


class Sessions(SyncResource):
    """Synchronous sessions resource."""

    def create(
        self, agent_id: Optional[str] = None, team_id: Optional[str] = None, **kwargs
    ) -> ChatSession:
        """Create a new chat session."""
        data = {}
        if agent_id:
            data["agent_id"] = agent_id
        if team_id:
            data["team_id"] = team_id
        response = self._post("/api/sessions/", json_data=data)
        return ChatSession.model_validate(response)

    def list(self, skip: int = 0, limit: int = 100, **filters) -> List[ChatSession]:
        """List chat sessions with optional filtering."""
        params = {"skip": skip, "limit": limit, **filters}
        response = self._get("/api/sessions/", params=params)
        return [ChatSession.model_validate(s) for s in response]

    def get(self, session_id: str) -> ChatSession:
        """Get a specific session by ID."""
        response = self._get(f"/api/sessions/{session_id}")
        return ChatSession.model_validate(response)

    def update(self, session_id: str, title: Optional[str] = None) -> ChatSession:
        """Update a session (e.g., rename)."""
        data = {}
        if title is not None:
            data["title"] = title
        response = self._patch(f"/api/sessions/{session_id}", json_data=data)
        return ChatSession.model_validate(response)

    def delete(self, session_id: str) -> Dict[str, str]:
        """Delete a session and all its messages."""
        return self._delete(f"/api/sessions/{session_id}")

    def clear_messages(self, session_id: str) -> Dict[str, str]:
        """Clear all messages from a session."""
        return self._delete(f"/api/sessions/{session_id}/messages")


class AsyncSessions(AsyncResource):
    """Asynchronous sessions resource."""

    async def create(
        self, agent_id: Optional[str] = None, team_id: Optional[str] = None, **kwargs
    ) -> ChatSession:
        """Create a new chat session."""
        data = {}
        if agent_id:
            data["agent_id"] = agent_id
        if team_id:
            data["team_id"] = team_id
        response = await self._post("/api/sessions/", json_data=data)
        return ChatSession.model_validate(response)

    async def list(self, skip: int = 0, limit: int = 100, **filters) -> List[ChatSession]:
        """List chat sessions with optional filtering."""
        params = {"skip": skip, "limit": limit, **filters}
        response = await self._get("/api/sessions/", params=params)
        return [ChatSession.model_validate(s) for s in response]

    async def get(self, session_id: str) -> ChatSession:
        """Get a specific session by ID."""
        response = await self._get(f"/api/sessions/{session_id}")
        return ChatSession.model_validate(response)

    async def update(self, session_id: str, title: Optional[str] = None) -> ChatSession:
        """Update a session (e.g., rename)."""
        data = {}
        if title is not None:
            data["title"] = title
        response = await self._patch(f"/api/sessions/{session_id}", json_data=data)
        return ChatSession.model_validate(response)

    async def delete(self, session_id: str) -> Dict[str, str]:
        """Delete a session and all its messages."""
        return await self._delete(f"/api/sessions/{session_id}")

    async def clear_messages(self, session_id: str) -> Dict[str, str]:
        """Clear all messages from a session."""
        return await self._delete(f"/api/sessions/{session_id}/messages")
