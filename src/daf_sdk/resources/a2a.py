"""A2A resource - Agent-to-Agent protocol management."""

from typing import Dict

from .._base import AsyncResource, SyncResource
from ..models import A2AKeyInfo, AgentCard


class A2A(SyncResource):
    """Synchronous A2A resource."""

    def get_agent_card(self, agent_id: str) -> AgentCard:
        """Get Agent Card for an agent."""
        response = self._get(f"/api/agents/{agent_id}/a2a/.well-known/agent.json")
        return AgentCard.model_validate(response)

    def get_agent_key(self, agent_id: str) -> A2AKeyInfo:
        """Get A2A API key info for an agent."""
        response = self._get(f"/api/agents/{agent_id}/a2a-key")
        return A2AKeyInfo.model_validate(response)

    def generate_agent_key(self, agent_id: str) -> Dict[str, str]:
        """Generate a new A2A API key for an agent."""
        return self._post(f"/api/agents/{agent_id}/a2a-key")

    def revoke_agent_key(self, agent_id: str) -> Dict[str, str]:
        """Revoke A2A API key for an agent."""
        return self._delete(f"/api/agents/{agent_id}/a2a-key")

    def get_team_card(self, team_id: str) -> AgentCard:
        """Get Agent Card for a team."""
        response = self._get(f"/api/teams/a2a/{team_id}/.well-known/agent.json")
        return AgentCard.model_validate(response)

    def generate_team_key(self, team_id: str) -> Dict[str, str]:
        """Generate a new A2A API key for a team."""
        return self._post(f"/api/teams/a2a/{team_id}/generate-key")

    def revoke_team_key(self, team_id: str) -> Dict[str, str]:
        """Revoke A2A API key for a team."""
        return self._delete(f"/api/teams/a2a/{team_id}/revoke-key")


class AsyncA2A(AsyncResource):
    """Asynchronous A2A resource."""

    async def get_agent_card(self, agent_id: str) -> AgentCard:
        """Get Agent Card for an agent."""
        response = await self._get(f"/api/agents/{agent_id}/a2a/.well-known/agent.json")
        return AgentCard.model_validate(response)

    async def get_agent_key(self, agent_id: str) -> A2AKeyInfo:
        """Get A2A API key info for an agent."""
        response = await self._get(f"/api/agents/{agent_id}/a2a-key")
        return A2AKeyInfo.model_validate(response)

    async def generate_agent_key(self, agent_id: str) -> Dict[str, str]:
        """Generate a new A2A API key for an agent."""
        return await self._post(f"/api/agents/{agent_id}/a2a-key")

    async def revoke_agent_key(self, agent_id: str) -> Dict[str, str]:
        """Revoke A2A API key for an agent."""
        return await self._delete(f"/api/agents/{agent_id}/a2a-key")

    async def get_team_card(self, team_id: str) -> AgentCard:
        """Get Agent Card for a team."""
        response = await self._get(f"/api/teams/a2a/{team_id}/.well-known/agent.json")
        return AgentCard.model_validate(response)

    async def generate_team_key(self, team_id: str) -> Dict[str, str]:
        """Generate a new A2A API key for a team."""
        return await self._post(f"/api/teams/a2a/{team_id}/generate-key")

    async def revoke_team_key(self, team_id: str) -> Dict[str, str]:
        """Revoke A2A API key for a team."""
        return await self._delete(f"/api/teams/a2a/{team_id}/revoke-key")
