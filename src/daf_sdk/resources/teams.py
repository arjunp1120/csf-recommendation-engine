"""
DAF SDK Teams Resource

CRUD operations and execution for teams (workflows).
"""

import json
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Generator, List, Optional

from .._base import AsyncResource, SyncResource, parse_list_response, parse_response
from ..models import Team

if TYPE_CHECKING:
    from .._http import AsyncHTTPClient, HTTPClient


class TeamExecution(SyncResource):
    """Resource for team execution operations."""

    def stream(
        self,
        team_id: str,
        message: str,
        session_id: Optional[str] = None,
        resume_from_hil_node_id: Optional[str] = None,
        hil_choice_value: Optional[str] = None,
        hil_choice_label: Optional[str] = None,
        hil_input: Optional[str] = None,
        **kwargs,
    ) -> Generator[Dict[str, Any], None, None]:
        """Execute a team workflow with streaming."""
        payload = {"team_id": team_id, "message": message}
        if session_id:
            payload["session_id"] = session_id
        if resume_from_hil_node_id:
            payload["resume_from_hil_node_id"] = resume_from_hil_node_id
        if hil_choice_value:
            payload["hil_choice_value"] = hil_choice_value
        if hil_choice_label:
            payload["hil_choice_label"] = hil_choice_label
        if hil_input:
            payload["hil_input"] = hil_input

        for line in self._client.stream("POST", "/api/teams/execute/stream", json_data=payload):
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    yield json.loads(data_str)
                except json.JSONDecodeError:
                    yield {"type": "text", "content": data_str}


class AsyncTeamExecution(AsyncResource):
    """Async resource for team execution."""

    async def stream(
        self, team_id: str, message: str, session_id: Optional[str] = None, **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute a team workflow with streaming."""
        payload = {"team_id": team_id, "message": message}
        if session_id:
            payload["session_id"] = session_id

        async for line in self._client.stream(
            "POST", "/api/teams/execute/stream", json_data=payload
        ):
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    yield json.loads(data_str)
                except json.JSONDecodeError:
                    yield {"type": "text", "content": data_str}


class TeamA2A(SyncResource):
    """Resource for team A2A operations."""

    def get_key(self, team_id: str) -> Dict[str, Any]:
        """Get A2A API key info for a team."""
        return self._get(f"/api/teams/{team_id}/a2a-key")

    def generate_key(self, team_id: str) -> Dict[str, str]:
        """Generate a new A2A API key for a team."""
        return self._post(f"/api/teams/a2a/{team_id}/generate-key")

    def revoke_key(self, team_id: str) -> Dict[str, str]:
        """Revoke the A2A API key."""
        return self._delete(f"/api/teams/a2a/{team_id}/revoke-key")

    def get_card(self, team_id: str) -> Dict[str, Any]:
        """Get A2A Agent Card for team."""
        return self._get(f"/api/teams/a2a/{team_id}/.well-known/agent.json")


class AsyncTeamA2A(AsyncResource):
    """Async resource for team A2A operations."""

    async def get_key(self, team_id: str) -> Dict[str, Any]:
        """Get A2A API key info for a team."""
        return await self._get(f"/api/teams/{team_id}/a2a-key")

    async def generate_key(self, team_id: str) -> Dict[str, str]:
        """Generate a new A2A API key for a team."""
        return await self._post(f"/api/teams/a2a/{team_id}/generate-key")

    async def revoke_key(self, team_id: str) -> Dict[str, str]:
        """Revoke the A2A API key."""
        return await self._delete(f"/api/teams/a2a/{team_id}/revoke-key")

    async def get_card(self, team_id: str) -> Dict[str, Any]:
        """Get A2A Agent Card for team."""
        return await self._get(f"/api/teams/a2a/{team_id}/.well-known/agent.json")


class Teams(SyncResource):
    """Sync resource for team operations."""

    def __init__(self, client: "HTTPClient"):
        super().__init__(client)
        self.execution = TeamExecution(client)
        self.a2a = TeamA2A(client)
        self.memory = TeamMemory(client)
        self.sessions = TeamSessions(client)

    def create(self, name: str, **kwargs) -> Team:
        """Create a new team."""
        payload = {"name": name}
        fields = ["description", "handoff_pattern", "nodes", "connections", "shared_memory_labels"]
        for f in fields:
            if f in kwargs and kwargs[f] is not None:
                payload[f] = kwargs[f]
        data = self._post("/api/teams/", json_data=payload)
        return parse_response(data, Team)

    def list(self, skip: int = 0, limit: int = 100, **kwargs) -> List[Team]:
        """List all teams."""
        data = self._get("/api/teams/", params={"skip": skip, "limit": limit})
        return parse_list_response(data, Team)

    def get(self, team_id: str, **kwargs) -> Team:
        """Get a team by ID."""
        data = self._get(f"/api/teams/{team_id}")
        return parse_response(data, Team)

    def update(self, team_id: str, **kwargs) -> Team:
        """Update a team."""
        fields = [
            "name",
            "description",
            "handoff_pattern",
            "nodes",
            "connections",
            "shared_memory_labels",
        ]
        payload = {k: v for k, v in kwargs.items() if k in fields and v is not None}
        data = self._put(f"/api/teams/{team_id}", json_data=payload)
        return parse_response(data, Team)

    def delete(self, team_id: str, **kwargs) -> Dict[str, Any]:
        """Delete a team."""
        return self._delete(f"/api/teams/{team_id}")

    def execute(self, team_id: str, message: str, session_id: Optional[str] = None, **kwargs):
        """Execute team workflow with streaming."""
        return self.execution.stream(team_id, message, session_id, **kwargs)

    # =========================================================================
    # Export/Import Methods
    # =========================================================================

    def export(self, team_id: str, include_agents: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Export a team configuration as JSON.

        Args:
            team_id: Team ID to export
            include_agents: Include agent configurations (default: True)

        Returns:
            Export data dict with team configuration

        Example:
            export_data = client.teams.export("team-123")
            with open("team.json", "w") as f:
                json.dump(export_data, f, indent=2)
        """
        params = {"include_agents": str(include_agents).lower()}
        return self._get(f"/api/teams/{team_id}/export", params=params, **kwargs)

    def import_team(
        self,
        config: Dict[str, Any],
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Import a team from exported configuration.

        Supports multiple formats:
        1. Frontend format: {"type": "team", "version": "2.0", "data": {...}}
        2. Backend export format: {"export_type": "team", "team": {...}, ...}
        3. Raw team data: {"name": "...", "nodes": [...], ...}

        Args:
            config: Export configuration dict
            api_key: API key for LLM provider (for agents)
            azure_endpoint: Azure endpoint
            azure_deployment: Azure deployment
            api_version: Azure API version

        Returns:
            Created team dict (full team data with id)

        Example:
            with open("team.json") as f:
                config = json.load(f)
            result = client.teams.import_team(config)
            print(f"Imported team: {result['id']}")
        """
        # Extract team data from various formats
        if config.get("type") == "team" and "data" in config:
            # Frontend format: {"type": "team", "data": {...}}
            team_data = config["data"].copy()
        elif config.get("export_type") == "team" and "team" in config:
            # Backend export format: {"export_type": "team", "team": {...}}
            team_data = config["team"].copy()
            # Merge additional fields
            if "nodes" in config:
                team_data["nodes"] = config["nodes"]
            if "connections" in config:
                team_data["connections"] = config["connections"]
        else:
            # Assume raw team data
            team_data = config.copy()

        # Remove fields that shouldn't be sent
        team_data.pop("id", None)
        team_data.pop("type", None)
        team_data.pop("created_at", None)
        team_data.pop("updated_at", None)

        return self._post("/api/teams/import", json_data=team_data, **kwargs)

    def create_from_template(
        self,
        template_path: str,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        name_override: Optional[str] = None,
        **kwargs,
    ) -> Team:
        """
        Create a team from a JSON template file.

        Args:
            template_path: Path to JSON template file
            api_key: API key for LLM provider
            azure_endpoint: Azure endpoint
            azure_deployment: Azure deployment
            api_version: Azure API version
            name_override: Override the name from template

        Returns:
            Created Team

        Example:
            team = client.teams.create_from_template(
                "templates/brainstorm_team.json",
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT")
            )
        """
        with open(template_path, "r") as f:
            config = json.load(f)

        if name_override:
            if "data" in config:
                config["data"]["name"] = name_override
            elif "team" in config:
                config["team"]["name"] = name_override

        result = self.import_team(
            config,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            **kwargs,
        )

        return self.get(result["team_id"])


class AsyncTeams(AsyncResource):
    """Async resource for team operations."""

    def __init__(self, client: "AsyncHTTPClient"):
        super().__init__(client)
        self.execution = AsyncTeamExecution(client)
        self.a2a = AsyncTeamA2A(client)
        self.memory = AsyncTeamMemory(client)
        self.sessions = AsyncTeamSessions(client)

    async def create(self, name: str, **kwargs) -> Team:
        """Create a new team."""
        payload = {"name": name}
        fields = ["description", "handoff_pattern", "nodes", "connections", "shared_memory_labels"]
        for f in fields:
            if f in kwargs and kwargs[f] is not None:
                payload[f] = kwargs[f]
        data = await self._post("/api/teams/", json_data=payload)
        return parse_response(data, Team)

    async def list(self, skip: int = 0, limit: int = 100, **kwargs) -> List[Team]:
        """List all teams."""
        data = await self._get("/api/teams/", params={"skip": skip, "limit": limit})
        return parse_list_response(data, Team)

    async def get(self, team_id: str, **kwargs) -> Team:
        """Get a team by ID."""
        data = await self._get(f"/api/teams/{team_id}")
        return parse_response(data, Team)

    async def update(self, team_id: str, **kwargs) -> Team:
        """Update a team."""
        fields = [
            "name",
            "description",
            "handoff_pattern",
            "nodes",
            "connections",
            "shared_memory_labels",
        ]
        payload = {k: v for k, v in kwargs.items() if k in fields and v is not None}
        data = await self._put(f"/api/teams/{team_id}", json_data=payload)
        return parse_response(data, Team)

    async def delete(self, team_id: str, **kwargs) -> Dict[str, Any]:
        """Delete a team."""
        return await self._delete(f"/api/teams/{team_id}")

    async def execute(self, team_id: str, message: str, session_id: Optional[str] = None, **kwargs):
        """Execute team workflow with streaming."""
        async for chunk in self.execution.stream(team_id, message, session_id, **kwargs):
            yield chunk


# =============================================================================
# Team Sessions Resource
# =============================================================================


class TeamSessions(SyncResource):
    """Team sessions management."""

    def list(self, team_id: str) -> List[Dict[str, Any]]:
        """
        List all sessions for a team.

        Args:
            team_id: Team ID

        Returns:
            List of session dicts
        """
        return self._get(f"/api/teams/{team_id}/sessions")

    def get(self, team_id: str, session_id: str) -> Dict[str, Any]:
        """
        Get a specific team session.

        Args:
            team_id: Team ID
            session_id: Session ID

        Returns:
            Session dict with messages
        """
        return self._get(f"/api/teams/{team_id}/sessions/{session_id}")

    def delete(self, team_id: str, session_id: str) -> Dict[str, Any]:
        """
        Delete a team session.

        Args:
            team_id: Team ID
            session_id: Session ID

        Returns:
            Deletion result
        """
        return self._delete(f"/api/teams/{team_id}/sessions/{session_id}")

    def clear_messages(self, team_id: str, session_id: str) -> Dict[str, Any]:
        """
        Clear all messages in a team session.

        Args:
            team_id: Team ID
            session_id: Session ID

        Returns:
            Result dict
        """
        return self._delete(f"/api/teams/{team_id}/sessions/{session_id}/messages")


class AsyncTeamSessions(AsyncResource):
    """Async team sessions management."""

    async def list(self, team_id: str) -> List[Dict[str, Any]]:
        """List all sessions for a team."""
        return await self._get(f"/api/teams/{team_id}/sessions")

    async def get(self, team_id: str, session_id: str) -> Dict[str, Any]:
        """Get a specific team session."""
        return await self._get(f"/api/teams/{team_id}/sessions/{session_id}")

    async def delete(self, team_id: str, session_id: str) -> Dict[str, Any]:
        """Delete a team session."""
        return await self._delete(f"/api/teams/{team_id}/sessions/{session_id}")

    async def clear_messages(self, team_id: str, session_id: str) -> Dict[str, Any]:
        """Clear all messages in a team session."""
        return await self._delete(f"/api/teams/{team_id}/sessions/{session_id}/messages")


# =============================================================================
# Team Memory Resource
# =============================================================================


class TeamMemory(SyncResource):
    """Team shared memory management."""

    def list(self, team_id: str) -> List[Dict[str, Any]]:
        """
        List team's shared memory blocks.

        Args:
            team_id: Team ID

        Returns:
            List of memory blocks
        """
        return self._get(f"/api/teams/{team_id}/shared-memory")

    def create(
        self, team_id: str, label: str, value: str, description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a shared memory block to a team.

        Args:
            team_id: Team ID
            label: Memory label
            value: Memory value
            description: Optional description

        Returns:
            Created memory block
        """
        data = {"label": label, "value": value}
        if description:
            data["description"] = description
        return self._post(f"/api/teams/{team_id}/shared-memory", json_data=data)

    def delete(self, team_id: str, label: str) -> Dict[str, Any]:
        """
        Delete a shared memory block from a team.

        Args:
            team_id: Team ID
            label: Memory label to delete

        Returns:
            Deletion result
        """
        from urllib.parse import quote

        encoded_label = quote(label, safe="")
        return self._delete(f"/api/teams/{team_id}/shared-memory/{encoded_label}")


class AsyncTeamMemory(AsyncResource):
    """Async team shared memory management."""

    async def list(self, team_id: str) -> List[Dict[str, Any]]:
        """List team's shared memory blocks."""
        return await self._get(f"/api/teams/{team_id}/shared-memory")

    async def create(
        self, team_id: str, label: str, value: str, description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add a shared memory block to a team."""
        data = {"label": label, "value": value}
        if description:
            data["description"] = description
        return await self._post(f"/api/teams/{team_id}/shared-memory", json_data=data)

    async def delete(self, team_id: str, label: str) -> Dict[str, Any]:
        """Delete a shared memory block from a team."""
        from urllib.parse import quote

        encoded_label = quote(label, safe="")
        return await self._delete(f"/api/teams/{team_id}/shared-memory/{encoded_label}")
