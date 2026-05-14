"""
DAF SDK Agents Resource

CRUD operations for agents.
"""

import json
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Generator, List, Optional

from .._base import AsyncResource, SyncResource, parse_list_response, parse_response
from ..models import Agent, ExecutionResponse

if TYPE_CHECKING:
    from .._http import AsyncHTTPClient, HTTPClient


class AgentMessages(SyncResource):
    """Resource for sending messages to agents."""

    def send(
        self,
        agent_id: str,
        message: str,
        session_id: Optional[str] = None,
        approved_tools: Optional[List[str]] = None,
        **kwargs,
    ) -> ExecutionResponse:
        """
        Send a message to an agent and get a response.

        Args:
            agent_id: ID of the agent
            message: Message content
            session_id: Optional session ID for conversation continuity
            approved_tools: Tools pre-approved for execution

        Returns:
            ExecutionResponse with agent's response
        """
        payload: Dict[str, Any] = {"agent_id": agent_id, "message": message}
        if session_id:
            payload["session_id"] = session_id
        if approved_tools:
            payload["approved_tools"] = approved_tools

        data = self._post("/api/execute/agent", json_data=payload, **kwargs)
        return parse_response(data, ExecutionResponse)

    def stream(
        self,
        agent_id: str,
        message: str,
        session_id: Optional[str] = None,
        approved_tools: Optional[List[str]] = None,
        **kwargs,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Send a message and stream the response.

        Args:
            agent_id: ID of the agent
            message: Message content
            session_id: Optional session ID
            approved_tools: Tools pre-approved for execution

        Yields:
            Stream chunks with type and content
        """
        payload: Dict[str, Any] = {"agent_id": agent_id, "message": message}
        if session_id:
            payload["session_id"] = session_id
        if approved_tools:
            payload["approved_tools"] = approved_tools

        for line in self._stream("POST", "/api/execute/agent/stream", json_data=payload):
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    yield json.loads(data_str)
                except json.JSONDecodeError:
                    pass


class AsyncAgentMessages(AsyncResource):
    """Async resource for sending messages to agents."""

    async def send(
        self,
        agent_id: str,
        message: str,
        session_id: Optional[str] = None,
        approved_tools: Optional[List[str]] = None,
        **kwargs,
    ) -> ExecutionResponse:
        """Send a message to an agent and get a response."""
        payload: Dict[str, Any] = {"agent_id": agent_id, "message": message}
        if session_id:
            payload["session_id"] = session_id
        if approved_tools:
            payload["approved_tools"] = approved_tools

        data = await self._post("/api/execute/agent", json_data=payload, **kwargs)
        return parse_response(data, ExecutionResponse)

    async def stream(
        self,
        agent_id: str,
        message: str,
        session_id: Optional[str] = None,
        approved_tools: Optional[List[str]] = None,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Send a message and stream the response."""
        payload: Dict[str, Any] = {"agent_id": agent_id, "message": message}
        if session_id:
            payload["session_id"] = session_id
        if approved_tools:
            payload["approved_tools"] = approved_tools

        async for line in self._stream("POST", "/api/execute/agent/stream", json_data=payload):
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    yield json.loads(data_str)
                except json.JSONDecodeError:
                    pass


class AgentMemory(SyncResource):
    """Resource for agent memory operations."""

    def list(self, agent_id: str, **kwargs) -> List[Dict[str, Any]]:
        """
        List memory blocks for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            List of memory blocks
        """
        return self._get(f"/api/agents/{agent_id}/memory", **kwargs)

    def get(self, agent_id: str, label: str, **kwargs) -> Dict[str, Any]:
        """
        Get a specific memory block.

        Args:
            agent_id: Agent ID
            label: Memory block label

        Returns:
            Memory block data
        """
        return self._get(f"/api/agents/{agent_id}/memory/{label}", **kwargs)

    def create(
        self, agent_id: str, label: str, value: str, description: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Create a memory block for an agent.

        Args:
            agent_id: Agent ID
            label: Unique label
            value: Block content
            description: Optional description

        Returns:
            Created memory block
        """
        payload = {"label": label, "value": value}
        if description:
            payload["description"] = description

        return self._post(f"/api/agents/{agent_id}/memory", json_data=payload, **kwargs)

    def update(
        self,
        agent_id: str,
        label: str,
        value: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Update a memory block.

        Args:
            agent_id: Agent ID
            label: Memory block label
            value: New value
            description: New description

        Returns:
            Updated memory block
        """
        payload = {}
        if value is not None:
            payload["value"] = value
        if description is not None:
            payload["description"] = description

        return self._patch(f"/api/agents/{agent_id}/memory/{label}", json_data=payload, **kwargs)

    def delete(self, agent_id: str, label: str, **kwargs) -> Dict[str, Any]:
        """
        Delete a memory block.

        Args:
            agent_id: Agent ID
            label: Memory block label

        Returns:
            Deletion confirmation
        """
        return self._delete(f"/api/agents/{agent_id}/memory/{label}", **kwargs)

    def search(
        self, agent_id: str, query: str, search_in: str = "all", **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search memory blocks for an agent.

        Args:
            agent_id: Agent ID
            query: Search query string
            search_in: Where to search - 'all', 'label', 'value', or 'description'

        Returns:
            List of matching memory blocks
        """
        params = {"query": query, "search_in": search_in}
        return self._get(f"/api/agents/{agent_id}/memory/search", params=params, **kwargs)


class AsyncAgentMemory(AsyncResource):
    """Async resource for agent memory operations."""

    async def list(self, agent_id: str, **kwargs) -> List[Dict[str, Any]]:
        """List memory blocks for an agent."""
        return await self._get(f"/api/agents/{agent_id}/memory", **kwargs)

    async def get(self, agent_id: str, label: str, **kwargs) -> Dict[str, Any]:
        """Get a specific memory block."""
        return await self._get(f"/api/agents/{agent_id}/memory/{label}", **kwargs)

    async def create(
        self, agent_id: str, label: str, value: str, description: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Create a memory block for an agent."""
        payload = {"label": label, "value": value}
        if description:
            payload["description"] = description

        return await self._post(f"/api/agents/{agent_id}/memory", json_data=payload, **kwargs)

    async def update(
        self,
        agent_id: str,
        label: str,
        value: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Update a memory block."""
        payload = {}
        if value is not None:
            payload["value"] = value
        if description is not None:
            payload["description"] = description

        return await self._patch(
            f"/api/agents/{agent_id}/memory/{label}", json_data=payload, **kwargs
        )

    async def delete(self, agent_id: str, label: str, **kwargs) -> Dict[str, Any]:
        """Delete a memory block."""
        return await self._delete(f"/api/agents/{agent_id}/memory/{label}", **kwargs)

    async def search(
        self, agent_id: str, query: str, search_in: str = "all", **kwargs
    ) -> List[Dict[str, Any]]:
        """Search memory blocks for an agent."""
        params = {"query": query, "search_in": search_in}
        return await self._get(f"/api/agents/{agent_id}/memory/search", params=params, **kwargs)


class AgentA2A(SyncResource):
    """Resource for agent A2A (Agent-to-Agent) operations."""

    def get_key(self, agent_id: str, **kwargs) -> Dict[str, Any]:
        """
        Get A2A API key info for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Key info with has_key, key_preview, full_key
        """
        return self._get(f"/api/agents/{agent_id}/a2a-key", **kwargs)

    def generate_key(self, agent_id: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a new A2A API key.

        Args:
            agent_id: Agent ID

        Returns:
            New API key
        """
        return self._post(f"/api/agents/{agent_id}/a2a-key/generate", **kwargs)

    def revoke_key(self, agent_id: str, **kwargs) -> Dict[str, Any]:
        """
        Revoke the A2A API key.

        Args:
            agent_id: Agent ID

        Returns:
            Revocation confirmation
        """
        return self._delete(f"/api/agents/{agent_id}/a2a-key", **kwargs)

    def get_card(self, agent_id: str, **kwargs) -> Dict[str, Any]:
        """
        Get A2A Agent Card.

        Args:
            agent_id: Agent ID

        Returns:
            Agent card (Google A2A protocol format)
        """
        return self._get(f"/api/agents/{agent_id}/.well-known/agent.json", **kwargs)


class AsyncAgentA2A(AsyncResource):
    """Async resource for agent A2A operations."""

    async def get_key(self, agent_id: str, **kwargs) -> Dict[str, Any]:
        """Get A2A API key info for an agent."""
        return await self._get(f"/api/agents/{agent_id}/a2a-key", **kwargs)

    async def generate_key(self, agent_id: str, **kwargs) -> Dict[str, Any]:
        """Generate a new A2A API key."""
        return await self._post(f"/api/agents/{agent_id}/a2a-key/generate", **kwargs)

    async def revoke_key(self, agent_id: str, **kwargs) -> Dict[str, Any]:
        """Revoke the A2A API key."""
        return await self._delete(f"/api/agents/{agent_id}/a2a-key", **kwargs)

    async def get_card(self, agent_id: str, **kwargs) -> Dict[str, Any]:
        """Get A2A Agent Card."""
        return await self._get(f"/api/agents/{agent_id}/.well-known/agent.json", **kwargs)


class Agents(SyncResource):
    """
    Resource for agent operations.

    Usage:
        # Create agent
        agent = client.agents.create(
            name="my-agent",
            system_instructions="You are helpful.",
            model_provider="azure",
            model_name="gpt-4o",
            api_key="..."
        )

        # List agents
        agents = client.agents.list()

        # Get agent
        agent = client.agents.get(agent_id)

        # Update agent
        agent = client.agents.update(agent_id, name="new-name")

        # Delete agent
        client.agents.delete(agent_id)

        # Send message
        response = client.agents.messages.send(agent_id, "Hello!")

        # Manage memory
        client.agents.memory.create(agent_id, label="persona", value="...")
    """

    def __init__(self, client: "HTTPClient"):
        super().__init__(client)
        self.messages = AgentMessages(client)
        self.memory = AgentMemory(client)
        self.a2a = AgentA2A(client)

    def create(
        self,
        name: str,
        system_instructions: Optional[str] = None,
        llm_endpoint_id: Optional[str] = None,
        model_provider: str = "azure",
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: Optional[str] = "0.7",
        max_tokens: Optional[int] = 4096,
        tools: Optional[List[str]] = None,
        tools_requiring_approval: Optional[List[str]] = None,
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        role: str = "worker",
        persistent_context: Optional[str] = None,
        auto_inject_memory: Optional[bool] = None,
        **kwargs,
    ) -> Agent:
        """
        Create a new agent.

        Args:
            name: Unique name for the agent
            system_instructions: System prompt
            model_provider: LLM provider (azure, openai, anthropic)
            model_name: Model name (gpt-4o, etc.)
            api_key: API key for the provider
            azure_endpoint: Azure OpenAI endpoint
            azure_deployment: Azure deployment name
            api_version: API version
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            tools: List of tool IDs to attach
            tools_requiring_approval: Tools that need human approval
            mcp_servers: MCP server configurations
            role: Agent role (worker, supervisor, coordinator)
            persistent_context: Static context always included in prompt
            auto_inject_memory: Auto-inject all memory blocks to prompt

        Returns:
            Created Agent object
        """
        payload: Dict[str, Any] = {
            "name": name,
            "model_provider": model_provider,
            "model_name": model_name,
            "role": role,
        }

        if system_instructions is not None:
            payload["system_instructions"] = system_instructions
        if llm_endpoint_id is not None:
            payload["llm_endpoint_id"] = llm_endpoint_id
        if api_key is not None:
            payload["api_key"] = api_key
        if azure_endpoint is not None:
            payload["azure_endpoint"] = azure_endpoint
        if azure_deployment is not None:
            payload["azure_deployment"] = azure_deployment
        if api_version is not None:
            payload["api_version"] = api_version
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools is not None:
            payload["tools"] = tools
        if tools_requiring_approval is not None:
            payload["tools_requiring_approval"] = tools_requiring_approval
        if mcp_servers is not None:
            payload["mcp_servers"] = mcp_servers
        if persistent_context is not None:
            payload["persistent_context"] = persistent_context
        if auto_inject_memory is not None:
            payload["auto_inject_memory"] = auto_inject_memory

        data = self._post("/api/agents/", json_data=payload)
        return parse_response(data, Agent)

    def list(self, skip: int = 0, limit: int = 100, **kwargs) -> List[Agent]:
        """
        List all agents.

        Args:
            skip: Number of agents to skip
            limit: Maximum agents to return

        Returns:
            List of Agent objects
        """
        data = self._get("/api/agents/", params={"skip": skip, "limit": limit}, **kwargs)
        # return parse_list_response(data, Agent)
        return data

    def get(self, agent_id: str, **kwargs) -> Agent:
        """
        Get an agent by ID.

        Args:
            agent_id: Agent ID (UUID string)

        Returns:
            Agent object
        """
        data = self._get(f"/api/agents/{agent_id}", **kwargs)
        return parse_response(data, Agent)

    def update(
        self,
        agent_id: str,
        name: Optional[str] = None,
        system_instructions: Optional[str] = None,
        llm_endpoint_id: Optional[str] = None,
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: Optional[str] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[str]] = None,
        tools_requiring_approval: Optional[List[str]] = None,
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        role: Optional[str] = None,
        persistent_context: Optional[str] = None,
        auto_inject_memory: Optional[bool] = None,
        **kwargs,
    ) -> Agent:
        """
        Update an agent.

        Args:
            agent_id: Agent ID
            name: New name
            system_instructions: New system prompt
            persistent_context: Static context always included in prompt
            auto_inject_memory: Auto-inject all memory blocks to prompt
            **kwargs: Other fields to update

        Returns:
            Updated Agent object
        """
        payload: Dict[str, Any] = {}

        if name is not None:
            payload["name"] = name
        if system_instructions is not None:
            payload["system_instructions"] = system_instructions
        if llm_endpoint_id is not None:
            payload["llm_endpoint_id"] = llm_endpoint_id
        if model_provider is not None:
            payload["model_provider"] = model_provider
        if model_name is not None:
            payload["model_name"] = model_name
        if api_key is not None:
            payload["api_key"] = api_key
        if azure_endpoint is not None:
            payload["azure_endpoint"] = azure_endpoint
        if azure_deployment is not None:
            payload["azure_deployment"] = azure_deployment
        if api_version is not None:
            payload["api_version"] = api_version
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools is not None:
            payload["tools"] = tools
        if tools_requiring_approval is not None:
            payload["tools_requiring_approval"] = tools_requiring_approval
        if mcp_servers is not None:
            payload["mcp_servers"] = mcp_servers
        if role is not None:
            payload["role"] = role
        if persistent_context is not None:
            payload["persistent_context"] = persistent_context
        if auto_inject_memory is not None:
            payload["auto_inject_memory"] = auto_inject_memory

        data = self._put(f"/api/agents/{agent_id}", json_data=payload)
        return parse_response(data, Agent)

    def delete(self, agent_id: str, force: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Delete an agent.

        Args:
            agent_id: Agent ID
            force: Force delete even if used in teams

        Returns:
            Deletion confirmation
        """
        params = {}
        if force:
            params["force"] = "true"

        return self._delete(f"/api/agents/{agent_id}", params=params, **kwargs)

    # =========================================================================
    # Export/Import Methods
    # =========================================================================

    def export(self, agent_id: str, **kwargs) -> Dict[str, Any]:
        """
        Export an agent configuration as JSON.

        Args:
            agent_id: Agent ID to export

        Returns:
            Export data dict with agent configuration and memory

        Example:
            export_data = client.agents.export("agent-123")
            with open("agent.json", "w") as f:
                json.dump(export_data, f, indent=2)
        """
        return self._get(f"/api/agents/{agent_id}/export", **kwargs)

    def import_agent(
        self,
        config: Dict[str, Any],
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Import an agent from exported configuration.

        Supports two formats:
        1. Backend format: {"export_type": "agent", "agent": {...}, "memory": [...]}
        2. Frontend format: {"type": "agent", "version": "1.0", "data": {...}}

        Args:
            config: Export configuration dict (from file or export())
            api_key: API key for LLM provider
            azure_endpoint: Azure endpoint
            azure_deployment: Azure deployment
            api_version: Azure API version

        Returns:
            Import result with agent_id and name

        Example:
            with open("agent.json") as f:
                config = json.load(f)
            result = client.agents.import_agent(
                config,
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT")
            )
            print(f"Imported: {result['agent_id']}")
        """
        # Handle frontend format (type/version/data)
        if config.get("type") == "agent" and "data" in config:
            config = {
                "export_type": "agent",
                "agent": config["data"],
                "memory": config.get("memory", []),
            }

        # Add credentials
        if api_key:
            config["api_key"] = api_key
        if azure_endpoint:
            config["azure_endpoint"] = azure_endpoint
        if azure_deployment:
            config["azure_deployment"] = azure_deployment
        if api_version:
            config["api_version"] = api_version

        return self._post("/api/agents/import", json_data=config, **kwargs)

    def create_from_template(
        self,
        template_path: str,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        name_override: Optional[str] = None,
        **kwargs,
    ) -> Agent:
        """
        Create an agent from a JSON template file.

        Args:
            template_path: Path to JSON template file
            api_key: API key for LLM provider
            azure_endpoint: Azure endpoint
            azure_deployment: Azure deployment
            api_version: Azure API version
            name_override: Override the name from template

        Returns:
            Created Agent

        Example:
            agent = client.agents.create_from_template(
                "templates/coder.json",
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
            elif "agent" in config:
                config["agent"]["name"] = name_override

        result = self.import_agent(
            config,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            **kwargs,
        )

        return self.get(result["agent_id"])

    def create_from_dict(
        self,
        template: Dict[str, Any],
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        **kwargs,
    ) -> Agent:
        """
        Create an agent from a dictionary template.

        Args:
            template: Agent configuration dict
            api_key: API key for LLM provider
            azure_endpoint: Azure endpoint
            azure_deployment: Azure deployment
            api_version: Azure API version

        Returns:
            Created Agent

        Example:
            template = {
                "name": "My Agent",
                "system_instructions": "You are helpful.",
                "model_provider": "Azure",
                "model_name": "gpt-4o",
                "temperature": "0.7",
                "max_tokens": 1000
            }
            agent = client.agents.create_from_dict(
                template,
                api_key=os.getenv("AZURE_OPENAI_API_KEY")
            )
        """
        config = {"export_type": "agent", "agent": template, "memory": template.get("memory", [])}

        result = self.import_agent(
            config,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_version=api_version,
            **kwargs,
        )

        return self.get(result["agent_id"])

    # Convenience method - send message directly
    def send_message(
        self, agent_id: str, message: str, session_id: Optional[str] = None, **kwargs
    ) -> ExecutionResponse:
        """
        Send a message to an agent (convenience method).

        Args:
            agent_id: Agent ID
            message: Message content
            session_id: Optional session ID

        Returns:
            ExecutionResponse
        """
        return self.messages.send(agent_id, message, session_id, **kwargs)


class AsyncAgents(AsyncResource):
    """Async resource for agent operations."""

    def __init__(self, client: "AsyncHTTPClient"):
        super().__init__(client)
        self.messages = AsyncAgentMessages(client)
        self.memory = AsyncAgentMemory(client)
        self.a2a = AsyncAgentA2A(client)

    async def create(
        self,
        name: str,
        system_instructions: Optional[str] = None,
        llm_endpoint_id: Optional[str] = None,
        model_provider: str = "azure",
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: Optional[str] = "0.7",
        max_tokens: Optional[int] = 4096,
        tools: Optional[List[str]] = None,
        tools_requiring_approval: Optional[List[str]] = None,
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        role: str = "worker",
        **kwargs,
    ) -> Agent:
        """Create a new agent."""
        payload: Dict[str, Any] = {
            "name": name,
            "model_provider": model_provider,
            "model_name": model_name,
            "role": role,
        }

        if system_instructions is not None:
            payload["system_instructions"] = system_instructions
        if llm_endpoint_id is not None:
            payload["llm_endpoint_id"] = llm_endpoint_id
        if api_key is not None:
            payload["api_key"] = api_key
        if azure_endpoint is not None:
            payload["azure_endpoint"] = azure_endpoint
        if azure_deployment is not None:
            payload["azure_deployment"] = azure_deployment
        if api_version is not None:
            payload["api_version"] = api_version
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools is not None:
            payload["tools"] = tools
        if tools_requiring_approval is not None:
            payload["tools_requiring_approval"] = tools_requiring_approval
        if mcp_servers is not None:
            payload["mcp_servers"] = mcp_servers

        data = await self._post("/api/agents/", json_data=payload, **kwargs)
        return parse_response(data, Agent)

    async def list(self, skip: int = 0, limit: int = 100, **kwargs) -> List[Agent]:
        """List all agents."""
        data = await self._get("/api/agents/", params={"skip": skip, "limit": limit}, **kwargs)
        return parse_list_response(data, Agent)

    async def get(self, agent_id: str, **kwargs) -> Agent:
        """Get an agent by ID."""
        data = await self._get(f"/api/agents/{agent_id}", **kwargs)
        return parse_response(data, Agent)

    async def update(self, agent_id: str, **kwargs) -> Agent:
        """Update an agent."""
        # Extract known fields from kwargs
        fields = [
            "name",
            "system_instructions",
            "llm_endpoint_id",
            "model_provider",
            "model_name",
            "api_key",
            "azure_endpoint",
            "azure_deployment",
            "api_version",
            "temperature",
            "max_tokens",
            "tools",
            "tools_requiring_approval",
            "mcp_servers",
            "role",
        ]
        payload = {k: v for k, v in kwargs.items() if k in fields and v is not None}

        data = await self._put(f"/api/agents/{agent_id}", json_data=payload)
        return parse_response(data, Agent)

    async def delete(self, agent_id: str, force: bool = False, **kwargs) -> Dict[str, Any]:
        """Delete an agent."""
        params = {}
        if force:
            params["force"] = "true"

        return await self._delete(f"/api/agents/{agent_id}", params=params, **kwargs)

    async def send_message(
        self, agent_id: str, message: str, session_id: Optional[str] = None, **kwargs
    ) -> ExecutionResponse:
        """Send a message to an agent (convenience method)."""
        return await self.messages.send(agent_id, message, session_id, **kwargs)
