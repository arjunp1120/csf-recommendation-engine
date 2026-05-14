"""
DAF SDK Client

Main entry points for the DAF SDK: DAF (sync) and AsyncDAF (async) clients.
"""

from typing import Dict, Optional

from ._http import AsyncHTTPClient, HTTPClient
from .auth import AsyncAuth, Auth
from .resources.a2a import A2A, AsyncA2A
from .resources.agents import Agents, AsyncAgents
from .resources.analytics import Analytics, AsyncAnalytics
from .resources.llm_endpoints import AsyncLLMEndpoints, LLMEndpoints
from .resources.mcp import MCP, AsyncMCP
from .resources.memory import AsyncMemory, Memory
from .resources.sessions import AsyncSessions, Sessions
from .resources.teams import AsyncTeams, Teams
from .resources.tools import AsyncTools, Tools
from .resources.triggers import AsyncTriggers, Triggers


class DAF:
    """
    Synchronous client for the DAF (Declarative Agentic Framework) API.

    Usage:
        from daf_sdk import DAF

        # With API key (recommended)
        client = DAF(base_url="http://localhost:8012", api_key="daf_...")

        # With token (if you already have one)
        client = DAF(base_url="http://localhost:8012", token="eyJ...")

        # List agents
        agents = client.agents.list()

        # Create an agent
        agent = client.agents.create(name="my_agent")

        # Send a message
        response = client.agents.messages.send(
            agent_id=agent.id,
            message="Hello!"
        )

        # Close when done
        client.close()

    Context manager usage:
        with DAF(base_url="...", api_key="...") as client:
            agents = client.agents.list()
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        token: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the DAF client.

        Args:
            base_url: Base URL of the DAF API
            token: JWT authentication token
            api_key: API key for authentication (starts with 'daf_')
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
            headers: Additional headers to include in requests
        """
        # Use api_key as token if provided (backend accepts both)
        auth_token = api_key or token

        self._http = HTTPClient(
            base_url=base_url,
            token=auth_token,
            timeout=timeout,
            max_retries=max_retries,
            headers=headers,
        )

        # Initialize resources
        self.auth = Auth(self._http, parent_client=self)
        self.agents = Agents(self._http)
        self.teams = Teams(self._http)
        self.sessions = Sessions(self._http)
        self.triggers = Triggers(self._http)
        self.analytics = Analytics(self._http)
        self.memory = Memory(self._http)
        self.tools = Tools(self._http)
        self.mcp = MCP(self._http)
        self.a2a = A2A(self._http)
        self.llm_endpoints = LLMEndpoints(self._http)

    @property
    def base_url(self) -> str:
        """Get the base URL."""
        return self._http.base_url

    def set_token(self, token: str) -> None:
        """Set or update the authentication token."""
        self._http.set_auth_token(token)

    def close(self) -> None:
        """Close the HTTP client."""
        self._http.close()

    def __enter__(self) -> "DAF":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def health_check(self) -> bool:
        """
        Check if the server is healthy.

        Returns:
            True if server is responding
        """
        try:
            self._http.get("/health")
            return True
        except Exception:
            return False


class AsyncDAF:
    """
    Asynchronous client for the DAF API.

    Usage:
        from daf_sdk import AsyncDAF
        import asyncio

        async def main():
            async with AsyncDAF(base_url="...", api_key="...") as client:
                agents = await client.agents.list()

                agent = await client.agents.create(name="my_agent")

                response = await client.agents.messages.send(
                    agent_id=agent.id,
                    message="Hello!"
                )

        asyncio.run(main())
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        token: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the async DAF client.

        Args:
            base_url: Base URL of the DAF API
            token: JWT authentication token
            api_key: API key for authentication (starts with 'daf_')
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts for failed requests
            headers: Additional headers to include in requests
        """
        auth_token = api_key or token

        self._http = AsyncHTTPClient(
            base_url=base_url,
            token=auth_token,
            timeout=timeout,
            max_retries=max_retries,
            headers=headers,
        )

        # Initialize resources
        self.auth = AsyncAuth(self._http, parent_client=self)
        self.agents = AsyncAgents(self._http)
        self.teams = AsyncTeams(self._http)
        self.sessions = AsyncSessions(self._http)
        self.triggers = AsyncTriggers(self._http)
        self.analytics = AsyncAnalytics(self._http)
        self.memory = AsyncMemory(self._http)
        self.tools = AsyncTools(self._http)
        self.mcp = AsyncMCP(self._http)
        self.a2a = AsyncA2A(self._http)
        self.llm_endpoints = AsyncLLMEndpoints(self._http)

    @property
    def base_url(self) -> str:
        """Get the base URL."""
        return self._http.base_url

    def set_token(self, token: str) -> None:
        """Set or update the authentication token."""
        self._http.set_auth_token(token)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http.close()

    async def __aenter__(self) -> "AsyncDAF":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def health_check(self) -> bool:
        """Check if the server is healthy."""
        try:
            await self._http.get("/health")
            return True
        except Exception:
            return False


# Aliases for backward compatibility
Client = DAF
AsyncClient = AsyncDAF
