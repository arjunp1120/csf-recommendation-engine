"""
DAF SDK - Python SDK for Declarative Agentic Framework
======================================================

A modern Python SDK for interacting with DAF API.

Quick Start:
    from daf_sdk import DAF

    client = DAF(
        base_url="http://localhost:8012",
        api_key="daf_your_api_key"
    )

    # Create an agent
    agent = client.agents.create(name="my_agent")

    # Send a message
    response = client.agents.messages.send(
        agent_id=agent.id,
        message="Hello!"
    )
    print(response.response)

Async Support:
    from daf_sdk import AsyncDAF

    async with AsyncDAF(base_url="...", api_key="...") as client:
        agents = await client.agents.list()

Custom Tools:
    from daf_sdk import DAF, custom_tool

    @custom_tool
    def my_tool(arg: str) -> str:
        '''Tool description.'''
        return f"Result: {arg}"

    client = DAF(...)
    client.tools.register(my_tool)
"""

__version__ = "0.2.0"
__author__ = "DAF Team"

# Main clients
from .client import DAF, AsyncDAF

# Decorator for custom tools
from .decorators import ToolFunction, custom_tool

# Exceptions
from .exceptions import (
    APIError,
    AuthenticationError,
    ConnectionError,
    DAFError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
    TimeoutError,
    ValidationError,
)

# Models (commonly used)
from .models import (
    Agent,
    AgentCreate,
    AgentUpdate,
    ChatSession,
    CustomTool,
    ExecutionResponse,
    MemoryBlock,
    Team,
    TeamCreate,
    TeamExecutionResponse,
    Trigger,
)

__all__ = [
    # Version
    "__version__",
    # Clients
    "DAF",
    "AsyncDAF",
    # Exceptions
    "DAFError",
    "APIError",
    "AuthenticationError",
    "NotFoundError",
    "ValidationError",
    "RateLimitError",
    "InternalServerError",
    "ConnectionError",
    "TimeoutError",
    # Models
    "Agent",
    "AgentCreate",
    "AgentUpdate",
    "Team",
    "TeamCreate",
    "ChatSession",
    "ExecutionResponse",
    "TeamExecutionResponse",
    "MemoryBlock",
    "Trigger",
    "CustomTool",
    # Decorators
    "custom_tool",
    "ToolFunction",
]
