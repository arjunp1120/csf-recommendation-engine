"""
DAF SDK Types

Pydantic models for request/response data structures.
Updated for CosmosDB backend (string IDs).
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# ============================================================================
# Enums
# ============================================================================


class AgentStatus(str, Enum):
    """Agent status enumeration."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    DELETED = "deleted"


class AgentRole(str, Enum):
    """Agent role enumeration."""

    WORKER = "worker"
    SUPERVISOR = "supervisor"
    COORDINATOR = "coordinator"


class ModelProvider(str, Enum):
    """Supported model providers."""

    OPENAI = "openai"
    AZURE = "azure"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
    CUSTOM = "custom"


class HandoffPattern(str, Enum):
    """Team handoff patterns."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    ITERATIVE = "iterative"


class MessageRole(str, Enum):
    """Message role in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class TriggerType(str, Enum):
    """Trigger types."""

    WEBHOOK = "webhook"
    SCHEDULE = "schedule"
    EMAIL = "email"


class HILPointType(str, Enum):
    """Human-in-the-loop point types."""

    BEFORE_AGENT = "before_agent"
    AFTER_AGENT = "after_agent"
    AT_CONNECTION = "at_connection"
    BEFORE_OUTPUT = "before_output"


# ============================================================================
# Request Options
# ============================================================================


class RequestOptions(BaseModel):
    """Options for API requests."""

    timeout: Optional[float] = Field(default=60.0, description="Request timeout in seconds")
    max_retries: Optional[int] = Field(default=2, description="Maximum number of retries")
    additional_headers: Optional[Dict[str, str]] = None


class RawResponse(BaseModel):
    """Wrapper for raw API response with headers."""

    data: Any
    headers: Dict[str, str]
    status_code: int

    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# Authentication
# ============================================================================


class AuthProvider(str, Enum):
    """Authentication provider types."""

    LOCAL = "local"
    GOOGLE = "google"
    MICROSOFT = "microsoft"
    GITHUB = "github"


class UserRole(str, Enum):
    """User role types."""

    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"


class TokenResponse(BaseModel):
    """JWT token response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: Optional[int] = None


class UserResponse(BaseModel):
    """User information response."""

    id: str
    email: str
    display_name: Optional[str] = None
    role: UserRole = UserRole.USER
    is_active: bool = True
    auth_provider: AuthProvider = AuthProvider.LOCAL
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None


class AuthStatusResponse(BaseModel):
    """Authentication status response."""

    auth_enabled: bool
    auth_mode: str = "local"
    available_providers: List[AuthProvider] = []
    registration_enabled: bool = True
    current_user: Optional[UserResponse] = None


class LoginRequest(BaseModel):
    """Login request."""

    email: str
    password: str


class RegisterRequest(BaseModel):
    """Registration request."""

    email: str
    password: str
    display_name: Optional[str] = None


# ============================================================================
# Memory Blocks
# ============================================================================


class MemoryBlockCreate(BaseModel):
    """Schema for creating a memory block."""

    label: str = Field(..., description="Unique label for the memory block")
    value: str = Field(..., description="Content of the memory block")
    description: Optional[str] = Field(None, description="Optional description")


class MemoryBlock(BaseModel):
    """Memory block attached to an agent."""

    id: Optional[str] = None  # String ID (CosmosDB)
    label: str
    value: str
    description: Optional[str] = None
    agent_id: Optional[str] = None
    is_shared: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class MemoryBlockUpdate(BaseModel):
    """Schema for updating a memory block."""

    value: Optional[str] = None
    description: Optional[str] = None


# ============================================================================
# Endpoint Configuration
# ============================================================================


class EndpointConfig(BaseModel):
    """LLM endpoint configuration."""

    id: Optional[str] = None
    name: str
    provider: str
    model_name: str
    api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    api_version: Optional[str] = None
    is_default: bool = False


# ============================================================================
# MCP Server Configuration
# ============================================================================


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""

    id: Optional[str] = None
    name: Optional[str] = Field(default=None, validation_alias="server_name")
    url: Optional[str] = Field(default=None, validation_alias="server_url")
    api_key: Optional[str] = None
    tools: List[Any] = Field(default_factory=list)  # Can be list of str or list of dict
    enabled: bool = True

    model_config = {"from_attributes": True, "extra": "ignore", "populate_by_name": True}

    @property
    def server_name(self) -> Optional[str]:
        """Alias for name."""
        return self.name

    @property
    def server_url(self) -> Optional[str]:
        """Alias for url."""
        return self.url


class MCPTool(BaseModel):
    """Tool definition from MCP server."""

    name: str
    description: str
    input_schema: Dict[str, Any] = Field(default_factory=dict, alias="inputSchema")

    class Config:
        populate_by_name = True


class MCPConnectResponse(BaseModel):
    """Response from MCP server connection."""

    success: bool
    tools: List[MCPTool] = []
    error: Optional[str] = None


# ============================================================================
# Tools
# ============================================================================


class ToolParameter(BaseModel):
    """Parameter definition for a tool."""

    name: str
    type: str
    description: str
    required: bool = True
    default: Optional[Any] = None


class ToolDefinition(BaseModel):
    """Tool definition."""

    id: Optional[str] = None
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    requires_approval: bool = False
    is_custom: bool = False


class ToolCallDetail(BaseModel):
    """Details of a tool call during execution."""

    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[str] = None
    status: str = "success"


class CustomTool(BaseModel):
    """Custom tool created via @tool decorator."""

    id: Optional[str] = None
    name: str
    description: str
    parameters: Dict[str, Any]
    code: Optional[str] = None
    requires_approval: bool = False
    user_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class CustomToolCreate(BaseModel):
    """Request to create a custom tool."""

    name: str
    description: str
    parameters: Dict[str, Any]
    code: str
    requires_approval: bool = False


class CustomToolUpdate(BaseModel):
    """Request to update a custom tool."""

    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    code: Optional[str] = None
    requires_approval: Optional[bool] = None


# ============================================================================
# Agents
# ============================================================================


class AgentCreate(BaseModel):
    """Schema for creating an agent."""

    name: str = Field(..., description="Unique name for the agent")
    system_instructions: Optional[str] = Field(None, description="System prompt")
    llm_endpoint_id: Optional[str] = None  # Reference to saved LLM endpoint
    model_provider: str = Field(default="azure", description="LLM provider")
    model_name: str = Field(default="gpt-4o", description="Model name")
    api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    api_version: Optional[str] = None
    temperature: Optional[str] = "0.7"
    max_tokens: Optional[int] = 4096
    tools: Optional[List[str]] = None
    tools_requiring_approval: Optional[List[str]] = None
    mcp_servers: Optional[List[Dict[str, Any]]] = None
    role: Optional[str] = "worker"
    persistent_context: Optional[str] = None
    auto_inject_memory: Optional[bool] = False


class AgentUpdate(BaseModel):
    """Schema for updating an agent."""

    name: Optional[str] = None
    system_instructions: Optional[str] = None
    llm_endpoint_id: Optional[str] = None  # Reference to saved LLM endpoint
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    api_version: Optional[str] = None
    temperature: Optional[str] = None
    max_tokens: Optional[int] = None
    tools: Optional[List[str]] = None
    tools_requiring_approval: Optional[List[str]] = None
    mcp_servers: Optional[List[Dict[str, Any]]] = None
    role: Optional[str] = None
    persistent_context: Optional[str] = None
    auto_inject_memory: Optional[bool] = None


class Agent(BaseModel):
    """Agent representation."""

    id: str  # String ID (CosmosDB UUID)
    name: str
    role: str = "worker"
    status: str = "active"
    system_instructions: Optional[str] = None
    llm_endpoint_id: Optional[str] = None  # Reference to saved LLM endpoint
    model_provider: str = "azure"
    model_name: str = "gpt-4o"
    api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    api_version: Optional[str] = None
    temperature: Optional[str] = "0.7"
    max_tokens: Optional[int] = 4096
    tools: List[str] = Field(default_factory=list)
    tools_requiring_approval: List[str] = Field(default_factory=list)
    mcp_servers: Optional[List[Dict[str, Any]]] = None
    persistent_context: Optional[str] = None
    auto_inject_memory: bool = False
    user_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_run: Optional[datetime] = None
    total_calls: int = 0
    a2a_api_key: Optional[str] = None

    @field_validator("tools", "tools_requiring_approval", mode="before")
    @classmethod
    def none_to_list(cls, v):
        return v if v is not None else []

    class Config:
        from_attributes = True


class AgentState(BaseModel):
    """Current state of an agent including memory."""

    agent: Agent
    memory: Dict[str, MemoryBlock] = Field(default_factory=dict)
    session_id: Optional[str] = None


# ============================================================================
# Sessions
# ============================================================================


class ChatMessage(BaseModel):
    """Chat message in a session."""

    role: str
    content: str
    timestamp: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChatSessionCreate(BaseModel):
    """Schema for creating a chat session."""

    agent_id: Optional[str] = None
    team_id: Optional[str] = None


class ChatSession(BaseModel):
    """Chat session representation."""

    id: str  # String ID
    agent_id: Optional[str] = None
    team_id: Optional[str] = None
    messages: List[ChatMessage] = Field(default_factory=list)
    user_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ============================================================================
# Teams
# ============================================================================


class TeamConnectionCreate(BaseModel):
    """Schema for creating a team connection."""

    from_node_id: str
    to_node_id: str
    from_agent_id: Optional[str] = None
    to_agent_id: Optional[str] = None
    source_handle: Optional[str] = None
    target_handle: Optional[str] = None
    condition: Optional[str] = None
    edge_type: str = "sync"
    output_label: Optional[str] = None
    output_number: Optional[int] = None


class TeamConnection(BaseModel):
    """Team connection representation."""

    id: Optional[str] = None
    from_node_id: str
    to_node_id: str
    from_agent_id: Optional[str] = None
    to_agent_id: Optional[str] = None
    source_handle: Optional[str] = None
    target_handle: Optional[str] = None
    condition: Optional[str] = None
    edge_type: str = "sync"
    output_label: Optional[str] = None
    output_number: Optional[int] = None


class TeamNode(BaseModel):
    """Node in a team workflow."""

    node_id: str
    type: str  # agent, input, output, parallel, aggregator, condition, loop, hil, team
    position: Dict[str, float] = Field(default_factory=dict)
    agent_snapshot: Optional[Dict[str, Any]] = None
    branches: Optional[int] = None
    strategy: Optional[str] = None
    custom_prompt: Optional[str] = None
    prompt: Optional[str] = None
    options: Optional[List[Dict[str, str]]] = None
    output_label: Optional[str] = None
    output_number: Optional[int] = None


class HILPoint(BaseModel):
    """Human-in-the-loop point definition."""

    type: HILPointType
    agent_id: Optional[str] = None
    connection_id: Optional[str] = None
    prompt: Optional[str] = None
    auto_approve_after_seconds: Optional[int] = None


class TeamCreate(BaseModel):
    """Schema for creating a team."""

    name: str
    description: Optional[str] = None
    handoff_pattern: str = "sequential"
    nodes: Optional[List[Dict[str, Any]]] = None
    connections: Optional[List[Dict[str, Any]]] = None
    shared_memory_labels: Optional[List[str]] = None
    hil_points: Optional[List[Dict[str, Any]]] = None
    input_node_position: Optional[Dict[str, float]] = None


class TeamUpdate(BaseModel):
    """Schema for updating a team."""

    name: Optional[str] = None
    description: Optional[str] = None
    handoff_pattern: Optional[str] = None
    nodes: Optional[List[Dict[str, Any]]] = None
    connections: Optional[List[Dict[str, Any]]] = None
    shared_memory_labels: Optional[List[str]] = None
    hil_points: Optional[List[Dict[str, Any]]] = None
    input_node_position: Optional[Dict[str, float]] = None


class Team(BaseModel):
    """Team representation."""

    id: str  # String ID (CosmosDB UUID)
    name: str
    type: Optional[str] = None  # CosmosDB document type field
    description: Optional[str] = None
    handoff_pattern: Optional[str] = "sequential"
    nodes: List[Dict[str, Any]] = Field(default_factory=list)
    connections: List[Dict[str, Any]] = Field(default_factory=list)
    shared_memory_labels: Optional[List[str]] = Field(default=None)
    hil_points: Optional[List[Dict[str, Any]]] = Field(default=None)
    input_node_position: Optional[Dict[str, float]] = None
    user_id: Optional[str] = None
    a2a_api_key: Optional[str] = None
    is_canvas_team: Optional[bool] = None
    entry_point_agent_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = {"from_attributes": True, "extra": "ignore"}


# ============================================================================
# Execution
# ============================================================================


class MessageRequest(BaseModel):
    """Request to send a message."""

    message: str
    session_id: Optional[str] = None
    approved_tools: Optional[List[str]] = None


class ExecutionResponse(BaseModel):
    """Response from agent execution."""

    response: str
    session_id: Optional[str] = None
    tool_calls: List[ToolCallDetail] = Field(default_factory=list)
    reasoning_steps: List[str] = Field(default_factory=list)
    pending_approval: Optional[Dict[str, Any]] = None


class StreamChunk(BaseModel):
    """Chunk from streaming response."""

    type: str  # text, tool_call, tool_result, reasoning, done, error, hil_required
    content: Optional[str] = None
    tool_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    result: Optional[str] = None


class TeamExecutionRequest(BaseModel):
    """Request to execute a team workflow."""

    message: str
    session_id: Optional[str] = None
    team_memory: Optional[List[Dict[str, Any]]] = None
    resume_from_hil_node_id: Optional[str] = None
    hil_choice_value: Optional[str] = None
    hil_choice_label: Optional[str] = None
    hil_input: Optional[str] = None


class TeamExecutionResponse(BaseModel):
    """Response from team execution."""

    execution_id: Optional[str] = None
    status: str = "completed"
    final_response: Optional[str] = None
    agent_results: List[Dict[str, Any]] = Field(default_factory=list)
    session_id: Optional[str] = None
    pending_hil: Optional[Dict[str, Any]] = None


# ============================================================================
# Triggers
# ============================================================================


class TriggerCreate(BaseModel):
    """Schema for creating a trigger."""

    name: str
    type: TriggerType
    target_type: str = "agent"  # agent or team
    target_id: str
    enabled: bool = True
    # Webhook specific
    webhook_path: Optional[str] = None
    # Schedule specific
    schedule_cron: Optional[str] = None
    schedule_timezone: Optional[str] = "UTC"
    # Email specific
    email_address: Optional[str] = None
    # Common
    input_template: Optional[str] = None
    description: Optional[str] = None


class TriggerUpdate(BaseModel):
    """Schema for updating a trigger."""

    name: Optional[str] = None
    enabled: Optional[bool] = None
    schedule_cron: Optional[str] = None
    schedule_timezone: Optional[str] = None
    input_template: Optional[str] = None
    description: Optional[str] = None


class Trigger(BaseModel):
    """Trigger representation."""

    id: str  # String ID
    name: str
    trigger_type: Optional[str] = None  # webhook, schedule, event
    type: Optional[str] = None  # Alias for trigger_type
    target_type: str = "agent"
    target_id: str
    enabled: bool = True
    description: Optional[str] = None
    trigger_config: Optional[Dict[str, Any]] = None
    input_template: Optional[str] = None
    default_input: Optional[str] = None
    webhook_token: Optional[str] = None
    webhook_url: Optional[str] = None
    webhook_path: Optional[str] = None  # Legacy
    schedule_cron: Optional[str] = None  # Legacy
    schedule_timezone: Optional[str] = "UTC"  # Legacy
    email_address: Optional[str] = None  # Legacy
    user_id: Optional[str] = None
    last_triggered: Optional[datetime] = None
    execution_count: int = 0
    trigger_count: int = 0  # Legacy
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
        extra = "allow"  # Allow extra fields from backend


# ============================================================================
# A2A (Agent-to-Agent)
# ============================================================================


class AgentCard(BaseModel):
    """A2A Agent Card (Google A2A Protocol)."""

    name: str
    description: Optional[str] = None
    url: str
    provider: Optional[Dict[str, str]] = None
    version: str = "1.0"
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    authentication: Optional[Dict[str, Any]] = None
    skills: List[Dict[str, Any]] = Field(default_factory=list)


class A2ATaskRequest(BaseModel):
    """A2A task send request."""

    message: str
    session_id: Optional[str] = None


class A2ATaskResponse(BaseModel):
    """A2A task response."""

    task_id: str
    status: str
    result: Optional[str] = None
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)


class A2AKeyInfo(BaseModel):
    """A2A API key information."""

    has_key: bool
    key_preview: Optional[str] = None
    full_key: Optional[str] = None


# ============================================================================
# Analytics
# ============================================================================


class AnalyticsOverview(BaseModel):
    """Analytics overview with key metrics."""

    period: str
    total_calls: int = 0
    total_calls_change: float = 0
    success_calls: int = 0
    error_calls: int = 0
    error_rate: float = 0
    error_rate_change: float = 0
    avg_latency_ms: int = 0
    avg_latency_change: float = 0
    latency_p50: int = 0
    latency_p95: int = 0
    latency_p99: int = 0
    total_tokens: int = 0
    total_tokens_change: float = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0
    estimated_cost_change: float = 0
    active_agents: int = 0
    active_teams: int = 0
    tool_calls: int = 0


class TimelinePoint(BaseModel):
    """Single point in timeline data."""

    time: str
    timestamp: Optional[str] = None
    calls: int = 0
    errors: int = 0
    avg_latency: int = 0
    tokens: int = 0
    cost_cents: int = 0


class AnalyticsTimeline(BaseModel):
    """Timeline data for charts."""

    timeline: List[TimelinePoint] = []
    period: str


class AgentAnalytics(BaseModel):
    """Analytics for a single agent."""

    agent_id: str
    agent_name: str
    calls: int = 0
    errors: int = 0
    error_rate: float = 0
    avg_latency_ms: int = 0
    p95_latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0
    tool_calls: int = 0
    last_call: Optional[str] = None


class AgentAnalyticsList(BaseModel):
    """List of agent analytics."""

    agents: List[AgentAnalytics] = []
    period: str


class ToolAnalytics(BaseModel):
    """Analytics for a single tool."""

    name: str
    calls: int = 0
    success_rate: float = 0
    errors: int = 0
    avg_latency_ms: int = 0


class ToolAnalyticsList(BaseModel):
    """List of tool analytics."""

    tools: List[ToolAnalytics] = []
    period: str


class ErrorExample(BaseModel):
    """Example of an error occurrence."""

    trace_id: str
    entity_name: str
    message: Optional[str] = None
    timestamp: str


class ErrorBreakdown(BaseModel):
    """Breakdown of error by type."""

    type: str
    count: int = 0
    percentage: float = 0
    examples: List[ErrorExample] = []


class AnalyticsErrors(BaseModel):
    """Error analytics."""

    total_errors: int = 0
    errors: List[ErrorBreakdown] = []
    period: str


class ProviderAnalytics(BaseModel):
    """Analytics for a model provider."""

    provider: str
    calls: int = 0
    tokens: int = 0
    cost_usd: float = 0
    error_rate: float = 0


class ModelAnalytics(BaseModel):
    """Analytics for a specific model."""

    model: str
    calls: int = 0
    tokens: int = 0
    cost_usd: float = 0
    avg_latency_ms: int = 0


class ModelsAnalytics(BaseModel):
    """Model and provider analytics."""

    providers: List[ProviderAnalytics] = []
    models: List[ModelAnalytics] = []
    period: str


class Trace(BaseModel):
    """Execution trace record."""

    trace_id: str
    parent_trace_id: Optional[str] = None
    entity_type: str
    entity_id: str
    entity_name: str
    status: str
    input_preview: Optional[str] = None
    input_text: Optional[str] = None
    input_tokens: Optional[int] = None
    output_preview: Optional[str] = None
    output_text: Optional[str] = None
    output_tokens: Optional[int] = None
    duration_ms: Optional[int] = None
    time_to_first_token_ms: Optional[int] = None
    tool_count: int = 0
    tools_called: Optional[List[Dict[str, Any]]] = None
    cost_usd: float = 0
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    memory_reads: int = 0
    memory_writes: int = 0
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    session_id: Optional[str] = None
    trigger_id: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    child_traces: Optional[List["Trace"]] = None


class TraceList(BaseModel):
    """List of traces with pagination."""

    traces: List[Trace] = []
    total: int = 0
    limit: int = 50
    offset: int = 0


# ============================================================================
# File Management
# ============================================================================


class FileInfo(BaseModel):
    """Uploaded file information."""

    id: str
    filename: str
    size: int
    content_type: Optional[str] = None
    uploaded_at: Optional[str] = None


class FileUploadResponse(BaseModel):
    """Response from file upload."""

    success: bool
    file: Optional[FileInfo] = None
    message: Optional[str] = None
    error: Optional[str] = None


# ============================================================================
# Session Info / Stats
# ============================================================================


class SessionInfo(BaseModel):
    """Server session information."""

    agents: int = 0
    teams: int = 0
    files: int = 0
    providers: bool = True


class SystemStats(BaseModel):
    """System-wide statistics."""

    agents: Dict[str, Any] = Field(default_factory=dict)
    teams: Dict[str, Any] = Field(default_factory=dict)
    sessions: Dict[str, Any] = Field(default_factory=dict)
    triggers: Dict[str, Any] = Field(default_factory=dict)
    memory: Dict[str, Any] = Field(default_factory=dict)
    files: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Approval (Tool Approval)
# ============================================================================


class ApprovalRequest(BaseModel):
    """Request for tool approval."""

    tool_name: str
    arguments: Dict[str, Any]
    agent_id: str
    session_id: Optional[str] = None
    expires_at: Optional[datetime] = None


class ApprovalResponse(BaseModel):
    """Response to an approval request."""

    approved: bool
    modified_args: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None


# ============================================================================
# LLM Endpoint
# ============================================================================


class LLMEndpoint(BaseModel):
    """LLM Endpoint configuration."""

    id: str
    name: str
    provider_type: str
    model_name: str
    api_key: Optional[str] = None  # Masked in response
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    api_version: Optional[str] = None
    custom_base_url: Optional[str] = None
    is_default: bool = False
    user_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
