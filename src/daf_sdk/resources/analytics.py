"""Analytics resource - Metrics, timeline, traces, errors."""

from typing import Any, Dict, List

from .._base import AsyncResource, SyncResource
from ..models import (
    AgentAnalytics,
    AnalyticsOverview,
    AnalyticsTimeline,
    ToolAnalytics,
    Trace,
    TraceList,
)


class Analytics(SyncResource):
    """Synchronous analytics resource."""

    def overview(self, period: str = "24h") -> AnalyticsOverview:
        """Get analytics overview with key metrics."""
        response = self._client.get("/api/analytics/overview", params={"period": period})
        return AnalyticsOverview.model_validate(
            response.json() if hasattr(response, "json") else response
        )

    def timeline(self, period: str = "24h") -> AnalyticsTimeline:
        """Get timeline data for charts."""
        response = self._client.get("/api/analytics/timeline", params={"period": period})
        return AnalyticsTimeline.model_validate(
            response.json() if hasattr(response, "json") else response
        )

    def agents(self, period: str = "24h") -> List[AgentAnalytics]:
        """Get per-agent analytics."""
        response = self._client.get("/api/analytics/agents", params={"period": period})
        data = response.json() if hasattr(response, "json") else response
        agents_data = data.get("agents", []) if isinstance(data, dict) else data
        return [AgentAnalytics.model_validate(a) for a in agents_data]

    def tools(self, period: str = "24h") -> List[ToolAnalytics]:
        """Get per-tool analytics."""
        response = self._client.get("/api/analytics/tools", params={"period": period})
        data = response.json() if hasattr(response, "json") else response
        tools_data = data.get("tools", []) if isinstance(data, dict) else data
        return [ToolAnalytics.model_validate(t) for t in tools_data]

    def models(self, period: str = "24h") -> Dict[str, Any]:
        """Get per-model analytics."""
        response = self._client.get("/api/analytics/models", params={"period": period})
        return response.json() if hasattr(response, "json") else response

    def errors(self, period: str = "24h") -> Dict[str, Any]:
        """Get error breakdown."""
        response = self._client.get("/api/analytics/errors", params={"period": period})
        return response.json() if hasattr(response, "json") else response

    def traces(self, period: str = "24h", skip: int = 0, limit: int = 50, **filters) -> TraceList:
        """List execution traces with filtering."""
        params = {"period": period, "skip": skip, "limit": limit, **filters}
        response = self._client.get("/api/analytics/traces", params=params)
        return TraceList.model_validate(response.json() if hasattr(response, "json") else response)

    def get_trace(self, trace_id: str) -> Trace:
        """Get a specific trace by ID."""
        response = self._client.get(f"/api/analytics/traces/{trace_id}")
        return Trace.model_validate(response.json() if hasattr(response, "json") else response)


class AsyncAnalytics(AsyncResource):
    """Asynchronous analytics resource."""

    async def overview(self, period: str = "24h") -> AnalyticsOverview:
        """Get analytics overview with key metrics."""
        response = await self._client.get("/api/analytics/overview", params={"period": period})
        return AnalyticsOverview.model_validate(
            response.json() if hasattr(response, "json") else response
        )

    async def timeline(self, period: str = "24h") -> AnalyticsTimeline:
        """Get timeline data for charts."""
        response = await self._client.get("/api/analytics/timeline", params={"period": period})
        return AnalyticsTimeline.model_validate(
            response.json() if hasattr(response, "json") else response
        )

    async def agents(self, period: str = "24h") -> List[AgentAnalytics]:
        """Get per-agent analytics."""
        response = await self._client.get("/api/analytics/agents", params={"period": period})
        data = response.json() if hasattr(response, "json") else response
        agents_data = data.get("agents", []) if isinstance(data, dict) else data
        return [AgentAnalytics.model_validate(a) for a in agents_data]

    async def tools(self, period: str = "24h") -> List[ToolAnalytics]:
        """Get per-tool analytics."""
        response = await self._client.get("/api/analytics/tools", params={"period": period})
        data = response.json() if hasattr(response, "json") else response
        tools_data = data.get("tools", []) if isinstance(data, dict) else data
        return [ToolAnalytics.model_validate(t) for t in tools_data]

    async def models(self, period: str = "24h") -> Dict[str, Any]:
        """Get per-model analytics."""
        response = await self._client.get("/api/analytics/models", params={"period": period})
        return response.json() if hasattr(response, "json") else response

    async def errors(self, period: str = "24h") -> Dict[str, Any]:
        """Get error breakdown."""
        response = await self._client.get("/api/analytics/errors", params={"period": period})
        return response.json() if hasattr(response, "json") else response

    async def traces(
        self, period: str = "24h", skip: int = 0, limit: int = 50, **filters
    ) -> TraceList:
        """List execution traces with filtering."""
        params = {"period": period, "skip": skip, "limit": limit, **filters}
        response = await self._client.get("/api/analytics/traces", params=params)
        return TraceList.model_validate(response.json() if hasattr(response, "json") else response)

    async def get_trace(self, trace_id: str) -> Trace:
        """Get a specific trace by ID."""
        response = await self._client.get(f"/api/analytics/traces/{trace_id}")
        return Trace.model_validate(response.json() if hasattr(response, "json") else response)
