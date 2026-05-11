"""Low-level async HTTP transport to the DAF backend.

Plan §11.1 mandates direct HTTP (no DAF SDK dependency). This module
wraps an `httpx.AsyncClient` with retry + timeout + structured-error
handling specific to the DAF execute endpoints.

Wire protocol (confirmed against `daf_sdk/resources/agents.py:AgentMessages.send`
and `daf_sdk/_http.py:HTTPClient._handle_response` in the venv):

* Endpoint: ``POST {base_url}/api/execute/agent``
* Body: ``{"agent_id": "...", "message": "...",
          "session_id"?: "...", "approved_tools"?: [...]}``
* Auth: ``Authorization: Bearer {DAF_API_KEY}``
* Response (success): JSON body matching `ExecutionResponse` from
  `daf_sdk/models.py` — at minimum a ``response: str`` field.
* Response (failure): non-2xx with a ``detail`` field in the body.

This module intentionally treats swarm_id and agent_id identically on
the wire; the user has confirmed they're addressed the same way and
the DAF backend currently routes both through `/api/execute/agent`.
If a separate endpoint is later needed for teams/swarms, swap the
``EXECUTE_AGENT_PATH`` constant or add a config knob.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from csf_recommendation_engine.core.config import Settings

logger = logging.getLogger(__name__)

# Path constant — single source of truth. If swarm/team endpoints are
# later split out, replace usage at the per-method site, not here.
EXECUTE_AGENT_PATH = "/api/execute/agent"

# Number of retries on transient transport failures. Set conservatively
# because callers (recommend, /ioi/accept) have sub-3s budgets.
DEFAULT_MAX_RETRIES = 1


class TransportError(RuntimeError):
    """Raised when the DAF backend is unreachable or returns an error.

    Distinct from validation/parsing failures (which are surfaced
    differently by the IntelligenceService layer).
    """


@dataclass
class ExecuteResult:
    """Successful parse of a DAF execute response.

    `response_text` is the agent/swarm's free-text reply (typically a
    JSON string the caller will further parse). `session_id` and
    auxiliary fields are passed through for callers that want them.
    """

    response_text: str
    session_id: str | None
    tool_calls: list[dict[str, Any]]
    reasoning_steps: list[str]
    pending_approval: dict[str, Any] | None
    raw_body: dict[str, Any]


class DAFTransport:
    """Thin async HTTP client over the DAF execute endpoint.

    Lifecycle: instantiate once at app startup, share across all
    `IntelligenceService` calls, close at shutdown via `aclose()`.
    """

    def __init__(self, settings: Settings, max_retries: int = DEFAULT_MAX_RETRIES) -> None:
        self._settings = settings
        self._max_retries = max(0, max_retries)
        # httpx.AsyncClient is created lazily so unit tests that mock
        # the transport don't need to set up a real network resource.
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._settings.daf_base_url,
                timeout=self._settings.llm_request_timeout_s,
                headers=self._default_headers(),
            )
        return self._client

    def _default_headers(self) -> dict[str, str]:
        """Match the DAF SDK's HTTPClient default headers exactly."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._settings.daf_api_key:
            headers["Authorization"] = f"Bearer {self._settings.daf_api_key}"
        return headers

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def execute_agent(
        self,
        target_id: str,
        message: str,
        session_id: str | None = None,
        approved_tools: list[str] | None = None,
    ) -> ExecuteResult:
        """POST `/api/execute/agent` to the configured DAF base URL and
        return the parsed agent text reply.

        Raises:
            TransportError: when the call cannot succeed (network error,
                non-2xx response, malformed body, missing `response`
                field).
        """
        if not target_id:
            raise TransportError("execute_agent called with empty target_id")

        payload: dict[str, Any] = {"agent_id": target_id, "message": message}
        if session_id:
            payload["session_id"] = session_id
        if approved_tools:
            payload["approved_tools"] = approved_tools

        client = self._ensure_client()
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await client.post(EXECUTE_AGENT_PATH, json=payload)
            except httpx.ConnectError as exc:
                last_error = exc
                logger.warning(
                    "DAF transport connect error (attempt %d/%d): %s",
                    attempt + 1, self._max_retries + 1, exc,
                )
            except httpx.TimeoutException as exc:
                last_error = exc
                logger.warning(
                    "DAF transport timeout (attempt %d/%d): %s",
                    attempt + 1, self._max_retries + 1, exc,
                )
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning(
                    "DAF transport HTTP error (attempt %d/%d): %s",
                    attempt + 1, self._max_retries + 1, exc,
                )
            else:
                return self._handle_response(response, target_id=target_id)

            if attempt < self._max_retries:
                # Mirror the SDK's exponential backoff: 0.5, 1.0, 2.0, ...
                await asyncio.sleep(0.5 * (2 ** attempt))

        raise TransportError(
            f"DAF execute_agent failed after {self._max_retries + 1} attempts: {last_error}"
        ) from last_error

    def _handle_response(self, response: httpx.Response, target_id: str) -> ExecuteResult:
        """Parse a 2xx response or raise TransportError on 4xx/5xx /
        malformed body."""
        try:
            body = response.json() if response.content else None
        except ValueError as exc:
            raise TransportError(
                f"DAF execute response was not valid JSON (status={response.status_code}, "
                f"target_id={target_id!r}): {exc}"
            ) from exc

        if response.status_code >= 400:
            detail = body.get("detail", str(body)) if isinstance(body, dict) else str(body)
            raise TransportError(
                f"DAF execute returned {response.status_code} for target_id={target_id!r}: {detail}"
            )

        if not isinstance(body, dict):
            raise TransportError(
                f"DAF execute response was not a JSON object (target_id={target_id!r}, "
                f"type={type(body).__name__})"
            )

        response_text = body.get("response")
        if not isinstance(response_text, str) or not response_text:
            raise TransportError(
                f"DAF execute response missing non-empty 'response' field "
                f"(target_id={target_id!r}, body_keys={list(body.keys())})"
            )

        return ExecuteResult(
            response_text=response_text,
            session_id=body.get("session_id"),
            tool_calls=body.get("tool_calls", []),
            reasoning_steps=body.get("reasoning_steps", []),
            pending_approval=body.get("pending_approval"),
            raw_body=body,
        )
