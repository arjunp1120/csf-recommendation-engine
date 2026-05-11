from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RecommendRequest(BaseModel):
    client_id: str
    desk: str
    structure: str
    side: str
    venue: str
    instrument_name: str
    quantity: int = Field(..., gt=0)
    top_k: int = 5

class RecommendResponse(BaseModel):
    counterparties: list[dict[str, Any]] = Field(..., description="List of recommended counterparties")
    queried_proxy: str = Field(..., description="The proxy string used for the query")


EngagementEventType = Literal[
    "impression",
    "click",
    "dwell",
    "thumbs_up",
    "thumbs_down",
    "in_progress",
    "coach_drafted",
    "copy_text",
]


class EngagementEventRequest(BaseModel):
    """Inbound payload for ``POST /events/engagement`` (plan §15.5).

    Exactly one of ``serve_id`` / ``match_id`` must be set. Validation
    matches the DB's ``one_target`` CHECK constraint so we reject bad
    payloads at the API boundary with a 422 rather than letting them
    reach Postgres.
    """

    model_config = ConfigDict(extra="forbid")

    event_type: EngagementEventType
    serve_id: UUID | None = None
    match_id: UUID | None = None
    user_id: UUID | None = None
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _exactly_one_target(self) -> "EngagementEventRequest":
        if (self.serve_id is not None) + (self.match_id is not None) != 1:
            raise ValueError(
                "exactly one of serve_id / match_id must be provided"
            )
        return self


class EngagementEventResponse(BaseModel):
    event_id: UUID
