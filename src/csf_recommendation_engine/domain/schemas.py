from pydantic import BaseModel, Field
from typing import Any


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


class VoiceInquiryMatchRequest(BaseModel):
    """POST /voice_inquiry/match payload."""
    ioi_text: str = Field(..., description="Free-text indication-of-interest from the broker")
    originator_entity_id: str = Field(..., description="Entity id of the originator (excluded from candidates)")
    top_k: int | None = Field(default=None, description="How many candidate counterparties to consider")


class VoiceInquiryMatchResponse(BaseModel):
    parsed_tags: dict[str, Any]
    candidates: list[dict[str, Any]] = Field(..., description="Top-K ranked candidates with dossiers + their open inquiries")
    best_match: dict[str, Any] | None = Field(default=None, description="Agent-selected best matching inquiry")
    matched_inquiry: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Convenience copy of the full inquiry record the matcher picked "
            "(inquiry_id, entity_id, ioi_text, tags, created_at), resolved from "
            "the candidate inquiry pool. Null if matcher failed or no match."
        ),
    )
    appended_inquiry: dict[str, Any] = Field(..., description="The new inquiry record persisted to local disk")
    # pipeline_trace: list[dict[str, Any]] = Field(
    #     default_factory=list,
    #     description=(
    #         "Ordered, timestamped record of every step the endpoint took: what was loaded, "
    #         "from where, what was filtered/selected, and what each DAF agent saw and returned. "
    #         "Each entry has step, elapsed_ms, summary, and step-specific data."
    #     ),
    # )
