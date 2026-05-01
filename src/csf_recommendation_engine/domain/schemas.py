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
