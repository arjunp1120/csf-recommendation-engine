from fastapi import APIRouter

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/models/{model_id}/approve")
async def approve_model(model_id: str) -> dict:
    # IMPORTANT TODO CHECKPOINT:
    # Finalize status lifecycle policy before implementing promotion mutation logic.
    return {
        "status": "pending-implementation",
        "model_id": model_id,
        "detail": "Promotion endpoint scaffolded. Status transition logic intentionally deferred.",
    }
