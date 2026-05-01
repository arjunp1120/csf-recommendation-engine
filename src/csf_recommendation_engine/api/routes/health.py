from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health/live")
async def live() -> dict:
    return {"status": "live"}


@router.get("/health/ready")
async def ready() -> dict:
    return {"status": "ready"}
