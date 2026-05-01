from datetime import datetime

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/internal", tags=["internal"])


@router.post("/shadow/serialize")
async def serialize_shadow_model(request: Request) -> dict:
	request_id = getattr(request.state, "request_id", None)
	if getattr(request.app.state, "shadow_model", None) is None:
		raise HTTPException(
			status_code=404,
			detail={
				"code": "SHADOW_MODEL_NOT_INITIALIZED",
				"message": "Shadow model not initialized",
				"request_id": request_id,
			},
		)

	# TODO: implement blob serialization once the trigger path and artifact policy are approved.
	return {
		"status": "accepted",
		"timestamp": datetime.utcnow().isoformat(),
		"request_id": request_id,
		"shadow_model_present": True,
		"detail": "Serialization endpoint scaffolded; blob write implementation pending.",
	}
