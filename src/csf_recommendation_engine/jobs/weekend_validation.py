import asyncio
from datetime import datetime, timezone


def build_weekend_validation_summary() -> dict:
    return {
        "status": "blocked",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detail": (
            "Weekend validation is scaffolded, but evaluation cannot proceed until the holdout dataset "
            "definition and serialization trigger path are approved."
        ),
    }


async def run_weekend_validation() -> None:
    summary = build_weekend_validation_summary()
    raise NotImplementedError(summary["detail"])


if __name__ == "__main__":
    asyncio.run(run_weekend_validation())
