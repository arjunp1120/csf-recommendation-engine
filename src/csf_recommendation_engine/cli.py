import asyncio

import uvicorn

from csf_recommendation_engine.jobs.nightly_pipeline import run_full_nightly_pipeline
from csf_recommendation_engine.jobs.rec_refresh_pipeline import run_rec_refresh_pipeline
from csf_recommendation_engine.jobs.weekend_validation import run_weekend_validation


def run_api() -> None:
    uvicorn.run("csf_recommendation_engine.main:app", host="0.0.0.0", port=8000, reload=True)


def run_nightly() -> None:
    asyncio.run(run_full_nightly_pipeline())


def run_rec_refresh() -> None:
    asyncio.run(run_rec_refresh_pipeline())


def run_weekend() -> None:
    asyncio.run(run_weekend_validation())
