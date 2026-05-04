"""Standalone job entrypoints.

Currently,
- rec_refresh_pipeline.py: Refreshes recommendations (timing set in main.py)
- nightly_heuristics.py: Pulls data from db and re-calculates "reranking heuristic" features.
    - Reranking heuristics are the second-pass on LightFM output candidates.
- nightly_pipeline.py: Imports nightly_heuristics only (later should import nightly_extraction as well).

    - Currently all use local disk.
    - TODO: Update to use Azure Blob storage when implemented.

- nightly_extraction.py: Not set up.
- weekend_validation.py: Not set up.




TODO: 
- Implement nightly_extraction.py for shadow model training.
- Implement weekend_validation.py for shadow model validation.
"""
