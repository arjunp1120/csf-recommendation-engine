"""Domain models and schema types.

- heuristics_build.py: Reranking heuristic builder (takes trade data, outputs artifacts for reranking)

- heuristics_index.py: Reranking heuristic indexer; loads on startup

- recommendation_engine.py: The recommendation engine inference function (also reranks).

- reranking.py: Reranking logic based on heuristics and absolute filters.

- schemas.py: Pydantic schemas for HTTP API requests/responses
"""
