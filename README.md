# csf-recommendation-engine

Python microservice for the CSF block trade recommendation system.

## Local setup

1. Create and activate a Python 3.11 virtual environment.
2. Install dependencies with `uv` using lock-aware sync.
3. Run the API service.

## Local development DB

All Intelligence Engine schema migrations and code changes target the **local Postgres** referenced by `DATABASE_URL` in `.env` — an exact mirror of production data as of 2026-05-08. **No production writes happen during development.**

Before applying any migration in `sql/migrations/`, run the pre-flight checks in [docs/runbooks/local-db-bootstrap.md](docs/runbooks/local-db-bootstrap.md). The runbook covers:

- Confirming `DATABASE_URL` points at the local DB (not Azure-managed).
- Connectivity and version probes.
- Inventory checks against expected row counts.
- Applying migrations with `psql`.
- pgvector status (currently deferred for v1; see runbook for revival steps).

For the implementation plan and decisions log, see [development-notes/Intelligence Engine Extended Implementation Plan.md](development-notes/Intelligence%20Engine%20Extended%20Implementation%20Plan.md) and [development-notes/Intelligence Engine Extended Notes.md](development-notes/Intelligence%20Engine%20Extended%20Notes.md).

## Dependency policy

- LightFM is pinned to a specific Git commit in `pyproject.toml`.
- Build prerequisites are defined in the optional `build` extra.
- Use `uv lock` and `uv sync` to keep installs reproducible.

### Recommended install flow

```bash
uv lock
uv sync --extra dev --extra build
```

### Build fallback (if LightFM compilation still fails)

```bash
uv sync --extra dev --extra build
uv pip install --no-build-isolation "git+https://github.com/lyst/lightfm.git@0c9c31e"
```

## Commands

- `csf-run-api`
- `csf-run-nightly`
- `csf-run-weekend`
- `python -m pytest`
