# csf-recommendation-engine

Python microservice for the CSF block trade recommendation system.

## Local setup

1. Create and activate a Python 3.11 virtual environment.
2. Install dependencies with `uv` using lock-aware sync.
3. Run the API service.

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
