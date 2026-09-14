# Copilot Instructions for docintel

## Project Overview

**docintel** is a Python 3.12+ CLI application. It's currently in early development with a basic CLI skeleton.

- **Package Manager**: `uv` (fast Python package installer)
- **Python Version**: 3.12 or higher (enforced by `pyproject.toml`)
- **Build System**: `uv_build`
- **CLI Entry Point**: `docintel.main()` (defined in `src/docintel/__init__.py`)

## Development Setup

```bash
# Install dependencies (uv handles virtual env automatically)
uv sync

# Run the CLI
uv run docintel

# Or after entering the environment
source .venv/bin/activate
python -m docintel
```

## Build and Packaging

```bash
# Build the package
uv build

# This creates distributions in the `dist/` directory using the uv_build backend
```

## Project Structure

- `src/docintel/` - Main package source code
  - `__init__.py` - CLI entry point (contains `main()`)
- `pyproject.toml` - Project metadata, dependencies, and build configuration
- `.venv/` - Virtual environment (auto-managed by `uv`)
- `uv.lock` - Dependency lock file (commit this for reproducible builds)

## Key Conventions

1. **CLI Module Structure**: All CLI logic should remain organized within `src/docintel/`. If functionality grows, create submodules as needed (e.g., `commands/`, `utils/`).

2. **Entry Point**: The `docintel` command-line script is defined in `pyproject.toml` under `[project.scripts]` and calls `docintel:main`. Keep the `main()` function in `__init__.py` as the CLI dispatcher.

3. **No External Dependencies**: The project currently has zero dependencies. Be intentional about adding any—document the rationale in commit messages if new dependencies are added.

4. **Python Version**: Maintain compatibility with Python 3.12+ only. No need to support older versions.

## Future Recommendations

- **Testing**: Once tests are added, follow a standard structure like `tests/` with `pytest.ini` or testing configuration in `pyproject.toml`.
- **Logging**: If the CLI needs structured logging, consider using Python's standard `logging` module.
- **CLI Framework**: As the CLI grows in complexity, consider adding a framework like `Click` or `Typer` for argument parsing.

## Git Workflow

- `main` branch: stable, deployable code
- Feature/fix branches: use descriptive names (e.g., `feature/add-document-parsing`)
- Include `uv.lock` in commits (ensures reproducible builds)
