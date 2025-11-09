# Triage development tasks

# Show available commands
default:
    @just --list

# Preview documentation locally
docs:
    cd docs && uv run mkdocs serve

# Run alembic commands (e.g., just alembic upgrade head)
alembic *ARGS:
    PYTHONPATH=src uv run alembic \
        -c src/triage/component/results_schema/alembic.ini \
        -x db_config_file=database.yaml \
        {{ARGS}}

# Run tests
test *ARGS:
    uv run pytest {{ARGS}}

# Run tests with coverage
test-cov:
    uv run pytest --cov=triage

# Run linter
lint:
    uv run flake8 src/triage src/tests

# Run type checker
typecheck:
    uv run basedpyright

# Sync dependencies
sync *EXTRA:
    uv sync {{EXTRA}}

# Install with dev dependencies
install:
    uv sync --extra dev

# Run triage CLI
triage *ARGS:
    uv run triage {{ARGS}}

# Launch the interactive TUI
tui:
    uv run triage tui
