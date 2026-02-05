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
    uv run ruff check src/

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

# Start DirtyDuck tutorial database
tutorial-up:
    docker compose -f dirtyduck/docker-compose.yml up -d food_db

# Stop DirtyDuck tutorial
tutorial-down:
    docker compose -f dirtyduck/docker-compose.yml stop

# Launch DirtyDuck bastion shell
tutorial-shell:
    docker compose -f dirtyduck/docker-compose.yml run --service-ports --rm bastion

# Build DirtyDuck images
tutorial-build:
    docker compose -f dirtyduck/docker-compose.yml build

# Rebuild DirtyDuck images (no cache)
tutorial-rebuild:
    docker compose -f dirtyduck/docker-compose.yml build --no-cache

# Show DirtyDuck container status
tutorial-status:
    docker compose -f dirtyduck/docker-compose.yml ps

# View DirtyDuck logs
tutorial-logs:
    docker compose -f dirtyduck/docker-compose.yml logs -f -t

# Clean up DirtyDuck resources (removes containers, images, volumes)
tutorial-clean:
    docker compose -f dirtyduck/docker-compose.yml down --rmi all --remove-orphans --volumes
