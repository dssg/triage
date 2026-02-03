# Triage Interactive TUI

An interactive terminal user interface for exploring and managing Triage experiments.

## Quick Start

```bash
# Launch the TUI (coming soon)
uv run triage tui

# Or from Python
python -c "from triage.tui import TriageApp; from triage import create_engine; app = TriageApp(create_engine('database.yaml')); app.run()"
```

## Current Status

🚧 **Work in Progress** - Basic structure created, implementation in progress

### Implemented
- ✅ Basic package structure
- ✅ ExperimentListScreen skeleton
- ✅ ExperimentDetailScreen skeleton
- ✅ Main App class

### TODO (Phase 1 - Foundation)
- [ ] Complete ExperimentListScreen with real data
- [ ] Database polling service
- [ ] CSS styling
- [ ] Basic keybindings
- [ ] CLI command integration

See `../../../docs/tui-design.md` for full design specification.

## Features (Planned)

### Experiment Dashboard
- Browse all experiments with status indicators
- Sort and filter by name, status, date
- Real-time updates for running experiments
- Quick stats overview

### Experiment Details
- Multi-tab view (Overview, Models, Matrices, Config, Logs)
- Live progress tracking
- Streaming logs
- Model performance metrics

### Model Comparison
- Side-by-side comparison
- Performance charts
- Feature importance analysis
- Export capabilities

### Interactive Commands
- Command palette (press `/`)
- Run experiments
- Audition models
- Aequitas bias audits
- Export predictions

### Advanced Features
- Fairness-aware model selection
- Bias audit visualization
- Production deployment
- Report generation

## Keyboard Shortcuts

### Global
- `/` - Open command palette
- `?` - Help
- `q` - Quit
- `r` - Refresh

### Navigation
- `↑/↓` or `j/k` - Move up/down
- `Enter` - Select/drill down
- `Esc` - Go back
- `Tab` - Switch tabs

### Experiment List
- `f` - Filter experiments
- `s` - Sort by column
- `Space` - Select experiment

## Architecture

```
triage/tui/
├── __init__.py           # Package exports
├── app.py                # Main App and Screens
├── widgets/              # Reusable UI components
│   ├── experiment_table.py
│   ├── model_metrics.py
│   ├── progress_tracker.py
│   ├── command_palette.py
│   └── log_stream.py
├── screens/              # Full-screen views
│   ├── dashboard.py      # Experiment list
│   ├── experiment.py     # Experiment details
│   ├── models.py         # Model comparison
│   └── audition.py       # Model selection
├── services/             # Background services
│   ├── db_poller.py      # Database polling
│   └── log_watcher.py    # Log streaming
└── styles/
    └── triage.tcss       # CSS styling
```

## Development

### Running in Development Mode

```bash
# Install dependencies
uv sync --extra dev

# Run the TUI (once integrated)
uv run triage tui --debug

# Or run directly for testing
uv run python -m triage.tui.app
```

### Testing

```bash
# Run TUI tests (coming soon)
uv run pytest src/tests/tui_tests/

# Manual testing
# 1. Set up test database with sample experiments
# 2. Launch TUI
# 3. Navigate and test features
```

### Contributing

See the main [CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines.

TUI-specific considerations:
- Follow Textual best practices
- Keep screens focused (single responsibility)
- Use widgets for reusable components
- Test with different terminal sizes
- Ensure keyboard-only navigation works

## Technical Notes

### Database Polling
- Polls every 5 seconds when experiments are running
- Uses efficient queries with proper indexes
- Caches static data (configs, model groups)

### Performance
- Lazy loading for large datasets
- Virtualized tables (only render visible rows)
- Background workers for expensive operations
- Debounced updates to avoid UI flicker

### Error Handling
- Graceful degradation when DB unavailable
- Clear error messages with recovery suggestions
- Retry logic for transient failures

## References

- [Textual Documentation](https://textual.textualize.io/)
- [Rich Documentation](https://rich.readthedocs.io/)
- [Design Document](../../../docs/tui-design.md)
- [TODO List](../../../TODO.org)

## Future Vision

The TUI aims to provide a complete interactive experience for Triage, making it easy to:
- Explore experiment results without SQL queries
- Monitor long-running experiments in real-time
- Compare and select models visually
- Ensure fairness with integrated bias audits
- Deploy models to production with confidence

Think of it as "k9s for Triage" or "lazygit for ML experiments".
