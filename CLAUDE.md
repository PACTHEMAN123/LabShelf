# LabShelf — Developer Guide

## Project Overview

LabShelf is a **pure data management framework** that provides metadata management and data provenance for Attention MoE decoupled inference experiments. It does not run experiments itself — it only handles:

- Receiving manually entered metadata (purpose, configuration, code version, etc.)
- Managing diverse experiment data files (json / nsys-rep / txt / sqlite / csv)
- Maintaining a clear **data → script → figure** provenance chain

## Architecture

```
labshelf.py          Single-file CLI, entry point for all commands (no pip install required)
config.yaml           Global configuration (plot parameters, supported formats, etc.)
catalog.yaml          Auto-generated global index (maintained by _rebuild_catalog_data(), do not edit manually)
templates/            Template files
  metadata.yaml       Metadata template for new experiments (Python str.format placeholders)
  plot_template.py    Visualization script skeleton (Python str.format placeholders)
  other_template.py   Processing script skeleton (Python str.format placeholders)
scripts/shared/       Shared utility library
  loaders.py          Unified data loading: load_data(exp_dir, metadata, logical_name)
  plot_utils.py       Plot utilities: setup_style() + save_figure()
experiments/          Experiment data root directory (experiment data files are large and confidential — reading this content is not permitted)
```

## Core Design Decisions

1. **metadata.yaml is the single source of truth** — All data registration and figure provenance are recorded here. catalog.yaml is a derived artifact auto-aggregated from each experiment's metadata.
2. **Flat experiment list + tag filtering** — No hierarchical categorization to avoid classification disputes.
3. **Experiment ID = `YYYY-MM-DD_<slug>`** — Date prefix ensures chronological sorting.
4. **Fuzzy matching** — `_resolve_experiment(query)` supports substring matching; entering `moe-routing` can match `2026-02-23_moe-routing-latency`.
5. **Each experiment is self-contained** — Can be directly packaged and shared with others.

## labshelf.py Internal Structure

```
Global path constants: ROOT, EXPERIMENTS_DIR, CATALOG_FILE, CONFIG_FILE, TEMPLATE_DIR

Utility functions:
  _now_iso()              → ISO timestamp string
  _today()                → YYYY-MM-DD date string
  _load_config()          → Read config.yaml
  _load_metadata(exp_dir) → Read experiment metadata.yaml
  _save_metadata(exp_dir, metadata) → Write metadata.yaml (automatically updates the 'updated' timestamp)
  _resolve_experiment(query) → Fuzzy-match experiment ID, returns (exp_id, exp_dir)
  _detect_format(file_path)  → Infer data format from file extension
  _rebuild_catalog_data()    → Scan all experiments to build catalog dict
  _save_catalog(catalog)     → Write catalog.yaml

Command functions: cmd_new, cmd_add_data, cmd_add_script, cmd_plot, cmd_add_other, cmd_run_other, cmd_list, cmd_show, cmd_info, cmd_rebuild_catalog, cmd_validate

CLI entry point: main() → argparse routes to command functions
```

## metadata.yaml Key Fields

- `environment`: Experiment environment, containing `machine` (machine name), `gpu` (GPU model), `notes`
- `provenance`: Code provenance, containing `code_repo`, `code_branch` (framework branch), `code_commit` (commit hash), `notes`
- `data`: Dictionary where key is the logical name and value contains `file`, `format`, `description`
- `figures`: Dictionary where key is the figure name and value contains `file`, `script`, `description`
- `others`: Dictionary where key is the output name and value contains `file`, `script`, `description` (same structure as figures)
- `status`: `active` / `complete` / `abandoned`

## Script Template Mechanism

`templates/plot_template.py` uses Python `str.format()` placeholders: `{description}`, `{exp_id}`, `{fig_name}`, `{output_file}`. `templates/other_template.py` uses the same mechanism with placeholders: `{description}`, `{exp_id}`, `{other_name}`, `{output_file}`. Scripts are not bound to specific data inputs but to the experiment; at runtime they dynamically load required data via `load_data(exp_dir, metadata, "logical_name")`. Generated scripts locate the project root via `PROJECT_ROOT = Path(__file__).resolve().parents[3]` to import `scripts.shared`.

## Modification Notes

- **Adding a new data format:** Add a loader function to the `_LOADERS` dict in `loaders.py`, add the extension mapping in `_detect_format` in `labshelf.py`, and add the format name to `supported_formats` in `config.yaml`.
- **Adding a new CLI command:** Add an argparse subcommand in `main()`, implement a `cmd_xxx` function, and add it to the `commands` dictionary.
- **Every command that modifies metadata** should call `_rebuild_catalog_data()` + `_save_catalog()` to keep the catalog in sync.
- **After adding/modifying a feature**, update the usage instructions and directory structure in `README.md` accordingly.