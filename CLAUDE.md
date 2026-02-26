# LabShelf — Developer Guide

## Project Overview

LabShelf is a **pure data management framework** that provides metadata management and data provenance for Attention MoE decoupled inference experiments. It does not run experiments itself — it only handles:

- Receiving manually entered metadata (purpose, tags, etc.)
- Managing diverse experiment data files with per-entry environment and provenance tracking
- Maintaining a clear **data → script → output** provenance chain

## Architecture

```
labshelf.py          Single-file CLI, entry point for all commands (no pip install required)
config.yaml           Global configuration (plot parameters, etc.)
catalog.yaml          Auto-generated global index (maintained by _rebuild_catalog_data(), do not edit manually)
templates/            Template files
  metadata.yaml       Metadata template for new experiments (Python str.format placeholders)
  script_template.py  Unified script skeleton (Python str.format placeholders)
scripts/              User scripts directory
  shared/             Shared utility library
    loaders.py        Unified data loading: load_data(exp_dir, metadata, logical_name)
    plot_utils.py     Plot utilities: setup_style() + save_figure()
  <user_script>.py    User-created scripts (not bound to experiments)
experiments/          Experiment data root directory (experiment data files are large and confidential — reading this content is not permitted)
```

## Core Design Decisions

1. **metadata.yaml is the single source of truth** — All data registration and output provenance are recorded here. catalog.yaml is a derived artifact auto-aggregated from each experiment's metadata.
2. **Flat experiment list + tag filtering** — No hierarchical categorization to avoid classification disputes.
3. **Experiment ID = `<slug>`** — No date prefix; each experiment represents a research direction, not a single run.
4. **Per-entry environment/provenance** — Each data entry independently records machine, GPU, code branch, and commit. Git info is auto-detected.
5. **Scripts decoupled from experiments** — Scripts live in `scripts/` and can be reused across experiments.
6. **Unified output** — `output/` replaces `figures/` + `others/`, no format restrictions.
7. **Fuzzy matching** — `_resolve_experiment(query)` supports substring matching.

## labshelf.py Internal Structure

```
Global path constants: ROOT, EXPERIMENTS_DIR, SCRIPTS_DIR, CATALOG_FILE, CONFIG_FILE, TEMPLATE_DIR

Utility functions:
  _now_iso()              → ISO timestamp string
  _load_config()          → Read config.yaml
  _load_metadata(exp_dir) → Read experiment metadata.yaml
  _save_metadata(exp_dir, metadata) → Write metadata.yaml (automatically updates the 'updated' timestamp)
  _resolve_experiment(query) → Fuzzy-match experiment ID, returns (exp_id, exp_dir)
  _auto_detect_git()      → Auto-detect current git branch and commit hash
  _rebuild_catalog_data()    → Scan all experiments to build catalog dict
  _save_catalog(catalog)     → Write catalog.yaml

Command functions: cmd_new, cmd_add_data, cmd_add_script, cmd_run, cmd_list, cmd_show, cmd_info, cmd_rebuild_catalog, cmd_validate

CLI entry point: main() → argparse routes to command functions
```

## metadata.yaml Key Fields

- `data`: Dictionary where key is the logical name and value contains `file`, `description`, `added`, `environment` (machine, gpu), `provenance` (code_branch, code_commit)
- `outputs`: Dictionary where key is the output name and value contains `files` (list), `script`, `inputs` (list of data logical names), `description`, `created`
- `status`: `active` / `complete` / `abandoned`

## Script Template Mechanism

`templates/script_template.py` uses Python `str.format()` placeholder: `{description}`. Scripts receive a JSON argument via `sys.argv[1]` containing `exp_dir`, `output_dir`, and `inputs` (dict of logical_name → file_path). Generated scripts locate the project root via `PROJECT_ROOT = Path(__file__).resolve().parents[1]` to import `scripts.shared`.

## Script Run Mechanism

`./lab run <script> <exp>` passes a JSON string via `sys.argv[1]`:
```json
{"exp_dir": "...", "output_dir": "...", "inputs": {"name": "path", ...}}
```
- `--inputs` defaults to all data in the experiment
- `--name` defaults to the script name
- Output files are auto-discovered in the output directory and recorded in metadata

## Modification Notes

- **Adding a new data format:** Add a loader function to `_LOADERS` dict and extension mapping to `_EXT_TO_FORMAT` in `loaders.py`.
- **Adding a new CLI command:** Add an argparse subcommand in `main()`, implement a `cmd_xxx` function, and add it to the `commands` dictionary.
- **Every command that modifies metadata** should call `_rebuild_catalog_data()` + `_save_catalog()` to keep the catalog in sync.
- **After adding/modifying a feature**, update the usage instructions and directory structure in `README.md` accordingly.
