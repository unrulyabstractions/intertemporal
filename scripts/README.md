# scripts/ - Experiment Scripts

Main scripts for running experiments and analysis.

## Full Pipeline

The complete workflow consists of these steps:

```bash
# 1. Generate dataset with preference questions
uv run python scripts/generate_dataset.py --config cityhousing

# 2. Query LLM and capture responses + activations
uv run python scripts/query_llm.py --config default_query

# 3. Train probes on activations to predict choices
uv run python scripts/train_probes.py --config default_probes

# 4. Use probe directions to steer model behavior
uv run python scripts/try_steering.py --config default_steering

# 5. Analyze choices with value functions (optional)
uv run python scripts/analyze_choices.py out/preference_data/*.json --viz
```

## Scripts

### generate_dataset.py
Generate preference question datasets.

```bash
uv run python scripts/generate_dataset.py --config <name>
# Example:
uv run python scripts/generate_dataset.py --config cityhousing
```

**Output:** `out/datasets/<name>_<id>.json`

### query_llm.py
Query language models with preference questions and capture internal activations.

```bash
uv run python scripts/query_llm.py --config <name>
# Example:
uv run python scripts/query_llm.py --config default_query
```

**Output:**
- `out/preference_data/<name>_<model>_<query_id>.json` - Choices and metadata
- `out/internals/<name>_<model>_<query_id>_sample_*.pt` - Activations

### train_probes.py
Train linear probes on model activations to predict choices.

```bash
uv run python scripts/train_probes.py --config <name>
# Example:
uv run python scripts/train_probes.py --config default_probes
```

**Output:** `out/probes/<probe_config_id>/` - Trained probe weights and metrics

### try_steering.py
Apply activation steering using trained probe directions.

```bash
uv run python scripts/try_steering.py --config <name>
# Example:
uv run python scripts/try_steering.py --config default_steering
```

**Output:** `out/steering/<steering_config_id>/` - Steering results and flip rates

### analyze_choices.py
Analyze model choices and train value functions (discount models).

```bash
uv run python scripts/analyze_choices.py <preference_data_path> [--viz]
# Example:
uv run python scripts/analyze_choices.py out/preference_data/cityhousing_*.json --viz
```

**Output:** `out/choice_modeling/analysis_<name>_<timestamp>.json`

## Common Utilities (scripts/common/)

Shared modules used across all scripts. Import via `from common import ...`.

### Core Modules
- `prompt_builder.py` - Build prompts from questions and formatting configs
- `utils.py` - Response parsing, model quirks, memory management
- `output_io.py` - Load/save output files (datasets, preferences, probes)
- `schemas.py` - Output file schemas (PreferenceDataOutput, etc.)

### Configuration
- `config_utils.py` - Find datasets/configs by ID, resolve paths

### Analysis
- `verification.py` - Data verification and diagnostic plotting
- `consistency.py` - θ consistency analysis across time horizons
- `token_positions.py` - Token position resolution for activation capture

### Formatting
- `formatting_variation.py` - Label styles, time unit conversion, number spelling

## Configuration (scripts/configs/)

JSON configuration files organized by type:

```
configs/
├── dataset/          # Dataset generation configs
├── formatting/       # Prompt formatting configs
├── query/            # LLM query configs
├── probes/           # Probe training configs
├── steering/         # Steering experiment configs
└── choice_modeling/  # Value function analysis configs
```

Each folder has a `test/` subfolder with minimal test fixtures for quick iteration.
