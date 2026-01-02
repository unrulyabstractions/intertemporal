# Intertemporal Preference Research

Framework for studying how language models make intertemporal choices (short-term vs long-term rewards).

## Overview

This project studies how LLMs make choices between options with different time horizons. It provides tools to:

1. **Generate datasets** with intertemporal preference questions
2. **Query language models** and capture their choices + internal activations
3. **Train value function models** to fit observed choices
4. **Analyze consistency** of choices across time horizons
5. **Train probes** on model internals to predict choices

## Project Structure

```
intertemporal/
├── src/                          # Core library modules
│   ├── types.py                  # Core types (TimeValue, PreferencePair, etc.)
│   ├── schemas.py                # Config schemas (DatasetConfig, QueryConfig)
│   ├── dataset_generator.py      # DatasetGenerator class
│   ├── model_runner.py           # TransformerLens model wrapper
│   ├── value_function.py         # ValueFunction for choice prediction
│   ├── discount_function.py      # Discount functions (exp, hyperbolic)
│   └── io.py                     # I/O utilities
├── scripts/                      # Executable scripts
│   ├── generate_dataset.py       # Generate preference datasets
│   ├── query_llm.py              # Query LLMs with datasets
│   ├── analyze_choices.py        # Analyze and train value functions
│   ├── train_contrastive_probe.py # Train activation probes
│   ├── common/                   # Shared utilities
│   │   ├── formatting_variation.py
│   │   ├── utils.py
│   │   ├── output_io.py
│   │   ├── verification.py
│   │   └── consistency.py
│   └── configs/                  # Configuration files
│       ├── dataset/              # Dataset configs
│       ├── formatting/           # Prompt formatting configs
│       ├── query/                # LLM query configs
│       └── probes/               # Probe training configs
├── tests/                        # Test suite
│   ├── RUN_ALL_TESTS.py         # Main test runner
│   ├── src/                      # Tests for src/
│   └── scripts/                  # Tests for scripts/
└── out/                          # Generated outputs
    ├── datasets/                 # Generated datasets
    ├── preference_data/          # Query results
    ├── internals/                # Saved activations
    └── choice_modeling/          # Analysis results
```

## Installation

```bash
# Install with uv
uv sync

# Install dev dependencies (pytest, coverage)
uv sync --extra dev
```

## Quick Start

### 1. Generate a Dataset

```bash
uv run python scripts/generate_dataset.py --config cityhousing
```

Creates: `out/datasets/cityhousing_<id>.json`

### 2. Query a Language Model

```bash
uv run python scripts/query_llm.py --config default_query
```

Creates: `out/preference_data/cityhousing_<model>_<query_id>.json`

### 3. Analyze Choices

```bash
uv run python scripts/analyze_choices.py out/preference_data/cityhousing_*.json --viz
```

Creates: `out/choice_modeling/analysis_cityhousing_<timestamp>.json`

## Configuration

### Dataset Config (`scripts/configs/dataset/`)

```json
{
  "name": "my_dataset",
  "context": {
    "reward_unit": "dollars",
    "role": "an investor",
    "situation": "Choose an investment.",
    "labels": ["a)", "b)"],
    "seed": 42
  },
  "options": {
    "short_term": {
      "reward_range": [100, 500],
      "time_range": [[1, "months"], [6, "months"]],
      "reward_steps": [2, "linear"],
      "time_steps": [2, "linear"]
    },
    "long_term": {
      "reward_range": [1000, 5000],
      "time_range": [[1, "years"], [10, "years"]],
      "reward_steps": [2, "logarithmic"],
      "time_steps": [2, "logarithmic"]
    }
  },
  "time_horizons": [null, [6, "months"], [5, "years"]],
  "add_formatting_variations": false
}
```

### Formatting Variations

When `add_formatting_variations: true`:
- Random label pairs: ("a)", "b)"), ("x)", "y)"), ("[1]", "[2]"), etc.
- Order flipping: short/long term randomly on left/right
- Time unit conversion: 1 year → 12 months → 365 days
- Number spelling: 1 → "one", 0.5 → "half"

## Key Concepts

### Value Function

Options are evaluated as:
```
U(option; θ) = utility(reward) × discount(time; θ)
```

### Discount Functions

- **Exponential**: `D(t) = exp(-θt)` - Constant discount rate
- **Hyperbolic**: `D(t) = 1/(1 + θt)` - Decreasing impatience

### Choice Prediction

The option with higher value is predicted:
```
predicted_choice = argmax U(option_i; θ)
```

### Training

Gradient descent on θ to minimize binary cross-entropy loss on observed choices.

## Running Tests

```bash
# Run all tests with coverage
python tests/RUN_ALL_TESTS.py

# Quick run without coverage
python tests/RUN_ALL_TESTS.py --quick

# Run specific tests
uv run pytest tests/src/test_types.py -v
```

## API Usage

```python
from src.dataset_generator import DatasetGenerator
from src.choice_model.value_function import ValueFunction
from src.types import DiscountType, UtilityType

# Generate dataset
gen = DatasetGenerator(
    "scripts/configs/dataset/cityhousing.json",
    "scripts/configs/formatting/default_formatting.json"
)
samples = gen.generate()

# Create and train value function
vf = create_value_function(
    utility_type=UtilityType.LINEAR,
    discount_type=DiscountType.EXPONENTIAL,
    theta=0.1
)
result = vf.train(training_samples, learning_rate=0.01)

# Predict choice
prediction = vf.predict(question)  # "short_term" or "long_term"
```

## References

Based on research exploring how LLMs encode time horizon preferences in intertemporal choice tasks.
