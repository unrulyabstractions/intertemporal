# Configuration Files

This directory contains JSON configuration files for the intertemporal preference research framework.

## Directory Structure

```
configs/
├── dataset/          # Dataset generation configs
├── formatting/       # Prompt formatting configs
├── query/            # LLM query configs
└── choice_modeling/  # Choice model analysis configs
```

## Config Types and IDs

Each config type has an associated deterministic ID computed from its content:

| Config Type | ID Name | Usage |
|------------|---------|-------|
| Dataset | `dataset_id` | Identifies dataset generation parameters |
| Formatting | `formatting_id` | Identifies prompt formatting |
| Query | `query_id` | Identifies query parameters (model, decoding, internals) |

IDs are computed via `config.get_id()` using a blake2b hash of the config content.

## Output File Naming

### Dataset Output
```
out/datasets/{dataset_name}_{dataset_id}.json
```

### Preference Data Output
```
out/preference_data/{dataset_name}_{model}_{query_id}.json
```

### Internals Output
```
out/internals/{dataset_name}_{model}_{query_id}_sample_{sample_id}.pt
```

## Config Schemas

### Dataset Config (`dataset/*.json`)
Defines the scenario, options, and time horizons for preference questions.

```json
{
  "name": "cityhousing",
  "context": {
    "reward_unit": "housing units",
    "role": "the city administration",
    "situation": "Plan for housing in the city.",
    "action_in_question": "build",
    "reasoning_ask": "why choice was made",
    "domain": "housing",
    "labels": ["a)", "b)"],
    "method": "grid",
    "seed": 42
  },
  "options": {
    "short_term": {
      "reward_range": [2000, 5000],
      "time_range": [[3, "months"], [1, "years"]],
      "reward_steps": [0, "linear"],
      "time_steps": [0, "linear"]
    },
    "long_term": { ... }
  },
  "time_horizons": [[5, "months"], [15, "years"]]
}
```

### Formatting Config (`formatting/*.json`)
Defines prompt templates and response format.

```json
{
  "question_template": "Situation: [SITUATION]\nTask: You, [ROLE]...",
  "response_format": "\n\nRespond in this format:\n[CHOICE_PREFIX] <...>",
  "choice_prefix": "I choose:",
  "max_reasoning_length": "1-2 sentences"
}
```

### Query Config (`query/*.json`)
Defines LLM query parameters including model, dataset reference, and internals capture.

```json
{
  "models": ["Qwen/Qwen2.5-7B-Instruct"],
  "dataset": {
    "name": "cityhousing",
    "id": "d4833b249c1ebee1e7260cd13cee1e7c"
  },
  "formatting": {
    "name": "default_formatting"
  },
  "decoding": {
    "max_new_tokens": 250,
    "temperature": 0.0,
    "top_k": 0,
    "top_p": 1.0
  },
  "internals": {
    "resid_post": {
      "layers": [8, 14, -8, -4, -2]
    }
  },
  "token_positions": [0, "I choose:"]
}
```

### Choice Modeling Config (`choice_modeling/*.json`)
Defines training/testing data and models for choice analysis.

```json
{
  "train_data": {
    "name": "cityhousing",
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "query_id": "c4732fa2a21c6f91..."
  },
  "test_data": {
    "name": "cityhousing",
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "query_id": "c4732fa2a21c6f91..."
  },
  "choice_models": [
    { "utility_type": "linear", "discount_type": "exponential" },
    { "utility_type": "linear", "discount_type": "hyperbolic" }
  ],
  "learning_rate": 0.01,
  "num_iterations": 100,
  "temperature": 1.0
}
```

## ID Tracking in Outputs

Preference data metadata includes all relevant IDs and configs for reproducibility:

```json
{
  "metadata": {
    "version": "1.0",
    "dataset_id": "d4833b249c1ebee1e7260cd13cee1e7c",
    "formatting_id": "a1b2c3d4e5f6...",
    "query_id": "f7e8d9c0b1a2...",
    "model": "Qwen2.5-7B-Instruct",
    "query_config": {
      "models": ["Qwen/Qwen2.5-7B-Instruct"],
      "dataset_name": "cityhousing",
      "dataset_id": "d4833b249c1ebee1e7260cd13cee1e7c",
      "formatting_name": "default_formatting",
      "formatting_id": "a1b2c3d4e5f6...",
      "decoding": { ... },
      "internals": { ... }
    }
  }
}
```
