# scripts/ Module Specifications

This document specifies the behavior of main scripts in `scripts/`.

---

## generate_dataset.py

Generates intertemporal preference datasets from configuration.

### Usage
```bash
python scripts/generate_dataset.py --config <config_name>
```

### Inputs
- Config file: `scripts/configs/dataset/<config_name>.json`
- Formatting config: `scripts/configs/formatting/<formatting_name>.json`

### Outputs
- Dataset file: `out/datasets/<name>_<dataset_id>.json`

### Behavior
1. Load dataset config and formatting config
2. Generate grid of (reward, time) combinations
3. For each combination × time horizon:
   - Create question with formatted prompt
   - Apply formatting variations if enabled
4. Save dataset with metadata

### Dataset ID
- Computed from hash of config content
- Same config always produces same ID
- Used for linking datasets to query results

---

## query_llm.py

Queries a language model with preference questions.

### Usage
```bash
python scripts/query_llm.py --config <config_name>
```

### Inputs
- Query config: `scripts/configs/query/<config_name>.json`
- Dataset: `out/datasets/<name>_<dataset_id>.json`

### Outputs
- Preference data: `out/preference_data/<name>_<model>_<query_id>.json`
- Internals (optional): `out/internals/<name>_<model>_<query_id>_sample_*.pt`

### Behavior
1. Load dataset and formatting config
2. Initialize model with TransformerLens
3. For each question:
   - Build prompt: question + response format
   - Run inference (optionally capture internals)
   - Parse response to extract chosen label
   - Map label to "short_term" / "long_term" / "unknown"
4. Save preferences with metadata

### Label Parsing
- Uses question-specific labels (important for variations)
- Searches for choice_prefix pattern
- Falls back to "option X" / "choice X" patterns
- Returns "unknown" if no match found

---

## analyze_choices.py

Analyzes model choices and trains value function models.

### Usage
```bash
python scripts/analyze_choices.py <train_data> [--test <test_data>] [--viz]
```

### Inputs
- Train data: Preference data JSON path
- Test data (optional): Separate test set

### Outputs
- Analysis: `out/choice_modeling/analysis_<name>_<timestamp>.json`
- Plots (optional): Consistency and verification visualizations

### Behavior
1. Load preference data
2. Group samples by time horizon bucket
3. For each horizon:
   - Train value function (fit θ parameter)
   - Evaluate on test set
4. Compute consistency analysis
5. Generate verification plots if --viz

### Time Horizon Buckets
- Based on formatted time horizon string
- "no_horizon" for samples without constraint
- e.g., "4 months", "1 years", "10 years"

### Consistency Analysis
- Checks if all choices can be explained by single θ
- Identifies conflicting constraint pairs
- Computes best achievable accuracy

---

## train_contrastive_probe.py

Trains linear probes on model activations to predict choices.

### Usage
```bash
python scripts/train_contrastive_probe.py --config <config_name>
```

### Inputs
- Config: `scripts/configs/probes/<config_name>.json`
- Preference data with internals

### Outputs
- Probe results: Console output and optional JSON

### Behavior
1. Load preference data with saved activations
2. For each specified layer:
   - Extract activation vectors
   - Train logistic regression classifier
   - Evaluate with cross-validation
3. Report accuracy per layer
4. Identify best performing layer

---

## Config File Specifications

### Dataset Config
```json
{
  "name": "string",
  "context": {
    "reward_unit": "string",
    "role": "string",
    "situation": "string",
    "action_in_question": "string",
    "reasoning_ask": "string",
    "domain": "string",
    "labels": ["string", "string"],
    "method": "grid" | "random",
    "seed": number
  },
  "options": {
    "short_term": { OptionRangeConfig },
    "long_term": { OptionRangeConfig }
  },
  "time_horizons": [null | [number, "unit"], ...],
  "add_formatting_variations": boolean
}
```

### Query Config
```json
{
  "models": ["model_name", ...],
  "dataset": {
    "name": "string",
    "id": "dataset_id"
  },
  "formatting": {
    "name": "formatting_config_name"
  },
  "decoding": {
    "max_new_tokens": number,
    "temperature": number,
    "top_k": number,
    "top_p": number
  },
  "internals": {
    "resid_post": { "layers": [int, ...] }
  },
  "token_positions": [int | "string", ...]
}
```

### Formatting Config
```json
{
  "question_template": "string with [PLACEHOLDERS]",
  "response_format": "string with [PLACEHOLDERS]",
  "choice_prefix": "string",
  "time_horizon_spec": "string",
  "max_reasoning_length": "string"
}
```
