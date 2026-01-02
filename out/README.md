# out/ - Output Directory

Generated outputs from experiments.

**Note:** Most contents are gitignored except `*/test/` fixtures.

## Structure

```
out/
├── datasets/           # Generated datasets
│   ├── <name>_<id>.json
│   └── test/          # Test fixtures (tracked in git)
├── preference_data/    # Model query results
│   ├── <name>_<model>_<query_id>.json
│   └── test/          # Test fixtures
├── internals/          # Saved activations
│   ├── <name>_<model>_<query_id>_sample_*.pt
│   └── test/          # Test fixtures
└── choice_modeling/    # Analysis results
    ├── analysis_<name>_<timestamp>.json
    └── test/          # Test fixtures
```

## File Formats

### Dataset Output
`<name>_<dataset_id>.json`
```json
{
  "metadata": {
    "version": "1.0",
    "dataset_id": "...",
    "config": { ... },
    "num_questions": 48
  },
  "questions": [...]
}
```

### Preference Data
`<name>_<model>_<query_id>.json`
```json
{
  "metadata": {
    "dataset_id": "...",
    "query_id": "...",
    "model": "..."
  },
  "preferences": [
    {
      "sample_id": 0,
      "choice": "short_term|long_term|unknown",
      "preference_pair": {...}
    }
  ]
}
```

### Internals
`<name>_<model>_<query_id>_sample_<n>.pt`

PyTorch tensors of activation vectors at specified layers/positions.

### Analysis Output
`analysis_<name>_<timestamp>.json`
```json
{
  "train_data": "...",
  "test_data": "...",
  "models": {
    "exponential": [
      {"train_horizon": "...", "theta": 0.1, "test_results": {...}}
    ]
  }
}
```

## ID Linking

Files are linked by IDs:
- `dataset_id`: Hash of dataset config (reproducible)
- `query_id`: Hash of query config
- `formatting_id`: Hash of formatting config

Example chain:
1. `cityhousing_d955f857.json` (dataset)
2. `cityhousing_Qwen2.5_b6dcc993.json` (preferences, links to dataset_id)
3. `analysis_cityhousing_20251230.json` (references preference data path)
