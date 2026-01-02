# scripts/common/ Module Specifications

This document specifies the behavior of utility modules in `scripts/common/`.

---

## formatting_variation.py - Formatting Variations

### LABEL_STYLES
List of valid label pairs for option identification.

**Invariants:**
- All pairs have distinct left and right labels
- Common styles: ("a)", "b)"), ("x)", "y)"), ("[1]", "[2]"), etc.
- No same-label pairs (e.g., ("-", "-") is forbidden - makes parsing impossible)

### Time Unit Conversion

**Conversion factors to years:**
| Unit | Factor |
|------|--------|
| years | 1.0 |
| months | 1/12 |
| weeks | 1/52.14 |
| days | 1/365 |
| hours | 1/8760 |
| decades | 10 |

**Functions:**
- `convert_time_value(tv, target_unit) -> TimeValue`
  - Converts time value to target unit
  - Preserves total time duration

- `get_sensible_units_for_time(tv) -> list[str]`
  - Returns units that give reasonable numeric values
  - Filters out extreme conversions (e.g., days for decades)

### Number Spelling

**spell_number(n) -> str | None**
Converts numbers to words.

| Value | Result |
|-------|--------|
| 0-12 | "zero" to "twelve" |
| 0.5 | "half" |
| 0.25 | "a quarter" |
| Other | None |

**format_time_spelled(tv) -> str | None**
Formats time with spelled number.

Examples:
- (1, "years") → "one year"
- (2, "months") → "two months"
- (0.5, "decades") → "half a decade"

### FormattingVariation

**Dataclass fields:**
- `labels: tuple[str, str]` - Label pair to use
- `flip_order: bool` - Swap short/long term positions
- `time_unit_variation: bool` - Convert to different time unit
- `spell_numbers: bool` - Use spelled numbers

**Class methods:**
- `default()` - Returns no-change variation
- `random(allow_all=True)` - Returns randomized variation

---

## utils.py - Response Parsing Utilities

### parse_label_from_response

**Signature:**
```python
parse_label_from_response(
    text: str,
    labels: list[str],
    choice_prefix: str,
    model_name: str | None = None
) -> str | None
```

**Behavior:**
1. Applies model-specific preprocessing (markdown stripping)
2. Searches for patterns (case-insensitive):
   - `{choice_prefix}: {label}` (e.g., "I choose: a)")
   - `option {label}` (e.g., "option a)")
   - `choice {label}` (e.g., "choice a)")
   - Label at start of response (e.g., "a). Because...")
3. Returns matched label or None

### determine_choice

**Signature:**
```python
determine_choice(
    chosen_label: str | None,
    short_term_label: str,
    long_term_label: str
) -> str
```

**Returns:**
- `"short_term"` if chosen_label matches short_term_label
- `"long_term"` if chosen_label matches long_term_label
- `"unknown"` if None or no match

**Note:** Matching is case-insensitive.

### strip_markdown

**Signature:**
```python
strip_markdown(text: str) -> str
```

**Removes:**
- Bold: `**text**`, `__text__`
- Italic: `*text*`, `_text_`

---

## output_io.py - I/O Functions

### Dataset I/O
- `save_dataset(path, metadata, questions)` - Save generated dataset
- `load_dataset_output(path) -> DatasetOutput` - Load dataset

### Preference Data I/O
- `save_preference_data(path, metadata, preferences)` - Save query results
- `load_preference_data(path) -> PreferenceDataOutput` - Load preferences

### Training I/O
- `save_train_output(path, output)` - Save training results
- `load_train_output(path) -> TrainOutput` - Load training results

---

## config_utils.py - Config Utilities

### find_dataset_by_id
```python
find_dataset_by_id(dataset_id: str) -> Path | None
```
Searches `out/datasets/` for matching dataset file.

### find_preference_data_by_query_id
```python
find_preference_data_by_query_id(query_id: str, model: str) -> Path | None
```
Searches `out/preference_data/` for matching file.

### get_expected_choice
```python
get_expected_choice(
    time_horizon: TimeValue | None,
    short_time: TimeValue,
    long_time: TimeValue
) -> str
```
Determines expected rational choice based on time horizon:
- Horizon < short_time → neither accessible → "long_term" (patient)
- Horizon < long_time → only short accessible → "short_term"
- Horizon ≥ long_time → both accessible → "long_term" (patient)

---

## verification.py - Data Verification

### verify_data_alignment
Analyzes choice patterns across time horizon categories.

**Returns DataVerification with:**
- Per-category statistics (short/long term counts)
- Alignment rates (does model match expected behavior?)
- Overall accuracy

### Categories:
- `short_horizon`: Time horizon < 1 year
- `medium_horizon`: 1 year ≤ horizon < 5 years
- `long_horizon`: Horizon ≥ 5 years
- `no_horizon`: No time constraint

---

## consistency.py - Consistency Analysis

### ThetaConstraint
Represents constraint on discount parameter θ from a single choice.

**For exponential discounting:**
- "short_term" choice → θ > threshold
- "long_term" choice → θ < threshold
- Threshold computed from indifference point

### ConsistencyAnalysis
Analyzes whether choices are consistent with any single θ.

**Fields:**
- `is_consistent: bool` - Whether feasible θ range exists
- `theta_range: tuple[float, float]` - Feasible range
- `best_theta: float` - Optimal θ if consistent
- `best_accuracy: float` - Accuracy at best θ
- `conflicting_pairs: list` - Contradictory constraint pairs

---

## schemas.py - Output Schemas

### DatasetOutput
```python
{
    "metadata": DatasetOutputMetadata,
    "questions": list[QuestionOutput]
}
```

### PreferenceDataOutput
```python
{
    "metadata": PreferenceDataMetadata,
    "preferences": list[PreferenceItem]
}
```

### PreferenceItem
```python
{
    "sample_id": int,
    "time_horizon": [value, unit] | null,
    "preference_pair": PreferencePairOutput,
    "choice": "short_term" | "long_term" | "unknown",
    "internals": InternalsReference | null,
    "debug": PreferenceDebugInfo | null
}
```
