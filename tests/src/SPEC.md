# src/ Module Specifications

This document specifies the behavior of core modules in `src/`.

---

## types.py - Core Type Definitions

### TimeValue
A time value with unit for temporal calculations.

**Properties:**
- `value: float` - Numeric time amount
- `unit: str` - Time unit ("years", "months", "weeks", "days", "hours", "decades")

**Methods:**
- `to_years() -> float` - Convert to years (standard unit for calculations)
- `to_list() -> list` - Return `[value, unit]` for JSON serialization

**Conversion factors:**
| Unit | To Years |
|------|----------|
| years | 1.0 |
| months | 1/12 |
| weeks | 1/52.14 |
| days | 1/365 |
| hours | 1/8760 |
| decades | 10 |

### RewardValue
A reward amount with optional unit.

**Properties:**
- `value: float` - Numeric reward amount
- `unit: str = "units"` - Unit description (e.g., "dollars", "points")

### IntertemporalOption
A single option in a preference pair.

**Properties:**
- `label: str` - Display label (e.g., "a)", "Option 1:")
- `time: TimeValue` - When reward is received
- `reward: RewardValue` - Amount of reward

### PreferencePair
A pair of options for comparison.

**Properties:**
- `short_term: IntertemporalOption` - Smaller/sooner option
- `long_term: IntertemporalOption` - Larger/later option

---

## discount_function.py - Discount Functions

### DiscountFunction (abstract)
Base class for temporal discount functions D(t) ∈ [0, 1].

**Methods:**
- `__call__(t: float) -> float` - Compute discount factor at time t (years)
- `discount(time: TimeValue) -> float` - Compute discount for TimeValue

### ExponentialDiscount
D(t) = exp(-θt)

**Parameters:**
- `theta: float` - Discount rate (higher = more discounting)

**Properties:**
- D(0) = 1 (no discounting at t=0)
- D(t) monotonically decreasing
- Higher theta → steeper decline

### HyperbolicDiscount
D(t) = 1 / (1 + θt)

**Parameters:**
- `theta: float` - Discount rate

**Properties:**
- D(0) = 1
- Exhibits present bias (steeper early, flatter late)

### QuasiHyperbolicDiscount
D(t) = β * δ^t for t > 0, D(0) = 1

**Parameters:**
- `beta: float` - Present bias factor (0 < β ≤ 1)
- `delta: float` - Per-period discount factor (0 < δ ≤ 1)

---

## value_function.py - Value Functions and Choice Models

### UtilityFunction (abstract)
Base class for utility functions u(r).

**Implementations:**
- `LinearUtility`: u(r) = r
- `LogUtility`: u(r) = log(r), returns -∞ for r ≤ 0
- `PowerUtility(alpha)`: u(r) = r^α

### ValueFunction
Computes option values: U(o) = u(r) * D(t)

**Methods:**
- `evaluate(option: IntertemporalOption) -> float` - Compute value
- `predict(question: PreferenceQuestion) -> str` - Predict "short_term" or "long_term"
- `train(samples, learning_rate, num_iterations) -> TrainingResult` - Fit parameters

**Choice Prediction:**
- Compare U(short_term) vs U(long_term)
- Return option with higher value
- Ties resolved by convention

**Training:**
- Binary cross-entropy loss
- Gradient descent on theta parameter
- Returns fitted parameters and final loss

---

## dataset_generator.py - Dataset Generation

### DatasetGenerator
Generates intertemporal preference datasets from config.

**Initialization:**
- Load dataset config (options, time horizons, context)
- Load formatting config (templates, placeholders)

**Methods:**
- `generate() -> list[DatasetSample]` - Generate all samples
- `create_sample(...) -> DatasetSample` - Create single sample
- `format_prompt(...) -> str` - Format question text

**Grid Generation:**
- Cartesian product of reward × time steps
- Linear or logarithmic stepping
- All combinations with each time horizon

**Formatting Variations (when enabled):**
- Random label pairs from LABEL_STYLES
- Random order flipping (short/long on left/right)
- Time unit conversion (1 year → 12 months)
- Number spelling (1 → "one")

---

## schemas.py - Config Schemas

### DatasetConfig
Configuration for dataset generation.

**Required fields:**
- `name: str` - Dataset identifier
- `context: ContextConfig` - Role, situation, labels
- `options: dict` - short_term and long_term OptionRangeConfig
- `time_horizons: list` - List of [value, unit] or null

**Optional fields:**
- `add_formatting_variations: bool = False` - Enable random formatting

### OptionRangeConfig
Range specification for an option type.

**Fields:**
- `reward_range: [min, max]`
- `time_range: [[min_val, min_unit], [max_val, max_unit]]`
- `reward_steps: [num_intervals, step_type]`
- `time_steps: [num_intervals, step_type]`

### FormattingConfig
Prompt formatting templates.

**Placeholders:**
- `[SITUATION]`, `[ROLE]`, `[ACTION_IN_QUESTION]`
- `[LEFT_TERM_LABEL]`, `[LEFT_TERM_REWARD]`, `[LEFT_TERM_TIME]`
- `[RIGHT_TERM_LABEL]`, `[RIGHT_TERM_REWARD]`, `[RIGHT_TERM_TIME]`
- `[TIME_HORIZON]`, `[TIME_HORIZON_SPEC]`
- `[CHOICE_PREFIX]`, `[MAX_REASONING_LENGTH]`
