# src/ - Core Library

Core library modules for intertemporal preference research.

## Modules

### types.py
Core type definitions used throughout the codebase.

- `TimeValue` - Time with unit (years, months, days, etc.)
- `RewardValue` - Reward amount with unit
- `IntertemporalOption` - Single choice option (label, time, reward)
- `PreferencePair` - Short-term vs long-term options
- `PreferenceQuestion` - Full question with optional time horizon
- `TrainingSample` - Question + observed choice

### discount_function.py
Temporal discount functions D(t) ∈ [0, 1].

- `ExponentialDiscount` - D(t) = exp(-θt)
- `HyperbolicDiscount` - D(t) = 1/(1+θt)
- `QuasiHyperbolicDiscount` - D(t) = β*δ^t

### value_function.py
Value functions for choice prediction.

U(option) = utility(reward) × discount(time)

- `LinearUtility`, `LogUtility`, `PowerUtility`
- `ValueFunction` - Evaluates options, predicts choices
- Training via gradient descent on θ parameter

### dataset_generator.py
Generates preference datasets from configuration.

- Grid-based sampling (reward × time combinations)
- Linear or logarithmic stepping
- Optional formatting variations (labels, time units)

### schemas.py
Configuration schemas for JSON configs.

- `DatasetConfig` - Dataset generation settings
- `FormattingConfig` - Prompt templates
- `QueryConfig` - LLM query settings
- `DecodingConfig` - Generation parameters

### model_runner.py
TransformerLens-based model inference.

- Load HuggingFace models via TransformerLens
- Run inference with decoding parameters
- Capture internal activations at specified layers/positions

### io.py
File I/O utilities.

- `load_json()`, `save_json()`
- `ensure_dir()` - Create directories if needed

### schema_utils.py
Base classes for schema validation.

- `SchemaClass` - Dataclass with validation

### profiling.py
Performance profiling utilities.

- Timer context manager
- Memory tracking
