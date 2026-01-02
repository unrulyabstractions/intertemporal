# tests/ - Test Suite

Comprehensive test suite for the intertemporal preference research framework.

## Running Tests

```bash
# Run all tests with coverage
python tests/RUN_ALL_TESTS.py

# Quick run without coverage
python tests/RUN_ALL_TESTS.py --quick

# Generate HTML coverage report
python tests/RUN_ALL_TESTS.py --html

# Run specific test file
uv run pytest tests/src/test_types.py -v

# Run specific test
uv run pytest tests/src/test_types.py::TestTimeValue::test_to_years_from_months -v
```

## Structure

```
tests/
├── RUN_ALL_TESTS.py      # Main test runner with coverage
├── conftest.py           # Shared fixtures
├── src/                  # Tests for src/ modules
│   ├── SPEC.md          # Module specifications
│   ├── test_types.py
│   ├── test_discount_function.py
│   ├── test_value_function.py
│   └── test_dataset_generator.py
└── scripts/              # Tests for scripts/ modules
    ├── SPEC.md          # Script specifications
    └── common/
        ├── SPEC.md      # Utility module specifications
        ├── test_formatting_variation.py
        └── test_utils.py
```

## Test Fixtures

### Configuration Fixtures
Located in `scripts/configs/*/test/`:
- `dataset/test/` - Test dataset configs
- `formatting/test/` - Test formatting configs
- `query/test/` - Test query configs
- `probes/test/` - Test probe configs

### Output Fixtures
Located in `out/*/test/`:
- `datasets/test/` - Sample generated datasets
- `preference_data/test/` - Sample preference data
- `choice_modeling/test/` - Sample analysis outputs

## Specifications

Each test folder contains a `SPEC.md` documenting:
- Module/script behavior
- Input/output formats
- Edge cases and invariants
- Expected behavior for each function

Tests are designed to verify these specifications.

## Writing Tests

1. Add test file matching module: `test_<module>.py`
2. Use fixtures from `conftest.py`
3. Follow pytest conventions
4. Test both happy path and edge cases
5. Update SPEC.md when adding new functionality
