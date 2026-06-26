# SPNI Orchestration Test Plan

This file describes the future test tree for the SPNI orchestration refactor.
It is intentionally non-executable and does not modify the current test suite.

## Proposed unit tree

`tests/unit/simulation/spni/`
- `test_config.py`
- `test_build.py`
- `test_data.py`
- `test_train.py`
- `test_evaluate.py`
- `test_results.py`
- `test_pipeline.py`

## Unit test cores

### `test_config.py`

Expectation cores:
- `build_run_config` validates required fields and normalizes options.
- `derive_seed_bundle` is deterministic for the same input.
- `derive_seed_sweep` returns a stable ordered list of unique seed bundles.
- `describe_run` returns only compact serializable fields.

### `test_build.py`

Expectation cores:
- synthetic configuration builds a grid graph
- real-world configuration delegates to CSV import
- optimization model construction is isolated from graph construction
- graph metadata records source and kind correctly

### `test_data.py`

Expectation cores:
- base synthetic data has expected shapes
- train/val/test split counts are exact
- normalization constant is stored and reused
- adverse/random/baseline loaders are all present
- baseline loaders come from scenario-zero non-adverse views

### `test_train.py`

Expectation cores:
- all predictor families are produced exactly once
- cache-tag assignment is deterministic
- returned logs have the expected curve lengths
- predictor bundle labels match the requested families

### `test_evaluate.py`

Expectation cores:
- uninterdicted outputs are aligned with test samples
- symmetric outputs are aligned with test samples
- asymmetric failures are represented explicitly, not dropped
- wrong-model evaluation can be skipped cleanly
- diagnostics expose skipped or failed solve counts

### `test_results.py`

Expectation cores:
- summary metrics match raw evaluation arrays
- legacy `all_data` export contains the required keys
- row flattening preserves simulation and sample indices
- divide-by-zero cases produce safe numeric outputs

### `test_pipeline.py`

Expectation cores:
- stage call order is fixed and explicit
- outputs from one stage are passed unchanged into the next
- optional persistence hooks are disabled by default
- `SimulationResult` is fully populated on success

## Integration tree

`tests/integration/simulation/spni/`
- `test_single_run_smoke.py`
- `test_seed_sweep_smoke.py`

## Integration test cores

### `test_single_run_smoke.py`

Expectation cores:
- tiny synthetic config completes one full run
- all four predictor families are present
- summary output is non-empty and internally consistent

### `test_seed_sweep_smoke.py`

Expectation cores:
- tiny two-seed sweep returns two single-run results
- seed bundles differ in the expected way
- aggregated summary combines runs without shape drift

