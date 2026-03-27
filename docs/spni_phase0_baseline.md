# SPNI Phase 0 Baseline

This document records the current SPNI orchestration baseline before the
orchestration-layer refactor is implemented.

Date recorded:
- 2026-03-26

Recorded against the current live code in:
- `src/dflintdpy/scripts/asym_spni_single_sim.py`
- `src/dflintdpy/scripts/Asym_SPNI_Simulator.py`
- `src/dflintdpy/scripts/setup.py`
- `src/dflintdpy/scripts/compare.py`
- `src/dflintdpy/utils/read_write_results.py`
- `src/dflintdpy/utils/analyse_results.py`

## 1. Current SPNI entrypoints

### `src/dflintdpy/scripts/asym_spni_single_sim.py`

Primary entrypoint:
- `single_sim(cfg, visualize=False, compute_asym_intd_2=True,
  compute_asym_intd=True)`

Current responsibilities:
- set RNG seeds from `cfg.random_seed`
- construct the synthetic grid graph and shortest-path model
- build adversarial and random SPNI training data
- derive a non-adverse training view from the adversarial loaders
- train or load four predictor families:
  - PFL
  - baseline DFL
  - random adverse DFL
  - adversarial DFL
- generate evaluation interdiction samples
- evaluate predictors under:
  - no interdiction
  - symmetric interdiction
  - asymmetric interdiction
  - optional wrong-model asymmetric experiments
- compute printed summary tables
- compute export dictionaries returned to the caller

Current side effects:
- prints training progress
- prints result tables with `tabulate`
- loads or writes cached datasets/predictors indirectly through setup helpers

### `src/dflintdpy/scripts/Asym_SPNI_Simulator.py`

Current role:
- top-level executable script with module-level side effects
- no wrapper function or `main()`

Current responsibilities:
- instantiate default `HP`
- set global cache replacement policy
- derive one random-seed triplet per sweep iteration
- mutate the shared `cfg` object in-place per sweep iteration
- call `single_sim(...)`
- collect one result dict per run
- write CSV results
- invoke analysis after the sweep

Current side effects:
- executes immediately on import/run
- writes a CSV file into `results/`
- calls `analyze_results()`
- prints progress and completion messages

### `src/dflintdpy/scripts/setup.py`

This file currently owns three orchestration roles.

#### `gen_train_data(...)`

Current responsibilities:
- load or generate synthetic base data
- normalize costs
- split train/test
- generate grouped SPNI adverse scenarios
- split grouped training data into train/validation
- create `AdvDataset` and `AdvLoader`
- return loader dicts plus testing data and generator metadata

Current side effects:
- reads/writes cached dataset artefacts
- may read/write cached interdiction artefacts indirectly through
  `SPNIAdverseDataGenerator`
- prints when cached or generated data is used

#### `gen_data(...)`

Current responsibilities:
- generate evaluation-time synthetic data
- normalize costs
- return a dict with `features` and `costs`

Current note:
- This function returns `features`, not `feats`, unlike `gen_train_data`.

#### `setup_pfl_predictor(...)` and `setup_dfl_predictor(...)`

Current responsibilities:
- build predictor architectures
- optionally load cached predictor state
- configure losses and optimizers
- train the predictor if not loaded
- optionally plot learning curves
- write predictor state to cache

Current side effects:
- loads/writes cached predictor artefacts
- prints training/caching status
- optionally writes figures

### `src/dflintdpy/scripts/compare.py`

Current role:
- low-level evaluation helper module called by `single_sim`

Current responsibilities by function:

#### `compare_shortest_paths(...)`
- evaluate oracle and predictor-selected shortest paths without interdiction
- return four parallel Python lists:
  - `true_objs`
  - `pfl_objs`
  - `dfl_objs`
  - `adfl_objs`

#### `compare_sym_intd(...)`
- evaluate predictor behavior under symmetric SPNI interdiction
- return a dict of NumPy arrays
- included keys depend on which optional predictors are supplied

#### `compare_asym_intd(...)`
- evaluate predictor behavior under asymmetric SPNI interdiction
- returns two NumPy arrays:
  - estimated follower objective
  - oracle follower objective
- currently preserves one output slot per input sample using `NaN`
  placeholders when asymmetric solves fail

#### `compare_wrong_asym_intd(...)`
- evaluate asymmetric interdiction under leader/follower model mismatch
- returns only the estimated objective array
- does not preserve input sample alignment on failure
- does not explicitly handle failed asymmetric solves

## 2. Current orchestration call graph

The current single-run flow is:

1. `single_sim(...)`
2. construct `Grid` and `ShortestPathGrb`
3. `gen_train_data(..., interdiction_policy="adversarial")`
4. `gen_train_data(..., interdiction_policy="random")`
5. `setup_pfl_predictor(...)`
6. `setup_dfl_predictor(..., cache_tag="adfl")`
7. `setup_dfl_predictor(..., cache_tag="rdfl")`
8. derive non-adverse loaders from adversarial loaders
9. `setup_dfl_predictor(..., cache_tag="dfl")`
10. `gen_data(...)`
11. `compare_shortest_paths(...)`
12. `compare_sym_intd(...)`
13. `compare_asym_intd(...)` for each predictor family if enabled
14. `compare_wrong_asym_intd(...)` for each model-pair experiment if enabled
15. compute summary dicts and return them

The current sweep flow is:

1. instantiate one shared `HP`
2. set global cache replacement policy
3. loop over `seed in range(seed_0, seed_0 + num_seeds)`
4. derive `seed1, seed2, seed3` using NumPy RNG
5. mutate `cfg.seed`, `cfg.random_seed`, `cfg.intd_seed`, `cfg.loader_seed`
6. call `single_sim(...)`
7. append one result dict to `results`
8. write sweep CSV via `save_results_to_csv(...)`
9. call `analyze_results()`

## 3. Current input contracts

### `single_sim(...)`

Inputs:
- `cfg`: config-like object with `.get(...)`
- `visualize`: controls predictor plotting and verbose training output
- `compute_asym_intd`: whether standard asymmetric experiments run
- `compute_asym_intd_2`: whether wrong-model asymmetric experiments run

Expected config fields used directly or indirectly:
- `grid_size`
- `random_seed`
- `intd_seed`
- `loader_seed`
- `budget`
- `num_scenarios`
- `num_features`
- `num_train_samples`
- `num_val_samples`
- `num_test_samples`
- `deg`
- `noise_width`
- `pred_model`
- `po_epochs`
- `spo_epochs`
- `po_lr`
- `spo_lr`
- `batch_size`
- `benders_max_count`
- `benders_eps`
- `lsd`
- `lam`
- `anchor`

### `gen_train_data(...)`

Input shape expectations:
- `opt_model.num_cost` defines the cost dimension
- `cfg.num_test_samples` and `cfg.num_val_samples` are interpreted as split
  sizes, not fractions

Output:
- 4-tuple:
  1. `{"train_loader": AdvLoader, "val_loader": AdvLoader}`
  2. `{"feats": X_test, "costs": c_test}`
  3. `normalization_constant: float`
  4. `{"data_generator": SPNIAdverseDataGenerator}`

### `gen_data(...)`

Output:
- dict with keys:
  - `features`
  - `costs`

### `compare_shortest_paths(...)`

Output:
- tuple of four Python lists:
  - `true_objs`
  - `pfl_objs`
  - `dfl_objs`
  - `adfl_objs`

### `compare_sym_intd(...)`

Guaranteed keys:
- `true_objective`
- `po_objective`
- `spo_objective`

Optional keys:
- `adv_spo_objective`
- `rand_adv_spo_objective`
- `mixed_rand_spo_objective`
- `mixed_adverse_spo_objective`

### `compare_asym_intd(...)`

Output:
- tuple `(est_objs, true_objs)`
- each is a NumPy array of length `cfg.num_test_samples`
- failed solves remain in place as `NaN`

### `compare_wrong_asym_intd(...)`

Output:
- single NumPy array
- length equals the number of successful appended evaluations, not
  necessarily the input sample count

## 4. Current `single_sim(...)` return structure

Current return value:

`prediction_mean_std, metrics, table_1, table_2, all_data`

### `prediction_mean_std`

Current keys:
- `test_mean`
- `train_mean`
- `intd_mean`
- `po_mean`
- `spo_mean`
- `rand_spo_mean`
- `adv_spo_mean`
- `test_std`
- `train_std`
- `intd_std`
- `po_std`
- `spo_std`
- `rand_spo_std`
- `adv_spo_std`

### `metrics`

Current keys:
- `metric_1`
- `metric_2`
- `metric_3`
- `metric_4`
- `metric_5`
- `metric_6`
- `metric_7`
- `metric_8`
- `asym_nan_rows_oracle`
- `asym_nan_rows_po`
- `asym_nan_rows_spo`
- `asym_nan_rows_rand_spo`
- `asym_nan_rows_adv_spo`

Notes:
- `metric_8` is `None` unless `compute_asym_intd_2=True`

### `table_1`

Current keys:
- `t1_o_n_mean`
- `t1_o_s_mean`
- `t1_o_s_std`
- `t1_o_a_mean`
- `t1_o_a_std`
- `t1_p_n_mean`
- `t1_p_s_mean`
- `t1_p_s_std`
- `t1_p_a_mean`
- `t1_p_a_std`
- `t1_s_n_mean`
- `t1_s_s_mean`
- `t1_s_s_std`
- `t1_s_a_mean`
- `t1_s_a_std`
- `t1_r_n_mean`
- `t1_r_s_mean`
- `t1_r_s_std`
- `t1_r_a_mean`
- `t1_r_a_std`
- `t1_a_n_mean`
- `t1_a_s_mean`
- `t1_a_s_std`
- `t1_a_a_mean`
- `t1_a_a_std`

### `table_2`

Current behavior:
- empty dict when `compute_asym_intd_2=False`
- otherwise includes:
  - `t2_p_s_mean`
  - `t2_p_s_std`
  - `t2_p_a_mean`
  - `t2_p_a_std`
  - `t2_s_p_mean`
  - `t2_s_p_std`
  - `t2_s_a_mean`
  - `t2_s_a_std`
  - `t2_a_p_mean`
  - `t2_a_p_std`
  - `t2_a_s_mean`
  - `t2_a_s_std`

### `all_data`

Always produced keys:
- `o_o`
- `o_p`
- `o_s`
- `o_r`
- `o_a`
- `s_o`
- `s_p`
- `s_s`
- `s_r`
- `s_a`
- `a_o`
- `a_p`
- `a_p_o`
- `a_s`
- `a_s_o`
- `a_r`
- `a_r_o`
- `a_a`
- `a_a_o`

Conditionally produced keys when `compute_asym_intd_2=True`:
- `a_s_p`
- `a_p_s`
- `a_a_p`
- `a_p_a`
- `a_a_s`
- `a_s_a`

Current naming convention:
- `o_*`: no interdiction
- `s_*`: symmetric interdiction
- `a_*`: asymmetric interdiction
- suffix is the predictor family or oracle variant

## 5. Current sweep result structure

Each entry appended by `Asym_SPNI_Simulator.py` has keys:
- `seed`
- `prediction_mean_std`
- `metrics`
- `table_1`
- `table_2`
- `all_data`

The CSV writer uses only:
- `result["all_data"]`

The summary dicts are currently not written into the CSV rows.

## 6. Current CSV row schema

`save_results_to_csv(...)` writes one row per:
- simulation index
- sample index

Guaranteed row columns:
- `simulation_index`
- `sample_index`
- one column per key in `all_data`

Value conversion rules:
- NumPy arrays are written as lists
- NumPy scalars are converted to Python scalars
- everything else is written as-is

Important current implication:
- CSV schema is driven entirely by the keys present in `all_data` at write
  time

## 7. Current analysis assumptions

`load_results_from_csv(...)` reconstructs only this fixed key set:
- `o_o`, `o_p`, `o_s`, `o_r`, `o_mr`, `o_ma`, `o_m`, `o_a`
- `s_o`, `s_p`, `s_s`, `s_r`, `s_mr`, `s_ma`, `s_m`, `s_a`
- `a_o`, `a_p`, `a_s`, `a_r`, `a_mr`, `a_ma`, `a_m`, `a_a`
- `a_p_o`, `a_s_o`, `a_r_o`, `a_a_o`

If a column is missing in the CSV:
- it is reconstructed as `np.full(len(sim_data), np.nan)`

Important current implication:
- analysis is tolerant of missing columns, but only for a predefined key set
- keys present in `all_data` but not in `load_results_from_csv(...)` are not
  reconstructed

`analyse_results.py` currently computes percentage improvements using:
- no interdiction:
  - `o_o`, `o_p`, `o_s`, `o_r`, `o_a`
- symmetric interdiction:
  - `s_o`, `s_p`, `s_s`, `s_r`, `s_a`
- asymmetric interdiction:
  - `a_o`, `a_p`, `a_s`, `a_r`, `a_a`

Important current implication:
- the plotting/analysis path depends only on this subset of the export schema

## 8. Current side effects by stage

### Single-run stage

Side effects:
- console printing
- dataset cache reads/writes
- interdiction cache reads/writes
- predictor cache reads/writes
- optional figure generation

### Sweep stage

Side effects:
- all single-run side effects
- CSV result writing
- analysis execution

## 9. Current baseline quirks and inconsistencies

1. `gen_train_data(...)` returns testing features under `feats`, while
   `gen_data(...)` returns them under `features`.
2. `compare_asym_intd(...)` preserves sample alignment using `NaN`
   placeholders; `compare_wrong_asym_intd(...)` does not.
3. `Asym_SPNI_Simulator.py` executes at module import time and has no thin
   `main()` wrapper.
4. The sweep-level `results` list stores summary dicts, but CSV export uses
   only `all_data`.
5. Analysis reconstructs some keys that the current SPNI writer does not
   produce, filling them with `NaN`.
6. `single_sim(...)` currently owns both:
   - numerical workflow orchestration
   - user-facing table construction

## 10. Current regression baseline

Regression command run:

```bash
MPLCONFIGDIR=/tmp/mpl pytest tests/regression -q
```

Observed status:
- 11 passed
- 1 skipped

Skipped regression:
- `tests/regression/test_spni_simulation_regressions.py`
  - `compare_wrong_asym_intd does not handle failed asymmetric solves.`

Current interpretation:
- the baseline code currently satisfies the active regression suite
- one known issue remains intentionally skipped rather than enforced

## 11. Phase 0 outcome

Phase 0 recording is complete when this document is treated as the current
baseline for:
- entrypoints
- responsibilities
- return schemas
- export schemas
- downstream analysis assumptions
- regression-suite status

