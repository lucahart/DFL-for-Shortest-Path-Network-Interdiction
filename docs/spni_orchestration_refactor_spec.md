# SPNI Orchestration Refactor Spec

This document specifies a refactor of the SPNI simulation orchestration
layer only.

Out of scope:
- solver core
- model core
- adverse-data core
- trainer core

Those parts stay where they are and continue to own the numerical and
optimization logic.

## Goals

1. Separate orchestration from computation.
2. Replace loose dictionaries with typed run artifacts.
3. Make each pipeline stage independently testable and debuggable.
4. Keep the current scripts working during migration.
5. Make seed handling, cache behavior, and result shapes explicit.

## Current pain points

The current script layer spreads responsibilities across
`scripts/setup.py`, `scripts/asym_spni_single_sim.py`,
`scripts/Asym_SPNI_Simulator.py`, and `scripts/compare.py`.

Main issues:
- config normalization is implicit
- seed derivation is duplicated
- cache behavior is mixed into data/training orchestration
- data bundles are passed around as raw dicts
- model bundles are passed around as raw variables
- evaluation results are flattened late and inconsistently
- debugging requires stepping through long script functions

## Proposed package

New package:

`src/dflintdpy/simulation/spni/`

Modules:
- `config.py`
- `types.py`
- `build.py`
- `data.py`
- `train.py`
- `evaluate.py`
- `results.py`
- `pipeline.py`

Thin package exports:
- `run_single_simulation`
- `run_seed_sweep`
- core dataclasses used by scripts and tests

## Module-by-module spec

### `config.py`

Purpose:
- validate incoming config
- normalize run options
- derive deterministic per-run seed bundles
- centralize cache-policy intent at the orchestration layer

Primary dataclasses:
- `SPNIRunConfig`
- `SeedBundle`
- `CachePolicy`

Primary functions:
- `build_run_config(...)`
- `derive_seed_bundle(...)`
- `derive_seed_sweep(...)`
- `describe_run(...)`

Rules:
- no model or solver creation here
- no file I/O here
- no training here
- may read existing `HP`, but should return a typed orchestration config

### `types.py`

Purpose:
- define the typed objects passed between pipeline stages
- document canonical shapes and semantics

Primary dataclasses:
- `GraphBundle`
- `DatasetBundle`
- `TrainingLogBundle`
- `PredictorBundle`
- `InterdictionSampleBundle`
- `EvaluationBundle`
- `SummaryBundle`
- `SimulationArtifacts`
- `SimulationResult`
- `SweepResult`

Rules:
- no heavy computation here
- no file I/O here
- types should be stable and easy to print in debugging sessions

### `build.py`

Purpose:
- build the graph and shortest-path optimization model used by one run

Primary functions:
- `build_graph(...)`
- `build_opt_model(...)`
- `build_problem_bundle(...)`

Rules:
- own real-world vs synthetic graph selection
- return constructed objects without training or evaluating them
- do not mutate config

### `data.py`

Purpose:
- generate or load base synthetic data
- create train/val/test splits
- build SPNI adverse and non-adverse loaders
- return one typed `DatasetBundle`

Primary functions:
- `generate_base_data(...)`
- `split_base_data(...)`
- `build_spni_training_data(...)`
- `build_nonadverse_views(...)`
- `assemble_dataset_bundle(...)`

Rules:
- own all dataset shape contracts
- own normalization constant creation
- own the distinction between adversarial, random, and baseline loaders

### `train.py`

Purpose:
- train all predictor families required by the SPNI experiments
- isolate cache tags and predictor-building choices

Primary functions:
- `train_pfl_predictor(...)`
- `train_dfl_predictor(...)`
- `train_all_predictors(...)`

Rules:
- return predictors and logs in typed bundles
- do not evaluate interdiction performance here
- no result aggregation here

### `evaluate.py`

Purpose:
- run all evaluation stages for one simulation
- keep the raw arrays close to the numerical routines

Primary functions:
- `evaluate_uninterdicted(...)`
- `evaluate_symmetric_interdiction(...)`
- `evaluate_asymmetric_interdiction(...)`
- `evaluate_wrong_model_asymmetry(...)`
- `evaluate_all(...)`

Rules:
- every function should state exact array shapes
- failed solves should be represented explicitly, not silently dropped
- no CSV writing here

### `results.py`

Purpose:
- convert raw evaluation arrays into consistent summaries and export-ready
  structures

Primary functions:
- `build_summary(...)`
- `flatten_result_rows(...)`
- `aggregate_sweep_results(...)`
- `to_legacy_all_data(...)`

Rules:
- own the mapping from typed results to the current CSV schema
- own percentage-improvement calculations
- keep all formatting and column naming in one place

### `pipeline.py`

Purpose:
- coordinate the full SPNI workflow

Primary functions:
- `run_single_simulation(...)`
- `run_seed_sweep(...)`

Rules:
- this is the only module allowed to call every stage in sequence
- this is the main replacement for the current script orchestration
- side effects should be optional and explicit

## Dataclass field spec

### `SPNIRunConfig`

Responsibilities:
- single source of truth for one run's orchestration settings
- contains only normalized and validated values

Required fields:
- `base_cfg`: original `HP` or config-like object
- `grid_size`
- `num_features`
- `num_train_samples`
- `num_val_samples`
- `num_test_samples`
- `budget`
- `num_scenarios`
- `deg`
- `noise_width`
- `benders_max_count`
- `benders_eps`
- `lsd`
- `intd_seed`
- `random_seed`
- `loader_seed`
- `seed`
- `pred_model`
- `po_epochs`
- `spo_epochs`
- `po_lr`
- `spo_lr`
- `compute_asym_intd`
- `compute_asym_intd_2`
- `load_real_world_graph`
- `cache_policy`
- `metadata`

### `SeedBundle`

Responsibilities:
- make every seed used by a run explicit

Fields:
- `sweep_seed`
- `random_seed`
- `intd_seed`
- `loader_seed`

### `GraphBundle`

Responsibilities:
- package the graph-facing objects used by one run

Fields:
- `graph`
- `opt_model`
- `graph_kind`
- `graph_source`

### `DatasetBundle`

Responsibilities:
- package every dataset/loader view needed by training and evaluation

Fields:
- `train_loader_adversarial`
- `val_loader_adversarial`
- `train_loader_random`
- `val_loader_random`
- `train_loader_baseline`
- `val_loader_baseline`
- `testing_features`
- `testing_costs`
- `interdiction_features`
- `interdiction_costs`
- `normalization_constant`
- `data_generator_adversarial`
- `data_generator_random`

### `TrainingLogBundle`

Responsibilities:
- store training curves without binding them to presentation logic

Fields per model family:
- `train_loss`
- `train_regret`
- `val_loss`
- `val_regret`

### `PredictorBundle`

Responsibilities:
- provide one stable place to access trained models

Fields:
- `pfl`
- `dfl`
- `rdfl`
- `adfl`
- `logs`

### `EvaluationBundle`

Responsibilities:
- hold raw arrays for every evaluation stage

Fields:
- `uninterdicted`
- `symmetric`
- `asymmetric`
- `wrong_model_asymmetry`
- `diagnostics`

### `SummaryBundle`

Responsibilities:
- hold the tables, metrics, and summary statistics derived from one run

Fields:
- `prediction_mean_std`
- `metrics`
- `table_1`
- `table_2`
- `all_data`

### `SimulationResult`

Responsibilities:
- top-level return type for one run

Fields:
- `run_config`
- `seed_bundle`
- `graph_bundle`
- `dataset_bundle`
- `predictor_bundle`
- `evaluation_bundle`
- `summary_bundle`

### `SweepResult`

Responsibilities:
- top-level return type for multi-seed execution

Fields:
- `run_config`
- `results`
- `aggregated_summary`

## Function responsibility spec

### `build_run_config(base_cfg, *, compute_asym_intd, compute_asym_intd_2, load_real_world_graph, cache_policy)`

Responsibilities:
- read orchestration-relevant values from the existing `HP`
- validate that required keys exist
- return a normalized `SPNIRunConfig`

Must not:
- create models
- touch caches
- start training

### `derive_seed_bundle(run_cfg, sweep_seed=None)`

Responsibilities:
- create the exact seed tuple for one run
- make the sweep-level seed explicit

Must not:
- mutate global random state unless explicitly requested by a caller

### `build_problem_bundle(run_cfg)`

Responsibilities:
- construct the graph
- construct the shortest path model
- return a `GraphBundle`

### `assemble_dataset_bundle(run_cfg, graph_bundle)`

Responsibilities:
- generate base data
- normalize costs
- split train/val/test
- build adverse/random/baseline loaders
- generate evaluation interdictions

### `train_all_predictors(run_cfg, graph_bundle, dataset_bundle)`

Responsibilities:
- train PFL
- train baseline DFL
- train random adverse DFL
- train adversarial DFL
- store logs in a structured way

### `evaluate_all(run_cfg, graph_bundle, dataset_bundle, predictor_bundle)`

Responsibilities:
- run all enabled evaluation families
- preserve sample alignment
- expose failure counts and skipped counts

### `build_summary(run_cfg, evaluation_bundle, predictor_bundle, dataset_bundle)`

Responsibilities:
- convert raw arrays into the current summary tables
- compute comparable metrics
- build the legacy `all_data` mapping if needed

### `run_single_simulation(run_cfg_or_base_cfg, **options)`

Responsibilities:
- call every stage once
- keep stage boundaries visible
- return a full `SimulationResult`

### `run_seed_sweep(run_cfg_or_base_cfg, *, num_seeds)`

Responsibilities:
- derive one `SeedBundle` per sweep element
- call `run_single_simulation`
- return a `SweepResult`

## Debugging rules

Each stage should emit or return diagnostics that answer:
- what config was used
- what seeds were used
- what cache entries were loaded
- what data shapes were created
- how many asymmetric solves failed
- which models were trained vs loaded
- which result arrays contain missing values

Diagnostics should be stored in typed fields, not only printed.

## Migration plan

1. Introduce typed dataclasses and config normalization.
2. Wrap existing script logic into stage functions.
3. Keep legacy scripts as thin callers.
4. Move CSV/result conversion behind `results.py`.
5. Migrate analysis scripts to consume `SimulationResult` or `SweepResult`.

## Proposed test structure

New tree:

`tests/unit/simulation/spni/`
- `test_config.py`
- `test_build.py`
- `test_data.py`
- `test_train.py`
- `test_evaluate.py`
- `test_results.py`
- `test_pipeline.py`

`tests/integration/simulation/spni/`
- `test_single_run_smoke.py`
- `test_seed_sweep_smoke.py`

`tests/regression/`
- keep regression repros close to current behavior and migration risks

### Unit test cores

#### `test_config.py`

Core expectations:
- valid `HP` becomes a normalized `SPNIRunConfig`
- missing required values fail early with useful messages
- seed derivation is deterministic
- cache policy is carried through unchanged

#### `test_build.py`

Core expectations:
- synthetic graph path builds a `Grid`
- real-world graph path calls the CSV importer once
- `ShortestPathGrb` is created exactly once
- graph bundle metadata matches the chosen graph source

#### `test_data.py`

Core expectations:
- base data generation returns expected shapes
- split counts match config exactly
- normalization constant is preserved and nonzero
- adverse/random/baseline loaders are all present
- baseline loaders are derived from the non-adverse scenario view

#### `test_train.py`

Core expectations:
- `train_all_predictors` trains or loads exactly four predictor families
- cache tags are assigned consistently
- each returned training-log bundle contains the expected curve vectors
- stage output preserves predictor identity and labels

#### `test_evaluate.py`

Core expectations:
- uninterdicted evaluation returns one result per test sample
- symmetric evaluation returns one result per test sample
- asymmetric evaluation preserves sample alignment even on failed solves
- wrong-model asymmetry respects the enable/disable flag
- diagnostics count failures explicitly

#### `test_results.py`

Core expectations:
- summary tables are derived consistently from raw arrays
- legacy `all_data` keys are produced exactly once
- flattening to CSV rows preserves simulation and sample indices
- percentage metrics handle divide-by-zero safely

#### `test_pipeline.py`

Core expectations:
- single-run pipeline calls stages in order
- stage outputs are threaded into the next stage unchanged
- optional analysis or persistence hooks are not executed unless requested
- returned `SimulationResult` contains all expected bundles

### Integration test cores

#### `test_single_run_smoke.py`

Core expectations:
- a tiny synthetic run completes end to end
- all expected model families are present
- summary outputs are non-empty

#### `test_seed_sweep_smoke.py`

Core expectations:
- a tiny two-seed sweep returns two `SimulationResult`s
- seeds differ across runs in the expected way
- aggregation combines the run outputs without shape drift

## Script migration targets

### Current `scripts/asym_spni_single_sim.py`

Future role:
- thin wrapper around `run_single_simulation`

### Current `scripts/Asym_SPNI_Simulator.py`

Future role:
- thin wrapper around `run_seed_sweep`
- optional persistence and analysis entrypoint

### Current `scripts/setup.py`

Future role:
- either deprecated or reduced to compatibility wrappers that call
  `simulation/spni/data.py` and `simulation/spni/train.py`

### Current `scripts/compare.py`

Future role:
- either migrated into `evaluate.py` or kept as an internal low-level helper
  behind typed wrappers

