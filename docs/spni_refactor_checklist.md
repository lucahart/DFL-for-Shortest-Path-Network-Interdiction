# SPNI Refactor Checklist

This checklist tracks the phased implementation of the SPNI orchestration
refactor.

Scope:
- orchestration layer only
- no solver core rewrite
- no model core rewrite
- no adverse-data core rewrite
- no trainer core rewrite

Reference documents:
- [`docs/spni_orchestration_refactor_spec.md`](./spni_orchestration_refactor_spec.md)
- [`tests/unit/simulation/spni/TEST_PLAN.md`](../tests/unit/simulation/spni/TEST_PLAN.md)

---

## Phase 0: Freeze Current Behavior

- [x] Record the current SPNI entrypoints and their responsibilities.
- [x] Record the current output structures from:
  - `src/dflintdpy/scripts/asym_spni_single_sim.py`
  - `src/dflintdpy/scripts/Asym_SPNI_Simulator.py`
  - `src/dflintdpy/scripts/setup.py`
  - `src/dflintdpy/scripts/compare.py`
- [x] Confirm the current regression repros still pass as expected.
- [x] Treat `src/dflintdpy/simulation/spni/` as the target architecture.

Implementation notes:
- Do not move code yet.
- This phase is only about making the current behavior explicit before
  introducing refactor steps.
- Baseline document:
  - `docs/spni_phase0_baseline.md`

Exit criteria:
- Current behavior is documented well enough to compare against the new
  pipeline during migration.

---

## Phase 1: Implement Config Normalization

- [x] Implement `CachePolicy` in
  `src/dflintdpy/simulation/spni/config.py`.
- [x] Implement `SeedBundle` in
  `src/dflintdpy/simulation/spni/config.py`.
- [x] Implement `SPNIRunConfig` in
  `src/dflintdpy/simulation/spni/config.py`.
- [x] Implement `build_run_config(...)`.
- [x] Implement `derive_seed_bundle(...)`.
- [x] Implement `derive_seed_sweep(...)`.
- [x] Implement `describe_run(...)`.
- [x] Add unit tests for config normalization and seed determinism.

Implementation notes:
- Read from the current `HP` object.
- Normalize only orchestration-relevant fields.
- Keep this module free of model creation, file I/O, training, and
  evaluation logic.

Exit criteria:
- A legacy `HP` can be converted into a validated `SPNIRunConfig`.
- Seed derivation is deterministic and test-covered.

Completed in:
- `src/dflintdpy/simulation/spni/config.py`
- `tests/unit/simulation/spni/test_config.py`

---

## Phase 2: Finalize Typed Stage Artifacts

- [x] Implement `GraphBundle` in
  `src/dflintdpy/simulation/spni/types.py`.
- [x] Implement `DatasetBundle`.
- [x] Implement `TrainingLogBundle`.
- [x] Implement `PredictorBundle`.
- [x] Implement `EvaluationBundle`.
- [x] Implement `SummaryBundle`.
- [x] Implement `SimulationArtifacts`.
- [x] Implement `SimulationResult`.
- [x] Implement `SweepResult`.
- [x] Add unit tests that instantiate each dataclass with lightweight stubs.

Implementation notes:
- Keep field names stable and explicit.
- Preserve debug visibility by storing diagnostics as structured fields.
- Avoid putting behavior-heavy methods on these dataclasses.

Exit criteria:
- Each pipeline stage has a concrete typed output target.

Completed in:
- `src/dflintdpy/simulation/spni/types.py`
- `src/dflintdpy/simulation/spni/__init__.py`
- `tests/unit/simulation/spni/test_types.py`

---

## Phase 3: Build Graph and Optimization Model

- [x] Implement `build_graph(...)` in
  `src/dflintdpy/simulation/spni/build.py`.
- [x] Implement `build_opt_model(...)`.
- [x] Implement `build_problem_bundle(...)`.
- [x] Add unit tests for synthetic-grid creation.
- [x] Add unit tests for optional real-world graph import.
- [x] Add unit tests for graph-bundle metadata.

Implementation notes:
- Use the existing `Grid`, `ShortestPathGrb`, and real-world CSV helper.
- Keep graph creation separately testable from optimization-model creation.

Exit criteria:
- One stage builds all graph-facing objects for a run.

Completed in:
- `src/dflintdpy/simulation/spni/build.py`
- `tests/unit/simulation/spni/test_build.py`

---

## Phase 4: Assemble Datasets and Loader Views

- [x] Implement `generate_base_data(...)` in
  `src/dflintdpy/simulation/spni/data.py`.
- [x] Implement `split_base_data(...)`.
- [x] Implement `build_spni_training_data(...)`.
- [x] Implement `build_nonadverse_views(...)`.
- [x] Implement `assemble_dataset_bundle(...)`.
- [x] Add unit tests for split counts and shape expectations.
- [x] Add unit tests for normalization handling.
- [x] Add unit tests for presence of adversarial, random, and baseline
  loaders.
- [x] Add unit tests for evaluation interdiction generation.

Implementation notes:
- Wrap the current `gen_syn_data`, `SPNIAdverseDataGenerator`, `AdvDataset`,
  and `AdvLoader` logic.
- Preserve the current normalization contract explicitly in the
  `DatasetBundle`.

Exit criteria:
- The data stage can provide every loader and array needed by training and
  evaluation.

Completed in:
- `src/dflintdpy/simulation/spni/data.py`
- `tests/unit/simulation/spni/test_data.py`

---

## Phase 5: Train All Predictor Families

- [ ] Implement `train_pfl_predictor(...)` in
  `src/dflintdpy/simulation/spni/train.py`.
- [ ] Implement `train_dfl_predictor(...)`.
- [ ] Implement `train_all_predictors(...)`.
- [ ] Add unit tests that monkeypatch current predictor setup helpers.
- [ ] Verify cache tags are assigned consistently for PFL, DFL, R-DFL, and
  A-DFL.
- [ ] Verify training-log bundles are captured for each predictor family.

Implementation notes:
- Reuse the existing predictor setup helpers instead of rewriting their
  internals.
- Keep the mapping from loader variant to predictor family explicit.

Exit criteria:
- One function returns a full `PredictorBundle` with four model families and
  logs.

---

## Phase 6: Implement Evaluation Stage Wrappers

- [ ] Implement `evaluate_uninterdicted(...)` in
  `src/dflintdpy/simulation/spni/evaluate.py`.
- [ ] Implement `evaluate_symmetric_interdiction(...)`.
- [ ] Implement `evaluate_asymmetric_interdiction(...)`.
- [ ] Implement `evaluate_wrong_model_asymmetry(...)`.
- [ ] Implement `evaluate_all(...)`.
- [ ] Add unit tests for uninterdicted evaluation alignment.
- [ ] Add unit tests for symmetric evaluation alignment.
- [ ] Add unit tests for asymmetric evaluation failure handling.
- [ ] Add unit tests for wrong-model asymmetry enable/disable behavior.
- [ ] Add unit tests for evaluation diagnostics.

Implementation notes:
- Wrap the current comparison helpers rather than duplicating solver logic.
- Preserve sample alignment explicitly.
- Represent failed solves through structured outputs and diagnostics.

Exit criteria:
- All evaluation families are stage-isolated and testable.

---

## Phase 7: Implement Summary and Export Adapters

- [ ] Implement `build_summary(...)` in
  `src/dflintdpy/simulation/spni/results.py`.
- [ ] Implement `to_legacy_all_data(...)`.
- [ ] Implement `flatten_result_rows(...)`.
- [ ] Implement `aggregate_sweep_results(...)`.
- [ ] Add unit tests for summary metric derivation.
- [ ] Add unit tests for legacy `all_data` compatibility.
- [ ] Add unit tests for row flattening and index preservation.
- [ ] Add unit tests for safe divide-by-zero handling.

Implementation notes:
- Centralize all legacy column naming here.
- Do not keep summary logic spread across multiple scripts after this phase.

Exit criteria:
- One module owns translation from raw evaluation outputs to summary and
  export formats.

---

## Phase 8: Implement Single-Run Pipeline

- [ ] Implement `run_single_simulation(...)` in
  `src/dflintdpy/simulation/spni/pipeline.py`.
- [ ] Wire the stages in order:
  - config
  - seed derivation
  - graph/model build
  - dataset assembly
  - predictor training/loading
  - evaluation
  - summary construction
- [ ] Return a full `SimulationResult`.
- [ ] Add unit tests that verify stage call order and argument flow.

Implementation notes:
- This becomes the authoritative single-run orchestration entrypoint.
- Keep side effects optional and explicit.

Exit criteria:
- A single function can execute the full SPNI run without relying on the
  legacy scripts.

---

## Phase 9: Implement Multi-Seed Sweep Pipeline

- [ ] Implement `run_seed_sweep(...)` in
  `src/dflintdpy/simulation/spni/pipeline.py`.
- [ ] Use `derive_seed_sweep(...)` for run ordering.
- [ ] Return a `SweepResult`.
- [ ] Aggregate sweep-level summaries.
- [ ] Add smoke tests for tiny 2-seed sweeps.

Implementation notes:
- Keep run ordering stable for aggregation and CSV output.
- Reuse `run_single_simulation(...)` instead of duplicating orchestration.

Exit criteria:
- The multi-seed sweep exists as a real pipeline API instead of a script body.

---

## Phase 10: Migrate `asym_spni_single_sim.py`

- [ ] Replace the current script body with a thin compatibility wrapper around
  `run_single_simulation(...)`.
- [ ] Preserve the current public return shape temporarily.
- [ ] Add compatibility tests for returned values and structure.

Implementation notes:
- Do not break current callers during this phase.
- The wrapper may adapt a `SimulationResult` back to the old tuple shape.

Exit criteria:
- The legacy single-run entrypoint delegates to the new pipeline.

---

## Phase 11: Migrate `Asym_SPNI_Simulator.py`

- [ ] Replace the current sweep script body with a thin wrapper around
  `run_seed_sweep(...)`.
- [ ] Move persistence and analysis calls behind explicit post-processing
  steps.
- [ ] Keep behavior compatible if external callers depend on the current
  script.

Implementation notes:
- The script should stop owning business logic after this phase.

Exit criteria:
- The legacy sweep script becomes a thin entrypoint.

---

## Phase 12: Reduce `setup.py` to Compatibility Helpers

- [ ] Move remaining orchestration logic out of
  `src/dflintdpy/scripts/setup.py`.
- [ ] Keep only thin wrappers if other code imports it.
- [ ] Ensure dataset and training responsibilities now live in
  `simulation/spni/data.py` and `simulation/spni/train.py`.

Implementation notes:
- This phase removes the last large mixed-responsibility orchestration file.

Exit criteria:
- `setup.py` is no longer the owner of core SPNI workflow assembly.

---

## Phase 13: Update Results and Analysis Consumers

- [ ] Update result-writing code to consume typed pipeline results first.
- [ ] Keep compatibility adapters for existing CSV workflows.
- [ ] Update analysis code to depend on exported summary structure rather than
  script-local dict assembly.
- [ ] Add integration tests for save-load-analyze flows.

Implementation notes:
- `read_write_results.py` and `analyse_results.py` should become adapters,
  not workflow owners.

Exit criteria:
- Reporting and analysis are cleanly downstream of the new pipeline.

---

## Phase 14: Deprecate or Remove Stale Orchestration Code

- [ ] Reassess duplicate entrypoints after wrappers are proven stable.
- [ ] Reassess stale exploratory scripts such as `spni_synthetic_data.py`.
- [ ] Document the canonical SPNI orchestration entrypoints.
- [ ] Remove dead code only after compatibility coverage is in place.

Implementation notes:
- This cleanup phase should happen only after all prior phases are stable.

Exit criteria:
- One canonical SPNI orchestration path remains.
- Old scripts are either wrappers or clearly deprecated.

---

## Cross-Cutting Verification Checklist

- [ ] New stage code does not change solver/model/adverse/trainer core logic.
- [ ] All newly introduced dataclasses are documented and type-stable.
- [ ] All stage boundaries are covered by unit tests.
- [ ] At least one tiny end-to-end SPNI run is covered by integration tests.
- [ ] Existing regression repros remain valid throughout migration.
- [ ] Legacy scripts continue working until explicitly deprecated.

---

## Recommended Execution Order

Suggested order:
1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 5
6. Phase 6
7. Phase 7
8. Phase 8
9. Phase 9
10. Phase 10
11. Phase 11
12. Phase 12
13. Phase 13
14. Phase 14

This order keeps the refactor incremental and minimizes the risk of breaking
the current SPNI simulation workflow before the replacement pipeline is ready.
