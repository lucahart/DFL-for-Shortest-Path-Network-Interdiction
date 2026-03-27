from types import SimpleNamespace

import numpy as np

from dflintdpy.simulation.spni.config import CachePolicy, SPNIRunConfig, SeedBundle
from dflintdpy.simulation.spni.types import (
    DatasetBundle,
    EvaluationBundle,
    GraphBundle,
    InterdictionSampleBundle,
    PredictorBundle,
    SimulationArtifacts,
    SimulationResult,
    SummaryBundle,
    SweepResult,
    TrainingLogBundle,
)


################
### Fixtures ###
################


############################
### Helper functionality ###
############################


def _build_run_config() -> SPNIRunConfig:
    """Return a compact run config for typed-artifact instantiation tests."""
    return SPNIRunConfig(
        base_cfg=SimpleNamespace(label="base"),
        grid_size=(3, 4),
        num_features=5,
        num_train_samples=10,
        num_val_samples=2,
        num_test_samples=3,
        batch_size=4,
        budget=1,
        num_scenarios=6,
        deg=2,
        noise_width=0.25,
        benders_max_count=7,
        benders_eps=1e-4,
        lsd=1e-5,
        seed=11,
        random_seed=13,
        intd_seed=17,
        loader_seed=19,
        pred_model="linear",
        po_epochs=8,
        spo_epochs=9,
        po_lr=1e-3,
        spo_lr=2e-3,
        compute_asym_intd=True,
        compute_wrong_asym_intd=False,
        load_real_world_graph=None,
        cache_policy=CachePolicy(replace_pred=True),
        metadata={"source_type": "SimpleNamespace"},
    )


def _build_seed_bundle() -> SeedBundle:
    """Return a compact seed bundle for simulation result tests."""
    return SeedBundle(
        sweep_seed=23,
        random_seed=29,
        intd_seed=31,
        loader_seed=37,
    )


def _sample_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Return small numeric arrays reused across bundle tests."""
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    costs = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=float)
    return features, costs


##################################
### test GraphBundle ###
##################################


def test_spni_types_graph_bundle_stores_graph_and_model_metadata():
    """Verify that `GraphBundle` stores the graph and provenance fields."""
    # Arrange lightweight graph and model stubs.
    graph = SimpleNamespace(name="graph")
    opt_model = SimpleNamespace(name="model")

    # Act by instantiating the bundle directly.
    bundle = GraphBundle(
        graph=graph,
        opt_model=opt_model,
        graph_kind="synthetic",
        graph_source=None,
    )

    # Assert that the bundle preserves the provided objects and metadata.
    assert bundle.graph is graph, "GraphBundle should retain the graph stub."
    assert bundle.opt_model is opt_model, \
        "GraphBundle should retain the optimization model stub."
    assert bundle.graph_kind == "synthetic", \
        "GraphBundle should record the graph kind."
    assert bundle.graph_source is None, \
        "GraphBundle should allow an absent graph source."
    assert bundle.diagnostics == {}, \
        "GraphBundle should default diagnostics to an empty dict."
    pass


###############################################
### test InterdictionSampleBundle ###
###############################################


def test_spni_types_interdiction_sample_bundle_stores_aligned_payloads():
    """Verify that `InterdictionSampleBundle` keeps aligned sample payloads."""
    # Arrange compact aligned evaluation inputs.
    features, costs = _sample_arrays()
    interdictions = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)

    # Act by instantiating the typed interdiction-sample bundle.
    bundle = InterdictionSampleBundle(
        features=features,
        costs=costs,
        interdictions=interdictions,
        sample_indices=np.array([4, 9]),
    )

    # Assert that the aligned arrays and optional indices are preserved.
    assert np.array_equal(bundle.features, features), \
        "InterdictionSampleBundle should preserve features exactly."
    assert np.array_equal(bundle.costs, costs), \
        "InterdictionSampleBundle should preserve costs exactly."
    assert np.array_equal(bundle.interdictions, interdictions), \
        "InterdictionSampleBundle should preserve interdictions exactly."
    assert np.array_equal(bundle.sample_indices, np.array([4, 9])), \
        "InterdictionSampleBundle should keep optional sample indices."
    assert bundle.diagnostics == {}, \
        "InterdictionSampleBundle should default diagnostics to an empty dict."
    pass


##################################
### test DatasetBundle ###
##################################


def test_spni_types_dataset_bundle_stores_loaders_and_defaults():
    """Verify that `DatasetBundle` stores loaders, arrays, and defaults."""
    # Arrange the loader and array inputs used by the data stage.
    features, costs = _sample_arrays()
    train_loader = SimpleNamespace(name="train")
    val_loader = SimpleNamespace(name="val")

    # Act by instantiating the bundle with lightweight stubs.
    bundle = DatasetBundle(
        train_loader_adversarial=train_loader,
        val_loader_adversarial=val_loader,
        train_loader_random=train_loader,
        val_loader_random=val_loader,
        train_loader_baseline=train_loader,
        val_loader_baseline=val_loader,
        testing_features=features,
        testing_costs=costs,
        interdiction_features=features + 10.0,
        interdiction_costs=costs + 10.0,
        normalization_constant=12.5,
    )

    # Assert that the bundle keeps the stage inputs and structured defaults.
    assert bundle.train_loader_adversarial is train_loader, \
        "DatasetBundle should keep the adversarial train loader."
    assert bundle.val_loader_baseline is val_loader, \
        "DatasetBundle should keep the baseline validation loader."
    assert np.array_equal(bundle.testing_features, features), \
        "DatasetBundle should preserve testing features exactly."
    assert np.array_equal(bundle.testing_costs, costs), \
        "DatasetBundle should preserve testing costs exactly."
    assert bundle.normalization_constant == 12.5, \
        "DatasetBundle should store the normalization constant."
    assert bundle.data_generator_adversarial is None, \
        "DatasetBundle should default the adversarial generator to None."
    assert bundle.data_generator_random is None, \
        "DatasetBundle should default the random generator to None."
    assert bundle.diagnostics == {}, \
        "DatasetBundle should default diagnostics to an empty dict."
    pass


##################################
### test TrainingLogBundle ###
##################################


def test_spni_types_training_log_bundle_stores_curve_vectors():
    """Verify that `TrainingLogBundle` stores training and validation curves."""
    # Arrange a compact curve history from the trainer stage.
    train_loss = [1.0, 0.5]
    train_regret = [0.25, 0.1]
    val_loss = [1.25, 0.75]
    val_regret = [0.3, 0.15]

    # Act by instantiating the log bundle directly.
    bundle = TrainingLogBundle(
        train_loss=train_loss,
        train_regret=train_regret,
        val_loss=val_loss,
        val_regret=val_regret,
    )

    # Assert that the bundle keeps the curve vectors intact.
    assert bundle.train_loss == train_loss, \
        "TrainingLogBundle should keep the train loss curve."
    assert bundle.train_regret == train_regret, \
        "TrainingLogBundle should keep the train regret curve."
    assert bundle.val_loss == val_loss, \
        "TrainingLogBundle should keep the validation loss curve."
    assert bundle.val_regret == val_regret, \
        "TrainingLogBundle should keep the validation regret curve."
    pass


##################################
### test PredictorBundle ###
##################################


def test_spni_types_predictor_bundle_defaults_are_structured_and_isolated():
    """Verify that `PredictorBundle` keeps predictors and default mappings."""
    # Arrange lightweight predictor and log stubs.
    log_bundle = TrainingLogBundle(
        train_loss=[1.0],
        train_regret=[0.5],
        val_loss=None,
        val_regret=None,
    )
    pfl = SimpleNamespace(name="pfl")
    dfl = SimpleNamespace(name="dfl")
    rdfl = SimpleNamespace(name="rdfl")
    adfl = SimpleNamespace(name="adfl")

    # Act by creating two bundles with distinct default containers.
    bundle_1 = PredictorBundle(
        pfl=pfl,
        dfl=dfl,
        rdfl=rdfl,
        adfl=adfl,
        logs={"pfl": log_bundle},
    )
    bundle_2 = PredictorBundle(
        pfl=pfl,
        dfl=dfl,
        rdfl=rdfl,
        adfl=adfl,
    )

    # Assert that the predictor identities and structured defaults are kept.
    assert bundle_1.pfl is pfl, \
        "PredictorBundle should keep the PFL predictor stub."
    assert bundle_1.logs["pfl"] is log_bundle, \
        "PredictorBundle should keep the provided log bundle."
    assert bundle_1.diagnostics == {}, \
        "PredictorBundle should default diagnostics to an empty dict."
    assert bundle_2.logs == {}, \
        "PredictorBundle should default logs to an empty dict."
    assert bundle_1.logs is not bundle_2.logs, \
        "PredictorBundle should allocate independent default log maps."
    assert bundle_1.diagnostics is not bundle_2.diagnostics, \
        "PredictorBundle should allocate independent diagnostics maps."
    pass


##################################
### test EvaluationBundle ###
##################################


def test_spni_types_evaluation_bundle_defaults_are_structured_and_isolated():
    """Verify that `EvaluationBundle` stores stage outputs and diagnostics."""
    # Arrange raw evaluation payloads from the comparison stage.
    uninterdicted = {"true_objs": np.array([1.0, 2.0], dtype=float)}
    symmetric = {"pfl_objs": np.array([3.0, 4.0], dtype=float)}
    asymmetric = {"estimated": np.array([5.0, 6.0], dtype=float)}
    wrong_model = {"fallback": np.array([7.0, 8.0], dtype=float)}

    # Act by creating two bundles with distinct default diagnostics.
    bundle_1 = EvaluationBundle(
        uninterdicted=uninterdicted,
        symmetric=symmetric,
        asymmetric=asymmetric,
        wrong_model_asymmetry=wrong_model,
        diagnostics={"failed_solves": 1},
    )
    bundle_2 = EvaluationBundle(
        uninterdicted=uninterdicted,
        symmetric=symmetric,
        asymmetric=asymmetric,
        wrong_model_asymmetry=wrong_model,
    )

    # Assert that the raw payloads and diagnostics are preserved.
    assert bundle_1.uninterdicted is uninterdicted, \
        "EvaluationBundle should keep the uninterdicted payload."
    assert bundle_1.symmetric is symmetric, \
        "EvaluationBundle should keep the symmetric payload."
    assert bundle_1.asymmetric is asymmetric, \
        "EvaluationBundle should keep the asymmetric payload."
    assert bundle_1.wrong_model_asymmetry is wrong_model, \
        "EvaluationBundle should keep the wrong-model payload."
    assert bundle_1.diagnostics == {"failed_solves": 1}, \
        "EvaluationBundle should keep the provided diagnostics."
    assert bundle_2.diagnostics == {}, \
        "EvaluationBundle should default diagnostics to an empty dict."
    assert bundle_1.diagnostics is not bundle_2.diagnostics, \
        "EvaluationBundle should allocate independent diagnostics maps."
    pass


##################################
### test SummaryBundle ###
##################################


def test_spni_types_summary_bundle_stores_summary_tables():
    """Verify that `SummaryBundle` stores the derived tables and metrics."""
    # Arrange compact summary payloads.
    prediction_mean_std = {"pfl": {"mean": 1.0, "std": 0.1}}
    metrics = {"accuracy": 0.95}
    table_1 = {"rows": 1}
    table_2 = {"rows": 2}
    all_data = {"legacy": True}

    # Act by instantiating the summary bundle directly.
    bundle = SummaryBundle(
        prediction_mean_std=prediction_mean_std,
        metrics=metrics,
        table_1=table_1,
        table_2=table_2,
        all_data=all_data,
    )

    # Assert that the summary bundle keeps all derived structures intact.
    assert bundle.prediction_mean_std is prediction_mean_std, \
        "SummaryBundle should keep the prediction statistics map."
    assert bundle.metrics is metrics, \
        "SummaryBundle should keep the summary metrics map."
    assert bundle.table_1 is table_1, \
        "SummaryBundle should keep table_1 intact."
    assert bundle.table_2 is table_2, \
        "SummaryBundle should keep table_2 intact."
    assert bundle.all_data is all_data, \
        "SummaryBundle should keep the legacy export payload."
    assert bundle.diagnostics == {}, \
        "SummaryBundle should default diagnostics to an empty dict."
    pass


##################################
### test SimulationArtifacts ###
##################################


def test_spni_types_simulation_artifacts_default_maps_are_independent():
    """Verify that `SimulationArtifacts` uses isolated default containers."""
    # Arrange two artifact bundles for independent default checks.
    bundle_1 = SimulationArtifacts()
    bundle_2 = SimulationArtifacts()

    # Act by mutating one bundle's paths.
    bundle_1.predictor_paths["pfl"] = "/tmp/pfl.pkl"
    bundle_1.figure_paths["curve"] = "/tmp/curve.png"

    # Assert that the optional paths and defaults are separated cleanly.
    assert bundle_1.predictor_paths == {"pfl": "/tmp/pfl.pkl"}, \
        "SimulationArtifacts should keep the predictor path mapping."
    assert bundle_1.result_path is None, \
        "SimulationArtifacts should default the result path to None."
    assert bundle_1.figure_paths == {"curve": "/tmp/curve.png"}, \
        "SimulationArtifacts should keep the figure path mapping."
    assert bundle_1.diagnostics == {}, \
        "SimulationArtifacts should default diagnostics to an empty dict."
    assert bundle_2.predictor_paths == {}, \
        "SimulationArtifacts should allocate independent predictor maps."
    assert bundle_2.figure_paths == {}, \
        "SimulationArtifacts should allocate independent figure maps."
    assert bundle_1.diagnostics is not bundle_2.diagnostics, \
        "SimulationArtifacts should allocate independent diagnostics maps."
    pass


##################################
### test SimulationResult ###
##################################


def test_spni_types_simulation_result_packs_stage_bundles():
    """Verify that `SimulationResult` packages one full run consistently."""
    # Arrange the stage bundles that a pipeline run would produce.
    run_config = _build_run_config()
    seed_bundle = _build_seed_bundle()
    graph_bundle = GraphBundle(
        graph=SimpleNamespace(name="graph"),
        opt_model=SimpleNamespace(name="model"),
        graph_kind="synthetic",
        graph_source=None,
    )
    features, costs = _sample_arrays()
    dataset_bundle = DatasetBundle(
        train_loader_adversarial=SimpleNamespace(name="train_adv"),
        val_loader_adversarial=SimpleNamespace(name="val_adv"),
        train_loader_random=SimpleNamespace(name="train_rnd"),
        val_loader_random=SimpleNamespace(name="val_rnd"),
        train_loader_baseline=SimpleNamespace(name="train_base"),
        val_loader_baseline=SimpleNamespace(name="val_base"),
        testing_features=features,
        testing_costs=costs,
        interdiction_features=features + 1.0,
        interdiction_costs=costs + 1.0,
        normalization_constant=1.0,
    )
    predictor_bundle = PredictorBundle(
        pfl=SimpleNamespace(name="pfl"),
        dfl=SimpleNamespace(name="dfl"),
        rdfl=SimpleNamespace(name="rdfl"),
        adfl=SimpleNamespace(name="adfl"),
        logs={"pfl": TrainingLogBundle([1.0], [0.5], None, None)},
    )
    evaluation_bundle = EvaluationBundle(
        uninterdicted={"true_objs": np.array([1.0], dtype=float)},
        symmetric={"pfl_objs": np.array([2.0], dtype=float)},
        asymmetric={"estimated": np.array([3.0], dtype=float)},
        wrong_model_asymmetry={"fallback": np.array([4.0], dtype=float)},
    )
    summary_bundle = SummaryBundle(
        prediction_mean_std={"pfl": {"mean": 1.0, "std": 0.0}},
        metrics={"gap": 0.1},
        table_1={"rows": 1},
        table_2={"rows": 1},
        all_data={"legacy": True},
    )

    # Act by creating the top-level result without explicit artifacts.
    result = SimulationResult(
        run_config=run_config,
        seed_bundle=seed_bundle,
        graph_bundle=graph_bundle,
        dataset_bundle=dataset_bundle,
        predictor_bundle=predictor_bundle,
        evaluation_bundle=evaluation_bundle,
        summary_bundle=summary_bundle,
    )

    # Assert that the result packages every stage bundle and default artifacts.
    assert result.run_config is run_config, \
        "SimulationResult should keep the run config."
    assert result.seed_bundle is seed_bundle, \
        "SimulationResult should keep the seed bundle."
    assert result.graph_bundle is graph_bundle, \
        "SimulationResult should keep the graph bundle."
    assert result.dataset_bundle is dataset_bundle, \
        "SimulationResult should keep the dataset bundle."
    assert result.predictor_bundle is predictor_bundle, \
        "SimulationResult should keep the predictor bundle."
    assert result.evaluation_bundle is evaluation_bundle, \
        "SimulationResult should keep the evaluation bundle."
    assert result.summary_bundle is summary_bundle, \
        "SimulationResult should keep the summary bundle."
    assert isinstance(result.artifacts, SimulationArtifacts), \
        "SimulationResult should create default simulation artifacts."
    assert result.artifacts.predictor_paths == {}, \
        "SimulationResult should default predictor paths to an empty dict."
    assert result.artifacts.figure_paths == {}, \
        "SimulationResult should default figure paths to an empty dict."
    assert result.diagnostics == {}, \
        "SimulationResult should default diagnostics to an empty dict."
    pass


##################################
### test SweepResult ###
##################################


def test_spni_types_sweep_result_stores_runs_and_aggregate_defaults():
    """Verify that `SweepResult` stores ordered runs and aggregate payloads."""
    # Arrange two run results and a shared top-level config.
    run_config = _build_run_config()
    simulation_result = SimulationResult(
        run_config=run_config,
        seed_bundle=_build_seed_bundle(),
        graph_bundle=GraphBundle(
            graph=SimpleNamespace(name="graph"),
            opt_model=SimpleNamespace(name="model"),
            graph_kind="synthetic",
            graph_source=None,
        ),
        dataset_bundle=DatasetBundle(
            train_loader_adversarial=SimpleNamespace(name="train_adv"),
            val_loader_adversarial=SimpleNamespace(name="val_adv"),
            train_loader_random=SimpleNamespace(name="train_rnd"),
            val_loader_random=SimpleNamespace(name="val_rnd"),
            train_loader_baseline=SimpleNamespace(name="train_base"),
            val_loader_baseline=SimpleNamespace(name="val_base"),
            testing_features=np.array([[1.0]], dtype=float),
            testing_costs=np.array([[2.0]], dtype=float),
            interdiction_features=np.array([[3.0]], dtype=float),
            interdiction_costs=np.array([[4.0]], dtype=float),
            normalization_constant=1.0,
        ),
        predictor_bundle=PredictorBundle(
            pfl=SimpleNamespace(name="pfl"),
            dfl=SimpleNamespace(name="dfl"),
            rdfl=SimpleNamespace(name="rdfl"),
            adfl=SimpleNamespace(name="adfl"),
        ),
        evaluation_bundle=EvaluationBundle(
            uninterdicted={},
            symmetric={},
            asymmetric={},
            wrong_model_asymmetry={},
        ),
        summary_bundle=SummaryBundle(
            prediction_mean_std={},
            metrics={},
            table_1={},
            table_2={},
            all_data={},
        ),
    )

    # Act by instantiating the sweep result with the ordered run list.
    sweep = SweepResult(
        run_config=run_config,
        results=[simulation_result],
    )

    # Assert that the sweep keeps the run order and default aggregate map.
    assert sweep.run_config is run_config, \
        "SweepResult should keep the sweep-level run config."
    assert sweep.results == [simulation_result], \
        "SweepResult should keep the ordered simulation results."
    assert sweep.aggregated_summary == {}, \
        "SweepResult should default the aggregate summary to an empty dict."
    assert sweep.diagnostics == {}, \
        "SweepResult should default diagnostics to an empty dict."
    pass
