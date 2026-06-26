from types import SimpleNamespace

import numpy as np
import torch

from dflintdpy.utils.read_write import CacheReplaceOptions

import dflintdpy.scripts.setup as script_module


############################
### Helper functionality ###
############################


def _cfg_stub() -> SimpleNamespace:
    """Return one compact legacy config stub for setup-wrapper tests."""
    values = {
        "grid_size": (3, 4),
        "num_features": 2,
        "num_train_samples": 6,
        "num_val_samples": 2,
        "num_test_samples": 3,
        "batch_size": 2,
        "budget": 1,
        "num_scenarios": 2,
        "deg": 5,
        "noise_width": 0.25,
        "benders_max_count": 2,
        "benders_eps": 1e-4,
        "lsd": 1e-5,
        "seed": 10,
        "random_seed": 11,
        "intd_seed": 12,
        "loader_seed": 13,
        "pred_model": "linear",
        "pfl_epochs": 1,
        "dfl_epochs": 1,
        "pfl_lr": 1e-3,
        "dfl_lr": 1e-3,
        "max_lr_reductions": 2,
        "surrogate_underprediction_penalty_weight": 7.0,
        "surrogate_underprediction_margin": 0.25,
        "lam": 0.0,
        "anchor": "mse",
    }
    return SimpleNamespace(
        **values,
        get=lambda key, default=None: values.get(key, default),
    )


########################
### test gen_train_data ###
########################


def test_scripts_setup_gen_train_data_delegates_to_spni_data_stage(
    monkeypatch,
):
    """Verify that `gen_train_data(...)` delegates into the SPNI data stage."""
    # Arrange stage stubs plus call capture for the compatibility wrapper.
    calls: dict[str, object] = {}

    def _fake_build_run_config(cfg, *, cache_policy):
        calls["build_run_config"] = {
            "cfg": cfg,
            "cache_policy": cache_policy,
        }
        return SimpleNamespace(label="run-cfg")

    def _fake_generate_base_data(run_cfg, graph_bundle):
        calls["generate_base_data"] = {
            "run_cfg": run_cfg,
            "graph_bundle": graph_bundle,
        }
        return SimpleNamespace(
            features=np.array([[1.0], [2.0]], dtype=float),
            costs=np.array([[3.0], [4.0]], dtype=float),
        )

    def _fake_split_base_data(run_cfg, features, costs):
        calls["split_base_data"] = {
            "run_cfg": run_cfg,
            "features": features,
            "costs": costs,
        }
        return SimpleNamespace(
            test_features=np.array([[8.0], [9.0]], dtype=float),
            test_costs=np.array([[10.0], [11.0]], dtype=float),
            normalization_constant=7.5,
        )

    def _fake_build_spni_training_view(
        run_cfg,
        graph_bundle,
        split_data,
        *,
        interdiction_policy,
    ):
        calls["build_spni_training_view"] = {
            "run_cfg": run_cfg,
            "graph_bundle": graph_bundle,
            "split_data": split_data,
            "interdiction_policy": interdiction_policy,
        }
        return SimpleNamespace(
            train_loader="train-loader",
            val_loader="val-loader",
            data_generator="generator",
        )

    monkeypatch.setattr(script_module, "build_run_config", _fake_build_run_config)
    monkeypatch.setattr(
        script_module,
        "generate_base_data",
        _fake_generate_base_data,
    )
    monkeypatch.setattr(
        script_module,
        "split_base_data",
        _fake_split_base_data,
    )
    monkeypatch.setattr(
        script_module,
        "build_spni_training_view",
        _fake_build_spni_training_view,
    )
    cfg = _cfg_stub()
    opt_model = SimpleNamespace(label="opt-model")
    cache_options = CacheReplaceOptions(
        replace_data=True,
        replace_intd_adv=True,
        replace_pred=True,
        archive_replaced=False,
    )

    # Act by calling the legacy helper.
    training_data, testing_data, normalization_constant, metadata = \
        script_module.gen_train_data(
            cfg,
            opt_model,
            interdiction_policy="random",
            cache_options=cache_options,
        )

    # Assert that the wrapper mapped inputs and preserved the legacy outputs.
    cache_policy = calls["build_run_config"]["cache_policy"]
    assert calls["build_run_config"]["cfg"] is cfg, \
        "gen_train_data should normalize the original legacy config object."
    assert cache_policy.replace_data is True, \
        "gen_train_data should forward replace_data into CachePolicy."
    assert cache_policy.replace_intd_adv is True, \
        "gen_train_data should forward replace_intd_adv into CachePolicy."
    assert cache_policy.replace_pred is True, \
        "gen_train_data should forward replace_pred into CachePolicy."
    assert cache_policy.archive_replaced is False, \
        "gen_train_data should forward archive_replaced into CachePolicy."
    assert calls["generate_base_data"]["run_cfg"].label == "run-cfg", \
        "gen_train_data should pass the normalized run config downstream."
    assert calls["generate_base_data"]["graph_bundle"].opt_model is opt_model, \
        "gen_train_data should preserve the caller-provided opt_model."
    assert calls["build_spni_training_view"]["interdiction_policy"] == "random", \
        "gen_train_data should preserve the requested interdiction policy."
    assert training_data == {
        "train_loader": "train-loader",
        "val_loader": "val-loader",
    }, "gen_train_data should preserve the legacy training-loader mapping."
    assert np.array_equal(
        testing_data["feats"],
        np.array([[8.0], [9.0]], dtype=float),
    ), "gen_train_data should preserve the legacy testing features key."
    assert np.array_equal(
        testing_data["costs"],
        np.array([[10.0], [11.0]], dtype=float),
    ), "gen_train_data should preserve the legacy testing costs key."
    assert normalization_constant == 7.5, \
        "gen_train_data should preserve the normalization constant output."
    assert metadata == {"data_generator": "generator"}, \
        "gen_train_data should preserve the legacy metadata mapping."
    pass


def test_scripts_setup_gen_train_data_uses_global_cache_options_when_missing(
    monkeypatch,
):
    """Verify that `gen_train_data(...)` falls back to global cache options."""
    # Arrange a minimal delegation path plus a global cache-option stub.
    cache_policies: list[object] = []

    def _fake_get_cache_replace_options():
        return CacheReplaceOptions(replace_result=True, replace_fig=True)

    def _fake_build_run_config(cfg, *, cache_policy):
        cache_policies.append(cache_policy)
        return SimpleNamespace(label="run-cfg")

    monkeypatch.setattr(
        script_module,
        "get_cache_replace_options",
        _fake_get_cache_replace_options,
    )
    monkeypatch.setattr(script_module, "build_run_config", _fake_build_run_config)
    monkeypatch.setattr(
        script_module,
        "generate_base_data",
        lambda *args, **kwargs: SimpleNamespace(features="f", costs="c"),
    )
    monkeypatch.setattr(
        script_module,
        "split_base_data",
        lambda *args, **kwargs: SimpleNamespace(
            test_features="tf",
            test_costs="tc",
            normalization_constant=1.0,
        ),
    )
    monkeypatch.setattr(
        script_module,
        "build_spni_training_view",
        lambda *args, **kwargs: SimpleNamespace(
            train_loader="train",
            val_loader="val",
            data_generator="generator",
        ),
    )

    # Act by omitting the explicit cache-options argument.
    script_module.gen_train_data(_cfg_stub(), SimpleNamespace(label="opt"))

    # Assert that the global cache settings were converted into CachePolicy.
    assert len(cache_policies) == 1, \
        "gen_train_data should resolve one cache policy per wrapper call."
    assert cache_policies[0].replace_result is True, \
        "gen_train_data should honor global replace_result settings."
    assert cache_policies[0].replace_fig is True, \
        "gen_train_data should honor global replace_fig settings."
    pass


################################
### test setup_dfl_predictor ###
################################


def test_scripts_setup_dfl_predictor_forwards_underprediction_options(
    monkeypatch,
):
    """Verify that DFL setup passes surrogate safety knobs to DFLTrainer."""
    # Arrange a trainer stub that records constructor and fit inputs.
    captured: dict[str, object] = {}

    class _FakeDFLTrainer:
        def __init__(self, **kwargs):
            captured["trainer_kwargs"] = kwargs

        def fit(self, train_loader, val_loader, **kwargs):
            captured["fit_loaders"] = (train_loader, val_loader)
            captured["fit_kwargs"] = kwargs
            return [0.0], [0.0], [0.0], [0.0]

    def _fake_write_pred(cfg, state_dict, *, artifact_tag, replace):
        captured["write_pred"] = {
            "cfg": cfg,
            "state_dict": state_dict,
            "artifact_tag": artifact_tag,
            "replace": replace,
        }

    monkeypatch.setattr(script_module, "DFLTrainer", _FakeDFLTrainer)
    monkeypatch.setattr(
        script_module.pyepo.func,
        "SPOPlus",
        lambda opt_model, processes: SimpleNamespace(
            opt_model=opt_model,
            processes=processes,
        ),
    )
    monkeypatch.setattr(script_module, "write_pred", _fake_write_pred)

    cfg = _cfg_stub()
    graph = SimpleNamespace(num_cost=3)
    opt_model = SimpleNamespace(label="opt-model")
    training_data = {
        "train_loader": SimpleNamespace(label="train"),
        "val_loader": SimpleNamespace(label="val"),
    }
    cache_options = CacheReplaceOptions(
        replace_pred=True,
        archive_replaced=False,
    )

    # Act by constructing a DFL predictor through the setup wrapper.
    predictor = script_module.setup_dfl_predictor(
        cfg,
        graph,
        opt_model,
        training_data,
        cache_options=cache_options,
    )

    # Assert that the DFLTrainer received the configured safety options.
    trainer_kwargs = captured["trainer_kwargs"]
    assert trainer_kwargs["surrogate_underprediction_penalty_weight"] == 7.0, \
        "setup_dfl_predictor did not forward the underprediction penalty."
    assert trainer_kwargs["surrogate_underprediction_margin"] == 0.25, \
        "setup_dfl_predictor did not forward the underprediction margin."
    assert captured["fit_kwargs"]["max_lr_reductions"] == 2, \
        "setup_dfl_predictor did not preserve the LR-reduction limit."
    assert isinstance(predictor, torch.nn.Module), \
        "setup_dfl_predictor should still return the predictor model."
    pass
