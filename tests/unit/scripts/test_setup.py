from types import SimpleNamespace

import numpy as np

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
        "po_epochs": 1,
        "spo_epochs": 1,
        "po_lr": 1e-3,
        "spo_lr": 1e-3,
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
