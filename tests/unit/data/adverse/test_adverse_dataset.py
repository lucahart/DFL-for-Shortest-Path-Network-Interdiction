from types import SimpleNamespace

import numpy as np
import pytest
import torch

import dflintdpy.data.adverse.adverse_dataset as adverse_dataset_module
from dflintdpy.data.adverse.adverse_dataset import (
    AdvDataset,
    generate_opt_dataset,
)


################
### Fixtures ###
################


@pytest.fixture
def feats() -> np.ndarray:
    """Return a small feature matrix for adverse-dataset tests."""
    return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)


@pytest.fixture
def costs_grouped() -> np.ndarray:
    """Return grouped scenario costs for adverse-dataset tests."""
    return np.array(
        [
            [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        ],
        dtype=float,
    )


@pytest.fixture
def intds_grouped() -> np.ndarray:
    """Return grouped interdictions matching the grouped costs."""
    return np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        ],
        dtype=float,
    )


############################
### Helper functionality ###
############################


class _OptModelStub:
    """Minimal optimization-model stub for dataset tests."""

    def __init__(self):
        self.label = "opt-model"


class _DatasetRecorder:
    """Recorder class used to capture dataset-construction arguments."""

    init_calls: list[dict] = []

    def __init__(
        self,
        opt_model,
        feats,
        costs_grouped,
        intds_grouped,
        mode="normal",
    ):
        type(self).init_calls.append(
            {
                "opt_model": opt_model,
                "feats": np.array(feats, dtype=float),
                "costs_grouped": np.array(costs_grouped, dtype=float),
                "intds_grouped": np.array(intds_grouped, dtype=float),
                "mode": mode,
            }
        )
        self.opt_model = opt_model
        self.feats = np.array(feats, dtype=float)
        self.costs_grouped = np.array(costs_grouped, dtype=float)
        self.intds_grouped = np.array(intds_grouped, dtype=float)
        self.mode = mode

    @classmethod
    def reset(cls) -> None:
        """Clear recorded constructor arguments."""
        cls.init_calls = []


def _patch_parent_dataset_init(monkeypatch) -> list[dict]:
    """Patch optDataset.__init__ with a deterministic lightweight stub."""
    init_calls: list[dict] = []

    def _fake_init(self, opt_model, feats, costs):
        init_calls.append(
            {
                "opt_model": opt_model,
                "feats": np.array(feats, dtype=float),
                "costs": np.array(costs, dtype=float),
            }
        )
        self.model = opt_model
        self.sols = np.array(costs, dtype=float) + 1.0
        self.objs = np.array(costs, dtype=float).sum(axis=1)

    monkeypatch.setattr(adverse_dataset_module.optDataset, "__init__", _fake_init)
    return init_calls


def _build_dataset(
    monkeypatch,
    feats: np.ndarray,
    costs_grouped: np.ndarray,
    intds_grouped: np.ndarray,
    mode: str = "normal",
) -> tuple[AdvDataset, list[dict]]:
    """Create an AdvDataset with the parent dataset logic stubbed out."""
    init_calls = _patch_parent_dataset_init(monkeypatch)
    dataset = AdvDataset(
        _OptModelStub(),
        feats,
        costs_grouped,
        intds_grouped,
        mode=mode,
    )
    return dataset, init_calls


#####################
### test __init__ ###
#####################


def test_adv_dataset_init_flattens_inputs_and_regroups_parent_outputs(
    monkeypatch,
    feats,
    costs_grouped,
    intds_grouped,
):
    """Verify that AdvDataset flattens scenarios for the parent dataset."""
    # Arrange / Act: construct the dataset with a lightweight parent stub.
    dataset, init_calls = _build_dataset(
        monkeypatch,
        feats,
        costs_grouped,
        intds_grouped,
    )

    # Assert: the parent saw flattened inputs and grouped outputs were restored.
    expected_feats_flat = np.repeat(feats, 2, axis=0)
    expected_costs_flat = costs_grouped.reshape(4, 3)
    assert len(init_calls) == 1, \
        "AdvDataset did not call the parent dataset constructor exactly once."
    assert np.array_equal(init_calls[0]["feats"], expected_feats_flat), \
        "AdvDataset did not repeat features for each scenario."
    assert np.array_equal(init_calls[0]["costs"], expected_costs_flat), \
        "AdvDataset did not flatten grouped costs for the parent dataset."
    assert dataset.sols.shape == (2, 2, 3), \
        "AdvDataset did not regroup the parent solutions correctly."
    assert dataset.objs.shape == (2, 2), \
        "AdvDataset did not regroup the parent objective values correctly."
    assert np.array_equal(dataset.feats, feats), \
        "AdvDataset did not preserve the original feature matrix."
    assert np.array_equal(dataset.intds, intds_grouped), \
        "AdvDataset did not preserve the grouped interdictions."
    pass


def test_adv_dataset_init_rejects_unknown_mode(
    monkeypatch,
    feats,
    costs_grouped,
    intds_grouped,
):
    """Verify that AdvDataset rejects unsupported retrieval modes."""
    # Arrange: patch the heavy parent dataset constructor.
    _patch_parent_dataset_init(monkeypatch)

    # Act / Assert: construction should fail for bad mode values.
    with pytest.raises(ValueError, match="Unknown mode: OTHER"):
        AdvDataset(
            _OptModelStub(),
            feats,
            costs_grouped,
            intds_grouped,
            mode="OTHER",
        )
    pass


########################
### test __getitem__ ###
########################


def test_adv_dataset_getitem_returns_normal_view(
    monkeypatch,
    feats,
    costs_grouped,
    intds_grouped,
):
    """Verify that normal mode returns only the first scenario."""
    # Arrange: create a normal-mode dataset with deterministic parent outputs.
    dataset, _ = _build_dataset(
        monkeypatch,
        feats,
        costs_grouped,
        intds_grouped,
        mode="normal",
    )

    # Act: fetch one item from the normal-mode dataset.
    feat, cost, sol, obj = dataset[0]

    # Assert: only the base scenario is returned in normal mode.
    assert isinstance(feat, torch.FloatTensor), \
        "AdvDataset did not return features as torch tensors."
    assert torch.equal(feat, torch.FloatTensor(feats[0])), \
        "AdvDataset returned the wrong feature vector in normal mode."
    assert torch.equal(cost, torch.FloatTensor(costs_grouped[0, 0])), \
        "AdvDataset returned the wrong cost vector in normal mode."
    assert torch.equal(
        sol,
        torch.FloatTensor(costs_grouped[0, 0] + 1.0),
    ), "AdvDataset returned the wrong solution vector in normal mode."
    assert obj.item() == pytest.approx(costs_grouped[0, 0].sum()), \
        "AdvDataset returned the wrong objective value in normal mode."
    pass


def test_adv_dataset_getitem_returns_adverse_view(
    monkeypatch,
    feats,
    costs_grouped,
    intds_grouped,
):
    """Verify that adverse mode returns all grouped scenarios and intds."""
    # Arrange: create a dataset and switch it to adverse mode.
    dataset, _ = _build_dataset(
        monkeypatch,
        feats,
        costs_grouped,
        intds_grouped,
        mode="normal",
    )
    dataset.adverse_mode()

    # Act: fetch one item from the adverse-mode dataset.
    feat, costs, sols, objs, intds = dataset[1]

    # Assert: all grouped scenarios are returned in adverse mode.
    assert torch.equal(feat, torch.FloatTensor(feats[1])), \
        "AdvDataset returned the wrong feature vector in adverse mode."
    assert torch.equal(costs, torch.FloatTensor(costs_grouped[1])), \
        "AdvDataset returned the wrong grouped costs in adverse mode."
    assert torch.equal(
        sols,
        torch.FloatTensor(costs_grouped[1] + 1.0),
    ), "AdvDataset returned the wrong grouped solutions in adverse mode."
    assert torch.equal(
        objs,
        torch.FloatTensor(costs_grouped[1].sum(axis=1)),
    ), "AdvDataset returned the wrong grouped objectives in adverse mode."
    assert torch.equal(intds, torch.FloatTensor(intds_grouped[1])), \
        "AdvDataset returned the wrong grouped interdictions in adverse mode."
    pass


######################
### test mode API ###
######################


def test_adv_dataset_mode_helpers_toggle_return_mode(
    monkeypatch,
    feats,
    costs_grouped,
    intds_grouped,
):
    """Verify that the dataset mode helpers stay in sync with get_mode."""
    # Arrange: create a dataset in normal mode.
    dataset, _ = _build_dataset(
        monkeypatch,
        feats,
        costs_grouped,
        intds_grouped,
        mode="normal",
    )

    # Act / Assert: toggle between normal and adverse views.
    assert dataset.get_mode() == "normal", \
        "AdvDataset did not start in normal mode."
    dataset.adverse_mode()
    assert dataset.get_mode() == "adverse", \
        "AdvDataset did not switch to adverse mode."
    dataset.normal_mode()
    assert dataset.get_mode() == "normal", \
        "AdvDataset did not switch back to normal mode."
    pass


######################################
### test get_nonadverse_dataset ###
######################################


def test_adv_dataset_get_nonadverse_dataset_slices_first_scenario(
    monkeypatch,
    feats,
    costs_grouped,
    intds_grouped,
):
    """Verify that get_nonadverse_dataset keeps only scenario zero."""
    # Arrange: create a grouped dataset and replace the constructor target.
    dataset, _ = _build_dataset(
        monkeypatch,
        feats,
        costs_grouped,
        intds_grouped,
        mode="adverse",
    )
    _DatasetRecorder.reset()
    monkeypatch.setattr(adverse_dataset_module, "AdvDataset", _DatasetRecorder)

    # Act: request the nonadverse dataset view.
    result = dataset.get_nonadverse_dataset()

    # Assert: the new dataset sees only the first scenario in normal mode.
    assert isinstance(result, _DatasetRecorder), \
        "get_nonadverse_dataset did not return the recorded dataset stub."
    assert len(_DatasetRecorder.init_calls) == 1, \
        "get_nonadverse_dataset did not construct exactly one dataset."
    assert np.array_equal(
        _DatasetRecorder.init_calls[0]["costs_grouped"],
        costs_grouped[:, :1, :],
    ), "get_nonadverse_dataset did not keep only the first cost scenario."
    assert np.array_equal(
        _DatasetRecorder.init_calls[0]["intds_grouped"],
        intds_grouped[:, :1, :],
    ), "get_nonadverse_dataset did not keep only the first interdiction."
    assert _DatasetRecorder.init_calls[0]["mode"] == "normal", \
        "get_nonadverse_dataset did not request normal mode."
    pass


################################
### test generate_opt_dataset ###
################################


def test_adv_dataset_generate_opt_dataset_wraps_single_scenario(monkeypatch):
    """Verify that generate_opt_dataset wraps costs with zero interdictions."""
    # Arrange: replace AdvDataset with a simple recorder class.
    _DatasetRecorder.reset()
    monkeypatch.setattr(adverse_dataset_module, "AdvDataset", _DatasetRecorder)
    feats = np.array([[1.0, 2.0]], dtype=float)
    costs = np.array([[3.0, 4.0, 5.0]], dtype=float)
    opt_model = SimpleNamespace(label="opt-model")

    # Act: generate the one-scenario adverse dataset wrapper.
    result = generate_opt_dataset(opt_model, feats, costs)

    # Assert: costs were wrapped and interdictions were zeroed out.
    assert isinstance(result, _DatasetRecorder), \
        "generate_opt_dataset did not return the recorded dataset stub."
    assert len(_DatasetRecorder.init_calls) == 1, \
        "generate_opt_dataset did not construct exactly one dataset."
    assert np.array_equal(
        _DatasetRecorder.init_calls[0]["costs_grouped"],
        costs[:, np.newaxis, :],
    ), "generate_opt_dataset did not add the scenario axis to costs."
    assert np.array_equal(
        _DatasetRecorder.init_calls[0]["intds_grouped"],
        np.zeros((1, 1, 3), dtype=float),
    ), "generate_opt_dataset did not create zero interdictions."
    assert _DatasetRecorder.init_calls[0]["mode"] == "normal", \
        "generate_opt_dataset did not request normal mode."
    pass
