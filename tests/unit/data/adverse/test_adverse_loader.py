from types import SimpleNamespace

import numpy as np

import dflintdpy.data.adverse.adverse_loader as adverse_loader_module
from dflintdpy.data.adverse.adverse_loader import AdvLoader


############################
### Helper functionality ###
############################


class _DatasetStub:
    """Minimal dataset stub for adverse-loader tests."""

    def __init__(self):
        self.normal_calls = 0
        self.adverse_calls = 0
        self.nonadverse_dataset = SimpleNamespace(label="nonadverse-dataset")

    def normal_mode(self) -> None:
        """Record one transition into normal mode."""
        self.normal_calls += 1

    def adverse_mode(self) -> None:
        """Record one transition into adverse mode."""
        self.adverse_calls += 1

    def get_nonadverse_dataset(self):
        """Return a deterministic nonadverse dataset stub."""
        return self.nonadverse_dataset


class _SamplerStub:
    """Record sampler construction and epoch updates."""

    init_calls: list[dict] = []

    def __init__(self, dataset, *, seed, shuffle):
        self.dataset = dataset
        self.seed = seed
        self.shuffle = shuffle
        self.set_epoch_calls: list[int] = []
        type(self).init_calls.append(
            {
                "dataset": dataset,
                "seed": seed,
                "shuffle": shuffle,
            }
        )

    @classmethod
    def reset(cls) -> None:
        """Clear recorded constructor calls."""
        cls.init_calls = []

    def set_epoch(self, epoch: int) -> None:
        """Record one epoch update."""
        self.set_epoch_calls.append(epoch)


class _DataLoaderStub:
    """Record DataLoader construction and expose iterable behavior."""

    init_calls: list[dict] = []

    def __init__(self, dataset, *, batch_size, shuffle, sampler, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.kwargs = kwargs
        type(self).init_calls.append(
            {
                "dataset": dataset,
                "batch_size": batch_size,
                "shuffle": shuffle,
                "sampler": sampler,
                "kwargs": kwargs,
            }
        )

    @classmethod
    def reset(cls) -> None:
        """Clear recorded constructor calls."""
        cls.init_calls = []

    def __iter__(self):
        """Return a deterministic iterator payload."""
        return iter([("batch", self.batch_size)])

    def __len__(self):
        """Return a deterministic loader length."""
        return 4


#####################
### test __init__ ###
#####################


def test_adv_loader_init_builds_sampler_and_loader(monkeypatch):
    """Verify that AdvLoader wires the sampler into DataLoader."""
    # Arrange: replace the sampler and DataLoader with recording stubs.
    _SamplerStub.reset()
    _DataLoaderStub.reset()
    monkeypatch.setattr(adverse_loader_module, "AdvSampler", _SamplerStub)
    monkeypatch.setattr(adverse_loader_module, "DataLoader", _DataLoaderStub)
    dataset = _DatasetStub()

    # Act: construct the adverse loader.
    loader = AdvLoader(
        dataset,
        batch_size=8,
        seed=17,
        shuffle=True,
        drop_last=True,
    )

    # Assert: sampler and DataLoader saw the expected arguments.
    assert len(_SamplerStub.init_calls) == 1, \
        "AdvLoader did not construct exactly one sampler."
    assert _SamplerStub.init_calls[0]["dataset"] is dataset, \
        "AdvLoader passed the wrong dataset into the sampler."
    assert _SamplerStub.init_calls[0]["seed"] == 17, \
        "AdvLoader passed the wrong seed into the sampler."
    assert _SamplerStub.init_calls[0]["shuffle"] is True, \
        "AdvLoader passed the wrong shuffle flag into the sampler."
    assert len(_DataLoaderStub.init_calls) == 1, \
        "AdvLoader did not construct exactly one DataLoader."
    assert _DataLoaderStub.init_calls[0]["shuffle"] is False, \
        "AdvLoader did not force DataLoader shuffle off."
    assert _DataLoaderStub.init_calls[0]["sampler"] is loader.sampler, \
        "AdvLoader did not pass the constructed sampler into DataLoader."
    assert _DataLoaderStub.init_calls[0]["kwargs"]["drop_last"] is True, \
        "AdvLoader did not forward extra DataLoader kwargs."
    pass


##########################
### test delegations ###
##########################


def test_adv_loader_delegates_epoch_and_mode_changes(monkeypatch):
    """Verify that loader helpers delegate to the sampler and dataset."""
    # Arrange: build a loader with stubbed collaborators.
    _SamplerStub.reset()
    _DataLoaderStub.reset()
    monkeypatch.setattr(adverse_loader_module, "AdvSampler", _SamplerStub)
    monkeypatch.setattr(adverse_loader_module, "DataLoader", _DataLoaderStub)
    dataset = _DatasetStub()
    loader = AdvLoader(dataset, batch_size=4, seed=3, shuffle=False)

    # Act: exercise the epoch and mode helpers.
    loader.set_epoch(5)
    loader.normal_mode()
    loader.adverse_mode()

    # Assert: the sampler and dataset recorded the delegated calls.
    assert loader.sampler.set_epoch_calls == [5], \
        "AdvLoader did not delegate epoch updates to the sampler."
    assert dataset.normal_calls == 1, \
        "AdvLoader did not delegate normal_mode to the dataset."
    assert dataset.adverse_calls == 1, \
        "AdvLoader did not delegate adverse_mode to the dataset."
    pass


#####################################
### test get_nonadverse_loader ###
#####################################


def test_adv_loader_get_nonadverse_loader_preserves_loader_settings(
    monkeypatch,
):
    """Verify that get_nonadverse_loader reuses the loader configuration."""
    # Arrange: replace the sampler and DataLoader with stubs.
    _SamplerStub.reset()
    _DataLoaderStub.reset()
    monkeypatch.setattr(adverse_loader_module, "AdvSampler", _SamplerStub)
    monkeypatch.setattr(adverse_loader_module, "DataLoader", _DataLoaderStub)
    dataset = _DatasetStub()
    loader = AdvLoader(dataset, batch_size=6, seed=13, shuffle=False)

    # Act: request the nonadverse loader variant.
    nonadverse_loader = loader.get_nonadverse_loader()

    # Assert: the new loader uses the nonadverse dataset with old settings.
    assert isinstance(nonadverse_loader, AdvLoader), \
        "get_nonadverse_loader did not return an AdvLoader instance."
    assert nonadverse_loader.dataset is dataset.nonadverse_dataset, \
        "get_nonadverse_loader did not use the nonadverse dataset."
    assert nonadverse_loader.loader.batch_size == 6, \
        "get_nonadverse_loader did not preserve the batch size."
    assert nonadverse_loader.sampler.seed == 13, \
        "get_nonadverse_loader did not preserve the sampler seed."
    assert nonadverse_loader.sampler.shuffle is False, \
        "get_nonadverse_loader did not preserve the shuffle flag."
    pass


###############################
### test wrapper behavior ###
###############################


def test_adv_loader_iter_and_len_forward_to_wrapped_loader(monkeypatch):
    """Verify that AdvLoader forwards iteration and length to DataLoader."""
    # Arrange: construct a loader with stubbed collaborators.
    _SamplerStub.reset()
    _DataLoaderStub.reset()
    monkeypatch.setattr(adverse_loader_module, "AdvSampler", _SamplerStub)
    monkeypatch.setattr(adverse_loader_module, "DataLoader", _DataLoaderStub)
    loader = AdvLoader(_DatasetStub(), batch_size=5, seed=1, shuffle=True)

    # Act: iterate once and query the wrapper length.
    batches = list(iter(loader))
    length = len(loader)

    # Assert: wrapper behavior matches the wrapped DataLoader stub.
    assert batches == [("batch", 5)], \
        "AdvLoader did not forward iteration to the wrapped DataLoader."
    assert length == 4, \
        "AdvLoader did not forward length to the wrapped DataLoader."
    pass
