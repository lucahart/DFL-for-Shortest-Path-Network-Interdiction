import torch

from dflintdpy.data.adverse.adverse_sampler import AdvSampler


############################
### Helper functionality ###
############################


class _DatasetStub:
    """Minimal dataset stub with a configurable length."""

    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        """Return the configured dataset length."""
        return self.size


#####################
### test __iter__ ###
#####################


def test_adv_sampler_iter_returns_sequential_indices_without_shuffle():
    """Verify that AdvSampler yields ordered indices when shuffle is off."""
    # Arrange: build a sampler over a five-item dataset.
    sampler = AdvSampler(_DatasetStub(5), seed=11, shuffle=False)

    # Act: materialize the iteration order.
    indices = list(iter(sampler))

    # Assert: non-shuffled sampling stays sequential.
    assert indices == [0, 1, 2, 3, 4], \
        "AdvSampler did not return sequential indices without shuffling."
    pass


def test_adv_sampler_iter_uses_seed_plus_epoch_for_shuffle():
    """Verify that AdvSampler shuffles deterministically per epoch."""
    # Arrange: build a shuffled sampler and compute expected permutations.
    sampler = AdvSampler(_DatasetStub(5), seed=11, shuffle=True)
    g0 = torch.Generator()
    g0.manual_seed(11)
    expected_epoch0 = torch.randperm(5, generator=g0).tolist()
    g3 = torch.Generator()
    g3.manual_seed(14)
    expected_epoch3 = torch.randperm(5, generator=g3).tolist()

    # Act: materialize the order at two epochs.
    epoch0 = list(iter(sampler))
    sampler.set_epoch(3)
    epoch3 = list(iter(sampler))

    # Assert: the sampler uses the base seed plus the epoch.
    assert epoch0 == expected_epoch0, \
        "AdvSampler did not use the base seed for epoch zero."
    assert epoch3 == expected_epoch3, \
        "AdvSampler did not use seed plus epoch for later shuffles."
    assert epoch0 != epoch3, \
        "AdvSampler produced the same shuffled order across epochs."
    pass


#########################
### test metadata API ###
#########################


def test_adv_sampler_len_and_set_epoch_update_sampler_state():
    """Verify that length and epoch updates track dataset metadata."""
    # Arrange: build a sampler over a four-item dataset.
    sampler = AdvSampler(_DatasetStub(4), seed=7, shuffle=True)

    # Act: query the length and update the epoch.
    length = len(sampler)
    sampler.set_epoch(9)

    # Assert: metadata stays in sync with the dataset and caller updates.
    assert length == 4, \
        "AdvSampler did not report the underlying dataset length."
    assert sampler.epoch == 9, \
        "AdvSampler did not store the requested epoch."
    assert sampler.seed == 7, \
        "AdvSampler did not preserve the configured base seed."
    pass
