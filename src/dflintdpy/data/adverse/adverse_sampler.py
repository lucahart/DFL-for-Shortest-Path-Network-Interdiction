from typing import Iterable
from dflintdpy.data.adverse.adverse_dataset import AdvDataset
import torch
from torch.utils.data import Sampler


class AdvSampler(Sampler[int]):
    """Sampler with epoch-based seeding for adversarial datasets."""

    def __init__(
        self,
        data_set: AdvDataset,
        *,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        """Initialize sampler.

        Parameters
        ----------
        data_set : AdvDataset
            Dataset to sample from.
        seed : int, optional
            Base random seed. Defaults to ``0``.
        shuffle : bool, optional
            If ``True`` draw a random permutation each epoch, otherwise
            iterate sequentially. Defaults to ``True``.
        """

        self.data_set = data_set
        self.n_full = len(data_set)
        self.seed = int(seed)
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self) -> Iterable[int]:
        """Return an iterator over dataset indices."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        if self.shuffle:
            perm = torch.randperm(self.n_full, generator=g).tolist()
        else:
            perm = list(range(self.n_full))

        return iter(perm)

    def __len__(self) -> int:
        """Number of samples provided by this sampler."""
        return self.n_full

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for deterministic shuffling."""
        self.epoch = int(epoch)

