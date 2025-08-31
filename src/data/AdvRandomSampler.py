from typing import Literal, Iterable
from .AdvDataset import AdvDataset
import torch
from torch.utils.data import Sampler

class AdvRandomSampler(Sampler[int]):
    """
    Yields indices in a shuffled order. In 'normal' mode, yields only the
    first n of that shuffle; in 'adverse' mode, yields all.
    """

    def __init__(self, 
                 data_set: AdvDataset, 
                 n_org_samples: int,
                 *,
                 mode: Literal["normal","adverse"] = "normal",
                 seed: int = 0, 
                 shuffle: bool = True):
        """
        Initializes the adversarial random sampler.

        Parameters:
        -----------
        data_set : AdvDataset
            The adversarial dataset.
        n_org_samples : int
            The number of original samples to return.
        mode : Literal["normal","adverse"], optional
            The mode of the sampler ('normal' or 'adverse'). 
            Defaults to 'normal'.
        seed : int, optional
            The random seed for shuffling. Defaults to 0.
        shuffle : bool, optional
            Whether to shuffle the data. Defaults to True.
        """

        # Store parameters
        self.data_set = data_set
        self.n_full = len(data_set)
        self.n_org_samples = min(int(n_org_samples), self.n_full)
        self.seed = int(seed)
        self.shuffle = shuffle
        self.epoch = 0
        self._last_perm = None

        # Set the interdiction mode
        if mode == 'normal':
            self._return_intds = False
        elif mode == 'adverse':
            self._return_intds = True
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __iter__(self) -> Iterable[int]:

        # Create a new random generator for this epoch
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Generate a random permutation of indices
        if self.shuffle:
            perm = torch.randperm(self.n_full, generator=g).tolist()
        else:
            perm = list(range(self.n_full))

        # Store the last generated permutation
        self._last_perm = perm
        if not self._return_intds:
            perm = perm[:self.n_org_samples]
        return iter(perm)

    def __len__(self) -> int:
        return self.n_full if self._return_intds else self.n_org_samples

    def normal_mode(self) -> None:
        """
        Sets the dataset to normal mode.
        The iterator will only return the original samples.
        """
        self._return_intds = False

    def adverse_mode(self) -> None:
        """
        Sets the dataset to adverse mode.
        The iterator will return all samples, including interdictions.
        """
        self._return_intds = True

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch for the sampler.

        Parameters:
        -----------
        epoch : int
            The epoch number.
        """
        self.epoch = epoch

    def get_mode(self) -> str:
        """
        Returns the current mode of the dataset.

        Returns:
        --------
        str
            The current mode ('normal' or 'adverse').
        """
        return 'adverse' if self._return_intds else 'normal'

    def first_n_indices(self) -> list[int]:
        """
        Optionally retrieve the first-n indices for the most recent shuffle.
        """
        if self._last_perm is None:
            # Not iterated yet this epoch; recompute deterministically
            g = torch.Generator(); g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(self.n_full, generator=g).tolist() if self.shuffle else list(range(self.n_full))
        else:
            perm = self._last_perm

        # Return the first-n indices
        return perm[:self.n_org_samples]