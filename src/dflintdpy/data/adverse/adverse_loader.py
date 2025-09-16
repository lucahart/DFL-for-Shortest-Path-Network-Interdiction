
from torch.utils.data import DataLoader

from dflintdpy.data.adverse.adverse_dataset import AdvDataset
from dflintdpy.data.adverse.adverse_sampler import AdvSampler


class AdvLoader:
    """
    Composition wrapper: holds dataset+sampler+DataLoader and keeps them in sync.
    Exposes .loader (a real DataLoader), but also acts like one (iter/len).
    """

    def __init__(
        self,
        dataset: AdvDataset,
        *,
        batch_size: int = 32,
        seed: int = 0,
        shuffle: bool = True,
        **kwargs,
    ):

        # Store adverse dataset and create sampler
        self.dataset = dataset
        self.sampler = AdvSampler(
            self.dataset,
            seed=seed,
            shuffle=shuffle,
        )

        # Pass sampler and keep shuffle=False (sampler handles it)
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=self.sampler,
            **kwargs,
        )

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for the sampler."""
        self.sampler.set_epoch(epoch)

    def normal_mode(self) -> None:
        """Sets the dataset to normal mode."""
        self.dataset.normal_mode()

    def adverse_mode(self) -> None:
        """Sets the dataset to adverse mode."""
        self.dataset.adverse_mode()

    # make this wrapper quack like a DataLoader
    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)
