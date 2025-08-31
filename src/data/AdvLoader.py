
from torch.utils.data import Dataset, DataLoader

from .AdvDataset import AdvDataset
from .AdvRandomSampler import AdvRandomSampler

class AdvLoader:
    """
    Composition wrapper: holds dataset+sampler+DataLoader and keeps them in sync.
    Exposes .loader (a real DataLoader), but also acts like one (iter/len).
    """

    def __init__(self, 
                 dataset: AdvDataset, 
                 n_org_samples: int, 
                 *,
                 batch_size: int = 32, 
                 seed: int = 0, 
                 shuffle: bool = True, 
                 **kwargs):

        # Store adverse dataset and create sampler
        self.dataset = dataset
        self.sampler = AdvRandomSampler(
            self.dataset, 
            n_org_samples=n_org_samples,
            seed=seed, 
            shuffle=shuffle,
            mode=dataset.get_mode() # initializes to mode of AdvDataset
        )

        # Pass sampler and keep shuffle=False (sampler handles it)
        self.loader  = DataLoader(
            self.dataset, 
            batch_size=batch_size,
            shuffle=False, 
            sampler=self.sampler, 
            **kwargs
        )

    def set_epoch(self, epoch: int) -> None: 
        """
        Sets the epoch for the sampler.
        """
        self.sampler.set_epoch(epoch)

    def normal_mode(self) -> None:
        """
        Sets the dataset to normal mode.
        When iterating over the dataset, 
        only uninterdicted samples will be returned.
        """
        self.dataset.normal_mode(); self.sampler.normal_mode()

    def adverse_mode(self) -> None:
        """
        Sets the dataset to adverse mode.
        When iterating over the dataset,
        interdictions will be returned.
        """
        self.dataset.adverse_mode(); self.sampler.adverse_mode()

    # make this wrapper quack like a DataLoader
    def __iter__(self):  return iter(self.loader)
    def __len__(self):   return len(self.loader)

    # optional helpers
    def first_n_indices(self): return self.sampler.first_n_indices()
