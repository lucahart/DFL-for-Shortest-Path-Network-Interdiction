from models.ShortestPathGrb import shortestPathGrb
import numpy as np

from pyepo.data.dataset import optDataset
import torch


class AdvDataset(optDataset):
    """Dataset for adverse network training under interdictions.

    Parameters
    ----------
    opt_model : :class:`shortestPathGrb`
        The optimization model used to solve each scenario.
    feats : ``np.ndarray``
        Feature matrix with shape ``(n_samples, n_feat)``.
    costs_grouped : ``np.ndarray``
        Cost tensors grouped by scenario with shape
        ``(n_samples, num_scenarios, m)``.
    intds_grouped : ``np.ndarray``
        Interdiction tensors grouped by scenario with the same shape as
        ``costs_grouped``.
    mode : str, optional
        ``"normal"`` or ``"adverse"``.  In normal mode only the first
        (uninterdicted) scenario is returned.
    """

    _return_intds: bool  # whether to return interdictions

    def __init__(self,
                 opt_model: shortestPathGrb,
                 feats: np.ndarray,
                 costs_grouped: np.ndarray,
                 intds_grouped: np.ndarray,
                 mode: str = "normal"):
        # store dimensions
        self.n_samples = feats.shape[0]
        self.num_scenarios = costs_grouped.shape[1]

        # Flatten scenarios so that ``optDataset`` can solve them
        # feats are repeated for each scenario
        feats_flat = np.repeat(feats, self.num_scenarios, axis=0)
        costs_flat = costs_grouped.reshape(self.n_samples * self.num_scenarios, -1)

        # call parent constructor to compute solutions/objs
        super().__init__(opt_model, feats_flat, costs_flat)

        # reshape solutions and objective values back to grouped form
        self.sols = self.sols.reshape(self.n_samples, self.num_scenarios, -1)
        self.objs = self.objs.reshape(self.n_samples, self.num_scenarios)

        # store original arrays for retrieval
        self.feats = feats
        self.costs = costs_grouped
        self.intds = intds_grouped

        # Set the interdiction mode
        if mode == "normal":
            self._return_intds = False
        elif mode == "adverse":
            self._return_intds = True
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __len__(self) -> int:  # type: ignore[override]
        """Number of original (non-scenario) samples."""
        return self.n_samples

    def __getitem__(self, index):  # type: ignore[override]
        """Retrieve data for a single instance.

        In normal mode only the first scenario is returned.  In adverse
        mode all scenarios together with their interdictions are returned.
        """

        feat = torch.FloatTensor(self.feats[index])
        costs = torch.FloatTensor(self.costs[index])
        sols = torch.FloatTensor(self.sols[index])
        objs = torch.FloatTensor(self.objs[index])
        intds = torch.FloatTensor(self.intds[index])

        if self._return_intds:
            return feat, costs, sols, objs, intds
        else:
            return feat, costs[0], sols[0], objs[0]

    def normal_mode(self) -> None:
        """
        Sets the dataset to normal mode.
        When iterating over the dataset, 
        interdictions will not be included in the output.
        """
        self._return_intds = False

    def adverse_mode(self) -> None:
        """
        Sets the dataset to adverse mode.
        When iterating over the dataset, 
        interdictions will be included in the output.
        """
        self._return_intds = True

    def get_mode(self) -> str:
        """
        Returns the current mode of the dataset.

        Returns:
        --------
        str
            The current mode ('normal' or 'adverse').
        """
        return 'adverse' if self._return_intds else 'normal'
