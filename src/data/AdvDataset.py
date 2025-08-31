from models.ShortestPathGrb import shortestPathGrb
import numpy as np

from pyepo.data.dataset import optDataset
import torch

class AdvDataset(optDataset):
    """
    Dataset for adverse network training under interdictions.

    Attributes:
    -----------
    intds : np.ndarray
        Interdictions for each instance.
    """

    intds : np.ndarray
    _return_intds: bool # whether to return interdictions

    def __init__(self, 
                 opt_model: shortestPathGrb, 
                 feats: np.ndarray, 
                 costs: np.ndarray, 
                 intds: np.ndarray,
                 mode: str = 'normal'
                 ):
        """
        Initializes the adversarial dataset.

        Parameters:
        -----------
        opt_model : shortestPathGrb
            The optimization model.
        feats : np.ndarray
            Dataset features.
        costs : np.ndarray
            Dataset costs.
        intds : np.ndarray
            Interdictions at instance.
        mode : str, optional
            Mode of the dataset ('normal' or 'adverse'). Defaults to 'normal'.

        Raises:
        -------
        ValueError
            If the mode is not 'normal' or 'adverse'.
        """

        # Call the parent constructor to create optDataset
        super().__init__(opt_model, feats, costs)

        # Store the interdictions
        self.intds = intds

        # Set the interdiction mode
        if mode == 'normal':
            self._return_intds = False
        elif mode == 'adverse':
            self._return_intds = True
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __getitem__(self, index):
        """
        A method to retrieve data

        Args:
            index (int): data index

        Returns:
            tuple: data features (torch.tensor), costs (torch.tensor), 
            optimal solutions (torch.tensor) and objective values (torch.tensor)
        """
        return (
            torch.FloatTensor(self.feats[index]),
            torch.FloatTensor(self.costs[index]),
            torch.FloatTensor(self.sols[index]),
            torch.FloatTensor(self.objs[index]),
            torch.FloatTensor(self.intds[index])
        ) if self._return_intds else (
            torch.FloatTensor(self.feats[index]),
            torch.FloatTensor(self.costs[index]),
            torch.FloatTensor(self.sols[index]),
            torch.FloatTensor(self.objs[index])
        )

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
