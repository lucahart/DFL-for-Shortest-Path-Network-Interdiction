import torch
import pyepo
import numpy as np
from copy import deepcopy

from data.config import HP
from models.ShortestPathGrb import shortestPathGrb
from solvers.BendersDecomposition import BendersDecomposition
from utils.versatile_utils import print_progress

class AdverseDataGenerator:
    """
    Class to augment existing datasets for SPO learning 
    with adversarial examples.
    """

    opt_model: shortestPathGrb
    interdictions: np.ndarray
    sym_interdictor: BendersDecomposition

    def __init__(self, 
                 cfg: HP,
                 opt_model: shortestPathGrb,
                 budget: int, 
                 normalization_constant: float,
                 n_additional_samples: int = 10,
                 **kwargs):
        """
        Initialize the AdverseDataGenerator.

        Parameters:
        -----------
        opt_model : ShortestPathGrb
            A shortest path optimization model 
            with correct network structure.
        budget : int
            The budget for the interdictor.
        **kwargs : dict
            Additional keyword arguments including:
            # For Bender's Decomposition:
            - max_cnt : int
                Maximum number of iterations for Bender's algorithm.
            - eps : float
                Epsilon for convergence criterion.
            # For AdverseDataGenerator:
            - intd_seed : int
                Seed for random number generation.
            - gen_additional_data_samples : int
                Number of additional data samples to generate.
        """

        # Copy optimization model instance
        self.opt_model = deepcopy(opt_model)
        self.n_additional_samples = n_additional_samples

        # Generate interdictions
        self.interdictions = AdverseDataGenerator.gen_interdictions(
            normalization_constant,
            cfg,
            **kwargs
        )

        # Create Benders decomposition instances for each interdiction
        self.sym_interdictor = BendersDecomposition(
            self.opt_model,
            k = budget,
            **kwargs
        )


    def generate(self, 
                 feats: np.ndarray,
                 costs: np.ndarray, 
                 versatile: bool = False,
                 seed: int = None
                 ) -> None:
        """
        Generate adversarial examples for the given dataset. 
        Adds new examples in place.

        Parameters:
        -----------
        dataset : pyepo.data.Dataset
            The dataset for which to generate adversarial examples.

        Returns:
        --------
        pyepo.data.Dataset
            A new dataset containing the adversarial examples.
        """

        # Print that generation started
        print(f"Generating adversarial examples with " + 
              f"{self.interdictions.shape[0]} interdictions...")

        # Set random seed if specified
        if seed is not None:
            np.random.seed(seed)

        # Create an array to hold the features for the adversarial examples
        n_samples = feats.shape[0]
        n_samples_new = n_samples * self.interdictions.shape[0]
        feats_result = np.concatenate([
                feats, 
                np.zeros((n_samples_new, feats.shape[1]))
            ])
        costs_result = np.concatenate([
                costs, 
                np.zeros((n_samples_new, costs.shape[1]))
            ])
        intds_result = np.zeros((n_samples_new + n_samples, costs.shape[1]))

        # Iterate over each example in the dataset
        for idx in range(n_samples):
            # Unpack features and costs for each sample
            feat = feats[idx]
            cost = costs[idx]

            # Select random interdictions
            selected_intds = np.random.choice(self.interdictions.shape[0], 
                                              self.n_additional_samples, 
                                              replace=False)

            # Iterate over each interdiction
            for idx_intd, intd in enumerate(self.interdictions[selected_intds]):
                # Solve the adversarial interdiction problem
                self.sym_interdictor.opt_model.setObj(cost)
                sym_intd, _, _ = self.sym_interdictor.benders_decomposition(
                    interdiction_cost=intd,
                    versatile=versatile
                )

                # Create new cost vector by adding the interdiction costs
                new_cost = cost + sym_intd * intd

                # Store the new costs
                store_idx = idx + n_samples * (idx_intd + 1)
                feats_result[store_idx,:] = feat
                costs_result[store_idx,:] = new_cost
                intds_result[store_idx,:] = sym_intd * intd

            # Print progress if versatile
            print_progress(idx, n_samples)

        return feats_result, costs_result, intds_result


    @staticmethod
    def gen_interdictions(normalization_constant,
                  cfg: HP,
                  *,
                  intd_seed: int = 157,
                  n_interdictions: int = 100) -> np.ndarray:
        """
        Generate adversarial interdictions for data generation. 
        If no configuration is provided, defaults will be used.

        Parameters:
        -----------
        cfg : HP
            The configuration object containing hyperparameters:
            - num_features : int
                The number of features in the dataset.
            - grid_size : int
                The size of the grid for the simulation.
            - deg : float
                The degree of the graph.
            - noise_width : float
                The width of the noise to add to the costs.
        normalization_constant : float
            The constant used to normalize the costs.
        intd_seed : int, Optional
            The seed for random number generation.
        n_interdictions : int, Optional
            The number of interdictions to generate.
            Defaults to 100.

        Returns:
        --------
        np.ndarray
            A numpy array containing the generated adversarial interdictions.
        """

        # Generate true network data for simulation
        _, costs = pyepo.data.shortestpath.genData(
            n_interdictions,
            cfg.get("num_features"),
            cfg.get("grid_size"),
            deg=cfg.get("deg"),
            noise_width=cfg.get("noise_width"),
            seed=intd_seed
        )

        # Normalize costs
        costs = costs / normalization_constant

        return costs
