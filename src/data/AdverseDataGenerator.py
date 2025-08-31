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


    def generate(
        self,
        feats: np.ndarray,
        costs: np.ndarray,
        versatile: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate adversarial examples for the given dataset.

        Parameters:
        -----------
        feats : np.ndarray
            Feature matrix of shape ``(n_samples, p)``.
        costs : np.ndarray
            Cost matrix of shape ``(n_samples, m)``.
        versatile : bool, optional
            Whether to print a progress bar during generation.

        Returns:
        --------
        tuple
            ``(feats, costs_grouped, interdictions_grouped)`` where ``feats``
            has shape ``(n_samples, p)`` and ``costs_grouped`` as well as
            ``interdictions_grouped`` have shape ``(n_samples, num_scenarios, m)``.
            Scenario index ``0`` corresponds to the original (unin­terdicted)
            costs; subsequent indices contain the results for each
            interdiction.
        """

        # Print that generation started
        print(
            f"Generating adversarial examples with "
            f"{self.interdictions.shape[0]} interdictions..."
        )

        # Determine sizes
        n_samples = feats.shape[0]
        m = costs.shape[1]
        num_scenarios = self.interdictions.shape[0] + 1

        # Allocate grouped arrays. Scenario 0 corresponds to the
        # original (unin­terdicted) cost. Remaining scenarios store the
        # result for each interdiction.
        costs_grouped = np.zeros((n_samples, num_scenarios, m))
        interdictions_grouped = np.zeros_like(costs_grouped)

        # Fill scenario 0 with the original costs
        costs_grouped[:, 0, :] = costs

        # Iterate over each example in the dataset
        for idx in range(n_samples):
            # Unpack costs for each sample
            cost = costs[idx]

            # Iterate over each interdiction
            for idx_intd, interdiction in enumerate(self.interdictions):
                # Solve the adversarial interdiction problem
                self.sym_interdictor.opt_model.setObj(cost)
                sym_intd, _, _ = self.sym_interdictor.benders_decomposition(
                    interdiction_cost=interdiction,
                    versatile=versatile,
                )

                # Create new cost vector by adding the interdiction costs
                new_cost = cost + sym_intd * interdiction

                # Store the new costs and applied interdiction
                costs_grouped[idx, idx_intd + 1, :] = new_cost
                interdictions_grouped[idx, idx_intd + 1, :] = (
                    sym_intd * interdiction
                )

            # Print progress if versatile
            print_progress(idx, n_samples)

        return feats, costs_grouped, interdictions_grouped


    @staticmethod
    def gen_interdictions(normalization_constant,
                  cfg: HP,
                  *,
                  intd_seed: int = 157,
                  gen_additional_data_samples: int = 10) -> np.ndarray:
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
        gen_additional_data_samples : int, Optional
            The number of samples to generate for the simulation. 
            Defaults to 10.
    
        Returns:
        --------
        np.ndarray
            A numpy array containing the generated adversarial interdictions.
        """

        # Generate true network data for simulation
        _, costs = pyepo.data.shortestpath.genData(
            gen_additional_data_samples,
            cfg.get("num_features"),
            cfg.get("grid_size"),
            deg=cfg.get("deg"),
            noise_width=cfg.get("noise_width"),
            seed=intd_seed
        )

        # Normalize costs
        costs = costs / normalization_constant

        return costs
