import pyepo
import numpy as np
from copy import deepcopy

from dflintdpy.data.config import HP
from dflintdpy.solvers.shortest_path_grb import ShortestPathGrb
from dflintdpy.solvers.BendersDecomposition import BendersDecomposition
from dflintdpy.utils.versatile_utils import print_progress

class AdvDataGenerator:
    """
    Class to augment existing datasets for SPO learning 
    with adversarial examples.

    Attributes:
    -----------
    opt_model : ShortestPathGrb
        A shortest path optimization model with correct network structure.
    select_intds : int
        Number of interdictions to select for each sample.
    interdictions : np.ndarray
        Array of generated interdictions.
    """

    opt_model: ShortestPathGrb
    num_scenarios: int
    interdictions: np.ndarray
    _sym_interdictor: BendersDecomposition

    def __init__(self, 
                 cfg: HP,
                 opt_model: ShortestPathGrb,
                 budget: int, 
                 normalization_constant: float,
                 *,
                 num_scenarios: int = 10,
                 seed: int = 0,
                 **kwargs):
        """
        Initialize the AdverseDataGenerator.

        Parameters:
        -----------
        cfg : HP
            Hyperparameter configuration.
        opt_model : ShortestPathGrb
            A shortest path optimization model 
            with correct network structure.
        budget : int
            The budget for the interdictor.
        normalization_constant : float
            Normalization constant for the costs.
        num_scenarios : int, optional
            Number of interdictions to select for each sample. Defaults to 10.
        seed : int, optional
            Seed for random number generation. Defaults to 0.
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
            - n_interdictions : int
                Number of interdictions to generate and choose from.
        """

        if num_scenarios is None:
            num_scenarios = 10

        # Copy optimization model instance
        self.opt_model = deepcopy(opt_model)
        self._base_seed = int(seed)
        self._rng = np.random.default_rng(self._base_seed)

        # Check correctness of num_scenarios
        n_intds = kwargs.get("n_interdictions")
        if n_intds is not None and n_intds <= num_scenarios - 1:
            # If there are more scenarios than interdictions, 
            # reduce num_scenarios to n_intds + 1
            Warning(f"Warning: Number of interdictions ({n_intds}) is less" +
                    f" than the number of scenarios ({num_scenarios - 1})." +
                    f" Setting num_scenarios to {n_intds + 1}.")
            self.num_scenarios = n_intds + 1
        elif n_intds is None and num_scenarios > 100 + 1:
             # If n_interdictions is not specified it defaults to 100,
             # so that the num_scenarios <= 101.
             Warning(f"Warning: Number of scenarios ({num_scenarios - 1})" + 
                   f" is greater than the default number of" + 
                   f" interdictions (100). Setting num_scenarios to 101.")
             self.num_scenarios = 101
        else:
            self.num_scenarios = num_scenarios

        # Generate interdictions
        self.interdictions = AdvDataGenerator.gen_interdictions(
            cfg,
            normalization_constant,
            **kwargs
        )

        # Create Benders decomposition instances for each interdiction
        self._sym_interdictor = BendersDecomposition(
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
            f"{self.num_scenarios} scenarios..."
        )

        # Determine sizes
        n_samples = feats.shape[0] # number of original samples
        m = costs.shape[1] # length of cost vector

        # Allocate grouped arrays. Scenario 0 corresponds to the
        # original (unin­terdicted) cost. Remaining scenarios store the
        # result for each interdiction.
        costs_grouped = np.zeros((n_samples, self.num_scenarios, m))
        interdictions_grouped = np.zeros_like(costs_grouped)

        # Fill scenario 0 with the original costs
        costs_grouped[:, 0, :] = costs

        # Iterate over each example in the dataset
        for idx in range(n_samples):
            # Unpack costs for each sample
            cost = costs[idx]

            # Select interdictions for scenarios at random
            selected_interdictions = self._rng.choice(
                self.interdictions.shape[0], 
                size=self.num_scenarios - 1, 
                replace=False)

            # Iterate over each interdiction
            for idx_intd, intd in enumerate(self.interdictions[selected_interdictions, :]):
                # Solve the adversarial interdiction problem
                self._sym_interdictor.opt_model.setObj(cost)
                sym_intd, _, _ = self._sym_interdictor.benders_decomposition(
                    interdiction_cost=intd,
                    versatile=versatile,
                )

                # Create new cost vector by adding the interdiction costs
                new_cost = cost + sym_intd * intd

                # Store the new costs and applied interdiction
                costs_grouped[idx, idx_intd + 1, :] = new_cost
                interdictions_grouped[idx, idx_intd + 1, :] = (
                    sym_intd * intd
                )

            # Print progress if versatile
            print_progress(idx, n_samples)

        return feats, costs_grouped, interdictions_grouped


    @staticmethod
    def gen_interdictions(
        cfg: HP,
        normalization_constant,
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
