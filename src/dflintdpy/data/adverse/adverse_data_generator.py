from dflintdpy.solvers.fast_solvers.fast_pricing_solver import FastBilevelPricingSolver
import pyepo
import numpy as np
import pandas as pd
from copy import deepcopy

from dflintdpy.data.config import HP
from dflintdpy.solvers.shortest_path_grb import ShortestPathGrb
from dflintdpy.solvers.symmetric_interdictor import SymmetricInterdictor
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
    _sym_interdictor: SymmetricInterdictor
    

    def __init__(self, 
                 cfg: HP,
                 opt_model: ShortestPathGrb,
                 budget: int, 
                 normalization_constant: float,
                 *,
                 num_scenarios: int = 10,
                 seed: int = 0,
                 adverse_problem: str = "SPNI",
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

        # Check correctness of adverse_problem
        if adverse_problem not in ["SPNI", "BPPO"]:
            raise ValueError(
                f"Unknown adverse problem type: {adverse_problem}"
            )
        else:
            self.adverse_problem = adverse_problem

        if self.adverse_problem == "SPNI":
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
            self._sym_interdictor = SymmetricInterdictor(
                self.opt_model._graph,
                k = budget,
                **kwargs
            )

        elif self.adverse_problem == "BPPO":
            self.num_scenarios = 2  # Original + one interdiction
            self._sym_interdictor = FastBilevelPricingSolver(
                opt_model.c, 
                opt_model.Sigma, 
                opt_model.gamma, 
                budget=opt_model.c.sum()*0.3
            )

    def _load_interdictions_from_cache(self, file_path, costs, feats):
        """
        Load cached interdiction data from CSV file.
        
        Returns:
            tuple: (feats, costs_grouped, interdictions_grouped) if successful, None otherwise
        """
        try:
            intd = pd.read_csv(file_path, header=None).values.astype(np.float32)
            
            n_samples = feats.shape[0]
            m = costs.shape[1]
            
            costs_grouped = np.zeros((n_samples, self.num_scenarios, m))
            interdictions_grouped = np.zeros_like(costs_grouped)
            
            # Fill scenario 0 with original costs
            costs_grouped[:, 0, :] = costs
            
            if self.adverse_problem == "BPPO":
                # BPPO: Single interdiction per sample
                costs_grouped[:, 1, :] = costs - intd
                interdictions_grouped[:, 1, :] = intd
            elif self.adverse_problem == "SPNI":
                # SPNI: Multiple interdictions per sample
                # Reshape loaded data: (n_samples * (num_scenarios-1), m) -> (n_samples, num_scenarios-1, m)
                intd_reshaped = intd.reshape(n_samples, self.num_scenarios - 1, m)
                for scenario_idx in range(self.num_scenarios - 1):
                    costs_grouped[:, scenario_idx + 1, :] = costs + intd_reshaped[:, scenario_idx, :]
                    interdictions_grouped[:, scenario_idx + 1, :] = intd_reshaped[:, scenario_idx, :]
            
            print("Loaded existing interdiction data from file.")
            return feats, costs_grouped, interdictions_grouped
        except Exception as e:
            print(f"Could not load cache (will generate new data): {e}")
            return None

    def _save_interdictions_to_cache(self, file_path, interdictions_grouped):
        """
        Save interdiction data to CSV file.
        """
        if self.adverse_problem == "BPPO":
            # BPPO: Save only scenario 1 (single interdiction per sample)
            np.savetxt(file_path, interdictions_grouped[:, 1, :], delimiter=',')
        elif self.adverse_problem == "SPNI":
            # SPNI: Save all scenarios (excluding scenario 0)
            # Reshape from (n_samples, num_scenarios-1, m) to (n_samples * (num_scenarios-1), m)
            n_samples = interdictions_grouped.shape[0]
            m = interdictions_grouped.shape[2]
            intd_flat = interdictions_grouped[:, 1:, :].reshape(-1, m)
            np.savetxt(file_path, intd_flat, delimiter=',')
        
        print(f"Saved interdiction data to {file_path}")

    def _generate_bppo_interdictions(self, feats, costs, versatile=False):
        """
        Generate interdictions for BPPO (Bilevel Pricing Problem).
        """
        n_samples = feats.shape[0]
        m = costs.shape[1]
        
        costs_grouped = np.zeros((n_samples, self.num_scenarios, m))
        interdictions_grouped = np.zeros_like(costs_grouped)
        
        # Fill scenario 0 with original costs
        costs_grouped[:, 0, :] = costs
        
        print(f"Generating BPPO adversarial examples for {n_samples} samples...")
        
        for idx in range(n_samples):
            cost = costs[idx].copy()
            cost[cost < 0] = 0  # Ensure non-negative costs
            
            # Solve the bilevel pricing problem
            solver = FastBilevelPricingSolver(
                cost, 
                self._sym_interdictor.Sigma, 
                self._sym_interdictor.gamma, 
                budget=self._sym_interdictor.budget
            )
            result_fast = solver.solve(n_starts=5, verbose=versatile)
            intd = result_fast['p_opt']
            
            # Store results
            new_cost = cost - intd
            costs_grouped[idx, 1, :] = new_cost
            interdictions_grouped[idx, 1, :] = intd
            
            print_progress(idx, n_samples)
        
        return feats, costs_grouped, interdictions_grouped

    def _generate_spni_interdictions(self, feats, costs, versatile=False):
        """
        Generate interdictions for SPNI (Stochastic Programming Network Interdiction).
        """
        n_samples = feats.shape[0]
        m = costs.shape[1]
        
        costs_grouped = np.zeros((n_samples, self.num_scenarios, m))
        interdictions_grouped = np.zeros_like(costs_grouped)
        
        # Fill scenario 0 with original costs
        costs_grouped[:, 0, :] = costs
        
        print(f"Generating SPNI adversarial examples for {n_samples} samples" + 
              f" with {self.num_scenarios} scenarios...")
        
        for idx in range(n_samples):
            cost = costs[idx]
            
            # Select interdictions for scenarios at random
            selected_interdictions = self._rng.choice(
                self.interdictions.shape[0], 
                size=self.num_scenarios - 1, 
                replace=False
            )
            
            # Iterate over each interdiction
            for scenario_idx, intd_idx in enumerate(selected_interdictions):
                intd = self.interdictions[intd_idx, :]
                
                # Solve the adversarial interdiction problem
                self._sym_interdictor.opt_model.setObj(cost)
                sym_intd, _, _ = self._sym_interdictor.benders_decomposition(
                    interdiction_cost=intd,
                    versatile=versatile,
                )
                
                # Create new cost vector by adding the interdiction costs
                new_cost = cost + sym_intd * intd
                
                # Store the new costs and applied interdiction
                costs_grouped[idx, scenario_idx + 1, :] = new_cost
                interdictions_grouped[idx, scenario_idx + 1, :] = sym_intd * intd
            
            print_progress(idx, n_samples)
        
        return feats, costs_grouped, interdictions_grouped

    def generate(self, feats, costs, file_path=None, versatile=False):
        """
        Main function to generate adversarial examples with caching support.
        
        Args:
            feats: Feature array
            costs: Cost array
            file_path: Optional path to cache file
            versatile: Verbose mode flag
            
        Returns:
            tuple: (feats, costs_grouped, interdictions_grouped)
        """
        # Try to load from cache if file path is provided
        if file_path is not None:
            cached_result = self._load_interdictions_from_cache(file_path, costs, feats)
            if cached_result is not None:
                return cached_result
        
        # Generate new interdictions based on problem type
        if self.adverse_problem == "BPPO":
            result = self._generate_bppo_interdictions(feats, costs, versatile)
        elif self.adverse_problem == "SPNI":
            result = self._generate_spni_interdictions(feats, costs, versatile)
        else:
            raise ValueError(f"Unknown adverse_problem: {self.adverse_problem}")
        
        feats, costs_grouped, interdictions_grouped = result
        
        # Save to cache if file path is provided
        if file_path is not None:
            self._save_interdictions_to_cache(file_path, interdictions_grouped)
        
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
