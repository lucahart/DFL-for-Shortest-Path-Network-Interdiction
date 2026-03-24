from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from types import SimpleNamespace
import warnings

import numpy as np
import pyepo

from dflintdpy.data.config import HP
from dflintdpy.solvers.fast_solvers.fast_pricing_solver import (
    FastBilevelPricingSolver,
)
from dflintdpy.solvers.shortest_path_grb import ShortestPathGrb
from dflintdpy.solvers.symmetric_interdictor import SymmetricInterdictor
from dflintdpy.utils.read_write import (
    Artefacts,
    CacheReplaceOptions,
    get_cache_replace_options,
    read_cache,
    write_adv_intd,
    write_rnd_intd,
)
from dflintdpy.utils.versatile_utils import print_progress


class SPNIInterdictionPolicy(ABC):
    """Strategy for converting one sampled interdiction cost into a scenario."""

    @abstractmethod
    def apply(
        self,
        *,
        cost: np.ndarray,
        interdiction_cost: np.ndarray,
        versatile: bool,
    ) -> np.ndarray:
        """Return the weighted interdiction applied to one scenario."""


class AdversarialSPNIInterdictionPolicy(SPNIInterdictionPolicy):
    """Run the symmetric interdictor to choose the interdicted edges."""

    def __init__(self, interdictor: SymmetricInterdictor):
        self._interdictor = interdictor

    def apply(
        self,
        *,
        cost: np.ndarray,
        interdiction_cost: np.ndarray,
        versatile: bool,
    ) -> np.ndarray:
        self._interdictor.opt_model.setObj(cost)
        sym_intd, _, _ = self._interdictor.benders_decomposition(
            interdiction_cost=interdiction_cost,
            versatile=versatile,
        )
        return sym_intd * interdiction_cost


class RandomSPNIInterdictionPolicy(SPNIInterdictionPolicy):
    """Sample a budget-feasible cardinality-constrained interdiction pattern."""

    def __init__(self, rng: np.random.Generator, budget: int):
        self._rng = rng
        self._budget = int(budget)

    def apply(
        self,
        *,
        cost: np.ndarray,
        interdiction_cost: np.ndarray,
        versatile: bool,
    ) -> np.ndarray:
        del cost
        del versatile

        m = interdiction_cost.shape[0]
        sym_intd = np.zeros(m, dtype=np.float32)
        k = min(self._budget, m)
        if k > 0:
            chosen = self._rng.choice(m, size=k, replace=False)
            sym_intd[chosen] = 1.0
        return sym_intd * interdiction_cost


class BaseAdverseDataGenerator(ABC):
    """Common orchestration for adverse-data generation."""

    adverse_problem: str = "BASE"
    opt_model: ShortestPathGrb
    num_scenarios: int

    def __init__(
        self,
        cfg: HP,
        opt_model: ShortestPathGrb,
        budget: int,
        normalization_constant: float,
        *,
        num_scenarios: int = 2,
        seed: int = 0,
        interdiction_policy: str = "adversarial",
        cache_options: CacheReplaceOptions | None = None,
    ):
        del cfg
        del normalization_constant

        if num_scenarios is None:
            num_scenarios = 10

        if interdiction_policy not in ["adversarial", "random"]:
            raise ValueError(
                f"Unknown interdiction policy: {interdiction_policy}"
            )

        self.opt_model = deepcopy(opt_model)
        self.budget = budget
        self.num_scenarios = int(num_scenarios)
        self.interdiction_policy = interdiction_policy
        self._base_seed = int(seed)
        self._rng = np.random.default_rng(self._base_seed)
        self._cache_options = cache_options or get_cache_replace_options()

    def _empty_grouped_arrays(
        self,
        costs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Allocate grouped cost/interdiction outputs with scenario 0 filled."""
        n_samples = costs.shape[0]
        m = costs.shape[1]
        costs_grouped = np.zeros(
            (n_samples, self.num_scenarios, m),
            dtype=float,
        )
        interdictions_grouped = np.zeros_like(costs_grouped)
        costs_grouped[:, 0, :] = costs
        return costs_grouped, interdictions_grouped

    def _load_interdictions_from_cache(
        self,
        cfg,
        costs: np.ndarray,
        feats: np.ndarray,
    ):
        """Return cached grouped data if this generator supports caching."""
        del cfg
        del costs
        del feats
        return None

    def _save_interdictions_to_cache(
        self,
        cfg,
        interdictions_grouped: np.ndarray,
    ) -> None:
        """Persist grouped interdictions if this generator supports caching."""
        del cfg
        del interdictions_grouped

    @abstractmethod
    def _generate(
        self,
        feats: np.ndarray,
        costs: np.ndarray,
        versatile: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate grouped adverse scenarios."""

    def generate(self, feats, costs, cfg=None, versatile=False):
        """
        Generate adverse examples and optionally read/write cached data.
        """
        if cfg is not None:
            cached_result = self._load_interdictions_from_cache(
                cfg,
                costs,
                feats,
            )
            if cached_result is not None:
                return cached_result

        feats, costs_grouped, interdictions_grouped = self._generate(
            feats,
            costs,
            versatile=versatile,
        )

        if cfg is not None:
            self._save_interdictions_to_cache(cfg, interdictions_grouped)

        return feats, costs_grouped, interdictions_grouped

    @staticmethod
    def gen_interdictions(
        cfg: HP,
        normalization_constant,
        num_cost,
        *,
        gen_intd_seed: int = 157,
        n_interdictions: int = 100,
    ) -> np.ndarray:
        """
        Generate normalized interdiction costs for SPNI scenario sampling.
        """
        _, costs = pyepo.data.shortestpath.genData(
            n_interdictions,
            cfg.get("num_features"),
            (num_cost + 1, 1),
            deg=cfg.get("deg"),
            noise_width=cfg.get("noise_width"),
            seed=gen_intd_seed,
        )
        return costs / normalization_constant


class SPNIAdverseDataGenerator(BaseAdverseDataGenerator):
    """Adverse-data generator for the shortest-path network interdiction."""

    adverse_problem = "SPNI"

    def __init__(
        self,
        cfg: HP,
        opt_model: ShortestPathGrb,
        budget: int,
        normalization_constant: float,
        *,
        num_scenarios: int = 2,
        seed: int = 0,
        interdiction_policy: str = "adversarial",
        cache_options: CacheReplaceOptions | None = None,
        n_training_interdictions: int = 100,
        gen_intd_seed: int = 157,
        max_cnt: int = 3,
        eps: float | None = None,
        **interdictor_kwargs,
    ):
        super().__init__(
            cfg,
            opt_model,
            budget,
            normalization_constant,
            num_scenarios=num_scenarios,
            seed=seed,
            interdiction_policy=interdiction_policy,
            cache_options=cache_options,
        )

        self.n_training_intds = int(n_training_interdictions)
        if self.n_training_intds <= self.num_scenarios - 1:
            self.num_scenarios = self.n_training_intds + 1
            warnings.warn(
                (
                    "Number of interdictions "
                    f"({self.n_training_intds}) is less than the requested "
                    f"number of interdicted scenarios "
                    f"({num_scenarios - 1}). "
                    f"Setting num_scenarios to {self.num_scenarios}."
                ),
                stacklevel=2,
            )

        benders_eps = cfg.get("benders_eps") if eps is None else eps
        self.interdictions = type(self).gen_interdictions(
            cfg,
            normalization_constant,
            self.opt_model.num_cost,
            gen_intd_seed=gen_intd_seed,
            n_interdictions=self.n_training_intds,
        )
        self._spni_interdictor = SymmetricInterdictor(
            self.opt_model._graph,
            k=budget,
            max_cnt=max_cnt,
            eps=benders_eps,
            **interdictor_kwargs,
        )
        self._sym_interdictor = self._spni_interdictor
        if self.interdiction_policy == "adversarial":
            self._policy = AdversarialSPNIInterdictionPolicy(
                self._spni_interdictor
            )
        else:
            self._policy = RandomSPNIInterdictionPolicy(
                self._rng,
                getattr(self._spni_interdictor, "k", budget),
            )

    def _target_artifact(self) -> Artefacts:
        """Return the cache artifact keyed by the SPNI interdiction policy."""
        if self.interdiction_policy == "adversarial":
            return Artefacts.INTD_ADV
        return Artefacts.INTD_RND

    def _load_interdictions_from_cache(self, cfg, costs, feats):
        """Load cached SPNI interdiction tensors and regroup by scenario."""
        target_artifact = self._target_artifact()
        if self._cache_options.for_artifact(target_artifact):
            return None

        intd = read_cache(cfg, target_artifact)
        if intd is None:
            print("No cached interdiction data found. Generating new data.")
            return None

        try:
            n_samples = feats.shape[0]
            m = costs.shape[1]
            costs_grouped, interdictions_grouped = self._empty_grouped_arrays(
                costs
            )
            intd_reshaped = intd.reshape(n_samples, self.num_scenarios - 1, m)
            for scenario_idx in range(self.num_scenarios - 1):
                costs_grouped[:, scenario_idx + 1, :] = (
                    costs + intd_reshaped[:, scenario_idx, :]
                )
                interdictions_grouped[:, scenario_idx + 1, :] = (
                    intd_reshaped[:, scenario_idx, :]
                )

            print("Loaded existing interdiction data from file.")
            return feats, costs_grouped, interdictions_grouped
        except Exception:
            print("No cached interdiction data found. Generating new data.")
            return None

    def _save_interdictions_to_cache(
        self,
        cfg,
        interdictions_grouped: np.ndarray,
    ) -> None:
        """Flatten and save SPNI interdictions through the cache helpers."""
        m = interdictions_grouped.shape[2]
        intd_flat = interdictions_grouped[:, 1:, :].reshape(-1, m)
        if self.interdiction_policy == "adversarial":
            write_adv_intd(
                cfg,
                intd_flat,
                replace=self._cache_options.for_artifact(Artefacts.INTD_ADV),
                archive_replaced=self._cache_options.archive_replaced,
            )
        elif self.interdiction_policy == "random":
            write_rnd_intd(
                cfg,
                intd_flat,
                replace=self._cache_options.for_artifact(Artefacts.INTD_RND),
                archive_replaced=self._cache_options.archive_replaced,
            )
        else:
            raise ValueError(
                f"Unknown interdiction policy: {self.interdiction_policy}"
            )

        print("Saved interdiction data to file.")

    def _generate_spni_interdictions(
        self,
        feats,
        costs,
        versatile=False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate grouped SPNI scenarios."""
        n_samples = feats.shape[0]
        costs_grouped, interdictions_grouped = self._empty_grouped_arrays(costs)

        print(
            f"Generating SPNI examples for {n_samples} samples"
            f" with {self.num_scenarios} scenarios"
            f" using {self.interdiction_policy} interdictions..."
        )

        for idx in range(n_samples):
            cost = costs[idx]
            selected_interdictions = self._rng.choice(
                self.interdictions.shape[0],
                size=self.num_scenarios - 1,
                replace=False,
            )

            for scenario_idx, intd_idx in enumerate(selected_interdictions):
                intd = self.interdictions[intd_idx, :]
                applied_intd = self._policy.apply(
                    cost=cost,
                    interdiction_cost=intd,
                    versatile=versatile,
                )
                costs_grouped[idx, scenario_idx + 1, :] = cost + applied_intd
                interdictions_grouped[idx, scenario_idx + 1, :] = applied_intd

            print_progress(idx, n_samples)

        return feats, costs_grouped, interdictions_grouped

    def _generate(
        self,
        feats: np.ndarray,
        costs: np.ndarray,
        versatile: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._generate_spni_interdictions(
            feats,
            costs,
            versatile=versatile,
        )


class BPPOAdverseDataGenerator(BaseAdverseDataGenerator):
    """Adverse-data generator for the bilevel pricing problem."""

    adverse_problem = "BPPO"

    def __init__(
        self,
        cfg: HP,
        opt_model,
        budget: int,
        normalization_constant: float,
        *,
        num_scenarios: int = 2,
        seed: int = 0,
        interdiction_policy: str = "adversarial",
        cache_options: CacheReplaceOptions | None = None,
        **kwargs,
    ):
        del cfg
        del normalization_constant
        del num_scenarios
        del kwargs

        super().__init__(
            HP(),
            opt_model,
            budget,
            1.0,
            num_scenarios=2,
            seed=seed,
            interdiction_policy=interdiction_policy,
            cache_options=cache_options,
        )
        self.num_scenarios = 2
        self._pricing_problem = SimpleNamespace(
            Sigma=np.array(self.opt_model.Sigma, dtype=float),
            gamma=float(self.opt_model.gamma),
            budget=float(budget),
        )
        self._sym_interdictor = self._pricing_problem

    def _load_interdictions_from_cache(self, cfg, costs, feats):
        """BPPO caching is intentionally disabled to avoid cache collisions."""
        del cfg
        del costs
        del feats
        return None

    def _save_interdictions_to_cache(
        self,
        cfg,
        interdictions_grouped: np.ndarray,
    ) -> None:
        """BPPO caching is intentionally disabled to avoid cache collisions."""
        del cfg
        del interdictions_grouped

    def _generate_bppo_interdictions(
        self,
        feats,
        costs,
        versatile=False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate one adversarial BPPO scenario per sample."""
        n_samples = feats.shape[0]
        costs_grouped, interdictions_grouped = self._empty_grouped_arrays(costs)

        print(f"Generating BPPO adversarial examples for {n_samples} samples...")

        for idx in range(n_samples):
            original_cost = costs[idx].copy()
            solver_cost = original_cost.copy()
            solver_cost[solver_cost < 0] = 0

            solver = FastBilevelPricingSolver(
                solver_cost,
                self._pricing_problem.Sigma,
                self._pricing_problem.gamma,
                budget=self._pricing_problem.budget,
            )
            result_fast = solver.solve(n_starts=5, verbose=versatile)
            intd = result_fast["p_opt"]

            costs_grouped[idx, 1, :] = solver_cost - intd
            interdictions_grouped[idx, 1, :] = intd

            print_progress(idx, n_samples)

        return feats, costs_grouped, interdictions_grouped

    def _generate(
        self,
        feats: np.ndarray,
        costs: np.ndarray,
        versatile: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._generate_bppo_interdictions(
            feats,
            costs,
            versatile=versatile,
        )


class AdvDataGenerator:
    """Compatibility wrapper around the concrete adverse-data generators."""

    _GENERATOR_TYPES = {
        "SPNI": SPNIAdverseDataGenerator,
        "BPPO": BPPOAdverseDataGenerator,
    }
    gen_interdictions = staticmethod(BaseAdverseDataGenerator.gen_interdictions)

    @classmethod
    def resolve_generator_cls(cls, adverse_problem: str):
        """Return the concrete generator type for the given problem family."""
        try:
            return cls._GENERATOR_TYPES[adverse_problem]
        except KeyError as exc:
            raise ValueError(
                f"Unknown adverse problem type: {adverse_problem}"
            ) from exc

    def __init__(
        self,
        cfg: HP,
        opt_model,
        budget: int,
        normalization_constant: float,
        *,
        num_scenarios: int = 2,
        seed: int = 0,
        adverse_problem: str = "SPNI",
        interdiction_policy: str = "adversarial",
        cache_options: CacheReplaceOptions | None = None,
        **kwargs,
    ):
        generator_cls = self.resolve_generator_cls(adverse_problem)
        self._delegate = generator_cls(
            cfg,
            opt_model,
            budget,
            normalization_constant,
            num_scenarios=num_scenarios,
            seed=seed,
            interdiction_policy=interdiction_policy,
            cache_options=cache_options,
            **kwargs,
        )

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def generate(self, feats, costs, cfg=None, versatile=False):
        return self._delegate.generate(
            feats,
            costs,
            cfg=cfg,
            versatile=versatile,
        )

