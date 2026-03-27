"""Dataset assembly for SPNI simulations.

This module should own all orchestration around:
- base synthetic data generation
- train/val/test splitting
- adverse/random data creation
- baseline non-adverse loader creation

It must not train predictors or compute evaluation metrics.
"""

from __future__ import annotations

from dflintdpy.simulation.spni.config import SPNIRunConfig
from dflintdpy.simulation.spni.types import DatasetBundle, GraphBundle


def generate_base_data(run_cfg: SPNIRunConfig, graph_bundle: GraphBundle):
    """Generate or load the base feature/cost arrays for one run.

    Future implementation responsibilities:
    - call the existing synthetic-data helper
    - keep base data generation separate from splitting
    - return raw arrays in a small internal bundle or tuple
    """

    raise NotImplementedError("Specification scaffold only.")


def split_base_data(run_cfg: SPNIRunConfig, features, costs):
    """Split base arrays into train/validation/test partitions.

    Future implementation responsibilities:
    - make split counts explicit
    - define and preserve sample ordering conventions
    - return a structured split result
    """

    raise NotImplementedError("Specification scaffold only.")


def build_spni_training_data(
    run_cfg: SPNIRunConfig,
    graph_bundle: GraphBundle,
    split_data,
):
    """Create adverse and random SPNI training loaders.

    Future implementation responsibilities:
    - build adversarial and random scenario generators
    - build training and validation loaders for both variants
    - preserve the generator objects for debugging and cache inspection
    """

    raise NotImplementedError("Specification scaffold only.")


def build_nonadverse_views(adverse_train_loader, adverse_val_loader):
    """Create baseline non-adverse loader views from adverse loaders.

    Future implementation responsibilities:
    - derive baseline loaders from scenario-zero data
    - preserve batch size and sampler configuration
    """

    raise NotImplementedError("Specification scaffold only.")


def assemble_dataset_bundle(
    run_cfg: SPNIRunConfig,
    graph_bundle: GraphBundle,
) -> DatasetBundle:
    """Build the full typed dataset bundle for one SPNI run.

    Future implementation responsibilities:
    - normalize costs once and store the normalization constant
    - create all loader variants required by the training stage
    - create evaluation interdiction arrays for the comparison stage
    """

    raise NotImplementedError("Specification scaffold only.")

