import random
import pyepo
import torch
import numpy as np
from torch import nn
from copy import deepcopy

from dflintdpy.data.config import HP
from dflintdpy.data.data_gen import gen_syn_data
from dflintdpy.models.grid import Grid
from dflintdpy.solvers.shortest_path_grb import ShortestPathGrb
from dflintdpy.predictors.hybrid_spop_loss import HybridSPOPLoss
from dflintdpy.utils.pfl_trainer import PFLTrainer
from dflintdpy.utils.dfl_trainer import DFLTrainer
from dflintdpy.utils.read_write import (
    Artefacts,
    CacheReplaceOptions,
    get_cache_replace_options,
    write_pred,
)
from dflintdpy.simulation.spni.config import CachePolicy, build_run_config
from dflintdpy.simulation.spni.data import (
    build_spni_training_view,
    generate_base_data,
    split_base_data,
)
from dflintdpy.simulation.spni.types import GraphBundle


def _cache_policy_from_options(
        cache_options: CacheReplaceOptions | None,
) -> CachePolicy:
    """Translate legacy cache-replace options into the SPNI cache policy."""
    options = cache_options or get_cache_replace_options()
    return CachePolicy(
        replace_data=options.replace_data,
        replace_intd_adv=options.replace_intd_adv,
        replace_intd_rnd=options.replace_intd_rnd,
        replace_pred=options.replace_pred,
        replace_result=options.replace_result,
        replace_fig=options.replace_fig,
        archive_replaced=options.archive_replaced,
    )

def set_seed(cfg: HP) -> None:    # Set the random seed for reproducibility
    np.random.seed(cfg.get("random_seed"))
    random.seed(cfg.get("random_seed"))
    torch.manual_seed(cfg.get("random_seed"))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.get("random_seed"))

def gen_train_data(
        cfg: HP,
        opt_model: 'ShortestPathGrb',
        # path_dir: str = None,
        interdiction_policy: str = "adversarial",
        cache_options: CacheReplaceOptions | None = None,
) -> dict:
    """
    Compatibility wrapper around the SPNI dataset-assembly stage.

    This preserves the legacy return structure while delegating the dataset
    orchestration into `simulation.spni.data`.
    """
    run_cfg = build_run_config(
        cfg,
        cache_policy=_cache_policy_from_options(cache_options),
    )
    graph_bundle = GraphBundle(
        graph=None,
        opt_model=opt_model,
        graph_kind="compatibility",
        graph_source="scripts.setup.gen_train_data",
    )
    base_data = generate_base_data(run_cfg, graph_bundle)
    split_data = split_base_data(run_cfg, base_data.features, base_data.costs)
    training_view = build_spni_training_view(
        run_cfg,
        graph_bundle,
        split_data,
        interdiction_policy=interdiction_policy,
    )

    return {
        "train_loader": training_view.train_loader,
        "val_loader": training_view.val_loader
    }, {
        "feats": split_data.test_features,
        "costs": split_data.test_costs
    }, split_data.normalization_constant,{
        "data_generator": training_view.data_generator
    }


def gen_data(cfg: HP,
            normalization_constant,
            opt_model: 'ShortestPathGrb' = None,
            seed: int = 31) -> dict:

    # Generate true network data for simulation
    features, costs = gen_syn_data(cfg, opt_model=opt_model, seed=seed)

    # Normalize costs
    costs = costs / normalization_constant

    return {
        "features": features,
        "costs": costs
    }


def get_nn(input_size, output_size):

    hidden_size_1 =  64   # number of neurons in the hidden layer
    return nn.Sequential(
        nn.Linear(input_size, hidden_size_1),  # first affine layer
        nn.ReLU(),                           # non‐linearity
        nn.Linear(hidden_size_1, output_size),  # third affine layer
        nn.Sigmoid()                         # output activation function
    )


def setup_pfl_predictor(
        cfg: HP,
        graph: Grid,
        opt_model: 'ShortestPathGrb',
        training_data: dict,
        *,
        cache_tag: str = "pfl",
        verbose: bool = False,
        file_name: str = None,
        train_type: str = "po",
        cache_options: CacheReplaceOptions | None = None,
        **kwargs
        ):
    
    # Specify a file name instead of verbose plotting without saving
    if verbose and file_name is None:
        print(
            "Warning: You have enabled verbose mode " \
             + "without specifying a file name. Plots will not be saved."
        )
    
    # Define your network dimensions
    input_size  =  cfg.get("num_features")   # e.g. number of features in your cost‐vector
    output_size =  graph.num_cost   # e.g. # of target outputs, or number of classes

    # This setup can be used for training a PO model or pre-training an SPO model
    if train_type == "po":
        lr = cfg.get("po_lr")
        epochs = cfg.get("po_epochs")
    elif train_type == "spo":
        lr = cfg.get("spo_po_lr")
        epochs = cfg.get("spo_po_epochs")
    else:
        raise(f"Training type {train_type} is not defined.")


    # Build the model with nn.Sequential
    po_model = get_nn(input_size, output_size) \
        if cfg.get("pred_model") is None or cfg.get("pred_model") == "nn" \
        else nn.Linear(input_size, output_size)

    cache_options = cache_options or get_cache_replace_options()
    replace_pred = cache_options.for_artifact(Artefacts.PRED)

    # Check if model has been cached already unless replacement is forced.
    state_dict = None if replace_pred else read_cache(cfg, Artefacts.PRED, artifact_tag=cache_tag)
    if state_dict is not None:
        po_model.load_state_dict(state_dict)
        print(f"Loaded existing predictor model '{cache_tag}' from file.")
        return po_model

    # Define the loss function and optimizer
    po_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(po_model.parameters(), lr=lr)

    po_trainer = PFLTrainer(
        pred_model=po_model,
        opt_model=opt_model,
        optimizer=optimizer,
        loss_fn=po_criterion
    )

    # Train the model
    train_loss_log, train_regret_log, val_loss_log, val_regret_log = po_trainer.fit(
        training_data["train_loader"], 
        training_data["val_loader"], 
        epochs=epochs
    )

    if verbose or file_name is not None:
        # Plot the learning curve
        PFLTrainer.vis_learning_curve(
            po_trainer,
            train_loss_log,
            train_regret_log,
            val_loss_log,
            val_regret_log,
            **kwargs
        )

        print("Final regret on validation set: ", val_regret_log[-1])

    write_pred(cfg, po_model.state_dict(), artifact_tag=cache_tag, replace=replace_pred)
    print(f"Saved predictor model '{cache_tag}' to file.")

    return po_model


def setup_dfl_predictor(
        cfg: HP,
        graph: Grid,
        opt_model: 'ShortestPathGrb',
        training_data: dict,
        *,
        cache_tag: str = "dfl",
        verbose: bool = False,
        file_name: str = None,
        transfer_model: nn.Sequential = None,
        dfl_variant: str = "a-dfl",
        cache_options: CacheReplaceOptions | None = None,
        **kwargs
        ):
    """
    Train a SPO model for the shortest path problem.
    """

    # Specify a file name instead of verbose plotting without saving
    if verbose and file_name is None:
        print(
            "Warning: You have enabled verbose mode " \
             + "without specifying a file name. Plots will not be saved."
        )

    # Define your network dimensions
    input_size  =  cfg.get("num_features")   # e.g. number of features in your cost‐vector
    output_size =  graph.num_cost   # e.g. # of target outputs, or number of classes

    # Set the random seed for reproducibility
    set_seed(cfg)

    # Build the model with nn.Sequential
    if transfer_model is None:
        spo_model = get_nn(input_size, output_size) \
            if cfg.get("pred_model") is None or cfg.get("pred_model") == "nn" \
            else nn.Linear(input_size, output_size)
    else:
        spo_model = deepcopy(transfer_model)

    cache_options = cache_options or get_cache_replace_options()
    replace_pred = cache_options.for_artifact(Artefacts.PRED)

    # Check if model has been cached already unless replacement is forced.
    state_dict = None if replace_pred else read_cache(cfg, Artefacts.PRED, artifact_tag=cache_tag)
    if state_dict is not None:
        spo_model.load_state_dict(state_dict)
        print(f"Loaded existing predictor model '{cache_tag}' from file.")
        return spo_model

    # Init SPO+ or hybrid SPO+ loss
    lam = cfg.get("lam")
    if lam == 0:
        loss_fn = pyepo.func.SPOPlus(opt_model, processes=1)
    else:
        loss_fn = HybridSPOPLoss(opt_model, lam=lam, anchor=cfg.get("anchor"))

    # Init optimizer
    optimizer = torch.optim.Adam(spo_model.parameters(), lr=cfg.get("spo_lr"))

    # Create a trainer instance
    spo_trainer = DFLTrainer(
        pred_model=spo_model, 
        opt_model=opt_model, 
        optimizer=optimizer, 
        loss_fn=loss_fn,
        dfl_variant=dfl_variant,
    )

    # Train the model
    train_loss_log, train_regret_log, val_loss_log, val_regret_log = spo_trainer.fit(
        training_data["train_loader"], 
        training_data["val_loader"], 
        epochs=cfg.get("spo_epochs")
    )

    if verbose or file_name is not None:
        # Plot the learning curve
        DFLTrainer.vis_learning_curve(
            spo_trainer,
            train_loss_log,
            train_regret_log,
            val_loss_log,
            val_regret_log,
            **kwargs
        )

        print("Final regret on validation set: ", val_regret_log[-1])

    write_pred(cfg, spo_model.state_dict(), artifact_tag=cache_tag, replace=replace_pred)
    print(f"Saved predictor model '{cache_tag}' to file.")

    return spo_model
