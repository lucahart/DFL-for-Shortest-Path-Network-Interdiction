
import random
import pyepo
import torch
import numpy as np
from torch import nn
from copy import deepcopy
from sklearn.model_selection import train_test_split

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
    read_cache,
    write_data,
    write_pred,
)
from dflintdpy.data.adverse.adverse_data_generator import AdvDataGenerator
from dflintdpy.data.adverse.adverse_dataset import AdvDataset
from dflintdpy.data.adverse.adverse_loader import AdvLoader

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
    Sets up the graph and data loaders for the shortest path problem.
    ``interdiction_policy`` controls whether scenario generation uses
    adversarial or random interdictions.
    """

    cache_options = cache_options or get_cache_replace_options()
    replace_data = cache_options.for_artifact(Artefacts.DATA)

    # Load data from cache if available and not forced to replace.
    data = None if replace_data else read_cache(cfg, Artefacts.DATA)
    if data is None: # TODO: num_seeds is not in the data information. What is loaded and how does it handle that the data is missing?
        # Generate synthetic data for training and testing
        features, costs = gen_syn_data(cfg, opt_model)

        # Save generated data if path is provided
        write_data(cfg, features, costs, replace=replace_data)
        print(f"Saved data to file.")
    else:
        features, costs = data["feats"], data["costs"]


    # Normalize costs
    normalization_constant = costs.max()
    costs = costs / normalization_constant

    # Split the data into training and testing sets
    X_train, X_test, c_train, c_test = train_test_split(
        features, 
        costs, 
        test_size=cfg.get("num_test_samples"), 
        random_state=cfg.get("random_seed")
    )

    # Generate adversarial examples for the validation set
    adversarial_generator = AdvDataGenerator(
        cfg, 
        opt_model, 
        budget=cfg.get("budget"), 
        normalization_constant=normalization_constant,
        num_scenarios=cfg.get("num_scenarios"),
        interdiction_policy=interdiction_policy,
        cache_options=cache_options,
        # gen_intd_seed=cfg.get("gen_intd_seed"), # 157 if not specified otherwise
    )

    X_train, c_train, i_train = adversarial_generator.generate(
        X_train, 
        c_train,
        cfg=cfg
    )
    
    # Split the training data into training and validation data
    idxs = np.arange(X_train.shape[0])
    X_train, X_val, idxs_train, idxs_val = train_test_split(
        X_train, 
        idxs, 
        test_size=cfg.get("num_val_samples"), 
        random_state=cfg.get("random_seed")
    )
    c_train, c_val = c_train[idxs_train], c_train[idxs_val]
    i_train, i_val = i_train[idxs_train], i_train[idxs_val]


    # Create data sets
    train_dataset = AdvDataset(opt_model, X_train, c_train, i_train)
    val_dataset = AdvDataset(opt_model, X_val, c_val, i_val)

    # Create data loaders for training and validation
    train_loader = AdvLoader(
        train_dataset,
        batch_size=cfg.get("batch_size"),
        seed=cfg.get("loader_seed"),
        shuffle=True,
    )
    val_loader = AdvLoader(
        val_dataset,
        batch_size=cfg.get("batch_size"),
        seed=cfg.get("loader_seed"),
        shuffle=False,
    )

    # Return the train and validation data loaders and the test data
    return {
        "train_loader": train_loader,
        "val_loader": val_loader
    }, {
        "feats": X_test,
        "costs": c_test
    }, normalization_constant,{
        "data_generator": adversarial_generator
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
