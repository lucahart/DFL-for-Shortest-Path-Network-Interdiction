from models.CalibratedPredictor import CalibratedPredictor
import pyepo
import torch
from torch import nn
from copy import deepcopy

from sklearn.model_selection import train_test_split
from models.ShortestPathGrid import ShortestPathGrid
from models.ShortestPathGrb import shortestPathGrb
from models.POTrainer import POTrainer
from models.SPOTrainer import SPOTrainer
from models.HybridSPOLoss import HybridSPOLoss
from data.config import HP

def gen_train_data(
        cfg: HP,
        opt_model: 'shortestPathGrb'
                ) -> dict:
    """
    Sets up the graph and data loaders for the shortest path problem.
    """

    # Generate synthetic data for training and testing
    features, costs = pyepo.data.shortestpath.genData(
        cfg.get("num_train_samples") + cfg.get("num_test_samples"), 
        cfg.get("num_features"), 
        cfg.get("grid_size"), 
        deg=cfg.get("deg"), 
        noise_width=cfg.get("noise_width"), 
        seed=cfg.get("random_seed")
    )

    # Normalize costs
    normalization_constant = costs.max()
    costs = costs / normalization_constant

    # Split the data into training and testing sets
    X_train, X_test, c_train, c_test = train_test_split(features, costs, test_size=cfg.get("num_test_samples"), random_state=cfg.get("random_seed"))

    # Split the training data into training and validation data
    X_train, X_val, c_train, c_val = train_test_split(X_train, c_train, test_size=cfg.get("validation_size"), random_state=cfg.get("random_seed"))

    # Create data sets
    train_dataset = pyepo.data.dataset.optDataset(opt_model, X_train, c_train)
    val_dataset = pyepo.data.dataset.optDataset(opt_model, X_val, c_val)

    # Create data loaders for training and validation
    g = torch.Generator().manual_seed(cfg.get("random_seed"))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.get("data_loader_batch_size"), shuffle=True, generator=g)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.get("data_loader_batch_size"), shuffle=False, generator=g)

    # Return the train and validation data loaders and the test data
    return {
        "train_loader": train_loader,
        "val_loader": val_loader
    }, {
        "feats": X_test,
        "costs": c_test
    }, normalization_constant


def gen_data(cfg: HP,
                  normalization_constant,
                  seed: int = 31) -> dict:

    # Generate true network data for simulation
    features, costs = pyepo.data.shortestpath.genData(
        cfg.get("sim_data_samples"),
        cfg.get("num_features"),
        cfg.get("grid_size"),
        deg=cfg.get("deg"),
        noise_width=cfg.get("noise_width"),
        seed=seed
    )

    # Normalize costs
    costs = costs / normalization_constant

    return {
        "features": features,
        "costs": costs
    }


@staticmethod
def get_nn(input_size, output_size):

    hidden_size_1 =  64   # number of neurons in the hidden layer
    return nn.Sequential(
        nn.Linear(input_size, hidden_size_1),  # first affine layer
        nn.ReLU(),                           # non‐linearity
        nn.Linear(hidden_size_1, output_size),  # third affine layer
        nn.Sigmoid()                         # output activation function
    )


def setup_po_model(
        cfg: HP,
        graph: ShortestPathGrid,
        opt_model: 'shortestPathGrb',
        training_data: dict,
        versatile: bool = False,
        train_type: str = "po",
        **kwargs
        ):
    
    # Define your network dimensions
    input_size  =  cfg.get("num_features")   # e.g. number of features in your cost‐vector
    output_size =  graph.num_cost   # e.g. # of target outputs, or number of classes

    # This setup can be used for training a PO model or pre-training an SPO model
    if train_type is "po":
        lr = cfg.get("po_lr")
        epochs = cfg.get("po_epochs")
    elif train_type is "spo":
        lr = cfg.get("spo_po_lr")
        epochs = cfg.get("spo_po_epochs")
    else:
        raise(f"Training type {train_type} is not defined.")


    # Build the model with nn.Sequential
    po_model = get_nn(input_size, output_size)

    # Define the loss function and optimizer
    po_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(po_model.parameters(), lr=lr)

    po_trainer = POTrainer(
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

    if versatile:
        # Plot the learning curve
        POTrainer.vis_learning_curve(
            po_trainer,
            train_loss_log,
            train_regret_log,
            val_loss_log,
            val_regret_log
        )

        print("Final regret on validation set: ", val_regret_log[-1])

    return po_model

def setup_spo_model(
        cfg: HP,
        graph: ShortestPathGrid,
        opt_model: 'shortestPathGrb',
        training_data: dict,
        versatile: bool = False,
        transfer_model: nn.Sequential = None,
        **kwargs
        ):
    """
    Train a SPO model for the shortest path problem.
    """

    # Define your network dimensions
    input_size  =  cfg.get("num_features")   # e.g. number of features in your cost‐vector
    output_size =  graph.num_cost   # e.g. # of target outputs, or number of classes

    # Build the model with nn.Sequential
    if transfer_model is None:
        spo_model = get_nn(input_size, output_size)
    else:
        spo_model = deepcopy(transfer_model)

    # Init SPO+ loss
    spop = pyepo.func.SPOPlus(opt_model, processes=1)

    # Init optimizer
    optimizer = torch.optim.Adam(spo_model.parameters(), lr=cfg.get("spo_lr"))

    # Create a trainer instance
    spo_trainer = SPOTrainer(
        pred_model=spo_model, 
        opt_model=opt_model, 
        optimizer=optimizer, 
        loss_fn=spop
    )

    # Train the model
    train_loss_log, train_regret_log, val_loss_log, val_regret_log = spo_trainer.fit(
        training_data["train_loader"], 
        training_data["val_loader"], 
        epochs=cfg.get("spo_epochs")
    )

    if versatile:
        # Plot the learning curve
        SPOTrainer.vis_learning_curve(
            spo_trainer,
            train_loss_log,
            train_regret_log,
            val_loss_log,
            val_regret_log
        )

        print("Final regret on validation set: ", val_regret_log[-1])

    return spo_model


def setup_hybrid_spo_model(
        cfg: HP,
        graph: ShortestPathGrid,
        opt_model: 'shortestPathGrb',
        training_data: dict,
        versatile: bool = False,
        transfer_model: nn.Sequential = None,
        **kwargs
        ):
    """
    Train a SPO model for the shortest path problem.
    """

    # Define your network dimensions
    input_size  =  cfg.get("num_features")   # e.g. number of features in your cost‐vector
    output_size =  graph.num_cost   # e.g. # of target outputs, or number of classes

    # Build the model with nn.Sequential
    if transfer_model is None:
        spo_model = get_nn(input_size, output_size)
    else:
        spo_model = deepcopy(transfer_model)

    # Add a calibration layer
    spo_model_calibrated = CalibratedPredictor(spo_model)

    # Init SPO+ loss
    spo_loss = HybridSPOLoss(opt_model, lam=cfg.get("lam"), anchor=cfg.get("anchor"))

    # assuming model has .log_s and .b
    calib_params = [spo_model_calibrated.log_s]
    backbone_params = [p for n,p in spo_model_calibrated.named_parameters() if n not in {'log_s','b'}]

    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': cfg.get("spo_lr"), 'weight_decay': 1e-4},
        {'params': calib_params,   'lr': cfg.get("spo_lr")*10, 'weight_decay': 0.0},  # 10× faster
    ])


    # # Init optimizer
    # optimizer = torch.optim.Adam(spo_model_calibrated.parameters(), lr=cfg.get("spo_lr"))

    # Create a trainer instance
    spo_trainer = SPOTrainer(
        pred_model=spo_model_calibrated, 
        opt_model=opt_model, 
        optimizer=optimizer, 
        loss_fn=spo_loss
    )

    # Train the model
    train_loss_log, train_regret_log, val_loss_log, val_regret_log = spo_trainer.fit(
        training_data["train_loader"], 
        training_data["val_loader"], 
        epochs=cfg.get("spo_epochs")
    )

    if versatile:
        # Plot the learning curve
        SPOTrainer.vis_learning_curve(
            spo_trainer,
            train_loss_log,
            train_regret_log,
            val_loss_log,
            val_regret_log
        )

        print("Final regret on validation set: ", val_regret_log[-1])

    return spo_model_calibrated

