import pyepo
import torch
from torch import nn

from sklearn.model_selection import train_test_split
from models.ShortestPathGrid import ShortestPathGrid
from models.ShortestPathGrb import shortestPathGrb
from models.POTrainer import POTrainer
from models.SPOTrainer import SPOTrainer
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
        cfg.get("num_data_samples"), 
        cfg.get("num_features"), 
        cfg.get("grid_size"), 
        deg=cfg.get("deg"), 
        noise_width=cfg.get("noise_width"), 
        seed=cfg.get("random_seed")
    )

    # Split the data into training and testing sets
    X_train, X_test, c_train, c_test = train_test_split(features, costs, test_size=cfg.get("test_size"), random_state=cfg.get("random_seed"))

    # Create data loaders for training and testing
    optnet_train_dataset = pyepo.data.dataset.optDataset(opt_model, X_train, c_train)
    optnet_test_dataset = pyepo.data.dataset.optDataset(opt_model, X_test, c_test)

    g = torch.Generator().manual_seed(cfg.get("random_seed"))
    optnet_train_loader = torch.utils.data.DataLoader(optnet_train_dataset, batch_size=cfg.get("data_loader_batch_size"), shuffle=True, generator=g)
    optnet_test_loader = torch.utils.data.DataLoader(optnet_test_dataset, batch_size=cfg.get("data_loader_batch_size"), shuffle=False, generator=g)

    return {
        "train_loader": optnet_train_loader,
        "test_loader": optnet_test_loader
    }

def gen_test_data(cfg: HP) -> dict:
    
    # Generate true network data for simulation
    features, costs = pyepo.data.shortestpath.genData(
        cfg.get("sim_data_samples"),
        cfg.get("num_features"),
        cfg.get("grid_size"),
        deg=cfg.get("deg"),
        noise_width=cfg.get("noise_width"),
        seed=cfg.get("random_seed")
    )

    return {
        "features": features,
        "costs": costs
    }


def setup_po_model(
        cfg: HP,
        graph: ShortestPathGrid,
        opt_model: 'shortestPathGrb',
        training_data: dict,
        versatile: bool = False,
        **kwargs
        ):
    
    # Define your network dimensions
    input_size  =  cfg.get("num_features")   # e.g. number of features in your cost‐vector
    hidden_size =  64   # number of neurons in the hidden layer
    output_size =  graph.num_cost   # e.g. # of target outputs, or number of classes

    # Build the model with nn.Sequential
    po_model = nn.Sequential(
        nn.Linear(input_size, hidden_size),  # first affine layer
        nn.ReLU(),                           # non‐linearity
        nn.Linear(hidden_size, output_size),  # second affine layer
        nn.Sigmoid()                         # output activation function
    )

    # Define the loss function and optimizer
    po_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(po_model.parameters(), lr=0.1)

    po_trainer = POTrainer(
        pred_model=po_model,
        opt_model=opt_model,
        optimizer=optimizer,
        loss_fn=po_criterion
    )

    # Train the model
    train_loss_log, train_regret_log, test_loss_log, test_regret_log = po_trainer.fit(
        training_data["train_loader"], 
        training_data["test_loader"], 
        epochs=cfg.get("epochs")
    )

    if versatile:
        # Plot the learning curve
        POTrainer.vis_learning_curve(
            po_trainer,
            train_loss_log,
            train_regret_log,
            test_loss_log,
            test_regret_log
        )

        print("Final regret on test set: ", test_regret_log[-1])

    return po_model

def setup_spo_model(
        cfg: HP,
        graph: ShortestPathGrid,
        opt_model: 'shortestPathGrb',
        training_data: dict,
        versatile: bool = False,
        **kwargs
        ):
    """
    Train a SPO model for the shortest path problem.
    """

    # Define your network dimensions
    input_size  =  cfg.get("num_features")   # e.g. number of features in your cost‐vector
    hidden_size =  64   # number of neurons in the hidden layer
    output_size =  graph.num_cost   # e.g. # of target outputs, or number of classes

    # Build the model with nn.Sequential
    spo_model = nn.Sequential(
        nn.Linear(input_size, hidden_size),  # first affine layer
        nn.ReLU(),                           # non‐linearity
        nn.Linear(hidden_size, output_size),  # second affine layer
        nn.Sigmoid()                         # output activation function
    )

    # Init SPO+ loss
    spop = pyepo.func.SPOPlus(opt_model, processes=1)

    # Init optimizer
    optimizer = torch.optim.Adam(spo_model.parameters(), lr=.1)

    # Create a trainer instance
    spo_trainer = SPOTrainer(
        pred_model=spo_model, 
        opt_model=opt_model, 
        optimizer=optimizer, 
        loss_fn=spop
    )

    # Train the model
    train_loss_log, train_regret_log, test_loss_log, test_regret_log = spo_trainer.fit(
        training_data["train_loader"], 
        training_data["test_loader"], 
        epochs=cfg.get("epochs")
    )

    if versatile:
        # Plot the learning curve
        SPOTrainer.vis_learning_curve(
            spo_trainer,
            train_loss_log,
            train_regret_log,
            test_loss_log,
            test_regret_log
        )

        print("Final regret on test set: ", test_regret_log[-1])
    
    return spo_model

