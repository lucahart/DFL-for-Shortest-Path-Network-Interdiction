import pyepo
import torch
import numpy as np
from torch import nn
from copy import deepcopy
from sklearn.model_selection import train_test_split

from dflintdpy.data.config import HP
from dflintdpy.models.grid import Grid
from dflintdpy.solvers.shortest_path_grb import ShortestPathGrb
from dflintdpy.predictors.hybrid_spop_loss import HybridSPOPLoss
from dflintdpy.utils.pfl_trainer import PFLTrainer
from dflintdpy.utils.dfl_trainer import DFLTrainer
from dflintdpy.data.adverse.adverse_data_generator import AdvDataGenerator
from dflintdpy.data.adverse.adverse_dataset import AdvDataset
from dflintdpy.data.adverse.adverse_loader import AdvLoader

def gen_train_data(
        cfg: HP,
        opt_model: 'ShortestPathGrb',
        path_dir: str = None
) -> dict:
    """
    Sets up the graph and data loaders for the shortest path problem.
    """
    file_found = False

    # Generate file path if directory is provided
    if path_dir is not None:
        file_name_body = "_samples_{samples}_m_{m}_n_{n}_deg_{deg}_noise_{noise}_seed_{seed}.csv".format(
            samples=cfg.get("num_train_samples") + cfg.get("num_val_samples") + cfg.get("num_test_samples"),
            m=cfg.get("grid_size")[0], 
            n=cfg.get("grid_size")[1], 
            deg=cfg.get("deg"), 
            noise=cfg.get("noise_width"), 
            seed=cfg.get("random_seed")
        )
        path_file = path_dir / ("xy" + file_name_body)
        
        try:
            # Load pre-generated data if available
            features, costs = _load_features_costs(path_file)
            print(f"Loaded existing cost and feature data from file.")
            file_found = True
        except (FileNotFoundError, OSError) as e:
            print(f"Could not load cached data: {e}")
            pass  # If file not found, generate new data below

    if not file_found:
        # Generate synthetic data for training and testing
        features, costs = pyepo.data.shortestpath.genData(
            cfg.get("num_train_samples") + cfg.get("num_val_samples") + cfg.get("num_test_samples"), 
            cfg.get("num_features"), 
            cfg.get("grid_size"), 
            deg=cfg.get("deg"), 
            noise_width=cfg.get("noise_width"), 
            seed=cfg.get("random_seed")
        )
        
        # Save generated data if path is provided
        if path_dir is not None:
            _save_features_costs(path_file, features, costs)

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
    
    # Split the training data into training and validation data
    X_train, X_val, c_train, c_val = train_test_split(
        X_train, 
        c_train, 
        test_size=cfg.get("num_val_samples"), 
        random_state=cfg.get("random_seed")
    )

    # Generate adversarial examples for the validation set
    adversarial_generator = AdvDataGenerator(
        cfg, 
        opt_model, 
        budget=cfg.get("budget"), 
        normalization_constant=normalization_constant,
        num_scenarios=cfg.get("num_scenarios")
    )
    X_train, c_train, i_train = adversarial_generator.generate(
        X_train, 
        c_train,
        file_path=path_dir / ("i_train" + file_name_body) if path_dir is not None else None
    )
    X_val, c_val, i_val = adversarial_generator.generate(
        X_val, 
        c_val,
        file_path=path_dir / ("i_valid" + file_name_body) if path_dir is not None else None
    )

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
    }, normalization_constant


def _save_features_costs(file_path, features, costs):
    """
    Save features and costs to CSV files.
    
    Args:
        file_path: Base file path (without extension)
        features: Feature array
        costs: Cost array
    """
    features_path = str(file_path).replace("xy_", "features_")
    costs_path = str(file_path).replace("xy_", "costs_")
    
    np.savetxt(features_path, features, delimiter=',')
    np.savetxt(costs_path, costs, delimiter=',')
    print(f"Saved features to {features_path}")
    print(f"Saved costs to {costs_path}")


def _load_features_costs(file_path):
    """
    Load features and costs from CSV files.
    
    Args:
        file_path: Base file path (without extension)
        
    Returns:
        tuple: (features, costs) as numpy arrays
        
    Raises:
        FileNotFoundError: If either file doesn't exist
    """
    features_path = str(file_path).replace("xy_", "features_")
    costs_path = str(file_path).replace("xy_", "costs_")
    
    features = np.loadtxt(features_path, delimiter=',', dtype=np.float32)
    costs = np.loadtxt(costs_path, delimiter=',', dtype=np.float32)
    
    return features, costs


def gen_data(cfg: HP,
            normalization_constant,
            seed: int = 31) -> dict:

    # Generate true network data for simulation
    features, costs = pyepo.data.shortestpath.genData(
        cfg.get("num_test_samples"),
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

    if versatile:
        # Plot the learning curve
        PFLTrainer.vis_learning_curve(
            po_trainer,
            train_loss_log,
            train_regret_log,
            val_loss_log,
            val_regret_log
        )

        print("Final regret on validation set: ", val_regret_log[-1])

    return po_model


def setup_dfl_predictor(
        cfg: HP,
        graph: Grid,
        opt_model: 'ShortestPathGrb',
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
    )

    # Train the model
    train_loss_log, train_regret_log, val_loss_log, val_regret_log = spo_trainer.fit(
        training_data["train_loader"], 
        training_data["val_loader"], 
        epochs=cfg.get("spo_epochs")
    )

    if versatile:
        # Plot the learning curve
        DFLTrainer.vis_learning_curve(
            spo_trainer,
            train_loss_log,
            train_regret_log,
            val_loss_log,
            val_regret_log
        )

        print("Final regret on validation set: ", val_regret_log[-1])

    return spo_model


# def setup_hybrid_spo_model(
#         cfg: HP,
#         graph: Grid,
#         opt_model: 'ShortestPathGrb',
#         training_data: dict,
#         versatile: bool = False,
#         transfer_model: nn.Sequential = None,
#         **kwargs
#         ):
#     """
#     Train a SPO model for the shortest path problem.
#     """

#     # Define your network dimensions
#     input_size  =  cfg.get("num_features")   # e.g. number of features in your cost‐vector
#     output_size =  graph.num_cost   # e.g. # of target outputs, or number of classes

#     # Build the model with nn.Sequential
#     if transfer_model is None:
#         spo_model = get_nn(input_size, output_size)
#     else:
#         spo_model = deepcopy(transfer_model)

#     # Add a calibration layer
#     spo_model_calibrated = CalibratedPredictor(spo_model)

#     # Init SPO+ loss
#     hybrid_loss = HybridSPOLoss(opt_model, lam=cfg.get("lam"), anchor=cfg.get("anchor"))

#     # assuming model has .log_s and .b
#     calib_params = [spo_model_calibrated.log_s]
#     backbone_params = [p for n,p in spo_model_calibrated.named_parameters() if n not in {'log_s','b'}]

#     optimizer = torch.optim.Adam([
#         {'params': backbone_params, 'lr': cfg.get("spo_lr"), 'weight_decay': 0.0},
#         {'params': calib_params,   'lr': cfg.get("spo_lr")*10, 'weight_decay': 0.0},  # 10× faster
#     ])


#     # # Init optimizer
#     # optimizer = torch.optim.Adam(spo_model_calibrated.parameters(), lr=cfg.get("spo_lr"))

#     # Create a trainer instance
#     spo_trainer = SPOTrainer(
#         pred_model=spo_model_calibrated, 
#         opt_model=opt_model, 
#         optimizer=optimizer, 
#         loss_fn=hybrid_loss,
#         method_name="hybrid",
#         cfg=cfg
#     )

#     # Train the model
#     train_loss_log, train_regret_log, val_loss_log, val_regret_log = spo_trainer.fit(
#         training_data["train_loader"], 
#         training_data["val_loader"], 
#         epochs=cfg.get("spo_epochs")
#     )

#     if versatile:
#         # Plot the learning curve
#         SPOTrainer.vis_learning_curve(
#             spo_trainer,
#             train_loss_log,
#             train_regret_log,
#             val_loss_log,
#             val_regret_log
#         )

#         print("Final regret on validation set: ", val_regret_log[-1])

#     return spo_model_calibrated

