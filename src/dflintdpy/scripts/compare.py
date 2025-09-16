
import pyepo
import torch
import numpy as np

from dflintdpy.models.grid import Grid
from dflintdpy.solvers.shortest_path_grb import ShortestPathGrb
from dflintdpy.solvers.asymmetric_interdictor import AsymmetricInterdictor
from dflintdpy.solvers.symmetric_interdictor import SymmetricInterdictor
from dflintdpy.utils.versatile_utils import print_progress


def compare_shortest_paths(cfg,
                   opt_model: 'ShortestPathGrb',
                   po_model: torch.nn.Module,
                   spo_model: torch.nn.Module,
                   test_data: dict,
                   spo_adv_model: torch.nn.Module = None
                   ) -> None:
    """
    Compare the performance of the PO and SPO models on a shortest path problem.
    """

    # Retrieve configuration parameters
    data_samples = cfg.get("num_test_samples") # number of training data
    m, n = cfg.get("grid_size")

    # Initialize lists to store results
    true_objs = []
    po_objs = []
    spo_objs = []
    adv_spo_objs = []

    # Iterate over the generated data samples
    for i in range(data_samples):
        # Store temporary values
        cost = test_data["costs"][i]
        feature = test_data["feats"][i]

        # Set the cost for the grid (Optionally specify the source and target nodes)
        opt_model.setObj(cost)

        # Solve shortest path problem
        _, obj = opt_model.solve()

        # Predict the shortest path with predict-then-optimize framework
        predicted_costs = po_model(torch.tensor(feature, dtype=torch.float32)).detach().numpy()
        po_graph = Grid(m, n, cost=predicted_costs)
        po_shortest_path = ShortestPathGrb(po_graph)
        po_path, _ = po_shortest_path.solve()

        # Predict shortest path with smart predict-then-optimize framework
        predicted_costs = spo_model(torch.tensor(feature, dtype=torch.float32)).detach().numpy()
        spo_graph = Grid(m, n, cost=predicted_costs)
        spo_shortest_path = ShortestPathGrb(spo_graph)
        spo_path, _ = spo_shortest_path.solve()

        if spo_adv_model is not None:
            # Predict shortest path with adverse SPO framework
            predicted_costs = spo_adv_model(torch.tensor(feature, dtype=torch.float32)).detach().numpy()
            adv_spo_graph = Grid(m, n, cost=predicted_costs)
            adv_spo_shortest_path = ShortestPathGrb(adv_spo_graph)
            adv_spo_path, _ = adv_spo_shortest_path.solve()

        # Evaluate the estimated paths
        po_obj = opt_model.evaluate(po_path)
        spo_obj = opt_model.evaluate(spo_path)
        adv_spo_obj = opt_model.evaluate(adv_spo_path) if spo_adv_model is not None else None

        # Store the results
        true_objs.append(obj)
        po_objs.append(po_obj)
        spo_objs.append(spo_obj)
        adv_spo_objs.append(adv_spo_obj)

    return true_objs, po_objs, spo_objs, adv_spo_objs


def compare_sym_intd(cfg, po_model, spo_model, test_data, interdictions, normalization_constant, idx = None, adv_spo_model = None):
    """
    Compare the performance of the PO and SPO models using symmetric shortest path interdiction.
    This function simulates the interdiction process and evaluates the objective values.
    """

    # Get the number of simulation data samples
    num_test_samples = cfg.get("num_test_samples")

    # Prepare lists to store results
    true_objs = []
    po_objs = []
    spo_objs = []
    adv_spo_objs = []

    # Print that the simulation is starting
    print(f"Running simulation with {num_test_samples} samples...")

    # Iterate through each data sample
    for i in range(num_test_samples) if idx is None else [idx]:
        # Store values for the current sample
        feature = test_data["feats"][i]
        cost = test_data["costs"][i] * normalization_constant
        interdiction = interdictions["costs"][i] * normalization_constant
        m, n = cfg.get("grid_size")
        
        # Update opt_model
        true_graph = Grid(m, n, cost=cost)
        opt_model = ShortestPathGrb(true_graph)

        # Update the estimated costs
        po_cost = po_model(torch.tensor(feature, dtype=torch.float32)).detach().numpy() * normalization_constant
        spo_cost = spo_model(torch.tensor(feature, dtype=torch.float32)).detach().numpy() * normalization_constant
        if adv_spo_model is not None:
            adv_spo_cost = adv_spo_model(torch.tensor(feature, dtype=torch.float32)).detach().numpy() * normalization_constant

        # Solutions without information asymmetry
        interdictor_I = SymmetricInterdictor(true_graph, 
                                             k=cfg.get("budget"), 
                                             interdiction_cost=interdiction, 
                                             max_cnt=cfg.get("benders_max_count"), 
                                             eps=cfg.get("benders_eps"))
        x_intd, _, _ = interdictor_I.solve(versatile=False if idx is None else True)

        # True shortest path after interdiction
        opt_model.setObj(cost + x_intd * interdiction)
        y_true, _ = opt_model.solve()

        # PO estimated shortest path after interdiction
        opt_model.setObj(po_cost + x_intd * interdiction)
        y_po, _ = opt_model.solve()

        # SPO estimated shortest path after interdiction
        opt_model.setObj(spo_cost + x_intd * interdiction)
        y_spo, _ = opt_model.solve()

        # Adverse SPO shortest path after interdiction
        if adv_spo_model is not None:
            opt_model.setObj(adv_spo_cost + x_intd * interdiction)
            y_adv_spo, _ = opt_model.solve()

        # Store the results
        true_objs.append(true_graph(y_true, interdictions=x_intd * interdiction))
        po_objs.append(true_graph(y_po, interdictions=x_intd * interdiction))
        spo_objs.append(true_graph(y_spo, interdictions=x_intd * interdiction))
        if adv_spo_model is not None:
            adv_spo_objs.append(true_graph(y_adv_spo, interdictions=x_intd * interdiction))

        print_progress(i, num_test_samples)

    # Evaluate performance
    return {
        "true_objective": np.array(true_objs),
        "po_objective": np.array(po_objs),
        "spo_objective": np.array(spo_objs),
    } if adv_spo_model is None else {
        "true_objective": np.array(true_objs),
        "po_objective": np.array(po_objs),
        "spo_objective": np.array(spo_objs),
        "adv_spo_objective": np.array(adv_spo_objs)
    }


def compare_asym_intd(cfg, test_data, interdictions, normalization_constant, pred_model = None):
    """
    Compare the performance of the predicted model with the true model
    using asymmetric shortest path interdiction.
    """ 

    # Get the number of simulation data samples
    num_test_samples = cfg.get("num_test_samples")

    # Print that the simulation is starting
    print(f"Running simulation with {num_test_samples} samples...")

    # Prepare lists to store results
    true_objs = []
    est_objs = []

    # Iterate through each data sample
    for i in range(num_test_samples):
        # Store values for the current sample
        cost = test_data["costs"][i] * normalization_constant
        interdiction = interdictions["costs"][i] * normalization_constant
        m, n = cfg.get("grid_size")

        # Compute estimated cost if an estimator is provided. Otherwise use the true costs
        if pred_model is not None:
            feature = test_data["feats"][i]
            pred_cost = pred_model(torch.tensor(feature, dtype=torch.float32)).detach().numpy() * normalization_constant
        else:
            pred_cost = cost
        
        # Create graph with true costs and opt_model for estimated shortest path
        true_graph = Grid(m, n, cost=cost)
        est_opt_model = ShortestPathGrb(true_graph)
        est_opt_model.setObj(pred_cost)

        # Solutions with information asymmetry
        asym_interdictor = AsymmetricInterdictor(
            true_graph, 
            budget=cfg.get("budget"), 
            true_costs=cost, 
            true_delays=interdiction, 
            est_costs=pred_cost, 
            est_delays=interdiction, 
            lsd=cfg.get("lsd")
        )
        x_intd, _ = asym_interdictor.solve()

        # # True shortest path after interdiction
        # opt_model.setObj(cost + x_intd * interdiction)
        # y_true, _ = opt_model.solve()

        # Estimated shortest path after interdiction
        est_opt_model.setObj(pred_cost + x_intd * interdiction)
        y_est, _ = est_opt_model.solve()

        # Store the results
        # true_objs.append(true_graph(y_true, interdictions=x_intd * interdiction))
        est_objs.append(true_graph(y_est, interdictions=x_intd * interdiction))

        # Print progress
        print_progress(i, num_test_samples)

    # Evaluate performance
    # return {
    #     "true_objective": np.array(true_objs),
    #     "estimated_objective": np.array(est_objs),
    # }
    return np.array(est_objs)


def compare_wrong_asym_intd(cfg, test_data, interdictions, normalization_constant, true_model, false_model):
    """
    Evader uses true model and interdictor uses asymmetric SPNI assuming false model.
    """ 

    # Get the number of simulation data samples
    num_test_samples = cfg.get("num_test_samples")

    # Print that the simulation is starting
    print(f"Running simulation with {num_test_samples} samples...")

    # Prepare lists to store results
    true_objs = []
    est_objs = []

    # Iterate through each data sample
    for i in range(num_test_samples):
        # Store values for the current sample
        cost = test_data["costs"][i] * normalization_constant
        interdiction = interdictions["costs"][i] * normalization_constant
        m, n = cfg.get("grid_size")

        # Compute estimated cost if an estimator is provided. Otherwise use the true costs
        feature = test_data["feats"][i]
        true_pred_cost = true_model(torch.tensor(feature, dtype=torch.float32)).detach().numpy() * normalization_constant
        false_pred_cost = false_model(torch.tensor(feature, dtype=torch.float32)).detach().numpy() * normalization_constant

        # Create graph with true costs and opt_model for estimated shortest path
        true_graph = Grid(m, n, cost=cost)
        est_opt_model = ShortestPathGrb(true_graph)

        # Solutions with information asymmetry
        asym_interdictor = AsymmetricInterdictor(
            true_graph,
            
            budget=cfg.get("budget"), 
            true_costs=cost, 
            true_delays=interdiction, 
            est_costs=false_pred_cost, 
            est_delays=interdiction, 
            lsd=cfg.get("lsd")
        )
        x_intd, _ = asym_interdictor.solve()

        # # True shortest path after interdiction
        # opt_model.setObj(cost + x_intd * interdiction)
        # y_true, _ = opt_model.solve()

        # Estimated shortest path after interdiction
        est_opt_model.setObj(true_pred_cost + x_intd * interdiction)
        y_est, _ = est_opt_model.solve()

        # Store the results
        # true_objs.append(true_graph(y_true, interdictions=x_intd * interdiction))
        est_objs.append(true_graph(y_est, interdictions=x_intd * interdiction))

        # Print progress
        print_progress(i, num_test_samples)

    # Evaluate performance
    # return {
    #     "true_objective": np.array(true_objs),
    #     "estimated_objective": np.array(est_objs),
    # }
    return np.array(est_objs)




