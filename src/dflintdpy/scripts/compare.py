
import pyepo
import torch
import numpy as np

from dflintdpy.models.graph import Graph
from dflintdpy.solvers.shortest_path_grb import ShortestPathGrb
from dflintdpy.solvers.asymmetric_interdictor import AsymmetricInterdictor
from dflintdpy.solvers.symmetric_interdictor import SymmetricInterdictor
from dflintdpy.utils.versatile_utils import print_progress


def compare_shortest_paths(cfg,
                   opt_model: 'ShortestPathGrb',
                   pfl_predictor: torch.nn.Module,
                   dfl_predictor: torch.nn.Module,
                   test_data: dict,
                   adfl_predictor: torch.nn.Module = None
                   ) -> None:
    """
    Compare the performance of the PO and SPO models on a shortest path problem.
    """

    # Retrieve configuration parameters
    data_samples = cfg.get("num_test_samples") # number of training data
    m, n = cfg.get("grid_size")

    # Initialize lists to store results
    true_objs = []
    pfl_objs = []
    dfl_objs = []
    adfl_objs = []

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
        predicted_costs = pfl_predictor(torch.tensor(feature, dtype=torch.float32)).detach().numpy()
        opt_model.setObj(predicted_costs)
        pfl_path, _ = opt_model.solve()

        # Predict shortest path with smart predict-then-optimize framework
        predicted_costs = dfl_predictor(torch.tensor(feature, dtype=torch.float32)).detach().numpy()
        opt_model.setObj(predicted_costs)
        dfl_path, _ = opt_model.solve()

        if adfl_predictor is not None:
            # Predict shortest path with adverse SPO framework
            predicted_costs = adfl_predictor(torch.tensor(feature, dtype=torch.float32)).detach().numpy()
            opt_model.setObj(predicted_costs)
            adfl_path, _ = opt_model.solve()

        # Evaluate the estimated paths
        opt_model.setObj(cost)
        pfl_obj = opt_model.evaluate(pfl_path)
        dfl_obj = opt_model.evaluate(dfl_path)
        adfl_obj = opt_model.evaluate(adfl_path) if adfl_predictor is not None else None

        # Store the results
        true_objs.append(obj)
        pfl_objs.append(pfl_obj)
        dfl_objs.append(dfl_obj)
        adfl_objs.append(adfl_obj)

    return true_objs, pfl_objs, dfl_objs, adfl_objs


def compare_sym_intd(
        cfg, 
        opt_model: 'ShortestPathGrb',
        pfl_predictor, 
        dfl_predictor, 
        test_data, 
        interdictions, 
        normalization_constant, 
        idx = None, 
        adfl_predictor = None
    ):
    """
    Compare the performance of the PO and SPO models using symmetric shortest path interdiction.
    This function simulates the interdiction process and evaluates the objective values.
    """

    # Get the number of simulation data samples
    num_test_samples = cfg.get("num_test_samples")

    # Prepare lists to store results
    true_objs = []
    pfl_objs = []
    dfl_objs = []
    adfl_objs = []

    # Print that the simulation is starting
    print(f"Running symmetric simulations with {num_test_samples} samples...")

    # Iterate through each data sample
    for i in range(num_test_samples) if idx is None else [idx]:
        # Store values for the current sample
        feature = test_data["feats"][i]
        cost = test_data["costs"][i] * normalization_constant
        interdiction = interdictions["costs"][i] * normalization_constant

        # Update the estimated costs
        pfl_cost = pfl_predictor(torch.tensor(feature, dtype=torch.float32)) \
            .detach().numpy() * normalization_constant
        dfl_cost = dfl_predictor(torch.tensor(feature, dtype=torch.float32)) \
            .detach().numpy() * normalization_constant
        if adfl_predictor is not None:
            adfl_cost = adfl_predictor(torch.tensor(feature, dtype=torch.float32)) \
                .detach().numpy() * normalization_constant

        # Solutions without information asymmetry
        interdictor_I = SymmetricInterdictor(
            graph=opt_model._graph, 
            k=cfg.get("budget"), 
            interdiction_cost=interdiction, 
            max_cnt=cfg.get("benders_max_count"), 
            eps=cfg.get("benders_eps")
        )
        x_intd, _, _ = interdictor_I.solve(versatile=False if idx is None else True)

        # True shortest path after interdiction
        opt_model.setObj(cost + x_intd * interdiction)
        y_true, _ = opt_model.solve()

        # PO estimated shortest path after interdiction
        opt_model.setObj(pfl_cost + x_intd * interdiction)
        y_po, _ = opt_model.solve()

        # SPO estimated shortest path after interdiction
        opt_model.setObj(dfl_cost + x_intd * interdiction)
        y_spo, _ = opt_model.solve()

        # Adverse SPO shortest path after interdiction
        if adfl_predictor is not None:
            opt_model.setObj(adfl_cost + x_intd * interdiction)
            y_adv_spo, _ = opt_model.solve()

        # Set true cost to evaluate objectives
        opt_model.setObj(cost + x_intd * interdiction)

        # Store the results
        true_objs.append(opt_model._graph(y_true, interdictions=x_intd * interdiction))
        pfl_objs.append(opt_model._graph(y_po, interdictions=x_intd * interdiction))
        dfl_objs.append(opt_model._graph(y_spo, interdictions=x_intd * interdiction))
        if adfl_predictor is not None:
            adfl_objs.append(opt_model._graph(y_adv_spo, interdictions=x_intd * interdiction))
        print_progress(i, num_test_samples)

    # Evaluate performance
    return {
        "true_objective": np.array(true_objs),
        "po_objective": np.array(pfl_objs),
        "spo_objective": np.array(dfl_objs),
    } if adfl_predictor is None else {
        "true_objective": np.array(true_objs),
        "po_objective": np.array(pfl_objs),
        "spo_objective": np.array(dfl_objs),
        "adv_spo_objective": np.array(adfl_objs)
    }


def compare_asym_intd(
        cfg, 
        opt_model: 'ShortestPathGrb',
        test_data, 
        interdictions, 
        normalization_constant, 
        pred_model = None
    ):
    """
    Compare the performance of the predicted model with the true model
    using asymmetric shortest path interdiction.
    """ 

    # Get the number of simulation data samples
    num_test_samples = cfg.get("num_test_samples")

    # Print that the simulation is starting
    print(f"Running asymmetric simulation with {num_test_samples} samples...")

    # Prepare lists to store results
    est_objs = []

    # Iterate through each data sample
    for i in range(num_test_samples):
        # Store values for the current sample
        cost = test_data["costs"][i] * normalization_constant
        interdiction = interdictions["costs"][i] * normalization_constant

        # Compute estimated cost if an estimator is provided. Otherwise use the true costs
        if pred_model is not None:
            feature = test_data["feats"][i]
            pred_cost = pred_model(torch.tensor(feature, dtype=torch.float32)) \
                .detach().numpy() * normalization_constant
        else:
            pred_cost = cost

        # Update the opt_model with estimated costs
        opt_model.setObj(cost)

        # Solutions with information asymmetry
        asym_interdictor = AsymmetricInterdictor(
            opt_model._graph, 
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
        opt_model.setObj(pred_cost + x_intd * interdiction)
        y_est, _ = opt_model.solve()

        # Store the results
        # true_objs.append(true_graph(y_true, interdictions=x_intd * interdiction))
        est_objs.append(opt_model._graph(y_est, interdictions=x_intd * interdiction))

        # Print progress
        print_progress(i, num_test_samples)

    # Evaluate performance
    # return {
    #     "true_objective": np.array(true_objs),
    #     "estimated_objective": np.array(est_objs),
    # }
    return np.array(est_objs)


def compare_wrong_asym_intd(
        cfg, 
        opt_model: 'ShortestPathGrb',
        test_data, 
        interdictions, 
        normalization_constant, 
        true_model, 
        false_model
    ):
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

        # Compute estimated cost if an estimator is provided. Otherwise use the true costs
        feature = test_data["feats"][i]
        true_pred_cost = true_model(torch.tensor(feature, dtype=torch.float32)).detach().numpy() * normalization_constant
        false_pred_cost = false_model(torch.tensor(feature, dtype=torch.float32)).detach().numpy() * normalization_constant

        # Update opt_model with true costs
        opt_model.setObj(cost)

        # Solutions with information asymmetry
        asym_interdictor = AsymmetricInterdictor(
            opt_model._graph,
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
        opt_model.setObj(true_pred_cost + x_intd * interdiction)
        y_est, _ = opt_model.solve()

        # Set correct cost to evaluate objectives
        opt_model.setObj(cost + x_intd * interdiction)

        # Store the results
        # true_objs.append(true_graph(y_true, interdictions=x_intd * interdiction))
        est_objs.append(opt_model._graph(y_est, interdictions=x_intd * interdiction))

        # Print progress
        print_progress(i, num_test_samples)

    # Evaluate performance
    # return {
    #     "true_objective": np.array(true_objs),
    #     "estimated_objective": np.array(est_objs),
    # }
    return np.array(est_objs)




