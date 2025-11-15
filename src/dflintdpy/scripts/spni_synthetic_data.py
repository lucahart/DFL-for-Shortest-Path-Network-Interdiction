from dflintdpy.data.adverse.adverse_data_generator import AdvDataGenerator
from dflintdpy.data.adverse.adverse_loader import AdvLoader
from dflintdpy.data.config import HP
from dflintdpy.solvers.shortest_path_grb import ShortestPathGrb
from dflintdpy.utils.dfl_trainer import DFLTrainer
from dflintdpy.utils.pfl_trainer import PFLTrainer
import numpy as np
import pandas as pd
import pyepo
import torch
import torch.optim as optim
from pathlib import Path
from typing import Tuple, Optional, Dict
import warnings

from dflintdpy.solvers.portfolio_optimization import PortfolioOptimization
from dflintdpy.solvers.bilevel_pricing import BilevelPricingProblem
from dflintdpy.models.grid import Grid
from dflintdpy.data.adverse.adverse_dataset import AdvDataset, generate_opt_dataset
from dflintdpy.predictors.linear_regression import LinearRegression
from dflintdpy.solvers.fast_solvers.fast_pricing_solver import FastBilevelPricingSolver
from dflintdpy.scripts.read_synthetic_data import read_synthetic_data
from dflintdpy.scripts.setup import gen_train_data
from dflintdpy.scripts.evaluate_models import evaluate_multiple_samples, summarize_results, print_summary

# Try to import optional dependencies
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    warnings.warn("CVXPY not available. Some methods will not work.")

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False
    warnings.warn("Gurobi not available. MIQCP method will not work.")

def simple_dfp_example(N = 100, noise=.5, deg=8, batch_size=32):
    """Simple example of solving a bilevel pricing problem with dfl data."""

    cfg = HP()

    ################## DataReading
    read_synthetic_data(N, noise, deg, file="SyntheticSPData")
    x_train, y_train, x_valid, y_valid, x_test, y_test = \
    read_synthetic_data(N, noise, deg, file="SyntheticSPData")

    num_test_samples = N

    ################## ModelCreation
    graph = Grid(m=5,n=5)
    opt_model = ShortestPathGrb(graph)

    ################## Generate Interdiction Data
    # Create adversarial data generator instance
    adverse_generator = AdvDataGenerator(
        cfg, 
        opt_model, 
        budget=0.3 * y_train[0,:].sum(),
        normalization_constant=1.0,
        num_scenarios=1,
        adverse_problem="SPNI",
    )
    # Generate adversarial training and validation data
    adfl_x_train, adfl_y_train, p_train = adverse_generator.generate(
        x_train, 
        y_train,
        file_path=Path(__file__).parent.parent.parent.parent / 'store_data' /\
            "p_train_N_{N}_noise_{noise}_deg_{deg}.csv" \
            .format(num_test_samples=num_test_samples, N=N, noise=noise, deg=deg)
    )
    adfl_x_valid, adfl_y_valid, p_valid = adverse_generator.generate(
        x_valid, 
        y_valid,
        file_path=Path(__file__).parent.parent.parent.parent / 'store_data' /\
            "p_valid_N_{N}_noise_{noise}_deg_{deg}.csv" \
            .format(num_test_samples=num_test_samples, N=N, noise=noise, deg=deg)
    )
    # Create adversarial datasets for training and validation
    dataset_train_adverse = AdvDataset(opt_model, adfl_x_train, 
                                       adfl_y_train, p_train)
    dataset_valid_adverse = AdvDataset(opt_model, adfl_x_valid, adfl_y_valid, p_valid)

    # Create data loaders for training and validation
    loader_train_adverse = AdvLoader(
        dataset_train_adverse,
        batch_size=cfg.get("batch_size"),
        seed=cfg.get("loader_seed"),
        shuffle=True,
    )
    loader_valid_adverse = AdvLoader(
        dataset_valid_adverse,
        batch_size=cfg.get("batch_size"),
        seed=cfg.get("loader_seed"),
        shuffle=False,
    )

    ################## DataLoader
    dataset_train = generate_opt_dataset(opt_model, x_train, y_train)
    dataset_valid = generate_opt_dataset(opt_model, x_valid, y_valid)
    dataset_test = generate_opt_dataset(opt_model, x_test, y_test)

    loader_train = AdvLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_valid = AdvLoader(dataset_valid, batch_size=batch_size, shuffle=False)
    loader_test = AdvLoader(dataset_test, batch_size=batch_size, shuffle=False)

    ################## Train simple model
    # Instantiate linear regression model
    pred_model_dfl = LinearRegression(
        num_feat=x_train.shape[1], 
        num_edges=opt_model.num_cost
    )
    pred_model_pfl = LinearRegression(
        num_feat=x_train.shape[1], 
        num_edges=opt_model.num_cost
    )
    pred_model_adfl = LinearRegression(
        num_feat=x_train.shape[1], 
        num_edges=opt_model.num_cost
    )

    # Init SPO+ loss
    dfl_loss = pyepo.func.SPOPlus(opt_model, processes=1)
    pfl_loss = torch.nn.MSELoss()
    adfl_loss = pyepo.func.SPOPlus(opt_model, processes=1)

    # Init optimizer
    optimizer = optim.Adam(pred_model_dfl.parameters(), lr=1e-2)
    optimizer_pfl = optim.Adam(pred_model_pfl.parameters(), lr=1e-2)
    optimizer_adfl = optim.Adam(pred_model_adfl.parameters(), lr=1e-2)

    # Set the number of epochs for training
    epochs = 10

    # Create a trainer instance
    dfl_trainer = DFLTrainer(
        pred_model=pred_model_dfl, 
        opt_model=opt_model,
        optimizer=optimizer, 
        loss_fn=dfl_loss
    )
    pfl_trainer = PFLTrainer(
        pred_model=pred_model_pfl,
        opt_model=opt_model,
        optimizer=optimizer_pfl,
        loss_fn=pfl_loss
    )
    adfl_trainer = DFLTrainer(
        pred_model=pred_model_adfl, 
        opt_model=opt_model,
        optimizer=optimizer_adfl, 
        loss_fn=adfl_loss
    )

    # Train the models
    dfl_train_loss_log, dfl_train_regret_log, dfl_val_loss_log, dfl_val_regret_log = \
        dfl_trainer.fit(loader_train, loader_valid, epochs=epochs)
    pfl_train_loss_log, pfl_train_regret_log, pfl_val_loss_log, pfl_val_regret_log = \
        pfl_trainer.fit(loader_train, loader_valid, epochs=epochs)
    adfl_train_loss_log, adfl_train_regret_log, adfl_val_loss_log, adfl_val_regret_log = \
        adfl_trainer.fit(loader_train_adverse, loader_valid_adverse, epochs=epochs)
    
    # Visualize learning curves
    DFLTrainer.vis_learning_curve(
        dfl_trainer,
        dfl_train_loss_log,
        dfl_train_regret_log,
        dfl_val_loss_log,
        dfl_val_regret_log,
        file_name="figures/dfl_learning_curve"
    )
    PFLTrainer.vis_learning_curve(
        pfl_trainer,
        pfl_train_loss_log,
        pfl_train_regret_log,
        pfl_val_loss_log,
        pfl_val_regret_log,
        file_name="figures/pfl_learning_curve"
    )
    DFLTrainer.vis_learning_curve(
        adfl_trainer,
        adfl_train_loss_log,
        adfl_train_regret_log,
        adfl_val_loss_log,
        adfl_val_regret_log,
        file_name="figures/adfl_learning_curve"
    )

    # Print final regrets

    print("DFL: Final regret on validation set:   ", dfl_val_regret_log[-1])
    print("PFL: Final regret on validation set:   ", pfl_val_regret_log[-1])
    print("A-DFL: Final regret on validation set: ", adfl_val_regret_log[-1])

    # Evaluate on test set
    loader_test.adverse_mode()
    _, dfl_test_regret = dfl_trainer.evaluate(loader_test)
    print("DFL: Final regret on test set: ", dfl_test_regret)
    loader_test.normal_mode()
    _, pfl_test_regret = pfl_trainer.evaluate(loader_test)
    print("PFL: Final regret on test set: ", pfl_test_regret)
    loader_test.adverse_mode()
    _, adfl_test_regret = adfl_trainer.evaluate(loader_test)
    print("A-DFL: Final regret on test set: ", adfl_test_regret)

        # Comparison of different means of costs to show similar 
    print(f"Mean value comparison:")
    print(f"\tTest:     {x_test.mean():.7f}")
    print(f"\tTrain:    {loader_train.dataset.costs.mean():.7f}")
    print(f"\tIntd:     {interdictions['costs'].mean():.7f}")
    print(f"\tPO:       {po_model(torch.tensor(testing_data['feats'], dtype=torch.float32)).mean().item():.7f}")
    print(f"\tSPO+:     {spo_model_non_adverse(torch.tensor(testing_data['feats'], dtype=torch.float32)).mean().item():.7f}")
    print(f"\tSPO+ adv: {spo_model(torch.tensor(testing_data['feats'], dtype=torch.float32)).std().item():.7f}")

    print(f"Std value comparison:")
    print(f"\tTest:     {testing_data['costs'].std():.7f}")
    print(f"\tTrain:    {training_data['train_loader'].dataset.costs.std():.7f}")
    print(f"\tIntd:     {interdictions['costs'].std():.7f}")
    print(f"\tPO:       {po_model(torch.tensor(testing_data['feats'], dtype=torch.float32)).std().item():.7f}")
    print(f"\tSPO+:     {spo_model_non_adverse(torch.tensor(testing_data['feats'], dtype=torch.float32)).std().item():.7f}")
    print(f"\tSPO+ adv: {spo_model(torch.tensor(testing_data['feats'], dtype=torch.float32)).std().item():.7f}")

    ################################################
    ##### Compare Shortest Paths of PO and SPO #####
    ################################################

    true_objs, po_objs, spo_objs, adv_spo_objs = \
        compare_shortest_paths(cfg, opt_model, po_model, spo_model_non_adverse, testing_data, spo_model)


    ###################################
    ##### Symmetric Interdictions #####
    ###################################
    all_pred_sym_intd = compare_sym_intd(
        cfg, 
        po_model, 
        spo_model_non_adverse, 
        testing_data, 
        interdictions, 
        normalization_constant, 
        adv_spo_model=spo_model
    )

    ####################################
    ##### Asymmetric Interdictions #####
    ####################################
    if compute_asym_intd:
        no_pred_asym_intd = compare_asym_intd(
            cfg, 
            testing_data, 
            interdictions, 
            normalization_constant
        )

        po_pred_asym_intd_I = compare_asym_intd(
            cfg, 
            testing_data, 
            interdictions, 
            normalization_constant, 
            po_model
        )

        spo_pred_asym_intd_I = compare_asym_intd(
            cfg, 
            testing_data, 
            interdictions, 
            normalization_constant, 
            spo_model_non_adverse
        )

        adv_spo_pred_asym_intd_I = compare_asym_intd(
            cfg, 
            testing_data, 
            interdictions, 
            normalization_constant, 
            spo_model
        )
    else:
        no_pred_asym_intd = np.zeros((cfg.get("num_test_samples"),))
        po_pred_asym_intd_I = np.zeros((cfg.get("num_test_samples"),))
        spo_pred_asym_intd_I = np.zeros((cfg.get("num_test_samples"),))
        adv_spo_pred_asym_intd_I = np.zeros((cfg.get("num_test_samples"),))

    ############################################################
    ##### Asymmetric Interdiction with wrong evader models #####
    ############################################################
    if not compute_asym_intd_2:
        print("Skipping asymmetric interdiction with wrong evader models...")
    if compute_asym_intd_2:
        true_nonadv_false_po_asym_intd = compare_wrong_asym_intd(
            cfg, 
            testing_data, 
            interdictions, 
            normalization_constant, 
            true_model=spo_model_non_adverse, 
            false_model=po_model
        )

        true_po_false_nonadv_asym_intd = compare_wrong_asym_intd(
            cfg, 
            testing_data, 
            interdictions, 
            normalization_constant, 
            true_model=po_model, 
            false_model=spo_model_non_adverse
        )

        true_spo_false_po_asym_intd = compare_wrong_asym_intd(
            cfg, 
            testing_data, 
            interdictions, 
            normalization_constant, 
            true_model=spo_model, 
            false_model=po_model
        )

        true_po_false_spo_asym_intd = compare_wrong_asym_intd(
            cfg, 
            testing_data, 
            interdictions, 
            normalization_constant, 
            true_model=po_model, 
            false_model=spo_model
        )

        true_adv_false_nonadv_asym_intd = compare_wrong_asym_intd(
            cfg, 
            testing_data, 
            interdictions, 
            normalization_constant, 
            true_model=spo_model, 
            false_model=spo_model_non_adverse
        )

        true_nonadv_false_adv_asym_intd = compare_wrong_asym_intd(
            cfg, 
            testing_data, 
            interdictions, 
            normalization_constant, 
            true_model=spo_model_non_adverse, 
            false_model=spo_model
        )

    ###########################################
    ##### Improvement Metrics and Results #####
    ###########################################

    true_mean = np.array(true_objs).mean() * normalization_constant
    po_mean = np.array(po_objs).mean() * normalization_constant
    spo_mean = np.array(spo_objs).mean() * normalization_constant
    adv_spo_mean = np.array(adv_spo_objs).mean() * normalization_constant

    print(f"DFL no intd. improvement = {po_mean - spo_mean:.2f}")
    print(f"Adv. DFL no intd. improvement = {po_mean - adv_spo_mean:.2f}")
    print(f"Adv. DFL sym. improvement = {all_pred_sym_intd['po_objective'].mean() - all_pred_sym_intd['adv_spo_objective'].mean():.2f}")
    if compute_asym_intd_2:
        print(f"Adv. DFL asym. improvement = {po_pred_asym_intd_I.mean() - adv_spo_pred_asym_intd_I.mean():.2f}")
        print(f"PO Asym. + Adv. Evader > Sym Asym. = {true_po_false_spo_asym_intd.mean() - all_pred_sym_intd['adv_spo_objective'].mean():.2f}")

    # Prepare no-interdiction results for printing
    true_mean = np.array(true_objs).mean() * normalization_constant
    po_mean = np.array(po_objs).mean() * normalization_constant
    spo_mean = np.array(spo_objs).mean() * normalization_constant
    adv_spo_mean = np.array(adv_spo_objs).mean() * normalization_constant

    # Print the results in a table format
    table_headers = ["Predictor", "No Interdictor", "Sym. Interdictor", "Asym. Interdictor", "Asym. Intd. Assumes PO", "Asym. Intd. Assumes SPO", "Asym. Intd Assumes Adv. SPO"]

    rows = [
        [
            "Oracle", 
            f"{true_mean:.4f}", 
            f"{all_pred_sym_intd['true_objective'].mean():.4f} +/- {all_pred_sym_intd['true_objective'].std():.4f}", 
            f"{no_pred_asym_intd.mean():.4f} +/- {no_pred_asym_intd.std():.4f}", 
            # "N/A", 
            # "N/A",
            # "N/A"
        ], [
            "PO", 
            f"{po_mean:.4f} ",  
            f"{all_pred_sym_intd['po_objective'].mean():.4f} +/- {all_pred_sym_intd['po_objective'].std():.4f}", 
            f"{po_pred_asym_intd_I.mean():.4f} +/- {po_pred_asym_intd_I.std():.4f}", 
            # "", 
            # f"{true_po_false_nonadv_asym_intd.mean():.4f} +/- {true_po_false_nonadv_asym_intd.std():.4f}",
            # f"{true_po_false_spo_asym_intd.mean():.4f} +/- {true_po_false_spo_asym_intd.std():.4f}"
        ], [
            "SPO", 
            f"{spo_mean:.4f} ", 
            f"{all_pred_sym_intd['spo_objective'].mean():.4f} +/- {all_pred_sym_intd['spo_objective'].std():.4f}", 
            f"{spo_pred_asym_intd_I.mean():.4f} +/- {spo_pred_asym_intd_I.std():.4f}", 
            # f"{true_nonadv_false_po_asym_intd.mean():.4f} +/- {true_nonadv_false_po_asym_intd.std():.4f}",
            # "",
            # f"{true_nonadv_false_adv_asym_intd.mean():.4f} +/- {true_nonadv_false_adv_asym_intd.std():.4f}", 
        ], [
            "SPO adv", 
            f"{adv_spo_mean:.4f} ", 
            f"{all_pred_sym_intd['adv_spo_objective'].mean():.4f} +/- {all_pred_sym_intd['adv_spo_objective'].std():.4f}", 
            f"{adv_spo_pred_asym_intd_I.mean():.4f} +/- {adv_spo_pred_asym_intd_I.std():.4f}", 
            # f"{true_spo_false_po_asym_intd.mean():.4f} +/- {true_spo_false_po_asym_intd.std():.4f}", 
            # f"{true_adv_false_nonadv_asym_intd.mean():.4f} +/- {true_adv_false_nonadv_asym_intd.std():.4f}",
            # ""
        ]
    ]
    print(tabulate(rows, headers=table_headers, tablefmt="github"))

    print("\n")

    if compute_asym_intd_2:
        table_headers = ["Predictor", "Asym. Intd. Assumes PO", "Asym. Intd. Assumes SPO", "Asym. Intd Assumes Adv. SPO"]

        rows = [
            [
                "Oracle", 
                # f"{true_mean:.4f}", 
                # f"{all_pred_sym_intd['true_objective'].mean():.4f} +/- {all_pred_sym_intd['true_objective'].std():.4f}", 
                # f"{no_pred_asym_intd.mean():.4f} +/- {no_pred_asym_intd.std():.4f}", 
                "N/A", 
                "N/A",
                "N/A"
            ], [
                "PO", 
                # f"{po_mean:.4f} ",  
                # f"{all_pred_sym_intd['po_objective'].mean():.4f} +/- {all_pred_sym_intd['po_objective'].std():.4f}", 
                # f"{po_pred_asym_intd_I.mean():.4f} +/- {po_pred_asym_intd_I.std():.4f}", 
                "", 
                f"{true_po_false_nonadv_asym_intd.mean():.4f} +/- {true_po_false_nonadv_asym_intd.std():.4f}",
                f"{true_po_false_spo_asym_intd.mean():.4f} +/- {true_po_false_spo_asym_intd.std():.4f}"
            ], [
                "SPO", 
                # f"{spo_mean:.4f} ", 
                # f"{all_pred_sym_intd['spo_objective'].mean():.4f} +/- {all_pred_sym_intd['spo_objective'].std():.4f}", 
                # f"{spo_pred_asym_intd_I.mean():.4f} +/- {spo_pred_asym_intd_I.std():.4f}", 
                f"{true_nonadv_false_po_asym_intd.mean():.4f} +/- {true_nonadv_false_po_asym_intd.std():.4f}",
                "",
                f"{true_nonadv_false_adv_asym_intd.mean():.4f} +/- {true_nonadv_false_adv_asym_intd.std():.4f}", 
            ], [
                "SPO adv", 
                # f"{adv_spo_mean:.4f} ", 
                # f"{all_pred_sym_intd['adv_spo_objective'].mean():.4f} +/- {all_pred_sym_intd['adv_spo_objective'].std():.4f}", 
                # f"{adv_spo_pred_asym_intd_I.mean():.4f} +/- {adv_spo_pred_asym_intd_I.std():.4f}", 
                f"{true_spo_false_po_asym_intd.mean():.4f} +/- {true_spo_false_po_asym_intd.std():.4f}", 
                f"{true_adv_false_nonadv_asym_intd.mean():.4f} +/- {true_adv_false_nonadv_asym_intd.std():.4f}",
                ""
            ]
        ]
        print(tabulate(rows, headers=table_headers, tablefmt="github"))



    ##########################
    ##### Return Results #####
    ##########################

    all_data = {
        # x_y means: x: follower, y: leader
        # a_x_y means: x: true follower model, y: follower model assumed by leader
        'o_o': np.array(true_objs) * normalization_constant, 
        'o_p': np.array(po_objs) * normalization_constant, 
        'o_s': np.array(spo_objs) * normalization_constant, 
        'o_a': np.array(adv_spo_objs) * normalization_constant, 
        's_o': all_pred_sym_intd['true_objective'],
        's_p': all_pred_sym_intd['po_objective'],
        's_s': all_pred_sym_intd['spo_objective'],
        's_a': all_pred_sym_intd['adv_spo_objective'],
        'a_o': no_pred_asym_intd,  
        'a_p': po_pred_asym_intd_I, 
        'a_s': spo_pred_asym_intd_I, 
        'a_a': adv_spo_pred_asym_intd_I,
    }
    if compute_asym_intd_2:
        all_data.update({
            'a_s_p': true_nonadv_false_po_asym_intd,
            'a_p_s': true_po_false_nonadv_asym_intd, 
            'a_a_p': true_spo_false_po_asym_intd, 
            'a_p_a': true_po_false_spo_asym_intd, 
            'a_a_s': true_adv_false_nonadv_asym_intd, 
            'a_s_a': true_nonadv_false_adv_asym_intd
        })

    prediction_mean_std = {
        "test_mean" : testing_data['costs'].mean(),
        "train_mean" : training_data['train_loader'].dataset.costs.mean(),
        "intd_mean" : interdictions['costs'].mean(),
        "po_mean" : po_model(torch.tensor(testing_data['feats'], dtype=torch.float32)).mean().item(),
        "spo_mean" : spo_model_non_adverse(torch.tensor(testing_data['feats'], dtype=torch.float32)).mean().item(),
        "adv_spo_mean" : spo_model(torch.tensor(testing_data['feats'], dtype=torch.float32)).std().item(),
        "test_std" : testing_data['costs'].std(),
        "train_std" : training_data['train_loader'].dataset.costs.std(),
        "intd_std" : interdictions['costs'].std(),
        "po_std" : po_model(torch.tensor(testing_data['feats'], dtype=torch.float32)).std().item(),
        "spo_std" : spo_model_non_adverse(torch.tensor(testing_data['feats'], dtype=torch.float32)).std().item(),
        "adv_spo_std" : spo_model(torch.tensor(testing_data['feats'], dtype=torch.float32)).std().item()
    }

    metrics = {
        "metric_1" : po_mean - spo_mean,
        "metric_2" : po_mean - adv_spo_mean,
        "metric_3" : all_pred_sym_intd['po_objective'].mean() - all_pred_sym_intd['adv_spo_objective'].mean(),
        "metric_4" : po_pred_asym_intd_I.mean() - adv_spo_pred_asym_intd_I.mean(),
        "metric_5" : true_po_false_spo_asym_intd.mean() - all_pred_sym_intd['adv_spo_objective'].mean() if compute_asym_intd_2 else None
    }

    table_1 = {
        "t1_o_n_mean" : true_mean,
        "t1_o_s_mean" : all_pred_sym_intd['true_objective'].mean(),
        "t1_o_s_std" : all_pred_sym_intd['true_objective'].std(),
        "t1_o_a_mean" : no_pred_asym_intd.mean(),
        "t1_o_a_std" : no_pred_asym_intd.std(),

        "t1_p_n_mean" : po_mean,
        "t1_p_s_mean" : all_pred_sym_intd['po_objective'].mean(),
        "t1_p_s_std" : all_pred_sym_intd['po_objective'].std(),
        "t1_p_a_mean" : po_pred_asym_intd_I.mean(),
        "t1_p_a_std" : po_pred_asym_intd_I.std(),

        "t1_s_n_mean" : spo_mean,
        "t1_s_s_mean" : all_pred_sym_intd['spo_objective'].mean(),
        "t1_s_s_std" : all_pred_sym_intd['spo_objective'].std(),
        "t1_s_a_mean" : spo_pred_asym_intd_I.mean(),
        "t1_s_a_std" : spo_pred_asym_intd_I.std(),

        "t1_a_n_mean" : adv_spo_mean,
        "t1_a_s_mean" : all_pred_sym_intd['adv_spo_objective'].mean(),
        "t1_a_s_std" : all_pred_sym_intd['adv_spo_objective'].std(),
        "t1_a_a_mean" : adv_spo_pred_asym_intd_I.mean(),
        "t1_a_a_std" : adv_spo_pred_asym_intd_I.std()
    }

    table_2 = {
        "t2_p_s_mean" : true_po_false_nonadv_asym_intd.mean(),
        "t2_p_s_std" : true_po_false_nonadv_asym_intd.std(),
        "t2_p_a_mean" : true_po_false_spo_asym_intd.mean(),
        "t2_p_a_std" : true_po_false_spo_asym_intd.std(),

        "t2_s_p_mean" : true_nonadv_false_po_asym_intd.mean(),
        "t2_s_p_std" : true_nonadv_false_po_asym_intd.std(),
        "t2_s_a_mean" : true_nonadv_false_adv_asym_intd.mean(),
        "t2_s_a_std" : true_nonadv_false_adv_asym_intd.std(),

        "t2_a_p_mean" : true_spo_false_po_asym_intd.mean(),
        "t2_a_p_std" : true_spo_false_po_asym_intd.std(),
        "t2_a_s_mean" : true_adv_false_nonadv_asym_intd.mean(),
        "t2_a_s_std" : true_adv_false_nonadv_asym_intd.std()
    } if compute_asym_intd_2 else {}


    
    pass



if __name__ == "__main__":
    simple_dfp_example()