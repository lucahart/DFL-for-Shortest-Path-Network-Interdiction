#####################
###### Imports ######
#####################

from pathlib import Path
import numpy as np

import torch
import random
from tabulate import tabulate

from dflintdpy.models.grid import Grid
from dflintdpy.solvers.shortest_path_grb import ShortestPathGrb

from dflintdpy.scripts.compare import (compare_shortest_paths,
                                       compare_sym_intd, 
                                       compare_asym_intd,
                                       compare_wrong_asym_intd)
from dflintdpy.scripts.setup import (gen_data, 
                                     gen_train_data, 
                                     setup_pfl_predictor,
                                     setup_dfl_predictor)


def single_sim(cfg, visualize=False, compute_asym_intd_2=True, compute_asym_intd=True):
    ############################
    ###### Set Parameters ######
    ############################

    # Set the random seed for reproducibility
    np.random.seed(cfg.get("random_seed"))
    random.seed(cfg.get("random_seed"))
    torch.manual_seed(cfg.get("random_seed"))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.get("random_seed"))


    ##################################
    ##### Generate Network Data ######
    ##################################

    # Define a graph with appropriate dimensions and an opt_model 
    # for solving the shortest path problem on the graph
    m,n = cfg.get("grid_size")
    graph = Grid(m,n)
    opt_model = ShortestPathGrb(graph)

    # Generate normalized training and testing data
    training_data, testing_data, normalization_constant = gen_train_data(
        cfg, 
        opt_model,
        path_dir=Path(__file__).parent.parent.parent.parent / 'store_data' 
    )

    cfg.set("po_epochs", 150)
    cfg.set("spo_epochs", 50)

    po_model = setup_pfl_predictor(
        cfg,
        graph,
        opt_model,
        training_data,
        versatile=visualize
    )

    spo_model = setup_dfl_predictor(
        cfg,
        graph,
        opt_model,
        training_data,
        versatile=visualize
    )

    cfg.set("num_scenarios", 1)

    # Generate normalized training and testing data
    training_data_non_adverse, testing_data_non_adverse, normalization_constant = gen_train_data(cfg, opt_model)

    spo_model_non_adverse = setup_dfl_predictor(
        cfg,
        graph,
        opt_model,
        training_data_non_adverse,
        versatile=visualize
    )

    cfg.set("num_scenarios", 10)


    #########################################
    ##### Prediction Algorithm Analysis #####
    #########################################

    interdictions = gen_data(cfg, seed=cfg.get("intd_seed"), normalization_constant=normalization_constant)

    # Comparison of different means of costs to show similar 
    print(f"Mean value comparison:")
    print(f"\tTest:     {testing_data['costs'].mean():.7f}")
    print(f"\tTrain:    {training_data['train_loader'].dataset.costs.mean():.7f}")
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

    true_objs, po_objs, spo_objs, adv_spo_objs = compare_shortest_paths(cfg, opt_model, po_model, spo_model_non_adverse, testing_data, spo_model)


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
        'o_o': true_objs, 
        'o_p': po_objs, 
        'o_s': spo_objs, 
        'o_a': adv_spo_objs, 
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

    return prediction_mean_std, metrics, table_1, table_2, all_data
