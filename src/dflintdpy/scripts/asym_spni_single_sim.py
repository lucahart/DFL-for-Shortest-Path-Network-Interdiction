#####################
###### Imports ######
#####################

from pathlib import Path
from dflintdpy.utils.real_world_spni_data_handling import csv_to_graph
import numpy as np

import torch
import random
from tabulate import tabulate

from dflintdpy.models.grid import Grid
from dflintdpy.models.dgrid import DGrid
from dflintdpy.solvers.shortest_path_grb import ShortestPathGrb

from dflintdpy.scripts.compare import (compare_shortest_paths,
                                       compare_sym_intd, 
                                       compare_asym_intd,
                                       compare_wrong_asym_intd)
from dflintdpy.scripts.setup import (gen_data, 
                                     gen_train_data, 
                                     setup_pfl_predictor,
                                     setup_dfl_predictor)


def single_sim(cfg, visualize=False, compute_asym_intd_2=True, 
               compute_asym_intd=True, load_real_world_graph: str | None = None):
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
    # Retrieve root directly
    root_dir = Path(__file__).parent.parent.parent.parent

    # Define a graph with appropriate dimensions and an opt_model 
    # for solving the shortest path problem on the graph
    if load_real_world_graph is not None:
        file_path = root_dir / 'real_world_spni_data' / load_real_world_graph
        graph = csv_to_graph(file_path)
        cfg.set("grid_size", (graph.num_cost+1, 1))
        dir = root_dir / 'store_data'
    else:
        m, n = cfg.get("grid_size")
        graph = DGrid(m, n)
        dir = None # only set to none for DGrid
        # dir = root_dir / 'store_data'
    opt_model = ShortestPathGrb(graph)

    # Generate normalized training and testing data
    training_data_adverse, testing_data, normalization_constant = gen_train_data(
        cfg, 
        opt_model,
        path_dir=dir,
        interdiction_policy="adversarial",
    )

    training_data_random, _, _ = gen_train_data(
        cfg,
        opt_model,
        path_dir=dir,
        interdiction_policy="random",
    )

    print(f"Training PFL prediction model.")
    po_model = setup_pfl_predictor(
        cfg,
        graph,
        opt_model,
        training_data_adverse,
        verbose=visualize
    )

    print(f"Training A-DFL prediction model.")
    spo_model_adversarial = setup_dfl_predictor(
        cfg,
        graph,
        opt_model,
        training_data_adverse,
        verbose=visualize
    )

    print(f"Training R-DFL prediction model.")
    spo_model_random = setup_dfl_predictor(
        cfg,
        graph,
        opt_model,
        training_data_random,
        verbose=visualize
    )

    spo_epochs = cfg.get("spo_epochs")
    cfg.set("spo_epochs", spo_epochs * 2)

    # Generate normalized training and testing data
    # nonadv_training_data, _, _ = gen_train_data(cfg, opt_model)
    nonadv_t_data = training_data_adverse["train_loader"].get_nonadverse_loader()
    nonadv_v_data = training_data_adverse["val_loader"].get_nonadverse_loader()
    nonadv_training_data = {
        "train_loader": nonadv_t_data,
        "val_loader": nonadv_v_data
    }

    print(f"Training DFL prediction model.")
    spo_model_non_adverse = setup_dfl_predictor(
        cfg,
        graph,
        opt_model,
        nonadv_training_data,
        verbose=visualize
    )

    cfg.set("spo_epochs", spo_epochs)


    #########################################
    ##### Prediction Algorithm Analysis #####
    #########################################

    interdictions = gen_data(
        cfg, 
        opt_model=opt_model, 
        seed=cfg.get("intd_seed"), 
        normalization_constant=normalization_constant
    )

    if False: # Skipping this data
        # Comparison of different means of costs to show similar 
        print(f"Mean value comparison:")
        print(f"\tTest:     {testing_data['costs'].mean():.7f}")
        print(f"\tTrain:    {training_data_adverse['train_loader'].dataset.costs.mean():.7f}")
        print(f"\tIntd:     {interdictions['costs'].mean():.7f}")
        print(f"\tPO:       {po_model(torch.tensor(testing_data['feats'], dtype=torch.float32)).mean().item():.7f}")
        print(f"\tSPO+:     {spo_model_non_adverse(torch.tensor(testing_data['feats'], dtype=torch.float32)).mean().item():.7f}")
        print(f"\tSPO+ rnd: {spo_model_random(torch.tensor(testing_data['feats'], dtype=torch.float32)).mean().item():.7f}")
        print(f"\tSPO+ adv: {spo_model_adversarial(torch.tensor(testing_data['feats'], dtype=torch.float32)).mean().item():.7f}")

        print(f"Std value comparison:")
        print(f"\tTest:     {testing_data['costs'].std():.7f}")
        print(f"\tTrain:    {training_data_adverse['train_loader'].dataset.costs.std():.7f}")
        print(f"\tIntd:     {interdictions['costs'].std():.7f}")
        print(f"\tPO:       {po_model(torch.tensor(testing_data['feats'], dtype=torch.float32)).std().item():.7f}")
        print(f"\tSPO+:     {spo_model_non_adverse(torch.tensor(testing_data['feats'], dtype=torch.float32)).std().item():.7f}")
        print(f"\tSPO+ rnd: {spo_model_random(torch.tensor(testing_data['feats'], dtype=torch.float32)).std().item():.7f}")
        print(f"\tSPO+ adv: {spo_model_adversarial(torch.tensor(testing_data['feats'], dtype=torch.float32)).std().item():.7f}")

    ################################################
    ##### Compare Shortest Paths of PO and SPO #####
    ################################################

    true_objs, po_objs, spo_objs, adv_spo_objs = compare_shortest_paths(
        cfg, opt_model, po_model, spo_model_non_adverse, testing_data, spo_model_adversarial
    )
    _, _, _, rand_spo_objs = compare_shortest_paths(
        cfg, opt_model, po_model, spo_model_non_adverse, testing_data, spo_model_random
    )


    ###################################
    ##### Symmetric Interdictions #####
    ###################################
    all_pred_sym_intd = compare_sym_intd(
        cfg, 
        opt_model,
        po_model, 
        spo_model_non_adverse, 
        testing_data, 
        interdictions, 
        normalization_constant, 
        adfl_predictor=spo_model_adversarial,
        rand_adfl_predictor=spo_model_random
    )

    ####################################
    ##### Asymmetric Interdictions #####
    ####################################
    if compute_asym_intd:
        no_pred_asym_intd = compare_asym_intd(
            cfg, 
            opt_model,
            testing_data, 
            interdictions, 
            normalization_constant
        )

        po_pred_asym_intd_I = compare_asym_intd(
            cfg, 
            opt_model,
            testing_data, 
            interdictions, 
            normalization_constant, 
            po_model
        )

        spo_pred_asym_intd_I = compare_asym_intd(
            cfg, 
            opt_model,
            testing_data, 
            interdictions, 
            normalization_constant, 
            spo_model_non_adverse
        )

        rand_spo_pred_asym_intd_I = compare_asym_intd(
            cfg,
            opt_model,
            testing_data,
            interdictions,
            normalization_constant,
            spo_model_random
        )

        adv_spo_pred_asym_intd_I = compare_asym_intd(
            cfg, 
            opt_model,
            testing_data, 
            interdictions, 
            normalization_constant, 
            spo_model_adversarial
        )
    else:
        no_pred_asym_intd = np.zeros((cfg.get("num_test_samples"),))
        po_pred_asym_intd_I = np.zeros((cfg.get("num_test_samples"),))
        spo_pred_asym_intd_I = np.zeros((cfg.get("num_test_samples"),))
        rand_spo_pred_asym_intd_I = np.zeros((cfg.get("num_test_samples"),))
        adv_spo_pred_asym_intd_I = np.zeros((cfg.get("num_test_samples"),))

    ############################################################
    ##### Asymmetric Interdiction with wrong evader models #####
    ############################################################
    if not compute_asym_intd_2:
        print("Skipping asymmetric interdiction with wrong evader models...")
    if compute_asym_intd_2:
        true_nonadv_false_po_asym_intd = compare_wrong_asym_intd(
            cfg, 
            opt_model,
            testing_data, 
            interdictions, 
            normalization_constant, 
            true_model=spo_model_non_adverse, 
            false_model=po_model
        )

        true_po_false_nonadv_asym_intd = compare_wrong_asym_intd(
            cfg, 
            opt_model,
            testing_data, 
            interdictions, 
            normalization_constant, 
            true_model=po_model, 
            false_model=spo_model_non_adverse
        )

        true_spo_false_po_asym_intd = compare_wrong_asym_intd(
            cfg, 
            opt_model,
            testing_data, 
            interdictions, 
            normalization_constant, 
            true_model=spo_model_adversarial, 
            false_model=po_model
        )

        true_po_false_spo_asym_intd = compare_wrong_asym_intd(
            cfg, 
            opt_model,
            testing_data, 
            interdictions, 
            normalization_constant, 
            true_model=po_model, 
            false_model=spo_model_adversarial
        )

        true_adv_false_nonadv_asym_intd = compare_wrong_asym_intd(
            cfg, 
            opt_model,
            testing_data, 
            interdictions, 
            normalization_constant, 
            true_model=spo_model_adversarial, 
            false_model=spo_model_non_adverse
        )

        true_nonadv_false_adv_asym_intd = compare_wrong_asym_intd(
            cfg, 
            opt_model,
            testing_data, 
            interdictions, 
            normalization_constant, 
            true_model=spo_model_non_adverse, 
            false_model=spo_model_adversarial
        )

    ###########################################
    ##### Improvement Metrics and Results #####
    ###########################################

    true_mean = np.array(true_objs).mean() * normalization_constant
    po_mean = np.array(po_objs).mean() * normalization_constant
    spo_mean = np.array(spo_objs).mean() * normalization_constant
    rand_spo_mean = np.array(rand_spo_objs).mean() * normalization_constant
    adv_spo_mean = np.array(adv_spo_objs).mean() * normalization_constant

    print(f"DFL no intd. improvement = {po_mean - spo_mean:.2f}")
    print(f"DFL+Rnd no intd. improvement = {po_mean - rand_spo_mean:.2f}")
    print(f"Adv. DFL no intd. improvement = {po_mean - adv_spo_mean:.2f}")
    print(
        f"DFL+Rnd sym. improvement = "
        f"{all_pred_sym_intd['po_objective'].mean() - all_pred_sym_intd['rand_adv_spo_objective'].mean():.2f}"
    )
    print(
        f"Adv. DFL sym. improvement = "
        f"{all_pred_sym_intd['po_objective'].mean() - all_pred_sym_intd['adv_spo_objective'].mean():.2f}"
    )
    if compute_asym_intd:
        print(f"DFL+Rnd asym. improvement = {po_pred_asym_intd_I.mean() - rand_spo_pred_asym_intd_I.mean():.2f}")
        print(f"Adv. DFL asym. improvement = {po_pred_asym_intd_I.mean() - adv_spo_pred_asym_intd_I.mean():.2f}")
    if compute_asym_intd_2:
        print(
            "PO Asym. + Adv. Evader > Sym Asym. = "
            f"{true_po_false_spo_asym_intd.mean() - all_pred_sym_intd['adv_spo_objective'].mean():.2f}"
        )

    # Prepare no-interdiction results for printing
    true_mean = np.array(true_objs).mean() * normalization_constant
    po_mean = np.array(po_objs).mean() * normalization_constant
    spo_mean = np.array(spo_objs).mean() * normalization_constant
    rand_spo_mean = np.array(rand_spo_objs).mean() * normalization_constant
    adv_spo_mean = np.array(adv_spo_objs).mean() * normalization_constant
    true_std = np.array(true_objs).std() * normalization_constant
    po_std = np.array(po_objs).std() * normalization_constant
    spo_std = np.array(spo_objs).std() * normalization_constant
    rand_spo_std = np.array(rand_spo_objs).std() * normalization_constant
    adv_spo_std = np.array(adv_spo_objs).std() * normalization_constant

    # Print the results in a table format
    table_headers = ["Predictor", "No Interdictor", "Sym. Interdictor", "Asym. Interdictor", "Asym. Intd. Assumes PO", "Asym. Intd. Assumes SPO", "Asym. Intd Assumes Adv. SPO"]

    rows = [
        [
            "Oracle", 
            f"{true_mean:.4f} +/- {true_std:.4f}", 
            f"{all_pred_sym_intd['true_objective'].mean():.4f} +/- {all_pred_sym_intd['true_objective'].std():.4f}", 
            f"{no_pred_asym_intd.mean():.4f} +/- {no_pred_asym_intd.std():.4f}", 
            # "N/A", 
            # "N/A",
            # "N/A"
        ], [
            "PO", 
            f"{po_mean:.4f} +/- {po_std:.4f}",  
            f"{all_pred_sym_intd['po_objective'].mean():.4f} +/- {all_pred_sym_intd['po_objective'].std():.4f}", 
            f"{po_pred_asym_intd_I.mean():.4f} +/- {po_pred_asym_intd_I.std():.4f}", 
            # "", 
            # f"{true_po_false_nonadv_asym_intd.mean():.4f} +/- {true_po_false_nonadv_asym_intd.std():.4f}",
            # f"{true_po_false_spo_asym_intd.mean():.4f} +/- {true_po_false_spo_asym_intd.std():.4f}"
        ], [
            "SPO", 
            f"{spo_mean:.4f} +/- {spo_std:.4f}", 
            f"{all_pred_sym_intd['spo_objective'].mean():.4f} +/- {all_pred_sym_intd['spo_objective'].std():.4f}", 
            f"{spo_pred_asym_intd_I.mean():.4f} +/- {spo_pred_asym_intd_I.std():.4f}", 
            # f"{true_nonadv_false_po_asym_intd.mean():.4f} +/- {true_nonadv_false_po_asym_intd.std():.4f}",
            # "",
            # f"{true_nonadv_false_adv_asym_intd.mean():.4f} +/- {true_nonadv_false_adv_asym_intd.std():.4f}", 
        ], [
            "SPO rnd", 
            f"{rand_spo_mean:.4f} +/- {rand_spo_std:.4f}", 
            f"{all_pred_sym_intd['rand_adv_spo_objective'].mean():.4f} +/- {all_pred_sym_intd['rand_adv_spo_objective'].std():.4f}", 
            f"{rand_spo_pred_asym_intd_I.mean():.4f} +/- {rand_spo_pred_asym_intd_I.std():.4f}", 
        ], [
            "SPO adv", 
            f"{adv_spo_mean:.4f} +/- {adv_spo_std:.4f}", 
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
        'o_r': np.array(rand_spo_objs) * normalization_constant,
        'o_a': np.array(adv_spo_objs) * normalization_constant, 
        's_o': all_pred_sym_intd['true_objective'],
        's_p': all_pred_sym_intd['po_objective'],
        's_s': all_pred_sym_intd['spo_objective'],
        's_r': all_pred_sym_intd['rand_adv_spo_objective'],
        's_a': all_pred_sym_intd['adv_spo_objective'],
        'a_o': no_pred_asym_intd,  
        'a_p': po_pred_asym_intd_I, 
        'a_s': spo_pred_asym_intd_I, 
        'a_r': rand_spo_pred_asym_intd_I,
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
        "train_mean" : training_data_adverse['train_loader'].dataset.costs.mean(),
        "intd_mean" : interdictions['costs'].mean(),
        "po_mean" : po_model(torch.tensor(testing_data['feats'], dtype=torch.float32)).mean().item(),
        "spo_mean" : spo_model_non_adverse(torch.tensor(testing_data['feats'], dtype=torch.float32)).mean().item(),
        "rand_spo_mean" : spo_model_random(torch.tensor(testing_data['feats'], dtype=torch.float32)).mean().item(),
        "adv_spo_mean" : spo_model_adversarial(torch.tensor(testing_data['feats'], dtype=torch.float32)).mean().item(),
        "test_std" : testing_data['costs'].std(),
        "train_std" : training_data_adverse['train_loader'].dataset.costs.std(),
        "intd_std" : interdictions['costs'].std(),
        "po_std" : po_model(torch.tensor(testing_data['feats'], dtype=torch.float32)).std().item(),
        "spo_std" : spo_model_non_adverse(torch.tensor(testing_data['feats'], dtype=torch.float32)).std().item(),
        "rand_spo_std" : spo_model_random(torch.tensor(testing_data['feats'], dtype=torch.float32)).std().item(),
        "adv_spo_std" : spo_model_adversarial(torch.tensor(testing_data['feats'], dtype=torch.float32)).std().item()
    }

    metrics = {
        "metric_1" : po_mean - spo_mean,
        "metric_2" : po_mean - rand_spo_mean,
        "metric_3" : po_mean - adv_spo_mean,
        "metric_4" : all_pred_sym_intd['po_objective'].mean() - all_pred_sym_intd['rand_adv_spo_objective'].mean(),
        "metric_5" : all_pred_sym_intd['po_objective'].mean() - all_pred_sym_intd['adv_spo_objective'].mean(),
        "metric_6" : po_pred_asym_intd_I.mean() - rand_spo_pred_asym_intd_I.mean(),
        "metric_7" : po_pred_asym_intd_I.mean() - adv_spo_pred_asym_intd_I.mean(),
        "metric_8" : (
            true_po_false_spo_asym_intd.mean() - all_pred_sym_intd['adv_spo_objective'].mean()
            if compute_asym_intd_2 else None
        )
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

        "t1_r_n_mean" : rand_spo_mean,
        "t1_r_s_mean" : all_pred_sym_intd['rand_adv_spo_objective'].mean(),
        "t1_r_s_std" : all_pred_sym_intd['rand_adv_spo_objective'].std(),
        "t1_r_a_mean" : rand_spo_pred_asym_intd_I.mean(),
        "t1_r_a_std" : rand_spo_pred_asym_intd_I.std(),

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
