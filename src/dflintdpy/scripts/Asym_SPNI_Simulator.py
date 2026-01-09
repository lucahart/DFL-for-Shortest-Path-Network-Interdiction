import numpy as np
from pathlib import Path
from tabulate import tabulate

from dflintdpy.data.config import HP
from dflintdpy.scripts.asym_spni_single_sim import single_sim
from dflintdpy.utils.read_write_results import save_results_to_csv

# Initialize the configuration class
cfg = HP()
num_seeds = cfg.get("num_seeds")
compute_asym_intd_2 = False # Matrix comparison with lack of evader knowledge (table 2)
compute_asym_intd = True # Asym. Interdictor column in table 1

# List to store results
results = []

# Loop over different random seeds
for seed in range(num_seeds):
    # Keep track of simulation progress
    print("="*80)
    print(f"STARTING SIMULATION {seed+1} / {num_seeds}")
    print("="*80)

    # Generate random seeds
    np.random.seed(seed)
    seed1, seed2, seed3 = np.random.randint(0, 150, 3).tolist()

    # Set random seeds
    cfg.set("random_seed", seed1)
    cfg.set("intd_seed", seed2)
    cfg.set("data_loader_seed", seed3)

    # Run the Asym_SPNI function
    prediction_mean_std, metrics, table_1, table_2, all_data = single_sim(
         cfg, 
         compute_asym_intd_2=compute_asym_intd_2,
         compute_asym_intd=compute_asym_intd,
        #  load_real_world_graph='my_graph.csv'
    )

    # Store values in list
    results.append({
        "seed": seed,
        "prediction_mean_std": prediction_mean_std,
        "metrics": metrics,
        "table_1": table_1,
        "table_2": table_2,
        "all_data": all_data
    })

# Store final results in a csv file
output_path = Path(__file__).parent.parent.parent.parent / 'results' /\
    "results_train_{train}_valid_{valid}_test_{test}_m_{m}_n_{n}_deg_{deg}_noise_{noise}_seeds_{num_seeds}.csv"\
    .format(train=cfg.get("num_train_samples"),
            valid=cfg.get("num_val_samples"),
            test=cfg.get("num_test_samples"),
            m=cfg.get("grid_size")[0],
            n=cfg.get("grid_size")[1],
            deg=cfg.get("deg"),
            noise=cfg.get("noise_width"),
            num_seeds=num_seeds
    )
save_results_to_csv(results, output_path=output_path)

# Combine prediction means and stds
test_mean = []
train_mean = []
intd_mean = []
po_mean = []
spo_mean = []
adv_spo_mean = []
test_std = []
train_std = []
intd_std = []
po_std = []
spo_std = []
adv_spo_std = []
for result in results:
        test_mean.append(result['prediction_mean_std']['test_mean'])
        train_mean.append(result['prediction_mean_std']['train_mean'])
        intd_mean.append(result['prediction_mean_std']['intd_mean'])
        po_mean.append(result['prediction_mean_std']['po_mean'])
        spo_mean.append(result['prediction_mean_std']['spo_mean'])
        adv_spo_mean.append(result['prediction_mean_std']['adv_spo_mean'])
        test_std.append(result['prediction_mean_std']['test_std'])
        train_std.append(result['prediction_mean_std']['train_std'])
        intd_std.append(result['prediction_mean_std']['intd_std'])
        po_std.append(result['prediction_mean_std']['po_std'])
        spo_std.append(result['prediction_mean_std']['spo_std'])
        adv_spo_std.append(result['prediction_mean_std']['adv_spo_std'])

# Combine metrics
metric_1 = []
metric_2 = []
metric_3 = []
metric_4 = []
metric_5 = []
for result in results:
    metric_1.append(result['metrics']['metric_1'])
    metric_2.append(result['metrics']['metric_2'])
    metric_3.append(result['metrics']['metric_3'])
    metric_4.append(result['metrics']['metric_4'])
    metric_5.append(result['metrics']['metric_5'])

# Combine table 1
t1_o_n_mean = []
t1_o_s_mean = []
t1_o_s_std = []
t1_o_a_mean = []
t1_o_a_std = []

t1_p_n_mean = []
t1_p_s_mean = []
t1_p_s_std = []
t1_p_a_mean = []
t1_p_a_std = []

t1_s_n_mean = []
t1_s_s_mean = []
t1_s_s_std = []
t1_s_a_mean = []
t1_s_a_std = []

t1_a_n_mean = []
t1_a_s_mean = []
t1_a_s_std = []
t1_a_a_mean = []
t1_a_a_std = []
for result in results:
    t1_o_n_mean.append(result['table_1']['t1_o_n_mean'])
    t1_o_s_mean.append(result['table_1']['t1_o_s_mean'])
    t1_o_s_std.append(result['table_1']['t1_o_s_std'])
    t1_o_a_mean.append(result['table_1']['t1_o_a_mean'])
    t1_o_a_std.append(result['table_1']['t1_o_a_std'])

    t1_p_n_mean.append(result['table_1']['t1_p_n_mean'])
    t1_p_s_mean.append(result['table_1']['t1_p_s_mean'])
    t1_p_s_std.append(result['table_1']['t1_p_s_std'])
    t1_p_a_mean.append(result['table_1']['t1_p_a_mean'])
    t1_p_a_std.append(result['table_1']['t1_p_a_std'])

    t1_s_n_mean.append(result['table_1']['t1_s_n_mean'])
    t1_s_s_mean.append(result['table_1']['t1_s_s_mean'])
    t1_s_s_std.append(result['table_1']['t1_s_s_std'])
    t1_s_a_mean.append(result['table_1']['t1_s_a_mean'])
    t1_s_a_std.append(result['table_1']['t1_s_a_std'])

    t1_a_n_mean.append(result['table_1']['t1_a_n_mean'])
    t1_a_s_mean.append(result['table_1']['t1_a_s_mean'])
    t1_a_s_std.append(result['table_1']['t1_a_s_std'])
    t1_a_a_mean.append(result['table_1']['t1_a_a_mean'])
    t1_a_a_std.append(result['table_1']['t1_a_a_std'])

# Combine table 2
if compute_asym_intd_2:
    t2_p_s_mean = []
    t2_p_s_std = []
    t2_p_a_mean = []
    t2_p_a_std = []

    t2_s_p_mean = []
    t2_s_p_std = []
    t2_s_a_mean = []
    t2_s_a_std = []

    t2_a_p_mean = []
    t2_a_p_std = []
    t2_a_s_mean = []
    t2_a_s_std = []
    for result in results:
        t2_p_s_mean.append(result['table_2']['t2_p_s_mean'])
        t2_p_s_std.append(result['table_2']['t2_p_s_std'])
        t2_p_a_mean.append(result['table_2']['t2_p_a_mean'])
        t2_p_a_std.append(result['table_2']['t2_p_a_std'])

        t2_s_p_mean.append(result['table_2']['t2_s_p_mean'])
        t2_s_p_std.append(result['table_2']['t2_s_p_std'])
        t2_s_a_mean.append(result['table_2']['t2_s_a_mean'])
        t2_s_a_std.append(result['table_2']['t2_s_a_std'])

        t2_a_p_mean.append(result['table_2']['t2_a_p_mean'])
        t2_a_p_std.append(result['table_2']['t2_a_p_std'])
        t2_a_s_mean.append(result['table_2']['t2_a_s_mean'])
        t2_a_s_std.append(result['table_2']['t2_a_s_std'])

print("Finished Simulations.")
