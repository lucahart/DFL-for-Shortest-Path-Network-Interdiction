
import numpy as np
from tabulate import tabulate

from dflintdpy.data.config import HP
from dflintdpy.scripts.asym_spni_single_sim import single_sim

# Initialize the configuration class
cfg = HP()
num_seeds = cfg.get("num_seeds")
compute_asym_intd = False

# List to store results
results = []

# Loop over different random seeds
for seed in range(num_seeds):
    # Keep track of simulation progress
    print("##########################################" +
          f"###### STARTING SIMULATION {seed+1} / {num_seeds} ######" +
          "##########################################")

    # Generate random seeds
    np.random.seed(seed)
    seed1, seed2, seed3 = np.random.randint(0, 150, 3).tolist()

    # Set random seeds
    cfg.set("random_seed", seed1)
    cfg.set("intd_seed", seed2)
    cfg.set("data_loader_seed", seed3)

    # Run the Asym_SPNI function
    prediction_mean_std, metrics, table_1, table_2 = single_sim(cfg, compute_asym_intd=compute_asym_intd)

    # Store values in list
    results.append({
        "seed": seed,
        "prediction_mean_std": prediction_mean_std,
        "metrics": metrics,
        "table_1": table_1,
        "table_2": table_2
    })

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
if compute_asym_intd:
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

# Mark final results
print("##########################################" + 
      "###### FINAL RESULTS OVER ALL SEEDS ######" + 
      "##########################################")

# Print prediction means and stds
print(f"Mean & std value comparison:")
print(f"\tTest:     {np.array(test_mean).mean():.4f} +/- {np.array(test_mean).std():.4f}")
print(f"\tTrain:    {np.array(train_mean).mean():.4f} +/- {np.array(train_mean).std():.4f}")
print(f"\tIntd:     {np.array(intd_mean).mean():.4f} +/- {np.array(intd_mean).std():.4f}")
print(f"\tPO:       {np.array(po_mean).mean():.4f} +/- {np.array(po_mean).std():.4f}")
print(f"\tSPO+:     {np.array(spo_mean).mean():.4f} +/- {np.array(spo_mean).std():.4f}")
print(f"\tSPO+ adv: {np.array(adv_spo_mean).mean():.4f}")

print(f"Std value comparison:")
print(f"\tTest:     {np.array(test_mean).std():.4f} +/- {np.array(test_std).mean():.4f}")
print(f"\tTrain:    {np.array(train_mean).std():.4f} +/- {np.array(train_std).mean():.4f}")
print(f"\tIntd:     {np.array(intd_mean).std():.4f} +/- {np.array(intd_std).mean():.4f}")
print(f"\tPO:       {np.array(po_mean).std():.4f} +/- {np.array(po_std).mean():.4f}")
print(f"\tSPO+:     {np.array(spo_mean).std():.4f} +/- {np.array(spo_std).mean():.4f}")
print(f"\tSPO+ adv: {np.array(adv_spo_mean).std():.4f} +/- {np.array(adv_spo_std).mean():.4f}")

# Print metrics
print(f"DFL no intd. improvement = {np.array(metric_1).mean():.4f}")
print(f"Adv. DFL no intd. improvement = {np.array(metric_2).mean():.4f}")
print(f"Adv. DFL sym. improvement = {np.array(metric_3).mean():.4f}")
print(f"Adv. DFL asym. improvement = {np.array(metric_4).mean():.4f}")
if compute_asym_intd:
    print(f"PO Asym. + Adv. Evader > Sym Asym. = {np.array(metric_5).mean():.4f}")

# Print table 1
table_headers = ["Predictor", "No Interdictor", "Sym. Interdictor", "Asym. Interdictor", "Asym. Intd. Assumes PO", "Asym. Intd. Assumes SPO", "Asym. Intd Assumes Adv. SPO"]
rows = [
    [
        "Oracle",
        f"{np.array(t1_o_n_mean).mean():.4f} +/- {np.array(t1_o_n_mean).std():.4f}",
        f"{np.array(t1_o_s_mean).mean():.4f} +/- {np.array(t1_o_s_mean).std():.4f}",
        f"{np.array(t1_o_a_mean).mean():.4f} +/- {np.array(t1_o_a_mean).std():.4f}",
    ], [
        "PO", 
        f"{np.array(t1_p_n_mean).mean():.4f} +/- {np.array(t1_p_n_mean).std():.4f}",
        f"{np.array(t1_p_s_mean).mean():.4f} +/- {np.array(t1_p_s_mean).std():.4f}", 
        f"{np.array(t1_p_a_mean).mean():.4f} +/- {np.array(t1_p_a_mean).std():.4f}",
    ], [
        "SPO", 
        f"{np.array(t1_s_n_mean).mean():.4f} +/- {np.array(t1_s_n_mean).std():.4f}", 
        f"{np.array(t1_s_s_mean).mean():.4f} +/- {np.array(t1_s_s_mean).std():.4f}", 
        f"{np.array(t1_s_a_mean).mean():.4f} +/- {np.array(t1_s_a_mean).std():.4f}", 
    ], [
        "SPO adv", 
        f"{np.array(t1_a_n_mean).mean():.4f} +/- {np.array(t1_a_n_mean).std():.4f}", 
        f"{np.array(t1_a_s_mean).mean():.4f} +/- {np.array(t1_a_s_mean).std():.4f}", 
        f"{np.array(t1_a_a_mean).mean():.4f} +/- {np.array(t1_a_a_mean).std():.4f}",
    ]
]
print(tabulate(rows, headers=table_headers, tablefmt="github"))

# Print table 2
if compute_asym_intd:
    table_headers = ["Predictor", "Asym. Intd. Assumes PO", "Asym. Intd. Assumes SPO", "Asym. Intd Assumes Adv. SPO"]
    rows = [
        [
            "Oracle", 
            "N/A", 
            "N/A",
            "N/A"
        ], [
            "PO", 
            f"{np.array(t1_p_a_mean).mean():.4f} +/- {np.array(t1_p_a_mean).std():.4f}",
            f"{np.array(t2_p_s_mean).mean():.4f} +/- {np.array(t2_p_s_mean).std():.4f}",
            f"{np.array(t2_p_a_mean).mean():.4f} +/- {np.array(t2_p_a_mean).std():.4f}"
        ], [
            "SPO", 
            f"{np.array(t2_s_p_mean).mean():.4f} +/- {np.array(t2_s_p_mean).std():.4f}",
            f"{np.array(t1_s_a_mean).mean():.4f} +/- {np.array(t1_s_a_mean).std():.4f}", 
            f"{np.array(t2_s_a_mean).mean():.4f} +/- {np.array(t2_s_a_mean).std():.4f}", 
        ], [
            "SPO adv", 
            f"{np.array(t2_a_p_mean).mean():.4f} +/- {np.array(t2_a_p_mean).std():.4f}", 
            f"{np.array(t2_a_s_mean).mean():.4f} +/- {np.array(t2_a_s_mean).std():.4f}",
            f"{np.array(t1_a_a_mean).mean():.4f} +/- {np.array(t1_a_a_mean).std():.4f}",
        ]
    ]
    print(tabulate(rows, headers=table_headers, tablefmt="github"))

print("Finished Simulations.")
