import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
RESULTS_DIR = "results"
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAPS = ["Nuerburgring_map", "Sochi_map", "Spa_map"]
LATENCIES = [0, 10]
STRATEGIES = ["round_robin", "sensitivity", "rl_simple", "rl_boot"]

# =========================
# LOAD DATA FUNCTION
# =========================
def load_csv(map_name, strategy, latency):
    filename = f"{map_name}_{strategy}_lat{latency}.csv"
    path = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(path):
        print(f"Missing file: {path}")
        return None

    df = pd.read_csv(path)
    return df["crosstrack_rmse_m"].values


# =========================
# MAIN PLOTTING
# =========================
for map_name in MAPS:
    for lat in LATENCIES:

        plt.figure()

        all_data = []

        for strat in STRATEGIES:
            rmse_values = load_csv(map_name, strat, lat)

            if rmse_values is None:
                all_data.append([float('nan')])
            else:
                all_data.append(rmse_values)

        # Boxplot (shows all 5 runs per strategy)
        plt.boxplot(all_data, labels=STRATEGIES)

        plt.title(f"{map_name} | Latency = {lat}")
        plt.xlabel("Strategy")
        plt.ylabel("Cross-track RMSE (m)")

        plt.grid(True)

        # Save figure
        out_path = os.path.join(
            OUTPUT_DIR, f"{map_name}_lat{lat}.pdf"
        )
        plt.savefig(out_path)
        plt.close()

        print(f"Saved: {out_path}")

print("All plots generated.")
