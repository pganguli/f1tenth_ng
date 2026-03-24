import re
import os
import csv
from collections import defaultdict

INPUT_LOG = "/Users/tingan/f1tenth_ng/packages/f110_scripts/src/f110_scripts/sim/slurm_logs/f1tenth_exp_38463301.out"
OUTPUT_DIR = "results_extracted"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Patterns
pattern_header = re.compile(
    r"Map:\s*(.*?)\s*\|\s*Latency:\s*(\d+)\s*\|\s*Strategy:\s*(\w+)"
)
pattern_run = re.compile(r"Run\s+(\d+)")
pattern_rmse = re.compile(r'"crosstrack_max_m":\s*([0-9.]+)')

# Storage: key → list of (run, value)
data = defaultdict(list)

current_map = None
current_latency = None
current_strategy = None
current_run = None

waiting_for_value = False

with open(INPUT_LOG, "r") as f:
    for line in f:

        # ---- Detect configuration ----
        header_match = pattern_header.search(line)
        if header_match:
            current_map = header_match.group(1)
            current_latency = int(header_match.group(2))
            current_strategy = header_match.group(3)
            continue

        # ---- Detect run ----
        run_match = pattern_run.search(line)
        if run_match:
            current_run = int(run_match.group(1))
            waiting_for_value = True
            continue

        # ---- Extract RMSE (ONLY first after run) ----
        if waiting_for_value:
            rmse_match = pattern_rmse.search(line)
            if rmse_match:
                value = float(rmse_match.group(1))

                key = (current_map, current_strategy, current_latency)
                data[key].append((current_run, value))

                waiting_for_value = False

# ---- Write separate CSV files ----
for (map_name, strategy, latency), runs in data.items():

    filename = f"{map_name}_{strategy}_lat{latency}.csv"
    path = os.path.join(OUTPUT_DIR, filename)

    # Sort by run index (important!)
    runs.sort(key=lambda x: x[0])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "crosstrack_max_m"])

        for run_id, value in runs:
            writer.writerow([run_id, value])

    print(f"Saved: {path}")

print("All CSV files generated.")