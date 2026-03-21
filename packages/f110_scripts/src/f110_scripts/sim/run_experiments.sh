#!/bin/bash

# =========================
# CONFIG
# =========================
PY_SCRIPT="packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py"
OUTPUT_DIR="results"
N_RUNS=1   # keep small for debugging

# Maps + waypoints (paired by index)
MAPS=(
"data/maps/F1/Nuerburgring/Nuerburgring_map"
"data/maps/F1/Sochi/Sochi_map"
"data/maps/F1/Spa/Spa_map"
)

WAYPOINTS=(
"data/maps/F1/Nuerburgring/Nuerburgring_centerline.tsv"
"data/maps/F1/Sochi/Sochi_centerline.tsv"
"data/maps/F1/Spa/Spa_centerline.tsv"
)

# Latencies
LATENCIES=(0 10)

# Strategies
STRATEGIES=("round_robin" "sensitivity" "rl_simple" "rl_boot")

# =========================
# INIT
# =========================
mkdir -p ${OUTPUT_DIR}
echo "Running in: $(pwd)"

# =========================
# MAIN LOOP
# =========================
for idx in ${!MAPS[@]}; do
    MAP=${MAPS[$idx]}
    WP=${WAYPOINTS[$idx]}
    MAP_NAME=$(basename $MAP)

    for LAT in "${LATENCIES[@]}"; do
        for STRAT in "${STRATEGIES[@]}"; do

            echo "----------------------------------------"
            echo "Map: $MAP_NAME | Latency: $LAT | Strategy: $STRAT"
            echo "----------------------------------------"

            OUTPUT_FILE="${OUTPUT_DIR}/${MAP_NAME}_${STRAT}_lat${LAT}.csv"
            echo "run,crosstrack_rmse_m" > $OUTPUT_FILE

            for ((i=1; i<=N_RUNS; i++)); do
                echo "Run $i"

                # -------------------------
                # Strategy handling
                # -------------------------
                EXTRA_ARGS=""

                if [ "$STRAT" == "round_robin" ]; then
                    EXTRA_ARGS="--cloud-strategy round_robin"

                elif [ "$STRAT" == "sensitivity" ]; then
                    EXTRA_ARGS="--cloud-strategy sensitivity --call-weights 1 1 1"

                elif [ "$STRAT" == "rl_simple" ]; then
                    MODEL="data/models/ppo_cte_k1_as0.7_asp0.2_lat${LAT}.zip"
                    EXTRA_ARGS="--cloud-strategy rl --rl-scheduler $MODEL"

                elif [ "$STRAT" == "rl_boot" ]; then
                    MODEL="data/models/ppo_cte_sensitivity_annealed_k1_as0.7_asp0.2_lat${LAT}.zip"
                    EXTRA_ARGS="--cloud-strategy rl --rl-scheduler $MODEL"
                fi

                # -------------------------
                # Run simulation
                # -------------------------
                OUTPUT=$(python $PY_SCRIPT \
                    --map $MAP \
                    --waypoints $WP \
                    --planner edge_cloud \
                    --cloud-latency $LAT \
                    --render-mode None \
                    --max-laps 2 \
                    --top-k 1 \
                    --randomize \
                    $EXTRA_ARGS 2>/dev/null)

                # -------------------------
                # Extract RMSE
                # -------------------------
                RMSE=$(echo "$OUTPUT" | grep '"crosstrack_rmse_m"' | sed -E 's/.*: ([0-9.]+).*/\1/')

                if [ -z "$RMSE" ]; then
                    echo "Warning: RMSE not found"
                    RMSE="NaN"
                fi

                echo "$i,$RMSE" >> $OUTPUT_FILE
            done

        done
    done
done

echo "All experiments completed."