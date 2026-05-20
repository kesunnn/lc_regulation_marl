#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="../training_logs_v4"
SEED=1001
LOG_INTERVAL=1000

mkdir -p "$LOG_DIR"

# =============================
# DQN lane5_3 jobs
# =============================
# nohup python -u ./run_dqn_task.py \
#   --config_template dqn_basic \
#   --config_file ../experiments/train5_3/dqn_lane5_3_rho_010_dummy_idm_grid100.json \
#   --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 39682 \
#   > "$LOG_DIR/dqn_lane5_3_rho_010_dummy_idm_grid100.out" 2>&1 &
# echo "Started dqn_lane5_3_rho_010_dummy_idm_grid100 on port 39682"
# sleep 1

# nohup python -u ./run_dqn_task.py \
#   --config_template dqn_basic \
#   --config_file ../experiments/train5_3/dqn_lane5_3_rho_010_lane_degrade_idm_grid100.json \
#   --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 39683 \
#   > "$LOG_DIR/dqn_lane5_3_rho_010_lane_degrade_idm_grid100.out" 2>&1 &
# echo "Started dqn_lane5_3_rho_010_lane_degrade_idm_grid100 on port 39683"
# sleep 1

# nohup python -u ./run_dqn_task.py \
#   --config_template dqn_basic \
#   --config_file ../experiments/train5_3/dqn_lane5_3_rho_010_vehicle_stop_idm_grid100.json \
#   --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 39684 \
#   > "$LOG_DIR/dqn_lane5_3_rho_010_vehicle_stop_idm_grid100.out" 2>&1 &
# echo "Started dqn_lane5_3_rho_010_vehicle_stop_idm_grid100 on port 39684"
# sleep 1

# nohup python -u ./run_dqn_task.py \
#   --config_template dqn_basic \
#   --config_file ../experiments/train5_3/dqn_lane5_3_rho_015_dummy_idm_grid100.json \
#   --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 39685 \
#   > "$LOG_DIR/dqn_lane5_3_rho_015_dummy_idm_grid100.out" 2>&1 &
# echo "Started dqn_lane5_3_rho_015_dummy_idm_grid100 on port 39685"
# sleep 1

# nohup python -u ./run_dqn_task.py \
#   --config_template dqn_basic \
#   --config_file ../experiments/train5_3/dqn_lane5_3_rho_015_lane_degrade_idm_grid100.json \
#   --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 39686 \
#   > "$LOG_DIR/dqn_lane5_3_rho_015_lane_degrade_idm_grid100.out" 2>&1 &
# echo "Started dqn_lane5_3_rho_015_lane_degrade_idm_grid100 on port 39686"
# sleep 1

# nohup python -u ./run_dqn_task.py \
#   --config_template dqn_basic \
#   --config_file ../experiments/train5_3/dqn_lane5_3_rho_015_vehicle_stop_idm_grid100.json \
#   --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 39687 \
#   > "$LOG_DIR/dqn_lane5_3_rho_015_vehicle_stop_idm_grid100.out" 2>&1 &
# echo "Started dqn_lane5_3_rho_015_vehicle_stop_idm_grid100 on port 39687"
# sleep 1

# nohup python -u ./run_dqn_task.py \
#   --config_template dqn_basic \
#   --config_file ../experiments/train5_3/dqn_lane5_3_rho_045_dummy_idm_grid100.json \
#   --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 39688 \
#   > "$LOG_DIR/dqn_lane5_3_rho_045_dummy_idm_grid100.out" 2>&1 &
# echo "Started dqn_lane5_3_rho_045_dummy_idm_grid100 on port 39688"
# sleep 1

# nohup python -u ./run_dqn_task.py \
#   --config_template dqn_basic \
#   --config_file ../experiments/train5_3/dqn_lane5_3_rho_045_lane_degrade_idm_grid100.json \
#   --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 39689 \
#   > "$LOG_DIR/dqn_lane5_3_rho_045_lane_degrade_idm_grid100.out" 2>&1 &
# echo "Started dqn_lane5_3_rho_045_lane_degrade_idm_grid100 on port 39689"
# sleep 1

# nohup python -u ./run_dqn_task.py \
#   --config_template dqn_basic \
#   --config_file ../experiments/train5_3/dqn_lane5_3_rho_045_vehicle_stop_idm_grid100.json \
#   --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 39690 \
#   > "$LOG_DIR/dqn_lane5_3_rho_045_vehicle_stop_idm_grid100.out" 2>&1 &
# echo "Started dqn_lane5_3_rho_045_vehicle_stop_idm_grid100 on port 39690"
# sleep 1

echo "All DQN jobs launched."
