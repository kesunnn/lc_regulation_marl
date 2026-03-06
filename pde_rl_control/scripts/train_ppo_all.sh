#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="../training_logs"
SEED=42
LOG_INTERVAL=1000

mkdir -p "$LOG_DIR"

# =============================
# PPO lane5_3 jobs
# =============================
# nohup python -u ./run_ppo_task.py \
#   --config_template ppo_basic \
#   --config_file ../experiments/train_ppo5_3/ppo_lane5_3_rho_010_dummy_idm_grid100.json \
#   --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 40682 \
#   > "$LOG_DIR/ppo_lane5_3_rho_010_dummy_idm_grid100.out" 2>&1 &
# echo "Started ppo_lane5_3_rho_010_dummy_idm_grid100 on port 40682"
# sleep 1

# nohup python -u ./run_ppo_task.py \
#   --config_template ppo_basic \
#   --config_file ../experiments/train_ppo5_3/ppo_lane5_3_rho_010_lane_degrade_idm_grid100.json \
#   --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 40683 \
#   > "$LOG_DIR/ppo_lane5_3_rho_010_lane_degrade_idm_grid100.out" 2>&1 &
# echo "Started ppo_lane5_3_rho_010_lane_degrade_idm_grid100 on port 40683"
# sleep 1

# nohup python -u ./run_ppo_task.py \
#   --config_template ppo_basic \
#   --config_file ../experiments/train_ppo5_3/ppo_lane5_3_rho_010_vehicle_stop_idm_grid100.json \
#   --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 40684 \
#   > "$LOG_DIR/ppo_lane5_3_rho_010_vehicle_stop_idm_grid100.out" 2>&1 &
# echo "Started ppo_lane5_3_rho_010_vehicle_stop_idm_grid100 on port 40684"
# sleep 1

# nohup python -u ./run_ppo_task.py \
#   --config_template ppo_basic \
#   --config_file ../experiments/train_ppo5_3/ppo_lane5_3_rho_015_dummy_idm_grid100.json \
#   --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 40685 \
#   > "$LOG_DIR/ppo_lane5_3_rho_015_dummy_idm_grid100.out" 2>&1 &
# echo "Started ppo_lane5_3_rho_015_dummy_idm_grid100 on port 40685"
# sleep 1

# nohup python -u ./run_ppo_task.py \
#   --config_template ppo_basic \
#   --config_file ../experiments/train_ppo5_3/ppo_lane5_3_rho_015_lane_degrade_idm_grid100.json \
#   --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 40686 \
#   > "$LOG_DIR/ppo_lane5_3_rho_015_lane_degrade_idm_grid100.out" 2>&1 &
# echo "Started ppo_lane5_3_rho_015_lane_degrade_idm_grid100 on port 40686"
# sleep 1

# nohup python -u ./run_ppo_task.py \
#   --config_template ppo_basic \
#   --config_file ../experiments/train_ppo5_3/ppo_lane5_3_rho_015_vehicle_stop_idm_grid100.json \
#   --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 40687 \
#   > "$LOG_DIR/ppo_lane5_3_rho_015_vehicle_stop_idm_grid100.out" 2>&1 &
# echo "Started ppo_lane5_3_rho_015_vehicle_stop_idm_grid100 on port 40687"
# sleep 1

# nohup python -u ./run_ppo_task.py \
#   --config_template ppo_basic \
#   --config_file ../experiments/train_ppo5_3/ppo_lane5_3_rho_045_dummy_idm_grid100.json \
#   --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 40688 \
#   > "$LOG_DIR/ppo_lane5_3_rho_045_dummy_idm_grid100.out" 2>&1 &
# echo "Started ppo_lane5_3_rho_045_dummy_idm_grid100 on port 40688"
# sleep 1

# nohup python -u ./run_ppo_task.py \
#   --config_template ppo_basic \
#   --config_file ../experiments/train_ppo5_3/ppo_lane5_3_rho_045_lane_degrade_idm_grid100.json \
#   --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 40689 \
#   > "$LOG_DIR/ppo_lane5_3_rho_045_lane_degrade_idm_grid100.out" 2>&1 &
# echo "Started ppo_lane5_3_rho_045_lane_degrade_idm_grid100 on port 40689"
# sleep 1

# nohup python -u ./run_ppo_task.py \
#   --config_template ppo_basic \
#   --config_file ../experiments/train_ppo5_3/ppo_lane5_3_rho_045_vehicle_stop_idm_grid100.json \
#   --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 40690 \
#   > "$LOG_DIR/ppo_lane5_3_rho_045_vehicle_stop_idm_grid100.out" 2>&1 &
# echo "Started ppo_lane5_3_rho_045_vehicle_stop_idm_grid100 on port 40690"
# sleep 1

# =============================
# PPO lane5_4 jobs
# =============================
nohup python -u ./run_ppo_task.py \
  --config_template ppo_basic \
  --config_file ../experiments/train_ppo5_4/ppo_lane5_4_rho_010_dummy_idm_grid100.json \
  --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 40691 \
  > "$LOG_DIR/ppo_lane5_4_rho_010_dummy_idm_grid100.out" 2>&1 &
echo "Started ppo_lane5_4_rho_010_dummy_idm_grid100 on port 40691"
sleep 1

nohup python -u ./run_ppo_task.py \
  --config_template ppo_basic \
  --config_file ../experiments/train_ppo5_4/ppo_lane5_4_rho_010_lane_degrade_idm_grid100.json \
  --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 40692 \
  > "$LOG_DIR/ppo_lane5_4_rho_010_lane_degrade_idm_grid100.out" 2>&1 &
echo "Started ppo_lane5_4_rho_010_lane_degrade_idm_grid100 on port 40692"
sleep 1

nohup python -u ./run_ppo_task.py \
  --config_template ppo_basic \
  --config_file ../experiments/train_ppo5_4/ppo_lane5_4_rho_010_vehicle_stop_idm_grid100.json \
  --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 40693 \
  > "$LOG_DIR/ppo_lane5_4_rho_010_vehicle_stop_idm_grid100.out" 2>&1 &
echo "Started ppo_lane5_4_rho_010_vehicle_stop_idm_grid100 on port 40693"
sleep 1

nohup python -u ./run_ppo_task.py \
  --config_template ppo_basic \
  --config_file ../experiments/train_ppo5_4/ppo_lane5_4_rho_015_dummy_idm_grid100.json \
  --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 40694 \
  > "$LOG_DIR/ppo_lane5_4_rho_015_dummy_idm_grid100.out" 2>&1 &
echo "Started ppo_lane5_4_rho_015_dummy_idm_grid100 on port 40694"
sleep 1

nohup python -u ./run_ppo_task.py \
  --config_template ppo_basic \
  --config_file ../experiments/train_ppo5_4/ppo_lane5_4_rho_015_lane_degrade_idm_grid100.json \
  --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 40695 \
  > "$LOG_DIR/ppo_lane5_4_rho_015_lane_degrade_idm_grid100.out" 2>&1 &
echo "Started ppo_lane5_4_rho_015_lane_degrade_idm_grid100 on port 40695"
sleep 1

nohup python -u ./run_ppo_task.py \
  --config_template ppo_basic \
  --config_file ../experiments/train_ppo5_4/ppo_lane5_4_rho_015_vehicle_stop_idm_grid100.json \
  --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 40696 \
  > "$LOG_DIR/ppo_lane5_4_rho_015_vehicle_stop_idm_grid100.out" 2>&1 &
echo "Started ppo_lane5_4_rho_015_vehicle_stop_idm_grid100 on port 40696"
sleep 1

nohup python -u ./run_ppo_task.py \
  --config_template ppo_basic \
  --config_file ../experiments/train_ppo5_4/ppo_lane5_4_rho_030_dummy_idm_grid100.json \
  --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 40700 \
  > "$LOG_DIR/ppo_lane5_4_rho_030_dummy_idm_grid100.out" 2>&1 &
echo "Started ppo_lane5_4_rho_030_dummy_idm_grid100 on port 40700"
sleep 1

nohup python -u ./run_ppo_task.py \
  --config_template ppo_basic \
  --config_file ../experiments/train_ppo5_4/ppo_lane5_4_rho_030_lane_degrade_idm_grid100.json \
  --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 40701 \
  > "$LOG_DIR/ppo_lane5_4_rho_030_lane_degrade_idm_grid100.out" 2>&1 &
echo "Started ppo_lane5_4_rho_030_lane_degrade_idm_grid100 on port 40701"
sleep 1

nohup python -u ./run_ppo_task.py \
  --config_template ppo_basic \
  --config_file ../experiments/train_ppo5_4/ppo_lane5_4_rho_030_vehicle_stop_idm_grid100.json \
  --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 40702 \
  > "$LOG_DIR/ppo_lane5_4_rho_030_vehicle_stop_idm_grid100.out" 2>&1 &
echo "Started ppo_lane5_4_rho_030_vehicle_stop_idm_grid100 on port 40702"
sleep 1

# nohup python -u ./run_ppo_task.py \
#   --config_template ppo_basic \
#   --config_file ../experiments/train_ppo5_4/ppo_lane5_4_rho_045_dummy_idm_grid100.json \
#   --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 40697 \
#   > "$LOG_DIR/ppo_lane5_4_rho_045_dummy_idm_grid100.out" 2>&1 &
# echo "Started ppo_lane5_4_rho_045_dummy_idm_grid100 on port 40697"
# sleep 1

# nohup python -u ./run_ppo_task.py \
#   --config_template ppo_basic \
#   --config_file ../experiments/train_ppo5_4/ppo_lane5_4_rho_045_lane_degrade_idm_grid100.json \
#   --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 40698 \
#   > "$LOG_DIR/ppo_lane5_4_rho_045_lane_degrade_idm_grid100.out" 2>&1 &
# echo "Started ppo_lane5_4_rho_045_lane_degrade_idm_grid100 on port 40698"
# sleep 1

# nohup python -u ./run_ppo_task.py \
#   --config_template ppo_basic \
#   --config_file ../experiments/train_ppo5_4/ppo_lane5_4_rho_045_vehicle_stop_idm_grid100.json \
#   --seed "$SEED" --log_interval "$LOG_INTERVAL" --port 40699 \
#   > "$LOG_DIR/ppo_lane5_4_rho_045_vehicle_stop_idm_grid100.out" 2>&1 &
# echo "Started ppo_lane5_4_rho_045_vehicle_stop_idm_grid100 on port 40699"
# sleep 1

echo "All PPO jobs launched."
