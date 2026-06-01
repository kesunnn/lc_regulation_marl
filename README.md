# Reinforcement Learning for Freeway Lane-Change Regulation via Connected Vehicles

This repository contains the code for the paper *Reinforcement Learning for Freeway Lane-Change Regulation via Connected Vehicles*.  
Paper link: [arXiv:2412.04341](https://arxiv.org/abs/2412.04341)

## Installation

Clone the repository:

```bash
git clone git@github.com:blackiny/lc_regulation_marl.git
cd lc_regulation_marl
```

### Environment Preparation

- OS: Ubuntu 20.04
- Python: 3.11.14
- SUMO: 1.26.0

Install PyTorch first (CUDA 12.8 build):

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

Then install Python dependencies and package:

```bash
pip install -r requirements.txt
pip install -e .
```

### Gymnasium Note

This project uses `import gymnasium as gym`.

- The environment is intentionally used in a loose style.
- `action_space`/`observation_space` are not strictly defined in the current env classes.
- `gym.make(..., disable_env_checker=True)` is used to avoid strict framework checks.
- Training/eval scripts accept both old (`obs, reward, done, info`) and new (`obs, reward, terminated, truncated, info`) step outputs.

### Performance-Related Simulation Options

These options are under `simulation` in experiment JSON files:

- `sumo_step_length`: SUMO simulation step length (default `0.1`).
- `num_parallel_envs`: number of independent SUMO training workers opened inside one training run (default `1`).
  When this is greater than 1, `total_steps` is treated as aggregate environment-control steps across workers.
  The base `--port` plus the next `num_parallel_envs` ports must be free, because training workers use
  `--port ... --port + num_parallel_envs - 1` and the local evaluation environment uses `--port + num_parallel_envs`.
- `keep_sumo_outputs`: if `false` (default), disable heavy SUMO XML outputs (`fcd`, `emission`, `tripinfo`, etc.) in generated run configs.
- `enable_ttc_metrics`: if `false`, skip TTC/TET/TIT collection (faster).
- `enable_detector_metrics`: if `false`, skip detector metric collection.
- `return_detector_data`: if `false` (default), avoid returning full detector history in each `env.step()` info dict.

## Directory Structure

```text
$PROJECT_ROOT_DIR/pde_rl_control
 ├─ agents          # RL training agents
 ├─ configs         # Configuration files for training and evaluation
 ├─ environments    # RL environments based on Gymnasium API and SUMO simulator
 ├─ experiments     # Experiment configuration JSON files
 ├─ scripts         # Scripts for running training and evaluation
 └─ utils           # Utility functions and modules
```

## SUMO Environment Variants

All registered environments use one straight SUMO edge (`e1`) with 5 lanes (`e1_0` to `e1_4`) and a 24.59 m/s speed limit. Agents are lane-grid controllers; the number of controlled grid cells per lane is `road_length / grid_length`. Actions are applied only to `controlled` vehicles. `uncontrolled` vehicles keep their normal SUMO lane-change behavior.

| Gym id | Road geometry | Buffer layout | Control scheme | Notes |
| --- | --- | --- | --- | --- |
| `TrafficEnv_lane5_1` | 5 lanes, 1000 m physical lane length, 1000 m controlled length | No upstream or downstream buffer | 2 actions per lane grid: `0` suppress controlled-vehicle lane changes, `1` allow lane changes | Base binary lane-change regulation environment. |
| `TrafficEnv_lane5_2` | 5 lanes, 1000 m physical lane length, 1000 m controlled length | No upstream or downstream buffer | 4 actions per lane grid: `0` suppress, `1` allow both directions, `2` left-only, `3` right-only | Directional regulation is implemented by switching controlled vehicles among `controlled`, `controlled:left`, and `controlled:right` SUMO vehicle types. |
| `TrafficEnv_lane5_3` | 5 lanes, 1000 m physical lane length, 1000 m controlled length | No upstream or downstream buffer | Same 4-action directional control as `TrafficEnv_lane5_2` | Adds separate evaluation-baseline route/config support via `traffic_eval_baseline.sumocfg` and `v1_routes_eval_baseline.rou.xml`. |
| `TrafficEnv_lane5_4` | 5 lanes, 1000 m physical lane length, 1000 m controlled length | No upstream or downstream buffer | Same 4-action directional control as `TrafficEnv_lane5_3` | Uses a more aggressive SL2015 lane-changing template, including higher assertiveness/pushiness and lower cooperation in the route files. |
| `TrafficEnv_lane5_5` | 5 lanes, 1200 m physical lane length, 1000 m controlled/metric section | No upstream buffer. 200 m downstream tail buffer from 1000 m to 1200 m | Same 4-action directional control inside 0-1000 m; lane changes are suppressed in the 1000-1200 m tail buffer | Buffered version of `TrafficEnv_lane5_4`. State and section metrics are computed only over the 0-1000 m control window. |

## Training and Evaluation

Example config files:

- Training: `$PROJECT_ROOT_DIR/pde_rl_control/experiments/train`
- Evaluation: `$PROJECT_ROOT_DIR/pde_rl_control/experiments/eval`

### Training

Run from `$PROJECT_ROOT_DIR/pde_rl_control/scripts`:

```bash
python ./run_dqn_task.py --config_template dqn_basic --config_file ../experiments/train/dqn_lane5_3_rho_010_dummy_idm_grid100.json --seed 12345 --log_interval 1000 --port 39682 > train.out
```

Parallel SUMO collection can be enabled from the command line:

```bash
python ./run_dqn_task.py --config_template dqn_basic \
  --config_file ../experiments/train/dqn_lane5_3_rho_010_dummy_idm_grid100.json \
  --seed 12345 --log_interval 1000 --port 39682 --num_parallel_envs 4 > train.out
```

The PPO training entry point accepts the same `--num_parallel_envs` option.

Training outputs are written under `$PROJECT_ROOT_DIR/results/$training_id`:

```text
$PROJECT_ROOT_DIR/results/$training_id
├─ config.json  # Dumped configuration JSON
├─ models       # Training checkpoints
└─ tf_logs      # TensorBoard logs
```

### Evaluation

Run from `$PROJECT_ROOT_DIR/pde_rl_control/scripts`:

```bash
python ./run_task_eval.py --agent dqn --config_template dqn_basic \
--config_file ../experiments/eval/dqn_lane5_3_rho_015_dummy_idm_grid100.json \
--model_path $MODEL_PATH --seed 12346 --log_interval 1000 --port 39505 > eval.out
```

The evaluation results directory structure is similar to training, except no model checkpoints are generated.

### Batch Helper Scripts

Run from `$PROJECT_ROOT_DIR/pde_rl_control/scripts`.

Generate an eval batch script from training `nohup` logs:

```bash
python ./generate_eval_bash.py \
  --checkpoint 50000 \
  --log_folder ../training_logs_v1 \
  --agent dqn \
  --port_start 45600 \
  --output_script ./eval_dqn_ckpt50000.sh
```

Then launch the generated eval jobs:

```bash
bash ./eval_dqn_ckpt50000.sh
```

Generate `train_data_tf_logs_locs.json` from a `nohup` log folder:

```bash
python ./generate_vis_data_locations.py \
  --log_folder ../training_logs_v1 \
  --output_file ./train_data_tf_logs_locs.json
```

`generate_vis_data_locations.py` uses a hard-coded mapping from output key name to `experiment_name`
(for example, `"tf_log_dir_010_dummy_dqn": "010_dummy_dqn"`), and fills each key with the extracted
`tf_logs` path from lines matching `logging outputs to .../tf_logs`.
