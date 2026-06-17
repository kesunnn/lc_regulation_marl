# Reinforcement Learning for Freeway Lane-Change Regulation via Connected Vehicles

## Abstract

Lane-change decision-making is difficult to regulate because freeway traffic involves coupled vehicle--vehicle and vehicle--infrastructure interactions. This repository contains the code for the paper **Reinforcement Learning for Freeway Lane-Change Regulation via Connected Vehicles**, which studies a lane-change regulation framework that improves freeway traffic efficiency through connected vehicles (CVs).

The proposed method computes lane-change regulation signals at a traffic management center and broadcasts them to CVs. Human-driven vehicles remain uncontrolled, while CVs follow grid-level signals such as allowing or disallowing left or right lane changes. This design avoids direct trajectory intervention, reduces positioning and communication requirements, and supports mixed traffic with different connectivity rates.

The framework combines a microscopic SUMO simulation environment with a macroscopic lane-grid representation motivated by multi-lane traffic partial differential equations (PDEs). SUMO executes vehicle-level dynamics, while the learning policy observes aggregated lane-grid states and regulates lane-change source-term exchanges between adjacent lanes. The policy is trained under a centralized-training/decentralized-execution multi-agent reinforcement learning formulation.

Experiments evaluate stable-flow, lane-degrade, and vehicle-stop scenarios under low, high, and congested-high demand. Results show that the learned regulation policy improves traffic efficiency in non-congested settings while maintaining comparable safety behavior in microscopic simulation.

## Repository Scope

This branch is prepared for paper review. It keeps only the code and configuration needed to train and evaluate the manuscript experiments.

```text
pde_rl_control/
├── agents/          # Double DQN and PPO agents
├── configs/         # Python config loaders and schedules
├── environments/    # SUMO/Gymnasium lane-change regulation environments
├── experiments/     # Manuscript experiment JSON files
├── scripts/         # Training and evaluation entry points
└── utils/           # Logging, TraCI, replay-buffer, and metric utilities
```

The retained SUMO environments are:

| Gymnasium id | Demand regime | Purpose |
| --- | --- | --- |
| `TrafficEnv_lane5_5` | Low and high demand | 5-lane freeway with a downstream buffer and baseline evaluation support. |
| `TrafficEnv_lane5_5_congested` | Congested-high demand | Congested variant used for high-density manuscript experiments. |

## Installation

The manuscript experiments were run with SUMO, TraCI, Gymnasium, PyTorch, and TensorBoard.

```bash
git clone git@github.com:blackiny/lc_regulation_marl.git
cd lc_regulation_marl
pip install -r requirements.txt
pip install -e .
```

Install PyTorch separately for your CUDA or CPU environment. For example, follow the official PyTorch selector for the wheel matching your hardware.

SUMO must be available on `PATH`, and the Python packages `traci` and `sumolib` should match the installed SUMO release.

## Experiment Configurations

The retained JSON files cover the manuscript scenarios:

| Folder | Algorithm/stage | Contents |
| --- | --- | --- |
| `pde_rl_control/experiments/train5_5` | Double DQN training | Stable flow, lane degrade, and vehicle stop for `rho_010`, `rho_015`, and `rho_045`. |
| `pde_rl_control/experiments/train_ppo5_5` | PPO training | PPO baseline configs for the same retained demand/scenario set. |
| `pde_rl_control/experiments/eval5_5` | Evaluation | Evaluation configs for trained Double DQN or PPO checkpoints. |

Demand identifiers correspond to the paper settings:

| Config suffix | Paper demand |
| --- | --- |
| `rho_010` | Low demand |
| `rho_015` | High demand |
| `rho_045` | Congested-high demand |

Scenario identifiers correspond to:

| Config suffix | Paper scenario |
| --- | --- |
| `dummy` | Stable Flow |
| `lane_degrade` | Lane Degrade |
| `vehicle_stop` | Vehicle Stop |

## Training

Run commands from `pde_rl_control/scripts`.

Double DQN example:

```bash
python ./run_dqn_task.py \
  --config_template dqn_basic \
  --config_file ../experiments/train5_5/dqn_lane5_5_rho_010_dummy_idm_grid100.json \
  --seed 1001 \
  --log_interval 1000 \
  --port 39682
```

PPO example:

```bash
python ./run_ppo_task.py \
  --config_template ppo_basic \
  --config_file ../experiments/train_ppo5_5/ppo_lane5_5_rho_010_dummy_idm_grid100.json \
  --seed 1001 \
  --log_interval 1000 \
  --port 40682
```

Training outputs are written to:

```text
results/<experiment_name>_<date>_<time>/
├── config.json
├── models/
└── tf_logs/
```

## Evaluation

Run commands from `pde_rl_control/scripts`.

```bash
python ./run_task_eval.py \
  --agent dqn \
  --config_template dqn_basic \
  --config_file ../experiments/eval5_5/lane5_5_rho_010_dummy_idm_grid100.json \
  --model_path ../../results/<training_run>/models/<checkpoint>.pt \
  --seed 1002 \
  --log_interval 1000 \
  --port 45682
```

Use `--agent ppo --config_template ppo_basic` when evaluating PPO checkpoints.

If `--num_parallel_envs` is set, the selected `--port` and the next worker ports must be free. The local evaluation environment uses the next port after the worker pool.

## Citation

If this repository is useful for your work, please cite the associated manuscript:

```bibtex
@article{sun2026lanechange,
  title={Reinforcement Learning for Freeway Lane-Change Regulation via Connected Vehicles},
  author={Sun, Ke and Yu, Huan},
  year={2026}
}
```
