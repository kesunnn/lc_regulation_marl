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

## Training and Evaluation

Example config files:

- Training: `$PROJECT_ROOT_DIR/pde_rl_control/experiments/train`
- Evaluation: `$PROJECT_ROOT_DIR/pde_rl_control/experiments/eval`

### Training

Run from `$PROJECT_ROOT_DIR/pde_rl_control/scripts`:

```bash
python ./run_dqn_task.py --config_template dqn_basic --config_file ../experiments/train/dqn_lane5_3_rho_010_dummy_idm_grid100.json --seed 12345 --log_interval 1000 --port 39682 > train.out
```

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
