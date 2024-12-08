# Reinforcement Learning for Freeway Lane-Change Regulation via Connected Vehicles

This repository contains the code for the paper *Reinforcement Learning for Freeway Lane-Change Regulation via Connected Vehicles*.  
You can access the paper here: [arXiv:2412.04341](https://arxiv.org/abs/2412.04341).



## Installation

Clone this repository using the following command:
```
git clone git@github.com:blackiny/lc_regulation_marl.git
```

### Environment preparation

System: Ubuntu 20.04.6 LTS  
Python version: 3.9.19  
SUMO version: 1.20.0

If needed, install missing Python dependencies:

```
pip install -r requirements.txt
```

Navigate to the `$PROJECT_ROOT_DIR` of the repository and install the package:

```
pip install -e .
```

## Directory Structure

The code directory structure is as follows:  
```
$PROJECT_ROOT_DIR/pde_rl_control
 ├─ agents          # RL training agents
 ├─ configs         # Configuration files for training and evaluation
 ├─ environments    # RL environments based on Gym API and SUMO simulator
 ├─ experiments     # Experiment configuration JSON files
 ├─ scripts         # Scripts for running training and evaluation
 └─ utils           # Utility functions and modules
```

## Training and Evaluation
Example configuration files for training and evaluation can be found in the following directories:  
Training: `$PROJECT_ROOT_DIR/pde_rl_control/experiments/train`  
Evaluation: `$PROJECT_ROOT_DIR/pde_rl_control/experiments/eval`

### Training

Navigate to `$PROJECT_ROOT_DIR/pde_rl_control/scripts` and run the training command:

```
python ./run_dqn_task.py --config_template dqn_basic --config_file ../experiments/train/dqn_lane5_3_rho_010_dummy_idm_grid100.json --seed 12345 --log_interval 1000 --port 39682 > train.out 
```

The training output will be logged in `train.out`, and the results will be saved in a subdirectory under `$PROJECT_ROOT_DIR/results`. The logging location is specified in `train.out`:

```
logging outputs to $PROJECT_ROOT_DIR/results/$training_id
```

The training results directory structure is as follows:

```
$PROJECT_ROOT_DIR/results/$training_id
├─ config.json  # The dumped configuration JSON file
├─ models       # All training checkpoints
└─ tf_logs      # TensorBoard logs
```

### Evaluation

To evaluate a training checkpoint, run the following command:

```
python ./run_dqn_task_eval.py --config_template dqn_basic \
--config_file ../experiments/eval/dqn_lane5_3_rho_015_dummy_idm_grid100.json \
--model_path $MODEL_PATH \
--seed 12346 --log_interval 1000 --port 39505 > eval.out
```

The evaluation results directory structure is similar to that of training, except no model checkpoints are generated during evaluation.