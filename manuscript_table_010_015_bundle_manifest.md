# Manuscript `010/015` Data Bundle

This bundle contains the `rho=010` and `rho=015` result folders and provenance log files used for the updated manuscript tables.

Scope:

- Tables 1-3: updated low/high DDQN standalone eval results
- Table 4: updated low/high DDQN cross-validation eval results
- Table 5: updated low/high PPO `150k`, DDQN `150k`, and DDQN converged/best eval results

Excluded:

- `rho=045` folders
- unrelated training outputs

## Included result folders

### DDQN standalone eval

- `results/dqn_lane5_5_rho_010_dummy_idm_grid100_250k_eval_19-05-2026_16-35-21`
- `results/dqn_lane5_5_rho_010_dummy_idm_grid100_250k_eval_19-05-2026_19-56-28`
- `results/dqn_lane5_5_rho_010_lane_degrade_idm_grid100_250k_eval_19-05-2026_22-24-42`
- `results/dqn_lane5_5_rho_010_lane_degrade_idm_grid100_250k_eval_20-05-2026_02-45-40`
- `results/dqn_lane5_5_rho_010_vehicle_stop_idm_grid100_250k_eval_19-05-2026_16-30-51`
- `results/dqn_lane5_5_rho_010_vehicle_stop_idm_grid100_250k_eval_19-05-2026_19-07-33`
- `results/dqn_lane5_5_rho_015_dummy_idm_grid100_250k_eval_21-05-2026_04-08-01`
- `results/dqn_lane5_5_rho_015_dummy_idm_grid100_250k_eval_21-05-2026_04-13-42`
- `results/dqn_lane5_5_rho_015_lane_degrade_idm_grid100_250k_eval_21-05-2026_04-13-43`
- `results/dqn_lane5_5_rho_015_lane_degrade_idm_grid100_250k_eval_21-05-2026_04-13-57`
- `results/dqn_lane5_5_rho_015_vehicle_stop_idm_grid100_250k_eval_21-05-2026_04-08-03`
- `results/dqn_lane5_5_rho_015_vehicle_stop_idm_grid100_250k_eval_21-05-2026_04-13-44`

### DDQN cross-validation eval

- `results/dqn_lane5_5_rho_010_lane_degrade_idm_grid100_250k_eval_24-05-2026_01-13-17`
- `results/dqn_lane5_5_rho_010_vehicle_stop_idm_grid100_250k_eval_24-05-2026_01-13-18`
- `results/dqn_lane5_5_rho_015_lane_degrade_idm_grid100_250k_eval_24-05-2026_01-13-19`
- `results/dqn_lane5_5_rho_015_vehicle_stop_idm_grid100_250k_eval_24-05-2026_01-13-20`

### PPO standalone eval

- `results/ppo_lane5_5_rho_010_dummy_idm_grid100_250k_eval_23-05-2026_11-36-39`
- `results/ppo_lane5_5_rho_010_lane_degrade_idm_grid100_250k_eval_23-05-2026_16-38-21`
- `results/ppo_lane5_5_rho_010_vehicle_stop_idm_grid100_250k_eval_23-05-2026_11-49-34`
- `results/ppo_lane5_5_rho_015_dummy_idm_grid100_250k_eval_23-05-2026_19-08-46`
- `results/ppo_lane5_5_rho_015_lane_degrade_idm_grid100_250k_eval_24-05-2026_02-36-43`
- `results/ppo_lane5_5_rho_015_vehicle_stop_idm_grid100_250k_eval_23-05-2026_16-52-35`

## Included provenance log files

### DDQN eval logs

- `pde_rl_control/eval_logs_lane5_5_rho010_015_045_250k/eval_dqn_lane5_5_rho_010_dummy_idm_grid100_250k_ckpt150000.out`
- `pde_rl_control/eval_logs_lane5_5_rho010_015_045_250k/eval_dqn_lane5_5_rho_010_dummy_idm_grid100_250k_ckpt200000.out`
- `pde_rl_control/eval_logs_lane5_5_rho010_015_045_250k/eval_dqn_lane5_5_rho_010_lane_degrade_idm_grid100_250k_ckpt150000.out`
- `pde_rl_control/eval_logs_lane5_5_rho010_015_045_250k/eval_dqn_lane5_5_rho_010_lane_degrade_idm_grid100_250k_ckpt200000.out`
- `pde_rl_control/eval_logs_lane5_5_rho010_015_045_250k/eval_dqn_lane5_5_rho_010_vehicle_stop_idm_grid100_250k_ckpt150000.out`
- `pde_rl_control/eval_logs_lane5_5_rho010_015_045_250k/eval_dqn_lane5_5_rho_010_vehicle_stop_idm_grid100_250k_ckpt200000.out`
- `pde_rl_control/eval_logs_lane5_5_rho010_015_045_250k/eval_dqn_lane5_5_rho_015_dummy_idm_grid100_250k_ckpt150000.out`
- `pde_rl_control/eval_logs_lane5_5_rho010_015_045_250k/eval_dqn_lane5_5_rho_015_dummy_idm_grid100_250k_ckpt250000.out`
- `pde_rl_control/eval_logs_lane5_5_rho010_015_045_250k/eval_dqn_lane5_5_rho_015_lane_degrade_idm_grid100_250k_ckpt150000.out`
- `pde_rl_control/eval_logs_lane5_5_rho010_015_045_250k/eval_dqn_lane5_5_rho_015_lane_degrade_idm_grid100_250k_ckpt200000.out`
- `pde_rl_control/eval_logs_lane5_5_rho010_015_045_250k/eval_dqn_lane5_5_rho_015_vehicle_stop_idm_grid100_250k_ckpt150000.out`
- `pde_rl_control/eval_logs_lane5_5_rho010_015_045_250k/eval_dqn_lane5_5_rho_015_vehicle_stop_idm_grid100_250k_ckpt250000.out`

### DDQN cross-validation logs

- `pde_rl_control/eval_logs_lane5_5_cross_dummy_best/eval_cross_rho010_dummy_to_lane_degrade_ckpt200000.out`
- `pde_rl_control/eval_logs_lane5_5_cross_dummy_best/eval_cross_rho010_dummy_to_vehicle_stop_ckpt200000.out`
- `pde_rl_control/eval_logs_lane5_5_cross_dummy_best/eval_cross_rho015_dummy_to_lane_degrade_ckpt250000.out`
- `pde_rl_control/eval_logs_lane5_5_cross_dummy_best/eval_cross_rho015_dummy_to_vehicle_stop_ckpt250000.out`

### PPO eval logs

- `pde_rl_control/eval_logs_ppo_lane5_5_rho010_015_045_250k/eval_ppo_lane5_5_rho_010_dummy_idm_grid100_250k_ckpt150000.out`
- `pde_rl_control/eval_logs_ppo_lane5_5_rho010_015_045_250k/eval_ppo_lane5_5_rho_010_lane_degrade_idm_grid100_250k_ckpt150000.out`
- `pde_rl_control/eval_logs_ppo_lane5_5_rho010_015_045_250k/eval_ppo_lane5_5_rho_010_vehicle_stop_idm_grid100_250k_ckpt150000.out`
- `pde_rl_control/eval_logs_ppo_lane5_5_rho010_015_045_250k/eval_ppo_lane5_5_rho_015_dummy_idm_grid100_250k_ckpt150000.out`
- `pde_rl_control/eval_logs_ppo_lane5_5_rho010_015_045_250k/eval_ppo_lane5_5_rho_015_lane_degrade_idm_grid100_250k_ckpt150000.out`
- `pde_rl_control/eval_logs_ppo_lane5_5_rho010_015_045_250k/eval_ppo_lane5_5_rho_015_vehicle_stop_idm_grid100_250k_ckpt150000.out`

## Archive intent

This bundle is for auditability of the updated `rho=010` and `rho=015` manuscript tables. Each included log file contains a `logging outputs to .../tf_logs` line that points to one of the included result directories.
