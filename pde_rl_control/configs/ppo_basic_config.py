# %%
import json, os, time
import numpy as np
from pde_rl_control.configs.schedule import ConstantSchedule, PiecewiseSchedule, LinearSchedule
import torch

# %%
import pde_rl_control.utils.event_generator as env_event_gen
import pde_rl_control.utils.vehicle_generator as env_vehicle_gen

# %%
ppo_basic_config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../experiments/ppo_basic.json")

# %%
def make_ppo_basic_config(config_file):
    with open(ppo_basic_config_file, "r") as f:
        basic_conf = json.load(f)
    with open(config_file, "r") as f:
        conf = json.load(f)
    config_file_name = os.path.basename(config_file)
    conf_fields = ['meta', 'simulation', 'reward', 'network', 'training', 'eval']
    for field in conf_fields:
        if field in conf:
            basic_conf[field].update(conf[field])
    conf_str = json.dumps(basic_conf)
    # update callable attributes
    event_generator = basic_conf['simulation']['event_generator']
    if hasattr(env_event_gen, event_generator):
        basic_conf['simulation']['event_generator'] = getattr(env_event_gen, event_generator)
    else:
        raise ValueError('Invalid event generator: {}'.format(event_generator))
    vehicle_generator = basic_conf['simulation']['vehicle_generator']
    if hasattr(env_vehicle_gen, vehicle_generator):
        basic_conf['simulation']['vehicle_generator'] = getattr(env_vehicle_gen, vehicle_generator)
    else:
        raise ValueError('Invalid vehicle generator: {}'.format(vehicle_generator))
    # lr scheduler
    lr_scheduler_mode = basic_conf['training']['lr_scheduler_mode']
    if lr_scheduler_mode not in ["constant", "exponential"]:
        if lr_scheduler_mode == "piecewise":
            def make_lr_schedule(optimizer):
                _total_steps = basic_conf['training']['total_steps']
                lr_scheduler_params = basic_conf['training'].get("lr_scheduler_params", {})
                _outside_value = lr_scheduler_params.get("outside_value", 5e-1)
                _decay_step = lr_scheduler_params.get("decay_step", 20000)
                _stop_step = lr_scheduler_params.get("stop_step", _total_steps / 2)
                if _stop_step <= _decay_step:
                    pieces = [(0, 1), (_stop_step, _outside_value)]
                else:
                    pieces = [(0, 1), (_decay_step, 1), (_stop_step, _outside_value)]
                return torch.optim.lr_scheduler.LambdaLR(optimizer,
                        PiecewiseSchedule(endpoints=pieces, outside_value=_outside_value).value)
            basic_conf['training']['lr_scheduler_mode'] = make_lr_schedule
        else:
            raise ValueError("Invalid lr_scheduler_mode: {}".format(lr_scheduler_mode))

    # log dir
    log_string = "{}_{}_d{}".format(
        basic_conf["meta"]["experiment_name"] if "experiment_name" in basic_conf["meta"] else "ppo",
        basic_conf["simulation"]["env_name"],
        int(basic_conf["training"]["discount"]*1000),
    )
    log_string += "_{}".format(config_file_name.split(".")[0])
    result_root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../results")
    if not (os.path.exists(result_root_path)):
        os.makedirs(result_root_path)
    log_dir = log_string + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    basic_conf["meta"]["log_dir"] = log_dir
    result_path = os.path.join(result_root_path, log_dir)
    basic_conf["meta"]["result_path"] = result_path
    if not (os.path.exists(result_path)):
        os.makedirs(result_path)

    return basic_conf, conf_str



