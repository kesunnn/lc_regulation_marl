import copy
import multiprocessing as mp
import traceback

import numpy as np


def _unwrap_reset(reset_output):
    return reset_output[0] if isinstance(reset_output, tuple) else reset_output


def _unwrap_step(step_output):
    if len(step_output) == 5:
        next_state, reward, terminated, truncated, info = step_output
        return next_state, reward, bool(terminated or truncated), info
    return step_output


def make_parallel_env_config(config, env_index, port):
    env_config = {
        field: copy.deepcopy(config[field])
        for field in ("meta", "simulation", "reward", "eval")
        if field in config
    }
    base_log_dir = env_config.setdefault("meta", {}).get("log_dir", "parallel_env")
    env_config["meta"]["log_dir"] = f"{base_log_dir}_env{env_index}"
    env_config.setdefault("simulation", {})["traci_port"] = int(port)
    return env_config


def make_local_env_config(config, port):
    env_config = copy.deepcopy(config)
    env_config.setdefault("simulation", {})["traci_port"] = int(port)
    return env_config


def _env_status(env):
    base_env = env.unwrapped
    return {
        "warm_start_begin": bool(getattr(base_env, "warm_start_begin", False)),
        "warm_start_finish": bool(getattr(base_env, "warm_start_finish", False)),
        "is_eval": bool(getattr(base_env, "is_eval", False)),
    }


def _make_env(config):
    import gymnasium as gym
    import pde_rl_control.environments  # noqa: F401

    return gym.make(
        config["simulation"]["env_name"],
        disable_env_checker=True,
        grid_length=config["simulation"]["grid_length"],
        control_rate=config["simulation"]["control_rate"],
        density_level=config["simulation"]["density_level"],
        event_generator=config["simulation"]["event_generator"],
        vehicle_generator=config["simulation"]["vehicle_generator"],
        config=config,
    )


def _env_worker(remote, config, seed):
    env = None
    try:
        np.random.seed(seed)
        env = _make_env(config)
        while True:
            command, payload = remote.recv()
            if command == "reset":
                state = np.asarray(_unwrap_reset(env.reset()))
                remote.send(("ok", (state, _env_status(env))))
            elif command == "step":
                next_state, reward, done, info = _unwrap_step(env.step(payload))
                info = dict(info)
                info["_worker_status"] = _env_status(env)
                remote.send(("ok", (np.asarray(next_state), reward, done, info)))
            elif command == "set_eval_flag":
                is_eval, reset_vehicles, reset_event_generator = payload
                env.unwrapped.set_eval_flag(is_eval, reset_vehicles, reset_event_generator)
                remote.send(("ok", _env_status(env)))
            elif command == "set_is_eval_baseline_flag":
                env.unwrapped.set_is_eval_baseline_flag(bool(payload))
                remote.send(("ok", _env_status(env)))
            elif command == "close":
                break
            else:
                raise ValueError(f"Unknown worker command: {command}")
    except EOFError:
        pass
    except Exception:
        remote.send(("error", traceback.format_exc()))
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        remote.close()


class AsyncEnvPool:
    def __init__(self, config, num_envs, base_port, seed=1):
        self.num_envs = int(num_envs)
        self.base_port = int(base_port)
        start_method = config["simulation"].get("parallel_start_method", "fork")
        if start_method not in mp.get_all_start_methods():
            start_method = mp.get_start_method(allow_none=True) or mp.get_all_start_methods()[0]
        self.ctx = mp.get_context(start_method)
        self.remotes = []
        self.processes = []

        for env_index in range(self.num_envs):
            parent_remote, child_remote = self.ctx.Pipe()
            env_config = make_parallel_env_config(config, env_index, self.base_port + env_index)
            process = self.ctx.Process(
                target=_env_worker,
                args=(child_remote, env_config, int(seed) + env_index),
            )
            process.daemon = True
            process.start()
            child_remote.close()
            self.remotes.append(parent_remote)
            self.processes.append(process)

    def _receive(self, index):
        status, payload = self.remotes[index].recv()
        if status == "error":
            raise RuntimeError(f"Parallel SUMO worker {index} failed:\n{payload}")
        return payload

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [self._receive(index) for index in range(self.num_envs)]
        states, statuses = zip(*results)
        return list(states), list(statuses)

    def reset_one(self, index):
        self.remotes[index].send(("reset", None))
        return self._receive(index)

    def step(self, actions):
        assert len(actions) == self.num_envs, "actions length must match num_envs"
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        return [self._receive(index) for index in range(self.num_envs)]

    def step_indices(self, indices, actions):
        assert len(indices) == len(actions), "indices and actions length must match"
        for index, action in zip(indices, actions):
            self.remotes[index].send(("step", action))
        return [self._receive(index) for index in indices]

    def command_one(self, index, command, payload=None):
        self.remotes[index].send((command, payload))
        return self._receive(index)

    def command_all(self, command, payload=None):
        for remote in self.remotes:
            remote.send((command, payload))
        return [self._receive(index) for index in range(self.num_envs)]

    def set_eval_flag(self, is_eval, reset_vehicles=True, reset_event_generator=True, indices=None):
        payload = (bool(is_eval), bool(reset_vehicles), bool(reset_event_generator))
        if indices is None:
            return self.command_all("set_eval_flag", payload)
        for index in indices:
            self.remotes[index].send(("set_eval_flag", payload))
        return [self._receive(index) for index in indices]

    def set_is_eval_baseline_flag(self, value, indices=None):
        if indices is None:
            return self.command_all("set_is_eval_baseline_flag", bool(value))
        for index in indices:
            self.remotes[index].send(("set_is_eval_baseline_flag", bool(value)))
        return [self._receive(index) for index in indices]

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for remote in self.remotes:
            remote.close()
