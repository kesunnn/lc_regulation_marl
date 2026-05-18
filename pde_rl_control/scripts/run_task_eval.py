# %%
import os, time, argparse, json, traceback, copy
import gymnasium as gym
import numpy as np
import torch
import tqdm

from pde_rl_control.agents.dqn import DQNAgent
from pde_rl_control.agents.ppo import PPOAgent
import pde_rl_control.utils.pytorch_util as ptu
import pde_rl_control.configs as configs

from pde_rl_control.utils.replay_buffer import ReplayBuffer
from pde_rl_control.utils.eval import calculate_episode_reward, eval_episode
from pde_rl_control.utils.parallel_env import AsyncEnvPool, make_local_env_config
from pde_rl_control.utils.u import process_data_by_method
import pde_rl_control.environments

AGENT_MAP = {
	"dqn": DQNAgent,
	"ppo": PPOAgent
}

# %%
def _unwrap_reset(reset_output):
	return reset_output[0] if isinstance(reset_output, tuple) else reset_output


def _unwrap_step(step_output):
	if len(step_output) == 5:
		next_state, reward, terminated, truncated, info = step_output
		return next_state, reward, bool(terminated or truncated), info
	return step_output


def _prefix_eval_result_dir(config: dict, agent: str):
	"""
	Prefix eval output folder with agent name to avoid collisions across agent types.
	"""
	meta = config.get("meta", {})
	log_dir = meta.get("log_dir", "")
	result_path = meta.get("result_path", "")
	if not log_dir or not result_path:
		return
	prefix = f"{agent}_"
	if log_dir.startswith(prefix):
		return
	result_root = os.path.dirname(os.path.normpath(result_path))
	prefixed_log_dir = prefix + log_dir
	prefixed_result_path = os.path.join(result_root, prefixed_log_dir)
	os.makedirs(prefixed_result_path, exist_ok=True)
	meta["log_dir"] = prefixed_log_dir
	meta["result_path"] = prefixed_result_path


def _make_env_from_config(config):
	return gym.make(
		config["simulation"]["env_name"],
		disable_env_checker=True,
		grid_length=config["simulation"]["grid_length"],
		control_rate=config["simulation"]["control_rate"],
		density_level=config["simulation"]["density_level"],
		event_generator=config["simulation"]["event_generator"],
		vehicle_generator=config["simulation"]["vehicle_generator"],
		config=config
	)


def _num_parallel_envs(config, args):
	if getattr(args, "num_parallel_envs", None) is not None:
		return int(args.num_parallel_envs)
	return int(config["eval"].get(
		"num_parallel_envs",
		config["simulation"].get("num_parallel_envs", 1)
	))


def _log_eval_episode(logger, epi_num, info, eval_metrics, agent_rewards, agent_global_rewards,
					  eval_episode_length_dummy, eval_metrics_dummy, eval_rewards_dummy,
					  eval_global_rewards_dummy, discount_factor, reward_metrics_methods):
	agent_avg_reward = calculate_episode_reward(agent_rewards, discount_factor)
	global_avg_reward = calculate_episode_reward(agent_global_rewards, discount_factor)
	eval_rewards_dummy = calculate_episode_reward(eval_rewards_dummy, discount_factor)
	eval_global_rewards_dummy = calculate_episode_reward(eval_global_rewards_dummy, discount_factor)

	if info["is_collision"]:
		logger.log_scalar(len(info["collision_vehicles"]), "eval_collisions", epi_num)
	else:
		logger.log_scalar(0, "eval_collisions", epi_num)

	epi_length_dict = {"episode_length": info["end_time"], "episode_length:dummy": eval_episode_length_dummy}
	logger.log_scalars(epi_length_dict, "eval_metrics/episode_length", epi_num)
	logger.flush()

	eval_metrics = copy.deepcopy(eval_metrics)
	for key, value in eval_metrics.items():
		if isinstance(value, dict):
			for key2, value2 in eval_metrics_dummy[key].items():
				value[key2 + ":dummy"] = value2
			logger.log_scalars(value, f'eval_metrics/{key}', epi_num)
		else:
			value_dummy = eval_metrics_dummy[key]
			value_dict = {key: value, key + ":dummy": value_dummy}
			logger.log_scalar(value_dict, f'eval_metrics/{key}', epi_num)
	logger.flush()

	if agent_avg_reward is not None and eval_rewards_dummy is not None:
		agent_reward_metrics = {}
		for method in reward_metrics_methods:
			agent_reward_metrics[method] = process_data_by_method(agent_avg_reward, method)
			agent_reward_metrics[method + ":dummy"] = process_data_by_method(eval_rewards_dummy, method)
		logger.log_scalars(agent_reward_metrics, "eval_metrics/rewards", epi_num)
	else:
		print("Warning: empty agent rewards at episode {}".format(epi_num))
	if global_avg_reward is not None and eval_global_rewards_dummy is not None:
		global_reward_metrics = {}
		for method in reward_metrics_methods:
			global_reward_metrics[method] = process_data_by_method(global_avg_reward, method)
			global_reward_metrics[method + ":dummy"] = process_data_by_method(eval_global_rewards_dummy, method)
		logger.log_scalars(global_reward_metrics, "eval_metrics/global_rewards", epi_num)
	else:
		print("Warning: empty global rewards at episode {}".format(epi_num))
	logger.flush()


# %%
def run_eval_loop(config: dict, logger, args: argparse.Namespace):
	num_envs = _num_parallel_envs(config, args)
	if num_envs > 1:
		return run_parallel_eval_loop(config, logger, args, num_envs)

	# set random seeds
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	use_gpu, gpu_id = config["training"]["use_gpu"], int(config["training"]["gpu_id"])
	ptu.init_gpu(use_gpu=use_gpu, gpu_id=gpu_id)

	# make the gym environment
	env = _make_env_from_config(config)
	base_env = env.unwrapped

	# create training agent
	network_config = config["network"]
	training_config = config["training"]
	agent_class = AGENT_MAP[args.agent]
	agent = agent_class(
		env=base_env,
		network_config=network_config,
		training_config=training_config
	)

	state = None
	discount_factor = float(training_config["discount"])
	episode_agents_reward, episode_global_reward = [], []
	reward_metrics_methods = ["avg", "50pt", "90pt"]

	def reset_env_training():
		nonlocal state

		state = _unwrap_reset(env.reset())
		state = np.asarray(state)
		return

	reset_env_training()
	base_env.set_eval_flag(True) # set the evaluation flag to True

	total_episodes = int(config["eval"]["num_episodes"])
	is_eval_baseline = bool(config["eval"]["is_eval_baseline"])
	exclude_warm_start = bool(config["eval"]["exclude_warm_start"])
	model_path = args.model_path # log model
	try:
		agent.load(model_path)
	except Exception as e:
		print("Failed to load model. Error: {}, Trackback: {}".format(e, traceback.format_exc()))
		return
	# dummy action used in warm_start phase
	dummy_action = np.ones((base_env.num_lanes, base_env.n_agents_per_lane), dtype=int)
	is_add_graph = False
	# log key information text in the tensorboard
	meta_text = "Env name: {}\n".format(config["simulation"]["env_name"])
	meta_text += "desired_rho: {}, desired_flow: {}, desired_velocity: {}, desired_traffic_condition: {}\n".\
				format(base_env.desired_rho, base_env.desired_flow, base_env.desired_velocity, base_env.desired_traffic_condition)
	meta_text += "grid_length: {}, control_rate: {}, density_level: {}\n".\
				format(base_env.grid_length, base_env.control_rate, base_env.density_level)
	meta_text += "event_generator: {}, event_generator_mode:{}, vehicle_generator: {}, fundamental_diagram_name:{}\n".\
				format(getattr(base_env.event_generator, "__name__", str(base_env.event_generator)), base_env.event_generator_mode, \
						getattr(base_env.vehicle_generator, "__name__", str(base_env.vehicle_generator)), base_env.fundamental_diagram_name)
	meta_text += "reward_gamma: {}, is_eval_baseline: {}, exclude_warm_start: {}\n".\
				format(discount_factor, is_eval_baseline, exclude_warm_start)
	logger.log_text(meta_text, "meta", 0)
	logger.flush()
	step = 0
	for epi_num in tqdm.trange(total_episodes, dynamic_ncols=True):
		done = False
		while not done:
			if not base_env.warm_start_finish:
				action = dummy_action
			else:
				# Compute action
				action = agent.get_action(state=state, epsilon=0.0)
			# Step the environment
			next_state, reward, done, info = _unwrap_step(env.step(action))

			if base_env.warm_start_finish:
				if step % args.log_interval == 0:
					# Log the action allow rate per lane
					action_allow_rate = info["action_allow_rate"]
					for lane_id, rate_dict in action_allow_rate.items():
						logger.log_scalars(rate_dict, f"action_allow_rate/{lane_id}", step)
					logger.flush()

			if not exclude_warm_start or base_env.warm_start_finish:
				# Update episode reward
				episode_agents_reward.append(info["reward"])
				episode_global_reward.append(info["global_reward"])
			state = next_state
			step += 1

		eval_metrics = copy.deepcopy(info["simulation_metrics"])
		if is_eval_baseline:
			base_env.set_is_eval_baseline_flag(True)
			eval_episode_length_dummy, eval_metrics_dummy, _, _, eval_rewards_dummy, eval_global_rewards_dummy = \
				eval_episode(env, agent, config["eval"]["num_steps"], exclude_warm_start, is_dummy_action=True, reset_vehicles=False, reset_event_generator=False)
			base_env.set_is_eval_baseline_flag(False)
		else:
			eval_episode_length_dummy, eval_metrics_dummy, _, _, eval_rewards_dummy, eval_global_rewards_dummy = \
				eval_episode(env, agent, config["eval"]["num_steps"], exclude_warm_start, is_dummy_action=True, reset_vehicles=False, reset_event_generator=False)
		_log_eval_episode(
			logger, epi_num, info, eval_metrics, episode_agents_reward, episode_global_reward,
			eval_episode_length_dummy, eval_metrics_dummy, eval_rewards_dummy,
			eval_global_rewards_dummy, discount_factor, reward_metrics_methods
		)
		episode_agents_reward, episode_global_reward = [], []
		base_env.set_eval_flag(True, reset_vehicles=True, reset_event_generator=True)
		reset_env_training()

	env.close()
	return


def _run_worker_episodes_batch(env_pool, env_indices, agent, dummy_action, max_steps,
							   exclude_warm_start, is_dummy_action=False):
	states = {}
	statuses = {}
	rewards = {env_idx: [] for env_idx in env_indices}
	global_rewards = {env_idx: [] for env_idx in env_indices}
	last_infos = {}

	for env_idx in env_indices:
		state, status = env_pool.reset_one(env_idx)
		states[env_idx] = np.asarray(state)
		statuses[env_idx] = status

	active = list(env_indices)
	for _ in range(max_steps):
		actions = []
		for env_idx in active:
			if is_dummy_action or not statuses[env_idx]["warm_start_finish"]:
				action = dummy_action
			else:
				action = agent.get_action(state=states[env_idx], epsilon=0.0)
			actions.append(action)

		results = env_pool.step_indices(active, actions)
		next_active = []
		for env_idx, (next_state, reward, done, info) in zip(active, results):
			status = info["_worker_status"]
			last_infos[env_idx] = info
			if not exclude_warm_start or status["warm_start_finish"]:
				rewards[env_idx].append(info["reward"])
				global_rewards[env_idx].append(info["global_reward"])
			if not done:
				states[env_idx] = np.asarray(next_state)
				statuses[env_idx] = status
				next_active.append(env_idx)
		active = next_active
		if not active:
			break

	result = {}
	for env_idx in env_indices:
		info = last_infos.get(env_idx)
		if info is None:
			raise RuntimeError("Parallel eval worker produced no episode info.")
		result[env_idx] = (
			info,
			copy.deepcopy(info["simulation_metrics"]),
			rewards[env_idx],
			global_rewards[env_idx],
		)
	return result


def run_parallel_eval_loop(config: dict, logger, args: argparse.Namespace, num_envs: int):
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	config["eval"]["num_parallel_envs"] = num_envs

	local_config = make_local_env_config(config, int(args.port) + num_envs)
	env = _make_env_from_config(local_config)
	base_env = env.unwrapped
	env_pool = None

	try:
		env_pool = AsyncEnvPool(config, num_envs=num_envs, base_port=args.port, seed=args.seed)

		use_gpu, gpu_id = config["training"]["use_gpu"], int(config["training"]["gpu_id"])
		ptu.init_gpu(use_gpu=use_gpu, gpu_id=gpu_id)

		network_config = config["network"]
		training_config = config["training"]
		agent_class = AGENT_MAP[args.agent]
		agent = agent_class(
			env=base_env,
			network_config=network_config,
			training_config=training_config
		)
		try:
			agent.load(args.model_path)
		except Exception as e:
			print("Failed to load model. Error: {}, Trackback: {}".format(e, traceback.format_exc()))
			return

		discount_factor = float(training_config["discount"])
		reward_metrics_methods = ["avg", "50pt", "90pt"]
		total_episodes = int(config["eval"]["num_episodes"])
		is_eval_baseline = bool(config["eval"]["is_eval_baseline"])
		exclude_warm_start = bool(config["eval"]["exclude_warm_start"])
		num_steps = int(config["eval"]["num_steps"])
		dummy_action = np.ones((base_env.num_lanes, base_env.n_agents_per_lane), dtype=int)

		meta_text = "Env name: {}\n".format(config["simulation"]["env_name"])
		meta_text += "parallel_eval_envs: {}, ports: {}-{}\n".format(
			num_envs, int(args.port), int(args.port) + num_envs - 1
		)
		meta_text += "desired_rho: {}, desired_flow: {}, desired_velocity: {}, desired_traffic_condition: {}\n".\
					format(base_env.desired_rho, base_env.desired_flow, base_env.desired_velocity, base_env.desired_traffic_condition)
		meta_text += "grid_length: {}, control_rate: {}, density_level: {}\n".\
					format(base_env.grid_length, base_env.control_rate, base_env.density_level)
		meta_text += "event_generator: {}, event_generator_mode:{}, vehicle_generator: {}, fundamental_diagram_name:{}\n".\
					format(getattr(base_env.event_generator, "__name__", str(base_env.event_generator)), base_env.event_generator_mode, \
							getattr(base_env.vehicle_generator, "__name__", str(base_env.vehicle_generator)), base_env.fundamental_diagram_name)
		meta_text += "reward_gamma: {}, is_eval_baseline: {}, exclude_warm_start: {}\n".\
					format(discount_factor, is_eval_baseline, exclude_warm_start)
		logger.log_text(meta_text, "meta", 0)
		logger.flush()

		episode_id = 0
		with tqdm.tqdm(total=total_episodes, dynamic_ncols=True) as pbar:
			while episode_id < total_episodes:
				env_indices = list(range(min(num_envs, total_episodes - episode_id)))
				env_pool.set_eval_flag(True, reset_vehicles=True, reset_event_generator=True, indices=env_indices)
				agent_results = _run_worker_episodes_batch(
					env_pool, env_indices, agent, dummy_action, num_steps,
					exclude_warm_start, is_dummy_action=False
				)

				if is_eval_baseline:
					env_pool.set_is_eval_baseline_flag(True, indices=env_indices)
				env_pool.set_eval_flag(True, reset_vehicles=False, reset_event_generator=False, indices=env_indices)
				dummy_results = _run_worker_episodes_batch(
					env_pool, env_indices, agent, dummy_action, num_steps,
					exclude_warm_start, is_dummy_action=True
				)
				if is_eval_baseline:
					env_pool.set_is_eval_baseline_flag(False, indices=env_indices)

				for local_idx, env_idx in enumerate(env_indices):
					info, eval_metrics, rewards, global_rewards = agent_results[env_idx]
					dummy_info, dummy_metrics, dummy_rewards, dummy_global_rewards = dummy_results[env_idx]
					_log_eval_episode(
						logger, episode_id + local_idx, info, eval_metrics, rewards, global_rewards,
						int(dummy_info["end_time"]), dummy_metrics, dummy_rewards, dummy_global_rewards,
						discount_factor, reward_metrics_methods
					)
				episode_id += len(env_indices)
				pbar.update(len(env_indices))
	finally:
		if env_pool is not None:
			env_pool.close()
		try:
			env.close()
		except Exception:
			pass
	return



# %%
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config_template", "-cfg_template", type=str, default="dqn_basic")
	parser.add_argument("--config_file", "-cfg", type=str, required=True)
	parser.add_argument(
		"--agent",
		choices=["dqn", "ppo"],
		default="dqn",
		help="Which type of agent to run: 'dqn' (default) or 'ppo'."
	)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--log_interval", type=int, default=1000)
	parser.add_argument("--port", type=int, required=True)
	parser.add_argument("--model_path", type=str, required=True)
	parser.add_argument("--num_parallel_envs", type=int, default=None,
						help="Number of SUMO eval workers to run inside one checkpoint eval.")

	args = parser.parse_args()
	cfg_template = args.config_template
	if cfg_template not in configs.config_map:
		raise ValueError(f"Invalid config template: {cfg_template}")

	config, config_str = configs.config_map[cfg_template](args.config_file)
	if args.num_parallel_envs is not None:
		config["eval"]["num_parallel_envs"] = int(args.num_parallel_envs)
	_prefix_eval_result_dir(config, args.agent)
	logger = configs.make_logger(config)
	# write config_str to file
	config_str = json.loads(config_str)
	if args.num_parallel_envs is not None:
		config_str.setdefault("eval", {})["num_parallel_envs"] = int(args.num_parallel_envs)
	config_str["meta"] = config["meta"].copy()
	config_str["meta"]["seed"] = args.seed
	config_str["meta"]["model_path"] = args.model_path
	with open(os.path.join(config["meta"]["result_path"], "config.json"), "w") as f:
		json.dump(config_str, f, indent=4)
	config["simulation"]["traci_port"] = args.port
	run_eval_loop(config, logger, args)


if __name__ == "__main__":
	main()
