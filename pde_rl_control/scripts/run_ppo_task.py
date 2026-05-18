# %%
import os, time, argparse, json
import gymnasium as gym
import numpy as np
import torch
import tqdm

from pde_rl_control.agents.ppo import PPOAgent
import pde_rl_control.utils.pytorch_util as ptu
import pde_rl_control.configs as configs

from pde_rl_control.utils.eval import calculate_episode_reward, eval_episode
from pde_rl_control.utils.parallel_env import AsyncEnvPool, make_local_env_config
from pde_rl_control.utils.u import process_data_by_method
import pde_rl_control.environments

# %%
def _unwrap_reset(reset_output):
	return reset_output[0] if isinstance(reset_output, tuple) else reset_output


def _unwrap_step(step_output):
	if len(step_output) == 5:
		next_state, reward, terminated, truncated, info = step_output
		return next_state, reward, bool(terminated or truncated), info
	return step_output


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


def _empty_episode_reward_dict():
	return {"reward": [], "global_reward": [],
			"reward_spi": [], "global_reward_spi": [],
			"reward_los": [], "global_reward_los": []
			}


def _empty_trajectory():
	return {"states": [], "actions": [], "log_probs": [], "rewards": [], "next_states": [], "dones": []}


def _num_parallel_envs(config, args):
	if getattr(args, "num_parallel_envs", None) is not None:
		return int(args.num_parallel_envs)
	return int(config["simulation"].get(
		"num_parallel_envs",
		config["training"].get("num_parallel_envs", 1)
	))


def _average_action_allow_rates(action_rate_infos):
	result = {}
	counts = {}
	for action_rate in action_rate_infos:
		for lane_id, rate_dict in action_rate.items():
			if lane_id not in result:
				result[lane_id] = {}
				counts[lane_id] = {}
			for key, value in rate_dict.items():
				result[lane_id][key] = result[lane_id].get(key, 0.0) + float(value)
				counts[lane_id][key] = counts[lane_id].get(key, 0) + 1
	for lane_id, rate_dict in result.items():
		for key in rate_dict:
			rate_dict[key] /= counts[lane_id][key]
	return result


# %%
def run_training_loop(config: dict, logger, args: argparse.Namespace):
	num_envs = _num_parallel_envs(config, args)
	if num_envs > 1:
		return run_parallel_training_loop(config, logger, args, num_envs)

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
	agent = PPOAgent(
		env=base_env,
		network_config=network_config,
		training_config=training_config
	)

	state = None
	discount_factor = float(training_config["discount"])
	episode_reward_dict = _empty_episode_reward_dict()
	reward_metrics_methods = ["avg", "50pt", "90pt"]

	def reset_env_training():
		nonlocal state

		state = _unwrap_reset(env.reset())
		state = np.asarray(state)
		return

	reset_env_training()
	states_list, actions_list, log_probs_list, rewards_list, next_states_list, dones_list = [], [], [], [], [], []
	update_info = {}

	total_steps = int(config["training"]["total_steps"])
	# learning_start_steps = int(config["training"]["learning_starts"])
	# batch_size = int(config["training"]["batch_size"])
	evaluation_period = int(config["eval"]["evaluation_period"])
	is_eval_baseline = bool(config["eval"]["is_eval_baseline"])
	exclude_warm_start = bool(config["eval"]["exclude_warm_start"])
	model_save_dir = os.path.join(config["meta"]["result_path"], "models")
	if not (os.path.exists(model_save_dir)):
		os.makedirs(model_save_dir)
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
	for step in tqdm.trange(total_steps, dynamic_ncols=True):
		if not base_env.warm_start_finish:
			action = dummy_action
			log_prob = None
		else:
			action, log_prob = agent.sample_action(state=state)

		# Step the environment
		next_state, reward, done, info = _unwrap_step(env.step(action))

		if base_env.warm_start_finish and log_prob is not None:
			if step % args.log_interval == 0:
				# Log the action allow rate per lane
				action_allow_rate = info["action_allow_rate"]
				for lane_id, rate_dict in action_allow_rate.items():
					logger.log_scalars(rate_dict, f"action_allow_rate/{lane_id}", step)
				logger.flush()
			states_list.append(state)
			actions_list.append(action)
			log_probs_list.append(log_prob)
			rewards_list.append(reward)
			next_states_list.append(next_state)
			dones_list.append(done)

		if not exclude_warm_start or base_env.warm_start_finish:
			# Update episode reward
			for k in episode_reward_dict.keys():
				episode_reward_dict[k].append(info[k])

		# Handle episode termination
		if done:
			states_np = np.stack(states_list, axis=0)             # (ep_len, H, W, C)
			actions_np = np.stack(actions_list, axis=0)           # (ep_len, H, W)
			log_probs_np = np.stack(log_probs_list, axis=0)       # (ep_len, H, W)
			rewards_np = np.stack(rewards_list, axis=0)           # (ep_len, H, W)
			next_states_np = np.stack(next_states_list, axis=0)
			dones_np = np.stack(dones_list, axis=0).astype(np.float32)

			# Add a leading batch dimension
			states_np = states_np[np.newaxis, ...]       # (1, ep_len, H, W, C)
			actions_np = actions_np[np.newaxis, ...]     # (1, ep_len, H, W)
			log_probs_np = log_probs_np[np.newaxis, ...] # (1, ep_len, H, W)
			rewards_np = rewards_np[np.newaxis, ...]     # (1, ep_len, H, W)
			next_states_np = next_states_np[np.newaxis, ...]
			dones_np = dones_np[np.newaxis, ...]         # (1, ep_len)

			# Convert to Torch tensors
			states_tensor = ptu.from_numpy(states_np)
			actions_tensor = ptu.from_numpy(actions_np)
			log_probs_tensor = ptu.from_numpy(log_probs_np)
			rewards_tensor = ptu.from_numpy(rewards_np)
			next_states_tensor = ptu.from_numpy(next_states_np)
			dones_tensor = ptu.from_numpy(dones_np)
			# Perform PPO update
			update_info = agent.update(
					states=states_tensor,
					actions=actions_tensor,
					rewards=rewards_tensor,
					next_states=next_states_tensor,
					dones=dones_tensor,
					step=step,
					old_log_probs=log_probs_tensor
			)
			update_info["lr"] = agent.lr_scheduler.get_last_lr()[0]
			reset_env_training()
			has_episode_reward = False
			for k, v in episode_reward_dict.items():
				reward_metrics = {}
				avg_reward = calculate_episode_reward(v, discount_factor)
				if avg_reward is None:
					continue
				has_episode_reward = True
				for method in reward_metrics_methods:
					reward_metrics[method] = process_data_by_method(avg_reward, method)
				logger.log_scalars(reward_metrics, f"training_episode_{k}", step)
			if not has_episode_reward:
				logger.log_scalar(1, "training_episode_empty", step)
			episode_reward_dict = _empty_episode_reward_dict()
			if info["is_collision"]:
				logger.log_scalar(len(info["collision_vehicles"]), "train_collisions", step)
			else:
				logger.log_scalar(0, "train_collisions", step)
			
			logger.log_scalar(info["end_time"], "episode_length", step)
			logger.flush()

			# Reset the lists for the next episode
			states_list, actions_list, log_probs_list, rewards_list, next_states_list, dones_list = [], [], [], [], [], []

			# detector metrics
			detector_metrics = info["detector_metrics"]
			for k, v in detector_metrics.items():
				if isinstance(v, dict):
					for lane_id, lane_data in v.items():
						logger.log_scalars(lane_data, f'{k}/lane_{lane_id}', step)
			logger.flush()

			# metrics of the episode
			eval_metrics = info["simulation_metrics"]
			for k, v in eval_metrics.items():
				if isinstance(v, dict):
					logger.log_scalars(v, f'episode_metrics/{k}', step)
				else:
					logger.log_scalar(v, f'episode_metrics/{k}', step)
			logger.flush()
		else:
			state = next_state

		if step % args.log_interval == 0 and step > 0 and len(update_info) > 0:
			# Log the training metrics
			for k, v in update_info.items():
				logger.log_scalar(v, k, step)
			# logger.log_scalars(info["allow_rate"], "action_allow_rate", step)
			logger.log_model(agent.actor_critic, "critic", step)

			# Log batch data and replay buffer size
			ep_len = states_np.shape[1]  # (1, ep_len, H, W, C) -> ep_len
			indices = np.random.choice(ep_len, size=min(32, ep_len), replace=False)
			batch_np = {
				"observations": states_np[0][indices],  # (H, W, C)
				"actions": actions_np[0][indices],      # (H, W)
				"rewards": rewards_np[0][indices],      # (H, W)
				"next_observations": next_states_np[0][indices],  # (H, W, C)
				"dones": dones_np[0][indices]           # (H, W)
			}
			# Convert to PyTorch tensors
			if not is_add_graph:
				logger._summ_writer.add_graph(agent.actor_critic, torch.permute(ptu.from_numpy(batch_np["observations"]), (0, 3, 1, 2)))
				logger.flush()
				is_add_graph = True
			batch_state_avg = np.mean(batch_np["observations"], axis=(1, 2))
			batch_next_state_avg = np.mean(batch_np["next_observations"], axis=(1, 2))
			batch_action_avg = np.mean(batch_np["actions"], axis=(1, 2))
			batch_reward_avg = np.mean(batch_np["rewards"], axis=(1, 2))
			for dim in range(batch_state_avg.shape[1]):
				logger.log_histogram(batch_state_avg[:, dim], f"batch_state/dim_{dim}", step)
				logger.log_histogram(batch_next_state_avg[:, dim], f"batch_next_state/dim_{dim}", step)
			logger.log_histogram(batch_action_avg, f"batch_action", step)
			logger.log_histogram(batch_reward_avg, f"batch_reward", step)
			logger.log_histogram(batch_np["dones"], f"batch_done", step)
			logger.flush()

		if step % evaluation_period == 0 and step > 0:
			# save model
			agent.save(model_save_dir, step)
			# Evaluate the agent vs baseline (dummy action)
			eval_episode_length, eval_metrics, _, _, eval_rewards, _ = \
				eval_episode(env, agent, config["eval"]["num_steps"], exclude_warm_start)
			if is_eval_baseline:
				base_env.set_is_eval_baseline_flag(True)
				eval_episode_length_dummy, eval_metrics_dummy, _, _, eval_rewards_dummy, _ = \
					eval_episode(env, agent, config["eval"]["num_steps"], exclude_warm_start, is_dummy_action=True, reset_vehicles=False, reset_event_generator=False)
				base_env.set_is_eval_baseline_flag(False)
			else:
				eval_episode_length_dummy, eval_metrics_dummy, _, _, eval_rewards_dummy, _ = \
					eval_episode(env, agent, config["eval"]["num_steps"], exclude_warm_start, is_dummy_action=True, reset_vehicles=False, reset_event_generator=False)
			eval_agent_avg_reward = calculate_episode_reward(eval_rewards, discount_factor)
			eval_agent_avg_reward_dummy = calculate_episode_reward(eval_rewards_dummy, discount_factor)
			logger.log_scalar(eval_episode_length, "eval_metrics/episode_length", step)
			logger.log_scalar(eval_episode_length_dummy, "eval_metrics/episode_length:dummy", step)
			for k, v in eval_metrics.items():
				if isinstance(v, dict):
					# merge eval_metrics with eval_metrics_dummy
					for k2, v2 in eval_metrics_dummy[k].items():
						v[k2 + ":dummy"] = v2
					logger.log_scalars(v, f'eval_metrics/{k}', step)
				else:
					v_dummy = eval_metrics_dummy[k]
					v_dict = {k: v, k + ":dummy": v_dummy}
					logger.log_scalar(v_dict, f'eval_metrics/{k}', step)
			if eval_agent_avg_reward is not None:
				eval_agent_reward_metrics = {}
				for method in reward_metrics_methods:
					eval_agent_reward_metrics[method] = process_data_by_method(eval_agent_avg_reward, method)
					eval_agent_reward_metrics[method + ":dummy"] = process_data_by_method(eval_agent_avg_reward_dummy, method)
				logger.log_scalars(eval_agent_reward_metrics, "eval_metrics/rewards", step)
			else:
				print("Warning: eval_agent_avg_reward is None at step {}".format(step))
			logger.flush()
			base_env.set_eval_flag(False, reset_vehicles=True, reset_event_generator=True) # set the evaluation flag to False
			reset_env_training()
			states_list, actions_list, log_probs_list, rewards_list, next_states_list, dones_list = [], [], [], [], [], []
	env.close()
	return



def run_parallel_training_loop(config: dict, logger, args: argparse.Namespace, num_envs: int):
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	config["simulation"]["num_parallel_envs"] = num_envs

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
		agent = PPOAgent(
			env=base_env,
			network_config=network_config,
			training_config=training_config
		)

		states, env_statuses = env_pool.reset()
		states = [np.asarray(state) for state in states]
		trajectories = [_empty_trajectory() for _ in range(num_envs)]
		episode_reward_dicts = [_empty_episode_reward_dict() for _ in range(num_envs)]
		update_info = {}
		last_batch_np = None
		discount_factor = float(training_config["discount"])
		reward_metrics_methods = ["avg", "50pt", "90pt"]

		total_steps = int(config["training"]["total_steps"])
		evaluation_period = int(config["eval"]["evaluation_period"])
		is_eval_baseline = bool(config["eval"]["is_eval_baseline"])
		exclude_warm_start = bool(config["eval"]["exclude_warm_start"])
		model_save_dir = os.path.join(config["meta"]["result_path"], "models")
		if not os.path.exists(model_save_dir):
			os.makedirs(model_save_dir)

		dummy_action = np.ones((base_env.num_lanes, base_env.n_agents_per_lane), dtype=int)
		is_add_graph = False
		global_step = 0
		gradient_step = 0
		collected_steps = 0
		log_interval = int(args.log_interval)
		next_log_step = log_interval if log_interval > 0 else None
		next_evaluation_step = evaluation_period if evaluation_period > 0 else None

		meta_text = "Env name: {}\n".format(config["simulation"]["env_name"])
		meta_text += "parallel_envs: {}, ports: {}-{}\n".format(
			num_envs, int(args.port), int(args.port) + num_envs
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

		with tqdm.tqdm(total=total_steps, dynamic_ncols=True) as pbar:
			while global_step < total_steps:
				actions = []
				log_probs = []
				for env_idx, state in enumerate(states):
					if not env_statuses[env_idx]["warm_start_finish"]:
						action = dummy_action
						log_prob = None
					else:
						action, log_prob = agent.sample_action(state=state)
					actions.append(action)
					log_probs.append(log_prob)

				results = env_pool.step(actions)
				prev_global_step = global_step
				global_step += len(results)
				pbar.update(min(global_step, total_steps) - min(prev_global_step, total_steps))
				should_log = log_interval > 0 and global_step >= next_log_step
				action_rate_infos = []

				for env_idx, (next_state, reward, done, info) in enumerate(results):
					next_state = np.asarray(next_state)
					status = info["_worker_status"]
					warm_start_finish = status["warm_start_finish"]

					if warm_start_finish and log_probs[env_idx] is not None:
						if "action_allow_rate" in info:
							action_rate_infos.append(info["action_allow_rate"])
						trajectory = trajectories[env_idx]
						trajectory["states"].append(states[env_idx])
						trajectory["actions"].append(actions[env_idx])
						trajectory["log_probs"].append(log_probs[env_idx])
						trajectory["rewards"].append(reward)
						trajectory["next_states"].append(next_state)
						trajectory["dones"].append(done)
						collected_steps += 1

					if not exclude_warm_start or warm_start_finish:
						for key in episode_reward_dicts[env_idx].keys():
							episode_reward_dicts[env_idx][key].append(info[key])

					if done:
						trajectory = trajectories[env_idx]
						if trajectory["states"]:
							states_np = np.stack(trajectory["states"], axis=0)[np.newaxis, ...]
							actions_np = np.stack(trajectory["actions"], axis=0)[np.newaxis, ...]
							log_probs_np = np.stack(trajectory["log_probs"], axis=0)[np.newaxis, ...]
							rewards_np = np.stack(trajectory["rewards"], axis=0)[np.newaxis, ...]
							next_states_np = np.stack(trajectory["next_states"], axis=0)[np.newaxis, ...]
							dones_np = np.stack(trajectory["dones"], axis=0).astype(np.float32)[np.newaxis, ...]
							states_tensor = ptu.from_numpy(states_np)
							actions_tensor = ptu.from_numpy(actions_np)
							log_probs_tensor = ptu.from_numpy(log_probs_np)
							rewards_tensor = ptu.from_numpy(rewards_np)
							next_states_tensor = ptu.from_numpy(next_states_np)
							dones_tensor = ptu.from_numpy(dones_np)
							gradient_step += 1
							update_info = agent.update(
								states=states_tensor,
								actions=actions_tensor,
								rewards=rewards_tensor,
								next_states=next_states_tensor,
								dones=dones_tensor,
								step=global_step,
								old_log_probs=log_probs_tensor
							)
							update_info["lr"] = agent.lr_scheduler.get_last_lr()[0]
							ep_len = states_np.shape[1]
							indices = np.random.choice(ep_len, size=min(32, ep_len), replace=False)
							last_batch_np = {
								"observations": states_np[0][indices],
								"actions": actions_np[0][indices],
								"rewards": rewards_np[0][indices],
								"next_observations": next_states_np[0][indices],
								"dones": dones_np[0][indices]
							}

						has_episode_reward = False
						for key, values in episode_reward_dicts[env_idx].items():
							reward_metrics = {}
							avg_reward = calculate_episode_reward(values, discount_factor)
							if avg_reward is None:
								continue
							has_episode_reward = True
							for method in reward_metrics_methods:
								reward_metrics[method] = process_data_by_method(avg_reward, method)
							logger.log_scalars(reward_metrics, f"training_episode_{key}", global_step)
						if not has_episode_reward:
							logger.log_scalar(1, "training_episode_empty", global_step)
						episode_reward_dicts[env_idx] = _empty_episode_reward_dict()

						if info["is_collision"]:
							logger.log_scalar(len(info["collision_vehicles"]), "train_collisions", global_step)
						else:
							logger.log_scalar(0, "train_collisions", global_step)
						logger.log_scalar(info["end_time"], "episode_length", global_step)
						logger.flush()

						trajectories[env_idx] = _empty_trajectory()
						detector_metrics = info["detector_metrics"]
						for key, value in detector_metrics.items():
							if isinstance(value, dict):
								for lane_id, lane_data in value.items():
									logger.log_scalars(lane_data, f'{key}/lane_{lane_id}', global_step)
						logger.flush()

						eval_metrics = info["simulation_metrics"]
						for key, value in eval_metrics.items():
							if isinstance(value, dict):
								logger.log_scalars(value, f'episode_metrics/{key}', global_step)
							else:
								logger.log_scalar(value, f'episode_metrics/{key}', global_step)
						logger.flush()

						reset_state, reset_status = env_pool.reset_one(env_idx)
						states[env_idx] = np.asarray(reset_state)
						env_statuses[env_idx] = reset_status
					else:
						states[env_idx] = next_state
						env_statuses[env_idx] = status

				if should_log and action_rate_infos:
					avg_action_allow_rate = _average_action_allow_rates(action_rate_infos)
					for lane_id, rate_dict in avg_action_allow_rate.items():
						logger.log_scalars(rate_dict, f"action_allow_rate/{lane_id}", global_step)
					logger.flush()

				if should_log:
					if update_info and last_batch_np is not None:
						for key, value in update_info.items():
							logger.log_scalar(value, key, global_step)
						logger.log_model(agent.actor_critic, "critic", global_step)
						if not is_add_graph:
							logger._summ_writer.add_graph(agent.actor_critic, torch.permute(ptu.from_numpy(last_batch_np["observations"]), (0, 3, 1, 2)))
							logger.flush()
							is_add_graph = True
						batch_state_avg = np.mean(last_batch_np["observations"], axis=(1, 2))
						batch_next_state_avg = np.mean(last_batch_np["next_observations"], axis=(1, 2))
						batch_action_avg = np.mean(last_batch_np["actions"], axis=(1, 2))
						batch_reward_avg = np.mean(last_batch_np["rewards"], axis=(1, 2))
						for dim in range(batch_state_avg.shape[1]):
							logger.log_histogram(batch_state_avg[:, dim], f"batch_state/dim_{dim}", global_step)
							logger.log_histogram(batch_next_state_avg[:, dim], f"batch_next_state/dim_{dim}", global_step)
						logger.log_histogram(batch_action_avg, f"batch_action", global_step)
						logger.log_histogram(batch_reward_avg, f"batch_reward", global_step)
						logger.log_histogram(last_batch_np["dones"], f"batch_done", global_step)
						logger.log_scalar(collected_steps, "parallel/collected_steps", global_step)
						logger.log_scalar(gradient_step, "parallel/gradient_steps", global_step)
						logger.flush()
					while next_log_step <= global_step:
						next_log_step += log_interval

				if next_evaluation_step is not None and gradient_step > 0 and global_step >= next_evaluation_step:
					agent.save(model_save_dir, global_step)
					eval_episode_length, eval_metrics, _, _, eval_rewards, _ = \
						eval_episode(env, agent, config["eval"]["num_steps"], exclude_warm_start)
					if is_eval_baseline:
						base_env.set_is_eval_baseline_flag(True)
						eval_episode_length_dummy, eval_metrics_dummy, _, _, eval_rewards_dummy, _ = \
							eval_episode(env, agent, config["eval"]["num_steps"], exclude_warm_start, is_dummy_action=True, reset_vehicles=False, reset_event_generator=False)
						base_env.set_is_eval_baseline_flag(False)
					else:
						eval_episode_length_dummy, eval_metrics_dummy, _, _, eval_rewards_dummy, _ = \
							eval_episode(env, agent, config["eval"]["num_steps"], exclude_warm_start, is_dummy_action=True, reset_vehicles=False, reset_event_generator=False)
					eval_agent_avg_reward = calculate_episode_reward(eval_rewards, discount_factor)
					eval_agent_avg_reward_dummy = calculate_episode_reward(eval_rewards_dummy, discount_factor)
					logger.log_scalar(eval_episode_length, "eval_metrics/episode_length", global_step)
					logger.log_scalar(eval_episode_length_dummy, "eval_metrics/episode_length:dummy", global_step)
					for key, value in eval_metrics.items():
						if isinstance(value, dict):
							for key2, value2 in eval_metrics_dummy[key].items():
								value[key2 + ":dummy"] = value2
							logger.log_scalars(value, f'eval_metrics/{key}', global_step)
						else:
							value_dummy = eval_metrics_dummy[key]
							value_dict = {key: value, key + ":dummy": value_dummy}
							logger.log_scalar(value_dict, f'eval_metrics/{key}', global_step)
					if eval_agent_avg_reward is not None:
						eval_agent_reward_metrics = {}
						for method in reward_metrics_methods:
							eval_agent_reward_metrics[method] = process_data_by_method(eval_agent_avg_reward, method)
							eval_agent_reward_metrics[method + ":dummy"] = process_data_by_method(eval_agent_avg_reward_dummy, method)
						logger.log_scalars(eval_agent_reward_metrics, "eval_metrics/rewards", global_step)
					else:
						print("Warning: eval_agent_avg_reward is None at step {}".format(global_step))
					logger.flush()
					base_env.set_eval_flag(False, reset_vehicles=True, reset_event_generator=True)
					while next_evaluation_step <= global_step:
						next_evaluation_step += evaluation_period
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
	parser.add_argument("--config_template", "-cfg_template", type=str, default="ppo_basic")
	parser.add_argument("--config_file", "-cfg", type=str, required=True)

	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--log_interval", type=int, default=1000)
	parser.add_argument("--port", type=int, required=True)
	parser.add_argument("--num_parallel_envs", type=int, default=None,
						help="Number of SUMO training workers to run in parallel. Defaults to simulation.num_parallel_envs or 1.")

	args = parser.parse_args()
	cfg_template = args.config_template
	if cfg_template not in configs.config_map:
		raise ValueError(f"Invalid config template: {cfg_template}")

	config, config_str = configs.config_map[cfg_template](args.config_file)
	if args.num_parallel_envs is not None:
		config["simulation"]["num_parallel_envs"] = int(args.num_parallel_envs)
	logger = configs.make_logger(config)
	# write config_str to file
	config_str = json.loads(config_str)
	if args.num_parallel_envs is not None:
		config_str.setdefault("simulation", {})["num_parallel_envs"] = int(args.num_parallel_envs)
	config_str["meta"] = config["meta"].copy()
	config_str["meta"]["seed"] = args.seed
	with open(os.path.join(config["meta"]["result_path"], "config.json"), "w") as f:
		json.dump(config_str, f, indent=4)
	config["simulation"]["traci_port"] = args.port
	run_training_loop(config, logger, args)


if __name__ == "__main__":
	main()
