# %%
import os, time, argparse, json
import gym
import numpy as np
import torch
import tqdm

from pde_rl_control.agents.dqn import DQNAgent
import pde_rl_control.utils.pytorch_util as ptu
import pde_rl_control.configs as configs

from pde_rl_control.utils.replay_buffer import ReplayBuffer
from pde_rl_control.utils.eval import calculate_episode_reward, eval_episode
from pde_rl_control.utils.u import process_data_by_method
import pde_rl_control.environments

# %%
def run_training_loop(config: dict, logger, args: argparse.Namespace):
	# set random seeds
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	use_gpu, gpu_id = config["training"]["use_gpu"], int(config["training"]["gpu_id"])
	ptu.init_gpu(use_gpu=use_gpu, gpu_id=gpu_id)

	# make the gym environment
	env = gym.make(
		config["simulation"]["env_name"],
		disable_env_checker=True,
		grid_length=config["simulation"]["grid_length"],
		control_rate=config["simulation"]["control_rate"],
		density_level=config["simulation"]["density_level"],
		event_generator=config["simulation"]["event_generator"],
		vehicle_generator=config["simulation"]["vehicle_generator"],
		config=config
	)
	exploration_schedule = config["training"]["exploration_schedule"]

	# create training agent
	network_config = config["network"]
	training_config = config["training"]
	agent = DQNAgent(
		env=env,
		network_config=network_config,
		training_config=training_config
	)

	# ep_len = env.spec.max_episode_steps

	state = None
	discount_factor = float(training_config["discount"])
	episode_reward_dict = {"reward": [], "global_reward": [],
						"reward_spi": [], "global_reward_spi": [],
						"reward_los": [], "global_reward_los": []
						}
	# episode_agents_reward, episode_global_reward = [], []
	# reward_metrics_methods = ["avg", "50pt", "90pt", "99pt", "min", "max"]
	reward_metrics_methods = ["avg", "50pt", "90pt"]

	# Replay buffer
	replay_buffer = ReplayBuffer(capacity=int(config["training"]["replay_buffer_size"]))

	def reset_env_training():
		nonlocal state

		state = env.reset()
		assert not isinstance(state, tuple), "env.reset() must return np.ndarray - make sure your Gym version uses the old step API"
		state = np.asarray(state)
		return

	reset_env_training()

	total_steps = int(config["training"]["total_steps"])
	learning_start_steps = int(config["training"]["learning_starts"])
	batch_size = int(config["training"]["batch_size"])
	evaluation_period = int(config["eval"]["evaluation_period"])
	is_eval_baseline = bool(config["eval"]["is_eval_baseline"])
	exclude_warm_start = bool(config["eval"]["exclude_warm_start"])
	model_save_dir = os.path.join(config["meta"]["result_path"], "models")
	if not (os.path.exists(model_save_dir)):
		os.makedirs(model_save_dir)
	# dummy action used in warm_start phase
	dummy_action = np.ones((env.num_lanes, env.n_agents_per_lane), dtype=int)
	is_add_graph = False
	# log key information text in the tensorboard
	meta_text = "Env name: {}\n".format(config["simulation"]["env_name"])
	meta_text += "desired_rho: {}, desired_flow: {}, desired_velocity: {}, desired_traffic_condition: {}\n".\
				format(env.desired_rho, env.desired_flow, env.desired_velocity, env.desired_traffic_condition)
	meta_text += "grid_length: {}, control_rate: {}, density_level: {}\n".\
				format(env.grid_length, env.control_rate, env.density_level)
	meta_text += "event_generator: {}, event_generator_mode:{}, vehicle_generator: {}, fundamental_diagram_name:{}\n".\
				format(getattr(env.event_generator, "__name__", str(env.event_generator)), env.event_generator_mode, \
						getattr(env.vehicle_generator, "__name__", str(env.vehicle_generator)), env.fundamental_diagram_name)
	meta_text += "reward_gamma: {}, is_eval_baseline: {}, exclude_warm_start: {}\n".\
				format(discount_factor, is_eval_baseline, exclude_warm_start)
	logger.log_text(meta_text, "meta", 0)
	logger.flush()
	for step in tqdm.trange(total_steps, dynamic_ncols=True):
		assert learning_start_steps * float(config["simulation"]["delta_T"]) > env.warm_start_time, "Learning starts should be larger than warm start steps to avoid empty replay buffer"
		if not env.warm_start_finish:
			action = dummy_action
		else:
			epsilon = exploration_schedule.value(step)
			# Compute action
			action = agent.get_action(state=state, epsilon=epsilon)

		# Step the environment
		next_state, reward, done, info = env.step(action)

		if env.warm_start_finish:
			# Add the data to the replay buffer
			replay_buffer.insert(state, action, reward, next_state, done)
			if step >= learning_start_steps and step % args.log_interval == 0 and epsilon < 1e-1:
				# Log the action allow rate per lane
				action_allow_rate = info["action_allow_rate"]
				for lane_id, rate_dict in action_allow_rate.items():
					logger.log_scalars(rate_dict, f"action_allow_rate/{lane_id}", step)
				logger.flush()

		if not exclude_warm_start or env.warm_start_finish:
			# Update episode reward
			for k in episode_reward_dict.keys():
				episode_reward_dict[k].append(info[k])
			# episode_agents_reward.append(info["reward"])
			# episode_global_reward.append(info["global_reward"])

		# Handle episode termination
		if done:
			reset_env_training()
			# agent_avg_reward = calculate_episode_reward(episode_agents_reward, discount_factor)
			# global_avg_reward = calculate_episode_reward(episode_global_reward, discount_factor)
			for k, v in episode_reward_dict.items():
				reward_metrics = {}
				avg_reward = calculate_episode_reward(v, discount_factor)
				for method in reward_metrics_methods:
					reward_metrics[method] = process_data_by_method(avg_reward, method)
				logger.log_scalars(reward_metrics, f"training_episode_{k}", step)
			episode_reward_dict = {"reward": [], "global_reward": [],
						"reward_spi": [], "global_reward_spi": [],
						"reward_los": [], "global_reward_los": []
						}
			# episode_agents_reward, episode_global_reward = [], []
			if info["is_collision"]:
				logger.log_scalar(len(info["collision_vehicles"]), "train_collisions", step)
			
			logger.log_scalar(info["end_time"], "episode_length", step)
			# episode reward
			# if agent_avg_reward is not None:
			# 	agent_reward_metrics = {}
			# 	for method in reward_metrics_methods:
			# 		agent_reward_metrics[method] = process_data_by_method(agent_avg_reward, method)
			# 	logger.log_scalars(agent_reward_metrics, "training_episode_agent_reward", step)
			# else:
			# 	print("Warning: agent_avg_reward is None at step {}".format(step))
			# if global_avg_reward is not None:
			# 	global_reward_metrics = {}
			# 	for method in reward_metrics_methods:
			# 		global_reward_metrics[method] = process_data_by_method(global_avg_reward, method)
			# 	logger.log_scalars(global_reward_metrics, "training_episode_global_reward", step)
			# else:
			# 	print("Warning: global_avg_reward is None at step {}".format(step))
			logger.flush()

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

		# Main DQN training loop
		if step >= learning_start_steps:
			# Sample config["batch_size"] samples from the replay buffer
			batch_np = replay_buffer.sample(batch_size)

			# Convert to PyTorch tensors
			batch = ptu.from_numpy(batch_np)
			if not is_add_graph:
				test_batch = ptu.from_numpy(replay_buffer.sample(5))
				logger._summ_writer.add_graph(agent.critic, torch.permute(test_batch["observations"], (0, 3, 1, 2)))
				logger.flush()
				is_add_graph = True

			#Train the agent. `batch` is a dictionary of numpy arrays
			update_info = agent.update(
				state=batch["observations"],
				action=batch["actions"],
				reward=batch["rewards"],
				next_state=batch["next_observations"],
				done=batch["dones"],
				step=step
			)

			# Logging code
			update_info["epsilon"] = epsilon
			update_info["lr"] = agent.lr_scheduler.get_last_lr()[0]

			if step % args.log_interval == 0:
				# Log the training metrics
				for k, v in update_info.items():
					logger.log_scalar(v, k, step)
				# logger.log_scalars(info["allow_rate"], "action_allow_rate", step)
				logger.log_model(agent.critic, "critic", step)
				# Log batch data and replay buffer size
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
				logger.log_scalar(len(replay_buffer), "replay_buffer_size", step)
				logger.flush()

			if step % evaluation_period == 0:
				# save model
				agent.save(model_save_dir, step)
				# Evaluate the agent vs baseline (dummy action)
				eval_episode_length, eval_metrics, _, _, eval_rewards, _ = \
					eval_episode(env, agent, config["eval"]["num_steps"], exclude_warm_start)
				if is_eval_baseline:
					env.set_is_eval_baseline_flag(True)
					eval_episode_length_dummy, eval_metrics_dummy, _, _, eval_rewards_dummy, _ = \
						eval_episode(env, agent, config["eval"]["num_steps"], exclude_warm_start, is_dummy_action=True, reset_vehicles=False, reset_event_generator=False)
					env.set_is_eval_baseline_flag(False)
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
				env.set_eval_flag(False, reset_vehicles=True, reset_event_generator=True) # set the evaluation flag to False
				reset_env_training()
	env.close()
	return



# %%
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config_template", "-cfg_template", type=str, default="dqn_basic")
	parser.add_argument("--config_file", "-cfg", type=str, required=True)

	# parser.add_argument("--eval_interval", "-ei", type=int, default=10000)
	# parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
	# parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--log_interval", type=int, default=1000)
	parser.add_argument("--port", type=int, required=True)

	args = parser.parse_args()
	cfg_template = args.config_template
	if cfg_template not in configs.config_map:
		raise ValueError(f"Invalid config template: {cfg_template}")

	config, config_str = configs.config_map[cfg_template](args.config_file)
	logger = configs.make_logger(config)
	# write config_str to file
	config_str = json.loads(config_str)
	config_str["meta"] = config["meta"].copy()
	config_str["meta"]["seed"] = args.seed
	with open(os.path.join(config["meta"]["result_path"], "config.json"), "w") as f:
		json.dump(config_str, f, indent=4)
	config["simulation"]["traci_port"] = args.port
	run_training_loop(config, logger, args)


if __name__ == "__main__":
	main()



