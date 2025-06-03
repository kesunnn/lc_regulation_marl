# %%
import os
import time
import argparse
import json
import gym
import numpy as np
import torch
import tqdm

from pde_rl_control.agents.ppo import PPOAgent
import pde_rl_control.utils.pytorch_util as ptu
import pde_rl_control.configs as configs

from pde_rl_control.utils.eval import calculate_episode_reward, eval_episode

# %%
def run_training_loop(config: dict, logger, args: argparse.Namespace):
    # 1. Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_gpu, gpu_id = config["training"]["use_gpu"], int(config["training"]["gpu_id"])
    ptu.init_gpu(use_gpu=use_gpu, gpu_id=gpu_id)

    # 2. Create the Gym environment
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

    # 3. Instantiate the PPO agent
    network_config = config["network"]
    training_config = config["training"]
    agent = PPOAgent(
        env=env,
        network_config=network_config,
        training_config=training_config
    )

    # 4. Logging setup
    total_updates = int(training_config["total_steps"])    # interpret as # of episodes
    eval_interval = int(config["eval"]["evaluation_period"])
    is_eval_baseline = bool(config["eval"]["is_eval_baseline"])
    exclude_warm_start = bool(config["eval"]["exclude_warm_start"])

    model_save_dir = os.path.join(config["meta"]["result_path"], "models")
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # 5. Meta information for TensorBoard
    meta_text = f"Env name: {config['simulation']['env_name']}\n"
    meta_text += f"grid_length: {env.grid_length}, control_rate: {env.control_rate}, density_level: {env.density_level}\n"
    meta_text += f"event_generator: {getattr(env.event_generator, '__name__', str(env.event_generator))}, "
    meta_text += f"vehicle_generator: {getattr(env.vehicle_generator, '__name__', str(env.vehicle_generator))}\n"
    meta_text += f"PPO discount: {agent.discount}, GAE lambda: {agent.gae_lambda}, clip_epsilon: {agent.clip_epsilon}\n"
    logger.log_text(meta_text, "meta", 0)
    logger.flush()

    global_step = 0
    for update_idx in tqdm.trange(total_updates, dynamic_ncols=True):
        # 6. Collect exactly one episode (no replay buffer)
        states_list, actions_list, rewards_list, next_states_list, dones_list = [], [], [], [], []

        state = env.reset()
        if isinstance(state, tuple):
            # Ensure state is a single array (old Gym API)
            state = state[0]
        state = np.asarray(state)

        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done:
            # 6.1. Get action (no exploration epsilon by default, can be scheduled if you like)
            action = agent.get_action(state, epsilon=0.0)

            next_state, reward, done, info = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            next_state = np.asarray(next_state)

            # 6.2. Store to trajectory buffers
            states_list.append(state)
            actions_list.append(action)
            rewards_list.append(reward)
            next_states_list.append(next_state)
            dones_list.append(done)

            state = next_state
            episode_reward += np.mean(reward)  # average reward over grid
            episode_length += 1

        # 7. Convert lists → NumPy arrays, then to Torch tensors
        # Shapes: (1, ep_len, H, W, C) / (1, ep_len, H, W)
        ep_len = len(states_list)
        states_np = np.stack(states_list, axis=0)        # (ep_len, H, W, C)
        actions_np = np.stack(actions_list, axis=0)      # (ep_len, H, W)
        rewards_np = np.stack(rewards_list, axis=0)      # (ep_len, H, W)
        next_states_np = np.stack(next_states_list, axis=0)
        dones_np = np.stack(dones_list, axis=0).astype(np.float32)  # treat done as float

        # Add a leading batch dimension
        states_np = states_np[np.newaxis, ...]       # (1, ep_len, H, W, C)
        actions_np = actions_np[np.newaxis, ...]     # (1, ep_len, H, W)
        rewards_np = rewards_np[np.newaxis, ...]     # (1, ep_len, H, W)
        next_states_np = next_states_np[np.newaxis, ...]
        dones_np = dones_np[np.newaxis, ...]         # (1, ep_len, H, W)

        # Convert to Torch tensors
        states_tensor = ptu.from_numpy(states_np)
        actions_tensor = ptu.from_numpy(actions_np)
        rewards_tensor = ptu.from_numpy(rewards_np)
        next_states_tensor = ptu.from_numpy(next_states_np)
        dones_tensor = ptu.from_numpy(dones_np)

        # 8. Perform a PPO update on this single‐episode batch
        update_info = agent.update(
            states=states_tensor,
            actions=actions_tensor,
            rewards=rewards_tensor,
            next_states=next_states_tensor,
            dones=dones_tensor,
            step=global_step
        )

        # 9. Logging: training losses, entropy, etc.
        update_info["episode_reward"] = episode_reward
        update_info["episode_length"] = episode_length
        logger.log_scalars(update_info, "ppo_train", global_step)
        logger.flush()

        # 10. Periodically save & evaluate
        if (update_idx + 1) % eval_interval == 0:
            # 10.1 Save model
            agent.save(model_save_dir, global_step)

            # 10.2 Evaluate agent for one episode (vs. baseline if requested)
            eval_len, eval_metrics, _, _, eval_rewards, _ = eval_episode(
                env, agent, config["eval"]["num_steps"], exclude_warm_start
            )

            if is_eval_baseline:
                env.set_is_eval_baseline_flag(True)
                eval_len_dummy, eval_metrics_dummy, _, _, eval_rewards_dummy, _ = eval_episode(
                    env, agent, config["eval"]["num_steps"],
                    exclude_warm_start, is_dummy_action=True,
                    reset_vehicles=False, reset_event_generator=False
                )
                env.set_is_eval_baseline_flag(False)
            else:
                eval_len_dummy, eval_metrics_dummy, _, _, eval_rewards_dummy, _ = eval_episode(
                    env, agent, config["eval"]["num_steps"],
                    exclude_warm_start, is_dummy_action=True,
                    reset_vehicles=False, reset_event_generator=False
                )

            # 10.3 Compute discounted returns for plotting (optional)
            eval_agent_return = calculate_episode_reward(eval_rewards, agent.discount)
            eval_dummy_return = calculate_episode_reward(eval_rewards_dummy, agent.discount)

            # 10.4 Log evaluation metrics
            # Episode length
            logger.log_scalar(eval_len, "eval/episode_length", global_step)
            logger.log_scalar(eval_len_dummy, "eval/episode_length:dummy", global_step)

            # For each metric (which may be dict or scalar)
            for k, v in eval_metrics.items():
                if isinstance(v, dict):
                    # merge with dummy
                    for subk, subv in eval_metrics_dummy[k].items():
                        v[subk + ":dummy"] = subv
                    logger.log_scalars(v, f"eval/{k}", global_step)
                else:
                    v_dummy = eval_metrics_dummy[k]
                    logger.log_scalar({k: v, k + ":dummy": v_dummy}, f"eval/{k}", global_step)

            # Log returns
            logger.log_scalar(eval_agent_return, "eval/discounted_return", global_step)
            logger.log_scalar(eval_dummy_return, "eval/discounted_return:dummy", global_step)
            logger.flush()

            env.set_eval_flag(False, reset_vehicles=True, reset_event_generator=True)

        # 11. Advance global_step
        global_step += 1

    env.close()
    return


# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_template", "-cfg_template", type=str, default="ppo_basic")
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--port", type=int, required=True)

    args = parser.parse_args()
    cfg_template = args.config_template
    if cfg_template not in configs.config_map:
        raise ValueError(f"Invalid config template: {cfg_template}")

    config, config_str = configs.config_map[cfg_template](args.config_file)
    logger = configs.make_logger(config)

    # Write out final config (with seed) to config.json
    config_str = json.loads(config_str)
    config_str["meta"] = config["meta"].copy()
    config_str["meta"]["seed"] = args.seed
    with open(os.path.join(config["meta"]["result_path"], "config.json"), "w") as f:
        json.dump(config_str, f, indent=4)

    # Inject TraCI port
    config["simulation"]["traci_port"] = args.port

    # Call the PPO training loop
    run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()
