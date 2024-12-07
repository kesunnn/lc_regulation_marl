import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import animation
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

def calculate_episode_reward(step_rewards, gamma):
    """
    Calculate the reward of an episode
    Args:
        step_rewards (list): list of rewards of each step
        gamma (float): discount factor
    Returns:
        float: the total reward of the episode
    """
    assert 0 < gamma <= 1, "Gamma should be in (0, 1]"
    if len(step_rewards) == 0:
        return None
    result = np.zeros(step_rewards[0].shape)
    for t in reversed(range(len(step_rewards))):
        result = step_rewards[t] + gamma * result
    return result

def eval_episode(env, agent, max_steps, exclude_warm_start, is_dummy_action=False, reset_vehicles=True, reset_event_generator=True):
    """
    Evaluate an episode
    Args:
        env (gym.Env): the environment
        agent (Agent): the agent
        max_steps (int): maximum number of steps
        gamma (float): discount factor
        exclude_warm_start (bool): whether to exclude the warm start phase when calculating metrics
        is_dummy_action (bool): whether to use dummy action
    Returns:
        float: the total reward of the episode
    """
    dummy_action = np.ones((env.num_lanes, env.n_agents_per_lane), dtype=int)
    actions, states, rewards, global_rewards = [], [], [], []
    env.set_eval_flag(True, reset_vehicles, reset_event_generator) # set the evaluation flag to True
    state = env.reset()
    for step in range(max_steps):
        if not env.warm_start_finish or is_dummy_action:
            action = dummy_action
        else:
            action = agent.get_action(state=state, epsilon=0.0)
        next_state, reward, done, info = env.step(action)
        agent_reward, global_reward = info["reward"], info["global_reward"]
        if not exclude_warm_start or env.warm_start_finish:
            rewards.append(agent_reward)
            global_rewards.append(global_reward)
            actions.append(action)
            states.append(state)
        if done:
            break
        state = next_state
    # metrics of the episode
    eval_metrics = env.get_current_simulation_metrics()
    episode_length = int(info["end_time"])
    env.set_eval_flag(False, reset_vehicles, reset_event_generator) # set the evaluation flag to False
    return episode_length, eval_metrics, actions, states, rewards, global_rewards
