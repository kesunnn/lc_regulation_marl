# %%
from typing import Sequence, Callable, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np

# %%
import pde_rl_control.utils.pytorch_util as ptu

# %%
class ActorCritic(nn.Module):
	kernel_dims = [32, 64, 64]
	def __init__(self, state_shape, n_actions, network_config):
		super(ActorCritic, self).__init__()
		self.receptive_field = 1
		_, _, input_channels = state_shape
		self.kernel_dims = network_config['kernel_dims'] if 'kernel_dims' in network_config else self.kernel_dims
		
		# Shared feature extraction layers
		shared_layers = []
		shared_layers.append(nn.Conv2d(input_channels, self.kernel_dims[0], kernel_size=3, stride=1, padding=1))
		shared_layers.append(nn.ReLU())
		self.receptive_field += 1
		for i in range(len(self.kernel_dims)-1):
			shared_layers.append(nn.Conv2d(self.kernel_dims[i], self.kernel_dims[i+1], kernel_size=3, stride=1, padding=1))
			self.receptive_field += 1
			shared_layers.append(nn.ReLU())
		
		self.shared_layers = nn.Sequential(*shared_layers)
		
		# Actor head (policy)
		self.actor_head = nn.Conv2d(self.kernel_dims[-1], n_actions, kernel_size=1, stride=1)
		
		# Critic head (value function)
		self.critic_head = nn.Conv2d(self.kernel_dims[-1], 1, kernel_size=1, stride=1)
		
		return

	def forward(self, x):
		shared_features = self.shared_layers(x)
		policy_logits = self.actor_head(shared_features)
		values = self.critic_head(shared_features).squeeze(1)  # Remove action dimension
		return policy_logits, values

	def get_action_and_value(self, x):
		policy_logits, values = self.forward(x)
		# Convert logits to probability distribution
		batch_size, n_actions, height, width = policy_logits.shape
		policy_logits_flat = policy_logits.permute(0, 2, 3, 1).reshape(-1, n_actions)  # (batch_size * H * W, n_actions)
		dist = Categorical(logits=policy_logits_flat)
		action_flat = dist.sample()
		action = action_flat.reshape(batch_size, height, width)  # (batch_size, H, W)
		log_prob_flat = dist.log_prob(action_flat)
		log_prob = log_prob_flat.reshape(batch_size, height, width)  # (batch_size, H, W)
		return action, log_prob, values

	def get_value(self, x):
		_, values = self.forward(x)
		return values

	def evaluate_actions(self, x, actions):
		policy_logits, values = self.forward(x)
		batch_size, n_actions, height, width = policy_logits.shape
		policy_logits_flat = policy_logits.permute(0, 2, 3, 1).reshape(-1, n_actions)  # (batch_size * H * W, n_actions)
		actions_flat = actions.reshape(-1)  # (batch_size * H * W)
		dist = Categorical(logits=policy_logits_flat)
		log_prob_flat = dist.log_prob(actions_flat)
		entropy_flat = dist.entropy()
		log_prob = log_prob_flat.reshape(batch_size, height, width)  # (batch_size, H, W)
		entropy = entropy_flat.reshape(batch_size, height, width)  # (batch_size, H, W)
		return log_prob, entropy, values


# %%
class PPOAgent(nn.Module):
	learning_rate: float = 3e-4
	adam_eps: float = 1e-5
	discount: float = 0.99
	gae_lambda: float = 0.95
	clip_epsilon: float = 0.2
	value_loss_coeff: float = 0.5
	entropy_coeff: float = 0.01
	clip_grad_norm: float = 0.5
	ppo_epochs: int = 10
	lr_scheduler_mode: str = "constant"

	def __init__(self, env, network_config, training_config):
		super(PPOAgent, self).__init__()

		self.env = env
		self.network_config = network_config
		self.training_config = training_config

		self.num_actions = env.num_actions
		self.state_shape = env.state_shape # (H, W, C)
		self.actor_critic = ptu.nn_to_device(ActorCritic(self.state_shape, self.num_actions, network_config))

		self.discount = training_config["discount"] if "discount" in training_config else self.discount
		self.gae_lambda = training_config["gae_lambda"] if "gae_lambda" in training_config else self.gae_lambda
		self.clip_epsilon = training_config["clip_epsilon"] if "clip_epsilon" in training_config else self.clip_epsilon
		self.value_loss_coeff = training_config["value_loss_coeff"] if "value_loss_coeff" in training_config else self.value_loss_coeff
		self.entropy_coeff = training_config["entropy_coeff"] if "entropy_coeff" in training_config else self.entropy_coeff
		self.clip_grad_norm = training_config["clip_grad_norm"] if "clip_grad_norm" in training_config else self.clip_grad_norm
		self.ppo_epochs = training_config["ppo_epochs"] if "ppo_epochs" in training_config else self.ppo_epochs
		self.lr_scheduler_mode = training_config["lr_scheduler_mode"] if "lr_scheduler_mode" in training_config else self.lr_scheduler_mode
		self.learning_rate = training_config["learning_rate"] if "learning_rate" in training_config else self.learning_rate
		self.adam_eps = training_config["adam_eps"] if "adam_eps" in training_config else self.adam_eps

		self.optimizer = self._make_optimizer(self.actor_critic.parameters())
		self.lr_scheduler = self._make_lr_schedule(self.optimizer)

	def _make_optimizer(self, params):
		return torch.optim.Adam(params, lr=self.learning_rate, eps=self.adam_eps)
	
	def _make_lr_schedule(self, optimizer):
		lr_scheduler_params = self.training_config.get("lr_scheduler_params", {})
		if self.lr_scheduler_mode == "constant":
			_factor = lr_scheduler_params.get("factor", 1.0)
			return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=_factor)
		elif self.lr_scheduler_mode == "exponential":
			_gamma = lr_scheduler_params.get("gamma", 0.999)
			return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=_gamma)
		elif callable(self.lr_scheduler_mode):
			return self.lr_scheduler_mode(optimizer)
		else:
			raise ValueError(f"Invalid lr_scheduler: {self.lr_scheduler_mode}")

	def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
		"""
		Used for evaluation.
		"""
		assert state.shape == self.state_shape, f"Invalid state shape: {state.shape} != {self.state_shape}"
		if np.random.rand() < epsilon:
			action = np.random.randint(0, self.num_actions, size=(self.state_shape[0], self.state_shape[1]))
		else:
			state_tensor = ptu.from_numpy(np.transpose(state, (2, 0, 1))).unsqueeze(0) # (H, W, C) -> (1, C, H, W)
			with torch.no_grad():
				policy_logits, _ = self.actor_critic(state_tensor) # (1, num_actions, H, W)
			action = policy_logits.argmax(dim=1).detach().cpu().numpy().squeeze(0) # (1, H, W) -> (H, W)
		return action

	def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Compute Generalized Advantage Estimation (GAE).
		
		If dones only indicate truncation (not termination):
		- We should NOT zero out the bootstrap value when done=True
		- The episode continues, we just don't have the next observation
		- Use the predicted value function as bootstrap instead of 0
		"""
		batch_size, seq_len = rewards.shape[:2]
		
		advantages = torch.zeros_like(rewards)
		returns = torch.zeros_like(rewards)
		
		gae = 0
		for t in reversed(range(seq_len)):
			if t == seq_len - 1:
				next_value = next_values
			else:
				next_value = values[:, t + 1]
			
			# Key change: If done only means truncation, don't zero out next_value
			# The environment continues, we just truncated the episode for practical reasons
			delta = rewards[:, t] + self.discount * next_value - values[:, t]
			gae = delta + self.discount * self.gae_lambda * gae
			
			advantages[:, t] = gae
			returns[:, t] = advantages[:, t] + values[:, t]
		
		return advantages, returns

	def update_actor_critic(self, states: torch.Tensor, actions: torch.Tensor, old_log_probs: torch.Tensor, 
						   returns: torch.Tensor, advantages: torch.Tensor) -> dict:
		"""Update the actor-critic network using PPO."""
		
		# Normalize advantages
		advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
		
		total_policy_loss = 0
		total_value_loss = 0
		total_entropy_loss = 0
		total_loss = 0
		
		for epoch in range(self.ppo_epochs):
			# Get current policy and value predictions
			states_input = torch.permute(states, (0, 3, 1, 2))  # (batch_size, H, W, C) -> (batch_size, C, H, W)
			log_probs, entropy, values = self.actor_critic.evaluate_actions(states_input, actions)
			
			# Compute policy loss with clipping
			ratio = torch.exp(log_probs - old_log_probs)
			surr1 = ratio * advantages
			surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
			policy_loss = -torch.min(surr1, surr2).mean()
			
			# Compute value loss
			value_loss = F.mse_loss(values, returns)
			
			# Compute entropy loss
			entropy_loss = -entropy.mean()
			
			# Combined loss
			loss = policy_loss + self.value_loss_coeff * value_loss + self.entropy_coeff * entropy_loss
			
			# Update network
			self.optimizer.zero_grad()
			loss.backward()
			grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
				self.actor_critic.parameters(), self.clip_grad_norm or float("inf")
			)
			self.optimizer.step()
			
			total_policy_loss += policy_loss.item()
			total_value_loss += value_loss.item()
			total_entropy_loss += entropy_loss.item()
			total_loss += loss.item()
		
		self.lr_scheduler.step()
		
		return {
			"policy_loss": total_policy_loss / self.ppo_epochs,
			"value_loss": total_value_loss / self.ppo_epochs,
			"entropy_loss": total_entropy_loss / self.ppo_epochs,
			"total_loss": total_loss / self.ppo_epochs,
			"values": values.mean().item(),
			"returns": returns.mean().item(),
			"advantages": advantages.mean().item(),
			"grad_norm": grad_norm.item(),
		}

	def update(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, 
			   next_states: torch.Tensor, dones: torch.Tensor, step: int) -> dict:
		"""
		Update the PPO agent using trajectory data.
		Expects trajectory data with shape (batch_size, seq_len, ...).
		"""
		batch_size, seq_len = rewards.shape[:2]
		
		# Get values for all states
		states_input = torch.permute(states, (0, 1, 4, 2, 3))  # (batch_size, seq_len, H, W, C) -> (batch_size, seq_len, C, H, W)
		states_flat = states_input.reshape(-1, *states_input.shape[2:])  # (batch_size * seq_len, C, H, W)
		next_states_input = torch.permute(next_states, (0, 1, 4, 2, 3))
		next_states_flat = next_states_input.reshape(-1, *next_states_input.shape[2:])
		
		with torch.no_grad():
			values_flat = self.actor_critic.get_value(states_flat)  # (batch_size * seq_len, H, W)
			next_values_flat = self.actor_critic.get_value(next_states_flat)  # (batch_size * seq_len, H, W)
			
			values = values_flat.reshape(batch_size, seq_len, *values_flat.shape[1:])  # (batch_size, seq_len, H, W)
			next_values = next_values_flat.reshape(batch_size, seq_len, *next_values_flat.shape[1:])  # (batch_size, seq_len, H, W)
			
			# Get old log probabilities
			actions_flat = actions.reshape(-1, *actions.shape[2:])  # (batch_size * seq_len, H, W)
			old_log_probs_flat, _, _ = self.actor_critic.evaluate_actions(states_flat, actions_flat)
			old_log_probs = old_log_probs_flat.reshape(batch_size, seq_len, *old_log_probs_flat.shape[1:])
		
		# Use the last next_value for GAE computation
		final_next_values = next_values[:, -1]  # (batch_size, H, W)
		
		# Compute GAE
		advantages, returns = self.compute_gae(rewards, values, final_next_values, dones)
		
		# Flatten for training
		states_train = states.reshape(-1, *states.shape[2:])  # (batch_size * seq_len, H, W, C)
		actions_train = actions.reshape(-1, *actions.shape[2:])  # (batch_size * seq_len, H, W)
		old_log_probs_train = old_log_probs.reshape(-1, *old_log_probs.shape[2:])  # (batch_size * seq_len, H, W)
		returns_train = returns.reshape(-1, *returns.shape[2:])  # (batch_size * seq_len, H, W)
		advantages_train = advantages.reshape(-1, *advantages.shape[2:])  # (batch_size * seq_len, H, W)
		
		# Update actor-critic
		actor_critic_stats = self.update_actor_critic(states_train, actions_train, old_log_probs_train, 
													  returns_train, advantages_train)
		
		return actor_critic_stats

	def save(self, save_dir, step):
		torch.save(self.actor_critic.state_dict(), f"{save_dir}/actor_critic_{step}.pth")
		return

	def load_by_step(self, save_dir, step):
		self.actor_critic.load_state_dict(torch.load(f"{save_dir}/actor_critic_{step}.pth"))
		return

	def load(self, file_path):
		self.actor_critic.load_state_dict(torch.load(file_path))
		return
