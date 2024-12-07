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
class DQN(nn.Module):
	kernel_dims = [32, 64, 64]
	def __init__(self, state_shape, n_actions, network_config):
		super(DQN, self).__init__()
		self.receptive_field = 1
		_, _, input_channels = state_shape
		self.kernel_dims = network_config['kernel_dims'] if 'kernel_dims' in network_config else self.kernel_dims
		layers = []
		layers.append(nn.Conv2d(input_channels, self.kernel_dims[0], kernel_size=3, stride=1, padding=1))
		layers.append(nn.ReLU())
		self.receptive_field += 1
		for i in range(len(self.kernel_dims)-1):
			layers.append(nn.Conv2d(self.kernel_dims[i], self.kernel_dims[i+1], kernel_size=3, stride=1, padding=1))
			self.receptive_field += 1
			layers.append(nn.ReLU())
		layers.append(nn.Conv2d(self.kernel_dims[-1], n_actions, kernel_size=1, stride=1))
		self.layers = nn.Sequential(*layers)
		return

	def forward(self, x):
		x = self.layers(x)
		return x


# %%
class DQNAgent(nn.Module):
	learning_rate: float = 1e-4
	adam_eps: float = 1e-4
	discount: float = 0.99
	target_update_period: int = 2000
	clip_grad_norm: float = 10.0
	use_double_q: bool = True
	lr_scheduler_mode: str = "constant"

	def __init__(self, env, network_config, training_config):
		super(DQNAgent, self).__init__()

		self.env = env
		self.network_config = network_config
		self.training_config = training_config

		self.num_actions = env.num_actions
		self.state_shape = env.state_shape # (H, W, C)
		self.critic = ptu.nn_to_device(DQN(self.state_shape, self.num_actions, network_config))
		self.target_critic = ptu.nn_to_device(DQN(self.state_shape, self.num_actions, network_config))

		self.discount = training_config["discount"] if "discount" in training_config else self.discount
		self.target_update_period = training_config["target_update_period"] if "target_update_period" in training_config else self.target_update_period
		self.clip_grad_norm = training_config["clip_grad_norm"] if "clip_grad_norm" in training_config else self.clip_grad_norm
		self.use_double_q = training_config["use_double_q"] if "use_double_q" in training_config else self.use_double_q
		self.lr_scheduler_mode = training_config["lr_scheduler_mode"] if "lr_scheduler_mode" in training_config else self.lr_scheduler_mode
		self.learning_rate = training_config["learning_rate"] if "learning_rate" in training_config else self.learning_rate
		self.adam_eps = training_config["adam_eps"] if "adam_eps" in training_config else self.adam_eps
		self.critic_loss = nn.MSELoss()

		self.critic_optimizer = self._make_optimizer(self.critic.parameters())
		self.lr_scheduler = self._make_lr_schedule(self.critic_optimizer)

		self.update_target_critic()

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

	def get_action(self, state: np.ndarray, epsilon: float = 0.02) -> int:
		"""
		Used for evaluation.
		"""
		assert state.shape == self.state_shape, f"Invalid state shape: {state.shape} != {self.state_shape}"
		if np.random.rand() < epsilon:
			action = np.random.randint(0, self.num_actions, size=(self.state_shape[0], self.state_shape[1]))
		else:
			state_tensor = ptu.from_numpy(np.transpose(state, (2, 0, 1))).unsqueeze(0) # (H, W, C) -> (1, C, H, W)
			with torch.no_grad():
				q_values = self.critic(state_tensor) # (1, num_actions, H, W)
			action = q_values.argmax(dim=1).detach().cpu().numpy().squeeze(0) # (1, H, W) -> (H, W)
		return action

	def update_critic(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor) -> dict:
		"""Update the DQN critic, and return stats for logging."""
		batch_size = reward.shape[0]
		state, next_state = torch.permute(state, (0, 3, 1, 2)), torch.permute(next_state, (0, 3, 1, 2)) # (batch_size, H, W, C) -> (batch_size, C, H, W)
		# Compute target values
		with torch.no_grad():
			next_qa_values = self.target_critic(next_state) # (batch_size, num_actions, H, W)

			if self.use_double_q:
				next_action = self.critic(next_state).argmax(dim=1) # (batch_size, H, W)
			else:
				next_action = self.target_critic(next_state).argmax(dim=1) # (batch_size, H, W)
			
			next_q_values = next_qa_values.gather(1, next_action.unsqueeze(1)).squeeze(1) # (batch_size, H, W)
			# done = done.unsqueeze(1).unsqueeze(1).expand(-1, self.state_shape[0], self.state_shape[1]) # (batch_size,) -> (batch_size, H, W)
			# target_values = reward + self.discount * next_q_values * (1 - done) # (batch_size, H, W)
			target_values = reward + self.discount * next_q_values # (batch_size, H, W)

		# train the critic with the target values
		qa_values = self.critic(state) # (batch_size, num_actions, H, W)
		q_values = qa_values.gather(1, action.unsqueeze(1)).squeeze(1) # (batch_size, H, W)
		loss = self.critic_loss(q_values, target_values)


		self.critic_optimizer.zero_grad()
		loss.backward()
		grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
			self.critic.parameters(), self.clip_grad_norm or float("inf")
		)
		self.critic_optimizer.step()

		self.lr_scheduler.step()

		return {
			"critic_loss": loss.item(),
			"q_values": q_values.mean().item(),
			"target_values": target_values.mean().item(),
			"grad_norm": grad_norm.item(),
		}

	def update_target_critic(self):
		self.target_critic.load_state_dict(self.critic.state_dict())

	def update(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor, step: int) -> dict:
		"""
		Update the DQN agent, including both the critic and target.
		"""
		#update the critic, and the target if needed
		critic_stats = self.update_critic(state, action, reward, next_state, done)
		if step % self.target_update_period == 0:
			self.update_target_critic()

		return critic_stats

	def save(self, save_dir, step):
		torch.save(self.critic.state_dict(), f"{save_dir}/critic_{step}.pth")
		torch.save(self.target_critic.state_dict(), f"{save_dir}/target_critic_{step}.pth")
		return

	def load_by_step(self, save_dir, step):
		self.critic.load_state_dict(torch.load(f"{save_dir}/critic_{step}.pth"))
		self.target_critic.load_state_dict(torch.load(f"{save_dir}/target_critic_{step}.pth"))
		return

	def load(self, file_path, target_file_path=None):
		self.critic.load_state_dict(torch.load(file_path))
		if target_file_path is not None:
			self.target_critic.load_state_dict(torch.load(target_file_path))
		else:
			self.target_critic.load_state_dict(torch.load(file_path))
		return


