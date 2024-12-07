# %%
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np

# %%
class Logger:
	def __init__(self, log_dir):
		self._log_dir = log_dir
		print('########################')
		print('logging outputs to ', log_dir)
		print('########################')
		self._summ_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)

	def log_scalar(self, scalar, name, step):
		self._summ_writer.add_scalar('{}'.format(name), scalar, step)

	def log_scalars(self, scalar_dict, group_name, step):
		"""Will log all scalars in the same plot."""
		self._summ_writer.add_scalars('{}'.format(group_name), scalar_dict, step)

	def log_text(self, text, name, step):
		self._summ_writer.add_text('{}'.format(name), text, step)

	def log_image(self, image, name, step):
		assert(len(image.shape) == 3)  # [C, H, W]
		self._summ_writer.add_image('{}'.format(name), image, step)

	def log_video(self, video_frames, name, step, fps=10):
		assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
		self._summ_writer.add_video('{}'.format(name), video_frames, step, fps=fps)

	def log_figures(self, figure, name, step):
		"""figure: matplotlib.pyplot figure handle"""
		assert figure.shape[0] > 0, "Figure logging requires input shape [batch x figures]!"
		self._summ_writer.add_figure('{}'.format(name), figure, step)

	def log_figure(self, figure, name, step):
		"""figure: matplotlib.pyplot figure handle"""
		self._summ_writer.add_figure('{}'.format(name), figure, step)

	def dump_scalars(self, log_path=None):
		log_path = os.path.join(self._log_dir, "scalar_data.json") if log_path is None else log_path
		self._summ_writer.export_scalars_to_json(log_path)

	def log_model(self, model, name, step):
		# Log the average weight of Conv layers
		for l_name, layer in model.named_modules():
			if isinstance(layer, nn.Conv2d):
				self._summ_writer.add_histogram(f'{name}/Weights/{l_name}', layer.weight, step)
				self._summ_writer.add_scalar(f'{name}/Avg Weight/{l_name}', torch.mean(layer.weight).item(), step)

		# Log gradient distributions
		for pname, param in model.named_parameters():
			self._summ_writer.add_histogram(f'{name}/Gradients/{pname}', param.grad, step)

	def log_histogram(self, data, name, step):
		self._summ_writer.add_histogram(name, data, step)

	def flush(self):
		self._summ_writer.flush()
		
	def close(self):
		self._summ_writer.close()
