# %%
import gym
import os, sys, math, time, copy
import numpy as np
import sumolib
import traci
import traci.constants as tc # https://sumo.dlr.de/pydoc/traci.constants.html
from traci.exceptions import FatalTraCIError, TraCIException
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pde_rl_control.utils.traci_u import extract_vehicle_root_type, extract_vehicle_root_type_from_id
import numpy as np
import traceback

from pde_rl_control.environments.lane5_1.traffic_env import Traffic_Env

# %%
class Traffic_Env_Four_Action(Traffic_Env):
	"""
	Environment for lane change control with 4 actions
	"""

	road_length = 1000.0
	num_lanes = 5
	vehicle_length_fd = 5.0
	T_fd = 1.4
	min_gap_fd = 2.5
	v0_fd = 24.6
	detector_interval = 60.0
	 
	def __init__(self, grid_length=50.0, control_rate=1.0, density_level=0.1, event_generator=None, vehicle_generator=None, config={}):
		super(Traffic_Env_Four_Action, self).__init__(grid_length=grid_length, control_rate=control_rate,
												density_level=density_level, event_generator=event_generator,
												vehicle_generator=vehicle_generator, config=config)
		self.num_actions = 4 # 0: no lane change, 1: lane change all allowed, 2: allow speedgain & disallow keepright 3: allow keepright & disallow speedgain
		self.sumo_config_template = os.path.join(os.path.dirname(__file__), "sumo_cfgs")
		self.sumo_config_dir = os.path.join(os.path.dirname(__file__), "sumo_cfgs_"+config["meta"]["log_dir"])
		return

	def _execute_action(self, action, info):
		# 1 means allow lane changes, 0 means disallow lane changes
		exemptive_vehicles = info.get("exemptive_vehicles", [])
		for lane in range(self.num_lanes):
			lane_id = f"e1_{lane}"
			veh_list = self.traci_conn.lane.getLastStepVehicleIDs(lane_id)
			for veh_id in veh_list:
				vtype = self.traci_conn.vehicle.getTypeID(veh_id)
				root_vtype = extract_vehicle_root_type(vtype)
				if root_vtype != "controlled" or veh_id in exemptive_vehicles:
					continue
				veh_pos = self.traci_conn.vehicle.getPosition(veh_id)
				veh_agent = int(veh_pos[0] // self.grid_length)
				if action[lane, veh_agent] == 0.:
					if vtype != "controlled":
						self.traci_conn.vehicle.setType(veh_id, "controlled")
					self.traci_conn.vehicle.setLaneChangeMode(veh_id, 256) # default mode
				elif action[lane, veh_agent] == 1.:
					if vtype != "controlled":
						self.traci_conn.vehicle.setType(veh_id, "controlled")
					self.traci_conn.vehicle.setLaneChangeMode(veh_id, 1621) # allow lane change
				elif action[lane, veh_agent] == 2.:
					if vtype != "controlled:left":
						self.traci_conn.vehicle.setType(veh_id, "controlled:left")
					self.traci_conn.vehicle.setLaneChangeMode(veh_id, 1621) # only allow lane change to the left
				elif action[lane, veh_agent] == 3.:
					if vtype != "controlled:right":
						self.traci_conn.vehicle.setType(veh_id, "controlled:right")
					self.traci_conn.vehicle.setLaneChangeMode(veh_id, 1621) # only allow lane change to the right
				else:
					raise ValueError("Invalid action value for lane: {}, agent: {}, value={}".format(lane, veh_agent, action[lane, veh_agent]))
		return

	def get_action_metrics(self, action):
		assert action.shape == (self.num_lanes, self.n_agents_per_lane), "action shape must be (num_lanes, n_agents_per_lane)"
		result = {}
		for lane in range(self.num_lanes):
			lane_dict = {}
			result[f"lane_{lane}"] = lane_dict
			# count number of agents that are allowed to change lanes
			allow_rate = np.sum(action[lane, :] >= 1.) / self.n_agents_per_lane
			allow_rate_left = np.sum(action[lane, :] == 2.) / self.n_agents_per_lane
			allow_rate_right = np.sum(action[lane, :] == 3.) / self.n_agents_per_lane
			allow_rate_both = np.sum(action[lane, :] == 1.) / self.n_agents_per_lane
			lane_dict["any"] = allow_rate
			lane_dict["left"] = allow_rate_left
			lane_dict["right"] = allow_rate_right
			lane_dict["both"] = allow_rate_both
		return result
