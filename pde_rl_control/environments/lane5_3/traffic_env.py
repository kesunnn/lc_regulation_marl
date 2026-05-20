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
class Traffic_Env_Four_Action_With_Baseline(Traffic_Env):
	"""
	Environment for lane change control with 4 actions with IDM fundamental diagram
	"""

	road_length = 1000.0
	num_lanes = 5
	vehicle_length_fd = 5.0
	T_fd = 1.4
	min_gap_fd = 2.5
	v0_fd = 24.6
	detector_interval = 60.0
	 
	def __init__(self, grid_length=50.0, control_rate=1.0, density_level=0.1, event_generator=None, vehicle_generator=None, config={}):
		super(Traffic_Env_Four_Action_With_Baseline, self).__init__(grid_length=grid_length, control_rate=control_rate,
												density_level=density_level, event_generator=event_generator,
												vehicle_generator=vehicle_generator, config=config)
		self.num_actions = 4 # 0: no lane change, 1: lane change all allowed, 2: allow speedgain & disallow keepright 3: allow keepright & disallow speedgain
		self.sumo_config_template = os.path.join(os.path.dirname(__file__), "sumo_cfgs")
		self.sumo_config_dir = os.path.join(os.path.dirname(__file__), "sumo_cfgs_"+config["meta"]["log_dir"])
		self.is_eval_baseline = False
		return

	def start(self):
		'''
		road structure is pre-defined.
		Flow is generated based on the density level and fundamental diagram "triangular_fd_params".
		'''
		self._build_sim_dir(self.sumo_config_template, self.sumo_config_dir)
		sumoBinary = sumolib.checkBinary('sumo-gui') if self.gui_flag else sumolib.checkBinary('sumo')
		self._add_vehicles()
		config_path = os.path.join(self.sumo_config_dir, "traffic.sumocfg")
		config_eval_path = os.path.join(self.sumo_config_dir, "traffic_eval.sumocfg")
		config_eval_baseline_path = os.path.join(self.sumo_config_dir, "traffic_eval_baseline.sumocfg")
		self.sumo_cmd = [sumoBinary, "-c", config_path]
		self.sumo_eval_cmd = [sumoBinary, "-c", config_eval_path]
		self.sumo_eval_baseline_cmd = [sumoBinary, "-c", config_eval_baseline_path]
		if not self.is_eval:
			traci.start(self.sumo_cmd, label=self.sim_name, port=self.traci_port)
		elif not self.is_eval_baseline:
			traci.start(self.sumo_eval_cmd, label=self.sim_name, port=self.traci_port)
		else:
			traci.start(self.sumo_eval_baseline_cmd, label=self.sim_name, port=self.traci_port)
		self.traci_conn = traci.getConnection(self.sim_name)
		self._step_count = 0
		self.traci_delta_t = self.traci_conn.simulation.getDeltaT()
		assert (self.delta_T / self.traci_delta_t).is_integer(), "delta_T must be a multiple of simulation time step"
		assert (self.reward_step / self.traci_delta_t).is_integer(), "reward_step must be a multiple of simulation time step"
		return

	def reset(self):
		# 	# get simulation metrics
		# 	_exclude_warm_start = self.config["eval"]["exclude_warm_start"]
		try:
			# If using SUMO, this would reload the simulation without closing the connection
			if not self.traci_conn:
				self.start()
			else:
				self._add_vehicles()
				if self.reset_event_generator:
					if self.event_generator_mode == "random":
						self.event_generator_index = np.random.randint(0, len(self.event_generator_parameters))
					elif self.event_generator_mode == "loop":
						self.event_generator_index = (self.event_generator_index + 1) % len(self.event_generator_parameters)
				time.sleep(0.5)
				if not self.is_eval:
					self.traci_conn.load(self.sumo_cmd[1:])
				elif not self.is_eval_baseline:
					self.traci_conn.load(self.sumo_eval_cmd[1:])
				else:
					self.traci_conn.load(self.sumo_eval_baseline_cmd[1:])
		except (FatalTraCIError, TraCIException) as e:
			print(f"Error in resetting traci simulation: {e}, traceback: {traceback.format_exc()}")
			raise Exception("Failed to reset traci simulation")
		self._step_count = 0
		self.current_arrived_vehicle_statistics = {}
		self.current_simulation_statistics = {}
		self.detector_data = {}
		initial_state = self._get_state()
		self.warm_start_begin = False
		self.warm_start_finish = False
		self._set_lane_change_permissions()
		self._execute_warm_start()
		return initial_state

	def _add_vehicles(self):
		use_cached = False
		cached_vehicles = None
		if not self.reset_vehicles:
			use_cached = True
			cached_vehicles = self.vehicle_generator_cached_vehicles
		if not self.is_eval:
			generated_vehicles = self.vehicle_generator(self.fd_params, num_lanes=self.num_lanes, max_time=self.max_time, desire_Q=self.desired_flow, dev=0.05,
						 control_rate=self.control_rate, traffic_cond=self.desired_traffic_condition,
						 input_file=os.path.join(self.sumo_config_dir, "v1_routes.rou.xml"),
						 output_file=os.path.join(self.sumo_config_dir, "v1_vehicles.rou.xml"),
						 use_cached=use_cached, cached_vehicles=cached_vehicles)
		elif not self.is_eval_baseline:
			generated_vehicles = self.vehicle_generator(self.fd_params, num_lanes=self.num_lanes, max_time=self.max_eval_time, desire_Q=self.desired_flow, dev=0.05,
						 control_rate=self.control_rate, traffic_cond=self.desired_traffic_condition,
						 input_file=os.path.join(self.sumo_config_dir, "v1_routes_eval.rou.xml"),
						 output_file=os.path.join(self.sumo_config_dir, "v1_vehicles_eval.rou.xml"),
						 use_cached=use_cached, cached_vehicles=cached_vehicles)
		else:
			generated_vehicles = self.vehicle_generator(self.fd_params, num_lanes=self.num_lanes, max_time=self.max_eval_time, desire_Q=self.desired_flow, dev=0.05,
						 control_rate=self.control_rate, traffic_cond=self.desired_traffic_condition,
						 input_file=os.path.join(self.sumo_config_dir, "v1_routes_eval_baseline.rou.xml"),
						 output_file=os.path.join(self.sumo_config_dir, "v1_vehicles_eval_baseline.rou.xml"),
						 use_cached=use_cached, cached_vehicles=cached_vehicles)
		if self.reset_vehicles:
			self.vehicle_generator_cached_vehicles = generated_vehicles
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

	def set_is_eval_baseline_flag(self, is_eval_baseline):
		self.is_eval_baseline = is_eval_baseline
		return
