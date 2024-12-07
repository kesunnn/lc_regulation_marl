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
import pde_rl_control.utils.vehicle_generator as vehicle_gen
from pde_rl_control.utils.u import generate_dict_from_list
from pde_rl_control.utils.traci_u import extract_vehicle_root_type, extract_vehicle_root_type_from_id
import numpy as np
import traceback

# %%
class Traffic_Env(gym.Env):

	road_length = 1000.0
	num_lanes = 5
	vehicle_length_fd = 5.0
	T_fd = 1.4
	min_gap_fd = 2.5
	v0_fd = 24.6
	a_fd = 0.73
	b_fd = 1.67
	detector_interval = 60.0
	ttc_thresholds = [1.3, 1.5, 2.0, 2.5, 3.0, 5.0]
	 
	def __init__(self, grid_length=50.0, control_rate=1.0, density_level=0.1, event_generator=None, vehicle_generator=None, config={}):
		assert self.road_length % grid_length == 0, "grid_length must be a divisor of road_length"
		assert 0.0 < control_rate <= 1, "control_rate must be in the range (0, 1]"
		assert 0.0 < density_level <= 1, "density_level must be in the range (0, 1] of maximum density"
		assert callable(event_generator), "event_generator must be a callable"
		assert callable(vehicle_generator), "vehicle_generator must be a callable"

		self.grid_length = grid_length
		self.control_rate = control_rate
		self.density_level = density_level
		# set up event generator
		self.event_generator = event_generator
		self.event_generator_parameters = config["simulation"]["event_generator_parameters"]
		self.event_generator_mode = config["simulation"]["event_generator_mode"]
		assert self.event_generator_mode in ["random", "loop"], "event_generator_mode must be either 'random' or 'loop'"
		self.event_generator_index = 0 if self.event_generator_mode == "loop" else np.random.randint(0, len(self.event_generator_parameters))

		self.vehicle_generator = vehicle_generator
		self.fundamental_diagram_name = config["simulation"]["fundamental_diagram_name"]
		assert self.fundamental_diagram_name in ["triangular", "IDM"], "fundamental_diagram_name must be in ['triangular', 'IDM']"
		self.set_funamental_diagram_by_name(self.fundamental_diagram_name)
		self.vehicle_generator_cached_vehicles = None

		self.n_agents_per_lane = int(self.road_length / self.grid_length)
		self.n_agents = int(self.n_agents_per_lane * self.num_lanes)
		self.state_shape = (self.num_lanes, self.n_agents_per_lane, 4)
		self.num_actions = 2 # 0: no lane change, 1: lane change
		self.traci_conn = None
		self.travel_times = {}
		self.lane_changes = {}
		
		self._step_count = 0
		if "max_time" in config["simulation"]: # in seconds
			self.max_time = int(config["simulation"]["max_time"])
		else:
			self.max_time = 3600
		if "max_time" in config["eval"]:
			self.max_eval_time = int(config["eval"]["max_time"])
		else:
			self.max_eval_time = self.max_time
		
		if "delta_T" in config["simulation"]:
			assert config["simulation"]["delta_T"] > 0, "delta_T must be positive"
			self.delta_T = config["simulation"]["delta_T"]
		else:
			self.delta_T = 1.0
		
		if "is_gui" in config["simulation"]:
			self.gui_flag = config["simulation"]["is_gui"]
		else:
			self.gui_flag = False

		if "state_normalize" in config["simulation"]:
			self.state_normalize = config["simulation"]["state_normalize"]
		else:
			self.state_normalize = False

		if "step" in config["reward"]:
			self.reward_step = config["reward"]["step"] # reward calculation time step
		else:
			self.reward_step = 1.0
		assert (self.delta_T / self.reward_step).is_integer(), "delta_T must be a multiple of reward_step"
		self.is_eval = False # simulation cfg and vehicle can be different in eval mode
		self.reset_vehicles= True
		self.reset_event_generator = True
		self.config = config

		#update desired flow info
		rho_m, rho_s = self.fd_params["rho_m"], self.fd_params["rho_s"]
		self.desired_rho = rho_m * self.density_level
		self.desired_flow = self.fd_rho_to_flow(self.desired_rho, self.fd_params) * 3600 # vehicles per hour
		self.desired_traffic_condition = "free" if self.desired_rho <= rho_s else "congested"
		self.desired_velocity = self.desired_flow / (3600 * self.desired_rho) # m/s
		print("Env initializing. desired_rho: {}, desired_flow: {}, desired_velocity: {}, desired_traffic_condition: {}".\
				format(self.desired_rho, self.desired_flow, self.desired_velocity, self.desired_traffic_condition))

		# update warm start info
		self.warm_start_begin = False
		self.warm_start_finish = False
		if "warm_start_time" in config["simulation"]:
			self.warm_start_time = int(config["simulation"]["warm_start_time"])
		else:
			self.warm_start_time = 0
		self.warm_start_time = max(self.warm_start_time, int(self.road_length / self.desired_velocity))

		self.sumo_config_template = os.path.join(os.path.dirname(__file__), "sumo_cfgs")
		self.sumo_config_dir = os.path.join(os.path.dirname(__file__), "sumo_cfgs_"+config["meta"]["log_dir"])

		self.traci_port = int(config["simulation"]["traci_port"])
		self.last_simulation_metrics = None
		self.current_arrived_vehicle_statistics = {}
		self.current_simulation_statistics = {}

		self.sim_name = config["meta"]["log_dir"]

		self.subscription_fields = [tc.VAR_LANEPOSITION, tc.VAR_LANE_INDEX, tc.VAR_TYPE, \
							tc.VAR_DEPART_DELAY, tc.VAR_DEPARTURE, tc.VAR_CO2EMISSION, \
							tc.VAR_WAITING_TIME, tc.VAR_SPEED]
		self.subscription_fields_map = {
			tc.VAR_LANEPOSITION: "lane_position",
			tc.VAR_LANE_INDEX: "lane_index",
			tc.VAR_TYPE: "vType",
			tc.VAR_DEPART_DELAY: "depart_delay",
			tc.VAR_DEPARTURE: "depart",
			tc.VAR_CO2EMISSION: "co2_emission",
			tc.VAR_WAITING_TIME: "waiting_time",
			tc.VAR_LEADER: "leader_info",
			tc.VAR_SPEED: "speed"
		}

		self.detector_data = {}

		return

	# <------------------------------------ basic funcs ------------------------------------>
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
		self.sumo_cmd = [sumoBinary, "-c", config_path]
		self.sumo_eval_cmd = [sumoBinary, "-c", config_eval_path]
		if not self.is_eval:
			traci.start(self.sumo_cmd, label=self.sim_name, port=self.traci_port)
		else:
			traci.start(self.sumo_eval_cmd, label=self.sim_name, port=self.traci_port)
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
				else:
					self.traci_conn.load(self.sumo_eval_cmd[1:])
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

	def step(self, action):
		# take action then simulate delta_T seconds
		info = {}
		action = self._process_action(action, info)
		# Check if done
		done = False
		reward_step_count = int(self.reward_step / self.traci_delta_t)
		step_rewards = []
		info["start_step"] = self._step_count
		info["start_time"] = self.traci_conn.simulation.getTime()
		for _idx in range(int(self.delta_T / self.traci_delta_t)):
			# if not skip_action:
			exemptive_vehicles = self.event_generator(self.traci_conn, **self.event_generator_parameters[self.event_generator_index])
			if exemptive_vehicles:
				info["exemptive_vehicles"] = exemptive_vehicles
			self._execute_action(action, info) # zero-order hold action
			if (_idx + 1) % reward_step_count == 0:
				_temp_state = self._get_state()
				step_rewards.append(self._get_step_reward(_temp_state, info))
			self.traci_conn.simulationStep()
			self._step_count += 1
			for veh_id in self.traci_conn.simulation.getDepartedIDList():
				self.traci_conn.vehicle.subscribe(veh_id, self.subscription_fields)
				self.traci_conn.vehicle.subscribeLeader(veh_id, dist=500)
			_subscription_results = self.traci_conn.vehicle.getAllSubscriptionResults()
			_subscription_results = self.__process_subscription_results(_subscription_results)
			self.__update_simulation_statistics(_subscription_results)
			arrived_vehs = self.traci_conn.simulation.getArrivedIDList()
			if len(arrived_vehs) > 0:
				self.__update_arrived_vehicles(arrived_vehs)
			curr_time = self.traci_conn.simulation.getTime()
			if curr_time > self.detector_interval and (curr_time - self.delta_T) % self.detector_interval == 0:
				self.__retrieve_detector_data()
			if curr_time > self.warm_start_time:
				self._execute_warm_start()
			max_episode_time = self.max_time if not self.is_eval else self.max_eval_time
			if curr_time >= max_episode_time or self.traci_conn.simulation.getMinExpectedNumber() <= 0 or self._check_collision(info):
				done = True
				metrics_result = self.__update_simulation_metrics() # update simulation metrics
				self.last_simulation_metrics = copy.deepcopy(metrics_result)
				info["simulation_metrics"] = metrics_result
				info["detector_metrics"] = self.detector_summary()
				break
		info["end_step"] = self._step_count
		info["end_time"] = self.traci_conn.simulation.getTime()
		state = self._get_state()
		if not step_rewards:
			print("done: {}, is_collision: {}, curr_time: {:.2f}".format(done, info["is_collision"], info["end_time"]))
		reward, global_reward = self._get_reward(step_rewards, info)
		info["reward"] = reward
		info["global_reward"] = global_reward
		info["detector_data"] = copy.deepcopy(self.detector_data)
		return state, reward, done, info

	def close(self):
		self.traci_conn = traci.getConnection(self.sim_name)
		if self.traci_conn:
			self.traci_conn.close()
		return

	# <------------------------------------ configuration funcs ---------------------------->
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
		else:
			generated_vehicles = self.vehicle_generator(self.fd_params, num_lanes=self.num_lanes, max_time=self.max_eval_time, desire_Q=self.desired_flow, dev=0.05,
						 control_rate=self.control_rate, traffic_cond=self.desired_traffic_condition,
						 input_file=os.path.join(self.sumo_config_dir, "v1_routes_eval.rou.xml"),
						 output_file=os.path.join(self.sumo_config_dir, "v1_vehicles_eval.rou.xml"),
						 use_cached=use_cached, cached_vehicles=cached_vehicles)
		if self.reset_vehicles:
			self.vehicle_generator_cached_vehicles = generated_vehicles
		return

	def set_eval_flag(self, is_eval, reset_vehicles=True, reset_event_generator=True):
		'''
		Set the evaluation flag
		Args:
			is_eval (bool): whether the environment is in evaluation mode
			reset_vehicles (bool): whether to reset the vehicles
			reset_event_generator (bool): whether to reset the event generator
		'''
		assert isinstance(is_eval, bool), "is_eval must be a boolean"
		self.reset_vehicles = reset_vehicles
		self.reset_event_generator = reset_event_generator
		self.is_eval = is_eval
		return

	def set_funamental_diagram_by_name(self, fd_name):
		vehicle_attribs = {"v0": self.v0_fd, "time_gap": self.T_fd, "min_gap": self.min_gap_fd, "v_len": self.vehicle_length_fd,\
					"a": self.a_fd, "b": self.b_fd}
		if fd_name == "triangular":
			fd_params_generator = getattr(vehicle_gen, "gen_triangular_fd_params")
			self.fd_params = fd_params_generator(vehicle_attribs)
			self.fd_rho_to_flow = getattr(vehicle_gen, "fundamental_diagram_triangular")
		elif fd_name == "IDM":
			fd_params_generator = getattr(vehicle_gen, "gen_IDM_fd_params")
			self.fd_params = fd_params_generator(vehicle_attribs)
			self.fd_rho_to_flow = getattr(vehicle_gen, "fundamental_diagram_IDM")
		else:
			raise ValueError("Invalid fundamental diagram name: {}".format(fd_name))
		return

	def _build_sim_dir(self, template_dir, dest_dir):
		if not os.path.exists(dest_dir):
			os.makedirs(dest_dir)
			dest_outputs_dir = os.path.join(dest_dir, "outputs")
			if not os.path.exists(dest_outputs_dir):
				os.makedirs(dest_outputs_dir)
		for file in os.listdir(template_dir):
			if os.path.isfile(os.path.join(template_dir, file)):
				os.system(f"cp {os.path.join(template_dir, file)} {os.path.join(dest_dir, file)}")
		return

	def _set_lane_change_permissions(self): # to do: set lane change direction permissions to achieve lane change control
		left_allow_classes = ["passenger", "vip"]
		right_allow_classes = ["passenger", "hov"]
		for lane in range(self.num_lanes):
			lane_id = f"e1_{lane}"
			if lane != self.num_lanes - 1:
				self.traci_conn.lane.setChangePermissions(lane_id, left_allow_classes, 1)
			if lane != 0:
				self.traci_conn.lane.setChangePermissions(lane_id, right_allow_classes, -1)
		return

	def _execute_warm_start(self):
		if not hasattr(self, "lane_speed_limit"):
			self.lane_speed_limit = {}
		if not self.warm_start_begin:
			for lane in range(self.num_lanes):
				lane_id = f"e1_{lane}"
				pre_max_speed = self.traci_conn.lane.getMaxSpeed(lane_id)
				self.lane_speed_limit[lane_id] = pre_max_speed
				self.traci_conn.lane.setMaxSpeed(lane_id, self.desired_velocity)
			self.warm_start_begin = True
		elif not self.warm_start_finish:
			curr_time = self.traci_conn.simulation.getTime()
			if curr_time >= self.warm_start_time:
				for lane in range(self.num_lanes):
					lane_id = f"e1_{lane}"
					self.traci_conn.lane.setMaxSpeed(lane_id, self.lane_speed_limit[lane_id])
				self.warm_start_finish = True
		return

	def __unsubscribe_all(self):
		for vid in self.traci_conn.vehicle.getAllSubscriptionResults():
			self.traci_conn.vehicle.unsubscribe(vid)
		return

	# <------------------------------------ state funcs ----------------------------------->
	def _get_state(self):
		'''
		return global state with shape (num_lanes, n_agents_per_lane, state_dim)
		'''
		_state_dim = 4
		global_state = np.zeros((self.num_lanes, self.n_agents_per_lane, _state_dim))
		_rho_max, _v_max = self.fd_params["rho_m"], self.fd_params["v0"]
		for lane in range(self.num_lanes):
			lane_id = f"e1_{lane}"
			veh_list = self.traci_conn.lane.getLastStepVehicleIDs(lane_id)
			agent_veh_list = {agent_idx : list() for agent_idx in range(self.n_agents_per_lane)}
			for i, veh_id in enumerate(veh_list):
				veh_pos = self.traci_conn.vehicle.getPosition(veh_id)
				veh_agent = int(veh_pos[0] // self.grid_length)
				agent_veh_list[veh_agent].append(veh_id)
			for i in range(self.n_agents_per_lane):
				avg_density, avg_speed, contr_avg_density, contr_avg_speed = self._aggregate_vehicle_states(agent_veh_list[i])
				if self.state_normalize:
					avg_density /= _rho_max
					avg_speed /= _v_max
					contr_avg_density /= _rho_max
					contr_avg_speed /= _v_max
				global_state[lane][i] = np.array([avg_density, avg_speed, contr_avg_density, contr_avg_speed])
		return global_state

	def _check_collision(self, info):
		collision_vids = self.traci_conn.simulation.getCollidingVehiclesIDList()
		if len(collision_vids) > 0:
			info["collision_vehicles"] = collision_vids
			info["is_collision"] = True
			return True
		info["is_collision"] = False
		return False

	def _aggregate_vehicle_states(self, vehicle_list):
		avg_density, avg_speed, contr_avg_density, contr_avg_speed = 0., 0., 0., 0.
		for veh_id in vehicle_list:
			veh_speed = self.traci_conn.vehicle.getSpeed(veh_id)
			vtype = self.traci_conn.vehicle.getTypeID(veh_id)
			root_vtype = extract_vehicle_root_type(vtype)
			avg_density += 1
			avg_speed += veh_speed
			if root_vtype == "controlled":
				contr_avg_density += 1
				contr_avg_speed += veh_speed
		if avg_density > 0:
			avg_speed /= avg_density
			avg_density /= self.grid_length
		if contr_avg_density > 0:
			contr_avg_speed /= contr_avg_density
			contr_avg_density /= self.grid_length
		return avg_density, avg_speed, contr_avg_density, contr_avg_speed

	# <------------------------------------ reward funcs ----------------------------------->
	def _get_step_reward(self, state, info):
		reward_config = self.config["reward"]
		_is_LoS_punish_free = bool(reward_config["is_LoS_punish_free"])
		_rho_max, _v_max = self.fd_params["rho_m"], self.fd_params["v0"]
		_rho_s = self.fd_params["rho_s"]
		rho, v, control_rho = state[:, :, 0], state[:, :, 1], state[:, :, 2]
		# spi_rewards, los_rewards, control_rates = (100 / _v_max) * v, (100 / _rho_max) * rho, np.divide(control_rho, rho+1e-6)
		if self.state_normalize:
			spi_rewards, los_rewards, control_rates = v, 1.0 - rho, np.divide(control_rho, rho+1e-6)
		else:
			spi_rewards, los_rewards, control_rates = v / _v_max, 1.0 - rho / _rho_max, np.divide(control_rho, rho+1e-6)
		if not _is_LoS_punish_free:
			los_threshold = 1.0 - _rho_s / _rho_max
			los_rewards = np.where(los_rewards >= los_threshold, 1.0, los_rewards)
		return (spi_rewards, los_rewards, control_rates)

	def _get_reward(self, step_rewards, info):
		reward_config = self.config["reward"]
		_SPI_w, _LoS_w = reward_config["SPI_w"], reward_config["LoS_w"]
		is_RSCI = reward_config["is_RSCI"]
		is_control_rate_weighted = reward_config["is_control_rate_weighted"]
		neighbor_radius = int(reward_config["neighbor_radius"])
		spi_rewards, los_rewards, control_rates = list(zip(*step_rewards))
		spi_rewards, los_rewards, control_rates = np.stack(spi_rewards, axis=0), np.stack(los_rewards, axis=0), np.stack(control_rates, axis=0)
		# SPI reward
		spi_reward = np.mean(spi_rewards, axis=0)
		if is_RSCI:
			congestion_threshold = 50
			rsci = np.where(spi_rewards < congestion_threshold, 0, 1)
			rsci = np.mean(rsci, axis=0)
			rsci = np.multiply(rsci, spi_reward)
			spi_reward = rsci
		# LoS reward
		los_reward = np.mean(los_rewards, axis=0)
		# Average neighbor reward
		avg_spi_reward = self.__reward_average_neighbor(spi_reward, neighbor_radius)
		avg_los_reward = self.__reward_average_neighbor(los_reward, neighbor_radius)
		# avg reward
		avg_reward = _SPI_w * avg_spi_reward + _LoS_w * avg_los_reward
		# original reward
		global_reward = _SPI_w * spi_reward + _LoS_w * los_reward
		info["reward_spi"] = avg_spi_reward
		info["reward_los"] = avg_los_reward
		info["global_reward_spi"] = spi_reward
		info["global_reward_los"] = los_reward
		# multiply by control rate
		if is_control_rate_weighted:
			control_rate = np.mean(control_rates, axis=0)
			control_rate = np.where(control_rate > 1-1e-3, 1, control_rate)
			control_rate = np.where(control_rate < 1e-3, 0, control_rate)
			avg_reward = np.multiply(avg_reward, control_rate)
			global_reward = np.multiply(global_reward, control_rate)
		return avg_reward, global_reward

	def __reward_average_neighbor(self, reward, neighbor_radius):
		assert reward.shape == (self.num_lanes, self.n_agents_per_lane), "reward shape must be (num_lanes, n_agents_per_lane)"
		cumu_reward = np.cumsum(np.cumsum(reward, axis=0), axis=1)
		cumu_reward = np.pad(cumu_reward, ((1, 0), (1, 0)), mode='constant', constant_values=0)
		result = np.zeros_like(reward)
		for lane in range(self.num_lanes):
			for agent in range(self.n_agents_per_lane):
				rb = min(self.num_lanes, lane+2), min(self.n_agents_per_lane, agent+neighbor_radius+1)
				lt = max(0, lane-1), max(0, agent-neighbor_radius)
				result[lane, agent] = cumu_reward[rb[0], rb[1]] - cumu_reward[lt[0], rb[1]] - cumu_reward[rb[0], lt[1]] + cumu_reward[lt[0], lt[1]]
				result[lane, agent] /= (rb[0]-lt[0]) * (rb[1]-lt[1])
		return result

	# <------------------------------------ action funcs ----------------------------------->
	def _process_action(self, action, info):
		if isinstance(action, list):
			action = np.array(action)
		assert action.shape == (self.num_lanes, self.n_agents_per_lane), "action shape must be (num_lanes, n_agents_per_lane)"
		if "action_allow_rate" not in info:
			info["action_allow_rate"] = self.get_action_metrics(action)
		return action

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
				if action[lane, veh_agent] == 1.:
					self.traci_conn.vehicle.setLaneChangeMode(veh_id, 1621) # default mode
				else:
					self.traci_conn.vehicle.setLaneChangeMode(veh_id, 256) # https://www.eclipse.org/lists/sumo-user/msg01811.html (mode 256 vs 512)
		return

	# <------------------------------------ simulation metrics funcs ----------------------->
	def __retrieve_detector_data(self):
		curr_time = self.traci_conn.simulation.getTime()
		_start_time = curr_time - self.detector_interval - self.delta_T
		detector_ids = self.traci_conn.inductionloop.getIDList()
		for detector_id in detector_ids:
			position = self.traci_conn.inductionloop.getPosition(detector_id)
			lane_id = self.traci_conn.inductionloop.getLaneID(detector_id)
			num = self.traci_conn.inductionloop.getLastIntervalVehicleNumber(detector_id)
			occupancy = self.traci_conn.inductionloop.getLastIntervalOccupancy(detector_id)
			speed = self.traci_conn.inductionloop.getLastIntervalMeanSpeed(detector_id)
			flow = num / self.detector_interval
			density = occupancy / (self.vehicle_length_fd * 100)
			if detector_id not in self.detector_data:
				self.detector_data[detector_id] = []
			self.detector_data[detector_id].append({"position": int(position), "lane": str(lane_id),\
										"start_time": _start_time, "end_time": _start_time + self.detector_interval,\
										"num": num, "speed": speed, "flow": flow, "density": density})
		return

	def detector_summary(self):
		_exclude_warm_start = self.config["eval"]["exclude_warm_start"]
		data = []
		for detector_id, detector_info in self.detector_data.items():
			if not _exclude_warm_start:
				data.extend(detector_info)
			else:
				data.extend([d for d in detector_info if d["start_time"] >= self.warm_start_time])
		if len(data) == 0:
			print("Error: No detector data found")
			return {}
		# lane-position data
		lane_position_flow = generate_dict_from_list(data, ["lane", "position"], "flow", "avg")
		lane_position_speed = generate_dict_from_list(data, ["lane", "position"], "speed", "avg")
		lane_position_density = generate_dict_from_list(data, ["lane", "position"], "density", "avg")
		# lane-time data
		lane_time_flow = generate_dict_from_list(data, ["lane", "start_time"], "flow", "sum")
		lane_time_speed = generate_dict_from_list(data, ["lane", "start_time"], "speed", "avg")
		lane_time_density = generate_dict_from_list(data, ["lane", "start_time"], "density", "avg")
		result = {
			"lane_position_flow": lane_position_flow,
			"lane_position_speed": lane_position_speed,
			"lane_position_density": lane_position_density,
			"lane_time_flow": lane_time_flow,
			"lane_time_speed": lane_time_speed,
			"lane_time_density": lane_time_density
		}
		return result

	def __process_subscription_results(self, result):
		if not result:
			return dict()
		processed_result = {}
		for vid, veh_info in result.items():
			processed_result[vid] = {self.subscription_fields_map[k]: v for k, v in veh_info.items()}
		return processed_result

	def __update_simulation_statistics(self, subscription_results):
		if not subscription_results:
			return
		for vid, veh_info in subscription_results.items():
			if vid not in self.current_simulation_statistics:
				self.current_simulation_statistics[vid] = {
					"depart": float(veh_info["depart"]),
					"depart_delay": float(veh_info["depart_delay"]),
					"vType": extract_vehicle_root_type(veh_info["vType"]),
					"lane": int(veh_info["lane_index"]),
					"waiting_time": 0.0,
					"co2_emission": 0.0,
					"lanechange_count": 0,
					"lane_position": float(veh_info["lane_position"]),
					"speed": float(veh_info["speed"])
				}
				for threshold in self.ttc_thresholds:
					self.current_simulation_statistics[vid][f"TET_{threshold}"] = 0.0
					self.current_simulation_statistics[vid][f"TIT_{threshold}"] = 0.0
			veh_stats = self.current_simulation_statistics[vid]
			if int(veh_info["lane_index"]) != veh_stats["lane"]:
				veh_stats["lanechange_count"] += 1
				veh_stats["lane"] = int(veh_info["lane_index"])
			veh_stats["co2_emission"] += (float(veh_info["co2_emission"]) * self.traci_delta_t)
			veh_stats["waiting_time"] += float(veh_info["waiting_time"])
			veh_stats["lane_position"] = float(veh_info["lane_position"])
			veh_stats["speed"] = float(veh_info["speed"])
		# get leader info
		for vid, veh_info in subscription_results.items():
			if not veh_info["leader_info"]: # None when there is no leader
				continue
			leader_vid, leader_gap = veh_info["leader_info"][0], float(veh_info["leader_info"][1]) + self.min_gap_fd
			if leader_vid in self.current_simulation_statistics:
				speed_diff = max(veh_info["speed"] - self.current_simulation_statistics[leader_vid]["speed"], 0.0)
			else:
				print(f"Error: Leader {leader_vid} not found in current_simulation_statistics")
				continue
			if speed_diff > 1e-3:
				ttc = leader_gap / speed_diff
			else:
				ttc = 10000.0
			for threshold in self.ttc_thresholds:
				if ttc < threshold:
					self.current_simulation_statistics[vid][f"TET_{threshold}"] += self.traci_delta_t * 100
					self.current_simulation_statistics[vid][f"TIT_{threshold}"] += (threshold - ttc) * self.traci_delta_t * 100
		return

	def __update_arrived_vehicles(self, arrived_vehicles):
		curr_time = self.traci_conn.simulation.getTime()
		for veh_id in arrived_vehicles:
			if veh_id not in self.current_simulation_statistics:
				print(f"Error: Vehicle {veh_id} not found in current_simulation_statistics")
				continue
			veh_stats = self.current_simulation_statistics.pop(veh_id, None)
			if veh_id in self.current_arrived_vehicle_statistics:
				print("Error: Vehicle {} already arrived".format(veh_id))
			veh_final_statistics = {}
			veh_final_statistics["depart"] = veh_stats["depart"]
			veh_final_statistics["delay"] = veh_stats["depart_delay"]
			veh_final_statistics["vType"] = veh_stats["vType"]
			veh_final_statistics["travel_time"] = curr_time - veh_stats["depart"]
			veh_final_statistics["average_speed"] = self.road_length / veh_final_statistics["travel_time"]
			veh_final_statistics["total_time"] = curr_time - veh_stats["depart"] + veh_stats["depart_delay"]
			veh_final_statistics["waiting_time"] = veh_stats["waiting_time"]
			veh_final_statistics["co2_emission"] = veh_stats["co2_emission"]
			veh_final_statistics["lanechange_count"] = veh_stats["lanechange_count"]
			for threshold in self.ttc_thresholds:
				veh_final_statistics[f"TTC_{threshold}"] = veh_stats[f"TET_{threshold}"]
				veh_final_statistics[f"TET_{threshold}"] = veh_stats[f"TET_{threshold}"] / veh_final_statistics["travel_time"]
				veh_final_statistics[f"TIT_{threshold}"] = veh_stats[f"TIT_{threshold}"] / (veh_final_statistics["travel_time"] * threshold)
			self.current_arrived_vehicle_statistics[veh_id] = veh_final_statistics
		return

	def __update_simulation_metrics(self):
		if not self.current_arrived_vehicle_statistics:
			print("Error: No vehicles arrived in the simulation")
			return dict()
		# update simulation metrics
		_exclude_warm_start = self.config["eval"]["exclude_warm_start"]
		metrics_result = self.aggregated_simulation_info(exclude_warm_start=_exclude_warm_start)
		return metrics_result

	def get_current_simulation_metrics(self):
		result = self.__update_simulation_metrics()
		return result

	def get_action_metrics(self, action):
		assert action.shape == (self.num_lanes, self.n_agents_per_lane), "action shape must be (num_lanes, n_agents_per_lane)"
		result = {}
		for lane in range(self.num_lanes):
			lane_dict = {}
			result[f"lane_{lane}"] = lane_dict
			allow_rate = np.sum(action[lane, :]) / self.n_agents_per_lane
			lane_dict["both"] = allow_rate
		return result

	def aggregated_simulation_info(self, exclude_warm_start=True):
		time_span = self.traci_conn.simulation.getTime()
		trips_info = self.current_arrived_vehicle_statistics.values()
		result = {}
		if exclude_warm_start:
			trips_info = [trip for trip in trips_info if trip["depart"] >= self.warm_start_time]
			time_span = time_span - self.warm_start_time
		vtypes = np.array([trip["vType"] for trip in trips_info])
		ttc_fields = [f"TET_{threshold}" for threshold in self.ttc_thresholds] + [f"TIT_{threshold}" for threshold in self.ttc_thresholds] + \
						[f"TTC_{threshold}" for threshold in self.ttc_thresholds]
		for field in ["delay", "travel_time", "total_time", "co2_emission", "waiting_time", "lanechange_count", "average_speed"] + \
						ttc_fields:
			values = np.array([trip[field] for trip in trips_info]).astype(float)
			avg_value = np.mean(values)
			controlled_avg_value = np.mean(values[vtypes == "controlled"])
			uncontrolled_avg_value = np.mean(values[vtypes == "uncontrolled"])
			result[field] = {"all": avg_value, "controlled": controlled_avg_value, "uncontrolled": uncontrolled_avg_value}
			for subfield in result[field]:
				if np.isnan(result[field][subfield]):
					result[field][subfield] = -1.0
		result["num_vehicles"] = {"all": len(trips_info), "controlled": np.sum(vtypes == "controlled"), "uncontrolled": np.sum(vtypes == "uncontrolled")}
		# average flow per lane per hour
		result["flow"] = {"all": 3600 * len(trips_info) / (time_span * self.num_lanes), "controlled": 3600 * np.sum(vtypes == "controlled") / (time_span * self.num_lanes),
						  "uncontrolled": 3600 * np.sum(vtypes == "uncontrolled") / (time_span * self.num_lanes)}
		return result
