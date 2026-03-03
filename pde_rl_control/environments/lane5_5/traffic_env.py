# %%
import copy
import os
import time
import traceback

import numpy as np
import sumolib
import traci
from traci.exceptions import FatalTraCIError, TraCIException

from pde_rl_control.environments.lane5_5_base import (
    Traffic_Env_Four_Action_Aggressive_With_Baseline,
)
from pde_rl_control.utils.traci_u import extract_vehicle_root_type


# %%
class Traffic_Env_Four_Action_Aggressive_With_Baseline_Buffered(
    Traffic_Env_Four_Action_Aggressive_With_Baseline
):
    """lane5_5 keeps a 1 km control window with only a 200 m tail buffer."""

    road_length = 1000.0
    T_fd = 1.4
    physical_road_length = 1200.0
    control_start_pos = 0.0
    control_end_pos = 1000.0
    lane_change_disable_start_pos = 1000.0

    def __init__(
        self,
        grid_length=50.0,
        control_rate=1.0,
        density_level=0.1,
        event_generator=None,
        vehicle_generator=None,
        config={},
    ):
        super().__init__(
            grid_length=grid_length,
            control_rate=control_rate,
            density_level=density_level,
            event_generator=event_generator,
            vehicle_generator=vehicle_generator,
            config=config,
        )
        self.control_length = self.control_end_pos - self.control_start_pos
        assert self.control_length == self.road_length, "control window must match road_length"
        self.sumo_config_template = os.path.join(os.path.dirname(__file__), "sumo_cfgs")
        self.sumo_config_dir = os.path.join(
            os.path.dirname(__file__), "sumo_cfgs_" + config["meta"]["log_dir"]
        )
        self.current_section_vehicle_statistics = {}
        self.current_arrived_section_statistics = {}
        self.section_free_flow_time = self.control_length / max(self.desired_velocity, 1e-6)
        self.warm_start_time = max(
            self.warm_start_time, int(self.physical_road_length / self.desired_velocity)
        )

    def reset(self):
        initial_state = super().reset()
        self.current_section_vehicle_statistics = {}
        self.current_arrived_section_statistics = {}
        return initial_state

    def _is_controlled_position(self, position_x):
        return self.control_start_pos <= position_x < self.control_end_pos

    def _position_to_agent(self, position_x):
        if not self._is_controlled_position(position_x):
            return None
        agent_idx = int((position_x - self.control_start_pos) // self.grid_length)
        if 0 <= agent_idx < self.n_agents_per_lane:
            return agent_idx
        return None

    def _is_lane_change_disabled_position(self, position_x):
        return self.lane_change_disable_start_pos <= position_x < self.physical_road_length

    def _get_state(self):
        state_dim = 4
        global_state = np.zeros((self.num_lanes, self.n_agents_per_lane, state_dim))
        rho_max, v_max = self.fd_params["rho_m"], self.fd_params["v0"]
        for lane in range(self.num_lanes):
            lane_id = f"e1_{lane}"
            veh_list = self.traci_conn.lane.getLastStepVehicleIDs(lane_id)
            agent_veh_list = {agent_idx: [] for agent_idx in range(self.n_agents_per_lane)}
            for veh_id in veh_list:
                veh_pos = self.traci_conn.vehicle.getPosition(veh_id)[0]
                veh_agent = self._position_to_agent(veh_pos)
                if veh_agent is None:
                    continue
                agent_veh_list[veh_agent].append(veh_id)
            for agent_idx in range(self.n_agents_per_lane):
                avg_density, avg_speed, contr_avg_density, contr_avg_speed = self._aggregate_vehicle_states(
                    agent_veh_list[agent_idx]
                )
                if self.state_normalize:
                    avg_density /= rho_max
                    avg_speed /= v_max
                    contr_avg_density /= rho_max
                    contr_avg_speed /= v_max
                global_state[lane][agent_idx] = np.array(
                    [avg_density, avg_speed, contr_avg_density, contr_avg_speed]
                )
        return global_state

    def _execute_action(self, action, info):
        exemptive_vehicles = info.get("exemptive_vehicles", [])
        for lane in range(self.num_lanes):
            lane_id = f"e1_{lane}"
            veh_list = self.traci_conn.lane.getLastStepVehicleIDs(lane_id)
            for veh_id in veh_list:
                veh_pos = self.traci_conn.vehicle.getPosition(veh_id)[0]
                if self._is_lane_change_disabled_position(veh_pos):
                    self.traci_conn.vehicle.setLaneChangeMode(veh_id, 256)
                    continue
                vtype = self.traci_conn.vehicle.getTypeID(veh_id)
                root_vtype = extract_vehicle_root_type(vtype)
                if root_vtype != "controlled" or veh_id in exemptive_vehicles:
                    continue
                veh_agent = self._position_to_agent(veh_pos)
                if veh_agent is None:
                    if vtype != "controlled":
                        self.traci_conn.vehicle.setType(veh_id, "controlled")
                    self.traci_conn.vehicle.setLaneChangeMode(veh_id, 1621)
                    continue
                if action[lane, veh_agent] == 0.0:
                    if vtype != "controlled":
                        self.traci_conn.vehicle.setType(veh_id, "controlled")
                    self.traci_conn.vehicle.setLaneChangeMode(veh_id, 256)
                elif action[lane, veh_agent] == 1.0:
                    if vtype != "controlled":
                        self.traci_conn.vehicle.setType(veh_id, "controlled")
                    self.traci_conn.vehicle.setLaneChangeMode(veh_id, 1621)
                elif action[lane, veh_agent] == 2.0:
                    if vtype != "controlled:left":
                        self.traci_conn.vehicle.setType(veh_id, "controlled:left")
                    self.traci_conn.vehicle.setLaneChangeMode(veh_id, 1621)
                elif action[lane, veh_agent] == 3.0:
                    if vtype != "controlled:right":
                        self.traci_conn.vehicle.setType(veh_id, "controlled:right")
                    self.traci_conn.vehicle.setLaneChangeMode(veh_id, 1621)
                else:
                    raise ValueError(
                        "Invalid action value for lane: {}, agent: {}, value={}".format(
                            lane, veh_agent, action[lane, veh_agent]
                        )
                    )
        return

    def _init_section_vehicle_stats(self, veh_id, veh_info, enter_time):
        stats = {
            "depart": float(enter_time),
            "vType": extract_vehicle_root_type(veh_info["vType"]),
            "lane": int(veh_info["lane_index"]),
            "waiting_time": 0.0,
            "last_waiting_time_total": float(veh_info["waiting_time"]),
            "co2_emission": 0.0,
            "lanechange_count": 0,
        }
        if self.enable_ttc_metrics:
            for threshold in self.ttc_thresholds:
                stats[f"TTC_{threshold}"] = 0.0
                stats[f"TET_{threshold}"] = 0.0
                stats[f"TIT_{threshold}"] = 0.0
        self.current_section_vehicle_statistics[veh_id] = stats
        return stats

    def _finalize_section_vehicle(self, veh_id, exit_time):
        veh_stats = self.current_section_vehicle_statistics.pop(veh_id, None)
        if veh_stats is None:
            return
        travel_time = max(float(exit_time) - veh_stats["depart"], self.traci_delta_t)
        final_stats = {
            "depart": veh_stats["depart"],
            "delay": max(travel_time - self.section_free_flow_time, 0.0),
            "vType": veh_stats["vType"],
            "travel_time": travel_time,
            "average_speed": self.control_length / travel_time,
            "total_time": travel_time,
            "waiting_time": veh_stats["waiting_time"],
            "co2_emission": veh_stats["co2_emission"],
            "lanechange_count": veh_stats["lanechange_count"],
        }
        if self.enable_ttc_metrics:
            for threshold in self.ttc_thresholds:
                tet_total = veh_stats[f"TET_{threshold}"]
                tit_total = veh_stats[f"TIT_{threshold}"]
                final_stats[f"TTC_{threshold}"] = tet_total
                final_stats[f"TET_{threshold}"] = tet_total / travel_time
                final_stats[f"TIT_{threshold}"] = tit_total / (travel_time * threshold)
        self.current_arrived_section_statistics[veh_id] = final_stats
        return

    def _update_section_vehicle_statistics(self, subscription_results):
        if not subscription_results:
            return
        curr_time = self.traci_conn.simulation.getTime()
        for veh_id, veh_info in subscription_results.items():
            position_x = self.traci_conn.vehicle.getPosition(veh_id)[0]
            in_zone = self._is_controlled_position(position_x)
            section_stats = self.current_section_vehicle_statistics.get(veh_id)
            if section_stats is None and in_zone:
                section_stats = self._init_section_vehicle_stats(veh_id, veh_info, curr_time)
            if section_stats is None:
                continue
            if position_x >= self.control_end_pos:
                self._finalize_section_vehicle(veh_id, curr_time)
                continue
            lane_index = int(veh_info["lane_index"])
            if lane_index != section_stats["lane"]:
                section_stats["lanechange_count"] += 1
                section_stats["lane"] = lane_index
            waiting_time_total = float(veh_info["waiting_time"])
            waiting_time_delta = max(
                waiting_time_total - section_stats["last_waiting_time_total"], 0.0
            )
            section_stats["waiting_time"] += waiting_time_delta
            section_stats["last_waiting_time_total"] = waiting_time_total
            section_stats["co2_emission"] += float(veh_info["co2_emission"]) * self.traci_delta_t
        if not self.enable_ttc_metrics:
            return
        for veh_id, veh_info in subscription_results.items():
            section_stats = self.current_section_vehicle_statistics.get(veh_id)
            if section_stats is None:
                continue
            if not veh_info.get("leader_info"):
                continue
            leader_vid = veh_info["leader_info"][0]
            leader_gap = float(veh_info["leader_info"][1]) + self.min_gap_fd
            leader_stats = self.current_simulation_statistics.get(leader_vid)
            if not leader_stats:
                continue
            speed_diff = max(float(veh_info["speed"]) - float(leader_stats.get("speed", 0.0)), 0.0)
            ttc = leader_gap / speed_diff if speed_diff > 1e-3 else 10000.0
            for threshold in self.ttc_thresholds:
                if ttc < threshold:
                    section_stats[f"TTC_{threshold}"] += self.traci_delta_t * 100
                    section_stats[f"TET_{threshold}"] += self.traci_delta_t * 100
                    section_stats[f"TIT_{threshold}"] += (threshold - ttc) * self.traci_delta_t * 100
        return

    def _update_section_simulation_metrics(self):
        if not self.current_arrived_section_statistics:
            print("Error: No vehicles traversed the control section in the simulation")
            return {}
        return self.aggregated_simulation_info(
            exclude_warm_start=self.config["eval"]["exclude_warm_start"]
        )

    def get_current_simulation_metrics(self):
        return self._update_section_simulation_metrics()

    def aggregated_simulation_info(self, exclude_warm_start=True):
        time_span = self.traci_conn.simulation.getTime()
        trips_info = list(self.current_arrived_section_statistics.values())
        if exclude_warm_start:
            trips_info = [trip for trip in trips_info if trip["depart"] >= self.warm_start_time]
            time_span = max(time_span - self.warm_start_time, self.traci_delta_t)
        if len(trips_info) == 0:
            print("Error: No section trips found after warm-start filtering")
            return {}
        vtypes = np.array([trip["vType"] for trip in trips_info])
        ttc_fields = []
        if self.enable_ttc_metrics:
            ttc_fields = (
                [f"TET_{threshold}" for threshold in self.ttc_thresholds]
                + [f"TIT_{threshold}" for threshold in self.ttc_thresholds]
                + [f"TTC_{threshold}" for threshold in self.ttc_thresholds]
            )
        result = {}
        for field in [
            "delay",
            "travel_time",
            "total_time",
            "co2_emission",
            "waiting_time",
            "lanechange_count",
            "average_speed",
        ] + ttc_fields:
            values = np.array([trip[field] for trip in trips_info]).astype(float)
            avg_value = np.mean(values)
            controlled_avg_value = np.mean(values[vtypes == "controlled"])
            uncontrolled_avg_value = np.mean(values[vtypes == "uncontrolled"])
            result[field] = {
                "all": avg_value,
                "controlled": controlled_avg_value,
                "uncontrolled": uncontrolled_avg_value,
            }
            for subfield in result[field]:
                if np.isnan(result[field][subfield]):
                    result[field][subfield] = -1.0
        result["num_vehicles"] = {
            "all": len(trips_info),
            "controlled": int(np.sum(vtypes == "controlled")),
            "uncontrolled": int(np.sum(vtypes == "uncontrolled")),
        }
        result["flow"] = {
            "all": 3600 * len(trips_info) / (time_span * self.num_lanes),
            "controlled": 3600 * np.sum(vtypes == "controlled") / (time_span * self.num_lanes),
            "uncontrolled": 3600 * np.sum(vtypes == "uncontrolled") / (time_span * self.num_lanes),
        }
        return result

    def step(self, action):
        info = {}
        action = self._process_action(action, info)
        done = False
        reward_step_count = int(self.reward_step / self.traci_delta_t)
        step_rewards = []
        info["start_step"] = self._step_count
        info["start_time"] = self.traci_conn.simulation.getTime()
        for step_idx in range(int(self.delta_T / self.traci_delta_t)):
            exemptive_vehicles = self.event_generator(
                self.traci_conn, **self.event_generator_parameters[self.event_generator_index]
            )
            if exemptive_vehicles:
                info["exemptive_vehicles"] = exemptive_vehicles
            self._execute_action(action, info)
            if (step_idx + 1) % reward_step_count == 0:
                temp_state = self._get_state()
                step_rewards.append(self._get_step_reward(temp_state, info))
            self.traci_conn.simulationStep()
            self._step_count += 1
            for veh_id in self.traci_conn.simulation.getDepartedIDList():
                self.traci_conn.vehicle.subscribe(veh_id, self.subscription_fields)
                if self.enable_ttc_metrics:
                    self.traci_conn.vehicle.subscribeLeader(veh_id, dist=500)
            subscription_results = self.traci_conn.vehicle.getAllSubscriptionResults()
            subscription_results = self._Traffic_Env__process_subscription_results(subscription_results)
            self._Traffic_Env__update_simulation_statistics(subscription_results)
            self._update_section_vehicle_statistics(subscription_results)
            arrived_vehs = self.traci_conn.simulation.getArrivedIDList()
            if len(arrived_vehs) > 0:
                self._Traffic_Env__update_arrived_vehicles(arrived_vehs)
                for veh_id in arrived_vehs:
                    self.current_section_vehicle_statistics.pop(veh_id, None)
            curr_time = self.traci_conn.simulation.getTime()
            if (
                self.enable_detector_metrics
                and curr_time > self.detector_interval
                and (curr_time - self.delta_T) % self.detector_interval == 0
            ):
                self._Traffic_Env__retrieve_detector_data()
            if curr_time > self.warm_start_time:
                self._execute_warm_start()
            max_episode_time = self.max_time if not self.is_eval else self.max_eval_time
            if (
                curr_time >= max_episode_time
                or self.traci_conn.simulation.getMinExpectedNumber() <= 0
                or self._check_collision(info)
            ):
                done = True
                metrics_result = self._update_section_simulation_metrics()
                self.last_simulation_metrics = copy.deepcopy(metrics_result)
                info["simulation_metrics"] = metrics_result
                info["detector_metrics"] = self.detector_summary() if self.enable_detector_metrics else {}
                break
        info["end_step"] = self._step_count
        info["end_time"] = self.traci_conn.simulation.getTime()
        state = self._get_state()
        if not step_rewards:
            print(
                "done: {}, is_collision: {}, curr_time: {:.2f}".format(
                    done, info["is_collision"], info["end_time"]
                )
            )
            step_rewards.append(self._get_step_reward(state, info))
        reward, global_reward = self._get_reward(step_rewards, info)
        info["reward"] = reward
        info["global_reward"] = global_reward
        if self.return_detector_data:
            info["detector_data"] = copy.deepcopy(self.detector_data)
        return state, reward, done, info
