from pde_rl_control.utils.traci_u import extract_vehicle_root_type, extract_vehicle_root_type_from_id


def dummy_event_generator(traci_conn):
	# exemptive_vehicles = []
	return None

def lane_degrade_event_generator(traci_conn, edge_id, lane_id, start_pos, end_pos, start_time, end_time, new_tau):
	"""
	Generate a lane block event
	Args:
		traci_conn (TraCI): TraCI connection
		lane_id (str): lane ID
		start_pos (float): start position
		end_pos (float): end position
		start_time (float): start time
		end_time (float): end time
	"""
	exemptive_vehicles = []
	# set all vehicles in the position [start_pos, end_pos] a new tau
	# setTau will change vtype to a unique vtype. Eg, controlled -> controlled@veh_0.3948
	curr_time = traci_conn.simulation.getTime()
	if curr_time >= start_time and curr_time < end_time + traci_conn.simulation.getDeltaT():
		all_vehicles = traci_conn.edge.getLastStepVehicleIDs(edge_id)
		vehicles = traci_conn.lane.getLastStepVehicleIDs(lane_id)
		for vehicle in all_vehicles:
			pos = traci_conn.vehicle.getPosition(vehicle)[0]
			if pos < start_pos:
				continue
			if pos <= end_pos and vehicle in vehicles and curr_time < end_time:
				traci_conn.vehicle.setTau(vehicle, new_tau)
				traci_conn.vehicle.setLaneChangeMode(vehicle, 256)
				exemptive_vehicles.append(vehicle)
			else:
				vtype = traci_conn.vehicle.getTypeID(vehicle)
				vtype_root = extract_vehicle_root_type(vtype)
				if vtype_root != vtype:
					traci_conn.vehicle.setType(vehicle, vtype_root)
	return exemptive_vehicles

degrade_vehicle_id = ""

def vehicle_degrade_event_generator(traci_conn, lane_id, start_pos, end_pos, trigger_time, new_tau):
	"""
	Generate a degrade vehicle event
	Args:
		traci_conn (TraCI): TraCI connection
		lane_id (str): lane ID
		start_pos (float): start position
		end_pos (float): end position
		trigger_time (float): trigger time
	"""
	global degrade_vehicle_id
	exemptive_vehicles = []
	# set all vehicles in the position [start_pos, end_pos] a new speed
	curr_time = traci_conn.simulation.getTime()
	if curr_time >= trigger_time and curr_time < trigger_time + traci_conn.simulation.getDeltaT():
		_success = False
		vehicles = traci_conn.lane.getLastStepVehicleIDs(lane_id)
		for vehicle in vehicles:
			pos = traci_conn.vehicle.getPosition(vehicle)[0]
			if pos >= start_pos and pos < end_pos:
				traci_conn.vehicle.setTau(vehicle, new_tau)
				traci_conn.vehicle.setLaneChangeMode(vehicle, 256)
				exemptive_vehicles.append(vehicle)
				degrade_vehicle_id = vehicle
				_success = True
				break
		if not _success:
			degrade_vehicle_id = ""
	elif curr_time >= trigger_time + traci_conn.simulation.getDeltaT():
		exemptive_vehicles.append(degrade_vehicle_id) # keep the vehicle in the list until the next trigger time
	return exemptive_vehicles

stop_vehicle_id = ""

def vehicle_stop_event_generator(traci_conn, edge_id, lane_index, start_pos, end_pos, trigger_time, duration):
	"""
	Generate a stop vehicle event
	Args:
		traci_conn (TraCI): TraCI connection
		edge_id (str): edge ID
		lane_index (int): lane index
		start_pos (float): start position
		end_pos (float): end position
		trigger_time (float): trigger time
	"""
	global stop_vehicle_id
	exemptive_vehicles = []
	# set all vehicles in the position [start_pos, end_pos] a new speed
	curr_time = traci_conn.simulation.getTime()
	if curr_time >= trigger_time and curr_time < trigger_time + traci_conn.simulation.getDeltaT():
		lane_id = edge_id + "_" + str(lane_index)
		lane_length = traci_conn.lane.getLength(lane_id)
		_success = False
		vehicles = traci_conn.lane.getLastStepVehicleIDs(lane_id)
		for vehicle in vehicles:
			pos = traci_conn.vehicle.getPosition(vehicle)[0]
			if pos >= start_pos and pos < end_pos:
				speed, decel = traci_conn.vehicle.getSpeed(vehicle), traci_conn.vehicle.getDecel(vehicle)
				decel_distance = speed * speed / (2 * 0.95 * decel)
				if decel_distance + pos > lane_length:
					continue
				traci_conn.vehicle.setStop(vehicle, edge_id, pos=decel_distance + pos, laneIndex=int(lane_index), duration=duration)
				traci_conn.vehicle.setLaneChangeMode(vehicle, 256)
				exemptive_vehicles.append(vehicle)
				stop_vehicle_id = vehicle
				_success = True
				break
		if not _success:
			stop_vehicle_id = ""
	elif curr_time >= trigger_time + traci_conn.simulation.getDeltaT():
		exemptive_vehicles.append(stop_vehicle_id)  # keep the vehicle in the list until the next trigger time
	return exemptive_vehicles

