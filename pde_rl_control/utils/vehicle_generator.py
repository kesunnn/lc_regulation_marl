# %%
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
from pde_rl_control.utils.u import prettify

# %%
def gen_triangular_fd_params(parameters):
    v0, T, min_gap, v_len = parameters["v0"], parameters["time_gap"], parameters["min_gap"], parameters["v_len"]
    rho_m = 1.0 / (min_gap + v_len)
    rho_s = 1.0 / (T * v0 + 1 / rho_m)
    fd = {
        "v0": v0,
        "rho_m": rho_m,
        "time_gap": T,
        "rho_s": rho_s,
        "max_flow": v0 * rho_s
    }
    return fd

default_triangular_fd_params = {
    "v0": 24.6, # m/s, from I80 speed limit
    "rho_m": 120 / 1000, # veh/m
    "time_gap": 1.4, # s
    "rho_s": 1.0 / (1.4 * 24.6 + 1000 / 120) # veh/m
}


def fundamental_diagram_triangular(rho, triangular_fd_params):
    if rho >= 0 and rho <= triangular_fd_params["rho_s"]:
        return rho * triangular_fd_params["v0"]
    elif rho > triangular_fd_params["rho_s"] and rho <= triangular_fd_params["rho_m"]:
        return (1/triangular_fd_params["time_gap"]) * (1 - rho/triangular_fd_params["rho_m"])
    else:
        raise ValueError("rho must be in range [0, rho_max]")

def fundamental_diagram_triangular_inverse(q, triangular_fd_params, traffic_condition="free"):
    Q_max = triangular_fd_params["rho_s"] * triangular_fd_params["v0"]
    if q < 0 or q > Q_max:
        raise ValueError("q must be in range [0, Q_max]")
    if traffic_condition == "free":
        rho = q / triangular_fd_params["v0"]
    elif traffic_condition == "congested":
        rho = triangular_fd_params["rho_m"] * (1 - q * triangular_fd_params["time_gap"])
    else:
        raise ValueError("traffic_condition must be 'free' or 'congested'")
    return rho

def fundamental_diagram_triangular_derivative(rho, triangular_fd_params):
    if rho >= 0 and rho <= triangular_fd_params["rho_s"]:
        return triangular_fd_params["v0"]
    elif rho > triangular_fd_params["rho_s"] and rho <= triangular_fd_params["rho_m"]:
        return -1/(triangular_fd_params["rho_m"]*triangular_fd_params["time_gap"])
    else:
        raise ValueError("rho must be in range [0, rho_max]")

# %%
def generate_vehicle_flows_from_triangular_FD(triangular_fd_params, num_lanes, max_time, desire_Q, dev, control_rate, traffic_cond, input_file, output_file,
                                              use_cached=False, cached_vehicles=None):
    '''
    triangular_fd_params: dict of triangular FD parameters
    num_lanes: number of lanes
    max_time: simulation time in seconds
    desire_Q: desired flow rate in veh/hr
    dev: deviation of speed in percentage
    control_rate: percentage of vehicles that are controlled
    traffic_cond: free or congested
    input_file: input route file
    output_file: output vehicle file

    the speed is calculated from desire_Q and traffic_cond based on triangular FD, and the speed deviation is from devs.
    The flow rate is then converted into emit probablity per second based on the speed, then randomly generate vehicles with the probablity.
    '''
    Q_max = triangular_fd_params["rho_s"] * triangular_fd_params["v0"] * 3600 # max flow rate in veh/hr
    assert traffic_cond in ["free", "congested"], "traffic condition must be either free or congested"
    assert 0 < desire_Q < Q_max, "desire_Q must be in range (0, rho_s * v0)"
    assert dev >= 0, "dev must be non-negative"
    assert num_lanes > 0, "num_lanes must be positive"
    assert max_time > 0, "max_time must be positive"
    assert 0 <= control_rate <= 1, "control_rate must be in range [0, 1]"

    insert_freq = 10 # every 1/10=0.1s
    rho = fundamental_diagram_triangular_inverse(desire_Q/3600, triangular_fd_params, traffic_cond)
    avg_speed = desire_Q / (3600*rho)
    emit_prob = desire_Q / (3600*insert_freq)

    tree = ET.parse(input_file)
    root = tree.getroot()
    generated_vehicles = []
    if use_cached:
        assert cached_vehicles, "cached_vehicles must be provided"
        generated_vehicles = cached_vehicles
        for vehicle in cached_vehicles:
            root.append(vehicle)
    else:
        route_id = '' # only one route is allowed
        for element in root:
            if element.tag == 'route':
                route_id = element.attrib['id']
                break
        assert route_id, "route id not found in the input file"

        for ts in range(max_time*insert_freq):
            for lane in range(num_lanes):
                if np.random.rand() > emit_prob:
                    continue
                vehicle_type = "controlled" if np.random.rand() < control_rate else "uncontrolled"
                vehicle = ET.Element("vehicle")
                vehicle.set("id", "veh_{}.{}".format(lane, ts))
                vehicle.set("type", vehicle_type)
                vehicle.set("route", route_id)
                vehicle.set("depart", str(ts/insert_freq))
                vehicle.set("departLane", str(lane))
                vehicle.set("departPos", "0")
                depart_speed = np.random.normal(avg_speed, avg_speed*dev)
                if depart_speed < max(0., avg_speed - 2*avg_speed*dev):
                    depart_speed = max(0., avg_speed - 2*avg_speed*dev)
                if depart_speed > avg_speed + 2*avg_speed*dev:
                    depart_speed = avg_speed + 2*avg_speed*dev
                depart_speed = round(depart_speed, 1)
                vehicle.set("departSpeed", str(depart_speed))
                if traffic_cond == "congested":
                    vehicle.set("arrivalSpeed", str(depart_speed))
                root.append(vehicle)
                generated_vehicles.append(vehicle)
    pretty_xml = prettify(root)
    with open(output_file, 'w') as f:
        f.write(pretty_xml)
    return generated_vehicles

idm_v_to_rho = []
idm_rho_to_flow = []

def gen_IDM_fd_params(parameters):
    v0, T, min_gap, v_len = parameters["v0"], parameters["time_gap"], parameters["min_gap"], parameters["v_len"]
    a, b = parameters["a"], parameters["b"]
    global idm_v_to_rho, idm_rho_to_flow
    idm_v_to_rho = []
    for v in np.linspace(0, v0*(1-1e-2), 1000):
        s = (min_gap + v*T) / np.sqrt(1 - (v/v0)**4)
        rho = 1.0 / (s + v_len) # veh/m
        idm_v_to_rho.append((v, rho))
    idm_rho_to_flow = []
    max_flow, rho_s = -1, -1
    for idx in reversed(range(len(idm_v_to_rho))):
        v, rho = idm_v_to_rho[idx]
        flow = rho * v
        if flow > max_flow:
            max_flow = flow
            rho_s = rho
        idm_rho_to_flow.append((rho, flow))
    fd = {
        "v0": v0,
        "rho_m": idm_v_to_rho[0][1],
        "time_gap": T,
        "rho_s": rho_s,
        "max_flow": max_flow
    }
    return fd

def get_fundamental_diagram_IDM_entries():
    global idm_v_to_rho, idm_rho_to_flow
    return idm_v_to_rho, idm_rho_to_flow

def fundamental_diagram_IDM(rho, idm_fd_params):
    global idm_rho_to_flow
    assert rho >= 0 and rho <= idm_fd_params["rho_m"], "rho must be in range [0, rho_m]"
    if rho <= idm_v_to_rho[-1][1]:
        v = idm_fd_params["v0"] 
        return rho * v
    for idx in range(len(idm_rho_to_flow)-1):
        if rho >= idm_rho_to_flow[idx][0] and rho <= idm_rho_to_flow[idx+1][0]:
            rho1, flow1 = idm_rho_to_flow[idx]
            rho2, flow2 = idm_rho_to_flow[idx+1]
            # linear interpolation
            return flow1 + (rho - rho1) * (flow2 - flow1) / (rho2 - rho1)
    raise ValueError("rho must be in range [0, rho_m]")

def fundamental_diagram_IDM_inverse(q, idm_fd_params, traffic_condition="free"):
    Q_max = idm_fd_params["max_flow"]
    global idm_rho_to_flow, idm_v_to_rho
    if q < 0 or q > Q_max:
        raise ValueError("q must be in range [0, Q_max]")
    if traffic_condition == "free":
        if q <= idm_v_to_rho[-1][1] * idm_fd_params["v0"]:
            return q / idm_fd_params["v0"]
        for idx in range(len(idm_rho_to_flow)-1):
            rho1, flow1 = idm_rho_to_flow[idx]
            rho2, flow2 = idm_rho_to_flow[idx+1]
            if q >= flow1 and q <= flow2:
                # linear interpolation
                return rho1 + (q - flow1) * (rho2 - rho1) / (flow2 - flow1)
    elif traffic_condition == "congested":
        for idx in reversed(range(len(idm_rho_to_flow))):
            rho1, flow1 = idm_rho_to_flow[idx]
            rho2, flow2 = idm_rho_to_flow[idx-1]
            if q >= flow1 and q <= flow2:
                # linear interpolation
                return rho1 + (q - flow1) * (rho2 - rho1) / (flow2 - flow1)
    raise ValueError("traffic_condition must be 'free' or 'congested'")

def generate_vehicle_flows_from_IDM_FD(idm_fd_params, num_lanes, max_time, desire_Q, dev, control_rate, traffic_cond, input_file, output_file,
                                       use_cached=False, cached_vehicles=None):
    '''
    idm_fd_params: dict of IDM FD parameters
    other parameters are the same as generate_vehicle_flows_from_triangular_FD
    '''

    Q_max = idm_fd_params["rho_s"] * idm_fd_params["v0"] * 3600 # max flow rate in veh/hr
    assert traffic_cond in ["free", "congested"], "traffic condition must be either free or congested"
    assert 0 < desire_Q < Q_max, "desire_Q must be in range (0, rho_s * v0)"
    assert dev >= 0, "dev must be non-negative"
    assert num_lanes > 0, "num_lanes must be positive"
    assert max_time > 0, "max_time must be positive"
    assert 0 <= control_rate <= 1, "control_rate must be in range [0, 1]"

    insert_freq = 10 # every 1/10=0.1s
    rho = fundamental_diagram_IDM_inverse(desire_Q/3600, idm_fd_params, traffic_cond)
    avg_speed = desire_Q / (3600*rho)
    emit_prob = desire_Q / (3600*insert_freq)

    tree = ET.parse(input_file)
    root = tree.getroot()
    generated_vehicles = []
    if use_cached:
        assert cached_vehicles, "cached_vehicles must be provided"
        generated_vehicles = cached_vehicles
        for vehicle in cached_vehicles:
            root.append(vehicle)
    else:
        route_id = '' # only one route is allowed
        for element in root:
            if element.tag == 'route':
                route_id = element.attrib['id']
                break
        assert route_id, "route id not found in the input file"

        for ts in range(max_time*insert_freq):
            for lane in range(num_lanes):
                if np.random.rand() > emit_prob:
                    continue
                vehicle_type = "controlled" if np.random.rand() < control_rate else "uncontrolled"
                vehicle = ET.Element("vehicle")
                vehicle.set("id", "veh_{}.{}".format(lane, ts))
                vehicle.set("type", vehicle_type)
                vehicle.set("route", route_id)
                vehicle.set("depart", str(ts/insert_freq))
                vehicle.set("departLane", str(lane))
                vehicle.set("departPos", "0")
                depart_speed = np.random.normal(avg_speed, avg_speed*dev)
                if depart_speed < max(0., avg_speed - 2*avg_speed*dev):
                    depart_speed = max(0., avg_speed - 2*avg_speed*dev)
                if depart_speed > avg_speed + 2*avg_speed*dev:
                    depart_speed = avg_speed + 2*avg_speed*dev
                depart_speed = round(depart_speed, 1)
                vehicle.set("departSpeed", str(depart_speed))
                if traffic_cond == "congested":
                    vehicle.set("arrivalSpeed", str(depart_speed))
                root.append(vehicle)
                generated_vehicles.append(vehicle)
    pretty_xml = prettify(root)
    with open(output_file, 'w') as f:
        f.write(pretty_xml)
    return generated_vehicles
