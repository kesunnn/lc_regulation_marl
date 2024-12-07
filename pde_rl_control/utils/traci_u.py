def extract_vehicle_root_type(vtype):
    # Eg: controlled:left@veh_0.3937 -> controlled
    type1 = vtype.split("@")[0]
    type2 = type1.split(":")[0]
    return type2


def extract_vehicle_root_type_from_id(traci_conn, vehicle_id):
    """
    Extract the root type of the vehicle
    Args:
        traci_conn (TraCI): TraCI connection
        vehicle_id (str): vehicle ID
    Returns:
        str: the root type of the vehicle
    """
    vtype = traci_conn.vehicle.getTypeID(vehicle_id)
    return extract_vehicle_root_type(vtype)