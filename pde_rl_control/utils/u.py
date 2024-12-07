# %%
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np

# %%
def prettify(element):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(element, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="\t")
    pretty_xml = '\n'.join(line for line in pretty_xml.split('\n') if line.strip())
    return pretty_xml

def generate_dict_from_list(data, groupby_keys, key, method="avg", key_to_str=True):
    """
    Generate a dictionary from a list of dictionaries
    Args:
        data (list): list of dictionaries
        groupby_keys (list): list of keys to group by
        key (str): key to calculate the average
        method (str): method to calculate the average
    Returns:
        dict: dictionary of the average values
    """
    result = {}
    assert method in ["avg", "sum", "max", "min", "50pt", "90pt", "99pt"], "Invalid method in generate_dict_from_list"
    assert isinstance(groupby_keys, list), "groupby_keys should be a list in generate_dict_from_list"
    if not groupby_keys:
        data_key = [d[key] for d in data]
        if not data_key:
            return None
        return process_data_by_method(data_key, method)
    for d in data:
        current = result
        for k in groupby_keys:
            v = d[k]
            if key_to_str:
                v = str(v)
            if v not in current:
                current[v] = {}
            current = current[v]
        if "data" not in current:
            current["data"] = []
        current["data"].append(d[key])
    process_dict_by_method(result, method)
    return result

def process_dict_by_method(data, method):
    """
    Process dictionary by method
    Args:
        data (dict): dictionary of data
        method (str): method to process the data
    """
    assert method in ["avg", "sum", "max", "min", "50pt", "90pt", "99pt"], "Invalid method in process_dict_by_method"
    for k, v in data.items():
        if "data" in v:
            data[k] = process_data_by_method(v["data"], method)
        else:
            process_dict_by_method(v, method)
    return

def process_data_by_method(data, method):
    """
    Process data by method
    Args:
        data (list): list of data
        method (str): method to process the data
    Returns:
        float: the processed data
    """
    assert method in ["avg", "sum", "max", "min", "50pt", "90pt", "99pt"], "Invalid method in process_data_by_method"
    if method == "avg":
        return np.mean(data)
    elif method == "sum":
        return np.sum(data)
    elif method == "max":
        return np.max(data)
    elif method == "min":
        return np.min(data)
    elif method == "50pt":
        return np.percentile(data, 50)
    elif method == "90pt":
        return np.percentile(data, 90)
    elif method == "99pt":
        return np.percentile(data, 99)
