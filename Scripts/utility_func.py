# Description: Utility functions for the project
import yaml
import torch
import random
import importlib
import numpy as np
from tqdm import tqdm

# Load YAML file
# input: path to the YAML file
# output: dictionary of the YAML file
def load_yaml(path):
    config = None
    try:
        with open(path, 'r') as file:
            config = yaml.full_load(file)
    except Exception as e:
        print(f"Error loading YAML config: {e}")
    return config

# set seed for pseudo-random number generators
# input: seed
# output: None
def set_seed(seed):
    if seed is not None:
        print(f"Global seed set to {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# update a configuration dictionary with a list of options
# input: config, options
# output: updated config
def merge_options_into_config(config, options):
    #recursive function to update config
    def update_config(config, keys, value):
        key = keys.pop(0)
        if keys:
            config[key] = update_config(config.get(key, {}), keys, value)
        else:
            config[key] = type(config.get(key, value))(value)
        return config

    if options:
        assert len(options) % 2 == 0, "Options should be given in pairs of name and value. The length must be an even number!"
        for i in range(0, len(options), 2):
            keys = options[i].split('.')
            value = options[i + 1]
            config = update_config(config, keys, value)
    return config


# input: configuration
# output: instance
def create_instance_from_config(configuration):
    if configuration is None:
        return None

    if "target" not in configuration:
        raise KeyError("The key `target` is required to create an instance.")

    module_name, class_name = configuration["target"].rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)

    params = configuration.get("params", {})
    instance = class_(**params)

    return instance


def data_generation(no, seq_len, dim, seed=123):
    st0 = np.random.get_state()
    set_seed(seed)
    local_data, global_data = list(), list()
    for i in tqdm(range(0, no), total=no, desc="Sampling Data"):
        local, glob = list(), list()
        for k in range(dim):
            a = np.random.uniform(0, 1)            
            b = np.random.uniform(0, 1)
            c = np.random.uniform(0.1, 0.5)
            e = np.random.uniform(1, 3)

            temp_data1 = [c * (np.sin(0.2 * np.pi * j * (k+1) + a) + 2 * np.sin(0.1 * np.pi * j * (k+1) + b)) for j in range(seq_len)]

            signal = np.random.uniform(0, 1)
            if signal > 0.5:
                temp_data2 = [e * np.sin(0.001 * np.pi * j) for j in range(seq_len)]
            else:
                temp_data2 = [- e * np.sin(0.001 * np.pi * j) for j in range(seq_len)]

            local.append(temp_data1)
            glob.append(temp_data2)
        
        local = np.transpose(np.asarray(local))
        local_data.append(local)
        glob = np.transpose(np.asarray(glob))
        global_data.append(glob)

    np.random.set_state(st0)
    local_data = np.array(local_data)
    global_data = np.array(global_data)
    return local_data, global_data

# Useful functions taken from the original implementation
def exists(x):
    return x is not None

def default(value, default_value):
    if exists(value):
        return value
    return default_value() if callable(default_value) else default_value

def identity(identity, *args, **kwargs):
    return identity

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# normalization functions
# used for normalizing the input data to the model
# in datasets like sines and mujocco
def normalize_to_neg_one_to_one(x):
    return x * 2 - 1

def unnormalize_to_zero_to_one(x):
    return (x + 1) * 0.5
