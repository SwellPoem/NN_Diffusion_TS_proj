# Description: Utility functions for the project
import yaml
import torch
import random
import importlib
import numpy as np
from tqdm import tqdm

# Load YAML file
#load_yaml_config before
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
# seed_everything before
# input: seed, cudnn_deterministic
# output: None
def set_seed(seed, cudnn_deterministic=False):
    if seed is not None:
        print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic

    # if cudnn_deterministic:
    #     warnings.warn('You have chosen to seed training. '
    #                   'This will turn on the CUDNN deterministic setting, '
    #                   'which can slow down your training considerably! '
    #                   'You may see unexpected behavior when restarting '
    #                   'from checkpoints.')


# update a configuration dictionary with a list of options
# merge_opts_into_config before
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


# instantiate_from_confiog before
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


def data_generation(no, seq_len, dim, seed=1234):
    st0 = np.random.get_state()
    set_seed(seed)
    # Initialize the output
    local_data, global_data = list(), list()

    # Generate sine data
    for i in tqdm(range(0, no), desc='Generating data'):
        local, glob = list(), list()
        for k in range(dim):

            # Randomly drawn frequency and phase for local data
            freq_local = np.random.uniform(0, 0.1)            
            phase_local = np.random.uniform(0, 0.1)
            # Randomly drawn frequency and phase for global data
            freq_global = np.random.uniform(0, 0.01)            
            phase_global = np.random.uniform(0, 0.01)
            # Generate sine signal based on the drawn frequency and phase for local and global data
            temp_data1 = [np.sin(freq_local * j + phase_local) for j in range(seq_len)]
            temp_data2 = [np.sin(freq_global * j + phase_global) for j in range(seq_len)]
            local.append(temp_data1)
            glob.append(temp_data2)

        # Align row/column and normalize to [0,1]
        local = np.transpose(np.asarray(local))
        local = (local + 1)*0.5
        local_data.append(local)
        glob = np.transpose(np.asarray(glob))
        glob = (glob + 1)*0.5
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
