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
# in sines dataset
def normalize_to_neg_one_to_one(x):
    return x * 2 - 1

def unnormalize_to_zero_to_one(x):
    return (x + 1) * 0.5
