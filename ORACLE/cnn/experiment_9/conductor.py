#! /usr/bin/env python3
import json

from steves_utils.ORACLE.utils_v2 import (
    ALL_DISTANCES_FEET,
    ALL_SERIAL_NUMBERS,
    ALL_RUNS,
    serial_number_to_id
)

###########################################
# Build all experiment json parameters
###########################################
experiment_jsons = []

base_parameters = {}
base_parameters = {}
base_parameters["experiment_name"] = "oracle_cnn_experiment_9"
base_parameters["lr"] = 0.0001
base_parameters["n_epoch"] = 10
base_parameters["batch_size"] = 256
base_parameters["patience"] = 10
base_parameters["seed"] = 1337
base_parameters["device"] = "cuda"
base_parameters["desired_serial_numbers"] = ALL_SERIAL_NUMBERS

base_parameters["window_stride"]=50
base_parameters["window_length"]=256 #Will break if not 256 due to model hyperparameters
base_parameters["desired_runs"]=[1]
base_parameters["num_examples_per_device"]=75000
base_parameters["max_cache_items"] = 4.5e6

#base_parameters["num_examples_per_device"]=260


base_parameters["x_net"] =     [# droupout, groups, 512 out
    {"class": "nnReshape", "kargs": {"shape":[-1, 1, 2, 128]}},
    {"class": "Conv2d", "kargs": { "in_channels":1, "out_channels":256, "kernel_size":(1,7), "bias":False, "padding":(0,3), },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "BatchNorm2d", "kargs": {"num_features":256}},

    {"class": "Conv2d", "kargs": { "in_channels":256, "out_channels":80, "kernel_size":(2,7), "bias":True, "padding":(0,3), },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "BatchNorm2d", "kargs": {"num_features":80}},
    {"class": "Flatten", "kargs": {}},

    {"class": "Linear", "kargs": {"in_features": 80*128, "out_features": 256}}, # 80 units per IQ pair
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "BatchNorm1d", "kargs": {"num_features":256}},

    {"class": "Linear", "kargs": {"in_features": 256, "out_features": 256}},
]

parameters = base_parameters

# These will get permuted so we cover every possible case
custom_parameters = {}
custom_parameters["seed"] = [1337, 1984, 2020, 18081994, 4321326]


import copy
import itertools
keys, values = zip(*custom_parameters.items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

#([], [2,8,14,20,26,32,38,44,50,56,62]),
for source_domains, target_domains in [
    ([2,38,62], [8,14,20,26,32,44,50,56]),
    ([2,8,14], [20,26,32,38,44,50,56,62]),
    ([26,32,38], [2,8,14,20,44,50,56,62]),
    ([50,56,62], [2,8,14,20,26,32,38,44]),
]:
    for d in permutations_dicts:
        parameters = copy.deepcopy(base_parameters)

        for key,val in d.items():
            parameters[key] = val

        parameters["source_domains"] = source_domains
        parameters["target_domains"] = target_domains
        
        j = json.dumps(parameters, indent=2)
        experiment_jsons.append(j)

import random
random.seed(1337)
random.shuffle(experiment_jsons)

###########################################
# Run all experiments using Conductor
###########################################
import os
from steves_utils.conductor import Conductor

conductor = Conductor(
    TRIALS_BASE_PATH=os.path.realpath("./results/"),
    EXPERIMENT_PATH=os.path.join(os.environ["DRIVER_ROOT_PATH"], "ORACLE/cnn/driver_1/"),
    KEEP_MODEL=True,
)
conductor.conduct_experiments(experiment_jsons)
