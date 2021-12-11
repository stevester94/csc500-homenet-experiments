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
base_parameters["experiment_name"] = "One Distance ORACLE PTN"
base_parameters["lr"] = 0.001
base_parameters["device"] = "cuda"
base_parameters["max_cache_items"] = 4.5e6

base_parameters["desired_serial_numbers"] = ALL_SERIAL_NUMBERS
# base_parameters["desired_serial_numbers"] = [
#     "3123D52",
#     "3123D65",
#     "3123D79",
#     "3123D80",
# ]
base_parameters["source_domains"] = [38,]
base_parameters["target_domains"] = [
    20,
    44,
    2,
    8,
    14,
    26,
    32,
    50,
    56,
    62
]

base_parameters["window_stride"]=50
base_parameters["window_length"]=256
base_parameters["desired_runs"]=[1]
base_parameters["num_examples_per_device"]=75000
base_parameters["num_examples_per_device"]=7500

base_parameters["n_shot"] = 3
base_parameters["n_way"]  = len(base_parameters["desired_serial_numbers"])
base_parameters["n_query"]  = 2
base_parameters["n_train_tasks"] = 2000
base_parameters["n_train_tasks"] = 100
base_parameters["n_val_tasks"]  = 100
base_parameters["n_test_tasks"]  = 100

base_parameters["n_epoch"] = 100
base_parameters["n_epoch"] = 3

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

base_parameters["patience"] = 10

parameters = base_parameters

# These will get permuted so we cover every possible case
custom_parameters = {}
custom_parameters["seed"] = [1337, 420]
# custom_parameters["patience"] = [10, 11]


import copy
import itertools
keys, values = zip(*custom_parameters.items())
permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

for d in permutations_dicts:
    parameters = copy.deepcopy(base_parameters)

    for key,val in d.items():
        parameters[key] = val
    
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
    EXPERIMENT_PATH=os.path.join(os.environ["DRIVER_ROOT_PATH"], "ORACLE/ptn/driver_1/"),
    KEEP_MODEL=True,
)
conductor.conduct_experiments(experiment_jsons)
