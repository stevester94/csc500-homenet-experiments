#! /usr/bin/env python3
import json
import os

from steves_utils.ORACLE.utils_v2 import (
    ALL_DISTANCES_FEET,
    ALL_SERIAL_NUMBERS,
    ALL_RUNS,
    serial_number_to_id
)

###########################################
# Build all experiment json parameters
###########################################
base_parameters = {}
base_parameters["experiment_name"] = "MANUAL ORACLE CNN"
base_parameters["lr"] = 0.0001
base_parameters["n_epoch"] = 3
base_parameters["batch_size"] = 256
base_parameters["patience"] = 10
base_parameters["seed"] = 1337
base_parameters["dataset_seed"] = 1337
base_parameters["device"] = "cuda"
base_parameters["desired_classes"] = ALL_SERIAL_NUMBERS
base_parameters["source_domains"] = [50,32,8]
base_parameters["target_domains"] = list(set(ALL_DISTANCES_FEET) - set([50,32,8]))

base_parameters["window_stride"]=50
base_parameters["window_length"]=512
base_parameters["desired_runs"]=[1]
base_parameters["num_examples_per_class_per_domain"]=1000
base_parameters["max_cache_items"] = 4.5e6

base_parameters["criteria_for_best"] = "source"
base_parameters["normalize_source"] = False
base_parameters["normalize_target"] = False

base_parameters["NUM_LOGS_PER_EPOCH"] = 10
base_parameters["RESULTS_DIR"] = "./results"
base_parameters["EXPERIMENT_JSON_PATH"] = os.path.join(base_parameters["RESULTS_DIR"], "experiment.json")
base_parameters["LOSS_CURVE_PATH"] = os.path.join(base_parameters["RESULTS_DIR"], "loss.png")
base_parameters["BEST_MODEL_PATH"] = os.path.join(base_parameters["RESULTS_DIR"], "best_model.pth")



base_parameters["x_net"] =     [
    {"class": "nnReshape", "kargs": {"shape":[-1, 1, 2, 256]}},
    {"class": "Conv2d", "kargs": { "in_channels":1, "out_channels":256, "kernel_size":(1,7), "bias":False, "padding":(0,3), },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "BatchNorm2d", "kargs": {"num_features":256}},

    {"class": "Conv2d", "kargs": { "in_channels":256, "out_channels":80, "kernel_size":(2,7), "bias":True, "padding":(0,3), },},
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "BatchNorm2d", "kargs": {"num_features":80}},
    {"class": "Flatten", "kargs": {}},

    {"class": "Linear", "kargs": {"in_features": 80*256, "out_features": 256}}, # 80 units per IQ pair
    {"class": "ReLU", "kargs": {"inplace": True}},
    {"class": "BatchNorm1d", "kargs": {"num_features":256}},

    {"class": "Linear", "kargs": {"in_features": 256, "out_features": len(base_parameters["desired_classes"])}},
]


parameters = base_parameters

# These will get permuted so we cover every possible case
custom_parameters = {}
custom_parameters["seed"] = [1337, 1984, 2020, 18081994, 4321326]

experiment_jsons = []
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
