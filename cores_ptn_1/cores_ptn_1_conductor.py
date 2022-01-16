#! /usr/bin/env python3
import json
import os

from steves_utils.CORES.utils import (
    ALL_NODES_MINIMUM_1000_EXAMPLES,
    ALL_NODES,
    ALL_DAYS
)

###########################################
# Build all experiment json parameters
###########################################
base_parameters = {}
base_parameters["experiment_name"] = "cores_ptn_1"
base_parameters["lr"] = 0.001
base_parameters["device"] = "cuda"

base_parameters["seed"] = 1337
base_parameters["desired_classes_source"] = ALL_NODES
# base_parameters["desired_classes_target"] = list(set(ALL_NODES) - set(ALL_NODES_MINIMUM_1000_EXAMPLES))
base_parameters["desired_classes_target"] = ALL_NODES

# base_parameters["source_domains"] = [1]
# base_parameters["target_domains"] = [2,3,4,5]

base_parameters["num_examples_per_class_per_domain_source"]=100
base_parameters["num_examples_per_class_per_domain_target"]=100

base_parameters["n_shot"] = 3
base_parameters["n_way"]  = len(base_parameters["desired_classes_source"])
base_parameters["n_query"]  = 2
base_parameters["train_k_factor"] = 1
base_parameters["val_k_factor"] = 2
base_parameters["test_k_factor"] = 2

base_parameters["n_epoch"] = 100

base_parameters["patience"] = 10
base_parameters["normalize_source"] = False
base_parameters["normalize_target"] = False

base_parameters["criteria_for_best"] = "target"

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

    {"class": "Linear", "kargs": {"in_features": 256, "out_features": 256}},
]


# Parameters relevant to results
# These parameters will basically never need to change
base_parameters["NUM_LOGS_PER_EPOCH"] = 10
base_parameters["RESULTS_DIR"] = "./results"
base_parameters["EXPERIMENT_JSON_PATH"] = os.path.join(base_parameters["RESULTS_DIR"], "experiment.json")
base_parameters["LOSS_CURVE_PATH"] = os.path.join(base_parameters["RESULTS_DIR"], "loss.png")
base_parameters["BEST_MODEL_PATH"] = os.path.join(base_parameters["RESULTS_DIR"], "best_model.pth")


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
    ([1], [2,3,4,5]),
    ([2], [1,3,4,5]),
    ([3], [1,2,4,5]),
    ([4], [1,2,3,5]),
    ([5], [1,2,3,4]),
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
    EXPERIMENT_PATH=os.path.join(os.environ["DRIVER_ROOT_PATH"], "CORES/ptn/driver_1/"),
    KEEP_MODEL=True,
)
conductor.conduct_experiments(experiment_jsons)
