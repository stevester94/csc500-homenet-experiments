#! /usr/bin/env python3

#
# What this tests:
# - if train, val, test datasets are disjoint
# - if each example in each set can be found in the original binary file
# - if train is randomized on each iteration
# - if each dataset is the right shape & length

import torch
import unittest
import numpy as np

from steves_utils.ORACLE.torch_utils import ORACLE_Torch_Dataset
from steves_utils.ORACLE.torch_utils import build_ORACLE_episodic_iterable
from steves_utils.ORACLE.utils_v2 import (
    ALL_DISTANCES_FEET,
    ALL_SERIAL_NUMBERS,
    ALL_RUNS,
    serial_number_to_id
)

num_examples_per_device = 75000
n_train_tasks = 2000
n_val_tasks = 1000
n_test_tasks = 100
max_cache_items = int(4.5e6)
desired_serial_numbers=ALL_SERIAL_NUMBERS
desired_distances=ALL_DISTANCES_FEET
desired_runs=[1]
n_way=len(ALL_SERIAL_NUMBERS)
n_shot=10
n_query=10
window_length=256
window_stride=50
seed=1337


n_train_tasks_per_distance=int(n_train_tasks/len(desired_distances))
n_val_tasks_per_distance=int(n_val_tasks/len(desired_distances))
n_test_tasks_per_distance=int(n_test_tasks/len(desired_distances))
max_cache_size_per_distance=int(max_cache_items/len(desired_distances))
num_examples_per_device_per_distance=int(num_examples_per_device/len(desired_distances))

    


import time

import timeit

ds = None
def build_ds():
    global ds
    ds = build_ORACLE_episodic_iterable(
        desired_serial_numbers=desired_serial_numbers,
        # desired_distances=[50],
        desired_distances=desired_distances,
        desired_runs=desired_runs,
        window_length=window_length,
        window_stride=window_stride,
        num_examples_per_device_per_distance=num_examples_per_device_per_distance,
        seed=seed,
        max_cache_size_per_distance=max_cache_size_per_distance,
        # n_way=len(ALL_SERIAL_NUMBERS),
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_train_tasks_per_distance=n_train_tasks_per_distance,
        n_val_tasks_per_distance=n_val_tasks_per_distance,
        n_test_tasks_per_distance=n_test_tasks_per_distance,
    )

def iterate_ds(ds):
    for x in ds[0]:
        pass
    for x in ds[1]:
        pass

print(
    "Time to load ds:",
    timeit.timeit(build_ds, number=1)
)

print(
    "Time to iterate train and val ds:",
    timeit.timeit(lambda: iterate_ds(ds), number=1)
)

print(
    "Time to iterate train and val ds:",
    timeit.timeit(lambda: iterate_ds(ds), number=1)
)

print(
    "Time to iterate train and val ds:",
    timeit.timeit(lambda: iterate_ds(ds), number=1)
)

print(
    "Time to iterate train and val ds:",
    timeit.timeit(lambda: iterate_ds(ds), number=1)
)