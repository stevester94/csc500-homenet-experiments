#! /usr/bin/env python3

from os import replace
import torch
import numpy as np
import math
from torch.utils.data import IterableDataset
# from typing import Tuple


"""
stratified_ds is expected to be in the form of
{
    <domain>: {
        <label>: np.ndarray whos shape=(num_examples_per_domain_per_class, x)
    }
}
"""
def create_datasets_from_stratified_ds(
    stratified_ds:dict,
    train_val_test_percents:tuple,
    num_examples_per_domain_per_class:int,
    x_transform_func=None,
)->tuple:
    all_train = []
    all_val   = []
    all_test  = []

    n_train = math.floor(num_examples_per_domain_per_class*train_val_test_percents[0])
    n_val = math.floor(num_examples_per_domain_per_class*train_val_test_percents[1])
    n_test = (num_examples_per_domain_per_class - n_train - n_val)

    if x_transform_func == False:
        x_transform_func = lambda x: x

    for domain, label_and_x_dict in stratified_ds.items():
        for label, all_x in label_and_x_dict.items():

            if len(all_x) < num_examples_per_domain_per_class:
                raise RuntimeError("Number of examples requested for (u={}, y={}) is too high, have only {} but wanted {}".format(
                    domain, label,
                    len(all_x), num_examples_per_domain_per_class
                ))

            train = all_x[:n_train]
            val   = all_x[n_train:n_train+n_val]
            test  = all_x[n_train+n_val:n_train+n_val+n_test]

            for e in train: all_train.append( (x_transform_func(e), label, domain) )
            for e in val: all_val.append(     (x_transform_func(e), label, domain) )
            for e in test: all_test.append(   (x_transform_func(e), label, domain) )
    
    # Lazy, in case I want to wrap them
    all_train = all_train
    all_val = all_val
    all_test = all_test
    
    return all_train, all_val, all_test