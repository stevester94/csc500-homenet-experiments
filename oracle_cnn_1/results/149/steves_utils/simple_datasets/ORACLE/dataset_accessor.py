#! /usr/bin/env python3
import numpy as np
import pickle
import os
from steves_utils.ORACLE.utils_v2 import (
    ALL_DISTANCES_FEET,
    ALL_SERIAL_NUMBERS,
    serial_number_to_id
)
from steves_utils.simple_datasets.general_dataset import create_datasets_from_stratified_ds
from steves_utils.utils_v2 import (norm, get_datasets_base_path)
from steves_utils.simple_datasets.ORACLE.utils import genericize_stratified_dataset

MAX_EXAMPLES = 10000

def get_datasets(
    serial_numbers:list,
    distances:list,
    num_examples_per_distance_per_serial:int,
    normalize_type:str=False,
    pickle_path:str=os.path.join(get_datasets_base_path(), "oracle.stratified_ds.2022A.pkl"),
    train_val_test_percents=(0.7,0.15,0.15)
)->tuple:
    with open(pickle_path, "rb") as f:
        stratified_ds_all = pickle.load(f)

    gsd = genericize_stratified_dataset(sds=stratified_ds_all["data"],
         domains=distances, labels=serial_numbers, n_per_u_per_y=num_examples_per_distance_per_serial)

    if normalize_type != False:
        x_transform_func = lambda x: norm(x, normalize_type)
    else:
        x_transform_func = False

    ds = create_datasets_from_stratified_ds(
        stratified_ds=gsd,
        train_val_test_percents=train_val_test_percents,
        num_examples_per_domain_per_class=num_examples_per_distance_per_serial,
        x_transform_func=x_transform_func
    )

    return ds


import unittest

def numpy_to_hash(n:np.ndarray):
    return hash(n.data.tobytes())

class Test_Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.labels = ALL_SERIAL_NUMBERS
        cls.domains = ALL_DISTANCES_FEET
        cls.n = 100

        cls.TRAIN, cls.VAL, cls.TEST = get_datasets(
            serial_numbers=cls.labels,
            distances=cls.domains,
            num_examples_per_distance_per_serial=cls.n,
            # normalize_type=None,
            # train_val_test_percents=(0.7,0.15,0.15)
        )
    def test_all_x_unique(self):
        all_h = []

        for ds in (self.TRAIN, self.VAL, self.TEST):
            all_h.extend( [numpy_to_hash(x) for (x,y,u) in ds] )

        self.assertEqual(
            len(all_h),
            len(set(all_h))
        )
    
    def test_expected_lens(self):
        self.assertEqual(
            len(self.TRAIN) + len(self.VAL) + len(self.TEST),
            self.n * len(self.labels) * len(self.domains)
        )

    def test_expected_domains(self):
        all_u = []

        for ds in (self.TRAIN, self.VAL, self.TEST):
            all_u.extend( [u for (x,y,u) in ds] )
        
        self.assertEqual(
            set(all_u),
            set(self.domains)
        )


    def test_expected_labels(self):
        all_y = []

        for ds in (self.TRAIN, self.VAL, self.TEST):
            all_y.extend( [y for (x,y,u) in ds] )
        
        self.assertEqual(
            set(all_y),
            set([serial_number_to_id(y) for y in self.labels])
        )


if __name__ == "__main__":
    unittest.main()
