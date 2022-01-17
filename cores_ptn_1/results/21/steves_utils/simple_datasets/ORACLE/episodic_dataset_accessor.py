#! /usr/bin/env python3
import random
import torch
import numpy as np
import pickle
import os
from steves_utils.ORACLE.utils_v2 import (
    ALL_DISTANCES_FEET,
    ALL_SERIAL_NUMBERS,
    serial_number_to_id
)
import steves_utils.simple_datasets.episodic_dataloader
from steves_utils.utils_v2 import (norm, get_datasets_base_path)
from steves_utils.simple_datasets.ORACLE.utils import genericize_stratified_dataset

def get_episodic_dataloaders(
    serial_numbers:list,
    distances:list,
    num_examples_per_distance_per_serial:int,
    iterator_seed:int,
    n_shot,
    n_way,
    n_query,
    train_val_test_k_factors,
    normalize_type:str=False,
    pickle_path:str=os.path.join(get_datasets_base_path(), "oracle.stratified_ds.2022A.pkl"),
    train_val_test_percents=(0.7,0.15,0.15)
)->tuple:
    with open(pickle_path, "rb") as f:
        stratified_ds_all = pickle.load(f)

    gsd = genericize_stratified_dataset(sds=stratified_ds_all["data"], domains=distances, labels=serial_numbers, n_per_u_per_y=num_examples_per_distance_per_serial)

    if normalize_type != False:
        x_transform_func = lambda x: norm(x, normalize_type)
    else:
        x_transform_func = False

    dataloaders = steves_utils.simple_datasets.episodic_dataloader.get_episodic_dataloaders(
        n_shot=n_shot,
        n_way=n_way,
        n_query=n_query,
        stratified_ds=gsd,
        train_val_test_percents=train_val_test_percents,
        num_examples_per_domain_per_class=num_examples_per_distance_per_serial,
        iterator_seed=iterator_seed,
        x_transform_func=x_transform_func,
        train_val_test_k_factors=train_val_test_k_factors,
    )

    return dataloaders


import unittest

def numpy_to_hash(n:np.ndarray):
    return hash(n.data.tobytes())

from steves_utils.simple_datasets.episodic_test_cases import(
    test_correct_domains,
    test_correct_labels,
    test_correct_example_count_per_domain_per_label,
    test_dls_disjoint,
    test_dls_equal,
    test_dls_notEqual,
    test_len,
    test_splits,
    test_episodes_have_no_repeats,
    test_normalization,
    test_shape,
    test_approximate_number_episodes,
    test_no_duplicates_in_dl,
    hash_episodic_dl
)

class Test_Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.desired_domains = ALL_DISTANCES_FEET
        cls.desired_labels  = ALL_SERIAL_NUMBERS
        cls.num_examples_per_domain_per_label=500
        cls.desired_seed=1337
        cls.desired_n_shot=2
        cls.desired_n_way=len(ALL_SERIAL_NUMBERS)
        cls.desired_n_query=2
        cls.desired_train_val_test_k_factors=(1,2,2)
        cls.train_val_test_percents=(0.7,0.15,0.15)

        cls.TRAIN, cls.VAL, cls.TEST = get_episodic_dataloaders(
            serial_numbers=cls.desired_labels,
            distances=cls.desired_domains,
            num_examples_per_distance_per_serial=cls.num_examples_per_domain_per_label,
            iterator_seed=cls.desired_seed,
            n_shot=cls.desired_n_shot,
            n_way=cls.desired_n_way,
            n_query=cls.desired_n_query,
            train_val_test_k_factors=cls.desired_train_val_test_k_factors
        )

        cls.ALL_DL = (cls.TRAIN, cls.VAL, cls.TEST)

        cls.generic_labels = [serial_number_to_id(y) for y in cls.desired_labels]

    def test_correct_domains(self):
        for dl in self.ALL_DL:
            test_correct_domains(self, dl, self.desired_domains)

    def test_correct_labels(self):
        for dl in self.ALL_DL:
            test_correct_labels(self, dl, self.generic_labels)


    def test_correct_example_count_per_domain_per_label(self):
        for dl,ratio in zip(self.ALL_DL[:1], self.train_val_test_percents[:1]):
            test_correct_example_count_per_domain_per_label(self, dl, int(self.num_examples_per_domain_per_label*ratio))


    def test_dls_disjoint(self):
        test_dls_disjoint(self, self.ALL_DL)

    
    def test_shape(self):
        for dl in self.ALL_DL:
            test_shape(self, dl, self.desired_n_way, self.desired_n_shot, self.desired_n_query)

    def test_repeatability(self):
        np.random.seed(self.desired_seed)
        random.seed(self.desired_seed)
        torch.manual_seed(self.desired_seed)
        TRAIN, VAL, TEST = get_episodic_dataloaders(
            serial_numbers=self.desired_labels,
            distances=self.desired_domains,
            num_examples_per_distance_per_serial=self.num_examples_per_domain_per_label,
            iterator_seed=self.desired_seed,
            n_shot=self.desired_n_shot,
            n_way=self.desired_n_way,
            n_query=self.desired_n_query,
            train_val_test_k_factors=self.desired_train_val_test_k_factors
        )
        
        NUM_ITERATIONS = 5
        first_h = []
        for dl in [TRAIN, VAL, TEST]:
            for _ in range(NUM_ITERATIONS):
                first_h.append(hash(tuple(hash_episodic_dl(dl))))


        np.random.seed(self.desired_seed)
        random.seed(self.desired_seed)
        torch.manual_seed(self.desired_seed)
        TRAIN, VAL, TEST = get_episodic_dataloaders(
            serial_numbers=self.desired_labels,
            distances=self.desired_domains,
            num_examples_per_distance_per_serial=self.num_examples_per_domain_per_label,
            iterator_seed=self.desired_seed,
            n_shot=self.desired_n_shot,
            n_way=self.desired_n_way,
            n_query=self.desired_n_query,
            train_val_test_k_factors=self.desired_train_val_test_k_factors
        )

        second_h = []
        for dl in [TRAIN, VAL, TEST]:
            for _ in range(NUM_ITERATIONS):
                second_h.append(hash(tuple(hash_episodic_dl(dl))))

        self.assertEqual(
            first_h,
            second_h
        )


    def test_seed_changes(self):
        np.random.seed(self.desired_seed)
        random.seed(self.desired_seed)
        torch.manual_seed(self.desired_seed)
        TRAIN, VAL, TEST = get_episodic_dataloaders(
            serial_numbers=self.desired_labels,
            distances=self.desired_domains,
            num_examples_per_distance_per_serial=self.num_examples_per_domain_per_label,
            iterator_seed=self.desired_seed,
            n_shot=self.desired_n_shot,
            n_way=self.desired_n_way,
            n_query=self.desired_n_query,
            train_val_test_k_factors=self.desired_train_val_test_k_factors
        )
        
        NUM_ITERATIONS = 5
        first_h = []
        for dl in [TRAIN, VAL, TEST]:
            for _ in range(NUM_ITERATIONS):
                first_h.append(hash(tuple(hash_episodic_dl(dl))))


        np.random.seed(self.desired_seed)
        random.seed(self.desired_seed)
        torch.manual_seed(self.desired_seed)
        TRAIN, VAL, TEST = get_episodic_dataloaders(
            serial_numbers=self.desired_labels,
            distances=self.desired_domains,
            num_examples_per_distance_per_serial=self.num_examples_per_domain_per_label,
            iterator_seed=self.desired_seed+1,
            n_shot=self.desired_n_shot,
            n_way=self.desired_n_way,
            n_query=self.desired_n_query,
            train_val_test_k_factors=self.desired_train_val_test_k_factors
        )

        second_h = []
        for dl in [TRAIN, VAL, TEST]:
            for _ in range(NUM_ITERATIONS):
                second_h.append(hash(tuple(hash_episodic_dl(dl))))

        self.assertNotEqual(
            first_h,
            second_h
        )

    def test_train_changes_between_iterations(self):
        test_dls_notEqual(self, self.TRAIN, self.TRAIN)

    def test_val_and_test_dont_change_between_iterations(self):
        test_dls_equal(self, self.VAL, self.VAL)
        test_dls_equal(self, self.TEST, self.TEST)

    def test_approximate_number_episodes(self):
        for i,dl in enumerate(self.ALL_DL):
            if i == 0: continue
            test_approximate_number_episodes(
                self,
                dl,
                self.desired_train_val_test_k_factors[i],
                int(self.num_examples_per_domain_per_label*self.train_val_test_percents[i]),
                len(self.desired_labels),
                len(self.desired_domains),
                self.desired_n_way,
                self.desired_n_shot,
                self.desired_n_query,
            )
    
    def test_len(self):
        for dl in self.ALL_DL:
            test_len(self, dl)

    def test_episodes_have_no_repeats(self):
        for i,dl in enumerate(self.ALL_DL):
            test_episodes_have_no_repeats(self, dl)
    
    def test_splits(self):
        test_splits(self, self.ALL_DL, self.train_val_test_percents, self.desired_train_val_test_k_factors)

    def test_no_duplicates_in_dl(self):
        for i,dl in enumerate(self.ALL_DL):
            test_no_duplicates_in_dl(self, dl)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "limited":
        suite = unittest.TestSuite()

        # suite.addTest(Test_Dataset("test_approximate_number_episodes"))
        # suite.addTest(Test_Dataset("test_correct_example_count_per_domain_per_label"))
        suite.addTest(Test_Dataset("test_val_and_test_dont_change_between_iterations"))
        
        # suite.addTest(Test_Dataset("test_no_duplicates_in_dl"))

        runner = unittest.TextTestRunner()
        runner.run(suite)
    elif len(sys.argv) > 1:
        Test_Dataset().test_reproducability()
    else:
        unittest.main()