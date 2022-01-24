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

num_examples_per_device = 7500
n_train_tasks = 200
n_val_tasks = 100
n_test_tasks = 10
max_cache_items = int(4.5e6)
desired_serial_numbers=ALL_SERIAL_NUMBERS[:5]
desired_distances=ALL_DISTANCES_FEET[:5]
desired_runs=[1]
n_way=len(ALL_SERIAL_NUMBERS[:5])
n_shot=2
n_query=2
window_length=256
window_stride=50
seed=1337


n_train_tasks_per_distance=int(n_train_tasks/len(desired_distances))
n_val_tasks_per_distance=int(n_val_tasks/len(desired_distances))
n_test_tasks_per_distance=int(n_test_tasks/len(desired_distances))
max_cache_size_per_distance=int(max_cache_items/len(desired_distances))
num_examples_per_device_per_distance=int(num_examples_per_device/len(desired_distances))

def iq_to_hash(iq):
    if isinstance(iq, torch.Tensor):
        return hash(iq.numpy().data.tobytes())
    else:
        return hash(iq.data.tobytes())

def hash_batch_examples(batch):
    return [iq_to_hash(iq) for iq in batch]
    


class test_oracle_episodic(unittest.TestCase):
    @classmethod
    def setUpClass(self) -> None:
        print("Building reusable dataset")
        self.train_dl, self.val_dl, self.test_dl = build_ORACLE_episodic_iterable(
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
        print("Done building dataset")

    def test_size(self):
        expected_n_train_tasks = n_train_tasks_per_distance * len(desired_distances)
        expected_n_val_tasks   = n_val_tasks_per_distance * len(desired_distances)
        expected_n_test_tasks  = n_test_tasks_per_distance * len(desired_distances)

        n = 0
        for _ in self.train_dl: n += 1
        self.assertEqual(n, expected_n_train_tasks)

        n = 0
        for _ in self.val_dl: n += 1
        self.assertEqual(n, expected_n_val_tasks)

        n = 0
        for _ in self.test_dl: n += 1
        self.assertEqual(n, expected_n_test_tasks)

    def test_shape(self):
        expected = {
            "support_x": torch.Size((n_way * n_shot, 2, 128)),
            "support_y": torch.Size((n_way * n_shot,)),
            "query_x": torch.Size((n_way * n_query, 2, 128)),
            "query_y": torch.Size((n_way * n_query,)),
            # "ds_class_ids": () # This is just a list of the classes that are in this episode (IE its like set(query_y))
        }

        for ex in self.train_dl:
            self.assertEqual(  ex[1][0].shape, expected["support_x"]  )
            self.assertEqual(  ex[1][1].shape, expected["support_y"]  )
            self.assertEqual(  ex[1][2].shape, expected["query_x"]  )
            self.assertEqual(  ex[1][3].shape, expected["query_y"]  )
    
    def test_sets_disjoint(self):
        train_hashes = set()
        val_hashes = set()
        test_hashes = set()

        for k in self.train_dl:
            train_hashes.update( hash_batch_examples(k[1][0]) )
            train_hashes.update( hash_batch_examples(k[1][2]) )

        for k in self.val_dl:
            val_hashes.update( hash_batch_examples(k[1][0]) )
            val_hashes.update( hash_batch_examples(k[1][2]) )

        for k in self.test_dl:
            test_hashes.update( hash_batch_examples(k[1][0]) )
            test_hashes.update( hash_batch_examples(k[1][2]) )


        self.assertEqual(
            len(train_hashes.intersection(val_hashes)),
            0
        )

        self.assertEqual(
            len(train_hashes.intersection(test_hashes)),
            0
        )

        self.assertEqual(
            len(val_hashes.intersection(test_hashes)),
            0
        )

    def test_train_randomized(self):
        NUM_ITERATIONS = 10

        first_elements = []

        for i in range(NUM_ITERATIONS):
            k = next(iter(self.train_dl))
            first_element = tuple(
                hash_batch_examples(k[1][0]) +
                hash_batch_examples(k[1][2])
            )

            first_elements.append(first_element)

        self.assertEqual(
            len(first_elements),
            len(set(first_elements))
        )

    def test_val_not_randomized(self):
        NUM_ITERATIONS = 10

        first_elements = []

        for i in range(NUM_ITERATIONS):
            k = next(iter(self.val_dl))
            first_element = tuple(
                hash_batch_examples(k[1][0]) +
                hash_batch_examples(k[1][2])
            )

            first_elements.append(first_element)

        self.assertEqual(
            len(set(first_elements)),
            1
        )
        
    def test_test_not_randomized(self):
        NUM_ITERATIONS = 10

        first_elements = []

        for i in range(NUM_ITERATIONS):
            k = next(iter(self.val_dl))
            first_element = tuple(
                hash_batch_examples(k[1][0]) +
                hash_batch_examples(k[1][2])
            )

            first_elements.append(first_element)

        self.assertEqual(
            len(set(first_elements)),
            1
        )

    def test_data_exists_in_oracle_sequence(self):
        """
        Verify that the binary numpy of each example in each episode
        actually exists in low level oracle dataset accessor.

        This works because the ORACLE_Torch_Dataset which backs the episodic iterators
        is deterministic based on the supplied seed
        """
        distance_hashes = {}
        for distance in desired_distances:
            distance_hashes[distance] = set()
            ds = ORACLE_Torch_Dataset(
                            desired_serial_numbers=desired_serial_numbers,
                            desired_distances=[distance],
                            desired_runs=desired_runs,
                            window_length=window_length,
                            window_stride=window_stride,
                            num_examples_per_device=num_examples_per_device_per_distance,
                            seed=seed,  
                            max_cache_size=0,
                            # transform_func=lambda x: (x["iq"], serial_number_to_id(x["serial_number"]), x["distance_ft"]),
                            transform_func=lambda x: (torch.from_numpy(x["iq"]), serial_number_to_id(x["serial_number"]), ), # Just (x,y)
                            prime_cache=False
            )

            for k in ds:
                distance_hashes[distance].add(iq_to_hash(k[0]))
    
        for dl in [self.train_dl, self.val_dl, self.test_dl]:
            for k in dl:
                distance = k[0]
                hashes = set(hash_batch_examples(k[1][0]) + hash_batch_examples(k[1][2]))

                self.assertEqual(
                    len(hashes),
                    len(distance_hashes[distance].intersection(hashes))
                )
    
    def test_determinism_1(self):
        first_hashes = []
        first_datasets = build_ORACLE_episodic_iterable(
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

        for ds in first_datasets:
            for k in ds:
                first_hashes.extend(
                    hash_batch_examples(k[1][0]) +
                    hash_batch_examples(k[1][2])
                )


        second_hashes = []
        second_datasets = build_ORACLE_episodic_iterable(
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

        for ds in second_datasets:
            for k in ds:
                second_hashes.extend(
                    hash_batch_examples(k[1][0]) +
                    hash_batch_examples(k[1][2])
                )

        for z in zip(first_hashes, second_hashes):
            self.assertEqual(
                z[0], z[1]
            )

    def test_determinism_2(self):
        first_hashes = []
        first_datasets = build_ORACLE_episodic_iterable(
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

        for ds in first_datasets:
            for k in ds:
                first_hashes.extend(
                    hash_batch_examples(k[1][0]) +
                    hash_batch_examples(k[1][2])
                )


        second_hashes = []
        second_datasets = build_ORACLE_episodic_iterable(
            desired_serial_numbers=desired_serial_numbers,
            # desired_distances=[50],
            desired_distances=desired_distances,
            desired_runs=desired_runs,
            window_length=window_length,
            window_stride=window_stride,
            num_examples_per_device_per_distance=num_examples_per_device_per_distance,
            seed=seed + 420,
            max_cache_size_per_distance=max_cache_size_per_distance,
            # n_way=len(ALL_SERIAL_NUMBERS),
            n_way=n_way,
            n_shot=n_shot,
            n_query=n_query,
            n_train_tasks_per_distance=n_train_tasks_per_distance,
            n_val_tasks_per_distance=n_val_tasks_per_distance,
            n_test_tasks_per_distance=n_test_tasks_per_distance,
        )

        for ds in second_datasets:
            for k in ds:
                second_hashes.extend(
                    hash_batch_examples(k[1][0]) +
                    hash_batch_examples(k[1][2])
                )

        for z in zip(first_hashes, second_hashes):
            self.assertNotEqual(
                z[0], z[1]
            )

    def test_each_example_in_episode_is_unique(self):
        for ds in [self.train_dl, self.val_dl, self.test_dl]:
            for k in ds:
                support_hashes = hash_batch_examples(k[1][0])
                query_hashes = hash_batch_examples(k[1][2])

                self.assertEqual(
                    len(set(support_hashes + query_hashes)),
                    len(support_hashes + query_hashes)
                )

    def test_got_expected_domains(self):
        expected_domains = set(desired_distances)
        seen_domains = set()
        for ds in [self.train_dl, self.val_dl, self.test_dl]:
            for k in ds:
                seen_domains.add(k[0])

        self.assertEqual(
            seen_domains, expected_domains
        )
                

unittest.main()

# Use this to test inter-run determinism
train_dl, val_dl, test_dl = build_ORACLE_episodic_iterable(
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

for ds in [train_dl, val_dl, test_dl]:
    for k in ds:
        support_hashes = hash_batch_examples(k[1][0])
        query_hashes = hash_batch_examples(k[1][2])

        for h in support_hashes+query_hashes: print(h)