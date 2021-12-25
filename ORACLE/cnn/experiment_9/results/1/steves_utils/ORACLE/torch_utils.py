#! /usr/bin/env python3

import math
import torch
import gc
import sys


from steves_utils.ORACLE.ORACLE_sequence import ORACLE_Sequence


from steves_utils.lazy_iterable_wrapper import Lazy_Iterable_Wrapper


from steves_utils.iterable_aggregator import Iterable_Aggregator
from steves_utils.ORACLE.utils_v2 import ALL_DISTANCES_FEET, ALL_RUNS, ALL_SERIAL_NUMBERS, serial_number_to_id
from steves_utils.fsl_utils import split_ds_into_episodes

class ORACLE_Torch_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        desired_serial_numbers,
        desired_runs,
        desired_distances,
        window_length,
        window_stride,
        num_examples_per_device,
        seed,
        max_cache_size=1e6,
        transform_func=None,
        prime_cache=False,
    ) -> None:
        super().__init__()

        self.os = ORACLE_Sequence(
            desired_serial_numbers,
            desired_runs,
            desired_distances,
            window_length,
            window_stride,
            num_examples_per_device,
            seed,
            max_cache_size,
            prime_cache=prime_cache
        )

        self.transform_func = transform_func

    def __len__(self):
        return len(self.os)
    
    def __getitem__(self, idx):
        if self.transform_func != None:
            return self.transform_func(self.os[idx])
        else:
            return self.os[idx]
    
def split_dataset_by_percentage(train:float, val:float, test:float, dataset, seed:int):
    assert train < 1.0
    assert val < 1.0
    assert test < 1.0
    assert train + val + test <= 1.0

    num_train = math.floor(len(dataset) * train)
    num_val   = math.floor(len(dataset) * val)
    num_test  = math.floor(len(dataset) * test)

    return torch.utils.data.random_split(dataset, (num_train, num_val, num_test), generator=torch.Generator().manual_seed(seed))



def build_ORACLE_episodic_iterable(
    desired_serial_numbers,
    desired_distances,
    desired_runs,
    window_length,
    window_stride,
    num_examples_per_device_per_distance,
    seed,
    max_cache_size_per_distance,
    n_way,
    n_shot,
    n_query,
    n_train_tasks_per_distance,
    n_val_tasks_per_distance,
    n_test_tasks_per_distance,
):
    """
    Each distance gets segregated such that an episode only consists of examples from the same distance
    """

    all_train = []
    all_val = []
    all_test = []

    for distance in desired_distances:
        print("Begin priming Cache for distance", distance)
        sys.stdout.flush()
        ds = ORACLE_Torch_Dataset(
                        desired_serial_numbers=desired_serial_numbers,
                        desired_distances=[distance],
                        desired_runs=desired_runs,
                        window_length=window_length,
                        window_stride=window_stride,
                        num_examples_per_device=num_examples_per_device_per_distance,
                        seed=seed,  
                        max_cache_size=max_cache_size_per_distance,
                        # transform_func=lambda x: (x["iq"], serial_number_to_id(x["serial_number"]), x["distance_ft"]),
                        transform_func=lambda x: (torch.from_numpy(x["iq"]), serial_number_to_id(x["serial_number"]), ), # Just (x,y)
                        prime_cache=True
        )
        print("Done priming Cache for distance", distance)

        labels = list(map(lambda k: serial_number_to_id(k["serial_number"]), ds.os.metadata))

        # del ds.os.metadata
        # gc.collect()


        train, val, test = split_ds_into_episodes(
            ds=ds,
            labels=labels,
            n_way=n_way,
            n_shot=n_shot,
            n_query=n_query,
            n_train_tasks=n_train_tasks_per_distance,
            n_val_tasks=n_val_tasks_per_distance,
            n_test_tasks=n_test_tasks_per_distance,
            seed=seed,
        )

        # We lazy_map the dataloaders such that they will return <domain, episode>
        # Note the bizarre distance=distance thing! This is because lambdas are actually a little shitty in that
        # they still use references, and if we are referencing an iterator value, then guess what, that value
        # is changing underneath us
        lam = lambda episode, distance=distance: (distance, episode)
        train = Lazy_Iterable_Wrapper(train, lam)
        val   = Lazy_Iterable_Wrapper(val, lam)
        test  = Lazy_Iterable_Wrapper(test, lam)

        # stick em on the pile
        all_train.append(train)
        all_val.append(val)
        all_test.append(test)

    
    # The final piece. We aggregate each of the iterables and randomize the order that their next()'s are returned
    # This is important, since we want each episode to be separated by distance, but we want the order they are
    # trained on to be randomized
    #
    # Note that val and test dont get randomized, only aggregated
    
    return (
        Iterable_Aggregator(all_train, randomizer_seed=seed),
        Iterable_Aggregator(all_val, randomizer_seed=None),
        Iterable_Aggregator(all_test, randomizer_seed=None),
    )


if __name__ == "__main__":
    import unittest
    import warnings

    class test_build_ORACLE_episodic_iterable(unittest.TestCase):
        # Necessary to cover up unclosed file errors from the very low level numpy memmap shit
        # def setUp(self):
        #     warnings.simplefilter("ignore", ResourceWarning)

        # @unittest.skip("Skipping for sake of time")
        def test_init(self):
            train, val, test = build_ORACLE_episodic_iterable(
                desired_serial_numbers=ALL_SERIAL_NUMBERS,
                # desired_distances=ALL_DISTANCES_FEET[:2],
                desired_distances=[2,8],
                desired_runs=ALL_RUNS,
                window_length=128,
                window_stride=50,
                num_examples_per_device_per_distance=200,
                seed=1337,
                max_cache_size_per_distance=1e9,
                n_way=len(ALL_SERIAL_NUMBERS),
                n_shot=2,
                n_query=5,
                n_train_tasks_per_distance=100,
                n_val_tasks_per_distance=5,
                n_test_tasks_per_distance=2,
            )

        # @unittest.skip("Skipping for sake of time")
        def test_max_file_handles(self):
            N_TRAIN_TASKS_PER_DISTANCE = 100
            N_VAL_TASKS_PER_DISTANCE = 5
            N_TEST_TASKS_PER_DISTANCE = 2
            DESIRED_DISTANCES = ALL_DISTANCES_FEET

            import os
            train, val, test = build_ORACLE_episodic_iterable(
                desired_serial_numbers=ALL_SERIAL_NUMBERS,
                desired_distances=ALL_DISTANCES_FEET,
                desired_runs=ALL_RUNS,
                window_length=128,
                window_stride=50,
                num_examples_per_device_per_distance=200,
                seed=1337,
                max_cache_size_per_distance=1e9,
                n_way=len(ALL_SERIAL_NUMBERS),
                n_shot=2,
                n_query=5,
                n_train_tasks_per_distance=N_TRAIN_TASKS_PER_DISTANCE,
                n_val_tasks_per_distance=N_VAL_TASKS_PER_DISTANCE,
                n_test_tasks_per_distance=N_TEST_TASKS_PER_DISTANCE,
            )

            for x in train:
                pass

        def test_num_tasks(self):
            N_TRAIN_TASKS_PER_DISTANCE = 100
            N_VAL_TASKS_PER_DISTANCE = 5
            N_TEST_TASKS_PER_DISTANCE = 2
            DESIRED_DISTANCES = [2,8]

            train, val, test = build_ORACLE_episodic_iterable(
                desired_serial_numbers=ALL_SERIAL_NUMBERS,
                desired_distances=DESIRED_DISTANCES,
                desired_runs=ALL_RUNS,
                window_length=128,
                window_stride=50,
                num_examples_per_device_per_distance=200,
                seed=1337,
                max_cache_size_per_distance=1e9,
                n_way=len(ALL_SERIAL_NUMBERS),
                n_shot=2,
                n_query=5,
                n_train_tasks_per_distance=N_TRAIN_TASKS_PER_DISTANCE,
                n_val_tasks_per_distance=N_VAL_TASKS_PER_DISTANCE,
                n_test_tasks_per_distance=N_TEST_TASKS_PER_DISTANCE,
            )

            l = 0
            for x in train:
                l += 1
            self.assertEqual(l, N_TRAIN_TASKS_PER_DISTANCE*len(DESIRED_DISTANCES))

            l = 0
            for x in val:
                l += 1
            self.assertEqual(l, N_VAL_TASKS_PER_DISTANCE*len(DESIRED_DISTANCES))

            l = 0
            for x in test:
                l += 1
            self.assertEqual(l, N_TEST_TASKS_PER_DISTANCE*len(DESIRED_DISTANCES))

        # @unittest.skip("Skipping for sake of time")
        def test_domains_are_correct(self):
            N_TRAIN_TASKS_PER_DISTANCE = 100
            N_VAL_TASKS_PER_DISTANCE = 5
            N_TEST_TASKS_PER_DISTANCE = 2
            DESIRED_DISTANCES = [2,8]
            train, val, test = build_ORACLE_episodic_iterable(
                desired_serial_numbers=ALL_SERIAL_NUMBERS,
                # desired_distances=ALL_DISTANCES_FEET[:2],
                desired_distances=DESIRED_DISTANCES,
                desired_runs=ALL_RUNS,
                window_length=128,
                window_stride=50,
                num_examples_per_device_per_distance=200,
                seed=1337,
                max_cache_size_per_distance=1e9,
                n_way=len(ALL_SERIAL_NUMBERS),
                n_shot=2,
                n_query=5,
                n_train_tasks_per_distance=N_TRAIN_TASKS_PER_DISTANCE,
                n_val_tasks_per_distance=N_VAL_TASKS_PER_DISTANCE,
                n_test_tasks_per_distance=N_TEST_TASKS_PER_DISTANCE,
            )

            self.assertEqual(
                set(map(lambda k: k[0], train)),
                set(DESIRED_DISTANCES)
            )
            self.assertEqual(
                set(map(lambda k: k[0], val)),
                set(DESIRED_DISTANCES)
            )
            self.assertEqual(
                set(map(lambda k: k[0], test)),
                set(DESIRED_DISTANCES)
            )

        # @unittest.skip("Skipping for sake of time")
        def test_domains_is_correct(self):
            N_TRAIN_TASKS_PER_DISTANCE = 100
            N_VAL_TASKS_PER_DISTANCE = 5
            N_TEST_TASKS_PER_DISTANCE = 2
            DESIRED_DISTANCES = [2]
            train, val, test = build_ORACLE_episodic_iterable(
                desired_serial_numbers=ALL_SERIAL_NUMBERS,
                # desired_distances=ALL_DISTANCES_FEET[:2],
                desired_distances=DESIRED_DISTANCES,
                desired_runs=ALL_RUNS,
                window_length=128,
                window_stride=50,
                num_examples_per_device_per_distance=200,
                seed=1337,
                max_cache_size_per_distance=1e9,
                n_way=len(ALL_SERIAL_NUMBERS),
                n_shot=2,
                n_query=5,
                n_train_tasks_per_distance=N_TRAIN_TASKS_PER_DISTANCE,
                n_val_tasks_per_distance=N_VAL_TASKS_PER_DISTANCE,
                n_test_tasks_per_distance=N_TEST_TASKS_PER_DISTANCE,
            )

            self.assertEqual(
                set(map(lambda k: k[0], train)),
                set(DESIRED_DISTANCES)
            )
            self.assertEqual(
                set(map(lambda k: k[0], val)),
                set(DESIRED_DISTANCES)
            )
            self.assertEqual(
                set(map(lambda k: k[0], test)),
                set(DESIRED_DISTANCES)
            )

        # @unittest.skip("Skipping for sake of time")
        def test_train_is_randomized(self):
            N_TRAIN_TASKS_PER_DISTANCE = 100
            N_VAL_TASKS_PER_DISTANCE = 5
            N_TEST_TASKS_PER_DISTANCE = 2
            DESIRED_DISTANCES = ALL_DISTANCES_FEET

            train, val, test = build_ORACLE_episodic_iterable(
                desired_serial_numbers=ALL_SERIAL_NUMBERS,
                # desired_distances=ALL_DISTANCES_FEET[:2],
                desired_distances=DESIRED_DISTANCES,
                desired_runs=ALL_RUNS,
                window_length=128,
                window_stride=50,
                num_examples_per_device_per_distance=200,
                seed=1337,
                max_cache_size_per_distance=1e9,
                n_way=len(ALL_SERIAL_NUMBERS),
                n_shot=2,
                n_query=5,
                n_train_tasks_per_distance=N_TRAIN_TASKS_PER_DISTANCE,
                n_val_tasks_per_distance=N_VAL_TASKS_PER_DISTANCE,
                n_test_tasks_per_distance=N_TEST_TASKS_PER_DISTANCE,
            )

            for k in train: pass
            # for k in train: pass

        # @unittest.skip("Skipping for sake of time")
        def test_n_tasks(self):
            N_TRAIN_TASKS_PER_DISTANCE = 100
            N_VAL_TASKS_PER_DISTANCE = 5
            N_TEST_TASKS_PER_DISTANCE = 2
            DESIRED_DISTANCES = [2,8]

            train, val, test = build_ORACLE_episodic_iterable(
                desired_serial_numbers=ALL_SERIAL_NUMBERS,
                # desired_distances=ALL_DISTANCES_FEET[:2],
                desired_distances=DESIRED_DISTANCES,
                desired_runs=ALL_RUNS,
                window_length=128,
                window_stride=50,
                num_examples_per_device_per_distance=200,
                seed=1337,
                max_cache_size_per_distance=1e9,
                n_way=len(ALL_SERIAL_NUMBERS),
                n_shot=2,
                n_query=5,
                n_train_tasks_per_distance=N_TRAIN_TASKS_PER_DISTANCE,
                n_val_tasks_per_distance=N_VAL_TASKS_PER_DISTANCE,
                n_test_tasks_per_distance=N_TEST_TASKS_PER_DISTANCE,
            )

            self.assertEqual(
                N_TRAIN_TASKS_PER_DISTANCE * len(DESIRED_DISTANCES),
                len(train)
            )

            self.assertEqual(
                N_VAL_TASKS_PER_DISTANCE * len(DESIRED_DISTANCES),
                len(val)
            )

            self.assertEqual(
                N_TEST_TASKS_PER_DISTANCE * len(DESIRED_DISTANCES),
                len(test)
            )



            self.assertEqual(
                N_TRAIN_TASKS_PER_DISTANCE * len(DESIRED_DISTANCES),
                len(train)
            )

            self.assertEqual(
                N_VAL_TASKS_PER_DISTANCE * len(DESIRED_DISTANCES),
                len(val)
            )

            self.assertEqual(
                N_TEST_TASKS_PER_DISTANCE * len(DESIRED_DISTANCES),
                len(test)
            )
        
    unittest.main()