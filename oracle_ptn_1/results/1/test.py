#! /usr/bin/python3 

from unittest.case import expectedFailure
import numpy as np
import torch
import unittest
import copy
from easydict import EasyDict
import os
import time

from oracle_prototypical_network_driver import (
    parse_and_validate_parameters,
    set_rng,
    build_network,
    build_datasets,
    base_parameters,
    train,
    evaluate_model_and_create_experiment_summary,
    build_model,
)

from steves_utils.ORACLE.utils_v2 import (
    ALL_SERIAL_NUMBERS,
    ALL_DISTANCES_FEET,
    serial_number_to_id
)

base_parameters = {}
base_parameters["experiment_name"] = "MANUAL ORACLE PTN"
base_parameters["lr"] = 0.001
base_parameters["device"] = "cuda"

base_parameters["seed"] = 1337
base_parameters["desired_classes_source"] = ALL_SERIAL_NUMBERS
base_parameters["desired_classes_target"] = ALL_SERIAL_NUMBERS

base_parameters["source_domains"] = [38,]
base_parameters["target_domains"] = [
    20,
    44,
    2,
    8,
    56,
    62
]

base_parameters["num_examples_per_class_per_domain_source"]=100
base_parameters["num_examples_per_class_per_domain_target"]=100

base_parameters["n_shot"] = 3
base_parameters["n_way"]  = len(base_parameters["desired_classes_source"])
base_parameters["n_query"]  = 2
base_parameters["train_k_factor"] = 1
base_parameters["val_k_factor"] = 2
base_parameters["test_k_factor"] = 2


base_parameters["n_epoch"] = 3

base_parameters["patience"] = 10
base_parameters["criteria_for_best"] = "target"
base_parameters["normalize_source"] = False
base_parameters["normalize_target"] = False


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

def prep_datasets(p:EasyDict)->dict:
    """
    {
        "source": {
            "original": {"train":<data>, "val":<data>, "test":<data>}
            "processed": {"train":<data>, "val":<data>, "test":<data>}
        },
        "target": {
            "original": {"train":<data>, "val":<data>, "test":<data>}
            "processed": {"train":<data>, "val":<data>, "test":<data>}
        },
    }
    """
    torch.set_default_dtype(torch.float64)
    set_rng(p)
    datasets = build_datasets(p)

    return datasets

def numpy_to_hash(n:np.ndarray):
    return hash(n.data.tobytes())


def every_x_in_datasets(datasets, original_or_processed, unbatch:bool):
    if not unbatch:
        for a in [datasets["source"], datasets["target"]]:
            for ds in a[original_or_processed].values():
                for u, support_x, support_y, query_x, query_y, real_classes in ds:
                    yield support_x
                    yield query_x
    if unbatch:
        for a in [datasets["source"], datasets["target"]]:
            for ds in a[original_or_processed].values():
                for support_x, support_y, query_x, query_y, real_classes in ds:
                    for x in support_x:
                        yield x
                    for x in query_x:
                        yield x

class Test_Datasets(unittest.TestCase):
    # @unittest.skip
    def test_correct_domains(self):
        params = copy.deepcopy(base_parameters)
        params = EasyDict(params)

        for source, target in [
            ([50,32,8],(2,14,44)),
            ([50,32,62],(2,14,44)),
            (ALL_DISTANCES_FEET,(2,14,44)),
        ]:

            params.source_domains = source
            params.target_domains = target

            p = parse_and_validate_parameters(params)
            datasets = prep_datasets(p)
        
            # source
            for ds in (datasets.source.original.values()):
                seen_domains = set()
                for u, ex in ds:
                    seen_domains.add(u)
                self.assertEqual(
                    seen_domains, set(params.source_domains)
                )

            # target
            for ds in (datasets.target.original.values()):
                seen_domains = set()
                for u, ex in ds:
                    seen_domains.add(u)
                self.assertEqual(
                    seen_domains, set(params.target_domains)
                )
                
    # @unittest.skip
    def test_correct_labels(self):
        params = copy.deepcopy(base_parameters)
        params = EasyDict(params)

        for source, target in [
            (ALL_SERIAL_NUMBERS, ALL_SERIAL_NUMBERS),
            (ALL_SERIAL_NUMBERS[:5], ALL_SERIAL_NUMBERS[5:]),
            (ALL_SERIAL_NUMBERS[:5], ALL_SERIAL_NUMBERS[:5]),
            (ALL_SERIAL_NUMBERS[:1], ALL_SERIAL_NUMBERS[4:5])
        ]:

            params.desired_classes_source = source
            params.desired_classes_target = target
            params.n_way                  = len(params.desired_classes_source)


            classes_as_ids_source = [params.desired_classes_source.index(y) for y in params.desired_classes_source]
            classes_as_ids_target = [params.desired_classes_target.index(y) for y in params.desired_classes_target]

            p = parse_and_validate_parameters(params)
            datasets = prep_datasets(p)
        
            # source
            for ds in (datasets.source.original.values()):
                seen_classes = set()
                for u, (support_x, support_y, query_x, query_y, real_classes) in ds:
                    for pseduo_y in torch.cat((support_y, query_y)):
                        seen_classes.add(real_classes[pseduo_y])

                self.assertEqual(
                    seen_classes, set(classes_as_ids_source)
                )

            # target
            for ds in (datasets.target.original.values()):
                seen_classes = set()
                for u, (support_x, support_y, query_x, query_y, real_classes) in ds:
                    for pseduo_y in torch.cat((support_y, query_y)):
                        seen_classes.add(real_classes[pseduo_y])
                self.assertEqual(
                    seen_classes, set(classes_as_ids_target)
                )

    # @unittest.skip
    def test_correct_example_count(self):
        params = copy.deepcopy(base_parameters)
        params = EasyDict(params)

        params.desired_classes_source = ALL_SERIAL_NUMBERS
        params.desired_classes_target = ALL_SERIAL_NUMBERS

        params.train_k_factor = 3
        params.val_k_factor   = 3
        params.test_k_factor  = 3

        for source, target in [
            (100, 100),
            (1000, 100),
            (100, 1000),
            (1000, 1000),
        ]:

            params.num_examples_per_class_per_domain_source = source
            params.num_examples_per_class_per_domain_target = target

            p = parse_and_validate_parameters(params)
            datasets = prep_datasets(p)
        
            # source
            examples_by_domain = {}
            for ds in (datasets.source.original.values()):
                
                for u, (support_x, support_y, query_x, query_y, real_classes) in ds:                   
                    for x in support_x:
                        if u not in examples_by_domain:
                            examples_by_domain[u] = set()
                        examples_by_domain[u].add(numpy_to_hash(x.numpy()))
                    for x in query_x:
                        if u not in examples_by_domain:
                            examples_by_domain[u] = set()
                        examples_by_domain[u].add(numpy_to_hash(x.numpy()))
                    
            for u, hashes in examples_by_domain.items():
                self.assertGreaterEqual(
                    len(hashes) / ( source * len(params.desired_classes_source) ),
                    0.9
                )

                self.assertLessEqual(
                    len(hashes) , ( source * len(params.desired_classes_source) )
                )


            # target
            examples_by_domain = {}
            for ds in (datasets.target.original.values()):
                
                for u, (support_x, support_y, query_x, query_y, real_classes) in ds:                   
                    for x in support_x:
                        if u not in examples_by_domain:
                            examples_by_domain[u] = set()
                        examples_by_domain[u].add(numpy_to_hash(x.numpy()))
                    for x in query_x:
                        if u not in examples_by_domain:
                            examples_by_domain[u] = set()
                        examples_by_domain[u].add(numpy_to_hash(x.numpy()))
                    
            for u, hashes in examples_by_domain.items():
                self.assertGreaterEqual(
                    len(hashes) / ( target * len(params.desired_classes_target) ),
                    0.9
                )
                self.assertLessEqual(
                    len(hashes) , ( target * len(params.desired_classes_target) )
                )

    # @unittest.skip
    def test_correct_episode_count(self):
        """
        This one is a little messy.
        We don't get the precise number of episodes expected due to dataset splits causing slight imbalances in the classes.
        This is alleviated with higher example counts per class/domain, but the problem is still present.

        On the plus side, this also validates our len methods
        """
        params = copy.deepcopy(base_parameters)
        params = EasyDict(params)

        params.desired_classes_source = ALL_SERIAL_NUMBERS
        params.desired_classes_target = ALL_SERIAL_NUMBERS

        for num_examples_per_class_per_domain_source, num_examples_per_class_per_domain_target, n_way, n_shot, n_query, k_factor  in [
            (500, 500, len(params.desired_classes_source), 2, 3, 1),
            (500, 500, len(params.desired_classes_source), 1, 1, 1),
            (500, 500, len(params.desired_classes_source), 5, 5, 1),

        ]:
            params.train_k_factor = k_factor
            params.val_k_factor   = k_factor
            params.test_k_factor  = k_factor
            params.num_examples_per_class_per_domain_source = num_examples_per_class_per_domain_source
            params.num_examples_per_class_per_domain_target = num_examples_per_class_per_domain_target
            params.n_shot = n_shot
            params.n_way = n_way
            params.n_query = n_query


            p = parse_and_validate_parameters(params)
            datasets = prep_datasets(p)
        
            # source
            expected_episodes = k_factor * num_examples_per_class_per_domain_source * len(params.desired_classes_source) * len(params.source_domains) / (n_way*(n_shot + n_query))
            len_episodes = sum([len(ds) for ds in datasets.source.original.values()])
            got_episodes = 0
            for ds in (datasets.source.original.values()):
                for u, (support_x, support_y, query_x, query_y, real_classes) in ds:                   
                    got_episodes +=1
            self.assertEqual(len_episodes, got_episodes)
            self.assertGreaterEqual(
                got_episodes / expected_episodes,
                0.8
            )

            self.assertLessEqual(
                got_episodes, expected_episodes
            )


            # target
            expected_episodes = k_factor * num_examples_per_class_per_domain_target * len(params.desired_classes_target) * len(params.target_domains) / (n_way*(n_shot + n_query))
            len_episodes = sum([len(ds) for ds in datasets.target.original.values()])
            got_episodes = 0
            for ds in (datasets.target.original.values()):
                for u, (support_x, support_y, query_x, query_y, real_classes) in ds:                   
                    got_episodes +=1
            self.assertEqual(len_episodes, got_episodes)
            self.assertGreaterEqual(
                got_episodes / expected_episodes,
                0.8
            )

            self.assertLessEqual(
                got_episodes, expected_episodes
            )

    # @unittest.skip
    def test_sets_disjoint(self):
        params = copy.deepcopy(base_parameters)
        params = EasyDict(params)

        params.desired_classes_source = ALL_SERIAL_NUMBERS
        params.desired_classes_target = ALL_SERIAL_NUMBERS

        for num_examples_per_class_per_domain_source, num_examples_per_class_per_domain_target, n_way, n_shot, n_query, k_factor  in [
            (500, 500, len(params.desired_classes_source), 2, 3, 1),

        ]:
            params.train_k_factor = k_factor
            params.val_k_factor   = k_factor
            params.test_k_factor  = k_factor
            params.num_examples_per_class_per_domain_source = num_examples_per_class_per_domain_source
            params.num_examples_per_class_per_domain_target = num_examples_per_class_per_domain_target
            params.n_shot = n_shot
            params.n_way = n_way
            params.n_query = n_query


            p = parse_and_validate_parameters(params)
            datasets = prep_datasets(p)
        
            # source
            train_hashes = set()
            for u, (support_x, support_y, query_x, query_y, real_classes) in datasets.source.original.train:                   
                for hash in [numpy_to_hash(x.numpy()) for x in support_x]: train_hashes.add(hash)
                for hash in [numpy_to_hash(x.numpy()) for x in query_x]: train_hashes.add(hash)

            val_hashes = set()
            for u, (support_x, support_y, query_x, query_y, real_classes) in datasets.source.original.val:                   
                for hash in [numpy_to_hash(x.numpy()) for x in support_x]: val_hashes.add(hash)
                for hash in [numpy_to_hash(x.numpy()) for x in query_x]: val_hashes.add(hash)

            test_hashes = set()
            for u, (support_x, support_y, query_x, query_y, real_classes) in datasets.source.original.test:                   
                for hash in [numpy_to_hash(x.numpy()) for x in support_x]: test_hashes.add(hash)
                for hash in [numpy_to_hash(x.numpy()) for x in query_x]: test_hashes.add(hash)
            

            self.assertEqual( len(train_hashes.intersection(val_hashes)),  0 )
            self.assertEqual( len(train_hashes.intersection(test_hashes)), 0 )
            self.assertEqual( len(val_hashes.intersection(test_hashes)),   0 )


            # target
            train_hashes = set()
            for u, (support_x, support_y, query_x, query_y, real_classes) in datasets.target.original.train:                   
                for hash in [numpy_to_hash(x.numpy()) for x in support_x]: train_hashes.add(hash)
                for hash in [numpy_to_hash(x.numpy()) for x in query_x]: train_hashes.add(hash)

            val_hashes = set()
            for u, (support_x, support_y, query_x, query_y, real_classes) in datasets.target.original.val:                   
                for hash in [numpy_to_hash(x.numpy()) for x in support_x]: val_hashes.add(hash)
                for hash in [numpy_to_hash(x.numpy()) for x in query_x]: val_hashes.add(hash)

            test_hashes = set()
            for u, (support_x, support_y, query_x, query_y, real_classes) in datasets.target.original.test:                   
                for hash in [numpy_to_hash(x.numpy()) for x in support_x]: test_hashes.add(hash)
                for hash in [numpy_to_hash(x.numpy()) for x in query_x]: test_hashes.add(hash)
            

            self.assertEqual( len(train_hashes.intersection(val_hashes)),  0 )
            self.assertEqual( len(train_hashes.intersection(test_hashes)), 0 )
            self.assertEqual( len(val_hashes.intersection(test_hashes)),   0 )

    # @unittest.skip
    def test_train_randomizes_episodes_val_and_test_dont(self):
        params = copy.deepcopy(base_parameters)
        params = EasyDict(params)

        params.desired_classes_source = ALL_SERIAL_NUMBERS
        params.desired_classes_target = ALL_SERIAL_NUMBERS

        NUM_ITERATIONS = 3

        for num_examples_per_class_per_domain_source, num_examples_per_class_per_domain_target, n_way, n_shot, n_query, k_factor  in [
            (100, 100, len(params.desired_classes_source), 2, 3, 1),
        ]:
            params.train_k_factor = k_factor
            params.val_k_factor   = k_factor
            params.test_k_factor  = k_factor
            params.num_examples_per_class_per_domain_source = num_examples_per_class_per_domain_source
            params.num_examples_per_class_per_domain_target = num_examples_per_class_per_domain_target
            params.n_shot = n_shot
            params.n_way = n_way
            params.n_query = n_query


            p = parse_and_validate_parameters(params)
            datasets = prep_datasets(p)
        
            train = set()
            val   = set()
            test  = set()

            for _ in range(NUM_ITERATIONS):
                # source
                train_hashes = []
                for u, (support_x, support_y, query_x, query_y, real_classes) in datasets.source.original.train:                   
                    for h in [numpy_to_hash(x.numpy()) for x in support_x]: train_hashes.append(h)
                    for h in [numpy_to_hash(x.numpy()) for x in query_x]: train_hashes.append(h)

                val_hashes = []
                for u, (support_x, support_y, query_x, query_y, real_classes) in datasets.source.original.val:                   
                    for h in [numpy_to_hash(x.numpy()) for x in support_x]: val_hashes.append(h)
                    for h in [numpy_to_hash(x.numpy()) for x in query_x]: val_hashes.append(h)

                test_hashes = []
                for u, (support_x, support_y, query_x, query_y, real_classes) in datasets.source.original.test:                   
                    for h in [numpy_to_hash(x.numpy()) for x in support_x]: test_hashes.append(h)
                    for h in [numpy_to_hash(x.numpy()) for x in query_x]: test_hashes.append(h)
                
                train.add(hash(tuple(train_hashes)))
                val.add(hash(tuple(val_hashes)))
                test.add(hash(tuple(test_hashes)))
            
            self.assertEqual(len(train), NUM_ITERATIONS)
            self.assertEqual(len(val), 1)
            self.assertEqual(len(test), 1)


            train = set()
            val   = set()
            test  = set()

            for _ in range(NUM_ITERATIONS):
                # source
                train_hashes = []
                for u, (support_x, support_y, query_x, query_y, real_classes) in datasets.target.original.train:                   
                    for h in [numpy_to_hash(x.numpy()) for x in support_x]: train_hashes.append(h)
                    for h in [numpy_to_hash(x.numpy()) for x in query_x]: train_hashes.append(h)

                val_hashes = []
                for u, (support_x, support_y, query_x, query_y, real_classes) in datasets.target.original.val:                   
                    for h in [numpy_to_hash(x.numpy()) for x in support_x]: val_hashes.append(h)
                    for h in [numpy_to_hash(x.numpy()) for x in query_x]: val_hashes.append(h)

                test_hashes = []
                for u, (support_x, support_y, query_x, query_y, real_classes) in datasets.target.original.test:                   
                    for h in [numpy_to_hash(x.numpy()) for x in support_x]: test_hashes.append(h)
                    for h in [numpy_to_hash(x.numpy()) for x in query_x]: test_hashes.append(h)
                
                train.add(hash(tuple(train_hashes)))
                val.add(hash(tuple(val_hashes)))
                test.add(hash(tuple(test_hashes)))
            
            self.assertEqual(len(train), NUM_ITERATIONS)
            self.assertEqual(len(val), 1)
            self.assertEqual(len(test), 1)
            
    #@unittest.skip 
    def test_iterator_changes_permutation(self):
        params = copy.deepcopy(base_parameters)
        params = EasyDict(params)

        params.desired_classes_source = ALL_SERIAL_NUMBERS
        params.desired_classes_target = ALL_SERIAL_NUMBERS

        source_train = set()
        source_val = set()
        source_test = set()

        target_val = set()
        target_test = set()
        target_train = set()

        combos = [
            (1337, 420),
            (54321, 420),
            (12332546, 420),
        ]

        for seed, dataset_seed  in combos:
            num_examples_per_class_per_domain_source= 100
            num_examples_per_class_per_domain_target= 100
            n_way= len(params.desired_classes_source)
            n_shot= 2
            n_query= 3
            k_factor= 100

            params.train_k_factor = k_factor
            params.val_k_factor   = k_factor
            params.test_k_factor  = k_factor
            params.num_examples_per_class_per_domain_source = num_examples_per_class_per_domain_source
            params.num_examples_per_class_per_domain_target = num_examples_per_class_per_domain_target
            params.n_shot = n_shot
            params.n_way = n_way
            params.n_query = n_query

            params.seed = seed


            p = parse_and_validate_parameters(params)
            datasets = prep_datasets(p)



            # source
            train_hashes = []
            for u, (support_x, support_y, query_x, query_y, real_classes) in datasets.source.original.train:                   
                for h in [numpy_to_hash(x.numpy()) for x in support_x]: train_hashes.append(h)
                for h in [numpy_to_hash(x.numpy()) for x in query_x]: train_hashes.append(h)

            val_hashes = []
            for u, (support_x, support_y, query_x, query_y, real_classes) in datasets.source.original.val:                   
                for h in [numpy_to_hash(x.numpy()) for x in support_x]: val_hashes.append(h)
                for h in [numpy_to_hash(x.numpy()) for x in query_x]: val_hashes.append(h)

            test_hashes = []
            for u, (support_x, support_y, query_x, query_y, real_classes) in datasets.source.original.test:                   
                for h in [numpy_to_hash(x.numpy()) for x in support_x]: test_hashes.append(h)
                for h in [numpy_to_hash(x.numpy()) for x in query_x]: test_hashes.append(h)
            
            source_train.add(hash(tuple(train_hashes)))
            source_val.add(hash(tuple(val_hashes)))
            source_test.add(hash(tuple(test_hashes)))
        

            # target
            train_hashes = []
            for u, (support_x, support_y, query_x, query_y, real_classes) in datasets.target.original.train:                   
                for h in [numpy_to_hash(x.numpy()) for x in support_x]: train_hashes.append(h)
                for h in [numpy_to_hash(x.numpy()) for x in query_x]: train_hashes.append(h)

            val_hashes = []
            for u, (support_x, support_y, query_x, query_y, real_classes) in datasets.target.original.val:                   
                for h in [numpy_to_hash(x.numpy()) for x in support_x]: val_hashes.append(h)
                for h in [numpy_to_hash(x.numpy()) for x in query_x]: val_hashes.append(h)

            test_hashes = []
            for u, (support_x, support_y, query_x, query_y, real_classes) in datasets.target.original.test:                   
                for h in [numpy_to_hash(x.numpy()) for x in support_x]: test_hashes.append(h)
                for h in [numpy_to_hash(x.numpy()) for x in query_x]: test_hashes.append(h)
            
            target_train.add(hash(tuple(train_hashes)))
            target_val.add(hash(tuple(val_hashes)))
            target_test.add(hash(tuple(test_hashes)))
            
        self.assertEqual(  len(source_train), len(combos)  )
        self.assertEqual(  len(source_val), len(combos)    )
        self.assertEqual(  len(source_test), len(combos)   )
        self.assertEqual(  len(target_val), len(combos)    )
        self.assertEqual(  len(target_test), len(combos)   )
        self.assertEqual(  len(target_train), len(combos)  )


    # @unittest.skip
    def test_reproducability(self):
        params = copy.deepcopy(base_parameters)
        params = EasyDict(params)

        params.desired_classes_source = ALL_SERIAL_NUMBERS
        params.desired_classes_target = ALL_SERIAL_NUMBERS

        NUM_ITERATIONS = 3

        params.train_k_factor = 1
        params.val_k_factor   = 1
        params.test_k_factor  = 1
        params.num_examples_per_class_per_domain_source = 100
        params.num_examples_per_class_per_domain_target = 100
        params.n_shot = 2
        params.n_way = len(params.desired_classes_source)
        params.n_query = 3

        all_hashes = set()


        for _ in range(NUM_ITERATIONS):
            p = parse_and_validate_parameters(params)
            datasets = prep_datasets(p)

            hashes = []

            # source
            for ds in datasets.source.original.values():
                for u, (support_x, support_y, query_x, query_y, real_classes) in ds:
                    hashes.append(numpy_to_hash(support_x.numpy()))
                    hashes.append(numpy_to_hash(query_x.numpy()))

            # target
            for ds in datasets.target.original.values():
                for u, (support_x, support_y, query_x, query_y, real_classes) in ds:
                    hashes.append(numpy_to_hash(support_x.numpy()))
                    hashes.append(numpy_to_hash(query_x.numpy()))

            all_hashes.add(
                hash(tuple(hashes))
            )
        
        print(all_hashes)
        self.assertEqual(len(all_hashes), 1)


    # @unittest.skip
    def test_splits(self):
        params = copy.deepcopy(base_parameters)
        params = EasyDict(params)

        params.desired_classes_source = ALL_SERIAL_NUMBERS
        params.desired_classes_target = ALL_SERIAL_NUMBERS

        for num_examples_per_class_per_domain_source, num_examples_per_class_per_domain_target, n_way, n_shot, n_query, k_factor  in [
            (500, 500, len(params.desired_classes_source), 2, 3, 3),
            (500, 500, len(params.desired_classes_source), 1, 1, 3),
            (500, 500, len(params.desired_classes_source), 5, 5, 3),

        ]:
            params.train_k_factor = k_factor
            params.val_k_factor   = k_factor
            params.test_k_factor  = k_factor
            params.num_examples_per_class_per_domain_source = num_examples_per_class_per_domain_source
            params.num_examples_per_class_per_domain_target = num_examples_per_class_per_domain_target
            params.n_shot = n_shot
            params.n_way = n_way
            params.n_query = n_query


            p = parse_and_validate_parameters(params)
            datasets = prep_datasets(p)
        


            # source
            train_hashes = set()
            for u, (support_x, support_y, query_x, query_y, real_classes) in datasets.source.original.train:                   
                for h in [numpy_to_hash(x.numpy()) for x in support_x]: train_hashes.add(h)
                for h in [numpy_to_hash(x.numpy()) for x in query_x]: train_hashes.add(h)

            val_hashes = set()
            for u, (support_x, support_y, query_x, query_y, real_classes) in datasets.source.original.val:                   
                for h in [numpy_to_hash(x.numpy()) for x in support_x]: val_hashes.add(h)
                for h in [numpy_to_hash(x.numpy()) for x in query_x]: val_hashes.add(h)

            test_hashes = set()
            for u, (support_x, support_y, query_x, query_y, real_classes) in datasets.source.original.test:                   
                for h in [numpy_to_hash(x.numpy()) for x in support_x]: test_hashes.add(h)
                for h in [numpy_to_hash(x.numpy()) for x in query_x]: test_hashes.add(h)
            
            total = len(train_hashes) + len(val_hashes) + len(test_hashes)
            self.assertAlmostEqual( len(train_hashes) / total, 0.7, places=1)
            self.assertAlmostEqual( len(val_hashes) / total, 0.15, places=1)
            self.assertAlmostEqual( len(test_hashes) / total, 0.15, places=1)


            # target
            train_hashes = set()
            for u, (support_x, support_y, query_x, query_y, real_classes) in datasets.target.original.train:                   
                for h in [numpy_to_hash(x.numpy()) for x in support_x]: train_hashes.add(h)
                for h in [numpy_to_hash(x.numpy()) for x in query_x]: train_hashes.add(h)

            val_hashes = set()
            for u, (support_x, support_y, query_x, query_y, real_classes) in datasets.target.original.val:                   
                for h in [numpy_to_hash(x.numpy()) for x in support_x]: val_hashes.add(h)
                for h in [numpy_to_hash(x.numpy()) for x in query_x]: val_hashes.add(h)

            test_hashes = set()
            for u, (support_x, support_y, query_x, query_y, real_classes) in datasets.target.original.test:                   
                for h in [numpy_to_hash(x.numpy()) for x in support_x]: test_hashes.add(h)
                for h in [numpy_to_hash(x.numpy()) for x in query_x]: test_hashes.add(h)
            
            total = len(train_hashes) + len(val_hashes) + len(test_hashes)
            self.assertAlmostEqual( len(train_hashes) / total, 0.7, places=1)
            self.assertAlmostEqual( len(val_hashes) / total, 0.15, places=1)
            self.assertAlmostEqual( len(test_hashes) / total, 0.15, places=1)

    # @unittest.skip
    def test_episode_has_no_repeats(self):
        params = copy.deepcopy(base_parameters)
        params = EasyDict(params)

        for source, target in [
            ([50,32,8],(2,14,44)),
            ([50,32,62],(2,14,44)),
            (ALL_DISTANCES_FEET,(2,14,44)),
        ]:

            params.source_domains = source
            params.target_domains = target

            p = parse_and_validate_parameters(params)
            datasets = prep_datasets(p)
        
            # source
            for ds in (datasets.source.original.values()):
                for u, (support_x, support_y, query_x, query_y, real_classes) in ds:                   
                    episode_hashes = []
                    for h in [numpy_to_hash(x.numpy()) for x in support_x]: episode_hashes.append(h)
                    for h in [numpy_to_hash(x.numpy()) for x in query_x]: episode_hashes.append(h)
                self.assertEqual( len(episode_hashes), len(set(episode_hashes)))


            # target
            for ds in (datasets.target.original.values()):
                for u, (support_x, support_y, query_x, query_y, real_classes) in ds:                   
                    episode_hashes = []
                    for h in [numpy_to_hash(x.numpy()) for x in support_x]: episode_hashes.append(h)
                    for h in [numpy_to_hash(x.numpy()) for x in query_x]: episode_hashes.append(h)
                self.assertEqual( len(episode_hashes), len(set(episode_hashes)))


    def test_normalization(self):
        params = copy.deepcopy(base_parameters)
        params = EasyDict(params)
        params.normalize_source = False
        params.normalize_target = False
        p = parse_and_validate_parameters(params)
        datasets = prep_datasets(p)

        
        non_norm_x = every_x_in_datasets(datasets, "processed", unbatch=True)
        

        for algo in ["dummy"]:
            params.normalize_source = algo
            params.normalize_target = algo
            p = parse_and_validate_parameters(params)
            datasets = prep_datasets(p)

            norm_x = every_x_in_datasets(datasets, "processed", unbatch=True)
            

            for non_norm, norm in zip(non_norm_x, norm_x):
                self.assertFalse(
                    np.array_equal( non_norm, norm)
                )

            for non_norm, norm in zip(non_norm_x, norm_x):
                self.assertTrue(
                    np.array_equal( 
                        norm(non_norm, algo), 
                        norm
                    )
                )


class Test_Reproducability(unittest.TestCase):
    def test_reproducability(self):
        params = copy.deepcopy(base_parameters)
        params = EasyDict(params)
        params.n_epoch = 3
        p = parse_and_validate_parameters(params)
        torch.set_default_dtype(torch.float64)

        # Run 1
        start_time_secs = time.time()
        set_rng(p)
        x_net = build_network(p)
        datasets = build_datasets(p)
        model, opt = build_model(p, x_net)
        jig = train(
            p,
            model=model,
            optimizer=opt,
            datasets=datasets
        )
        total_experiment_time_secs = time.time() - start_time_secs

        experiment = evaluate_model_and_create_experiment_summary(
            p,
            jig=jig,
            ds=datasets,
            total_experiment_time_secs=total_experiment_time_secs,
            model=model,
        )
        hist_1 = jig.get_history()

        # Run 2
        start_time_secs = time.time()
        set_rng(p)
        x_net = build_network(p)
        datasets = build_datasets(p)
        model, opt = build_model(p, x_net)
        jig = train(
            p,
            model=model,
            optimizer=opt,
            datasets=datasets
        )
        total_experiment_time_secs = time.time() - start_time_secs

        experiment = evaluate_model_and_create_experiment_summary(
            p,
            jig=jig,
            ds=datasets,
            total_experiment_time_secs=total_experiment_time_secs,
            model=model,
        )
        hist_2 = jig.get_history()


        self.assertEqual(
            hist_1,
            hist_2
        )

    @unittest.expectedFailure
    def test_nonreproducability(self):
        params = copy.deepcopy(base_parameters)
        params = EasyDict(params)
        params.n_epoch = 3
        p = parse_and_validate_parameters(params)

        # Run 1
        start_time_secs = time.time()
        # set_rng(p)
        x_net = build_network(p)
        datasets = build_datasets(p)
        model, opt = build_model(p, x_net)
        jig = train(
            p,
            model=model,
            optimizer=opt,
            datasets=datasets
        )
        total_experiment_time_secs = time.time() - start_time_secs

        experiment = evaluate_model_and_create_experiment_summary(
            p,
            jig=jig,
            ds=datasets,
            total_experiment_time_secs=total_experiment_time_secs,
            model=model,
        )
        hist_1 = jig.get_history()

        # Run 2
        start_time_secs = time.time()
        # set_rng(p)
        x_net = build_network(p)
        datasets = build_datasets(p)
        model, opt = build_model(p, x_net)
        jig = train(
            p,
            model=model,
            optimizer=opt,
            datasets=datasets
        )
        total_experiment_time_secs = time.time() - start_time_secs

        experiment = evaluate_model_and_create_experiment_summary(
            p,
            jig=jig,
            ds=datasets,
            total_experiment_time_secs=total_experiment_time_secs,
            model=model,
        )
        hist_2 = jig.get_history()


        self.assertEqual(
            hist_1,
            hist_2
        )

                    
        

import sys
if len(sys.argv) > 1 and sys.argv[1] == "limited":
    suite = unittest.TestSuite()
    suite.addTest(Test_Reproducability("test_reproducability"))
    suite.addTest(Test_Reproducability("test_nonreproducability"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
elif len(sys.argv) > 1:
    Test_Datasets().test_reproducability()
else:
    unittest.main()