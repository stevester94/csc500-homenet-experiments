#! /usr/bin/env python3

import enum
import random

import itertools
import numpy as np
from numpy.core.numeric import indices

# import steves_utils.utils_v2 as steves_utils_v2

from steves_utils.ORACLE.utils_v2 import (
    ALL_DISTANCES_FEET,
    ALL_RUNS,
    ALL_SERIAL_NUMBERS,
    filter_paths,
    get_oracle_dataset_path,
    get_oracle_data_files_based_on_criteria
)

from steves_utils import file_as_windowed_list
from steves_utils import lazy_map
from steves_utils import sequence_aggregator
from steves_utils import sequence_mask
from steves_utils import sequence_cache

class ORACLE_Sequence:
    def __init__(
        self,
        desired_serial_numbers,
        desired_runs,
        desired_distances,
        window_length,
        window_stride,
        num_examples_per_device_per_distance_per_run,
        seed,
        max_cache_size=1e6, # IDK
        prime_cache=False,
        return_IQ_as_tuple_with_offset=False, # Used for debugging
    ) -> None:
        self.rng = np.random.default_rng(seed)

        # We're going to group based on device so that we can aggregate them and then pull 'num_examples_per_device' from the aggregate.
        # As part of the quest to speed up FSL episode generation, we build the metadata as a datastructure
        # parallel to the actual IQ. That is why you'll see a lot of duplicated actions
        devices = {}
        for serial in desired_serial_numbers:
            devices[serial] = {"IQ": [], "metadata": []}


        
        """
        Get the data file path
        windowize the data file
        attach metadata to that windowed view of the datafile
        Group the metadata'd windowed view by device serial number
        Select a random subset from each serial_number X run X distance
        """
        for serial_number, run, distance in set(itertools.product(desired_serial_numbers, desired_runs, desired_distances)):
            path = ORACLE_Sequence._get_a_data_file_path(serial_number, run, distance)

            windowed_sequence = ORACLE_Sequence._windowize_data_file_and_reshape(path, window_length, window_stride, return_IQ_as_tuple_with_offset)
            metadata = ORACLE_Sequence._apply_metadata(range(len(windowed_sequence)), serial_number,run,distance) # what a hack

            mask = self.rng.choice(len(windowed_sequence), size=num_examples_per_device_per_distance_per_run, replace=False)

            windowed_sequence = sequence_mask.Sequence_Mask(windowed_sequence, mask)
            metadata = sequence_mask.Sequence_Mask(metadata, mask)

            devices[serial_number]["IQ"].append(windowed_sequence)
            devices[serial_number]["metadata"].append(metadata)
        
        for d in devices.values():
            assert(
                len(d["IQ"]) == len(d["metadata"])
            )

        """
        Aggregate them based on device
        """
        masked_devices_iq  = []
        masked_devices_metadata  = []
        for device in devices.values():
            aggregated_device_iq_sequence = sequence_aggregator.Sequence_Aggregator(device["IQ"])
            aggregated_device_metadata_sequence = sequence_aggregator.Sequence_Aggregator(device["metadata"])

            assert(len(aggregated_device_iq_sequence) == len(aggregated_device_metadata_sequence))

            masked_devices_iq.append(aggregated_device_iq_sequence)        
            masked_devices_metadata.append(aggregated_device_metadata_sequence)     

        assert(len(masked_devices_iq) == len(masked_devices_metadata))

        """
        Final step!
        We aggregate all the devices
        Randomize across the entire aggregate (We use every example)
        Wrap this in a sequence cache
        """
        aggregated_devices_iq = sequence_aggregator.Sequence_Aggregator(masked_devices_iq)
        aggregated_devices_metadata = sequence_aggregator.Sequence_Aggregator(masked_devices_metadata)

        mask = self.rng.choice(len(aggregated_devices_iq), size=len(aggregated_devices_iq), replace=False)

        masked_devices_iq = sequence_mask.Sequence_Mask(aggregated_devices_iq, mask)
        masked_devices_metadata = sequence_mask.Sequence_Mask(aggregated_devices_metadata, mask)

        # OK, so we have everything we need. We lazy map a dumb range so that we can combine up IQ and metadata
        #    based on index.
        lm = lazy_map.Lazy_Map(
            range(len(masked_devices_iq)),
            lambda x: {
                "serial_number":masked_devices_metadata[x]["serial_number"],
                "run":masked_devices_metadata[x]["run"],
                "distance_ft":masked_devices_metadata[x]["distance_ft"],
                "iq":masked_devices_iq[x]
            }
        )

        # The cache is now complete and ready to be accessed. Metadata is kept so that it can be
        # accessed separately
        self.cache = sequence_cache.Sequence_Cache(lm, max_cache_size)
        self.metadata = masked_devices_metadata



        if prime_cache:
            sorted_mask = list(enumerate(mask))
            sorted_mask.sort(key=lambda x: x[1])
            
            indexes = [x[0] for x in sorted_mask][:max_cache_size]

            for i in indexes:
                _ = self[i]
    
    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        return self.cache[idx]

    # The iterable could probably just be iter(self.cache)
    def __iter__(self):
        self.iter_idx = 0
        return self
    
    def __next__(self):
        if self.iter_idx >= len(self):
            raise StopIteration
        else:
            result = self[self.iter_idx]
            self.iter_idx += 1
            return result
    
    @staticmethod
    def _get_a_data_file_path(desired_serial_number, desired_run, desired_distance):
        paths = get_oracle_data_files_based_on_criteria(
            desired_serial_numbers=[desired_serial_number],
            desired_runs=[desired_run],
            desired_distances=[desired_distance],
        )
        assert len(paths) == 1
        path = paths[0]

        return path
    
    @staticmethod
    def _windowize_data_file_and_reshape(path, window_length, window_stride, return_IQ_as_tuple_with_offset):
        faws = file_as_windowed_list.File_As_Windowed_Sequence(
            path=path,
            window_length=window_length,
            stride=window_stride,
            numpy_dtype=np.double,
            return_as_tuple_with_offset=return_IQ_as_tuple_with_offset
        )

        # print("WARNING WE ARE NOT RESHAPING")
        if return_IQ_as_tuple_with_offset:
            lm = lazy_map.Lazy_Map(
                faws,
                lambda x: (x[0], x[1].reshape((2,int(len(x[1])/2)), order="F"))
                # lambda x: x
            )
        else:
            lm = lazy_map.Lazy_Map(
                faws,
                lambda x: x.reshape((2,int(len(x)/2)), order="F")
                # lambda x: x
            )



        return lm
    
    @staticmethod
    def _apply_metadata(sequence, serial_number, run, distance):
        lm = lazy_map.Lazy_Map(
            sequence,
            lambda x: {"serial_number":serial_number, "run":run, "distance_ft":distance, "iq":None}
        )

        return lm





if __name__ == "__main__":
    import time
    import unittest

    ALL_RUNS = [1]

    class Test_ORACLE_Sequence(unittest.TestCase):
        def check_oracle_elements_equivalent(self, x,y):
            all_true = True
            all_true = all_true and (x["distance_ft"] == y["distance_ft"])
            all_true = all_true and (x["run"] == y["run"])
            all_true = all_true and (x["serial_number"] == y["serial_number"])

            # Handle the case if we are getting IQ tuple with offset
            # if len(x["iq"]) == 2 and len(y["iq"]) == 2:
            #     all_true = all_true and np.array_equal(x["iq"][1], y["iq"][1])
            #     all_true = all_true and (x["iq"][0] == y["iq"][0])
            
            # if len(x["iq"]) != len(y["iq"]):
            #     raise Exception("IQ lengths are not equal")

            # else:
            all_true = all_true and np.array_equal(x["iq"], y["iq"])

            return all_true
        
        def check_oracle_elements_equivalent(self, x,y):
            all_true = True
            all_true = all_true and (x["distance_ft"] == y["distance_ft"])
            all_true = all_true and (x["run"] == y["run"])
            all_true = all_true and (x["serial_number"] == y["serial_number"])
            all_true = all_true and np.array_equal(x["iq"], y["iq"])

            return all_true

        # @unittest.skip("Skipping for sake of time")
        def test_accessing_same_index_gives_same_values(self):
            oseq = ORACLE_Sequence(
                desired_serial_numbers=ALL_SERIAL_NUMBERS,
                desired_distances=ALL_DISTANCES_FEET,
                desired_runs=ALL_RUNS,
                window_length=256,
                # window_length=24,
                window_stride=1,
                num_examples_per_device=1000,
                seed=1337,
                max_cache_size=100000*16,
                return_IQ_as_tuple_with_offset=False
            )

            rand_indices = np.random.default_rng(1337).integers(0, len(oseq), len(oseq))

            all_items = {}

            for i in rand_indices:
                all_items[i] = oseq[i]
            
            for i in rand_indices:            
                self.assertTrue(self.check_oracle_elements_equivalent(all_items[i], oseq[i]))


        # @unittest.skip("Skipping for sake of time")
        def test_iteration(self):
            oseq = ORACLE_Sequence(
                desired_serial_numbers=ALL_SERIAL_NUMBERS,
                desired_distances=ALL_DISTANCES_FEET,
                desired_runs=ALL_RUNS,
                window_length=256,
                window_stride=1,
                num_examples_per_device=1000,
                seed=1337,  
                max_cache_size=100000*16,
                return_IQ_as_tuple_with_offset=False
            )

            for i,x in enumerate(oseq):
                self.assertTrue(self.check_oracle_elements_equivalent(oseq[i], x))


        # @unittest.skip("Skipping for sake of time")
        def test_same_seed_same_results(self):
            all_equivalent = True

            oseq_1 = ORACLE_Sequence(
                desired_serial_numbers=ALL_SERIAL_NUMBERS,
                desired_distances=ALL_DISTANCES_FEET,
                desired_runs=ALL_RUNS,
                window_length=256,
                window_stride=1,
                num_examples_per_device=1000,
                seed=1337,  
                max_cache_size=100000*16,
                return_IQ_as_tuple_with_offset=False
            )

            oseq_2 = ORACLE_Sequence(
                desired_serial_numbers=ALL_SERIAL_NUMBERS,
                desired_distances=ALL_DISTANCES_FEET,
                desired_runs=ALL_RUNS,
                window_length=256,
                window_stride=1,
                num_examples_per_device=1000,
                seed=1337,
                max_cache_size=100000*16,
                return_IQ_as_tuple_with_offset=False
            )

            for x in zip(oseq_1, oseq_2):
                all_equivalent = all_equivalent and self.check_oracle_elements_equivalent(*x)
            
            self.assertTrue(all_equivalent)

        @unittest.expectedFailure
        # @unittest.skip("Skipping for sake of time")
        def test_different_seed_different_results(self):
            all_equivalent = True

            oseq_1 = ORACLE_Sequence(
                desired_serial_numbers=ALL_SERIAL_NUMBERS,
                desired_distances=ALL_DISTANCES_FEET,
                desired_runs=ALL_RUNS,
                window_length=256,
                window_stride=1,
                num_examples_per_device=1000,
                seed=1337,  
                max_cache_size=100000*16,
                return_IQ_as_tuple_with_offset=False
            )

            oseq_2 = ORACLE_Sequence(
                desired_serial_numbers=ALL_SERIAL_NUMBERS,
                desired_distances=ALL_DISTANCES_FEET,
                desired_runs=ALL_RUNS,
                window_length=256,
                window_stride=1,
                num_examples_per_device=1000,
                seed=1338,  
                max_cache_size=100000*16,
                return_IQ_as_tuple_with_offset=False
            )

            for x in zip(oseq_1, oseq_2):
                all_equivalent = all_equivalent and self.check_oracle_elements_equivalent(*x)
            
            self.assertTrue(all_equivalent)

        # @unittest.skip("Skipping for sake of time")
        def test_get_expected_meta(self):
            """
            Check that we get the expected serials, distances, and runs
            """
            distances_to_check = (
                ALL_DISTANCES_FEET,
                ALL_DISTANCES_FEET[:int(len(ALL_DISTANCES_FEET)/2)],
                (14,2,44,8),
                (14,),
                (8,),
                (56,)
            )
            serials_to_check = (
                ALL_SERIAL_NUMBERS,
                ALL_SERIAL_NUMBERS[:int(len(ALL_SERIAL_NUMBERS)/2)],
                ("3123D52",),
                ("3123D52","3123D80","3123D58","3123D64"),
                ("3123D65",),
                ("3123D89",),
                ("3123D78",),
                ("3124E4A",),
            )
            runs_to_check = (
                ALL_RUNS,
                (1,),
                (2,),
            )

            for distances,serials,runs in itertools.product(distances_to_check, serials_to_check, runs_to_check):
                oseq = ORACLE_Sequence(
                    desired_serial_numbers=serials,
                    desired_distances=distances,
                    desired_runs=runs,
                    window_length=256,
                    window_stride=1,
                    num_examples_per_device=1000,
                    seed=1337,  
                    max_cache_size=100000*16,
                    return_IQ_as_tuple_with_offset=False
                )

                d = set()
                s = set()
                r = set()
                for x in oseq:
                    r.add(x["run"])
                    d.add(x["distance_ft"])
                    s.add(x["serial_number"])
                
                self.assertEqual(set(distances), d)
                self.assertEqual(set(serials), s)
                self.assertEqual(set(runs), r)
        
        # @unittest.skip("Skipping for sake of time")
        def test_get_correct_num_examples(self):
            num_examples_per_device = 1000
            oseq = ORACLE_Sequence(
                desired_serial_numbers=ALL_SERIAL_NUMBERS,
                desired_distances=ALL_DISTANCES_FEET,
                desired_runs=ALL_RUNS,
                window_length=256,
                window_stride=1,
                num_examples_per_device=num_examples_per_device,
                seed=1337,  
                max_cache_size=100000*16,
                return_IQ_as_tuple_with_offset=False
            )

            device_count = {}
            for serial in ALL_SERIAL_NUMBERS:
                device_count[serial] = 0
            

            for x in oseq:
                device_count[x["serial_number"]] += 1
            

            for key,count in device_count.items():
                self.assertEqual(count, num_examples_per_device)
        
        # @unittest.skip("Skipping for sake of time")
        def test_offset_data_is_equivalent(self):
            all_equivalent = True

            oseq_with_offset = ORACLE_Sequence(
                desired_serial_numbers=ALL_SERIAL_NUMBERS,
                desired_distances=ALL_DISTANCES_FEET,
                desired_runs=ALL_RUNS,
                window_length=256,
                window_stride=1,
                num_examples_per_device=1000,
                seed=1337,  
                max_cache_size=100000*16,
                return_IQ_as_tuple_with_offset=True
            )

            oseq_without_offset = ORACLE_Sequence(
                desired_serial_numbers=ALL_SERIAL_NUMBERS,
                desired_distances=ALL_DISTANCES_FEET,
                desired_runs=ALL_RUNS,
                window_length=256,
                window_stride=1,
                num_examples_per_device=1000,
                seed=1337,  
                max_cache_size=100000*16,
                return_IQ_as_tuple_with_offset=False
            )

            for x,y in zip(oseq_with_offset, oseq_without_offset):
                x["iq"] = x["iq"][1]
                all_equivalent = all_equivalent and self.check_oracle_elements_equivalent(x,y)
            
            self.assertTrue(all_equivalent)

        def test_window_size(self):
            window_sizes_to_test = (
                256,
                128,
                2,
            )


            for w in window_sizes_to_test:
                oseq = ORACLE_Sequence(
                    desired_serial_numbers=ALL_SERIAL_NUMBERS,
                    desired_distances=ALL_DISTANCES_FEET,
                    desired_runs=ALL_RUNS,
                    window_length=w,
                    window_stride=1,
                    num_examples_per_device=1000,
                    seed=1337,  
                    max_cache_size=100000*16,
                    return_IQ_as_tuple_with_offset=False
                )

                for x in oseq:
                    self.assertEqual(
                        int(w/2), # Divide by two since we are channelizing the I and Q
                        x["iq"].shape[1]
                    )

        # @unittest.skip("Skipping for sake of time")
        def test_data_is_accurate(self):
            oseq = ORACLE_Sequence(
                desired_serial_numbers=ALL_SERIAL_NUMBERS,
                desired_distances=ALL_DISTANCES_FEET,
                desired_runs=ALL_RUNS,
                window_length=256,
                window_stride=1,
                num_examples_per_device=1000,
                seed=1337,  
                max_cache_size=100000*16,
                return_IQ_as_tuple_with_offset=True
            )

            for x in oseq:
                distance = x["distance_ft"]
                serial = x["serial_number"]
                run = x["run"]
                offset = x["iq"][0]
                iq = x["iq"][1]

                paths = get_oracle_data_files_based_on_criteria(
                    desired_distances=[distance],
                    desired_runs=[run],
                    desired_serial_numbers=[serial]
                )
                assert len(paths) == 1
                path = paths[0]
                mm = np.memmap(path, np.double)


                # We have an I channel and a Q channel. The original data interleaves (IE: IQIQIQIQ).
                for idx, f in enumerate(iq[0]):
                    self.assertEqual(mm[offset+idx*2], f)

                for idx, f in enumerate(iq[1]):
                    self.assertEqual(mm[offset+1+idx*2], f)

        # @unittest.skip("Skipping for sake of time")
        def test_metadata_is_accurate(self):
            oseq = ORACLE_Sequence(
                desired_serial_numbers=ALL_SERIAL_NUMBERS,
                desired_distances=ALL_DISTANCES_FEET,
                desired_runs=ALL_RUNS,
                window_length=256,
                window_stride=1,
                num_examples_per_device=1000,
                seed=1337,  
                max_cache_size=100000*16,
                return_IQ_as_tuple_with_offset=True
            )

            for example, metadata in zip(oseq, oseq.metadata):
                self.assertEqual(
                    example["serial_number"],
                    metadata["serial_number"]
                )
                self.assertEqual(
                    example["run"],
                    metadata["run"]
                )
                self.assertEqual(
                    example["distance_ft"],
                    metadata["distance_ft"]
                )

            


    unittest.main()

    print("Cheesed to meet you")

    oracle_sequence = ORACLE_Sequence(
        # desired_distances=[14,2,56],
        # desired_runs=[1],
        # desired_serial_numbers=["3123D52","3123D65","3123D79"],
        desired_serial_numbers=ALL_SERIAL_NUMBERS,
        desired_distances=ALL_DISTANCES_FEET,
        desired_runs=ALL_RUNS,
        window_length=256,
        window_stride=1,
        num_examples_per_device=100,
        # num_examples_per_device=100,
        # seed=int(1337
        seed=int(time.time()),
        max_cache_size=100000*16
        # max_cache_size=1
    )


    for x in oracle_sequence:
        print(x)

    
#     import timeit
# # <average time in seconds> = timeit.timeit(lambda: <muh shit>, number=<number of iterations>)
#     def iterate():
#         for i in oracle_sequence:
#             pass
    
#     print(timeit.timeit(lambda: iterate(), number=1))
#     print(timeit.timeit(lambda: iterate(), number=100))
#     # print(timeit.timeit(lambda: iterate(), number=1))

#     print(len(oracle_sequence))
