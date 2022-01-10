#! /usr/bin/env python3

import unittest
import steves_utils.CORES.utils as suc
from episodic_iterable import EpisodicIterable






class Test_Episodic(unittest.TestCase):
    def test_shape(self):
        d, _, _ = suc.get_it(
            num_examples_per_node_per_day=1000,
            nodes_to_get=suc.ALL_NODES_MINIMUM_1000_EXAMPLES,
            days_to_get=suc.ALL_DAYS,
            seed=1337
        )

        d = d[1]

        labels = list(
            map(lambda ex: ex[1],d)
        )

        d = list(
            map(lambda ex: ex[:2], d)
        )

        for x,y in d:
            self.assertTrue(
                x.shape == (2,256)
            )

        # indices_by_label = {}
        # for item, label in enumerate(labels):
        #     if label in indices_by_label.keys():
        #         indices_by_label[label].append(item)
        #     else:
        #         indices_by_label[label] = [item]

        # for label, indices in indices_by_label.items():
        #     print(label,":",len(indices))

        # # print(len(d))
        # # print(labels)

        # # Do the thing
        # for seed in [1337, 1984, 2020, 18081994, 4321326]:
        #     # ei = EpisodicIterable(
        #     #     dataset=d,
        #     #     labels=labels,
        #     #     n_way=len(suc.ALL_NODES_MINIMUM_1000_EXAMPLES)-5,
        #     #     n_shot=2,
        #     #     n_query=3,
        #     #     k_factor=1,
        #     #     seed=seed,
        #     #     randomize_each_iter=False,
        #     # )

        ei = EpisodicIterable(
            dataset=d,
            labels=labels,
            n_way=len(suc.ALL_NODES_MINIMUM_1000_EXAMPLES)-5,
            n_shot=2,
            n_query=3,
            k_factor=1,
            seed=1337,
            randomize_each_iter=False,
        )

        for episode in ei:
            for member in episode:
                print(type(member))


        #     print(len(ei))







unittest.main()