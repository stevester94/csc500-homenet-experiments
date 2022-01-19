#! /usr/bin/env python3


import numpy as np
import pickle
import os
from steves_utils.CORES.utils import  get_dataset_by_day_and_node_name, get_cores_dataset_path
from steves_utils.utils_v2 import norm
from typing import List
import torch

class CORES_Torch_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        days_to_get:List[int],
        num_examples_per_node_per_day,
        nodes_to_get,
        seed:int,
        normalize:str=False,
        root_dir:str=get_cores_dataset_path(),
	    transform_func=None,
    ) -> None:
        self.ds = get_dataset_by_day_and_node_name(
            num_examples_per_node_per_day=num_examples_per_node_per_day,
            nodes_to_get=nodes_to_get,
            seed=seed,
            days_to_get=days_to_get,
            root_dir=root_dir
        )

        self.transform_func = transform_func
        self.normalize = normalize

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        d = self.ds[idx]
        if self.normalize is not False:
            d["IQ"] = norm(d["IQ"], self.normalize)

        if self.transform_func != None:
            return self.transform_func(d)
        else:
            return d

if __name__ == "__main__":
    from utils import ALL_NODES, ALL_DAYS

    ctd = CORES_Torch_Dataset(
        days_to_get=ALL_DAYS,
        num_examples_per_node_per_day=100,
        nodes_to_get=ALL_NODES,
        seed=1337,
	    # transform_func=None,
    )
    print(ctd[0])

    ctd = CORES_Torch_Dataset(
        days_to_get=ALL_DAYS,
        num_examples_per_node_per_day=100,
        nodes_to_get=ALL_NODES,
        seed=1337,
	    transform_func=lambda ex: (ex["day"], ex["node_name"])
    )
    print(ctd[0])

