#! /usr/bin/env python3
from dataclasses import replace
from operator import index
from signal import valid_signals
from torch.utils.data import Sampler, DataLoader
import numpy as np
import torch
from typing import List, Tuple
import copy
import math

def get_episodic_dataloaders(
    stratified_ds:dict,
    train_val_test_percents:tuple,
    num_examples_per_domain_per_class:int,
    n_shot:int,
    n_way:int,
    n_query:int,
    train_val_test_k_factors:tuple,
    iterator_seed:int,
    x_transform_func=False,
    dataloader_kwargs:dict={},
)->Tuple[DataLoader, DataLoader, DataLoader]:
    train_sds = {}
    val_sds = {}
    test_sds = {}

    n_train = math.floor(num_examples_per_domain_per_class*train_val_test_percents[0])
    n_val = math.floor(num_examples_per_domain_per_class*train_val_test_percents[1])
    n_test = (num_examples_per_domain_per_class - n_train - n_val)

    for domain, label_and_x_dict in stratified_ds.items():
        train_sds[domain] = {}
        val_sds[domain] = {}
        test_sds[domain] = {}
        for label, all_x in label_and_x_dict.items():
            train_sds[domain][label] = all_x[:n_train]
            val_sds[domain][label]   = all_x[n_train:n_train+n_val]
            test_sds[domain][label]  = all_x[n_train+n_val:n_train+n_val+n_test]

            # print(f"train_sds[{domain}][{label}]", len(train_sds[domain][label]))
            # print(f"val_sds[{domain}][{label}]", len(val_sds[domain][label]))
            # print(f"test_sds[{domain}][{label}]", len(test_sds[domain][label]))

    # Default params for dl
    default_dataloader_kwargs = {
        "num_workers": 1,
        "pin_memory": True,
        "prefetch_factor": 10,
    }

    for key, val in default_dataloader_kwargs.items():
        if key not in dataloader_kwargs:
            dataloader_kwargs[key] = val

    train_dl = get_single_episodic_dataloader(
        stratified_ds=train_sds,
        n_shot=n_shot,
        n_way=n_way,
        n_query=n_query,
        seed=iterator_seed,
        x_transform_func=x_transform_func,
        randomize_each_iter=True,
        k_factor=train_val_test_k_factors[0],
        dataloader_kwargs = dataloader_kwargs
    )

    val_dl = get_single_episodic_dataloader(
        stratified_ds=val_sds,
        n_shot=n_shot,
        n_way=n_way,
        n_query=n_query,
        seed=iterator_seed,
        x_transform_func=x_transform_func,
        randomize_each_iter=False,
        k_factor=train_val_test_k_factors[1],
        dataloader_kwargs = dataloader_kwargs
    )

    test_dl = get_single_episodic_dataloader(
        stratified_ds=test_sds,
        n_shot=n_shot,
        n_way=n_way,
        n_query=n_query,
        seed=iterator_seed,
        x_transform_func=x_transform_func,
        randomize_each_iter=False,
        k_factor=train_val_test_k_factors[2],
        dataloader_kwargs = dataloader_kwargs
    )

    return train_dl, val_dl, test_dl
    




def get_single_episodic_dataloader(
    stratified_ds:dict,
    n_way: int,
    n_shot: int,
    n_query: int,
    seed: int,
    randomize_each_iter: bool,
    k_factor:int,
    dataloader_kwargs:dict,
    x_transform_func=None,
):
    # Build the index and data
    index = {}
    data  = []
    i = 0


    for domain, label_and_x in stratified_ds.items():
        index[domain] = {}
        for label, all_x in label_and_x.items():
            index[domain][label] = []
            for x in all_x:
                x = torch.from_numpy(x)
                if x_transform_func != False:
                    data.append( (x_transform_func(x), label, domain) )
                else:
                    data.append( (x, label, domain) )

                index[domain][label].append(i)

                i += 1

    sampler = stratified_dataset_episodic_sampler(
        index=index,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        seed=seed,
        randomize_each_iter=randomize_each_iter,
        k_factor=k_factor
    )


    dl = DataLoader(data, batch_sampler=sampler, collate_fn=sampler.episodic_collate_fn, **dataloader_kwargs)

    return dl


            

class stratified_dataset_episodic_sampler(Sampler):
    def __init__(
        self, 
        index:dict,
        n_way: int,
        n_shot: int,
        n_query: int,
        seed: int,
        randomize_each_iter: bool,
        k_factor:int,
    ):
        """
        Args:
            dataset: dataset from which to sample classification tasks. Must have a field 'label': a
                list of length len(dataset) containing containing the labels of all images.
            n_way: number of classes in one task
            n_shot: number of support images for each class in one task
            n_query: number of query images for each class in one task
            n_tasks: number of tasks to sample
        """
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.randomize_each_iter = randomize_each_iter
        self.k_factor = k_factor

        self.index = index

    def __len__(self):
        """
        Hahhahahahaha, kill me
        Because we need the ability to randomize which examples are used for each episode,
        our len can change between iterations if we have n_way < |classes|.

        Therefore, in order to get an accurate len we must manually iterate over ourself.
        HOWEVER this is a problem because normally the next iteration would potentially be 
        different! We get around this by deepcopying our rng, iterating, then restoring that RNG.

        Note that this is not necessary for the case where randomize_each_iter == False, however
        we just keep the same code here for that case because it makes no difference
        """
        count = 0
        self.disable_colate = True
        original_rng = copy.deepcopy(self.rng)
        for _ in iter(self): count += 1
        self.rng = original_rng

        return count

    def get_items_in_index(self, index):
        total_indices = 0
        for u, y_x in index.items():
            for y, x in y_x.items():
                total_indices += len(x)
        return total_indices


    # Clean the label portion of an index in place, IE
    # self.clean_label(index[<a single domain>])
    def clean_index(self, index)->None:
        # First pass, delete all the labels that have too few indices
        for domain, label_and_indices in index.items():
            labels_to_delete = [label for label, indices in index[domain].items() if len(indices) < self.n_shot + self.n_query]
            for label in labels_to_delete: del index[domain][label]

        # Second pass, delete all the domains that have too few labels left in them
        domains_to_delete = [domain for domain, label_and_indices in index.items() if len(label_and_indices) < self.n_way]
        for domain in domains_to_delete: del index[domain]
        

    def stat_index(self, index):
        for domain, label_and_indices in index.items():
            print(domain)
            for label, indices in label_and_indices.items():
                print("  ", label,":",len(indices))


    def __iter__(self):
        # If we don't randomize each iteration then we just reset the RNG to its default state
        if not self.randomize_each_iter:
            self.rng = np.random.default_rng(self.seed)

        """
        A foreword, by Steven Mackey
        This madness needs some explanation. We are maintaning a dict which gives us the index of each
        element in our data, by its domain and then label.
        Using this, we select n_way labels, and then n_shot+n_query indices from that list.
        We then delete those indices from the list.
        If upon deletion that [domain,label] could not support another round of indices, we purge it.

        We do this until the entire list is empty

        We do this all k_factor times

        Why is this necessary? Because we want each episode to be drawn from a single domain
        """

        # K factor is just a higher level loop, but because we are using RNG on the 
        # building of the episodes it means we generate different episode on each loop
        for _ in range(self.k_factor):

            # We keep a dictionary of available examples by index for each label
            index_copy = copy.deepcopy(self.index)

            # Do an initial purge of labels that don't have enough elements. This can happen with small datasets
            self.clean_index(index_copy)

            try:
                # While we have at least domain, we continue
                while len(index_copy) > 0:
                    # The indices used for this episode, built up for each label
                    episode_indices = []
                    domain = self.rng.choice(list(index_copy.keys()), 1)[0]

                    # Make an n_way choice of labels for this episode, and select indices for each one of them
                    labels_for_this_episode = self.rng.choice(list(index_copy[domain].keys()), self.n_way, replace=False)
                    for label in labels_for_this_episode:
                        indices_for_this_label = torch.tensor(
                            self.rng.choice(
                                index_copy[domain][label], self.n_shot + self.n_query, replace=False
                            )
                        )

                        episode_indices.append(indices_for_this_label)

                        # Remove the indices we just used from the available indices for this label
                        for i in indices_for_this_label: index_copy[domain][label].remove(i)
                        
                    self.clean_index(index_copy)

                    episode_indices = torch.cat(episode_indices)

                    yield episode_indices
            except Exception as e:
                self.stat_index(index_copy)
                raise

    def episodic_collate_fn(
        self, input_data: List[Tuple[torch.Tensor, int, int]]
    ) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]]:
        """
        Collate function to be used as argument for the collate_fn parameter of episodic
            data loaders.
        Args:
            input_data: each element is a tuple containing: (x, y, u)
        Returns:

        """
        true_class_ids = list({x[1] for x in input_data})

        # I dont care if it slows us down, we must be sure that the entire episode is 
        # from the same domain
        all_domains = set([u for x,y,u in input_data])
        assert(
            len(
                all_domains
            ) == 1
        )

        domain = all_domains.pop()


        all_images = torch.cat([x[0].unsqueeze(0) for x in input_data])
        all_images = all_images.reshape(
            (self.n_way, self.n_shot + self.n_query, *all_images.shape[1:])
        )
        # pylint: disable=not-callable
        all_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in input_data]
        ).reshape((self.n_way, self.n_shot + self.n_query))
        # pylint: enable=not-callable

        support_images = all_images[:, : self.n_shot].reshape(
            (-1, *all_images.shape[2:])
        )
        query_images = all_images[:, self.n_shot :].reshape((-1, *all_images.shape[2:]))
        support_labels = all_labels[:, : self.n_shot].flatten()
        query_labels = all_labels[:, self.n_shot :].flatten()

        return (
            domain, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                true_class_ids,
            )
        )


class dumb_sampler(Sampler):
    def __init__(
        self, 
    ):
        super().__init__(data_source=None)
        self.rng = np.random.default_rng(1337)
        self.len = None

    def __len__(self):
        if self.len != None:
            return self.len
        else:
            return self.rng.choice(1000, 1)[0]

    def __iter__(self):
        self.len = 42
        # If we don't randomize each iteration then we just reset the RNG to its default state
        if not self.randomize_each_iter:
            self.rng = np.random.default_rng(self.seed)
        for _ in range(self.n_tasks):
            yield 0


if __name__ == "__main__":
    dl = DataLoader([0,0,0,0], batch_size=1, shuffle=False, sampler=dumb_sampler())

    print(len(dl))
    print(len(dl))
    print(len(dl))


# def __iter__(self):
#     # If we don't randomize each iteration then we just reset the RNG to its default state
#     if not self.randomize_each_iter:
#         self.rng = np.random.default_rng(self.seed)

#     # K factor is just a higher level loop, but because we are using RNG on the 
#     # building of the episodes it means we generate different episode on each loop
#     for _ in range(self.k_factor):

#         # We keep a dictionary of available examples by index for each label
#         available_indices_by_label = copy.deepcopy(self.indices_by_label)

#         # Do an initial purge of labels that don't have enough elements. This can happen with small datasets
#         to_delete = [label for label, indices in available_indices_by_label.items() if len(indices) < self.n_shot + self.n_query]
#         for d in to_delete: del available_indices_by_label[d]

#         # If we have fewer labels than n_way then we break
#         while len(available_indices_by_label) >= self.n_way:
#             # The indices used for this episode, built up for each label
#             episode_indices = []
#             try:
#                 # Make an n_way choice of labels for this episode, and select indices for each one of them
#                 for label in self.rng.choice(list(available_indices_by_label.keys()), self.n_way, replace=False):
#                     indices_for_this_label = torch.tensor(
#                         self.rng.choice(
#                             available_indices_by_label[label], self.n_shot + self.n_query, replace=False
#                         )
#                     )

#                     episode_indices.append(indices_for_this_label)

#                     # Remove the indices we just used from the available indices for this label
#                     for i in indices_for_this_label: available_indices_by_label[label].remove(i)
                    
#                     # If we have exhausted this label then delete it
#                     if len(available_indices_by_label[label]) < self.n_shot + self.n_query:
#                         del available_indices_by_label[label]


#             except KeyError:
#                 raise
                    

#             indices = torch.cat(episode_indices)
            
#             if not self.disable_colate:
#                 yield self.episodic_collate_fn([
#                     (torch.from_numpy(self.dataset[i][0]), self.dataset[i][1]) for i in indices
#                 ])
#             else:
#                 yield None

# def episodic_collate_fn(
#     self, input_data: List[Tuple[torch.Tensor, int]]
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
#     """
#     Collate function to be used as argument for the collate_fn parameter of episodic
#         data loaders.
#     Args:
#         input_data: each element is a tuple containing:
#             - an image as a torch Tensor
#             - the label of this image
#     Returns:
#         tuple(Tensor, Tensor, Tensor, Tensor, list[int]): respectively:
#             - support images,
#             - their labels,
#             - query images,
#             - their labels,
#             - the dataset class ids of the class sampled in the episode
#     """

#     true_class_ids = list({x[1] for x in input_data})

#     all_images = torch.cat([x[0].unsqueeze(0) for x in input_data])
#     all_images = all_images.reshape(
#         (self.n_way, self.n_shot + self.n_query, *all_images.shape[1:])
#     )
#     # pylint: disable=not-callable
#     all_labels = torch.tensor(
#         [true_class_ids.index(x[1]) for x in input_data]
#     ).reshape((self.n_way, self.n_shot + self.n_query))
#     # pylint: enable=not-callable

#     support_images = all_images[:, : self.n_shot].reshape(
#         (-1, *all_images.shape[2:])
#     )
#     query_images = all_images[:, self.n_shot :].reshape((-1, *all_images.shape[2:]))
#     support_labels = all_labels[:, : self.n_shot].flatten()
#     query_labels = all_labels[:, self.n_shot :].flatten()

#     return (
#         support_images,
#         support_labels,
#         query_images,
#         query_labels,
#         true_class_ids,
#     )