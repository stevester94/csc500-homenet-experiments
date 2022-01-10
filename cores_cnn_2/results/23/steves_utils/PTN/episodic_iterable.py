#! /usr/bin/env python3


from typing import List, Tuple
import numpy as np
import torch
import copy

"""
Steven Mackey: This code was adapted from https://pypi.org/project/easyfsl/
"""
class EpisodicIterable:
    """
    Samples batches in the shape of few-shot classification tasks. At each iteration, it will sample
    n_way classes, and then sample support and query images from these classes.
    """

    def __init__(
        self, dataset: list, labels:list, n_way: int, n_shot: int, n_query: int, k_factor: int, seed: int, randomize_each_iter: bool
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
        super().__init__()
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.k_factor = k_factor
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.randomize_each_iter = randomize_each_iter
        self.dataset = dataset

        self.disable_colate = False

        self.indices_by_label = {}
        for item, label in enumerate(labels):
            if label in self.indices_by_label.keys():
                self.indices_by_label[label].append(item)
            else:
                self.indices_by_label[label] = [item]

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

        Programming is fun
        """
        count = 0
        self.disable_colate = True
        original_rng = copy.deepcopy(self.rng)
        for _ in iter(self): count += 1
        self.rng = original_rng
        self.disable_colate = False

        return count

    def __iter__(self):
        # If we don't randomize each iteration then we just reset the RNG to its default state
        if not self.randomize_each_iter:
            self.rng = np.random.default_rng(self.seed)

        # K factor is just a higher level loop, but because we are using RNG on the 
        # building of the episodes it means we generate different episode on each loop
        for _ in range(self.k_factor):

            # We keep a dictionary of available examples by index for each label
            available_indices_by_label = copy.deepcopy(self.indices_by_label)

            # Do an initial purge of labels that don't have enough elements. This can happen with small datasets
            to_delete = [label for label, indices in available_indices_by_label.items() if len(indices) < self.n_shot + self.n_query]
            for d in to_delete: del available_indices_by_label[d]

            # If we have fewer labels than n_way then we break
            while len(available_indices_by_label) >= self.n_way:
                # The indices used for this episode, built up for each label
                episode_indices = []
                try:
                    # Make an n_way choice of labels for this episode, and select indices for each one of them
                    for label in self.rng.choice(list(available_indices_by_label.keys()), self.n_way, replace=False):
                        indices_for_this_label = torch.tensor(
                            self.rng.choice(
                                available_indices_by_label[label], self.n_shot + self.n_query, replace=False
                            )
                        )

                        episode_indices.append(indices_for_this_label)

                        # Remove the indices we just used from the available indices for this label
                        for i in indices_for_this_label: available_indices_by_label[label].remove(i)
                        
                        # If we have exhausted this label then delete it
                        if len(available_indices_by_label[label]) < self.n_shot + self.n_query:
                            del available_indices_by_label[label]


                except KeyError:
                    raise
                        

                indices = torch.cat(episode_indices)
                
                if not self.disable_colate:
                    yield self.episodic_collate_fn([
                        (torch.from_numpy(self.dataset[i][0]), self.dataset[i][1]) for i in indices
                    ])
                else:
                    yield None

    def episodic_collate_fn(
        self, input_data: List[Tuple[torch.Tensor, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Collate function to be used as argument for the collate_fn parameter of episodic
            data loaders.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor
                - the label of this image
        Returns:
            tuple(Tensor, Tensor, Tensor, Tensor, list[int]): respectively:
                - support images,
                - their labels,
                - query images,
                - their labels,
                - the dataset class ids of the class sampled in the episode
        """

        true_class_ids = list({x[1] for x in input_data})

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
            support_images,
            support_labels,
            query_images,
            query_labels,
            true_class_ids,
        )


if __name__ == "__main__":
    import steves_utils.CORES.utils as suc

    d, _, _ = suc.get_it(
        num_examples_per_node_per_day=100,
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

    ei = EpisodicIterable(
        dataset=d,
        labels=labels,
        n_way=len(suc.ALL_NODES_MINIMUM_1000_EXAMPLES),
        n_shot=2,
        n_query=3,
        k_factor=10,
        seed=1337,
        randomize_each_iter=False
    )


    print(len(list(ei)))
    print(len(list(ei)))
    print(len(list(ei)))
    print(len(list(ei)))


    for i in ei:
        pass

    next(iter(ei))