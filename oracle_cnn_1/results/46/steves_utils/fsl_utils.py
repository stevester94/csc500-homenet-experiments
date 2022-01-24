from math import floor
import random
import torch

# Note I am using my own version of easyfsl
from easyfsl.data_tools import TaskSampler


def split_ds_into_episodes(
    ds,
    labels,
    n_way,
    n_shot,
    n_query,
    n_train_tasks,
    n_val_tasks,
    n_test_tasks,
    seed,
):
    """
    Splits ds into train,val,test sets, each being a episodic
    Returns:
        tuple(
            train_dl,
            val_list,
            test_list
        )
        Val and test are returned as lists so that they are not resampled dynamically
    """
    import copy 

    train_len = floor(len(ds)*0.7)
    val_len   = floor(len(ds)*0.15)
    test_len  = len(ds) - train_len - val_len

    train_ds, val_ds, test_ds = torch.utils.data.random_split(ds, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(seed))
    train_labels, val_labels, test_labels = torch.utils.data.random_split(labels, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(seed))

    train_ds.labels = train_labels
    val_ds.labels   = val_labels
    test_ds.labels  = test_labels

    def wrap_in_dataloader(ds,n_tasks, seed, randomize_each_iter):
        sampler = TaskSampler(
                ds,
                n_way=n_way,
                n_shot=n_shot,
                n_query=n_query,
                n_tasks=n_tasks,
                seed=seed,
                randomize_each_iter=randomize_each_iter
            )

        return torch.utils.data.DataLoader(
            ds,
            num_workers=1,
            persistent_workers=True,
            prefetch_factor=25,
            pin_memory=True,
            batch_sampler=sampler,
            collate_fn=sampler.episodic_collate_fn
        )

    # Deep copy necessary because of how tensors are retrieved from workers
    # val_list = []
    # for k in wrap_in_dataloader(val_ds, n_val_tasks, seed, randomize_each_iter=False):
    #     val_list.append(copy.deepcopy(k))

    # test_list = []
    # for k in wrap_in_dataloader(test_ds, n_test_tasks, seed, randomize_each_iter=False):
    #     test_list.append(copy.deepcopy(k))

    
    return (
        wrap_in_dataloader(train_ds, n_train_tasks, seed, randomize_each_iter=True),
        wrap_in_dataloader(val_ds, n_val_tasks, seed, randomize_each_iter=False),
        wrap_in_dataloader(test_ds, n_test_tasks, seed, randomize_each_iter=False),
        # val_list,
        # test_list
    )