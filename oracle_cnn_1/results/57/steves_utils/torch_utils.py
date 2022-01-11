from re import U
import torch
import math
from easydict import EasyDict
import numpy as np
from torch._C import Value


def independent_accuracy_assesment(model, dl):
    correct = 0
    total = 0
    for x,y in dl:
        z = model.forward(x.cuda())
        y_hat = torch.max(z,dim=1,)[1].cuda()

        n_correct = (y_hat.cuda() == y.cuda()).sum().item()
        n_total   = y.shape[0]

        correct += n_correct
        total   += n_total

    return correct / total

def numpy_to_hash(n:np.ndarray):
    return hash(n.data.tobytes())

"""
Input:
    datasets: standardized dict of datasets 
    ds_type: "ptn" or "cnn"

Dataset structure:
    {
        "source": {
            "original": {"train":source_original_train, "val":source_original_val, "test":source_original_test},
            "processed": {"train":source_processed_train, "val":source_processed_val, "test":source_processed_test}
        },
        "target": {
            "original": {"train":target_original_train, "val":target_original_val, "test":target_original_test},
            "processed": {"train":target_processed_train, "val":target_processed_val, "test":target_processed_test}
        },
    }
"""
def get_dataset_metrics(datasets:EasyDict, ds_type:str):
    metrics = {
        "source": {
            "train": {
                "n_unique_x": 0,
                "n_unique_y": 0,
                "n_batch/episode": 0,
            },
            "val": {
                "n_unique_x": 0,
                "n_unique_y": 0,
                "n_batch/episode": 0,
            },
            "test": {
                "n_unique_x": 0,
                "n_unique_y": 0,
                "n_batch/episode": 0,
            }
        },
        "target": {
            "train": {
                "n_unique_x": 0,
                "n_unique_y": 0,
                "n_batch/episode": 0,
            },
            "val": {
                "n_unique_x": 0,
                "n_unique_y": 0,
                "n_batch/episode": 0,
            },
            "test": {
                "n_unique_x": 0,
                "n_unique_y": 0,
                "n_batch/episode": 0,
            }
        },
    }

    if ds_type == "ptn":
        for source_or_target, d_1 in datasets.items():
            for split, ds in d_1["original"].items():
                unique_x = set()
                unique_y = set()
                n_batches = 0
                for u, (support_x, support_y, query_x, query_y, real_classes) in ds:
                    n_batches += 1
                    for x in support_x: unique_x.add( numpy_to_hash(x.numpy()) )
                    for x in query_x: unique_x.add( numpy_to_hash(x.numpy()) )

                    for y in support_y: unique_y.add( real_classes[y] )
                    for y in query_y: unique_y.add( real_classes[y] )

                metrics[source_or_target][split]["n_unique_x"] = len(unique_x)
                metrics[source_or_target][split]["n_unique_y"] = len(unique_y)
                metrics[source_or_target][split]["n_batch/episode"] = n_batches

    elif ds_type == "cnn":
        for source_or_target, d_1 in datasets.items():
            for split, ds in d_1["original"].items():
                unique_x = set()
                unique_y = set()
                for x,y,u in ds:
                    unique_x.add( numpy_to_hash(x) )
                    unique_y.add( y )

                metrics[source_or_target][split]["n_unique_x"] = len(unique_x)
                metrics[source_or_target][split]["n_unique_y"] = len(unique_y)
            for split, ds in d_1["processed"].items():
                n_batches = sum( 1 for _ in ds )
                metrics[source_or_target][split]["n_batch/episode"] = n_batches


    else:
        raise ValueError("ds_type incorrect")

    return metrics


    


def split_dataset_by_percentage(train:float, val:float, test:float, dataset, seed:int):
    assert train < 1.0
    assert val < 1.0
    assert test < 1.0
    assert train + val + test <= 1.0

    num_train = math.floor(len(dataset) * train)
    num_val   = math.floor(len(dataset) * val)
    num_test  = math.floor(len(dataset) * test)

    return torch.utils.data.random_split(dataset, (num_train, num_val, num_test), generator=torch.Generator().manual_seed(seed))

def split_dataset_by_percentage_v2(train:float, val:float, dataset, seed:int):
    """
    test just gets the remainder
    """
    assert train < 1.0
    assert val < 1.0
    assert train + val < 1.0

    num_train = math.floor(len(dataset) * train)
    num_val   = math.floor(len(dataset) * val)
    num_test  = len(dataset) - num_train - num_val

    assert num_test > 0

    return torch.utils.data.random_split(dataset, (num_train, num_val, num_test), generator=torch.Generator().manual_seed(seed))

"""
Assumes batch is in the form (X,Y,U) or (X,Y)
"""
def predict_batch(model, device, batch, forward_uses_domain):
    model.eval()
    tup = [t.to(device) for t in batch]

    if forward_uses_domain:
        y_hat, u_hat = model.forward(tup[0], tup[2])
    else:
        y_hat = model.forward(tup[0])
    pred = y_hat.data.max(1, keepdim=True)[1]
    pred = torch.flatten(pred).cpu()

    return pred


"""
Returns dict in the form of
{
    <domain>:
    {
        <y>: {
            <y_hat>: int
        }
    },
}

IE [domain][y][y_hat]:count

IE a confusion matrix for each domain
"""
def confusion_by_domain_over_dataloader(model, device, dl, forward_uses_domain, denormalize_domain_func=None):
    confusion_by_domain = {}

    for batch in dl:
        pred = predict_batch(model, device, batch, forward_uses_domain)

        batch.append(pred)



        for x,y,u,y_hat in zip(*batch):
            x = x.cpu().detach().numpy()
            y = int(y.cpu().detach().numpy())
            u = float(u.cpu().detach().numpy())
            y_hat = int(y_hat.cpu().detach().numpy())

            if denormalize_domain_func is not None:
                u = denormalize_domain_func(u)

            u = int(u)

            # Yeah yeah I know...
            if u not in confusion_by_domain:
                confusion_by_domain[u] = {}
            if y not in confusion_by_domain[u]:
                confusion_by_domain[u][y] = {}
            if y_hat not in confusion_by_domain[u][y]:
                confusion_by_domain[u][y][y_hat] = 0

            confusion_by_domain[u][y][y_hat] += 1

    return confusion_by_domain


def ptn_confusion_by_domain_over_dataloader(model, device, dl):
    confusion_by_domain = {}
    correct_and_total_by_domain = {} # Going to use this to validate

    for u, (support_x, support_pseudo_y, query_x, query_pseudo_y, classes) in dl:
        # Returns pseudo labels so we need to fetch them from the classes list that 
        # is generated for each episode
        pseudo_y_hat = model.predict_on_one_task(
            support_x,
            support_pseudo_y,
            query_x,
            query_pseudo_y,
        )

        y_hat = [classes[idx] for idx in pseudo_y_hat]
        query_y = [classes[idx] for idx in query_pseudo_y]

        # These two chunks are just for sanity checking our confusion matrix
        correct, total, _loss = model.evaluate_on_one_task(
            support_x, support_pseudo_y, query_x, query_pseudo_y
        )

        if u not in correct_and_total_by_domain:
            correct_and_total_by_domain[u] = [0, 0]
        correct_and_total_by_domain[u][0] += correct
        correct_and_total_by_domain[u][1] += total


         
        for y, y_hat in zip(query_y, y_hat):
            # Yeah yeah I know...
            if u not in confusion_by_domain:
                confusion_by_domain[u] = {}
            if y not in confusion_by_domain[u]:
                confusion_by_domain[u][y] = {}
            if y_hat not in confusion_by_domain[u][y]:
                confusion_by_domain[u][y][y_hat] = 0
            confusion_by_domain[u][y][y_hat] += 1

    # import pprint
    # pp = pprint.PrettyPrinter(indent=2)
    # pp.pprint(confusion_by_domain)

    # ok now check the matrix by domain
    for domain, y_and_y_hat in confusion_by_domain.items():
        n_correct = 0
        n_total = 0
        for y, y_hat_dict in y_and_y_hat.items():
            if y in y_hat_dict: # Handle the case if we never had a single correct guess
                n_correct += y_hat_dict[y]
            n_total += sum(y_hat_dict.values())
        
        if correct_and_total_by_domain[domain][0] != n_correct:
            print("[n_correct]","Got:",n_correct, "Expected:", correct_and_total_by_domain[domain][0])
            print("[n_total]",  "Got:",n_total,   "Expected:", correct_and_total_by_domain[domain][1])
            raise RuntimeError(f"Error in confusion matrix calculation: n_correct is wrong for domain {domain}")

        if correct_and_total_by_domain[domain][1] != n_total:
            raise RuntimeError(f"Error in confusion matrix calculation: n_total is wrong for domain {domain}")       


    return confusion_by_domain