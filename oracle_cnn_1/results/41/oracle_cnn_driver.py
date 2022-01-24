#! /usr/bin/env python3
import os, json, sys, time, random
import numpy as np
import torch
from  easydict import EasyDict
from math import floor

from steves_utils.torch_utils import get_dataset_metrics
from steves_models.configurable_vanilla import Configurable_Vanilla
from steves_utils.vanilla_train_eval_test_jig import  Vanilla_Train_Eval_Test_Jig
from steves_utils.torch_sequential_builder import build_sequential
from steves_utils.lazy_map import Lazy_Map
from steves_utils.sequence_aggregator import Sequence_Aggregator

from steves_utils.simple_datasets.ORACLE.dataset_accessor import get_datasets
from steves_utils.ORACLE.utils_v2 import (
    ALL_DISTANCES_FEET,
    ALL_SERIAL_NUMBERS,
    ALL_RUNS,
    serial_number_to_id
)

from steves_utils.torch_utils import (
    confusion_by_domain_over_dataloader,
    independent_accuracy_assesment
)

from steves_utils.utils_v2 import (
    per_domain_accuracy_from_confusion,
)

from do_report import do_report


###################################
# Parse Args, Set paramaters
###################################

base_parameters = {}
base_parameters["experiment_name"] = "MANUAL ORACLE CNN"
base_parameters["lr"] = 0.0001
base_parameters["n_epoch"] = 3
base_parameters["batch_size"] = 256
base_parameters["patience"] = 10
base_parameters["seed"] = 1337
base_parameters["device"] = "cuda"
base_parameters["desired_classes"] = ALL_SERIAL_NUMBERS
base_parameters["source_domains"] = [50,32,8]
base_parameters["target_domains"] = list(set(ALL_DISTANCES_FEET) - set([50,32,8]))

base_parameters["num_examples_per_class_per_domain"]=1000

base_parameters["criteria_for_best"] = "source"
base_parameters["normalize_source"] = False
base_parameters["normalize_target"] = False

base_parameters["NUM_LOGS_PER_EPOCH"] = 10
base_parameters["RESULTS_DIR"] = "./results"
base_parameters["EXPERIMENT_JSON_PATH"] = os.path.join(base_parameters["RESULTS_DIR"], "experiment.json")
base_parameters["LOSS_CURVE_PATH"] = os.path.join(base_parameters["RESULTS_DIR"], "loss.png")
base_parameters["BEST_MODEL_PATH"] = os.path.join(base_parameters["RESULTS_DIR"], "best_model.pth")



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

    {"class": "Linear", "kargs": {"in_features": 256, "out_features": len(base_parameters["desired_classes"])}},
]


def wrap_in_dataloader(p, ds):
    return torch.utils.data.DataLoader(
        ds,
        batch_size=p.batch_size,
        shuffle=True,
        num_workers=1,
        persistent_workers=True,
        prefetch_factor=50,
        pin_memory=True
    )

def parse_and_validate_parameters(d:dict)->EasyDict:
    ed = EasyDict(d)
    allowed_keys =set(base_parameters.keys())
    supplied_keys = set(ed.keys())

    

    if  supplied_keys != allowed_keys:
        print("Parameters are incorrect")
        if len(supplied_keys - allowed_keys)>0: print("Shouldn't have:", str(supplied_keys - allowed_keys))
        if len(allowed_keys - supplied_keys)>0: print("Need to have:", str(allowed_keys - supplied_keys))
        raise RuntimeError("Parameters are incorrect")

    return ed

###################################
# Set the RNGs and make it all deterministic
###################################
def set_rng(p:EasyDict)->None:
    np.random.seed(p.seed)
    random.seed(p.seed)
    torch.manual_seed(p.seed)

    torch.use_deterministic_algorithms(True) 




###################################
# Build the network(s)
# Note: It's critical to do this AFTER setting the RNG
###################################
def build_network(p:EasyDict):
    return build_sequential(p.x_net)

###################################
# Build the dataset
###################################
def build_datasets(p:EasyDict)->EasyDict:
    """
    returns {
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


    source_original_train, source_original_val, source_original_test = get_datasets(
        serial_numbers=p.desired_classes,
        distances=p.source_domains,
        num_examples_per_distance_per_serial=p.num_examples_per_class_per_domain,
        normalize_type=p.normalize_source,
    )


    target_original_train, target_original_val, target_original_test = get_datasets(
        serial_numbers=p.desired_classes,
        distances=p.target_domains,
        num_examples_per_distance_per_serial=p.num_examples_per_class_per_domain,
        normalize_type=p.normalize_source,
    )

    # For CNN We only use X and Y. And we only train on the source.
    # Properly form the data using a transform lambda and Lazy_Map. Finally wrap them in a dataloader

    transform_lambda = lambda ex: ex[:2] # Strip the tuple to just (x,y)

    source_processed_train = wrap_in_dataloader(
        p,
        Lazy_Map(source_original_train, transform_lambda)
    )
    source_processed_val = wrap_in_dataloader(
        p,
        Lazy_Map(source_original_val, transform_lambda)
    )
    source_processed_test = wrap_in_dataloader(
        p,
        Lazy_Map(source_original_test, transform_lambda)
    )

    target_processed_train = wrap_in_dataloader(
        p,
        Lazy_Map(target_original_train, transform_lambda)
    )
    target_processed_val = wrap_in_dataloader(
        p,
        Lazy_Map(target_original_val, transform_lambda)
    )
    target_processed_test  = wrap_in_dataloader(
        p,
        Lazy_Map(target_original_test, transform_lambda)
    )



    return EasyDict({
        "source": {
            "original": {"train":source_original_train, "val":source_original_val, "test":source_original_test},
            "processed": {"train":source_processed_train, "val":source_processed_val, "test":source_processed_test}
        },
        "target": {
            "original": {"train":target_original_train, "val":target_original_val, "test":target_original_test},
            "processed": {"train":target_processed_train, "val":target_processed_val, "test":target_processed_test}
        },
    })

###################################
# Build the model
###################################
def build_model(p:EasyDict, model)->tuple:

    return Configurable_Vanilla(
        x_net=model,
        label_loss_object=torch.nn.NLLLoss(),
        learning_rate=p.lr
    )


###################################
# Build the tet jig, train
###################################
def train(p:EasyDict, model, ds:EasyDict)->Vanilla_Train_Eval_Test_Jig:
    jig = Vanilla_Train_Eval_Test_Jig(
        model=model,
        path_to_best_model=p.BEST_MODEL_PATH,
        device=p.device,
        label_loss_object=torch.nn.NLLLoss(),
    )

    jig.train(
        train_iterable=ds.source.processed.train,
        source_val_iterable=ds.source.processed.val,
        target_val_iterable=ds.target.processed.val,
        patience=p.patience,
        num_epochs=p.n_epoch,
        num_logs_per_epoch=p.NUM_LOGS_PER_EPOCH,
        criteria_for_best=p.criteria_for_best
    )

    return jig


###################################
# Evaluate the model
###################################
def evaluate_model_and_create_experiment_summary(
    p:EasyDict,
    jig:Vanilla_Train_Eval_Test_Jig,
    total_experiment_time_secs,
    ds:EasyDict,
    model
    )->dict:
    source_test_label_accuracy, source_test_label_loss = jig.test(ds.source.processed.test)
    target_test_label_accuracy, target_test_label_loss = jig.test(ds.target.processed.test)

    source_val_label_accuracy, source_val_label_loss = jig.test(ds.source.processed.val)
    target_val_label_accuracy, target_val_label_loss = jig.test(ds.target.processed.val)

    history = jig.get_history()

    total_epochs_trained = len(history["epoch_indices"])

    val_dl = wrap_in_dataloader(p, Sequence_Aggregator((ds.source.original.val, ds.target.original.val)))

    confusion = confusion_by_domain_over_dataloader(model, p.device, val_dl, forward_uses_domain=False)
    per_domain_accuracy = per_domain_accuracy_from_confusion(confusion)

    # Add a key to per_domain_accuracy for if it was a source domain
    for domain, accuracy in per_domain_accuracy.items():
        per_domain_accuracy[domain] = {
            "accuracy": accuracy,
            "source?": domain in p.source_domains
        }

    # Do an independent accuracy assesment JUST TO BE SURE!
    _source_test_label_accuracy = independent_accuracy_assesment(model, ds.source.processed.test)
    _target_test_label_accuracy = independent_accuracy_assesment(model, ds.target.processed.test)
    _source_val_label_accuracy = independent_accuracy_assesment(model, ds.source.processed.val)
    _target_val_label_accuracy = independent_accuracy_assesment(model, ds.target.processed.val)

    assert(_source_test_label_accuracy == source_test_label_accuracy)
    assert(_target_test_label_accuracy == target_test_label_accuracy)
    assert(_source_val_label_accuracy == source_val_label_accuracy)
    assert(_target_val_label_accuracy == target_val_label_accuracy)

    ###################################
    # Write out the results
    ###################################

    experiment = {
        "experiment_name": p.experiment_name,
        "parameters": p,
        "results": {
            "source_test_label_accuracy": source_test_label_accuracy,
            "source_test_label_loss": source_test_label_loss,
            "target_test_label_accuracy": target_test_label_accuracy,
            "target_test_label_loss": target_test_label_loss,
            "source_val_label_accuracy": source_val_label_accuracy,
            "source_val_label_loss": source_val_label_loss,
            "target_val_label_accuracy": target_val_label_accuracy,
            "target_val_label_loss": target_val_label_loss,
            "total_epochs_trained": total_epochs_trained,
            "total_experiment_time_secs": total_experiment_time_secs,
            "confusion": confusion,
            "per_domain_accuracy": per_domain_accuracy,
        },
        "history": history,
        "dataset_metrics": get_dataset_metrics(ds, "cnn"),
    }

    return experiment


###################################
# Write out the results
###################################
def write_results(p:EasyDict, experiment)->None:
    with open(p.EXPERIMENT_JSON_PATH, "w") as f:
        json.dump(experiment, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "-":
        parameters = json.loads(sys.stdin.read())
    elif len(sys.argv) == 1:
        parameters = base_parameters
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Unit Testing!")
        raise RuntimeError("Unimplemented")

    start_time_secs = time.time()

    p = parse_and_validate_parameters(parameters)

    if not os.path.exists(p.RESULTS_DIR):
        os.mkdir(p.RESULTS_DIR)

    set_rng(p)
    x_net = build_network(p)
    datasets = build_datasets(p)
    model = build_model(p, x_net)
    jig = train(
        p,
        model=model,
        ds=datasets
    )
    total_experiment_time_secs = time.time() - start_time_secs

    experiment = evaluate_model_and_create_experiment_summary(
        p,
        jig=jig,
        ds=datasets,
        total_experiment_time_secs=total_experiment_time_secs,
        model=model,
    )

    print("Source Test Label Accuracy:", experiment["results"]["source_test_label_accuracy"], "Target Test Label Accuracy:", experiment["results"]["target_test_label_accuracy"])
    print("Source Val Label Accuracy:", experiment["results"]["source_val_label_accuracy"], "Target Val Label Accuracy:", experiment["results"]["target_val_label_accuracy"])

    write_results(p, experiment)

    do_report(p.EXPERIMENT_JSON_PATH, p.LOSS_CURVE_PATH)