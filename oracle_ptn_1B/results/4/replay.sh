#! /bin/sh
export PYTHONPATH=
cat << EOF | ./run.sh -
{
  "experiment_name": "oracle_ptn_1",
  "lr": 0.0001,
  "device": "cuda",
  "seed": 1337,
  "desired_classes_source": [
    "3123D52",
    "3123D65",
    "3123D79",
    "3123D80",
    "3123D54",
    "3123D70",
    "3123D7B",
    "3123D89",
    "3123D58",
    "3123D76",
    "3123D7D",
    "3123EFE",
    "3123D64",
    "3123D78",
    "3123D7E",
    "3124E4A"
  ],
  "desired_classes_target": [
    "3123D52",
    "3123D65",
    "3123D79",
    "3123D80",
    "3123D54",
    "3123D70",
    "3123D7B",
    "3123D89",
    "3123D58",
    "3123D76",
    "3123D7D",
    "3123EFE",
    "3123D64",
    "3123D78",
    "3123D7E",
    "3124E4A"
  ],
  "num_examples_per_class_per_domain_source": 1000,
  "num_examples_per_class_per_domain_target": 1000,
  "n_shot": 3,
  "n_way": 16,
  "n_query": 2,
  "train_k_factor": 1,
  "val_k_factor": 2,
  "test_k_factor": 2,
  "n_epoch": 100,
  "patience": 10,
  "criteria_for_best": "source",
  "normalize_source": false,
  "normalize_target": false,
  "x_net": [
    {
      "class": "nnReshape",
      "kargs": {
        "shape": [
          -1,
          1,
          2,
          256
        ]
      }
    },
    {
      "class": "Conv2d",
      "kargs": {
        "in_channels": 1,
        "out_channels": 256,
        "kernel_size": [
          1,
          7
        ],
        "bias": false,
        "padding": [
          0,
          3
        ]
      }
    },
    {
      "class": "ReLU",
      "kargs": {
        "inplace": true
      }
    },
    {
      "class": "BatchNorm2d",
      "kargs": {
        "num_features": 256
      }
    },
    {
      "class": "Conv2d",
      "kargs": {
        "in_channels": 256,
        "out_channels": 80,
        "kernel_size": [
          2,
          7
        ],
        "bias": true,
        "padding": [
          0,
          3
        ]
      }
    },
    {
      "class": "ReLU",
      "kargs": {
        "inplace": true
      }
    },
    {
      "class": "BatchNorm2d",
      "kargs": {
        "num_features": 80
      }
    },
    {
      "class": "Flatten",
      "kargs": {}
    },
    {
      "class": "Linear",
      "kargs": {
        "in_features": 20480,
        "out_features": 256
      }
    },
    {
      "class": "ReLU",
      "kargs": {
        "inplace": true
      }
    },
    {
      "class": "BatchNorm1d",
      "kargs": {
        "num_features": 256
      }
    },
    {
      "class": "Linear",
      "kargs": {
        "in_features": 256,
        "out_features": 256
      }
    }
  ],
  "NUM_LOGS_PER_EPOCH": 10,
  "RESULTS_DIR": "./results",
  "EXPERIMENT_JSON_PATH": "./results/experiment.json",
  "LOSS_CURVE_PATH": "./results/loss.png",
  "BEST_MODEL_PATH": "./results/best_model.pth",
  "source_domains": [
    2
  ],
  "target_domains": [
    8,
    14,
    20,
    26,
    32,
    38,
    44,
    50,
    56,
    62
  ]
}
EOF