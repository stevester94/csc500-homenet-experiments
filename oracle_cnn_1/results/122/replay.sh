#! /bin/sh
export PYTHONPATH=
cat << EOF | ./run.sh -
{
  "experiment_name": "oracle_cnn_1",
  "lr": 0.0001,
  "n_epoch": 100,
  "batch_size": 64,
  "patience": 10,
  "seed": 1984,
  "dataset_seed": 1337,
  "device": "cuda",
  "desired_classes": [
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
  "window_stride": 50,
  "window_length": 512,
  "desired_runs": [
    1
  ],
  "max_cache_items": 4500000.0,
  "criteria_for_best": "source",
  "normalize_source": false,
  "normalize_target": false,
  "NUM_LOGS_PER_EPOCH": 10,
  "RESULTS_DIR": "./results",
  "EXPERIMENT_JSON_PATH": "./results/experiment.json",
  "LOSS_CURVE_PATH": "./results/loss.png",
  "BEST_MODEL_PATH": "./results/best_model.pth",
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
        "out_features": 16
      }
    }
  ],
  "num_examples_per_class_per_domain": 1000,
  "source_domains": [
    50
  ],
  "target_domains": [
    2,
    8,
    14,
    20,
    26,
    32,
    38,
    44,
    56,
    62
  ]
}
EOF