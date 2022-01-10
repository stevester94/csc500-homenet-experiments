#! /bin/sh
export PYTHONPATH=
cat << EOF | ./run.sh -
{
  "experiment_name": "cores_cnn_1",
  "lr": 0.001,
  "device": "cuda",
  "seed": 1984,
  "dataset_seed": 1337,
  "desired_classes": [
    "17-11.",
    "10-7.",
    "8-20.",
    "14-7.",
    "19-1.",
    "7-14.",
    "3-13.",
    "15-1.",
    "4-1.",
    "19-19.",
    "5-5.",
    "15-20.",
    "13-8.",
    "11-1.",
    "2-6.",
    "8-3.",
    "16-16.",
    "6-15."
  ],
  "batch_size": 128,
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
        "out_features": 18
      }
    }
  ],
  "NUM_LOGS_PER_EPOCH": 10,
  "RESULTS_DIR": "./results",
  "EXPERIMENT_JSON_PATH": "./results/experiment.json",
  "LOSS_CURVE_PATH": "./results/loss.png",
  "BEST_MODEL_PATH": "./results/best_model.pth",
  "num_examples_per_class_per_domain": 1000,
  "source_domains": [
    5
  ],
  "target_domains": [
    1,
    2,
    3,
    4
  ]
}
EOF