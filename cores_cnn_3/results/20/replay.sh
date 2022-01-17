#! /bin/sh
export PYTHONPATH=
cat << EOF | ./run.sh -
{
  "experiment_name": "cores_cnn_3",
  "lr": 0.001,
  "device": "cuda",
  "seed": 1337,
  "desired_classes": [
    "1-10.",
    "1-11.",
    "1-15.",
    "1-16.",
    "1-17.",
    "1-18.",
    "1-19.",
    "10-4.",
    "10-7.",
    "11-1.",
    "11-14.",
    "11-17.",
    "11-20.",
    "11-7.",
    "13-20.",
    "13-8.",
    "14-10.",
    "14-11.",
    "14-14.",
    "14-7.",
    "15-1.",
    "15-20.",
    "16-1.",
    "16-16.",
    "17-10.",
    "17-11.",
    "17-2.",
    "19-1.",
    "19-16.",
    "19-19.",
    "19-20.",
    "19-3.",
    "2-10.",
    "2-11.",
    "2-17.",
    "2-18.",
    "2-20.",
    "2-3.",
    "2-4.",
    "2-5.",
    "2-6.",
    "2-7.",
    "2-8.",
    "3-13.",
    "3-18.",
    "3-3.",
    "4-1.",
    "4-10.",
    "4-11.",
    "4-19.",
    "5-5.",
    "6-15.",
    "7-10.",
    "7-14.",
    "8-18.",
    "8-20.",
    "8-3.",
    "8-8."
  ],
  "batch_size": 128,
  "n_epoch": 100,
  "patience": 10,
  "criteria_for_best": "target",
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
        "out_features": 58
      }
    }
  ],
  "NUM_LOGS_PER_EPOCH": 10,
  "RESULTS_DIR": "./results",
  "EXPERIMENT_JSON_PATH": "./results/experiment.json",
  "LOSS_CURVE_PATH": "./results/loss.png",
  "BEST_MODEL_PATH": "./results/best_model.pth",
  "num_examples_per_class_per_domain": 100,
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