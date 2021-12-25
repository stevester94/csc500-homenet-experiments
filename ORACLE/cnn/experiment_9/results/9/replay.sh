#! /bin/sh
export PYTHONPATH=
cat << EOF | ./run.sh -
{
  "experiment_name": "oracle_cnn_experiment_9",
  "lr": 0.0001,
  "n_epoch": 10,
  "batch_size": 256,
  "patience": 10,
  "seed": 18081994,
  "device": "cuda",
  "desired_serial_numbers": [
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
  "window_length": 256,
  "desired_runs": [
    1
  ],
  "num_examples_per_device": 75000,
  "max_cache_items": 4500000.0,
  "x_net": [
    {
      "class": "nnReshape",
      "kargs": {
        "shape": [
          -1,
          1,
          2,
          128
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
        "in_features": 10240,
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
  "source_domains": [
    2,
    38,
    62
  ],
  "target_domains": [
    8,
    14,
    20,
    26,
    32,
    44,
    50,
    56
  ]
}
EOF