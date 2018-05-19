# A Note About this Branch: Tensor Compiler Comparisons

This branch contains a pre-release of the code we used to generate performance
metrics comparing PlaidML with Tensor Comprehensions and TVM.

Currently we don't provide any help in setting those up, so in order to compare
against them, you'll need to have them installed in your virtualenv

*Feedback is very welcome, especially if we're using tc or tvm incorrectly*

## Methodology

As we developed support for each backend (tc, tvm, plaid), great care was taken to ensure the results we capture are accurate. `nvprof` was used to manually compare results (see below).

The relevant files are `frontend_ops.py`, `networks/ops/dense.py`, `networks/ops/conv2d.py`

Files of interest are `frontend_ops.py`, `networks/ops/dense.py`, `networks/ops/conv2d.py`

## Instructions
```
pip install plaidml-keras
<install TVM & Tensor Comprehensions>
python setup.py install
mkdir pbresults
```

### PlaidML
To use PlaidML, ensure you run `plaidml-setup` to choose a device
```
plaidml-setup
plaidbench --results=pbresults --blanket-run ops
```

### Tensor Comprehensions
Tensor Comprehensions has more options to control the autotuner. By default all autotuning is cached in `~/.plaidbench-tccache`
```
 --tc [--tc-cachedir --tc-at-generations, --tc-at-population,--tc-no-autotune]
```
Our results were generated with the default settings (13 gens, 13 pop), with a timeout of 1 hour (per op/batch_size)
```
plaidbench --results=pbresults --timeout 3600 --blanket-run ops --tc
```
Once you've autotuned to can run with `--tc-no-autotune` to recheck a particular op.

### TVM
TVM lets you pick its driver via `--tvm-driver`. Our results were generated with the default cuda backend.
```
plaidbench --results=pbresults --blanket-run ops --tc
```

When you're done you can look at the results and also auto generate some charts
```
pip install pandas seaborn matplotlib
python plaidplotter --path=pbresults --gflops=<peak gflops for your card>
```
Then open the `compare.html` file inside `pbresults`:
<img src="compare_example.png"></img>

### Using nvprof
You can use nvprof to validate the numbers plaidbench spits out:
`nvprof --profile-child-processes plaidbench -n 30 ops conv2d_vgg_lrg --cuda-profile`

### Other Options
`--no-large-ops` will exclude the heaviest ops, if things are taking too long.
Feedback very much appreciated



# Vertex.AI Machine Learning Benchmarks
Plaidbench measures the performance of machine-learning networks.

Plaidbench supports:

* Benchmarking across various frontends (the packages that provide ways to represent ML networks)

* Benchmarking across various backends (the packages used by the frontends to actually run the network).

Plaidbench was created to quantify the performance of [PlaidML](http://www.github.com/plaidml/plaidml) relative to other frameworks' backends across a variety of hardware, and to help determine which networks provide acceptable performance in various application deployment scenarios.



## Current Status

[![Build Status](https://travis-ci.org/plaidml/plaidbench.svg?branch=master)](https://travis-ci.org/plaidml/plaidbench)
[![Build status](https://ci.appveyor.com/api/projects/status/307lhqu7kp2m0j0v?svg=true)](https://ci.appveyor.com/project/earhart/plaidbench)

## Installation

To get the basic framework and command-line interface:

    pip install plaidbench

If you know which ML frontends you'll want to use, you can install their pre-requisites ahead of time:

    pip install plaidbench[keras]
    pip install plaidbench[onnx]

You can also install various ML backends -- for example,

    pip install plaidml-keras
    pip install tensorflow
    pip install caffe2
    pip install onnx-plaidml
    pip install onnx-tf

If you don't have a particular package installed, and you run benchmarks that require the package, Plaidbench will try to determine what needs to be installed and tell you how to install it.

If you're using PlaidML as a backend, you'll want to run `plaidml-setup` to configure it correctly for your hardware.

## Usage

Plaidbench provides a simple command-line interface; global flags are provided immediately, and subcommands are used to select the frontend framework and to provide framework-specific options.

For example, to benchmark [ShuffleNet](https://arxiv.org/abs/1707.01083) on [ONNX](https://onnx.ai/) using PlaidML, writing results to the directory `~/shuffle_results`, you can run:

    plaidbench --result ~/shuffle_results onnx --plaid shufflenet

For a complete overview of the supported global flags, use `plaidbench --help`; for the individual subcommand flags, specify `--help` with the subcommand (e.g. `plaidbench keras --help`).

## Supported Configurations

Plaidbench supports:

* Keras

  * Backends: PlaidML and Tensorflow

  * Networks: Inception-V3, ResNet50, Vgg16, Vgg19, Xception, and (in Keras 2.0.6 and later) MobileNet.

  * Training vs. Inference performance, and fp16 vs. fp32 performance.

* ONNX

  * Backends: PlaidML, Caffe2, and Tensorflow

  * Networks: AlexNet, DenseNet, Inception-V1, Inception-V2, Resnet50, ShuffleNet, SqueezeNet, Vgg16, and Vgg19.
