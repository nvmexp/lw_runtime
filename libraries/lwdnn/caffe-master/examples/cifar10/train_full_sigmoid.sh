#!/usr/bin/elw sh

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_sigmoid_solver.prototxt

