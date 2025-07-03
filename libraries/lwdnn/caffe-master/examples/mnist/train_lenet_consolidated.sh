#!/usr/bin/elw sh

./build/tools/caffe train \
  --solver=examples/mnist/lenet_consolidated_solver.prototxt
