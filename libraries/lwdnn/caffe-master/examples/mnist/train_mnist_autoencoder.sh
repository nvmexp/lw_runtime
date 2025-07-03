#!/usr/bin/elw sh

./build/tools/caffe train \
  --solver=examples/mnist/mnist_autoencoder_solver.prototxt
