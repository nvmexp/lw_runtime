#!/usr/bin/elw sh

./build/tools/caffe train \
    --solver=examples/solvers/adam/solver.prototxt -gpu=all \
    2>&1 | tee examples/solvers/adam/adam.log
