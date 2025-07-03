#!/bin/bash

# Usage: result.sh <lwtensor_root>

for n in 2 3 4 5 7; do
    echo $n
    for b in mxnet random1000; do
      python3 "$1"/test/regression/contraction2.py  ${n}_*.$b.out 6_*.$b.out | grep avg 
    done
done
