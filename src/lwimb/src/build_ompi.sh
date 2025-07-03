#!/bin/bash

export MPI_HOME=/usr/local/openmpi
export LWDA_HOME=/usr/local/lwca

make clean
make -f make_ompi
sleep 2

