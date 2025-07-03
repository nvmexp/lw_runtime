#!/bin/bash

DATA_TYPE=$1
ARCH=$2
SRC_FILE="src/tensorContraction_sm${ARCH}_${DATA_TYPE}.lw"

cd build
COMPILE_COMMAND=$(make -n -B | grep "lwcc.*${SRC_FILE}")
eval "${COMPILE_COMMAND} -keep --ptxas-options=-dm,-v 2>&1"
cd ..
