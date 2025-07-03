#!/bin/bash

# Usage: run.sh <lwtensorTest> <lwtensor_root> <prefix>

set -euxo pipefail

"$1" -file "$2"/test/regression/contraction/mxnet.sh > "$3".mxnet.out 2>&1
"$1" -file "$2"/test/regression/contraction/random1000.sh > "$3".random1000.out 2>&1 2>&1
