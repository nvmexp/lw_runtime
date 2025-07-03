#!/bin/bash

set -euxo pipefail

# Usage: analyze.sh <dryRunBinary> <lwtensor_root>

for bench in mxnet random1000; do
LWTENSOR_GROUND_TRUTH=1_base.$bench.out LWTENSOR_DISABLE_LWBLAS=1 "$1" -file "$2"/test/regression/contraction/${bench}-gett.sh > 2_nomod.$bench.out
LWTENSOR_GROUND_TRUTH=1_base.$bench.out LWTENSOR_DISABLE_LWBLAS=1 ILP_MU=0 ILP_LO=1 REUSE_MU=0 REUSE_LO=1 "$1" -file "$2"/test/regression/contraction/${bench}-gett.sh > 3_disable.$bench.out
LWTENSOR_GROUND_TRUTH=1_base.$bench.out LWTENSOR_DISABLE_LWBLAS=1 ILP_MU=0 ILP_LO=1 "$1" -file "$2"/test/regression/contraction/${bench}-gett.sh > 4_disable_ilp.$bench.out
LWTENSOR_GROUND_TRUTH=1_base.$bench.out LWTENSOR_DISABLE_LWBLAS=1 REUSE_MU=0 REUSE_LO=1 "$1" -file "$2"/test/regression/contraction/${bench}-gett.sh > 5_disable_reuse.$bench.out
LWTENSOR_GROUND_TRUTH=1_base.$bench.out LWTENSOR_DISABLE_LWBLAS=1 LWTENSOR_HEUR=old "$1" -file "$2"/test/regression/contraction/${bench}-gett.sh > 6_old.$bench.out
LWTENSOR_GROUND_TRUTH=1_base.$bench.out LWTENSOR_DISABLE_LWBLAS=1 LWTENSOR_HEUR=paul "$1" -file "$2"/test/regression/contraction/${bench}-gett.sh > 7_paul.$bench.out
done
