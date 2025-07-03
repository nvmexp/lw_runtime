#!/bin/bas

set -euxo pipefail

build="build_precision"
result="result_precision"
benchmarks="test/regression/rand1000_ew.sh test/regression/easy_ew.sh"

precisionName="$1"
shift

precision="-Pa${precisionName:0:1} -Pb${precisionName:1:1} -Pc${precisionName:2:1} -Pcomp${precisionName:3:1}"

step=0

source /usr/share/modules/init/bash

module load lwca/11.1.52-internal
mkdir -p "${build}"
mkdir -p "${result}"
python3 misc/generateElementwise.py
pushd "${build}"
cmake -DDEVELOP=ON -DCMAKE_BUILD_TYPE=ON -DLWTENSOR_EXPOSE_INTERNAL=ON ..
make -j99
popd

while true; do
   LWTENSOR_GENERATE_TARGET="${precisionName},${step}" python3 misc/generateElementwise.py
   pushd "${build}"
   make -j99
   popd
   for benchmark in $benchmarks; do
       benchmarkName="${benchmark##*/}"
       benchmarkName="${benchmarkName%%.sh}"
       "${build}/bin/lwtensorTest" ${precision} -file "${benchmark}" -disableVerify -numRuns1 > "${result}/${precisionName}_${benchmarkName}_${step}.out"
   done
   step=$((step+1))
done
