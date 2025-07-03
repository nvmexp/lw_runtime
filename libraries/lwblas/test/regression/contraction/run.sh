#! /bin/bash

numCachelines=0
numRuns=0
incCount=0
algo=-1

######### Don't modify below this line #####################

LWTENSOR_BIN_DIR=${LWTENSOR_ROOT}/build/bin/
timestamp=`date +%m_%d_%y_%H_%M_%S`
host=`hostname`
commit=`git rev-parse HEAD`
GPU=$1
types="dddd ddds ssss hhhh sssh"

counter=0
for type in ${types}
do
   if ((${counter} == ${GPU}))
   then
       #for benchmark in peps_nx8_d4_chi8 gemm_sq_nt gemm_k_nt qFlex rand1000 mxnet
       for benchmark in rand1000
       do
           echo $benchmark
           outputFile=${timestamp}_${benchmark}_${type}_${host}_${numRuns}_${incCount}_${host}_${commit:0:8}_${GPU}.dat
           rm -f ${outputFile}
           touch ${outputFile}
       
           rm -f ${benchmark}_tmp.sh
           taskset -c ${GPU} ${LWTENSOR_BIN_DIR}/lwtensorTest -d${GPU} -disableVerify -Pa${type:0:1} -Pb${type:1:1} -Pc${type:2:1} -Pcomp${type:3:1} ${args} -file ${benchmark}.sh > ${outputFile}
           rm -f ${benchmark}_tmp.sh
       
           # analyze data with contraction.py
           #python3 ../misc/analyzeBenchmarks.py ${outputFile}
           mv ${outputFile} ${LWTENSOR_ROOT}/test/regression/contraction/data/
       done
   fi
   let counter+=1
done

