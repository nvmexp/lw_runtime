TAG=""
GPU=gv100
#LWTENSOR_TEST_VERBOSE=1 LWTENSOR_DISABLE_LWBLAS=1 ../../build/bin/lwtensorTest -d0 -Pas -Pbs -Pcs -Pcomps -file ./contraction/rand300.sh  > ${GPU}_ssss_tc300_${TAG}.log&
#LWTENSOR_TEST_VERBOSE=1 LWTENSOR_DISABLE_LWBLAS=1 ../../build/bin/lwtensorTest -d1 -Pas -Pbs -Pcs -Pcompt -file ./contraction/rand300.sh  > ${GPU}_ssst_tc300_${TAG}.log&
#LWTENSOR_TEST_VERBOSE=1 LWTENSOR_DISABLE_LWBLAS=1 ../../build/bin/lwtensorTest -d2 -Pah -Pbh -Pch -Pcomph -file ./contraction/rand300.sh  > ${GPU}_hhhh_tc300_${TAG}.log&
#LWTENSOR_TEST_VERBOSE=1 LWTENSOR_DISABLE_LWBLAS=1 ../../build/bin/lwtensorTest -d3 -Pac -Pbc -Pcc -Pcompt -file ./contraction/rand300.sh  > ${GPU}_ccct_tc300_${TAG}.log&
#LWTENSOR_TEST_VERBOSE=1 LWTENSOR_DISABLE_LWBLAS=1 ../../build/bin/lwtensorTest -d4 -Pad -Pbd -Pcd -Pcompd -file ./contraction/rand300.sh  > ${GPU}_dddd_tc300_${TAG}.log&
#LWTENSOR_TEST_VERBOSE=1 LWTENSOR_DISABLE_LWBLAS=1 ../../build/bin/lwtensorTest -d5 -Pad -Pbd -Pcd -Pcompd -file ./elementwise/rand300_ew.sh  > ${GPU}_dddd_ew300_${TAG}.log&
#LWTENSOR_TEST_VERBOSE=1 LWTENSOR_DISABLE_LWBLAS=1 ../../build/bin/lwtensorTest -d6 -Pas -Pbs -Pcs -Pcomps -file ./elementwise/rand300_ew.sh  > ${GPU}_ssss_ew300_${TAG}.log&
#LWTENSOR_TEST_VERBOSE=1 LWTENSOR_DISABLE_LWBLAS=1 ../../build/bin/lwtensorTest -d7 -Pah -Pbh -Pch -Pcomph -file ./elementwise/rand300_ew.sh  > ${GPU}_hhhh_ew300_${TAG}.log&
LWTENSOR_TEST_VERBOSE=1 LWTENSOR_DISABLE_LWBLAS=1 ../../build/bin/lwtensorTest -d5 -Pad -Pbd -Pcd -Pcompd -file ./reduction/red300.sh  > ${GPU}_dddd_red300_${TAG}.log&
LWTENSOR_TEST_VERBOSE=1 LWTENSOR_DISABLE_LWBLAS=1 ../../build/bin/lwtensorTest -d6 -Pas -Pbs -Pcs -Pcomps -file ./reduction/red300.sh  > ${GPU}_ssss_red300_${TAG}.log&
LWTENSOR_TEST_VERBOSE=1 LWTENSOR_DISABLE_LWBLAS=1 ../../build/bin/lwtensorTest -d7 -Pah -Pbh -Pch -Pcomph -file ./reduction/red300.sh  > ${GPU}_hhhh_red300_${TAG}.log&
