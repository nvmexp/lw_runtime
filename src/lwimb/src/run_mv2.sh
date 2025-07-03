#!/bin/bash

export MPI_HOME=/ivylogin/home/spotluri/MPI/mvapich2-2.0-lwca-gnu-install
export LWDA_HOME=/usr/local/lwca-6.5

TESTLIST="PingPong PingPing Sendrecv Exchange Reduce Allreduce Reduce_scatter Allgather Allgatherv Gather Gatherv Scatter Scatterv Alltoall Alltoallv Bcast Barrier"

make clean
make -f make_mpich IMB-MPI1
sleep 2

${MPI_HOME}/bin/mpirun -f hfile -ppn 1 -elw MV2_USE_LWDA 1 ./IMB-MPI1 -lwca 1 $TESTLIST
