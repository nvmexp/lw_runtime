#!/bin/bash

if [ "$#" -eq "2" ]; then
    TARGET=$1
    exelwtable=$2
else
    echo "$0: Incorrect number of arguments!"
    exit 1
fi
HOST=10.0.0.1
TEST=$(basename $exelwtable)
DIR=$(dirname $exelwtable)
echo "Mounting on target and running the test..."
sshpass -p root ssh -tt root@${TARGET} "cd ${DIR}; ./${TEST}; sync; sleep 0.5"
echo "Exelwtion complete"
