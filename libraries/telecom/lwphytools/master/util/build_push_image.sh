#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR/../../
ls h5/*.h5 &>/dev/null
if [ $? -ne 0 ]; then echo 'Please put .h5 files in LWRAN/h5/'; exit; fi

sudo docker build -f lwPHYTools/util/Dockerfile.gen -t gitlab-master.lwpu.com:5005/gputelecom/dpdk/5gphy-gen . \
&& sudo docker push gitlab-master.lwpu.com:5005/gputelecom/dpdk/5gphy-gen

#sudo docker build -f lwPHYTools/util/Dockerfile.gen -t gitlab-master.lwpu.com:5005/gputelecom/dpdk/5gphy-generator .

sudo docker build -f lwPHYTools/util/Dockerfile.recv -t gitlab-master.lwpu.com:5005/gputelecom/dpdk/5gphy-recv . \
&& sudo docker push gitlab-master.lwpu.com:5005/gputelecom/dpdk/5gphy-recv

