#!/bin/bash

sudo docker build -t gitlab-master.lwpu.com:5005/gputelecom/dpdk .

sudo docker push gitlab-master.lwpu.com:5005/gputelecom/dpdk
