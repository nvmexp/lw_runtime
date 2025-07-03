#!/bin/bash

#sudo docker login gitlab-master.lwpu.com:5005

sudo docker build --no-cache -t gitlab-master.lwpu.com:5005/gputelecom/lwphytools .

sudo docker push gitlab-master.lwpu.com:5005/gputelecom/lwphytools

