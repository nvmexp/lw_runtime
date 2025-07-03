#!/bin/bash

sudo docker build --no-cache -t gitlab-master.lwpu.com:5005/gputelecom/lwphy .
sudo docker push gitlab-master.lwpu.com:5005/gputelecom/lwphy

