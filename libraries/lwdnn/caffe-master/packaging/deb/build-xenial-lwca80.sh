#!/bin/bash
set -e

export DOCKER_BASE=caffe-lw-debuild-xenial-lwda80
docker build --pull -t $DOCKER_BASE -f Dockerfile.xenial-lwda80 .
./build.sh
