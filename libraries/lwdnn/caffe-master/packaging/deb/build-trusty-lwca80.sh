#!/bin/bash
set -e

export DOCKER_BASE=caffe-lw-debuild-trusty-lwda80
docker build --pull -t $DOCKER_BASE -f Dockerfile.trusty-lwda80 .
./build.sh
