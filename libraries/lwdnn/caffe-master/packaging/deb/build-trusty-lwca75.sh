#!/bin/bash
set -e

export DOCKER_BASE=caffe-lw-debuild-trusty-lwda75
docker build --pull -t $DOCKER_BASE -f Dockerfile.trusty-lwda75 .
./build.sh
