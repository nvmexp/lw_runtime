#!/usr/bin/elw bash
# test the project

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

if $WITH_LWDA ; then
  echo "Skipping tests for LWCA build"
  exit 0
fi

if ! $WITH_CMAKE ; then
  make -j"$(nproc)"
  make runtest
  make pytest
else
  cd build
  make -j"$(nproc)"
  make runtest
  make pytest
fi
