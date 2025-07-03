#!/usr/bin/elw bash
# CMake configuration

mkdir -p build
cd build

#Travis doesn't have LWCA driver installed
ARGS="-DCMAKE_BUILD_TYPE=Release -DBLAS=Open -DNO_LWML=On"

if $WITH_PYTHON3 ; then
  ARGS="$ARGS -Dpython_version=3"
fi

if $WITH_IO ; then
  ARGS="$ARGS -DUSE_LMDB=On -DUSE_LEVELDB=On"
else
  ARGS="$ARGS -DUSE_LMDB=Off -DUSE_LEVELDB=Off"
fi

if $WITH_LWDA ; then
  # Only build SM50
  ARGS="$ARGS -DLWDA_ARCH_NAME=Manual -DLWDA_ARCH_BIN=\"50\" -DLWDA_ARCH_PTX=\"\""
fi

if $WITH_LWDNN ; then
  ARGS="$ARGS -DUSE_LWDNN=On"
else
  ARGS="$ARGS -DUSE_LWDNN=Off"
fi

cmake .. $ARGS

