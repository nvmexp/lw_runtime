#!/bin/bash
export LWPHYTOOLS_PATH=/root/share/lwPHYTools
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install -DLWPHYCONTROLLER=1
make -j36
make install
