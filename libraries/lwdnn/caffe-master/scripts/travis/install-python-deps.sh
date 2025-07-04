#!/usr/bin/elw bash
# install extra Python dependencies
# (must come after setup-velw)

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

if ! $WITH_PYTHON3 ; then
  # Python2
  pip install Pillow
else
  # Python3
  pip install --pre protobuf==3.0.0b3
  pip install pydot
fi
