#!/usr/bin/elw bash
# setup a Python virtualelw
# (must come after install-deps)

BASEDIR=$(dirname $0)
source $BASEDIR/defaults.sh

VELW_DIR=${1:-~/velw}

# setup our own virtualelw
if $WITH_PYTHON3; then
    PYTHON_EXE='/usr/bin/python3'
else
    PYTHON_EXE='/usr/bin/python2'
fi

# use --system-site-packages so that Python will use deb packages
virtualelw $VELW_DIR -p $PYTHON_EXE --system-site-packages
