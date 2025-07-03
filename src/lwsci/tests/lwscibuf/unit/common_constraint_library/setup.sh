#!/bin/bash -e

## Set unit name
export UNIT=LWSCIBUF_CONSTRAINT_LIB_COMMON

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. $SCRIPTDIR/../../../unit/common.sh $@
