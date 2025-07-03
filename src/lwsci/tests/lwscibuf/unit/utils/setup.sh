#!/bin/bash -e

## Set unit name
export UNIT=LWSCIBUF_UTILS

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. $SCRIPTDIR/../../../unit/common.sh $@
