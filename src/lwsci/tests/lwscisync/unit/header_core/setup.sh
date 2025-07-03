#!/bin/bash -e

## Set unit name
export UNIT=LWSCISYNC_HEADER_CORE

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. $SCRIPTDIR/../../../unit/common.sh $@
