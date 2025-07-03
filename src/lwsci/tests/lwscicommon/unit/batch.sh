#!/bin/bash

run()
{
    echo EXECUTE UNIT $1
    $SCRIPTDIR/$1/setup.sh > $1.log 2>&1
    echo UNIT $1 exelwtion status: $?
}

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Run unit test suites one-by-one
run header_platform_utilities
run lwscicommon_transportutils
run objref
run platform_utilities

# TODO: gather HTML reports and process them to extract summary for all units
# and present it in a colwenient way