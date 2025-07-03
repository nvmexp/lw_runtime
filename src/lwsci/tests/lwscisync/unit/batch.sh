#!/bin/bash -e

run()
{
    echo EXECUTE UNIT $1
    $SCRIPTDIR/$1/setup.sh > $1.log 2>&1
    echo UNIT $1 exelwtion status: $?
}

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Run unit test suites one-by-one
run attribute_core
run attribute_reconcile
run attribute_transport
run core
run cpu_wait_context
run fence
run header_core
run ipc_table
run module
run object_core
run object_external
run primitives
run rm_backend

# TODO: gather HTML reports and process them to extract summary for all units
# and present it in a colwenient way
