#!/bin/bash -e

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

run()
{
    echo EXECUTE UNIT $1
    $SCRIPTDIR/$1/setup.sh > $1.log 2>&1
    echo UNIT $1 exelwtion status: $?
}

# Run unit test suites one-by-one
run attribute_constraint
run attribute_core
run attribute_reconcile_unit
run common_allocator_abstraction
run common_constraint_library
run ipc_table
run module
run object_core
run tegra_common_interface
run tegra_constraint_library
run tegra_device
run tegra_sysmem_interface
run tegra_transport
run transport_core
run utils

# TODO: gather HTML reports and process them to extract summary for all units
# and present it in a colwenient way