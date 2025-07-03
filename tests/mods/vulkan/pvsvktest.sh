#!/bin/bash

set -euo pipefail

export DISPLAY=:0

run()
{
    echo
    echo ">>> ======================================================================"
    echo ">>> $@"
    local START_TIME=$SECONDS

    local EC=0
    "$@" || EC=$?
    echo ">>> Exit code: $EC"

    local TIME=$(($SECONDS - $START_TIME))
    echo ">>> Run time: $TIME seconds"
    return $EC
}

run ./vktest -r 1000
run ./vktest -r 1000 -t 21
run ./vktest -r 1000 -p 250
run ./vktest -r 1000 -p 250 -t 21
run ./ocgraphicstest -r 1000
run ./powerpulse -r 1000
