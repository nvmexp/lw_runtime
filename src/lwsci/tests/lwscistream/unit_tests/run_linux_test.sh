#!/bin/bash -e

UNIT_SKPD="NONE"

# Help on usage
if [[ "$1" == "--help" ]]; then
    echo "Usage:"
    echo "For single unit-test : $./run_linux_test.sh <unit name>"
    echo "For batch exelwtion  : $./run_linux_test.sh all [--skip <list of whitespace-seperated unit names to be skipped>]"
    exit 0
fi

# Check for correct args
if [[ "$#" -eq "1" ]]; then
    UNIT=$1
elif [[ "$#" -gt "1" ]]; then
    if [[ "$1" == "all" ]] && [[ "$2" == "--skip" ]] && [[ -n ${@:3} ]]; then
        UNIT=$1
        UNIT_SKPD="${@:3}"
    else
        echo "Incorrect usage. For help on usage: ./run_linux_test.sh --help"
        exit 1
    fi
else
    echo "Incorrect usage. For help on usage: ./run_linux_test.sh --help"
    exit 1
fi


# Enable unit-test build
export LWSCISTREAM_ENABLE_UNIT_TEST_BUILD=1

# Get the TEGRA_TOP folder path
cd ../../../../../../..
TEGRA_TOP=$(pwd)

# Export necessary paths
export QEMU_LD_PREFIX="${P4ROOT}/sw/embedded/external-prebuilt/hv/qemu-aarch64-ld-prefix/2.27"
export LD_LIBRARY_PATH="${TEGRA_TOP}/../QEMU_LIBS"

# Setup tmake elw
source tmake/scripts/elwsetup.sh
choose embedded-linux t186ref none release external aarch64


################### Build & run all unit tests ###################

if [ "$UNIT" == "all" ]
then
    UNIT=""

    # Get the list of all the available unit tests from umbrella makefile
    UNITS=$(grep "    tests/lwscistream/unit_tests/" ${TEGRA_TOP}/gpu/drv/drivers/lwsci/Makefile.umbrella.tmk | cut -c 1-33 --complement | xargs)

    # Build & run each unit test
    for i in ${UNITS}
    do
        # Search for unit tests to be skipped
        for j in ${UNIT_SKPD}
        do
            if [[ "${j}" == "${i}" ]]; then
                continue 2
            fi
        done

        UNIT+="${i} "

        UT_PATH="${TEGRA_TOP}/gpu/drv/drivers/lwsci/tests/lwscistream/unit_tests/${i}"
        cd ${UT_PATH}
        tmm

        OUT_PATH="${TEGRA_TOP}/out/embedded-linux-t186ref-release/lwpu/gpu/drv/drivers/lwsci/tests/lwscistream/unit_tests/${i}-embedded-linux_64"
        qemu-aarch64 ${OUT_PATH}/test_lwscistream_${i}
    done

    echo ""
    echo "[=============================]"
    echo "[        BUILD SUMMARY        ]"
    echo "[=============================]"
    for i in ${UNIT}
    do
        printf "[ %-18s\t   OK ]\n" "${i}"
    done
    for i in ${UNIT_SKPD}
    do
        printf "[ %-18s\t SKIP ]\n" "${i}"
    done
    echo "[-----------------------------]"
    echo "Builds & run on qemu successful!"
    echo ""
    exit 0
fi

################### Build & run selected unit test ###################

# Build the selected unit-test
UT_PATH="${TEGRA_TOP}/gpu/drv/drivers/lwsci/tests/lwscistream/unit_tests/${UNIT}"
cd ${UT_PATH}
tmm

# Run the selected unit test on qemu
OUT_PATH="${TEGRA_TOP}/out/embedded-linux-t186ref-release/lwpu/gpu/drv/drivers/lwsci/tests/lwscistream/unit_tests/${UNIT}-embedded-linux_64"
qemu-aarch64 ${OUT_PATH}/test_lwscistream_${UNIT}

echo "${UNIT} unit-test build & run on qemu successful!"
exit 0

