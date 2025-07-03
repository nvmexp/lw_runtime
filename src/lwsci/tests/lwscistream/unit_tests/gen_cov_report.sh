#!/bin/bash

start=`date +%s`

# Prompt for user input
clear
echo ""
echo "****************************************************************** Prerequisites ******************************************************************"
echo "* IPP repos need to be cloned at the location \"\$TEGRA_TOP/../ipp-framework\" using the below commands:"
echo "   - git clone ssh://git-master.lwpu.com:12001/tools/vectorcast/core -b rel-33"
echo "   - git clone ssh://git-master.lwpu.com:12001/tools/vectorcast/data -b rel-33"
echo "* \"\${P4ROOT}/sw/tools/VectorCAST/linux/vectorcast_elw.sh\" script need to be synced in P4."
echo "* Set the elw variable \"\$VECTORCAST_DIR\" in ~/.profile to point to the VectorCAST root directory of latest tool version in P4."
echo "   eg: export VECTORCAST_DIR=\$P4ROOT/sw/tools/VectorCAST/linux/20XXspX"
echo "* Insall qemu with the command \"sudo apt install qemu-user\"."
echo "* Aarch64 ELF interpreter libraries required by quemu-aarch64 need to be synced in P4 at the path \"\${P4ROOT}/sw/embedded/external-prebuilt/hv/qemu-aarch64-ld-prefix\"."
echo "* Copy the below libs synced in P4 to the location \"\$TEGRA_TOP/../QEMU_LIBS\" using the below commands:"
echo "   - cp \${P4ROOT}/sw/mobile/tools/linux/linaro/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc/lib/libgcc_s.so* \$TEGRA_TOP/../QEMU_LIBS"
echo "   - cp \${P4ROOT}/sw/mobile/tools/linux/linaro/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc/lib/libstdc++* \$TEGRA_TOP/../QEMU_LIBS"
echo "   - cp \${P4ROOT}/sw/mobile/tools/linux/linaro/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc/lib/librt* \$TEGRA_TOP/../QEMU_LIBS"
echo "***************************************************************************************************************************************************"
echo ""
echo ""
read -p "Enter the unit name: " UNIT

# Select project ID in .json based on <unit name> arg
case ${UNIT} in

    producer)
        P_ID="1"
    ;;

    trackcount)
        P_ID="2"
    ;;

    trackarray)
        P_ID="3"
    ;;

    safeconnection)
        P_ID="4"
    ;;

    consumer)
        P_ID="5"
    ;;

    pool)
        P_ID="6"
    ;;

    queue)
        P_ID="7"
    ;;

    block)
        P_ID="8"
    ;;

    packet)
        P_ID="9"
    ;;

    multicast)
        P_ID="10"
    ;;

    lwscistreamcommon)
        P_ID="11"
    ;;

    lwsciwrap)
        P_ID="12"
    ;;

    publicAPI)
        P_ID="13"
    ;;

    ipcsrc)
        P_ID="14"
    ;;

    ipcdst)
        P_ID="15"
    ;;

    limiter)
        P_ID="16"
    ;;

    ipccomm)
        P_ID="17"
    ;;

    *)
        echo "Error: Invalid unit selected."
        echo ""
        exit 1
    ;;
esac

# Get the TEGRA_TOP folder path
cd ../../../../../../..
TEGRA_TOP=$(pwd)

# Export necessary paths
export QEMU_LD_PREFIX="${P4ROOT}/sw/embedded/external-prebuilt/hv/qemu-aarch64-ld-prefix/2.27"
export LD_LIBRARY_PATH="${TEGRA_TOP}/../QEMU_LIBS"

# Test binary path
BIN_DIR=${TEGRA_TOP}/out/embedded-linux-t186ref-release/lwpu/gpu/drv/drivers/lwsci/tests/lwscistream/unit_tests/${UNIT}-embedded-linux_64

# Enable unit-test build
export LWSCISTREAM_ENABLE_UNIT_TEST_BUILD=1

# Setup tmake elw
source tmake/scripts/elwsetup.sh
choose embedded-linux t186ref none release external aarch64

# Setup VectorCAST elw
source ${P4ROOT}/sw/tools/VectorCAST/linux/vectorcast_elw.sh

# Pre clean up
cd ${TEGRA_TOP}/out
rm -rf vcshell.db vcshell.txt vcast_build_elw.sh vectorcast/LwSciStream/test_lwscistream_${UNIT}
rm -rf ${BIN_DIR}
rm -f /tmp/*.DAT

# Generate vcshell.db
export LW_BUILD_CONFIGURATION_IS_VCAST=1
cd ${TEGRA_TOP}/gpu/drv/drivers/lwsci/tests/lwscistream/unit_tests/${UNIT}
command=$(show tmm -B)
echo "export LD_PRELOAD=${VECTORCAST_DIR}/recorder/lib64/\$LD_PRELOAD" > vcshell_cmd.sh
echo ${command/*COMMAND LINE:/} >> vcshell_cmd.sh
chmod 777 vcshell_cmd.sh
mkdir -p ${TEGRA_TOP}/out
vcshell --db=${TEGRA_TOP}/out/vcshell.db --echo --out=${TEGRA_TOP}/out/vcshell.txt ./vcshell_cmd.sh

# Instrument component project
cd ${TEGRA_TOP}/../ipp-framework/core
source tools/createBuildElwFile.sh
python tools/command.py --json ${TEGRA_TOP}/gpu/drv/drivers/lwsci/tests/lwscistream/unit_tests/vectorcast/json/lwscistream_linux_aarch64.json -i -p ${P_ID}

# Build instrumented binary
cd ${TEGRA_TOP}/gpu/drv/drivers/lwsci/tests/lwscistream/unit_tests/${UNIT}
tmm

# Run the binary and get instrumented data dump
qemu-aarch64 ${BIN_DIR}/test_lwscistream_${UNIT}

# Get coverage data  and report from the data dump
cd ${TEGRA_TOP}/out/vectorcast/LwSciStream/test_lwscistream_${UNIT}/vcast-workarea/vc_coverage
clicast -e test_lwscistream_${UNIT}_coverage Cover Result Add /tmp/*.DAT unit
clicast -e test_lwscistream_${UNIT}_coverage Cover Report Aggregate
cp test_lwscistream_${UNIT}_coverage_aggregate_coverage_report.html ${TEGRA_TOP}/gpu/drv/drivers/lwsci/tests/lwscistream/unit_tests

# Post clean up
cd ${TEGRA_TOP}/gpu/drv/drivers/lwsci/tests/lwscistream/unit_tests/${UNIT}
rm -rf vcdebug vcshell_cmd.sh CCAST.CFG
cd ${TEGRA_TOP}/../ipp-framework/core
rm -f instrumentation_log.txt tmp.json vectorcast_build_debug.log vectorcast_build_info.log

echo "### ${UNIT} coverage report generated successfully. Runtime: $((($(date +%s)-$start)/60)) min ###"
exit 0
