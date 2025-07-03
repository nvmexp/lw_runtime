#!/bin/bash

export SUMMARY_LOG_DEFAULT="summarylog.txt"
export SUMMARY_LOG_TEMP="summarylogtemp.txt"
export SUMMARY_LOG="${SUMMARY_LOG_DEFAULT}"

cd ../../
lwsci_dir=`pwd`
outputdir="_out/Linux_amd64_release"

# set LD_LIBRARY_PATH to the library location
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${lwsci_dir}/lwscisync/${outputdir}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${lwsci_dir}/lwscistream/${outputdir}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${lwsci_dir}/lwsciipc/${outputdir}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${lwsci_dir}/lwscibuf/${outputdir}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${lwsci_dir}/lwscievent/${outputdir}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${lwsci_dir}/lwscicommon/${outputdir}

# set PATH containing the test binaries
export PATH=${PATH}:${lwsci_dir}/tests/lwscisync/api/${outputdir}
export PATH=${PATH}:${lwsci_dir}/tests/lwscistream/component_tests/${outputdir}
export PATH=${PATH}:${lwsci_dir}/tests/lwscibuf/api/${outputdir}

# copy lwsciipc.cfg to /etc
cp ${lwsci_dir}/lwsciipc/lwsciipc_dvs.cfg /etc/lwsciipc.cfg

cd - > /dev/NULL

# List of tests
LWSCIBUF_TEST=test_lwscibuf_api
LWSCISYNC_TEST=test_lwscisync_api
LWSCISTREAM_TEST=test_lwscistream_api


## Logging
rm -f ${SUMMARY_LOG}

function log {
    echo "$@" >> ${SUMMARY_LOG}
}

num_tests=3
num_success=0

# Enabling lwpu-persistence mode for faster test exelwtion
OUTPUT=`lwpu-smi -pm 1`

log "Running LwSciBuf tests..."
log ""
OUTPUT=`${LWSCIBUF_TEST} > ${SUMMARY_LOG_TEMP} 2>&1`
# 'OUTPUT=' avoids crashing the dvs script if the above test crashes due to SegFault, Abort
if [ $? -eq 0 ]; then
# $? == 0 means test returned EXIT_SUCCESS, so now test for gtest ASSERT_ failures
    if [ $(cat ${SUMMARY_LOG_TEMP} | grep ": Failure" | wc -l) -eq 0 ]; then
        # When ASSERT_ fails it will print ":__LINE__: Failure"
        let num_success=num_success+1
    fi
fi
cat ${SUMMARY_LOG_TEMP} >> ${SUMMARY_LOG}

log ""
log "LwScibuf tests ended."

log "Running LwSciSync tests..."
log ""
OUTPUT=`${LWSCISYNC_TEST} > ${SUMMARY_LOG_TEMP} 2>&1`
# 'OUTPUT=' avoids crashing the dvs script if the above test crashes due to SegFault, Abort
if [ $? -eq 0 ]; then
# $? == 0 means test returned EXIT_SUCCESS, so now test for gtest ASSERT_ failures
    if [ $(cat ${SUMMARY_LOG_TEMP} | grep ": Failure" | wc -l) -eq 0 ]; then
        # When ASSERT_ fails it will print ":__LINE__: Failure"
        let num_success=num_success+1
    fi
fi
cat ${SUMMARY_LOG_TEMP} >> ${SUMMARY_LOG}

log ""
log "LwSciSync tests ended."

log "Running LwSciStream tests..."
log ""
OUTPUT=`${LWSCISTREAM_TEST} > ${SUMMARY_LOG_TEMP} 2>&1`
# 'OUTPUT=' avoids crashing the dvs script if the above test crashes due to SegFault, Abort
if [ $? -eq 0 ]; then
# $? == 0 means test returned EXIT_SUCCESS, so now test for gtest ASSERT_ failures
    if [ $(cat ${SUMMARY_LOG_TEMP} | grep ": Failure" | wc -l) -eq 0 ]; then
        # When ASSERT_ fails it will print ":__LINE__: Failure"
        let num_success=num_success+1
    fi
fi
cat ${SUMMARY_LOG_TEMP} >> ${SUMMARY_LOG}

log ""
log "LwSciStream tests ended."

## All done
## Print out the score
log "Number of tests: ${num_tests}"
log "Passed tests:    ${num_success}"
log ""

let score=100*${num_success}/${num_tests}
log "Score: ${score}.00"

# Disabling lwpu-persistence mode for original config
OUTPUT=`lwpu-smi -pm 0`

# delete lwsciipc.cfg from /etc
rm -rf /etc/lwsciipc.cfg
rm -f ${SUMMARY_LOG_TEMP}

exit 0
