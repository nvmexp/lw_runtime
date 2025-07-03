#!/bin/bash
#
# This shell script is intended to be used by DVS to ilwoke a build of
# lwsci and create an intermediate tarball containing the files
# to be propagated to the machine that will create the final package.
#
# See the comments at the top of drivers/common/build/unix/dvs-util.sh for usage details.

# determine LW_SOURCE by cd'ing to the directory containing this script,
# and then backing up the appropriate number of directories
cd `dirname $0`
cd ../../
lw_source=`pwd`

# include the helper functions; this also parses the commandline
. ${lw_source}/drivers/common/build/unix/dvs-util.sh

# assign variables needed below and in the helper functions called below
assign_common_variables lwsci

lwsci_dir=${lw_source}/drivers/lwsci

# build the driver
run_lwmake ${lwsci_dir}

# No mangle_files
mangle_files=

lwsci_libs="drivers/lwsci/lwscisync/${outputdir}/liblwscisync.so \
  drivers/lwsci/lwscibuf/${outputdir}/liblwscibuf.so \
  drivers/lwsci/lwscistream/${outputdir}/liblwscistream.so \
  drivers/lwsci/lwscicommon/${outputdir}/liblwscicommon.so \
  drivers/lwsci/lwscievent/${outputdir}/liblwscievent.so \
  drivers/lwsci/lwsciipc/${outputdir}/liblwsciipc.so \
  drivers/lwsci/lwscisync/${outputdir}/liblwscisync.so.1 \
  drivers/lwsci/lwscibuf/${outputdir}/liblwscibuf.so.1 \
  drivers/lwsci/lwscistream/${outputdir}/liblwscistream.so.1 \
  drivers/lwsci/lwscicommon/${outputdir}/liblwscicommon.so.1 \
  drivers/lwsci/lwscievent/${outputdir}/liblwscievent.so.1 \
  drivers/lwsci/lwsciipc/${outputdir}/liblwsciipc.so.1"

# create the tar file in LW_SOURCE
tar_output_files \
  ${lwsci_libs} \
  drivers/lwsci/lwsciipc/lwsciipc_dvs.cfg \
  drivers/lwsci/tests/lwscibuf/api/${outputdir}/test_lwscibuf_api \
  drivers/lwsci/tests/lwscistream/component_tests/${outputdir}/test_lwscistream_api \
  drivers/lwsci/tests/lwscisync/api/${outputdir}/test_lwscisync_api \
  drivers/lwsci/tests/lwsciipc/${outputdir}/test_lwsciipc_read \
  drivers/lwsci/tests/lwsciipc/${outputdir}/test_lwsciipc_write \
  drivers/lwsci/tests/lwsciipc/${outputdir}/test_lwsciipc_perf \
  drivers/lwsci/tests/dvs/test_lwsci.sh \

# success
exit 0

