#!/bin/bash
#
# This shell script is intended to be used by DVS to ilwoke a build of
# LWSwitch Test and create an intermediate tarball containing the files.
#
# See the comments at the top of drivers/common/build/unix/dvs-util.sh for usage details.

# determine LW_SOURCE by cd'ing to the directory containing this script,
# and then backing up the appropriate number of directories
cd `dirname $0`
cd ../../../../
lw_source=`pwd`

# include the helper functions; this also parses the commandline
. ${lw_source}/drivers/common/build/unix/dvs-util.sh

# assign variables needed below and in the helper functions called below
assign_common_variables lwswitch_test

lwswitch_test_dir=${lw_source}/drivers/lwswitch/tests

# build LWSwitch Test
run_lwmake ${lwswitch_test_dir}

# success
exit 0
