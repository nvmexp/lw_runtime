#!/bin/bash
#
#  Copyright (c) 2018 LWPU Corporation.  All rights reserved.
#
#  LWPU Corporation and its licensors retain all intellectual property and proprietary
#  rights in and to this software, related documentation and any modifications thereto.
#  Any use, reproduction, disclosure or distribution of this software and related
#  documentation without an express license agreement from LWPU Corporation is strictly
#  prohibited.
#
#  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
#  AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
#  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
#  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
#  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
#  BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
#  INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
#  SUCH DAMAGES
#
# This shell script is intended to be used by DVS to ilwoke a build of optix and
# create an intermediate tarball containing the files to be propagated to the
# machine that will create the final package.
#
# See the comments at the top of drivers/common/build/unix/dvs-util.sh for usage
# details.

# determine LW_SOURCE by cd'ing to the directory containing this script,
# and then backing up the appropriate number of directories
cd `dirname $0`
cd ../../../
lw_source=`pwd`

# include the helper functions; this also parses the commandline
. ${lw_source}/drivers/common/build/unix/dvs-util.sh

# To avoid memory exhaustion during the link phase, build all of LWVM first,
# then build everything else.

if [ -n "${COVERITY_COV_BUILD}" ]; then
    echo "This seems to be a coverity build. Make will be parsed through \"${COVERITY_COV_BUILD}\""
fi

OPTIX_RUN_LWMAKE() {
  if [ -n "${COVERITY_COV_BUILD}" ]; then
    # unix_build_lwmake_cmd is an array now with this definition
    # LWMAKE_CMD=lwmake
    # unix_build_lwmake_cmd=(${unix_build_cmd})
    # unix_build_lwmake_cmd+=(${LWMAKE_CMD})
    # unix_build_lwmake_cmd+=(${dvs_elw_vars})
    # unix_build_lwmake_cmd+=(LW_TARGET_OS=${target_os} LW_TARGET_ARCH=${target_arch} LW_BUILD_TYPE=${build_type})
    # unix_build_lwmake_cmd+=(${_lwmake_target_arch_abi})
    # unix_build_lwmake_cmd+=(${_lwmake_default_jN} ${lwmake_args})
    # unix_build_lwmake_cmd+=(--keep-going)

    # We want to get rid of ${unix_build_cmd} because we are in a unix-build already at this point
    # So we take a slice starting after unix_build_cmd
    unix_build_cmd_array=(${unix_build_cmd})
    unix_build_cmd_len=${#unix_build_cmd_array[@]}
    echo "lwmake settings for for Coverity: ${unix_build_lwmake_cmd[@]:${unix_build_cmd_len}}"
    echo "Coverity: final command: ${COVERITY_COV_BUILD} ${COVERITY_COV_BUILD_PARAMS} ${unix_build_lwmake_cmd[@]:${unix_build_cmd_len}} optix"
    ${COVERITY_COV_BUILD} ${COVERITY_COV_BUILD_PARAMS} ${unix_build_lwmake_cmd[@]:${unix_build_cmd_len}} optix
  else
    run_lwmake ${lw_source}/apps/optix
  fi
}

# LWVM:
additional_lwmake_args="@lwvm-build"
assign_common_variables optix
OPTIX_RUN_LWMAKE

# lwmake_args needs to be reset because assign_common_variables aclwmulates arguments
# onto it and it is initialized in dvs-util.sh.
lwmake_args=$@
# Build OptiX libraries
additional_lwmake_args="@lwoptix @optix_static"
assign_common_variables optix
OPTIX_RUN_LWMAKE

# Build everything else (tests and tools), but limit number of processes so
# test linking doesn't exhaust the build node's memory
lwmake_args=$@
additional_lwmake_args="-j$(expr $(nproc) / 3)"
assign_common_variables optix
OPTIX_RUN_LWMAKE

# check if we ran lwmake on WSL
wsl_build=0
for lwmake_arg in ${lwmake_args} ; do
    if test "$lwmake_arg" = "winnext" ; then wsl_build=1 ; fi
    if test "$lwmake_arg" = "winfuture" ; then wsl_build=1 ; fi
    if test "$lwmake_arg" = "wddm2" ; then wsl_build=1 ; fi
done

# assign mangle_files, which will be used by tar_output_files in release builds
mangle_files=" \
    apps/optix/${outputdir}/liblwoptix-mangle-list.txt"

# create the tar file in LW_SOURCE
if [ $wsl_build -eq 0 ] ; then
  tar_output_files \
      apps/optix/${outputdir}/liblwoptix.so.1
else
  echo "WSL build complete"
fi

# success
exit 0
