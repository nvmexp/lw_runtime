#!/bin/bash -eux
cd $(dirname $0)
export HERE=$(pwd)

# Runs a coverity build for OptiX on Linux

# We expect //sw in P4_ROOT (DVS Sets P4_ROOT)
export P4_ROOT=${P4_ROOT:-$(pwd)}

if [ ! -d $P4_ROOT/sw ]; then
  echo Error: \$P4_ROOT $P4_ROOT does not contain the directory sw. 
  echo Error: Maybe $P4_ROOT is not the root of a checkout?
  exit
fi

# Path to <branch>/apps/optix
if [ -n "${LW_COV_MODULE_PATH}" ]; then
    # We are in DVS. Make sure to find apps optix as an absolute path
    echo "This seems to be a DVS run. We assume that DVS set paths like P4_ROOT and LW_COV_MODULE_PATH for us"
    if [ -z ${LW_COV_MODULE_PATH} ]; then
        echo "Error: The variable LW_COV_MODULE_PATH is unset. It is set by the Coverity build of DVS".
        exit
    fi
    export OPTIX_SRC=${P4_ROOT}/${LW_COV_MODULE_PATH}
else
    # Standalone run. Figuring out our path (not implemented to release-branches - as you see)
    export OPTIX_SRC=${P4_ROOT}/sw/dev/gpu_drv/bugfix_main/apps/optix
fi

echo "We will cd ing to apps/optix (${OPTIX_SRC}): unix-build wants to find the top-level dir"
cd ${OPTIX_SRC}

# Make the Coverity compiler modifyable
test -d ${P4_ROOT}/config_template_tmp/gnu/ && chmod -R a+w ${P4_ROOT}/config_template_tmp/gnu/

# Environment section
# Many of them are set by DVS, but we define plausible values for Standalone runs
if [ -z "${LW_TOOLS}" ]; then
  echo "Error: LW_TOOLS is unset! Please set to your checkout of //sw/tools. Exit $0" 
  exit
fi

if [ -n "${LW_TOOLCHAIN}" ]; then
    echo "WARNING: You have \${LW_TOOLCHAIN} defined as \${LW_TOOLCHAIN}. This file needs to know some include paths of the toolchain an will set its own toolchain."
fi

# Manually setting things similar to sw\tools\unix\hosts\Linux-x86\targets\Linux-x86_64\configs\gcc-7.3.0as2-glibc-2.11.3.mk as we cannot include it
# Check this file for the correct LW_TOOLCHAIN to Path mappings: //sw/tools/unix/hosts/Linux-x86/targets/Linux-x86_64/configs/gcc-7.3.0as2-glibc-2.11.3.mk
export LW_TOOLCHAIN=${LW_TOOLCHAIN:-gcc-7.3.0as2-glibc-2.11.3}
# Referenced by gcc-7.3.0as2-glibc-2.11.3.mk. 
export _LW_TOOLCHAIN_glibc=glibc-2.11.3
# Referenced by gcc-7.3.0as2-glibc-2.11.3.mk. This should exist in //sw/tools/unix/hosts/Linux-x86/targets/Linux-x86_64

export _LW_TOOLCHAIN_gcc=gcc-7.3.0as2-2
export _LW_TOOLCHAIN_gcc_version=7.3.0
export COMPILER_DIR_x86_64=${LW_TOOLS}/unix/hosts/Linux-x86/targets/Linux-x86_64/${_LW_TOOLCHAIN_gcc}

# -B is unsupported by cov but needed to run the compiler in p4 (as not found). 
# -E - "is a skip argument" - no stdin compilations supported by cov
# Compiler options: 
# 	Use what cov-configure returns (grep cov-configure /local/home/skoerner/p4/skoerner_optix_coverty_linux/sw/CP/Linux64/emit-Coverity_Linux_AMD64_OptiX/build-log.txt
#       Add binutils for as to be found
#       Add the isystem and ignore the warnings
export COMPILER_OPTIONS_x86_64=${COMPILER_OPTIONS_x86_64:-" \
-isystem$COMPILER_DIR_x86_64/lib/gcc/x86_64-pc-linux-gnu/${_LW_TOOLCHAIN_gcc_version}/include \
-isystem$COMPILER_DIR_x86_64/lib/gcc/x86_64-pc-linux-gnu/${_LW_TOOLCHAIN_gcc_version}/include-fixed \
-isystem$COMPILER_DIR_x86_64/x86_64-pc-linux-gnu/include/c++/${_LW_TOOLCHAIN_gcc_version} \
-isystem$COMPILER_DIR_x86_64/x86_64-pc-linux-gnu/include/c++/${_LW_TOOLCHAIN_gcc_version}/x86_64-pc-linux-gnu \
-isystem${LW_TOOLS}/unix/targets/Linux-x86_64/${_LW_TOOLCHAIN_gcc}/include \
-B${LW_TOOLS}/unix/hosts/Linux-x86/targets/Linux-x86_64/binutils-2.29.1/bin \
-B${LW_TOOLS}/unix/targets/Linux-x86_64/${_LW_TOOLCHAIN_glibc}/lib \
-B${LW_TOOLS}/unix/hosts/Linux-x86/targets/Linux-x86_64/${_LW_TOOLCHAIN_gcc}/libexec/gcc/x86_64-pc-linux-gnu/${_LW_TOOLCHAIN_gcc_version} \
-B${LW_TOOLS}/unix/hosts/Linux-x86/targets/Linux-x86_64/${_LW_TOOLCHAIN_gcc}/x86_64-pc-linux-gnu/lib64 \
-nostdinc \
--sysroot=${LW_TOOLS}/unix/targets/Linux-x86_64/${_LW_TOOLCHAIN_glibc}"}

# Another set of call options found by coverity.
export COMPILER_OPTIONS_x86_64_1=${COMPILER_OPTIONS_x86_64_1:-" \
-isystem$COMPILER_DIR_x86_64/lib/gcc/x86_64-pc-linux-gnu/${_LW_TOOLCHAIN_gcc_version}/include \
-isystem$COMPILER_DIR_x86_64/lib/gcc/x86_64-pc-linux-gnu/${_LW_TOOLCHAIN_gcc_version}/include-fixed \
-isystem$COMPILER_DIR_x86_64/x86_64-pc-linux-gnu/include/c++/${_LW_TOOLCHAIN_gcc_version} \
-isystem$COMPILER_DIR_x86_64/x86_64-pc-linux-gnu/include/c++/${_LW_TOOLCHAIN_gcc_version}/x86_64-pc-linux-gnu \
-isystem${LW_TOOLS}/unix/targets/Linux-x86_64/${_LW_TOOLCHAIN_gcc}/include \
-B${LW_TOOLS}/unix/hosts/Linux-x86/targets/Linux-x86_64/binutils-2.29.1/bin \
-B${LW_TOOLS}/unix/targets/Linux-x86_64/${_LW_TOOLCHAIN_glibc}/lib \
-nostdinc \
--sysroot=${LW_TOOLS}/unix/targets/Linux-x86_64/${_LW_TOOLCHAIN_glibc}"}

export COMPONENT_NAME_LONG=${COMPONENT_NAME_LONG:-Coverity_Linux_AMD64_OptiX}
export COMPONENT_NAME_SHORT=${COMPONENT_NAME_SHORT:-OptiX}
export COVERITY_BRANCH_PROJECT=${COVERITY_BRANCH_PROJECT:-unix-build_OptiX}
export COVERITY_BRANCH=${COVERITY_BRANCH:-gpu_drv_bugfix_main}
export COVERITY_BRANCH_STREAM=${COVERITY_BRANCH_STREAM:-Linux_${COVERITY_BRANCH_PROJECT}_${COVERITY_BRANCH}_$COMPONENT_NAME_SHORT}
export COVERITY_DIR=${COVERITY_DIR:-${P4_ROOT}/sw/CP/Linux64}
# Next line to match DVS Build-Config
export COVERITY_EMIT_DIR=${COVERITY_EMIT_DIR:-${COVERITY_DIR}/emit-$COMPONENT_NAME_LONG}
# For the defect submission. b for the automatic builds. d for the virtuals
export COVERITY_LOGIN_OPTIONS_AUTOMATIC=${COVERITY_LOGIN_OPTIONS_AUTOMATIC:-"--user reporter --password coverity --host txcovlinb "}
export COVERITY_LOGIN_OPTIONS_VIRTUAL=${COVERITY_LOGIN_OPTIONS_VIRTUAL:-"--user reporter --password coverity --host txcovlind "}
export COVERITY_SCRIPTS_DIR=${COVERITY_SCRIPTS_DIR:-${LW_TOOLS}/Coverity/ps-scripts}
export TOOLSDIR=${TOOLSDIR:-${LW_TOOLS}}
export VERBOSE=${VERBOSE:-1}

# Build section
mkdir -p $COVERITY_EMIT_DIR
#
# Workaround for Coverity not handling -wrapper
mkdir -p ${OPTIX_SRC}/config_template_tmp
chmod -R a+w ${OPTIX_SRC}/config_template_tmp
cp -r $COVERITY_DIR/config/templates/gnu ${OPTIX_SRC}/config_template_tmp
chmod +w ${OPTIX_SRC}/config_template_tmp/gnu/gnu_config.xml
sed -i -e '/<skip_arg>-wrapper/d' ${OPTIX_SRC}/config_template_tmp/gnu/gnu_config.xml

export COVERITY_COV_BUILD=$COVERITY_DIR/bin/cov-build
export COVERITY_COV_BUILD_PARAMS="--verbose 2 --prevent-root ${P4_ROOT}/sw/CP/Linux64/ --dir ${COVERITY_EMIT_DIR}"
# Configure all compilers that are in use! 
# Taken from ${COVERITY_EMIT_DIR}/build-log.txt: Look for ERROR This <pathto> compiler command specifies the following arguments which should be marked as required in your configuration:
# Don't try verbose 4 on DVS! The output will contain error-lines from feature checking
cat >${OPTIX_SRC}/temp_build.sh  << EOF
# Configure-Section
# compiler in $PATH or with full-path. --template can be used only w/o full-path
#   Unable to run native compiler sanity test:  The compiler itself could not run. Increase verbose level to 4 and check the log. Maybe -- <compiler_options> are needed for this compiler? I had "as execvpe File not found"
set -e
$COVERITY_DIR/bin/cov-configure --verbose 1 --template-dir ${OPTIX_SRC}/config_template_tmp --comptype gcc --compiler $COMPILER_DIR_x86_64/bin/x86_64-pc-linux-gnu-gcc -- ${COMPILER_OPTIONS_x86_64} -fPIC -std=gnu++11 -std=c++0x -std=c++0x -std=gnu++11 -std=c++0x -std=c++0x
$COVERITY_DIR/bin/cov-configure --verbose 1 --template-dir ${OPTIX_SRC}/config_template_tmp --comptype gcc --compiler $COMPILER_DIR_x86_64/bin/x86_64-pc-linux-gnu-gcc -- ${COMPILER_OPTIONS_x86_64} -fPIC -std=gnu++11
$COVERITY_DIR/bin/cov-configure --verbose 1 --template-dir ${OPTIX_SRC}/config_template_tmp --comptype gcc --compiler $COMPILER_DIR_x86_64/bin/x86_64-pc-linux-gnu-gcc -- ${COMPILER_OPTIONS_x86_64} -fPIC -std=gnu++11 -std=c++0x
$COVERITY_DIR/bin/cov-configure --verbose 1 --template-dir ${OPTIX_SRC}/config_template_tmp --comptype gcc --compiler $COMPILER_DIR_x86_64/bin/x86_64-pc-linux-gnu-gcc -- ${COMPILER_OPTIONS_x86_64_1} -fPIC -std=gnu++11 -std=c++0x
export PATH=/usr/local/bin:${PATH}
set -x
./make/dvs.sh "unix-build --tools ${LW_TOOLS}" Linux amd64 release v
EOF
chmod +x ${OPTIX_SRC}/temp_build.sh
unix-build \
--verbose \
--no-read-only-bind-mounts \
--no-devrel \
--tools ${LW_TOOLS} \
--extra /proc \
--extra ${P4_ROOT}/sw/CP/Linux64 \
--extra ${P4_ROOT} \
--extra ${OPTIX_SRC} \
--extra ${P4_ROOT} \
--extra /usr/bin \
--extra /usr/local/bin \
--elwvar COVERITY_COV_BUILD="${COVERITY_COV_BUILD}" \
--elwvar COVERITY_COV_BUILD_PARAMS="${COVERITY_COV_BUILD_PARAMS}" \
--elwvar COMPONENT_NAME_LONG="${COMPONENT_NAME_LONG}" \
--elwvar COMPONENT_NAME_SHORT="${COMPONENT_NAME_SHORT}" \
--elwvar COVERITY_BRANCH_PROJECT="${COVERITY_BRANCH_PROJECT}" \
--elwvar COVERITY_BRANCH="${COVERITY_BRANCH}" \
--elwvar COVERITY_BRANCH_STREAM="${COVERITY_BRANCH_STREAM}" \
--elwvar COVERITY_DIR="${COVERITY_DIR}" \
--elwvar COVERITY_EMIT_DIR="${COVERITY_EMIT_DIR}" \
--elwvar COVERITY_LOGIN_OPTIONS_AUTOMATIC="${COVERITY_LOGIN_OPTIONS_AUTOMATIC}" \
--elwvar COVERITY_LOGIN_OPTIONS_VIRTUAL="${COVERITY_LOGIN_OPTIONS_VIRTUAL}" \
--elwvar COVERITY_SCRIPTS_DIR="${COVERITY_SCRIPTS_DIR}" \
--elwvar TOOLSDIR="${TOOLSDIR}" \
--elwvar VERBOSE="${VERBOSE}" \
${OPTIX_SRC}/temp_build.sh || ( echo "Error running ${OPTIX_SRC}/temp_build.sh" ; exit 1 ) || exit 1
echo "End of the Build"

echo "#### For detailed errors check ${COVERITY_EMIT_DIR}/build-log.txt  ####"

# Analyze
#
echo "Starting analyze"
$COVERITY_DIR/bin/cov-analyze --dir ${COVERITY_EMIT_DIR} --verbose 2 --security --enable AUDIT.SPELWLATIVE_EXELWTION_DATA_LEAK --checker-option AUDIT.SPELWLATIVE_EXELWTION_DATA_LEAK:max_sensitive_read_size:"2" --conlwrrency --enable-fnptr --enable-virtual --enable-constraint-fpp -j auto --strip-path /dvs/p4/build

# Commiting results depending on the build-mode (Virtual/Automatic)
echo "Commiting defects"

#
# Commit - Automatic
#
if [ "${IS_VIRTUAL_BUILD}" == "false" ]; then
    $COVERITY_DIR/bin/cov-commit-defects  --certs /dvs/p4/build/sw/tools/scripts/CoverityAutomation/SSLCert/ca-chain.crt --https-port 8443 $COVERITY_LOGIN_OPTIONS_AUTOMATIC --dir $COVERITY_EMIT_DIR --stream $COVERITY_BRANCH_STREAM --description ${COMPONENT_NAME_LONG}_$TESTED_CHANGELIST
fi
#
# Commit - Virtual
#
if [ "${IS_VIRTUAL_BUILD}" == "true" ]; then
    perl $COVERITY_SCRIPTS_DIR/create-stream.pl --config $COVERITY_SCRIPTS_DIR/coverity_pse_config_txcovlind.xml --stream $SUBMITTED_BY_USERNAME --project "Developer Streams"
    $COVERITY_DIR/bin/cov-commit-defects  --certs /dvs/p4/build/sw/tools/scripts/CoverityAutomation/SSLCert/ca-chain.crt --https-port 8443 $COVERITY_LOGIN_OPTIONS_VIRTUAL --dir $COVERITY_EMIT_DIR --stream $SUBMITTED_BY_USERNAME --description ${COMPONENT_NAME_LONG}_$TESTED_CHANGELIST
fi
