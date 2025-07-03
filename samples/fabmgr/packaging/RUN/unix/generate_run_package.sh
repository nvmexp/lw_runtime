#!/bin/bash
#
# This script generate a .RUN package for Fabric Manager using makeself shell script.
# makeself generates a self-extractable compressed tar archive from a directory and 
# start an embedded bash script during extraction. This embedded bash script will
# act as the installer to copy required files to target machine.
#

TARGET=$1

# cachine our package current directory
package_source_dir=${PWD}

# default to lwmake style target of 'amd64'
if [ -z "${TARGET}" ]; then
    TARGET=amd64
fi

# validate the target
if [ "${TARGET}" != "amd64" ] && [ "${TARGET}" != "aarch64" ]; then
    echo "Error: target build type has to be amd64 or aarch64 "
    exit 1
fi

if [ "${TARGET}" = "amd64" ]; then
    RUN_PKG_ARCH_PREFIX=x86_64
elif [ "${TARGET}" = "aarch64" ]; then
    RUN_PKG_ARCH_PREFIX=aarch64
fi

fm_rootdir=$(readlink -f ../../..)
fm_output_dir=${fm_rootdir}/_out/Linux_${TARGET}_release

# make fabric manager
cd ${fm_rootdir}; unix-build lwmake release -j12 ${TARGET} clobber 
cd ${fm_rootdir}; unix-build lwmake release -j12 ${TARGET}

# check whether the build output directory is created
if ![ -d ${fm_output_dir} ]; then
    echo "Error: cannot find Fabric Manager output directory, build might have failed, exiting..."
    exit 1
fi

mkdir ${fm_rootdir}/_out/packaging  > /dev/null 2>&1
#first cleanup our RUN package location
rm -rf ${fm_rootdir}/_out/packaging/RUN > /dev/null 2>&1
#re-create the directory again
mkdir ${fm_rootdir}/_out/packaging/RUN  > /dev/null 2>&1
pkg_working_dir=${fm_rootdir}/_out/packaging/RUN
pkg_binary_dir=${pkg_working_dir}/fabric_manager_pkg

# first cleanup and recreate our working directory for package
rm -rf ${pkg_binary_dir}
mkdir ${pkg_binary_dir}

# copy required installation file to the package directory
cp ${fm_output_dir}/../../scripts/systemd/lwpu-fabricmanager.service ${pkg_binary_dir}
cp ${fm_output_dir}/../../config/default_unix.cfg ${pkg_binary_dir}/fabricmanager.cfg
cp ${fm_output_dir}/../../sdk/public/lw_fm_agent.h ${pkg_binary_dir}
cp ${fm_output_dir}/../../sdk/public/lw_fm_types.h ${pkg_binary_dir}
cp ${fm_output_dir}/../../docs/LICENSE ${pkg_binary_dir}
cp ${fm_output_dir}/../../docs/third-party-notices.txt ${pkg_binary_dir}

cp ${fm_output_dir}/liblwfm.so.1 ${pkg_binary_dir}
cp -P ${fm_output_dir}/liblwfm.so ${pkg_binary_dir}
cp ${fm_output_dir}/lw-fabricmanager ${pkg_binary_dir}
cp ${fm_output_dir}/lwswitch-audit ${pkg_binary_dir}
cp ${fm_output_dir}/topology/dgx2_hgx2_topology  ${pkg_binary_dir}
cp ${fm_output_dir}/topology/dgxa100_hgxa100_topology  ${pkg_binary_dir}

# this file is specific to .RUN. This script will actually install/copy the 
# fabric manager binaries files to desired directory on target when .RUN file
# is exelwted.
cp ${package_source_dir}/fm_run_package_installer.sh ${pkg_binary_dir}

# this information is used to name the .RUN file.
BUILD_BRANCH_LOOK_UP_HEADER_FILE=${fm_rootdir}/../../drivers/common/inc/lwBldVer.h
# get our branch details like r445_00 or chips_a
SUB_BRANCH=$(grep -m 1 -w "#define LW_BUILD_BRANCH" ${BUILD_BRANCH_LOOK_UP_HEADER_FILE} | awk '{print $3}' | sed 's/[()]//g')
# get the current change list number
CHANGELIST_NUM=$(grep -m 1 -w "#define LW_BUILD_CHANGELIST_NUM" ${BUILD_BRANCH_LOOK_UP_HEADER_FILE} | awk '{print $3}' | sed 's/[()]//g') 
# get our build type which is Nightly or Official
BUILD_TYPE_TEMP=$(grep -m 1 -w "#define LW_BUILD_TYPE" ${BUILD_BRANCH_LOOK_UP_HEADER_FILE} | awk '{print $3}' | sed 's/[()]//g') 
BUILD_TYPE=`echo ${BUILD_TYPE_TEMP} | cut -d '"' -f 2`

# build_branch_version_temp will be like "rel/gpu_drv/r445/r445_00-35"
BUILD_BRANCH_VERSION_TEMP=$(grep -m 1 -w "#define LW_BUILD_BRANCH_VERSION " ${BUILD_BRANCH_LOOK_UP_HEADER_FILE} | awk '{print $3}' | sed 's/[()]//g')
# remove " from the build_branch_version_temp strin
BUILD_BRANCH_VERSION=`echo ${BUILD_BRANCH_VERSION_TEMP} | cut -d '"' -f 2`
# branch type will be dev or rel
BRANCH_TYPE=`echo ${BUILD_BRANCH_VERSION} | cut -d '/' -f 1`
# build tree will be like gpu_drv
BRANCH_TREE=`echo ${BUILD_BRANCH_VERSION} | cut -d '/' -f 2`
# branch name will be like r445. Branch name is applicable only for release branches
BRANCH_NAME=`echo ${BUILD_BRANCH_VERSION} | cut -d '/' -f 3`

# get the build number for release (official) builds
BUILD_VERSION_LOOK_UP_HEADER_FILE=${fm_rootdir}/../../drivers/common/inc/lwUnixVersion.h
OFFICIAL_BUILD_VERSION_TEMP=$(grep -m 1 -w "#define LW_VERSION_STRING " ${BUILD_VERSION_LOOK_UP_HEADER_FILE} | awk '{print $3}' | sed 's/[()]//g')
OFFICIAL_BUILD_VERSION=`echo ${OFFICIAL_BUILD_VERSION_TEMP} | cut -d '"' -f 2`

DATE=`date +%Y%m%d`

# construct the run package name based on build type and branch type
if [ "${BUILD_TYPE}" = "Official" ]; then
    RUN_PKG_NAME=lwpu-fabricmanager-${RUN_PKG_ARCH_PREFIX}-${OFFICIAL_BUILD_VERSION}-internal.run
elif [ "${BUILD_TYPE}" = "Nightly" ]; then
    # Branch name is applicable only for release branches
    if [ "${BRANCH_TYPE}" = "rel" ]; then
        RUN_PKG_NAME=lwpu-fabricmanager-${RUN_PKG_ARCH_PREFIX}-${BRANCH_TYPE}_${BRANCH_TREE}_${BRANCH_NAME}_${SUB_BRANCH}-${DATE}_${CHANGELIST_NUM}-internal.run
    else
        RUN_PKG_NAME=lwpu-fabricmanager-${RUN_PKG_ARCH_PREFIX}-${BRANCH_TYPE}_${BRANCH_TREE}_${SUB_BRANCH}-${DATE}_${CHANGELIST_NUM}-internal.run
    fi
fi

# copy all the required files to working directory
cp ${package_source_dir}/fm_run_package_installer.sh ${pkg_working_dir}
cp -R ${package_source_dir}/makeself-2.1.5 ${pkg_working_dir}

cd ${pkg_working_dir}
# create the .RUN file
makeself-2.1.5/makeself.sh ${pkg_binary_dir} ./${RUN_PKG_NAME} "LWPU Fabric Manager" ./fm_run_package_installer.sh

#done with packaging, clear the temp working directory
rm -rf ${pkg_binary_dir}
rm -rf ${pkg_working_dir}/fm_run_package_installer.sh
rm -rf ${pkg_working_dir}/makeself-2.1.5

echo "Packaging completed. Check ${pkg_working_dir} for .RUN file"
