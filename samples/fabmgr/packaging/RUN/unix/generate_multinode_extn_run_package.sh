#!/bin/bash
#
#This script will generate a fabric manager multinode extension. RUN package. This extension package model is
#used as LWLink multinode is not yet officially released, however for selected lwstomers, we need to provide
#the same. So, FM multinode specific files, topology files etc. cannot be packaged into the existing FM main
#package for now. When LWLink multinode is ready for official release, all these can move to the FM main package.
#

TARGET=$1

# cache our package current directory
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
    EXT_RUN_PKG_ARCH_PREFIX=x86_64
elif [ "${TARGET}" = "aarch64" ]; then
    EXT_RUN_PKG_ARCH_PREFIX=aarch64
fi

fm_rootdir=$(readlink -f ../../..)

# Note: as of now nothing to compile/build for this extension package. So skipping FM build steps

mkdir ${fm_rootdir}/_out/ext_packaging  > /dev/null 2>&1
#first cleanup our RUN package location
rm -rf ${fm_rootdir}/_out/ext_packaging/RUN > /dev/null 2>&1
#re-create the directory again
mkdir ${fm_rootdir}/_out/ext_packaging/RUN  > /dev/null 2>&1
pkg_working_dir=${fm_rootdir}/_out/ext_packaging/RUN
pkg_binary_dir=${pkg_working_dir}/fabric_manager_ext_pkg

# first cleanup and recreate our working directory for package
rm -rf ${pkg_binary_dir}
mkdir ${pkg_binary_dir}

# copy required installation file to the package directory
cp ${fm_rootdir}/config/multinode_topology/dgxa100_all_to_all_9node_topology ${pkg_binary_dir}
cp ${fm_rootdir}/config/multinode_topology/dgxa100_all_to_all_9node_trunk.csv ${pkg_binary_dir}

# this file is specific to .RUN. This script will actually install/copy the 
# fabric manager binaries files to desired directory on target when .RUN file
# is exelwted.
cp ${package_source_dir}/fm_multinode_extn_run_package_installer.sh ${pkg_binary_dir}

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
EXT_RUN_PKG_NAME_COMMON_PREFIX="lwpu-fabricmanager-multinode-extn"
if [ "${BUILD_TYPE}" = "Official" ]; then
    EXT_RUN_PKG_NAME=${EXT_RUN_PKG_NAME_COMMON_PREFIX}-${EXT_RUN_PKG_ARCH_PREFIX}-${OFFICIAL_BUILD_VERSION}-internal.run
elif [ "${BUILD_TYPE}" = "Nightly" ]; then
    # Branch name is applicable only for release branches
    if [ "${BRANCH_TYPE}" = "rel" ]; then
        EXT_RUN_PKG_NAME=${EXT_RUN_PKG_NAME_COMMON_PREFIX}-${EXT_RUN_PKG_ARCH_PREFIX}-${BRANCH_TYPE}_${BRANCH_TREE}_${BRANCH_NAME}_${SUB_BRANCH}-${DATE}_${CHANGELIST_NUM}-internal.run
    else
        EXT_RUN_PKG_NAME=${EXT_RUN_PKG_NAME_COMMON_PREFIX}-${EXT_RUN_PKG_ARCH_PREFIX}-${BRANCH_TYPE}_${BRANCH_TREE}_${SUB_BRANCH}-${DATE}_${CHANGELIST_NUM}-internal.run
    fi
fi

# copy all the required files to working directory
cp ${package_source_dir}/fm_multinode_extn_run_package_installer.sh ${pkg_working_dir}
cp -R ${package_source_dir}/makeself-2.1.5 ${pkg_working_dir}

cd ${pkg_working_dir}
# create the .RUN file
makeself-2.1.5/makeself.sh ${pkg_binary_dir} ./${EXT_RUN_PKG_NAME} "LWPU Fabric Manager" ./fm_multinode_extn_run_package_installer.sh

#done with packaging, clear the temp working directory
rm -rf ${pkg_binary_dir}
rm -rf ${pkg_working_dir}/fm_multinode_extn_run_package_installer.sh
rm -rf ${pkg_working_dir}/makeself-2.1.5

echo "Packaging completed. Check ${pkg_working_dir} for .RUN file"
