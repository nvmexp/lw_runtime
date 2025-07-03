#/bin/bash
#
# This script copies lwntest and lwn sample code from the source
# location in gpu/drv to the directories specified by the two
# command line arguments $1 and $2
# $1: target directory for lwn samples
# $2: target directory for lwntest
#
# Optional command-line arguments 3 & 4 specify TEGRA_TOP and the root of gpu/drv.
# $3: TEGRA_TOP
# $4: root of gpu/drv
# These arguments are required if running the copy script from a Windows P4 directory
# (TEGRA_TOP should point to the HOS tree and gpu/drv should point to Windows driver root).
#
# All parameters must be absolute directories and the script will error out if not.
#
# Typically the following targets are used:
# $TEGRA_TOP/tests-hos/lwn is the target location for lwntest
# $TEGRA_TOP/lwn-samples is the target location for the lwn samples
#
# If the target directory does not exist the script will create it.
# All required sub folders are created as well.
# ATTENTION: The target directory is supposed to be empty. If it
# contains files or folders they will be deleted!
#
# exclude.txt contains a list of files that are not copied.
#
# The required headers for LWN, GL, EGL and KHR are copied to $1/include
#
# The final step will replace Lwpu copyrights with the Hovi
# copyright.
#
# Usage example:
# cd $TEGRA_TOP
# <path to script>/cp_drv_to_rel.sh /e/git/lwn-samples /e/git/tests-hos/lwn
#
# Or example running from a Windows P4 dir (where the P4 driver tree is under different path from HOS TEGRA_TOP):
# <path to script>/cp_drv_to_rel.sh /e/git-hos/lwn-samples /e/git-hos/tests-hos/lwn /e/git-hos/ /c/p4/sw/dev_a/
#

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    echo "Error: Illegal number of parameters"
    echo "Usage: cp_drv_to_rel.sh <Destination dir for samples> <Destination dir for lwntest> <TEGRA_TOP override (optional; defaults to elw var TEGRA_TOP)> <gpu_drv root dir override (optional; defaults to TEGRA_TOP/gpu/drv)>"
    exit 1
fi

LWNSAMPLE_DEST=$1
LWNTEST_DEST=$2

if [[ ! "$LWNSAMPLE_DEST" = /* ]]; then
    echo "Error: Script expects absolute paths for input parameters and $LWNSAMPLE_DEST is not absolute."
    exit 1
fi

if [[ ! "$LWNTEST_DEST" = /* ]]; then
    echo "Error: Script expects absolute paths for input parameters and $LWNTEST_DEST is not absolute."
    exit 1
fi

# Set up defaults (input parameters may override)
IN_TEGRA_TOP=$TEGRA_TOP
IN_GPU_DRV_ROOT=$TEGRA_TOP/gpu/drv

if [ "$#" -gt 2 ]; then
    if [[ ! "$3" = /* ]]; then
        echo "Error: Script expects absolute paths for input parameters and $3 is not absolute."
        exit 1
    fi

    IN_TEGRA_TOP=$3
else
    if [ -z "$TEGRA_TOP" ]; then
        echo "Error: TEGRA_TOP not set"
        exit 1
    fi
fi

if [ "$#" -gt 3 ]; then
    if [[ ! "$4" = /* ]]; then
        echo "Error: Script expects absolute paths for input parameters and $4 is not absolute."
        exit 1
    fi

    IN_GPU_DRV_ROOT=$4
fi

if [ ! -d "$IN_TEGRA_TOP" ]; then
    echo "Error: TEGRA_TOP directory $IN_TEGRA_TOP does not exist"
    exit 1
fi

if [ ! -d "$IN_GPU_DRV_ROOT" ]; then
    echo "Error: gpu/drv directory $IN_GPU_DRV_ROOT does not exist"
    exit 1
fi

if [ -z `which rsync` ]; then
   echo "Error: rsync cannot be found"
   exit 1
fi

EXCLFILE=$SCRIPT_DIR/exclude.txt

# Specify the file that contains the Hovi copyright
HOVI_COPYRIGHT=$IN_TEGRA_TOP/tests-hos/copyright/HoviCopyright.txt
if [ ! -f $HOVI_COPYRIGHT ]; then
    echo "Error: Hovi copyright file "$HOVI_COPYRIGHT" does not exist"
    exit 1
fi


# Add write permisions in case the file is coming from a RO repo like Perforce.
COPY_CMD="rsync --exclude-from=$EXCLFILE -r --chmod=ug+w"
COPY_CMD_NOEXCLUDE="rsync -r --chmod=ug+w"

##################################################
# Copy lwn samples and public headers
##################################################
if [ ! -d "$LWNSAMPLE_DEST" ]; then
   mkdir -p $LWNSAMPLE_DEST
else
   # Folder already exist, we will delete its content to make
   # sure no files will remain there that are not in the $SOURCE
   rm -rf $LWNSAMPLE_DEST/*
fi

if [ ! -d "$LWNSAMPLE_DEST/include" ]; then
  mkdir $LWNSAMPLE_DEST/include
fi

if [ ! -d "$LWNSAMPLE_DEST/common" ]; then
  mkdir $LWNSAMPLE_DEST/common
fi

if [ ! -d "$LWNSAMPLE_DEST/external" ]; then
  mkdir $LWNSAMPLE_DEST/external
fi

if [ ! -d "$LWNSAMPLE_DEST/include/lwca" ]; then
  mkdir $LWNSAMPLE_DEST/include/lwca
fi

if [ ! -d "$LWNSAMPLE_DEST/include/llgd" ]; then
  mkdir $LWNSAMPLE_DEST/include/llgd
fi

if [ ! -d "$LWNSAMPLE_DEST/include/aftermath" ]; then
  mkdir $LWNSAMPLE_DEST/include/aftermath
fi

# Specify folders in $SOURCE/apps/lwn/ that should be copied
LWN_SAMPLES_DIRS="microbench samples win32 external"
for dir in $LWN_SAMPLES_DIRS
do
   $COPY_CMD $IN_GPU_DRV_ROOT/apps/lwn/$dir $LWNSAMPLE_DEST
done

# copy public LWN header files
$COPY_CMD $IN_GPU_DRV_ROOT/drivers/lwn/public/* $LWNSAMPLE_DEST/include

# copy public LWCA header files
LWDA_PATH=${TEGRA_TOP}/gpu/drv_lwda-9.0_odin
$COPY_CMD ${LWDA_PATH}/drivers/gpgpu/lwca/inc/lwca.h $LWNSAMPLE_DEST/include/lwca
$COPY_CMD ${LWDA_PATH}/drivers/gpgpu/lwca/inc/lwdaLWN.h $LWNSAMPLE_DEST/include/lwca
$COPY_CMD ${LWDA_PATH}/drivers/gpgpu/lwca/inc/lwdaNNAllocator.h $LWNSAMPLE_DEST/include/lwca

# copy lwnUtil folder from lwn/common
$COPY_CMD $IN_GPU_DRV_ROOT/apps/lwn/common/lwnUtil $LWNSAMPLE_DEST/common/
# copy lwnWin folder from lwn/common
$COPY_CMD $IN_GPU_DRV_ROOT/apps/lwn/common/lwnWin $LWNSAMPLE_DEST/common/

# copy top level nact files
NACT_FILES="!.nact !.nart LwnProcessBuildRules.narl"
for nact_file in $NACT_FILES
do
   $COPY_CMD_NOEXCLUDE $IN_GPU_DRV_ROOT/apps/lwn/$nact_file $LWNSAMPLE_DEST
done

# copy llgd header file and rename it
$COPY_CMD_NOEXCLUDE $IN_GPU_DRV_ROOT/drivers/lwn/devtools/llgd/llgd-target/inc/LlgdApi.h $LWNSAMPLE_DEST/include/llgd/llgd.h

# copy Aftermath header file and rename it
$COPY_CMD_NOEXCLUDE $IN_GPU_DRV_ROOT/drivers/lwn/devtools/aftermath/aftermath-target/inc/AftermathApi.h $LWNSAMPLE_DEST/include/aftermath/aftermath.h

# copy vcxproj and sln file for TearTest and corresponding freeglut vcxproj file
# also fix include path in TearTest vcxproj file for "lwn" headers
$COPY_CMD_NOEXCLUDE $IN_GPU_DRV_ROOT/apps/lwn/samples/TearTest/vcproj/TearTest.vcxproj  $LWNSAMPLE_DEST/samples/TearTest/vcproj
$COPY_CMD_NOEXCLUDE $IN_GPU_DRV_ROOT/apps/lwn/samples/TearTest/vcproj/TearTest.sln $LWNSAMPLE_DEST/samples/TearTest/vcproj
$COPY_CMD_NOEXCLUDE $IN_GPU_DRV_ROOT/apps/lwn/external/freeglut/freeglut.vcxproj $LWNSAMPLE_DEST/external/freeglut
sed -i -e 's#[\.\\]*drivers\\lwn\\public#..\\..\\..\\include#g' $LWNSAMPLE_DEST/samples/TearTest/vcproj/TearTest.vcxproj

# fix include paths
# All paths that need to get patched must be in the top level narl file.
# Since lwntest and samples will be in different destination directories, the narl
# should only contain paths relative to HosRoot (which is equal to $IN_TEGRA_TOP) or
# relative paths that are valid for samples and lwntest like e.g. ./common.
sed -i -e 's#gpu/drv/drivers/lwn/public#lwn-samples/include#g' $LWNSAMPLE_DEST/LwnProcessBuildRules.narl
sed -i -e 's#gpu/drv/drivers/lwn/interface#lwn-samples/private-include#g' $LWNSAMPLE_DEST/LwnProcessBuildRules.narl
sed -i -e 's#gpu/drv_lwda-9.0_odin/drivers/gpgpu/lwca/inc#lwn-samples/include/lwca#g' $LWNSAMPLE_DEST/LwnProcessBuildRules.narl

# Delete paths that are not needed
sed -i -e '\#HosRoot.Combine("gpu/drv/drivers/khronos/interface/apps-mirror"),#d' $LWNSAMPLE_DEST/LwnProcessBuildRules.narl

# Generate !.nact for lwn samples
echo "include \"LwnProcessBuildRules.narl\";
if FileExists(\"samples\/\!.nact\")
{
    build \"samples\";
}
if FileExists(\"microbench\/\!.nact\")
{
    build \"microbench\";
}" > $LWNSAMPLE_DEST/\!.nact

##################################################
# Copy lwntest
##################################################

if [ ! -d "$LWNTEST_DEST" ]; then
   mkdir -p $LWNTEST_DEST
else
   # Folder already exist, we will delete its content to make
   # sure no files will remain there that are not in the $SOURCE
   rm -rf $LWNTEST_DEST/*
fi

if [ ! -d "$LWNTEST_DEST/common" ]; then
   mkdir $LWNTEST_DEST/common
fi

# copy common lwnUtil folder
$COPY_CMD $IN_GPU_DRV_ROOT/apps/lwn/common/lwnUtil $LWNTEST_DEST/common/
# copy common lwnWin folder
$COPY_CMD $IN_GPU_DRV_ROOT/apps/lwn/common/lwnWin $LWNTEST_DEST/common/

# copy common files
LWNTEST_COMMON_FILES="lwwinsys.h"

for file in $LWNTEST_COMMON_FILES
do
   $COPY_CMD_NOEXCLUDE $IN_GPU_DRV_ROOT/apps/lwn/samples/common/$file $LWNTEST_DEST/common/
done

# Get patched narl file from samples
$COPY_CMD_NOEXCLUDE $LWNSAMPLE_DEST/LwnProcessBuildRules.narl $LWNTEST_DEST
$COPY_CMD_NOEXCLUDE $LWNSAMPLE_DEST/\!.nart $LWNTEST_DEST

# Delete path that is not used by lwntest. lwntest will include from $LWNTEST_DEST/common
sed -i -e '\#LwnRootInfo.Path.Combine("./samples/common"),#d' $LWNTEST_DEST/LwnProcessBuildRules.narl

# generate !.nact for lwntest
echo "include \"LwnProcessBuildRules.narl\";
if FileExists(\"lwntest\/\!.nact\")
{
    build \"lwntest\";
}" > $LWNTEST_DEST/\!.nact

# copy lwntest
pushd $IN_GPU_DRV_ROOT/apps/lwn/
$COPY_CMD lwntest $LWNTEST_DEST
popd > /dev/null

# lwntest uses private headers which will be copied from drv/drivers/lwn/liblwn-etc1 to
# lwn-test/include/liblwn-etc1. Copy header and change the path in the lwntest nact file.
mkdir -p $LWNTEST_DEST/lwntest/include/liblwn-etc1
$COPY_CMD_NOEXCLUDE $IN_GPU_DRV_ROOT/drivers/lwn/liblwn-etc1/*.h $LWNTEST_DEST/lwntest/include/liblwn-etc1
sed -i -e 's#gpu/drv/drivers/lwn/liblwn-etc1#lwn-tests/lwntest/include/liblwn-etc1#g' $LWNTEST_DEST/lwntest/\!.nact

##################################################
# fix copyright
##################################################
HOVI_DIR="$LWNTEST_DEST $LWNSAMPLE_DEST/common $LWNSAMPLE_DEST/include/lwn $LWNSAMPLE_DEST/include/lwnTool $LWNSAMPLE_DEST/samples $LWNSAMPLE_DEST/include/aftermath"
for dir in $HOVI_DIR
do
   if [ -d "$dir" ]; then
      # Only run fix copyright if the directory really exists
      echo fixing copyright in $dir
      pushd $dir > /dev/null
      perl $SCRIPT_DIR/fix-copyright.pl $HOVI_COPYRIGHT
      popd > /dev/null
   fi
done

# Copy lwn internal extension header. This file is provided to Hovi but should not be released in
# their SDK. See Bug 200258560.
mkdir -p $LWNSAMPLE_DEST/private-include/lwnExt
mkdir -p $LWNSAMPLE_DEST/private-include/lwnUtil
$COPY_CMD $IN_GPU_DRV_ROOT/drivers/lwn/interface/lwnExt/lwnExt_Internal.h $LWNSAMPLE_DEST/private-include/lwnExt
$COPY_CMD $IN_GPU_DRV_ROOT/drivers/lwn/interface/lwnExt/lwnExt_interception.h $LWNSAMPLE_DEST/private-include/lwnExt
$COPY_CMD $IN_GPU_DRV_ROOT/drivers/lwn/interface/lwnUtil/g_lwnObjectList.h $LWNSAMPLE_DEST/private-include/lwnUtil/g_lwnObjectList.h
$COPY_CMD $IN_GPU_DRV_ROOT/drivers/lwn/interface/lwnUtil/g_lwnObjectListCpp.h $LWNSAMPLE_DEST/private-include/lwnUtil/g_lwnObjectListCpp.h

# Copy GL, GLES, EGL and KHR headers. Those headers are suitable for release to anyone outside of LWPU
# See: https://wiki.lwpu.com/engwiki/index.php/OpenGL/Header_Files
$COPY_CMD $IN_TEGRA_TOP/gpu/apps-graphics/gpu/drivers/common/include/release/EGL $LWNSAMPLE_DEST/include
$COPY_CMD $IN_TEGRA_TOP/gpu/apps-graphics/gpu/drivers/common/include/release/GL $LWNSAMPLE_DEST/include
$COPY_CMD $IN_TEGRA_TOP/gpu/apps-graphics/gpu/drivers/common/include/release/GLES2 $LWNSAMPLE_DEST/include
$COPY_CMD $IN_TEGRA_TOP/gpu/apps-graphics/gpu/drivers/common/include/release/GLES3 $LWNSAMPLE_DEST/include
$COPY_CMD $IN_TEGRA_TOP/gpu/apps-graphics/gpu/drivers/common/include/release/KHR $LWNSAMPLE_DEST/include

# Copy Hovi-specific eglplatform.h.
$COPY_CMD $IN_TEGRA_TOP/tests-hos/external/include/egl/eglplatform.h $LWNSAMPLE_DEST/include/EGL

# Copy Vulkan headers, including extensions for Hovi.
mkdir -p $LWNSAMPLE_DEST/include/vulkan
$COPY_CMD $IN_TEGRA_TOP/3rdparty/khronos/vulkan/include/vulkan/vulkan.h $LWNSAMPLE_DEST/include/vulkan/
$COPY_CMD $IN_TEGRA_TOP/3rdparty/khronos/vulkan/include/vulkan/vulkan.hpp $LWNSAMPLE_DEST/include/vulkan/
$COPY_CMD $IN_TEGRA_TOP/3rdparty/khronos/vulkan/include/vulkan/vulkan_core.h $LWNSAMPLE_DEST/include/vulkan/
$COPY_CMD $IN_TEGRA_TOP/3rdparty/khronos/vulkan/include/vulkan/vulkan_vi.h $LWNSAMPLE_DEST/include/vulkan/
$COPY_CMD $IN_TEGRA_TOP/3rdparty/khronos/vulkan/include/vulkan/vulkan_win32.h $LWNSAMPLE_DEST/include/vulkan/
$COPY_CMD $IN_TEGRA_TOP/3rdparty/khronos/vulkan/include/vulkan/vk_platform.h $LWNSAMPLE_DEST/include/vulkan/

# Call script to copy the LWN, GLSLC and TEXPKG documentation to the LWNSAMPLE_DEST directory
$SCRIPT_DIR/cp_doc_to_rel.sh $LWNSAMPLE_DEST/docs ../include $IN_TEGRA_TOP $IN_GPU_DRV_ROOT

