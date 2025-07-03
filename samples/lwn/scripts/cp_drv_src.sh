#!/bin/bash
#
# This script copies dedicated LWN driver sources and files that are
# required to build LWN, LWWSI and LWRM_GPU to the destination folder
# which is specified by the first command line argument. This should
# be the root of the customer build tree.
#
# The following files will be copied by the script:
#   1: LWN files under gpu/drv/drivers/lwn
#   2: Files specified in driver_files.txt. These are files from
#      a) gpu/drv/drivers/khronos/egl
#      b) gpu/drv/sdk/lwpu/
#      c) core-private
#   3: LWSTL headers from core-private/include/lwstl
#
# Files listed in driver_exclude.txt will not be copied.
#
# ATTENTION: The gpu/drv and core-private folders in the destination
# directory are supposed to be empty. If they exist, they will be
# deleted.
#
# Usage example:
# cd $TEGRA_TOP/gpu/drv
# ./apps/lwn/scripts/cp_drv_src.sh <path to root of customer tree>
#
set -e

readonly SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ "$#" -ne 1 ]; then
    echo "Error: Illegal number of parameters"
    echo "Usage: cp_drv_src.sh <Destination dir for sources>"
    exit 1
fi

if [ -z "$TEGRA_TOP" ]; then
    echo "Error: TEGRA_TOP not set"
    exit 1
fi

if [ ! -d "$TEGRA_TOP" ]; then
    echo "Error: $TEGRA_TOP does not exist"
    exit 1
fi

if [ -z `which rsync` ]; then
   echo "Error: rsync cannot be found"
   exit 1
fi

PYTHON_CMD="$P4ROOT/sw/tools/win64/python/275/python"

if [ ! -f "$PYTHON_CMD" ]; then
    echo "Error: Invalid path to python: $PYTHON_CMD"
    exit 1
fi

# Specify the file that contains the Hovi copyright
HOVI_COPYRIGHT=$TEGRA_TOP/tests-hos/copyright/HoviCopyright.txt
if [ ! -f $HOVI_COPYRIGHT ]; then
    echo "Error: Hovi copyright file "$HOVI_COPYRIGHT" does not exist"
    exit 1
fi

DEST_ROOT=`readlink -f $1` || \
   (echo "Error: Path to destination folder does not exist."; exit 1)

# Check if the gpu/drv folder exists and create it if needed.
if [ ! -d "$DEST_ROOT/gpu/drv" ]; then
   mkdir -p $DEST_ROOT/gpu/drv
else
   # Folder already exist, we will delete its content to make sure
   # no files will remain there that are not in the $DEST_ROOT/gpu/drv
   rm -rf $DEST_ROOT/gpu/drv/*
fi

# If the core-private folder exists in the destination path
# delete it to make sure it will only contain files that are
# copied by this script.
if [ -d "$DEST_ROOT/core-private" ]; then
   rm -rf "$DEST_ROOT/core-private"
fi

EXCLFILE=$SCRIPT_DIR/driver_exclude.txt
EXCLCOPYRIGHT=$SCRIPT_DIR/driver_copyright_exclude.txt

COPY_CMD="rsync"

echo ===================================================
echo SOURCE $TEGRA_TOP
echo DEST   $DEST_ROOT
echo ===================================================

pushd $TEGRA_TOP > /dev/null

###############################################################
# Copy driver files specified in driver_files.txt
# The files listed in driver_files.txt must contain the
# relative path from $TEGRA_TOP to the file.
###############################################################
echo Copying extra source files and headers.
# Colwert to unix line ending, filter out comments and blank lines
sed -e 's/\r//g' -e '/#.*/d' -e '/^$/d' $SCRIPT_DIR/driver_files.txt |
while read f
do
   if [ -f "$f" ]; then
      # Check if there's update in ctrl2080gsp.h since current version
      # is only allowed to release at this moment.
      # checksum of current version: ctrl2080gsph_v1
      if [[ "$f" =~ ctrl2080gsp.h ]]; then
         echo "Check if crl2080gsp.h is updated or not!"
         ctrl2080gsph_v2=3765668371
         chk=$(cksum "$TEGRA_TOP/gpu/drv/sdk/lwpu/inc/ctrl/ctrl2080/ctrl2080gsp.h" | cut -d' ' -f1)
         if [ $ctrl2080gsph_v2 -eq $chk ]; then
            echo "No change in ctrl2080gsp.h. Copy it!"
            $COPY_CMD -R "$f" $DEST_ROOT || \
            echo "Warning: file $f was not copied."
         else
            echo "Warning: ctrl2080gsp.h is updated! Don't copy."
         fi
      else
         # Copy file including path to the destination folder
         $COPY_CMD -R "$f" $DEST_ROOT || \
           echo "Warning: file $f was not copied."
      fi
   else
      echo "Warning: file $f does not exist or is not a file."
   fi
done

popd > /dev/null

###############################################################
# Copy LWN files
###############################################################
mkdir -p $DEST_ROOT/gpu/drv/drivers/lwn

pushd $TEGRA_TOP/gpu/drv > /dev/null

echo Copying core LWN source files.
$COPY_CMD -r -R --exclude-from=$EXCLFILE drivers/lwn $DEST_ROOT/gpu/drv/

# Check for files in the destination without LWPU copyrights.
echo Checking for files missing LWPU copyrights.
pushd $DEST_ROOT/gpu/drv > /dev/null
egrep -i -r -L --exclude-from=$EXCLCOPYRIGHT \
      "Copyright.*LWPU,? CORPORATION" . |
while read f
do
    echo "Warning:  $f is missing an LWPU copyright."
done

# Remove LWNCFG blocks related to Volta or later GPUs
$PYTHON_CMD $SCRIPT_DIR/rm_lwncfg_blocks.py ./drivers/lwn \
GLOBAL_ARCH_TURING GLOBAL_ARCH_VOLTA GLOBAL_ARCH_AMPERE GLOBAL_GPU_FAMILY_GA10X

popd > /dev/null

# Adjust the copyrights of LWN public headers.
lwnpublic="$DEST_ROOT/gpu/drv/drivers/lwn/public"
echo Fixing copyrights for LWN headers.
if [ -d $lwnpublic ]; then
   # Only run fix copyright if the directory really exists
   pushd $lwnpublic > /dev/null
   perl $SCRIPT_DIR/fix-copyright.pl $HOVI_COPYRIGHT
   popd > /dev/null
fi

popd > /dev/null

###############################################################
# Copy LWSTL files
###############################################################
pushd $TEGRA_TOP/core-private > /dev/null

echo Copying core LWSTL header files.
$COPY_CMD -r -R --exclude-from=$EXCLFILE include/lwstl $DEST_ROOT/core-private

popd > /dev/null

###############################################################
# Patch Makefile
###############################################################

# Lwrrently liblwn and liblwn-etc1 are built using lwmake and tmake. For
# the tmake build the targets are named lwn-tmake and lwn-etc1-tmake. The
# customer build uses only tmake and we want the targets to be named lwn
# and lwn-etc1 therefore the Makefiles are patched.
# This can be removed once liblwn and liblwn-etc1 are only built using tmake.

if [ -d "$DEST_ROOT/gpu/drv/drivers/lwn/liblwn-tmake" ]; then
   # Rename component name in Makefile.tmk from lwn-tmake to lwn.
   sed -i -e 's#:= lwn-tmake#:= lwn#g' $DEST_ROOT/gpu/drv/drivers/lwn/liblwn-tmake/Makefile.tmk
   # Rename interface name in Makefile.interface.tmk from lwn-tmake to lwn.
   sed -i -e 's#:= lwn-tmake#:= lwn#g' $DEST_ROOT/gpu/drv/drivers/lwn/liblwn-tmake/Makefile.interface.tmk
else
   echo "WARNING: $DEST_ROOT/gpu/drv/drivers/lwn/liblwn-tmake does not exist!"
fi

if [ -d "$DEST_ROOT/gpu/drv/drivers/lwn/liblwn-etc1-tmake" ]; then
   # Rename component name in Makefile.tmk from lwn-etc1-tmake to lwn-etc1.
   sed -i -e 's#:= lwn-etc1-tmake#:= lwn-etc1#g' $DEST_ROOT/gpu/drv/drivers/lwn/liblwn-etc1-tmake/Makefile.tmk
   # Rename interface name in Makefile.interface.tmk from lwn-etc1-tmake to lwn-etc1.
   sed -i -e 's#:= lwn-etc1-tmake#:= lwn-etc1#g' $DEST_ROOT/gpu/drv/drivers/lwn/liblwn-etc1-tmake/Makefile.interface.tmk
else
   echo "WARNING: $DEST_ROOT/gpu/drv/drivers/lwn/liblwn-etc1-tmake does not exist!"
fi

###############################################################
# Patch core-private filelist.mk
###############################################################

# When building lwrm_gpu as part of buildable source only Maxwell
# GPUs are supported. Filter out all non-Maxwell related source
# files and remove non-Maxwell GPUs from the list of supported GPUs.
sed -i -e '/src_files += v2\/hwhal\/lwrm_gpu_hwhal_\(g[^m]\|tu\)[0-9]\+.\?.cpp/d' \
       -e '/src_files\s*+= v2\/hwhal\/lwrm_gpu_hw_platform_t19x.*.cpp/d' \
       -e '/supported_gpus += \(g[^m]\|tu\)[0-9]\+.\?/d' $DEST_ROOT/core-private/drivers/lwrm/gpu/filelist-v2.mk
