#/bin/bash
#
# This script copies the source files that are required to build the
# doxygen dolwmentations for: LWN, GLSLC, TEXPKG, LWNTEST, GPU_OVERVIEW,
# GPU_ERROR_GUIDE, and KHRONOS to a common documentation folder.
#
# $1 Destination directory
# $2 Relative path from the destination directory to the LWN header files
#
# Command-line arguments 3 & 4 specify TEGRA_TOP and the root of gpu/drv.
# $3: TEGRA_TOP
# $4: root of gpu/drv
#
# The script will patch the LWN doxygen file to use the LWN headers specified
# by $2 as input to generate the documentation. These headers should contain the
# Hovi copyright.
#
# Typically the destination directory will be $TEGRA_TOP/lwn-samples/docs
#
#The script requires that the tests-hos repository is accessible.
set -e

if [ "$#" -ne 4 ]; then
   echo "Error: Illegal number of arguments"
   echo "Usage: cp_doc_to_rel.sh <Destination directory> <Relative path to LWN header folder> <TEGRA_TOP HOS root dir> <gpu_drv root dir>"
   exit 1
fi

# Destination directory for the documentation
DOLWMENT_DIR=`readlink -f $1`
DOXYGEN_DIR=$DOLWMENT_DIR/doxygen

# Relative path from $DOLWMENT_DIR to the folder containing
# the LWN headers
INCLUDE_DIR=$2
IN_TEGRA_TOP=$3
IN_GPU_DRV_ROOT=$4

if [ ! -d "$IN_TEGRA_TOP" ]; then
    echo "Error: $IN_TEGRA_TOP for <TEGRA_TOP HOS root dir> parameter does not exist"
    exit 1
fi

if [ ! -d "$IN_GPU_DRV_ROOT" ]; then
    echo "Error: $IN_GPU_DRV_ROOT for <gpu_drv root dir> does not exist"
    exit 1
fi

if [ ! -d "$DOLWMENT_DIR" ]; then
   mkdir -p $DOLWMENT_DIR
else
   # Folder already exist, we will delete its content
   rm -rf $DOLWMENT_DIR/*
fi

pushd $DOLWMENT_DIR > /dev/null

if [ ! -d "$INCLUDE_DIR/lwn" ]; then
   echo "Error: Invalid include path"
   exit 1
fi

if [ ! -d "$INCLUDE_DIR/lwnTool" ]; then
   echo "Error: Invalid include path"
   exit 1
fi

popd > /dev/null

# Source directories of the documentation
LWN_DOC_DIR=$IN_GPU_DRV_ROOT/drivers/lwn/generate/docs
GLSLC_DOC_DIR=$IN_GPU_DRV_ROOT/drivers/lwn/glslc/docs
TEXPKG_DOC_DIR=$IN_GPU_DRV_ROOT/drivers/lwn/tools/lwntexpkg/docs
LWNTEST_DOC_DIR=$IN_GPU_DRV_ROOT/apps/lwn/lwntest/docs
GPU_OVERVIEW_DOC_DIR=$IN_TEGRA_TOP/tests-hos/sourcedocs/GPU_OVERVIEW
GPU_ERROR_GUIDE_DOC_DIR=$IN_TEGRA_TOP/tests-hos/sourcedocs/GPU_ERROR_GUIDE
KHRONOS_DOC_DIR=$IN_GPU_DRV_ROOT/apps/lwn/khronos/docs
RELNOTES_DIR=$IN_TEGRA_TOP/tests-hos/sourcedocs/releasenotes

HOVI_COPYRIGHT_DIR=$IN_TEGRA_TOP/tests-hos/copyright

# Common doxygen files
# Use footer and header from tests-hos since the footer.html contains Hovi Copyright.
DOXYGEN_FILES="Doxyfile $HOVI_COPYRIGHT_DIR/header.html $HOVI_COPYRIGHT_DIR/footer.html"

TWEAK_GUIDE_CMD=$LWN_DOC_DIR/tweak-guide.pl

mkdir $DOXYGEN_DIR

##################################################
# Copy LWN documentation
##################################################
mkdir $DOXYGEN_DIR/LWN

pushd $LWN_DOC_DIR > /dev/null

perl  $TWEAK_GUIDE_CMD lwn 2 g_lwnformats_doc.md 0 ProgrammingGuide.md > $DOXYGEN_DIR/LWN/lwn_ProgrammingGuide.md
cp $DOXYGEN_FILES $DOXYGEN_DIR/LWN

popd > /dev/null

# Patch path for the LWN headers that Doxygen takes as input
sed -i -e "s|lwn/|../../$INCLUDE_DIR/lwn/|g" $DOXYGEN_DIR/LWN/Doxyfile
# restore lines commented out in the LWPU build environment (which uses Doxygen 1.8.1).
sed -i -e "s|###||g" $DOXYGEN_DIR/LWN/Doxyfile

##################################################
# Copy GLSLC documentation
##################################################
mkdir $DOXYGEN_DIR/GLSLC

GLSLC_FILES="GL_LW_extended_pointer_atomics.txt GL_LW_separate_texture_types.txt"

pushd $GLSLC_DOC_DIR > /dev/null

perl $TWEAK_GUIDE_CMD lwn_glslc 2 GLSLCProgrammingGuide.md > $DOXYGEN_DIR/GLSLC/lwnTool_GlslcProgrammingGuide.md
cp $DOXYGEN_FILES $GLSLC_FILES $DOXYGEN_DIR/GLSLC/

popd > /dev/null

sed -i -e "s|lwnTool/|../../$INCLUDE_DIR/lwnTool/|g" $DOXYGEN_DIR/GLSLC/Doxyfile
sed -i -e "s|###||g" $DOXYGEN_DIR/GLSLC/Doxyfile

##################################################
# Copy TEXPKG documentation
##################################################
mkdir $DOXYGEN_DIR/TEXPKG

TEXPKG_FILES="lwntexpkg.png lwnfd.png texbl.png"

pushd $TEXPKG_DOC_DIR > /dev/null

perl $TWEAK_GUIDE_CMD lwn_texpkg 2 TexturePackager.md > $DOXYGEN_DIR/TEXPKG/lwnTool_TexturePackagerGuide.md
cp $DOXYGEN_FILES $TEXPKG_FILES $DOXYGEN_DIR/TEXPKG

popd > /dev/null

sed -i -e "s|lwnTool/|../../$INCLUDE_DIR/lwnTool/|g" $DOXYGEN_DIR/TEXPKG/Doxyfile
sed -i -e "s|###||g" $DOXYGEN_DIR/TEXPKG/Doxyfile

##################################################
# Copy LWN TEST documentation
##################################################
mkdir $DOXYGEN_DIR/LWNTEST

LWNTEST_SRC="mainpage.md testlist.md"

pushd $LWNTEST_DOC_DIR > /dev/null

cp $DOXYGEN_FILES $LWNTEST_SRC $DOXYGEN_DIR/LWNTEST

popd > /dev/null

##################################################
# Copy GPU_OVERVIEW documentation
##################################################
mkdir $DOXYGEN_DIR/GPU_OVERVIEW

pushd $GPU_OVERVIEW_DOC_DIR > /dev/null

perl $TWEAK_GUIDE_CMD gpuOverview_MaxwellBestPractices 4 MaxwellBestPractices.md > $DOXYGEN_DIR/GPU_OVERVIEW/gpuOverview_MaxwellBestPractices.md
perl $TWEAK_GUIDE_CMD gpuOverview_MaxwellTechnicalOverview 4 MaxwellTechnicalOverview.md > $DOXYGEN_DIR/GPU_OVERVIEW/gpuOverview_MaxwellTechnicalOverview.md
perl $TWEAK_GUIDE_CMD gpuOverview_MaxwellProfilingGuide 4 MaxwellProfilingGuide.md > $DOXYGEN_DIR/GPU_OVERVIEW/gpuOverview_MaxwellProfilingGuide.md

cp -r ./imgs $DOXYGEN_DIR/GPU_OVERVIEW
cp $DOXYGEN_FILES $DOXYGEN_DIR/GPU_OVERVIEW

popd > /dev/null

sed -i -e "s|###||g" $DOXYGEN_DIR/GPU_OVERVIEW/Doxyfile

##################################################
# Copy GPU_ERROR_GUIDE documentation
##################################################
mkdir $DOXYGEN_DIR/GPU_ERROR_GUIDE

pushd $GPU_ERROR_GUIDE_DOC_DIR > /dev/null

perl $TWEAK_GUIDE_CMD gpuOverview_GpuErrorGuide 4 GpuErrorGuide.md > $DOXYGEN_DIR/GPU_ERROR_GUIDE/gpuOverview_GpuErrorGuide.md

cp -r ./imgs $DOXYGEN_DIR/GPU_ERROR_GUIDE
cp $DOXYGEN_FILES $DOXYGEN_DIR/GPU_ERROR_GUIDE

popd > /dev/null

sed -i -e "s|###||g" $DOXYGEN_DIR/GPU_ERROR_GUIDE/Doxyfile

##################################################
# Copy KHRONOS documentation
##################################################
mkdir $DOXYGEN_DIR/KHRONOS

KHRONOS_SRC="KhronosApis.md"

pushd $KHRONOS_DOC_DIR > /dev/null

perl $TWEAK_GUIDE_CMD khronos 2 KhronosApis.md > $DOXYGEN_DIR/KHRONOS/khronos_ApiProgrammingGuide.md
cp $DOXYGEN_FILES $DOXYGEN_DIR/KHRONOS

popd > /dev/null

sed -i -e "s|###||g" $DOXYGEN_DIR/GPU_ERROR_GUIDE/Doxyfile

##################################################
# Copy release notes
##################################################
RELEASE_NOTES="GLSLC-Changes.txt GPU_OVERVIEW-Changes.txt LWN-Changes.txt TEXPKG-Changes.txt GPU_ERROR_GUIDE-Changes.txt"

for file in $RELEASE_NOTES
do
   if [ -f $RELNOTES_DIR/$file ]; then
       cp $RELNOTES_DIR/$file $DOLWMENT_DIR
   fi
done
