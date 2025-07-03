Update RTXStich (instructions for windows)
==========================================

These instructions are based on the old mostly outdated readme, added at the end of this document for completness.

Current parser state (2021-12-09): CL 30726927

Get required files to build driver tools
----------------------------------------

Use DVS target Develop_Windows_AMD64_GPGPU_COMPILER_tools.txt to get current viewspec, PATH, etc.

Viewspec 2021-12-09:
//sw/dev/gpu_drv/module_compiler/drivers/common/build/... //rtxstich_workspace/dev/gpu_drv/module_compiler/drivers/common/build/...
//sw/dev/gpu_drv/module_compiler/drivers/common/cop/... //rtxstich_workspace/dev/gpu_drv/module_compiler/drivers/common/cop/...
//sw/dev/gpu_drv/module_compiler/drivers/common/dwarf/... //rtxstich_workspace/dev/gpu_drv/module_compiler/drivers/common/dwarf/...
//sw/dev/gpu_drv/module_compiler/drivers/common/inc/... //rtxstich_workspace/dev/gpu_drv/module_compiler/drivers/common/inc/...
//sw/dev/gpu_drv/module_compiler/drivers/common/merlwry/... //rtxstich_workspace/dev/gpu_drv/module_compiler/drivers/common/merlwry/...
//sw/dev/gpu_drv/module_compiler/drivers/common/lwi/... //rtxstich_workspace/dev/gpu_drv/module_compiler/drivers/common/lwi/...
//sw/dev/gpu_drv/module_compiler/drivers/common/src/... //rtxstich_workspace/dev/gpu_drv/module_compiler/drivers/common/src/...
//sw/dev/gpu_drv/module_compiler/drivers/compiler/... //rtxstich_workspace/dev/gpu_drv/module_compiler/drivers/compiler/...
//sw/dev/gpu_drv/module_compiler/drivers/gpgpu/lwca/common/... //rtxstich_workspace/dev/gpu_drv/module_compiler/drivers/gpgpu/lwca/common/...
//sw/dev/gpu_drv/module_compiler/sdk/lwpu/inc/... //rtxstich_workspace/dev/gpu_drv/module_compiler/sdk/lwpu/inc/...
//sw/dev/gpu_drv/module_compiler/drivers/common/utils/utilsfiles.inc //rtxstich_workspace/dev/gpu_drv/module_compiler/drivers/common/utils/utilsfiles.inc
//sw/dev/gpu_drv/module_compiler/drivers/common/utils/compiler/... //rtxstich_workspace/dev/gpu_drv/module_compiler/drivers/common/utils/compiler/...
//sw/dev/gpu_drv/module_compiler/drivers/common/utils/lwstl/... //rtxstich_workspace/dev/gpu_drv/module_compiler/drivers/common/utils/lwstl/...
//sw/dev/gpu_drv/module_compiler/drivers/common/utils/lwstl_std/... //rtxstich_workspace/dev/gpu_drv/module_compiler/drivers/common/utils/lwstl_std/...
//sw/dev/gpu_drv/module_compiler/drivers/common/utils/generic/... //rtxstich_workspace/dev/gpu_drv/module_compiler/drivers/common/utils/generic/...
//sw/dev/gpu_drv/module_compiler/drivers/common/utils/merlwry/... //rtxstich_workspace/dev/gpu_drv/module_compiler/drivers/common/utils/merlwry/...
//sw/dev/gpu_drv/module_compiler/drivers/common/utils/machine/... //rtxstich_workspace/dev/gpu_drv/module_compiler/drivers/common/utils/machine/...
//sw/devrel/SDK/inc/GL/... //rtxstich_workspace/devrel/SDK/inc/GL/...

Tools Viewspec 2020-08-14 (most of these are already in the OptiX tools workspace):
//sw/tools/ddk/nt8/9600/... //tools_workspace/ddk/nt8/9600/...
//sw/tools/sei/robust/... //tools_workspace/sei/robust/...
//sw/tools/win32/ActivePerl/5.10.0.1004/... //tools_workspace/win32/ActivePerl/5.10.0.1004/...
//sw/tools/win32/bison/bison_2_3/... //tools_workspace/win32/bison/bison_2_3/...
//sw/tools/win32/bison/m4-1.4.14/... //tools_workspace/win32/bison/m4-1.4.14/...
//sw/tools/win32/infozip/... //tools_workspace/win32/infozip/...
//sw/tools/win32/MiscBuildTools/... //tools_workspace/win32/MiscBuildTools/...
//sw/tools/win32/msvc120u2/... //tools_workspace/win32/msvc120u2/...
//sw/tools/win32/msvc141/... //tools_workspace/win32/msvc141/...
//sw/tools/win32/msvc141u3/... //tools_workspace/win32/msvc141u3/...
//sw/tools/win32/msvc141u5/... //tools_workspace/win32/msvc141u5/...
//sw/tools/win32/msvc141u6/... //tools_workspace/win32/msvc141u6/...
//sw/tools/win32/msvc141u7/... //tools_workspace/win32/msvc141u7/...
//sw/tools/win32/python/254/... //tools_workspace/win32/python/254/...

Preparation
-----------
Look out for any files containing OPTIX_HAND_EDIT comments under PTXStitch. Some of these code pieces need to be ported into your local 
module_compiler sources before generating code as they are needed for code generation and allow generating the parser and lexer without
having to hand edit the generated files afterwards.
This set lwrrently includes (2021-12-09):

//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/lib/ptxparse/ptx.y
//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/lib/ptxparse/ptx.l
//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/lib/ptxparse/ptx.h
//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/lib/ptxparse/ptxConstructors.h
//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/lib/ptxparse/ptxInstructionTemplates.c
//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/lib/ptxparse/ptxIR.c
//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/lib/ptxparse/ptxIR.h
//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/lib/ptxparse/ptxMacroUtils.c
//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/lib/ptxparse/ptxparseMessageDefs.h
//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/lib/ptxaslib/lwdwarf.c
//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/lib/ptxaslib/lwdwarf.h
//sw/dev/gpu_drv/module_compiler/drivers/compiler/utilities/ucodeToElf/DebugInfo.c
//sw/dev/gpu_drv/module_compiler/drivers/compiler/utilities/std/stdBitSet.c
//sw/dev/gpu_drv/module_compiler/drivers/compiler/utilities/std/stdLocal.h

With these edits in place, it should be possible to copy the generated files without the need to further edit them.
Note that you will have to apply those edits to the files in //sw/dev/gpu_drv/bugfix_main/apps/optix/src/FrontEnd/PTX/PTXStitch/ after merging them
using the branch spec mentioned below since they will not be picked up by the merge as they are local changes.

Generate generated_private: suitable for internal use (future and internal instructions exposed)
------------------------------------------------------------------------------------------------

In a regular CMD window (might need admin rights):

set LW_TOOLS=<rtxstich_workspace>\tools

set PATH=<rtxstich_workspace>\tools\win32\MiscBuildTools;C:\Windows;C:\Windows\System32;C:\windows\system32\wbem;C:\Windows\;%PATH%

cd <rtxstich_workspace>\dev\gpu_drv\module_compiler\drivers\compiler

build\lwmake.exe tools

Copy relevant files from the "built" folder to apps\optix\src\FrontEnd\PTX\PTXStitch\generated_private\. I updated all files already there, removed files not generated anymore, and added files that I found were missing when compiling in the end.

Look out for any OPTIX_HAND_EDIT comments in generated_private files before overwriting them, these needs to be ported.


Generate generated_public: suitable for public use
---------------------------------------------------

build\lwmake.exe clobber

delete the built directory (built/)

set RELEASE=1

build\lwmake.exe LWCFG_OPTIONS="--override=LWCFG_GLOBAL_FEATURE_PTX_ISA_FUTURE:disable  --override=LWCFG_GLOBAL_FEATURE_PTX_ISA_INTERNAL:disable" tools

Copy relevant files from the "built" folder to apps\optix\src\FrontEnd\PTX\PTXStitch\generated_public\. I updated all files already there, removed files not generated anymore, and added files that I found were missing when compiling in the end.

Look out for any OPTIX_HAND_EDIT comments in generated_public files before overwriting them, these needs to be ported.


Update the parser
-----------------

Use the mapping ptxparse_bfm to integrate the parser files. Might need to be updated. - Maybe we should rename this branch mapping to ptxparse2optixBFM or similar?

Look out for patches marked with OPTIX_HAND_EDIT when resolving the files, these needs to be ported to the new version. See section "Preparation".


Build optix
-----------

Fix any compiler/link errors because of changes to the parser. This might also indicate that additional files needs to be added to the parser mapping. 


Patch ptx_lib_gen.py
--------------------

The python script generating ptx_instructions.ll might need to be updated to cope with new commands or changed syntax. To ilwoke manually cd to base of workspace and call:

  python apps/optix/src/FrontEnd/scripts/ptx_lib_gen.py -a ptx_instructions.ll apps/optix/src/FrontEnd/PTX/PTXStitch/generated_private/ptxInstructionDefs.incl  >& out.txt
  
  ptxparse/ptxInstructionDefs.table can be of help.





================================== OLD INSTRUCTIONS ==================================

There are several parts to integrating a new parser.  There are basically three parts to this.

1. Obtaining the generated files as by-products of compiling lwcc, and copying them into
   our tree.
2. Integrating the remaining parts of the compiler.
3. Updating our interface to the parser (Instruction.cpp and IRManipulator.cpp are the two
   biggest places).

******************************************************************************   
   1. Obtain the generated files from the parser
******************************************************************************

There are two directories of generated files in our parser:

generated_private: suitable for internal use (future and internal instructions exposed)
generated_public:  suitable for public use

In order to get both, you will have to compile lwcc twice and copy the generated files
after each build.

There are instructions around for compiling lwcc.  I found this site to be helpful:
https://wiki.lwpu.com/engwiki/index.php/Compiler_and_OCG/Compute_Compiler#How_to_build_Compute_Compiler

------------------------------------------
unix
------------------------------------------
 I used this client spec

 //sw/compiler/gpgpu/... //jbigler-mlt-rtsdk/ptxparser/compiler/gpgpu/...
 -//sw/compiler/gpgpu/export/... //jbigler-mlt-rtsdk/ptxparser/compiler/gpgpu/export/...
 //sw/dev/gpu_drv/module_compiler/drivers/cglang/... //jbigler-mlt-rtsdk/ptxparser/dev/gpu_drv/module_compiler/drivers/cglang/...
 //sw/dev/gpu_drv/module_compiler/drivers/common/... //jbigler-mlt-rtsdk/ptxparser/dev/gpu_drv/module_compiler/drivers/common/...
 //sw/dev/gpu_drv/module_compiler/drivers/compiler/build/... //jbigler-mlt-rtsdk/ptxparser/dev/gpu_drv/module_compiler/drivers/compiler/build/...
 //sw/dev/gpu_drv/module_compiler/drivers/compiler/edg/... //jbigler-mlt-rtsdk/ptxparser/dev/gpu_drv/module_compiler/drivers/compiler/edg/...
 //sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/... //jbigler-mlt-rtsdk/ptxparser/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/...
 -//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/build/... //jbigler-mlt-rtsdk/ptxparser/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/build/...
 //sw/dev/gpu_drv/module_compiler/drivers/compiler/Makefile //jbigler-mlt-rtsdk/ptxparser/dev/gpu_drv/module_compiler/drivers/compiler/Makefile
 //sw/dev/gpu_drv/module_compiler/drivers/compiler/mdFiles/... //jbigler-mlt-rtsdk/ptxparser/dev/gpu_drv/module_compiler/drivers/compiler/mdFiles/...
 //sw/dev/gpu_drv/module_compiler/drivers/gpgpu/... //jbigler-mlt-rtsdk/ptxparser/dev/gpu_drv/module_compiler/drivers/gpgpu/...
 //sw/dev/gpu_drv/module_compiler/sdk/lwpu/inc/... //jbigler-mlt-rtsdk/ptxparser/dev/gpu_drv/module_compiler/sdk/lwpu/inc/...

 
------------------------------------------
windows
------------------------------------------
Make your client include the following:

	//sw/compiler/gpgpu/... //davemc_ptxparse/compiler/gpgpu/...
	-//sw/compiler/gpgpu/export/... //davemc_ptxparse/compiler/gpgpu/export/...
	//sw/dev/gpu_drv/module_compiler/drivers/cglang/... //davemc_ptxparse/dev/gpu_drv/module_compiler/drivers/cglang/...
	//sw/dev/gpu_drv/module_compiler/drivers/common/... //davemc_ptxparse/dev/gpu_drv/module_compiler/drivers/common/...
	//sw/dev/gpu_drv/module_compiler/drivers/compiler/build/... //davemc_ptxparse/dev/gpu_drv/module_compiler/drivers/compiler/build/...
	//sw/dev/gpu_drv/module_compiler/drivers/compiler/edg/... //davemc_ptxparse/dev/gpu_drv/module_compiler/drivers/compiler/edg/...
	//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/... //davemc_ptxparse/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/...
	//sw/dev/gpu_drv/module_compiler/drivers/compiler/Makefile //davemc_ptxparse/dev/gpu_drv/module_compiler/drivers/compiler/Makefile
	//sw/dev/gpu_drv/module_compiler/drivers/compiler/mdFiles/... //davemc_ptxparse/dev/gpu_drv/module_compiler/drivers/compiler/mdFiles/...
	//sw/dev/gpu_drv/module_compiler/drivers/gpgpu/... //davemc_ptxparse/dev/gpu_drv/module_compiler/drivers/gpgpu/...
	//sw/dev/gpu_drv/module_compiler/sdk/lwpu/inc/... //davemc_ptxparse/dev/gpu_drv/module_compiler/sdk/lwpu/inc/...
	//sw/tools/msys/... //davemc_ptxparse/tools/msys/...
	//sw/tools/pkzip25/... //davemc_ptxparse/tools/pkzip25/...
	//sw/tools/sdk/DirectX9/... //davemc_ptxparse/tools/sdk/DirectX9/...
	//sw/tools/sdk/DirectX9c/... //davemc_ptxparse/tools/sdk/DirectX9c/...
	//sw/tools/win32/ActivePerl/584/bin/... //davemc_ptxparse/tools/win32/ActivePerl/584/bin/...
	//sw/tools/win32/ActivePerl/584/lib/... //davemc_ptxparse/tools/win32/ActivePerl/584/lib/...
	//sw/tools/win32/bison/bison_2_3/... //davemc_ptxparse/tools/win32/bison/bison_2_3/...
	//sw/tools/win32/cygnus/Apr2008/... //davemc_ptxparse/tools/win32/cygnus/Apr2008/...
	//sw/tools/win32/gnumake/... //davemc_ptxparse/tools/win32/gnumake/...
	//sw/tools/win32/infozip/... //davemc_ptxparse/tools/win32/infozip/...
	//sw/tools/win32/MiscBuildTools/... //davemc_ptxparse/tools/win32/MiscBuildTools/...
	//sw/tools/win32/msvc80sp1/... //davemc_ptxparse/tools/win32/msvc80sp1/...
	//sw/tools/win32/python/254/... //davemc_ptxparse/tools/win32/python/254/...

Change your Windows PATH environment variable to include these three folders:
set path=C:\sw\tools\win32\cygnus\Apr2008\bin;C:\sw\tools\win32\MiscBuildTools;C:\sw\tools\win32\gnumake\native;%path%

*** Use a CMD (DOS) shell for all build commands!!!!!!
    with administrative rights because of unclear reason

In 2015 I found the below to not succeed, and also not be needed. It appeared that I did have to rename my personal cygwin directory to force this build to find the checked-in one, though.
Read and follow section 7 of this: https://wiki.lwpu.com/engwiki/index.php/Compiler_and_OCG/NewUser#Before_you_Build
That should be:
cd C:\sw\dev\gpu_drv\module_compiler\drivers\compiler
build\lwmake.exe patch_cygwin

Now I do the rest of this in cygwin.

--------------------------------------------------------
  Building
--------------------------------------------------------
  
--------------------------------------------------------
unix
--------------------------------------------------------
After mapping the files, you will need to change your current working directory to this
and then run make (the path to build on the wiki doesn't seem to work) (note, p4 where
requires a file, so I provide a dummy one that I used dirname to get rid of later):

d=`p4 where //sw/dev/gpu_drv/module_compiler/drivers/compiler/build/dummy | awk '{print $3}'`;cd `dirname $d`
make tools
------------------------------------------
windows
------------------------------------------
cd C:\sw\dev\gpu_drv\module_compiler\drivers\compiler
build\lwmake.exe tools

Continue to use lwmake.exe instead of make below
--------------------------------------------------------

This should make files appropriate for a private build.  p4 edit the files in
generated_private and copy the new ones over:

d=`p4 where //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/generated_private/ptx_parser.c | awk '{print $3}'`;d=`dirname $d`
p4 edit $d/...
for h in cmdoptMessageDefs.c cmdoptMessageDefs.h g_lwconfig.h gpuInfoMessageDefs.c gpuInfoMessageDefs.h GPUSpec.inc ptx_lexer.c ptx_parser.c ptx_parser.h ptxAs.h ptxInstructionDefs.incl ptxInstructionMacrosFermi.c ptxInstructions.h ptxparseMessageDefs.c ptxparseMessageDefs.h stdMessageDefs.c stdMessageDefs.h threadsMessageDefs.c threadsMessageDefs.h; do cp `find . -name $h` $d;done

The public ones take a bit more care to generate.  You need to set the environment
variable RELEASE to be true (very important!!), then make the project with a bunch of
additional flags. LW_VERBOSE=1 displays flags.  You'll want to run this command twice,
once with the 'clobber' target and once with the 'tools' target.

elw RELEASE=1 make LWCFG_OPTIONS="--override=LWCFG_GLOBAL_FEATURE_PTX_ISA_FUTURE:disable --override=LWCFG_GLOBAL_ARCH_PASCAL:disable --override=LWCFG_GLOBAL_ARCH_VOLTA:disable --override=LWCFG_GLOBAL_FEATURE_PTX_ISA_INTERNAL:disable --override=LWCFG_GLOBAL_FEATURE_PTX_ISA_VERSION_43:enable --override=LWCFG_GLOBAL_GPU_FAMILY_GP00X:disable --override=LWCFG_GLOBAL_GPU_FAMILY_GP10X:disable --override=LWCFG_GLOBAL_GPU_FAMILY_GV10X:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP000:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP100:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP102:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP104:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP106:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP107:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP107F:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP10B:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP10D:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP10E:disable --override=LWCFG_GLOBAL_GPU_FAMILY_GP10XF:disable --override=LWCFG_GLOBAL_GPU_IMPL_GV100:disable" clobber
rm -Rf built
elw RELEASE=1 make LWCFG_OPTIONS="--override=LWCFG_GLOBAL_FEATURE_PTX_ISA_FUTURE:disable --override=LWCFG_GLOBAL_ARCH_PASCAL:disable --override=LWCFG_GLOBAL_ARCH_VOLTA:disable --override=LWCFG_GLOBAL_FEATURE_PTX_ISA_INTERNAL:disable --override=LWCFG_GLOBAL_FEATURE_PTX_ISA_VERSION_43:enable --override=LWCFG_GLOBAL_GPU_FAMILY_GP00X:disable --override=LWCFG_GLOBAL_GPU_FAMILY_GP10X:disable --override=LWCFG_GLOBAL_GPU_FAMILY_GV10X:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP000:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP100:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP102:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP104:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP106:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP107:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP107F:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP10B:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP10D:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP10E:disable --override=LWCFG_GLOBAL_GPU_FAMILY_GP10XF:disable --override=LWCFG_GLOBAL_GPU_IMPL_GV100:disable" tools

d=`p4 where //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/generated_public/ptx_parser.c | awk '{print $3}'`;d=`dirname $d`
p4 edit $d/...
for h in cmdoptMessageDefs.c cmdoptMessageDefs.h g_lwconfig.h gpuInfoMessageDefs.c gpuInfoMessageDefs.h GPUSpec.inc ptx_lexer.c ptx_parser.c ptx_parser.h ptxAs.h ptxInstructionDefs.incl ptxInstructionMacrosFermi.c ptxInstructions.h ptxparseMessageDefs.c ptxparseMessageDefs.h stdMessageDefs.c stdMessageDefs.h threadsMessageDefs.c threadsMessageDefs.h; do cp `find . -name $h` $d;done

# For rtmain only
Reimplement #if OPTIX_DOES_SUPPORT_PRAGMAS in addPragmaStatement in both generated_private/ptx_parser.c and generated_public/ptx_parser.c.

You have to watch what it does when it configures the build.  There will be a part right
at the beginning where it tells you what it is enabling.  It looks this.  What you want to
be careful of is the architectures supported as well as the PTX_ISA_VERSION list in the
FTS Features section.  In addition you should double check the
generated_public/ptxInstructionDefs.incl file.  This file will be filtered by the macros
matching this feature list below.  See ptxparse/ptxInstructionDefs.table for the before
files with all the #if's inside it.

When passing these lwconfig args, if you disable an arch or family, then you need to
individually disable all of the families or impls (chips) within that. If you don't, it
will error out. Just disable the impls it errors on and try again.

For OptiX 3.7:
build\lwmake.exe LWCFG_OPTIONS="--override=LWCFG_GLOBAL_ARCH_MAXWELL:enable --override=LWCFG_GLOBAL_GPU_FAMILY_GM10X:enable --override=LWCFG_GLOBAL_GPU_FAMILY_GM20X:enable --override=LWCFG_GLOBAL_GPU_IMPL_GM200:enable  --override=LWCFG_GLOBAL_GPU_IMPL_GM108:enable --override=LWCFG_GLOBAL_GPU_IMPL_GM107:enable --override=LWCFG_GLOBAL_GPU_IMPL_GM204:enable --override=LWCFG_GLOBAL_GPU_IMPL_GM206:enable --override=LWCFG_GLOBAL_GPU_IMPL_GM20B:enable --override=LWCFG_GLOBAL_GPU_IMPL_GM20D:enable --override=LWCFG_GLOBAL_ARCH_PASCAL:disable --override=LWCFG_GLOBAL_GPU_FAMILY_GP00X:disable --override=LWCFG_GLOBAL_GPU_FAMILY_GP10X:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP000:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP100:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP107:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP10B:disable --override=LWCFG_GLOBAL_ARCH_KEPLER:enable --override=LWCFG_GLOBAL_ARCH_FERMI:enable --override=LWCFG_GLOBAL_FEATURE_PTX_ISA_INTERNAL:disable --override=LWCFG_GLOBAL_FEATURE_PTX_ISA_VERSION_41:enable" tools

For OptiX 3.8:
build\lwmake.exe LWCFG_OPTIONS="--override=LWCFG_GLOBAL_ARCH_FERMI:enable --override=LWCFG_GLOBAL_ARCH_KEPLER:enable --override=LWCFG_GLOBAL_ARCH_MAXWELL:enable --override=LWCFG_GLOBAL_ARCH_PASCAL:disable --override=LWCFG_GLOBAL_ARCH_VOLTA:disable --override=LWCFG_GLOBAL_FEATURE_PTX_ISA_INTERNAL:disable --override=LWCFG_GLOBAL_FEATURE_PTX_ISA_VERSION_42:enable --override=LWCFG_GLOBAL_FEATURE_PTX_ISA_VERSION_43:disable --override=LWCFG_GLOBAL_GPU_FAMILY_GP00X:disable --override=LWCFG_GLOBAL_GPU_FAMILY_GP10X:disable --override=LWCFG_GLOBAL_GPU_FAMILY_GV10X:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP000:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP100:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP102:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP104:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP107:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP10B:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP10D:disable --override=LWCFG_GLOBAL_GPU_IMPL_GP10E:disable --override=LWCFG_GLOBAL_GPU_IMPL_GV100:disable --override=LWCFG_GLOBAL_GPU_IMPL_GV10B:disable" tools



==========================================================================================
      Profile: lwda_macosx_unified_internal_profile

Architectures: FERMI KEPLER MAXWELL TESLA
     Families: G8X G9X GF10X GF10XF GF11X GK10X GK11X GK20X GM10X GM20X GT20X GT21X
      Devices: GK104 GK106 GK107 GK110 GK110B GK110C GK208 GK20A GM107 GM108 GM200 GM204 GM206 GM20B GM20D
     Features: LWCA                                    LWDA_TOOLKIT_6_5                        
               LWDA_TOOLKIT_7_0                        DISPLAYPORT                             
               ... < bunch of stuff I left out >
 FTS Features: ASN191_NON_VOLATILE_LWAPI_DISPLAY_ID    COMPUTE_COMPILER_INTERNAL               
               ... < bunch of stuff I left out >
               FSN993_SPARSE_TEXTURE                   LWML_7_0                                
               OPENGL_T114_EXTENSIONS                  PTX_ISA_VERSION_32                      
               PTX_ISA_VERSION_40                      PTX_ISA_VERSION_41                      
               PTX_ISA_VERSION_42                      PTX_ISA_VERSION_43                      
               ... < bunch of stuff I left out >
               SRS1967_LWDA_GEFORCE_DOUBLE_PRECISION   TWINPEAKS                               
    Overrides:
               Disable LWCFG_GLOBAL_FEATURE_PTX_ISA_FUTURE
               Disable LWCFG_GLOBAL_ARCH_PASCAL
               Disable LWCFG_GLOBAL_ARCH_VOLTA
               Disable LWCFG_GLOBAL_FEATURE_PTX_ISA_INTERNAL
               Enable LWCFG_GLOBAL_FEATURE_PTX_ISA_VERSION_43
               Disable LWCFG_GLOBAL_GPU_FAMILY_GP00X
               Disable LWCFG_GLOBAL_GPU_FAMILY_GP10X
               Disable LWCFG_GLOBAL_GPU_FAMILY_GV10X
               Disable LWCFG_GLOBAL_GPU_IMPL_GP000
               Disable LWCFG_GLOBAL_GPU_IMPL_GP100
               Disable LWCFG_GLOBAL_GPU_IMPL_GP102
               Disable LWCFG_GLOBAL_GPU_IMPL_GP104
               Disable LWCFG_GLOBAL_GPU_IMPL_GP106
               Disable LWCFG_GLOBAL_GPU_IMPL_GP107
               Disable LWCFG_GLOBAL_GPU_IMPL_GP107F
               Disable LWCFG_GLOBAL_GPU_IMPL_GP10B
               Disable LWCFG_GLOBAL_GPU_IMPL_GP10D
               Disable LWCFG_GLOBAL_GPU_IMPL_GP10E
               Disable LWCFG_GLOBAL_GPU_FAMILY_GP10XF
               Disable LWCFG_GLOBAL_GPU_IMPL_GV100
==========================================================================================

******************************************************************************   
   2. Integrate remaining parser files
******************************************************************************

Once you have the generated files, you need to integrate the rest of the parser.  There
are two mappings for both goldenrod and rtmain.

p4 integrate -b ptxparse_rtmain
p4 integrate -b ptxparse_goldenrod

p4 resolve

If any symbols are not found you will probably need to edit CMakeLists.txt to force the
appropriate .h file to be included in the compilation of the offending file. See
add_header_include().

Here is a copy of the rtmain branch spec:

	//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/lib/ptxparse/... //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/ptxparse/...
	-//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/lib/ptxparse/Makefile //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/ptxparse/Makefile
	-//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/lib/ptxparse/Messages.table //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/ptxparse/Messages.table
	-//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/lib/ptxparse/geninstr.c //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/ptxparse/geninstr.c
	//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/tools/jas/1.0/lib/utilities/std/... //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/std/...
	-//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/tools/jas/1.0/lib/utilities/std/Makefile //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/std/Makefile
	-//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/tools/jas/1.0/lib/utilities/std/Messages.table //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/std/Messages.table
	-//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/tools/jas/1.0/lib/utilities/std/stdBitSet.c //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/std/stdBitSet.c
	-//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/tools/jas/1.0/lib/utilities/std/stdEndianTypes.h //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/std/stdEndianTypes.h
	-//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/tools/jas/1.0/lib/utilities/std/stdMacroCodeGen.h //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/std/stdMacroCodeGen.h
	-//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/tools/jas/1.0/lib/utilities/std/stdSharedLibraries.c //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/std/stdSharedLibraries.c
	-//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/tools/jas/1.0/lib/utilities/std/stdSharedLibraries.h //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/std/stdSharedLibraries.h
	//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/tools/jas/1.0/lib/utilities/cmdopt/... //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/cmdopt/...
	-//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/tools/jas/1.0/lib/utilities/cmdopt/Makefile //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/cmdopt/Makefile
	-//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/tools/jas/1.0/lib/utilities/cmdopt/Messages.table //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/cmdopt/Messages.table
	-//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/tools/jas/1.0/lib/utilities/cmdopt/stdCommandParse.inc //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/cmdopt/stdCommandParse.inc
	//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/tools/jas/1.0/lib/utilities/threads/... //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/threads/...
	-//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/tools/jas/1.0/lib/utilities/threads/Makefile //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/threads/Makefile
	-//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/tools/jas/1.0/lib/utilities/threads/Messages.table //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/threads/Messages.table
	//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/lib/gpuInfo/gpuInfo.c //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/gpuInfo/gpuInfo.c
	//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/lib/gpuInfo/gpuInfo.h //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/gpuInfo/gpuInfo.h
	//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/lib/ptxopt/ptxOptEnums.h //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/ptxopt/ptxOptEnums.h
	//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/lib/ptxdci/API.h //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/ptxdci/API.h
	//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/lib/ptxdci/ptxDCI.h //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/ptxdci/ptxDCI.h
	//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/lib/utilities/compilerTools/ctMessages.c //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/ptxparse/ctMessages.c
	//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/lib/utilities/compilerTools/ctMessages.h //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/ptxparse/ctMessages.h
	//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/lib/utilities/compilerTools/ctArch.c //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/ptxparse/ctArch.c
	//sw/dev/gpu_drv/module_compiler/drivers/compiler/gpgpucomp/lib/utilities/compilerTools/ctArch.h //sw/wsapps/raytracing/rtsdk/rtmain/src/FrontEnd/PTX/PTXStitch/ptxparse/ctArch.h


-------------------------------------------------------------------------------
OPTIX EDITS
-------------------------------------------------------------------------------
You might need to fix some warnings.  I wanted to disable as many warnings as possible for
the parser code, since we don't want to fix their warnings so much.  Any hand
modifications should be marked with OPTIX_HAND_EDIT in a comment.

Most other changes are marked by OPTIX EDIT comments.  Most changes should be superficial,
since the PTXFrontEnd should abstract a lot of the changes to the parser.

******************************************************************************   
   3. Changes to our usage
******************************************************************************

Once it builds correctly, you will probably hit runtime errors. Most of these can be
solved by running in separate trees with the old and new parsers and seeing where the
behavior diverges.

Most of the changes will be in the ptx front end.
1. PTXFrontEnd.cpp - adding new types (such as half float)
2. ptx_lib_gen.py
3. ptx_instr_enum.py

There are several places where things can go wrong.  You should double check:

generated_*/ptxInstructions.h for any new or different instructions.

ptx.l for new modifiers.  These will need to be dealt with in ptx_instr_enum.py.  There
are comments in ptx_instr_enum.py on how to deal with these enums.


******************************************************************************   
   Handy notes
******************************************************************************

Older versions of this file (in Perforce) have older notes.

p4 describe 5661115
People who know stuff about this: Sharad Sanap, Richard Johnson, Fahim Shafi, Gautam Chakrabarti

Try these elw. variables:
./scripts/rules.mk:     $(CYGWIN_INSTALL)/bin/dos2unix.exe $(<F:.y=_parser.h)
./build/Platform.mk:export DOS2UNIX           := $(TOOL_PREFIX)dos2unix
Can you please try setting CYGWIN_INSTALL=<p4_root>/sw/tools/win32/cygnus/Apr2008 and also add that to your p4 client (and sync) if it is not there already.
