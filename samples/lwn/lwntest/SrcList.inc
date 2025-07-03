# -*- makefile -*-
# tests/OpenGL/lwntest/SrcList.inc
#
# List of source files for lwntest.
# Must be included in a separate makefile...
#
#   Windows: makefile.lwmk
#   MacOSX:  makefile.lwmk
#   Linux:   makefile.lwmk
#   MODS:    mods/Makefile
#   Android: Android.mk
#
# Please see this Wiki page for more information:
#   http://swwiki/index.php/Building_lwntest
#
# Copyright (c) 2004-2020 by LWPU Corporation.  All rights reserved.
##############################################################################


# This file sets some variables:
#
#   LWNTEST_CFILES    = List of .c files to build
#   LWNTEST_CCFILES    = List of .cc files to build
#   LWNTEST_CPPFILES    = List of .cpp files to build
#
# In addition, the above variables with _PCH and _NO_PCH suffixes are set.  The
# former are built with pre-compiled headers enabled (include/ogtest.h for the
# C files, include/cppogtest.h for the C++ files), and the latter are not.
#
# For platform-specific source files, the variable LWNTEST_PLATFORM should
# be set by the makefile that includes this file.  Valid values:
#
#   win = Standalone lwntest (Windows)
#   hos = Standalone lwntest (HOS)
#

ifeq ($(filter win hos linux,$(LWNTEST_PLATFORM)),)
$(error LWNTEST_PLATFORM is not supported.)
endif

SAMPLES_COMMON_DIR ?= ../../lwn/samples/common


########################################################################
#
# Core lwntest C files that have been configured to use pre-compiled headers.
#
LWNTEST_CFILES_PCH =

########################################################################
#
# Core lwntest C files that have not been configured to use pre-compiled
# headers or can not use them.
#
LWNTEST_CFILES_NO_PCH =

########################################################################
#
# Core lwntest C++ files that have been configured to use pre-compiled headers
# (none yet).
#
LWNTEST_CPPFILES_PCH  =
LWNTEST_CPPFILES_PCH += elw/cells.cpp
LWNTEST_CPPFILES_PCH += elw/cmdline.cpp
ifeq ($(LWNTEST_LWDA_ENABLED),1)
LWNTEST_CPPFILES_PCH += elw/lwda_interop.cpp
endif
LWNTEST_CPPFILES_PCH += elw/float_util.cpp
LWNTEST_CPPFILES_PCH += elw/global.cpp
LWNTEST_CPPFILES_PCH += elw/md5.cpp
LWNTEST_CPPFILES_PCH += elw/lwnTest_AllocationTracker.cpp
LWNTEST_CPPFILES_PCH += elw/lwnTest_Formats.cpp
LWNTEST_CPPFILES_PCH += elw/lwnTest_Mislwtils.cpp
LWNTEST_CPPFILES_PCH += elw/lwnTest_DeviceState.cpp
LWNTEST_CPPFILES_PCH += elw/lwnTest_Objects.cpp
LWNTEST_CPPFILES_PCH += elw/lwnTest_Framebuffer.cpp
LWNTEST_CPPFILES_PCH += elw/lwnTest_VertexState.cpp
LWNTEST_CPPFILES_PCH += elw/lwnTest_GlslcHelper.cpp
LWNTEST_CPPFILES_PCH += elw/lwnTest_WindowFramebuffer.cpp
LWNTEST_CPPFILES_PCH += elw/rand.cpp
LWNTEST_CPPFILES_PCH += elw/sums.cpp
LWNTEST_CPPFILES_PCH += elw/str_util.cpp
LWNTEST_CPPFILES_PCH += elw/testloop.cpp
LWNTEST_CPPFILES_PCH += elw/tests.cpp

########################################################################
#
# Core lwntest C++ files that have not been configured to use pre-compiled
# headers or can not use them.
#
LWNTEST_CPPFILES_NO_PCH =
LWNTEST_CPPFILES_NO_PCH += elw/cppcheck.cpp
LWNTEST_CPPFILES_NO_PCH += elw/cppshaders.cpp
ifeq ($(LWNTEST_PLATFORM),win)
LWNTEST_CPPFILES_NO_PCH += elw/windows_main.cpp
else
LWNTEST_CPPFILES_NO_PCH += elw/tegra_main.cpp
LWNTEST_CPPFILES_NO_PCH += elw/tegra_utils.cpp
endif
LWNTEST_CPPFILES_NO_PCH += elw/lwn_interface_c.cpp
LWNTEST_CPPFILES_NO_PCH += elw/lwn_interface_cpp.cpp
LWNTEST_CPPFILES_NO_PCH += elw/lwn_utils.cpp
LWNTEST_CPPFILES_NO_PCH += elw/readpixels.cpp
LWNTEST_CPPFILES_NO_PCH += elw/tga.cpp

ifneq ($(filter $(LWNTEST_PLATFORM),hos linux),)
LWNTEST_CPPFILES_NO_PCH += ../common/lwnWin/lwn_win.cpp
LWNTEST_CPPFILES_NO_PCH += ../common/lwnWin/lwwinsys_win.cpp
ifeq ($(LWNTEST_PLATFORM),linux)
LWNTEST_CPPFILES_NO_PCH += ../common/lwnWin/wayland_win.cpp
LWNTEST_CFILES_NO_PCH   += ../common/lwnWin/dmabuf-protocol.c
LWNTEST_CFILES_NO_PCH   += ../common/lwnWin/synchronization-protocol.c
endif
endif

# For SPIR-V functionality
# Builds in shaderc/glslang for GLSL -> SPIR-V compilation
# Disable this flag to compile lwntest without building in
# any shaderc/glslang sources or SPIR-V support in general.
SHADERC_INCLUDE=
LW_LWN_ENABLE_SPIRV ?= 1
ifeq ($(LW_LWN_ENABLE_SPIRV),1)

# spir-v interface file for interacting with shader-c and the glslang compiler
ifeq ($(LWNTEST_PLATFORM),win)
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/OSDependent/Windows/ossource.cpp
else
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/OSDependent/Unix/ossource.cpp
endif
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/OGLCompilersDLL/InitializeDll.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/attribute.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/Constant.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/InfoSink.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/Initialize.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/IntermTraverse.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/Intermediate.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/ParseHelper.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/PoolAlloc.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/RemoveTree.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/Scan.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/ShaderLang.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/SymbolTable.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/Versions.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/glslang_tab.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/intermOut.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/iomapper.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/limits.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/linkValidate.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/parseConst.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/ParseContextBase.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/reflection.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/preprocessor/Pp.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/preprocessor/PpAtom.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/preprocessor/PpContext.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/preprocessor/PpScanner.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/preprocessor/PpTokens.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/MachineIndependent/propagateNoContraction.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/GenericCodeGen/CodeGen.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/glslang/GenericCodeGen/Link.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/hlsl/hlslAttributes.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/hlsl/hlslGrammar.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/hlsl/hlslOpMap.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/hlsl/hlslParseables.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/hlsl/hlslParseHelper.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/hlsl/hlslScanContext.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/hlsl/hlslTokenStream.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/SPIRV/GlslangToSpv.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/SPIRV/InReadableOrder.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/SPIRV/Logger.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/SPIRV/SpvBuilder.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/SPIRV/SpvPostProcess.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/SPIRV/SPVRemapper.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/SPIRV/SpvTools.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/SPIRV/doc.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/third_party/glslang/SPIRV/disassemble.cpp
LWNTEST_CPPFILES_NO_PCH += shaderc/libshaderc_util/src/compiler.cc
LWNTEST_CPPFILES_NO_PCH += shaderc/libshaderc_util/src/file_finder.cc
ifneq ($(LWNTEST_PLATFORM),hos)
# This source file, which uses std::cin, causes a build break on HOS. Remove it from the HOS build,
# where it is lwrrently unused.
LWNTEST_CPPFILES_NO_PCH += shaderc/libshaderc_util/src/io.cc
endif
LWNTEST_CPPFILES_NO_PCH += shaderc/libshaderc_util/src/message.cc
LWNTEST_CPPFILES_NO_PCH += shaderc/libshaderc_util/src/resources.cc
LWNTEST_CPPFILES_NO_PCH += shaderc/libshaderc_util/src/shader_stage.cc
LWNTEST_CPPFILES_NO_PCH += shaderc/libshaderc_util/src/version_profile.cc
LWNTEST_CPPFILES_NO_PCH += shaderc/libshaderc/src/shaderc.cc

SHADERC_INCLUDE += shaderc/libshaderc_util/include/
SHADERC_INCLUDE += shaderc/third_party/glslang/
SHADERC_INCLUDE += shaderc/third_party/spirv-tools/
SHADERC_INCLUDE += shaderc/libshaderc/include
SHADERC_INCLUDE += shaderc/third_party/spirv-tools/include/
SHADERC_INCLUDE += shaderc/third_party/spirv-tools/source/
SHADERC_INCLUDE += shaderc/third_party/spirv-tools/external/spirv-headers/include/
SHADERC_INCLUDE += shaderc/third_party/spirv-tools/external/spirv-headers/include/spirv/unified1
LW_INCLUDES += $(SHADERC_INCLUDE)

endif # ifeq ($(LW_LWN_ENABLE_SPIRV),1)

########################################################################
#
# LWN tests using the pure C interface for LWN.
#
LWNTEST_C_INTERFACE_TESTS =
LWNTEST_C_INTERFACE_TESTS += tests/lwn/lwn_01tri_c.cpp
LWNTEST_C_INTERFACE_TESTS += tests/lwn/lwn_mempool.cpp
LWNTEST_C_INTERFACE_TESTS += tests/lwn/lwn_natvis_c.cpp

########################################################################
#
# LWN tests using the pure C++ interface for LWN.
#
LWNTEST_CPP_INTERFACE_TESTS =
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_01tri_cpp.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_alloc.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_alphatest.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_basic.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_basic_cpp.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_bgr5.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_bigtex.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_bind_deferred.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_blendadv.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_c2c.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_clear_buffer.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_clear_texture.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_clear_tex_overwrite.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_client_allocator.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_cmdbuf_add_memory.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_cmdbuf_callcopy.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_cmdbuf_disable_oom.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_color_reduction.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_compare_objects.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_compression.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_compute.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_conditional_render.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_conservative_raster.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_copy_compressible.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_copy_image.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_copy_image_mipmaps.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_copy_image_mirror.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_copy_image_ms.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_copy_texture_limits.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_copy_texture_stride.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_counters.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_counters_zlwll.cpp
ifeq ($(LWNTEST_LWDA_ENABLED),1)
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_lwda_interop.cpp
endif
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_dbounds.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_debug.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_debug_walk.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_debuglabel.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_defaults.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_depth_range.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_depth_stencil.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_depth_write_war.cpp
ifneq ($(filter $(LWNTEST_PLATFORM),hos linux),)
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_disable_both_compression.cpp
endif
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_downsample.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_draw_basic.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_draw_texture.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_event.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_fastpath.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_force_sample_shading.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_formats.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_fs_interlock.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_getters.cpp
ifneq ($(LWNTEST_PLATFORM),linux)
# LWN/GL interop not supported on L4T
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_gl_interop.cpp
endif
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_glsl.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_glsl_bindings.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_glsl_bindings_unused.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_glsl_derivative_control.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_glsl_initialized_array.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_glslc.cpp
ifeq ($(LWNTEST_PLATFORM),win)
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_glslc_include.cpp
endif # ifneq ($(LWNTEST_PLATFORM),win)
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_gs_passthrough.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_hlsl.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_image.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_image_targets.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_image_swizzle.cpp
ifeq ($(LWNTEST_PLATFORM),win)
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_internal_shader_pool.cpp
endif # ifneq ($(LWNTEST_PLATFORM),win)
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_lowp_mediump_mixed.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_mempool_external.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_mempool_mapping.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_mempool_virtual.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_mt_texture_creation.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_multi_draw_indirect_count.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_multibind_compute.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_multisample_a2cov.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_natvis_cpp.cpp
ifeq ($(LWNTEST_PLATFORM),hos)
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_vic_interop.cpp
endif
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_polygon_offset.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_pool_bufaddr.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_post_depth_coverage.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_present_interval.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_psprite.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_programs_mt.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_provoking_vertex.cpp
ifneq ($(filter $(LWNTEST_PLATFORM),hos linux),)
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_pte.cpp
endif
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_query.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_raster.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_raw_storage_class_switch.cpp
ifneq ($(filter $(LWNTEST_PLATFORM),hos linux),)
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_robustness.cpp
endif
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_rt_attachments.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_sample_control.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_sample_location.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_sampler.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_separate_sampler.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_shader_scratch.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_shader_specialization.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_shader_spirv.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_shader_subgroup.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_shader_subgroup_ballot_vote.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_shader_subroutine.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_shader_tracking.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_shadowmap.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_sparse_texture.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_ssbo.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_ssbo_extended_atomics.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_stateobjects.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_subtile_override.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_tess.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_tess_tcsparams.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_texbind_random.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_texture_basic.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_texture_binding.cpp
ifneq ($(filter $(LWNTEST_PLATFORM),hos linux),)
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_texture_etc.cpp
endif
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_texture_packager.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_texture_pitch.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_texture_pool.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_texture_rt_compress.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_texture_storagelayout.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_texture_storagesize.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_texture_swizzle.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_texture_views.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_texture_wr_texels.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_tiled_cache.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_timestamp.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_timestamp_colwersion.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_tir.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_trilinopt.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_ubo_zerosize.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_unbound_id.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_unfinalized_queues.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_uniformbuffer.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_util_datatypes.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_vertex_attrib.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_vertex_streams.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_view_offset.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_viewports.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_window.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_xfb_basic.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_zbc.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_zlwll.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_zlwll_layered.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_zlwll_stencil.cpp
LWNTEST_CPP_INTERFACE_TESTS += tests/lwn/lwn_zlwll_zf32.cpp

########################################################################
#
# Headers to include in Visual Studio projects.
#
LWNTEST_HEADERS =
LWNTEST_HEADERS += elw\cmdline.h
LWNTEST_HEADERS += elw\elw.h
LWNTEST_HEADERS += elw\md5.h
LWNTEST_HEADERS += elw\readpixels.h
LWNTEST_HEADERS += elw\sums.h
LWNTEST_HEADERS += elw\testloop.h
LWNTEST_HEADERS += elw\tests.h
LWNTEST_HEADERS += elw\tga.h
LWNTEST_HEADERS += include\cells.h
LWNTEST_HEADERS += include\contexts.h
LWNTEST_HEADERS += include\cppcheck.h
LWNTEST_HEADERS += include\cppogtest.h
LWNTEST_HEADERS += include\cppshaders.h
LWNTEST_HEADERS += include\cppsharedobject.h
LWNTEST_HEADERS += include\cppstring.h
LWNTEST_HEADERS += include\float16.h
LWNTEST_HEADERS += include\floatu10.h
LWNTEST_HEADERS += include\floatu11.h
LWNTEST_HEADERS += include\float_packed.h
LWNTEST_HEADERS += include\float_util.h
LWNTEST_HEADERS += include\lwctassert.h
LWNTEST_HEADERS += include\lwn_utils.h
LWNTEST_HEADERS += include\lwogtypes.h
LWNTEST_HEADERS += include\ogtest.h
LWNTEST_HEADERS += include\ossymbols.h
LWNTEST_HEADERS += include\printf_like.h
LWNTEST_HEADERS += include\shaders.h
LWNTEST_HEADERS += include\str_util.h
LWNTEST_HEADERS += tests\lwn\g_lwn_formats_data.h
LWNTEST_HEADERS += tests\lwn\lwn_basic.h

########################################################################
#
# Full lists of files.
#

LWNTEST_CFILES =
LWNTEST_CFILES += $(LWNTEST_CFILES_PCH)
LWNTEST_CFILES += $(LWNTEST_CFILES_NO_PCH)

LWNTEST_CPPFILES =
LWNTEST_CPPFILES += $(LWNTEST_CPPFILES_PCH)
LWNTEST_CPPFILES += $(LWNTEST_CPPFILES_NO_PCH)
LWNTEST_CPPFILES += $(LWNTEST_C_INTERFACE_TESTS)
LWNTEST_CPPFILES += $(LWNTEST_CPP_INTERFACE_TESTS)

LWNTEST_CCFILES =
