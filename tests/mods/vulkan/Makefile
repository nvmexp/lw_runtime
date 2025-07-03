#/*
# * LWIDIA_COPYRIGHT_BEGIN
# *
# * Copyright 2017-2021 by LWPU Corporation.  All rights reserved.  All
# * information contained herein is proprietary and confidential to LWPU
# * Corporation.  Any use, reproduction, or disclosure without the written
# * permission of LWPU Corporation is prohibited.
# *
# * LWIDIA_COPYRIGHT_END
# */

#-------------------------------------------------------------------------------
# Standalone Vulkan tests make file.
# See //sw/<tree>/diag/mods/makedefs.inc for more information about the MODS
# makefiles.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Targets:
#
#    all             Build the exelwtable, default target.
#    clean           Delete all object files.
#-------------------------------------------------------------------------------

MODS_DIR_ORG ?= ../../mods
MODS_DIR     ?= $(MODS_DIR_ORG)
DRIVERS_DIR  ?= ../../../drivers
ROOT_DIR     ?= .

ENABLE_UNIT_TESTING ?= false
ENABLE_OCGRAPHICSTEST ?= false
ENABLE_POWERPULSETEST ?= false
ENABLE_LIBVKSTRESS ?= false

dll_build = false
base_name = vktest
dedicated_app = false
ifeq "$(ENABLE_UNIT_TESTING)" "true"
    base_name = vktestut
else ifeq "$(ENABLE_OCGRAPHICSTEST)" "true"
    base_name = ocgraphicstest
    dedicated_app = true
else ifeq "$(ENABLE_POWERPULSETEST)" "true"
    base_name = powerpulse
    dedicated_app = true
else ifeq "$(ENABLE_LIBVKSTRESS)" "true"
    base_name = libvkstress
    dedicated_app = true
endif

MODS_OUTPUT_DIR ?= $(abspath $(ROOT_DIR)/artifacts)
MODS_RUNSPACE ?= $(MODS_OUTPUT_DIR)

ifeq "$(MAKE_HOST)" "Windows32"
    override BUILD_OS := win32
else
    override BUILD_OS := $(shell uname)
    ifeq "$(BUILD_OS)" "Linux"
        override BUILD_OS := linux
    else
        $(error Standalone Vulkan test does not support your host OS with uname=$(BUILD_OS))
    endif
endif

include $(MODS_DIR)/makedefs.inc

ifeq "$(ENABLE_LIBVKSTRESS)" "true"
    bin_name = $(lib_name)
else
    bin_name = $(exe_name)
endif

GLSLANG_PARENT_DIR_VKTEST := $(COMMON_WS_DIR)/sw/apps/gpu/drivers/opengl/lwogtest/thirdparty/shaderc/third_party
GLSLANG_PARENT_DIR := $(GLSLANG_PARENT_DIR_VKTEST)

VULKAN_SDK_DIR := $(COMMON_WS_DIR)/sw/mods/vulkan/validationlayers/1.2.162.1

defines += -DVULKAN_STANDALONE
defines += -DINCLUDE_STD_NAMESPACE
defines += -DUSE_NEW_TASKER

ifneq "$(dedicated_app)" "true"
    defines += -DVULKAN_STANDALONE_KHR
    defines += -DVULKAN_ENABLE_DEBUG_MARKERS
endif
ifeq "$(ENABLE_POWERPULSETEST)" "true"
    defines += -DVULKAN_STANDALONE_POWERPULSE
endif

cpp_files += $(MODS_DIR_ORG)/core/cpu/cpu.cpp
cpp_files += $(MODS_DIR_ORG)/core/filetype/png.cpp
cpp_files += $(MODS_DIR_ORG)/core/main/script.cpp
cpp_files += $(MODS_DIR_ORG)/core/main/taskerm.cpp
cpp_files += $(MODS_DIR_ORG)/core/main/thread_mill.cpp
cpp_files += $(MODS_DIR_ORG)/core/utility/color.cpp
cpp_files += $(MODS_DIR_ORG)/core/utility/memory.cpp
cpp_files += $(MODS_DIR_ORG)/core/utility/ptrnclss.cpp
cpp_files += $(MODS_DIR_ORG)/core/utility/coreutility.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/swapchain.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/util.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/vkasgen.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/vkbuffer.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/vkcmdbuffer.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/vkdescriptor.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/vkdev.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/vkfence.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/vkframebuffer.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/vkgoldensurfaces.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/vkimage.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/vkinstance.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/vkphysdev.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/vkpipeline.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/vkrenderpass.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/vksampler.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/vksemaphore.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/vkshader.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/vktexfill.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/vktexture.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/vkutil.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/vkerror.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/vkmods.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/vkquery.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/tests/vkstress.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/tests/vkstress/vkstressray.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/tests/vkfusion.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/tests/vkfusion/vkfusion_job.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/tests/vkfusion/vkfusion_graphics.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/tests/vkfusion/vkfusion_raytracing.cpp
cpp_files += $(MODS_DIR_ORG)/vulkan/tests/vkfusion/vkfusion_mats.cpp
ifneq "$(dedicated_app)" "true"
    cpp_files += $(MODS_DIR_ORG)/vulkan/tests/vktriangle.cpp
endif
cpp_files += $(ROOT_DIR)/util_init.cpp
cpp_files += $(ROOT_DIR)/vkmodssub.cpp

ifeq "$(ENABLE_UNIT_TESTING)" "true"
    defines += -DENABLE_UNIT_TESTING=1

    cpp_files += $(MODS_DIR_ORG)/vulkan/unittesting/vklayersut.cpp
    cpp_files += $(MODS_DIR_ORG)/vulkan/unittesting/vulkandrivermock.cpp
    cpp_files += $(ROOT_DIR)/vktestut.cpp
else ifeq "$(dedicated_app)" "true"
    cpp_files += $(ROOT_DIR)/vulkan/vklayerssub.cpp
    cpp_files += $(ROOT_DIR)/$(base_name).cpp
else
    cpp_files += $(ROOT_DIR)/vulkan/vklayerssub.cpp
    cpp_files += $(ROOT_DIR)/vktest.cpp
endif

defines += -DINCLUDE_VKLAYERS=1

force_include_files += $(MODS_DIR)/$(base_os)/modsinc.h
force_include_files += $(MODS_DIR)/core/include/jsprotect.h

VULKAN_INCLUDE_DIR = $(VULKAN_SDK_DIR)

include_dirs += $(BOOST_DIR)
include_dirs += $(MODS_DIR)
include_dirs += $(MODS_DIR)/../../sdk/lwpu/inc
include_dirs += $(MODS_DIR)/../js1.7
include_dirs += $(MODS_DIR)/../utils
include_dirs += $(MODS_DIR)/../standalonemodstests/vulkan
include_dirs += $(MODS_DIR)/vulkan
include_dirs += $(BUILD_TOOLS_DIR)/mods/glm/0.9.9.7/glm

# For GLSL2SPV, needs to be in front of the sdk to pickup the proper TBuiltInResource:
include_dirs += $(GLSLANG_PARENT_DIR_VKTEST)/glslang
include_dirs += $(VULKAN_INCLUDE_DIR)/install/include
include_dirs += $(VULKAN_INCLUDE_DIR)/install/include/vulkan

$(o_dir)/png$(obj_suffix): include_dirs += $(BUILD_TOOLS_DIR)/lpng1251 $(MODS_DIR_ORG)/core/include

libraries += boost
libraries += js17
libraries += lwdiag_utils
libraries += png
libraries += z
libraries += glslang

lib_dirs  += $(MODS_OUTPUT_DIR)/$(rel_o_dir)/libglslang
lib_dirs  += $(MODS_OUTPUT_DIR)/$(rel_o_dir)/lwmake/diag/js1.7
lib_dirs  += $(MODS_OUTPUT_DIR)/$(rel_o_dir)/lwmake/diag/utils

ifeq "$(base_os)" "win32"
    defines += -DVK_USE_PLATFORM_WIN32_KHR

    cpp_files += $(MODS_DIR_ORG)/win32/winthread.cpp
    cpp_files += $(ROOT_DIR)/vktest_win32.cpp

    extra_libs += gdi32.lib
    extra_libs += user32.lib
    extra_libs += ws2_32.lib
    extra_libs += crypt32.lib
    extra_libs += selwr32.lib

    # Must turn on C++ exception handling when using the C++ standard library.
    CFLAGS += /EHsc
    # Treat warnings as errors
    warnings += /WX

    # Windows builds have some libraries that are different in debug/release
    # builds. Include those paths here.
    lib_dirs += $(MODS_DIR)/$(base_mods_lib_dir)/$(BUILD_CFG)

    #extra_libs += shaderc_combined.lib
    #lib_dirs += $(shell cygpath -m '$(VULKAN_SDK_DIR)/Lib/msvc140')
else
    defines  += -DVK_USE_PLATFORM_XCB_KHR
    warnings += -Wno-unused-value

    include_dirs += $(COMMON_WS_DIR)/sw/tools/mods/xcb/include
    lib_dirs     += $(COMMON_WS_DIR)/sw/tools/mods/xcb/lib
    libraries    += xcb
    LDFLAGS      += -Wl,-rpath-link=$(COMMON_WS_DIR)/sw/tools/mods/xcb/lib

    cpp_files += $(MODS_DIR_ORG)/linux/posixthread.cpp
    cpp_files += $(ROOT_DIR)/vktest_linux.cpp

    lib_dirs += $(MODS_DIR)/$(base_mods_lib_dir)

    libraries += dl
    libraries += rt
endif

submake.glslang.%:
	$(Q)$(MAKE) -C ../../glslang GLSLANG_PARENT_DIR=$(GLSLANG_PARENT_DIR) MODS_OUTPUT_DIR=$(MODS_OUTPUT_DIR) $* BUILD_OS=$(BUILD_OS)

#-------------------------------------------------------------------------------
# Targets:
#-------------------------------------------------------------------------------

.PHONY: all clean

# The default target makes the vktest exelwtable (vktest.exe).
all: build

build: print_start_build_message $(o_dir)/$(exe_name) print_end_build_message

include $(MODS_DIR_UNIX)/makerule.inc

$(o_dir)/$(exe_name) : submake.glslang.build $(o_files)
	@$(ECHO) "Linking object files and libraries: $(libraries)"
	@$(ECHO) "Exe $@"
	$(Q)$(LD) $(LDFLAGS_EXE) $(o_files) $(link_libs)

$(o_dir)/$(lib_name) : submake.glslang.build $(o_files)

$(o_dir)/$(bin_name): submake.lwdiag_utils.build

submake.lwdiag_utils.%:
	$(Q)$(MAKE) -C $(MODS_DIR)/../utils -f makefile.lwmk $(subst submake.lwdiag_utils.,,$@) $(LWMAKE_ARGS) $*

LWCFG_ROOT ?= ../../../drivers/common/build
LWCFG_OUTPUTDIR ?= $(gen_cpp_dir)
include $(LWCFG_ROOT)/lwconfig.lwmk
include_dirs += $(LWCFG_OUTPUTDIR)

define LWMAKE_COMPONENT
submake.$1.build:
	$$(ECHO) $$(Q)$$(MAKE) -C $2 -f makefile.lwmk build_install_target $$(LWMAKE_ARGS)
	$$(Q)$$(MAKE) -C $2 -f makefile.lwmk build_install_target $$(LWMAKE_ARGS)

$(o_dir)/$(bin_name): submake.$1.build
endef

$(eval $(call LWMAKE_COMPONENT,js17,../../js1.7))

install:
	@$(ECHO) "Copying $(o_dir)/$(exe_name) to $(MODS_RUNSPACE)..."
	$(Q)$(MKDIR_P) $(MODS_RUNSPACE)
	$(Q)$(CP_IF_CHANGED) $(CP_RM_DEST) $(o_dir)/$(exe_name) $(MODS_RUNSPACE)/$(exe_name)

uninstall:
	@$(ECHO) "Deleting installed files."
	$(Q)-$(RM) -f $(install_target_dir)/$(exe_name)

build_install: build
	@$(ECHO) "Copying $(o_dir)/$(exe_name) to $(MODS_RUNSPACE)..."
	$(Q)$(MKDIR_P) $(MODS_RUNSPACE)
	$(Q)$(RM) $(MODS_RUNSPACE)/$(exe_name)
	$(Q)$(CP_IF_CHANGED) $(CP_RM_DEST) $(o_dir)/$(exe_name) $(MODS_RUNSPACE)/$(exe_name)

build_lib: $(o_dir)/$(lib_name)
	@$(ECHO) "Creating combined static lib as $(MODS_RUNSPACE)/$(lib_name)"
	$(Q)$(MKDIR_P) $(MODS_RUNSPACE)
	$(Q)$(RM) $(MODS_RUNSPACE)/$(lib_name)
	$(Q)lib.exe /out:$(MODS_RUNSPACE)/$(lib_name) $(addprefix /LIBPATH:, $(lib_dirs)) $(o_dir)/$(lib_name) $(lib_files)

clobber: clean

print_start_build_message :
	@echo "*********************************************************************"
	@echo "* Building $(BUILD_CFG) version of $(exe_name)."
	@echo "*********************************************************************"

print_end_build_message :
	@echo "Built $(BUILD_CFG) version of $(exe_name)."

clean:
	@echo "Deleting object, lib, and dependency files."
	@-rm -f $(o_dir)/*$(obj_suffix)
	@-rm -f $(o_dir)/*$(lib_suffix)
	@-rm -f $(o_dir)/*$(dll_suffix)
	@-rm -f $(o_dir)/*.d
	@-rm -f $(o_dir)/*.pdb
	@-rm -f $(o_dir)/$(bin_name)
	@-rm -f *.d

# Include dependency information.
-include $(depend_files)
