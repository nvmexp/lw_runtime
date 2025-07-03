#
# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright 2017-2022 by LWPU Corporation.  All rights reserved.  All
# information contained herein is proprietary and confidential to LWPU
# Corporation.  Any use, reproduction, or disclosure without the written
# permission of LWPU Corporation is prohibited.
#
# LWIDIA_COPYRIGHT_END
#

PROTOUTIL_DIR   ?= .
MODS_DIR        ?= $(PROTOUTIL_DIR)/../mods
DRIVERS_DIR     ?= $(PROTOUTIL_DIR)/../../drivers
PROTOBUF_DIR    ?= $(PROTOUTIL_DIR)/../../apps/fabricmanager/libs/protobuf-2.6.0
BUILD_HOST      ?= false

GEN_LIB_DIR  = $(MODS_OUTPUT_DIR)/$(rel_o_dir)
GEN_LIB_DIR_UNIX = $(MODS_OUTPUT_DIR_UNIX)/$(rel_o_dir)

override SYMBOLIZE := false

exe_build = true

ifeq "$(BUILD_HOST)" "true"
    base_name = protoutil_host
else
    base_name = protoutil
endif

# Disable all sanitizers when using this tool at MODS build-time
ifeq "$(BUILD_HOST)" "true"
    override SANITIZER :=
endif

include $(MODS_DIR)/makedefs.inc

include_dirs += $(PROTOUTIL_DIR)
include_dirs += $(BOOST_DIR)
include_dirs += $(BUILD_TOOLS_DIR)/mods/rapidjson
include_dirs += $(MODS_DIR)

ifneq "$(base_os)" "linux"
    $(error protobuf library only supported on linux MODS platforms)
endif
	   
lib_dirs  = $(MODS_DIR)/$(base_mods_lib_dir)

ifeq "$(BUILD_HOST)" "true"
    libraries += protobuf_host
    lib_dirs += $(o_dir)/../lwmake/diag/protoutil/protobuf_host
    include_dirs += $(PROTOBUF_DIR)/include
    include_dirs += $(GEN_LIB_DIR_UNIX)/lwmake/diag/protoutil/protobuf_host
else
    libraries += protobuf
    lib_dirs += $(o_dir)/../lwmake/diag/protoutil/protobuf
    include_dirs += $(PROTOBUF_DIR)/include
    include_dirs += $(GEN_LIB_DIR_UNIX)/lwmake/diag/protoutil/protobuf
endif

# Generate the lwConfig data: LWCFG_* makefile vars and C #defines in g_lwconfig.h
# Note: We cannot include lwconfig.lwmk before makelib.inc in this makefile because the latter
#       defines the overall build target -- and that has to be the 1st target in the makefile

LWCFG_ROOT = $(DRIVERS_DIR)/common/build
LWCFG_OUTPUTDIR = $(gen_cpp_dir)
include $(LWCFG_ROOT)/lwconfig.lwmk
include_dirs += $(LWCFG_OUTPUTDIR)

libraries += boost z

protoutil_files += protoutil.cpp
protoutil_files += protoutil_setecids.cpp
protoutil_files += protoutil_dump.cpp
protoutil_files += protoutil_common.cpp
protoutil_files += protoutil_create.cpp
protoutil_files += protoutil_create_lr.cpp
protoutil_files += protoutil_tograph.cpp
protoutil_files += protoutil_maxtraffic.cpp
protoutil_files += protoutil_routing.cpp
protoutil_files += protoutil_routing_lr.cpp

cpp_files += $(protoutil_files)

LDFLAGS += -static-libstdc++

defines += -DMASSERT=assert

.PHONY : all clean install uninstall build_all build_install clean_all

install_target_dir = $(MODS_OUTPUT_DIR)/$(rel_o_dir)/mods

ifneq "$(filter 7.%,$(toolchain_rev))" ""
    CFLAGS += -Wno-placement-new
endif

# The default target makes the protoutil or protoutil_host program
all build_all:
	$(Q)$(MAKE) $(o_dir)/$(exe_name) 

clean:
	@echo Cleaning object, binary, and dependency files.
	@-rm -rf $(gen_cpp_dir)
	@-rm -rf $(o_dir)

install:
	@$(ECHO) "Copying $(o_dir)/$(exe_name) to $(install_target_dir)..."
	$(Q)$(MKDIR_P) $(install_target_dir)
	$(Q)$(CP_IF_CHANGED) $(CP_RM_DEST) $(o_dir)/$(exe_name) $(install_target_dir)/$(exe_name)

uninstall:
	@$(ECHO) "Deleting installed files."
	$(Q)-$(RM) -f $(install_target_dir)/$(exe_name)

build_install: build_all
	@$(ECHO) "Copying $(o_dir)/$(exe_name) to $(install_target_dir)..."
	$(Q)$(MKDIR_P) $(install_target_dir)
	$(Q)$(CP_IF_CHANGED) $(CP_RM_DEST) $(o_dir)/$(exe_name) $(install_target_dir)/$(exe_name)

clean_all : clean

protoutil_o_files = $(addprefix $(o_dir)/, $(notdir $(protoutil_files:.cpp=$(obj_suffix))))

ifeq "$(BUILD_HOST)" "true"
clean_all : submake.protobuf_host.clean
all build_all : submake.protobuf_host.build_install
else
clean_all : submake.protobuf.clean
all build_all : submake.protobuf.build_install
endif

$(o_dir)/$(exe_name) : $(o_files)
	@$(ECHO) "Linking object files and libraries: $(libraries)"
	$(Q)$(LD) $(LDFLAGS_EXE) $(o_files) $(link_libs)

define LWMAKE_COMPONENT
submake.$1.build:
	$$(Q)$$(MAKE) -C $2/$3 TOPOLOGY_PROTO_ONLY=1 -f makefile.lwmk build $$(LWMAKE_ARGS)
submake.$1.clean:
	$$(Q)-$$(RM_RF) $$(GEN_LIB_DIR)/lwmake/$3 $$(GEN_LIB_DIR)/lwmake/drv/$3
submake.$1.install:
	$$(Q)$$(MAKE) -C $2/$3 TOPOLOGY_PROTO_ONLY=1 -f makefile.lwmk install_shared $$(LWMAKE_ARGS)
submake.$1.build_install: submake.$1.build
	$$(Q)$$(MAKE) -C $2/$3 TOPOLOGY_PROTO_ONLY=1 -f makefile.lwmk install_shared $$(LWMAKE_ARGS)
endef

$(eval $(call LWMAKE_COMPONENT,protobuf,$(PROTOUTIL_DIR)/../..,diag/protoutil/protobuf))
$(eval $(call LWMAKE_COMPONENT,protobuf_host,$(PROTOUTIL_DIR)/../..,diag/protoutil/protobuf_host))

include $(MODS_DIR_UNIX)/makerule.inc

# Include dependency information.
-include $(depend_files)
