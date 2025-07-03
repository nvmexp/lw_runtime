# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright 2013 by LWPU Corporation.  All rights reserved.  All
# information contained herein is proprietary and confidential to LWPU
# Corporation.  Any use, reproduction, or disclosure without the written
# permission of LWPU Corporation is prohibited.
#
# LWIDIA_COPYRIGHT_END
#
#
# Set variables to their default state if they are not already set.
MODS_DIR ?= ../../../diag/mods
LWWATCH_INC_DIR ?= ../inc
LWWATCH_LIB_DIR ?= ../lib
LWWATCH_INC_DIR += ../../../sdk/lwpu/inc

dll_build = false
base_name = libflcndwarf

include $(MODS_DIR)/makedefs.inc

LDFLAGS += -L$(LWWATCH_LIB_DIR) -llwdwarf -llwelf

defines += -DHAVE_CONFIG_H

CFLAGS += -Wall -fPIC

cpp_files += flcndwarf.cpp
cpp_files += flcndump.cpp
cpp_files += flcnextractinfo.cpp
cpp_files += naming.cpp
cpp_files += uri.cpp

include_dirs += ./libdwarf
include_dirs += ./
include_dirs += ./linux
include_dirs += ./libelf
include_dirs += ./libelf/linux
include_dirs += $(LWWATCH_INC_DIR)

ifeq "$(toolchain)" "gnu"
    # Set visibility=hidden on gcc 4.0 on up
    ifeq "$(filter 0 1.% 2.% 3.%,$(toolchain_rev))" ""
        CFLAGS += -fvisibility=hidden
    endif
endif

# Generic rules for a library build
include $(MODS_DIR)/makelib.inc
include $(MODS_DIR)/makerule.inc

install: flcndwarf_install
build_install: flcndwarf_install
flcndwarf_install:
	mkdir -p $(LWWATCH_LIB_DIR)
	@$(ECHO) "Copying $(o_dir)/$(lib_name) to $(LWWATCH_LIB_DIR)"
	$(Q)$(CP_IF_CHANGED) $(CP_RM_DEST) $(o_dir)/$(lib_name) $(LWWATCH_LIB_DIR)/$(lib_name)

# Include dependency information.
-include $(depend_files)
