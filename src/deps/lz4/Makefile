#/*
# * LWIDIA_COPYRIGHT_BEGIN
# *
# * Copyright 2018 by LWPU Corporation.  All rights reserved.  All
# * information contained herein is proprietary and confidential to LWPU
# * Corporation.  Any use, reproduction, or disclosure without the written
# * permission of LWPU Corporation is prohibited.
# *
# * LWIDIA_COPYRIGHT_END
# */

#-------------------------------------------------------------------------------
# JavaScript make file.
# See //sw/<tree>/diag/mods/makedefs.inc for more information about the MODS
# makefiles.
#-------------------------------------------------------------------------------

MODS_DIR = ../mods

dll_build = false
base_name = liblz4

include $(MODS_DIR)/makedefs.inc

LZ4_PARENT_DIR = $(COMMON_WS_DIR)/sw/mods/tools/lz4-1.8.1.2
ifeq "$(base_os)" "win32"
    ifdef CYGWIN_MAKE
        LZ4_PARENT_DIR := $(shell $(MODS_DIR)/tools/abs2rel.sh '$(LZ4_PARENT_DIR)')
    endif

    # Must turn on C++ exception handling when using the C++ standard library.
    # Would really prefer to not have to turn this on... unwinding logic is not
    # free, and we don't really want C++ exceptions.
    CFLAGS += /EHsc
endif

LZ4_DIR = $(LZ4_PARENT_DIR)/lz4
include_dirs += $(LZ4_PARENT_DIR)
include_dirs += $(LZ4_DIR)

# Source files
lz4_files += lz4.c
lz4_files += lz4frame.c
lz4_files += lz4hc.c
lz4_files += xxhash.c
c_files += $(addprefix $(LZ4_DIR)/, $(lz4_files))

#-------------------------------------------------------------------------------
# Includes for standard functionality
#-------------------------------------------------------------------------------
include $(MODS_DIR_UNIX)/makelib.inc
include $(MODS_DIR_UNIX)/makerule.inc

# Include dependency information.
-include $(depend_files)
