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
base_name = libbzip2

include $(MODS_DIR)/makedefs.inc

BZIP2_PARENT_DIR = $(BUILD_TOOLS_DIR)/mods/bzip2-1.0.6
ifeq "$(base_os)" "win32"
    ifdef CYGWIN_MAKE
        BZIP2_PARENT_DIR := $(shell $(MODS_DIR)/tools/abs2rel.sh '$(BZIP2_PARENT_DIR)')
    endif

    # Must turn on C++ exception handling when using the C++ standard library.
    # Would really prefer to not have to turn this on... unwinding logic is not
    # free, and we don't really want C++ exceptions.
    CFLAGS += /EHsc
endif

BZIP2_DIR = $(BZIP2_PARENT_DIR)/src
include_dirs += $(BZIP2_PARENT_DIR)
include_dirs += $(BZIP2_DIR)

# Source files
bzip2_files += blocksort.c
bzip2_files += huffman.c
bzip2_files += crctable.c
bzip2_files += randtable.c
bzip2_files += compress.c
bzip2_files += decompress.c
bzip2_files += bzip2.c
bzip2_files += bzlib.c
c_files += $(addprefix $(BZIP2_DIR)/, $(bzip2_files))

ifneq "$(filter 7.%,$(toolchain_rev))" ""
    CFLAGS += -Wno-unused-but-set-variable
    CFLAGS += -Wno-unused-result
endif

#-------------------------------------------------------------------------------
# Includes for standard functionality
#-------------------------------------------------------------------------------
include $(MODS_DIR_UNIX)/makelib.inc
include $(MODS_DIR_UNIX)/makerule.inc

# Include dependency information.
-include $(depend_files)
