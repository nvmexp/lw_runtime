#
# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright 2018 by LWPU Corporation.  All rights reserved.  All
# information contained herein is proprietary and confidential to LWPU
# Corporation.  Any use, reproduction, or disclosure without the written
# permission of LWPU Corporation is prohibited.
#
# LWIDIA_COPYRIGHT_END
#

#-------------------------------------------------------------------------------
# MODS version checker utility make file.
# See //sw/<tree>/diag/mods/makedefs.inc for more information about the MODS
# makefiles.
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#
# Environment variables: (set default values if not defined by user)
#
#-------------------------------------------------------------------------------
# Set variables to their default state if they are not already set.

CHANGELIST        ?= 0
VERSION           ?= 000
OFFICIAL_BUILD    ?= false

VERSION_CHECKER_DIR   ?= .
MODS_DIR              ?= $(VERSION_CHECKER_DIR)/../mods
COMMON_WS_DIR         ?= $(P4ROOT)
SHARED_MODS_FILES_DIR ?= $(COMMON_WS_DIR)/sw/mods
LINUX_DRV_DIR         ?= $(SHARED_MODS_FILES_DIR)/linuxdrv/v3.2

exe_build = true

base_name = version_checker

include $(MODS_DIR)/makedefs.inc

include_dirs += $(VERSION_CHECKER_DIR)
include_dirs += $(MODS_DIR) # for version.h, types.h
include_dirs += $(LINUX_DRV_DIR)

cpp_files += version_checker.cpp


#-------------------------------------------------------------------------------
# Targets:
#-------------------------------------------------------------------------------

.PHONY : all clean FORCE

all :  $(o_dir)/$(exe_name)

include $(MODS_DIR_UNIX)/makerule.inc

$(o_dir)/$(exe_name) : $(o_files)
	$(Q)$(LD) $(LDFLAGS_EXE) $(o_files) $(link_libs)
	@cp $(o_dir)/$(exe_name) $(MODS_RUNSPACE)

clean :
	@echo "Deleting object, lib, and dependency files."
	@-rm -f $(o_dir)/$(exe_name)

# Include dependency information.
-include $(depend_files)
