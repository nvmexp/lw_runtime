#/*
# * LWIDIA_COPYRIGHT_BEGIN
# *
# * Copyright 1999-2008 by LWPU Corporation.  All rights reserved.  All
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
base_name = libxml

include $(MODS_DIR)/makedefs.inc

include_dirs += .

# Source files
c_files += xmlparse.c
c_files += xmlrole.c
c_files += xmltok.c

defines += -DXML_STATIC

ifeq "$(toolchain)" "msvc"
   defines += -DCOMPILED_FROM_DSP
else
   defines += -DHAVE_EXPAT_CONFIG_H
endif

ifeq "$(base_os)" "macosx"
   defines += -DMACOSX
endif

ifeq "$(BUILD_OS)" "linux"
   defines += -Dlinux
endif

ifeq "$(BUILD_OS)" "sim"
   defines += -Dlinux
endif


#-------------------------------------------------------------------------------
# Targets:
#-------------------------------------------------------------------------------

# The default target makes the xml library (libxml.a).
all : $(o_dir)/$(lib_name)

.PHONY : all clean install

include $(MODS_DIR_UNIX)/makerule.inc

clean :
	@echo "Deleting object, lib, and dependency files."
	@-rm -f $(o_dir)/*$(obj_suffix)
	@-rm -f $(o_dir)/*$(lib_suffix)
	@-rm -f $(o_dir)/*$(dll_suffix)
	@-rm -f $(o_dir)/*.d
	@-rm -f $(o_dir)/*.pdb
	@-rm -f *.d

install:
	@echo "Copying $(o_dir)/$(lib_name) to $(MODS_DIR)/$(base_mods_lib_dir)/$(lib_name)..."
	@cp $(o_dir)/$(lib_name) $(MODS_DIR)/$(base_mods_lib_dir)/$(lib_name)

# Include dependency information.
-include $(depend_files)
