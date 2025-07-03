# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright 2014-2019 by LWPU Corporation. All rights reserved. All
# information contained herein is proprietary and confidential to LWPU
# Corporation. Any use, reproduction, or disclosure without the written
# permission of LWPU Corporation is prohibited.
#
# LWIDIA_COPYRIGHT_END

MODS_DIR         = ../../mods
DRIVERS_DIR      = ../../../drivers
JS17_DIR         = ../../js1.7
LWDIAG_UTILS_DIR = ../../utils
SDK_DIR          = ../../../sdk

include $(MODS_DIR)/makedefs.inc

base_name = jsc

include_dirs += $(BOOST_DIR)
include_dirs += $(LWDIAG_UTILS_DIR)
include_dirs += $(JS17_DIR)
include_dirs += $(MODS_DIR)
include_dirs += $(SDK_DIR)/lwpu/inc
include_dirs += ../include
include_dirs += $(OPCODE_TABLE_DIR)

defines += -DINCLUDE_STD_NAMESPACE

cpp_files = jsc.cpp

include $(JS17_DIR)/makesrc.inc

ifeq ($(base_os),win32)
	lib_dirs = $(MODS_DIR)/$(base_mods_lib_dir)/$(BUILD_CFG)
else
	lib_dirs = $(MODS_DIR)/$(base_mods_lib_dir)
endif

ifeq "$(base_os)" "linux"
    libraries += dl
endif
libraries += lwdiag_utils boost z

lib_dirs += $(MODS_OUTPUT_DIR)/$(rel_o_dir)/lwmake/diag/utils

$(o_dir)/$(exe_name) : $(o_files)
	@echo "Linking $@"
	$(Q)$(LD) $(LDFLAGS_EXE) $(o_files) $(link_libs)

$(o_dir)/$(exe_name): submake.lwdiag_utils.build

submake.lwdiag_utils.%:
	$(Q)$(MAKE) -C $(MODS_DIR)/../utils -f makefile.lwmk $(subst submake.lwdiag_utils.,,$@) $(LWMAKE_ARGS) $*

include $(MODS_DIR_UNIX)/makerule.inc

-include $(depend_files)

# Reset the default goal.
.DEFAULT_GOAL :=

.PHONY : all clean

all : $(o_dir)/dummy.txt $(o_dir)/$(exe_name)

clean :
	@-rm -f $(o_dir)/*$(obj_suffix)
	@-rm -f $(o_dir)/*.d
	@-rm -f $(o_dir)/*.pdb
	@-rm -f *.d
	@-rm -f $(o_dir)/$(exe_name)
