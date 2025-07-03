# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright 2017 by LWPU Corporation. All rights reserved. All information
# contained herein is proprietary and confidential to LWPU Corporation. Any
# use, reproduction, or disclosure without the written permission of LWPU
# Corporation is prohibited.
#
# LWIDIA_COPYRIGHT_END

MODS_DIR ?= ../mods

base_name = libusb-1.0
USB_DIR = $(BUILD_TOOLS_DIR)/mods/libusb/libusb-1.0.21

BUILD_RELEASE_DLLS=true
dll_build = true

old_defines := $(defines)
defines := $(filter-out LW_MODS,$(defines))
defines += -DHAVE_CONFIG_H

include $(MODS_DIR)/makedefs.inc

include_dirs += .						#for config.h
include_dirs += $(USB_DIR)/libusb		#for generic libusb headers
include_dirs += $(USB_DIR)/libusb/os	#for os specific headers

ifeq "$(base_os)" "linux"
	c_files += $(USB_DIR)/libusb/os/poll_posix.c
	c_files += $(USB_DIR)/libusb/os/threads_posix.c
	c_files += $(USB_DIR)/libusb/os/linux_usbfs.c
	c_files += $(USB_DIR)/libusb/os/linux_netlink.c
endif

c_files += $(USB_DIR)/libusb/core.c
c_files += $(USB_DIR)/libusb/descriptor.c
c_files += $(USB_DIR)/libusb/hotplug.c
c_files += $(USB_DIR)/libusb/io.c
c_files += $(USB_DIR)/libusb/strerror.c
c_files += $(USB_DIR)/libusb/sync.c

#-------------------------------------------------------------------------------
# Targets:
#-------------------------------------------------------------------------------
.PHONY : all clean install

all : $(o_dir)/$(lib_name)

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

# Restore defines
defines := $(old_defines)
