#
# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright 2016-2018 by LWPU Corporation.  All rights reserved.  All
# information contained herein is proprietary and confidential to LWPU
# Corporation.  Any use, reproduction, or disclosure without the written
# permission of LWPU Corporation is prohibited.
#
# LWIDIA_COPYRIGHT_END
#

# Set variables to their default state if they are not already set.
INCLUDE_GPU ?= true
LW_SOURCE ?= ../../..
MODS_DIR ?= $(LW_SOURCE)/diag/mods
PROTOBUF_DIR := ../libs/protobuf-2.6.0

dll_build = false
base_name = libprotobuf

include $(MODS_DIR)/makedefs.inc

ifneq "$(base_os)" "linux"
    $(error protobuf library only supported on linux MODS platforms)
endif

include makesrc.inc

ifneq "$(base_os)" "win32"
   CFLAGS += -Wno-sign-compare -Wno-unused-local-typedefs -Wno-deprecated -Wno-return-type -Wno-maybe-uninitialized
endif

ifeq "$(base_os)" "win32"
   warnings += -WX
else
   warnings += -Werror
endif

ifneq "$(filter 7.%,$(toolchain_rev))" ""
    CFLAGS += -Wno-misleading-indentation
endif

ifneq "$(filter 9.%,$(toolchain_rev))" ""
    CFLAGS += -Wno-misleading-indentation
    CFLAGS += -Wno-class-memaccess
endif

# Generic rules for a library build
include $(MODS_DIR_UNIX)/makelib.inc
include $(MODS_DIR_UNIX)/makerule.inc
