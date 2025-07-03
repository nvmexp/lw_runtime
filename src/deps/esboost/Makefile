#
# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright 2015-2018 by LWPU Corporation.  All rights reserved.  All
# information contained herein is proprietary and confidential to LWPU
# Corporation.  Any use, reproduction, or disclosure without the written
# permission of LWPU Corporation is prohibited.
#
# LWIDIA_COPYRIGHT_END
#


ROOT_DIR ?= .
MODS_DIR ?= ../mods
BUILD_OS ?= linuxmfg

base_name = libesboost

include $(MODS_DIR)/makedefs.inc

ifeq ($(base_os),win32)
   # Must turn on C++ exception handling when using the C++ standard library on windows.
   CFLAGS += /EHsc
endif

include_dirs += $(ROOT_DIR)
include_dirs += $(BOOST_DIR)
include_dirs += $(MODS_DIR)

libraries += esboost

cpp_files  = $(ROOT_DIR)/es_json_parser.cpp
cpp_files += $(ROOT_DIR)/es_ptree.cpp

#-------------------------------------------------------------------
# Targets
#-------------------------------------------------------------------

all: build_install

include $(MODS_DIR_UNIX)/makelib.inc
include $(MODS_DIR_UNIX)/makerule.inc
