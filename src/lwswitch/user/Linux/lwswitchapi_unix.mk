##############################################################################
#
# _LWRM_COPYRIGHT_BEGIN_
#
# Copyright 2017-2020 by LWPU Corporation.  All rights reserved.  All
# information contained herein is proprietary and confidential to LWPU
# Corporation.  Any use, reproduction, or disclosure without the written
# permission of LWPU Corporation is prohibited.
#
# _LWRM_COPYRIGHT_END_
#
# This makefile will define:
#
# LWSWITCHAPI_SOURCES : a list of source files, based on $(LWSWITCHAPI_LW_SOURCE);
# the includer should compile each of these source files and link the
# resulting object files into the includer's RM client(s)
#
# LWSWITCHAPI_DEFINES : a list of preprocessor defines needed by
# $(LWSWITCHAPI_SOURCES); the includer should compile $(LWSWITCHAPI_SOURCES) with
# these defines; e.g.,
#
#   $(LWSWITCHAPI_OBJECTS): CFLAGS += $(addprefix -D,$(LWSWITCHAPI_DEFINES))
#
# LWSWITCHAPI_INCLUDES : a list of include paths, based on
# $(LWSWITCHAPI_LW_SOURCE), needed by $(LWSWITCHAPI_SOURCES); the includer should
# compile $(LWSWITCHAPI_SOURCES) with these include paths; e.g.,
#
#   $(LWSWITCHAPI_OBJECTS): CFLAGS += $(addprefix -I,$(LWSWITCHAPI_INCLUDES))
#
# Note that LWSWITCHAPI depends on lwconfig; it is the includer's
# responsibility to ensure that lwconfig is run and available to
# $(LWSWITCHAPI_SOURCES)
#
##############################################################################

LWSWITCHAPI_INCLUDES  = $(LWSWITCHAPI_SOURCE_ROOT)/sdk/lwpu/inc
LWSWITCHAPI_INCLUDES += $(LWSWITCHAPI_SOURCE_ROOT)/drivers/common/inc
LWSWITCHAPI_INCLUDES += $(LWSWITCHAPI_SOURCE_ROOT)/drivers/lwswitch/interface
LWSWITCHAPI_INCLUDES += $(LWSWITCHAPI_SOURCE_ROOT)/drivers/resman/arch/lwalloc/unix/lib/utils
LWSWITCHAPI_INCLUDES += $(LWSWITCHAPI_SOURCE_ROOT)/drivers/unix/common/inc

LWSWITCHAPI_SOURCES  =

ifeq ($(LWSWITCH_ENABLE_CROSS_PLATFORM_USER_API), 1)
    LWSWITCHAPI_SOURCES  += $(LWSWITCHAPI_SOURCE_ROOT)/drivers/lwswitch/user/Linux/lwswitch_user_api_linux.c
    LWSWITCHAPI_INCLUDES += $(LWSWITCHAPI_SOURCE_ROOT)/drivers/lwswitch/user
    LWSWITCHAPI_INCLUDES += $(LWSWITCHAPI_SOURCE_ROOT)/drivers/lwlink/user/lwlink
else
    LWSWITCHAPI_SOURCES  += $(LWSWITCHAPI_SOURCE_ROOT)/drivers/lwswitch/user/Linux/lwswitch_user_linux.c
    LWSWITCHAPI_INCLUDES += $(LWSWITCHAPI_SOURCE_ROOT)/drivers/lwswitch/user/Linux
endif

LWSWITCHAPI_SOURCES  += $(LWSWITCHAPI_SOURCE_ROOT)/drivers/resman/arch/lwalloc/unix/lib/utils/lwpu-modprobe-utils.c
LWSWITCHAPI_SOURCES  += $(LWSWITCHAPI_SOURCE_ROOT)/drivers/resman/arch/lwalloc/unix/lib/utils/lwpu-modprobe-client-utils.c
LWSWITCHAPI_SOURCES  += $(LWSWITCHAPI_SOURCE_ROOT)/drivers/resman/arch/lwalloc/unix/lib/utils/pci-sysfs.c
