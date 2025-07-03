##############################################################################
#
# _LWRM_COPYRIGHT_BEGIN_
#
# Copyright 2020 by LWPU Corporation.  All rights reserved.  All
# information contained herein is proprietary and confidential to LWPU
# Corporation.  Any use, reproduction, or disclosure without the written
# permission of LWPU Corporation is prohibited.
#
# _LWRM_COPYRIGHT_END_
#
##############################################################################

LWSWITCHAPI_DIR = $(DRIVERS_DIR)/lwswitch/user

include_dirs += $(LWSWITCHAPI_DIR)
include_dirs += $(DRIVERS_DIR)/sdk/lwpu/inc
include_dirs += $(DRIVERS_DIR)/common/inc
include_dirs += $(DRIVERS_DIR)/lwswitch/interface

ifneq (,$(filter $(BUILD_OS),linux))
    include_dirs += $(DRIVERS_DIR)/resman/arch/lwalloc/unix/lib/utils
    include_dirs += $(DRIVERS_DIR)/unix/common/inc
    include_dirs += $(DRIVERS_DIR)/lwswitch/user
    include_dirs += $(DRIVERS_DIR)/lwlink/user/lwlink
    
    c_files  += $(LWSWITCHAPI_DIR)/Linux/lwswitch_user_api_linux.c
else
    c_files += $(LWSWITCHAPI_DIR)/mods/lwswitch_user_api_mods.c
endif
