/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*
 * Please *DO NOT* list MODS-only IOCTLs in this file.
 *
 * This file defines IOCTLs that work with lwpu-lwswitchX (device) nodes.
 *
 * The IOCTLs are for internal use only, for example LWSwitch GTEST, and hence
 * they do not contribute to the driver ABI version.
 */

#ifndef _IOCTL_DEVICE_INTERNAL_LWSWITCH_H_
#define _IOCTL_DEVICE_INTERNAL_LWSWITCH_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "ioctl_common_lwswitch.h"
#include "ctrl_dev_internal_lwswitch.h"

#define IOCTL_LWSWITCH_REGISTER_READ \
    LWSWITCH_IOCTL_CODE(LWSWITCH_DEV_IO_TYPE, CTRL_LWSWITCH_REGISTER_READ, LWSWITCH_REGISTER_READ, \
                        LWSWITCH_IO_WRITE_READ)

#define IOCTL_LWSWITCH_REGISTER_WRITE \
    LWSWITCH_IOCTL_CODE(LWSWITCH_DEV_IO_TYPE, CTRL_LWSWITCH_REGISTER_WRITE, LWSWITCH_REGISTER_WRITE, \
                        LWSWITCH_IO_WRITE_ONLY)

#ifdef __cplusplus
}
#endif

#endif //_IOCTL_DEVICE_INTERNAL_LWSWITCH_H_
