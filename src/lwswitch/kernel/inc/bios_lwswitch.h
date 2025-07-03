/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _BIOS_LWSWITCH_H_
#define _BIOS_LWSWITCH_H_

#include "common_lwswitch.h"

LwlStatus lwswitch_bios_read(lwswitch_device *, LwU32, void *);
LwlStatus lwswitch_bios_read_size(lwswitch_device *, LwU32 *);
LwlStatus lwswitch_bios_get_image(lwswitch_device *device);

#endif //_BIOS_LWSWITCH_H_
