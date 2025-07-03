/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _SOE_LS10_H_
#define _SOE_LS10_H_

#include "ls10.h"

//
// Functions shared with LR10
//

LwlStatus lwswitch_soe_set_ucode_core_ls10(lwswitch_device *device, LwBool bFalcon);
LwlStatus lwswitch_init_soe_ls10(lwswitch_device *device);


#endif //_SOE_LS10_H_
