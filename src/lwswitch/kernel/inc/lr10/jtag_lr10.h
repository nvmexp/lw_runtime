/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */


#ifndef _JTAG_LR10_H_
#define _JTAG_LR10_H_

LwlStatus
lwswitch_jtag_read_seq_lr10
(
    lwswitch_device *device,
    LwU32 chainLen,
    LwU32 chipletSel,
    LwU32 instrId,
    LwU32 *data,
    LwU32 dataArrayLen          // in bytes
);

LwlStatus
lwswitch_jtag_write_seq_lr10
(
    lwswitch_device *device,
    LwU32 chainLen,
    LwU32 chipletSel,
    LwU32 instrId,
    LwU32 *data,
    LwU32 dataArrayLen          // in bytes
);

#endif //_JTAG_LR10_H_
