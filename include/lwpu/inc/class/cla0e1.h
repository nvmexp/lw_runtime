/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef _cla0e1_h_
#define _cla0e1_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  GK110_SUBDEVICE_FB                                       (0x0000a0e1)

typedef struct {
    LwU32 Reserved00[0x400];                                       /* 0x0000 - 0x0fff */
} LWA0E1MapTypedef, GK110SubdeviceFBMap;
#define  LWA0E1_TYPEDEF                                            GK110SubdeviceFBMap

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cla0e1_h_ */
