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

#ifndef _cla0e0_h_
#define _cla0e0_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  GK110_SUBDEVICE_GRAPHICS                                  (0x0000a0e0)

typedef struct {
    LwU32 Reserved00[0x200000];                                    /* 0x000000 - 0x1fffff */                                   
} LWA0E0MapTypedef, GK110SubdeviceGRMap;
#define  LWA0E0_TYPEDEF                                            GK110SubdeviceGRMap

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cla0e0_h_ */
