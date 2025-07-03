/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2010 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef _cl90e1_h_
#define _cl90e1_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  GF100_SUBDEVICE_FB                                        (0x000090e1)

typedef struct {
    LwU32 Reserved00[0x400];                                       /* 0x0000 - 0x0fff */
} Lw90e1MapTypedef, GF100FBMap;
#define  LW90e1_TYPEDEF                                            GF100FBMap

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl90e1_h_ */
