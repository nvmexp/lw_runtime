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

#ifndef _cl90e0_h_
#define _cl90e0_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  GF100_SUBDEVICE_GRAPHICS                                  (0x000090e0)

typedef struct {
    LwU32 Reserved00[0x200000];                                    /* 0x000000 - 0x1fffff */                                   
} Lw90e0MapTypedef, GF100GRMap;
#define  LW90e0_TYPEDEF                                            GF100GRMap

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl90e0_h_ */
