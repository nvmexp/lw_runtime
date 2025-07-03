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

#ifndef _cl90e3_h_
#define _cl90e3_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  GF100_SUBDEVICE_FLUSH                                     (0x000090e3)

typedef struct {
    LwU32 Reserved00[0x400];                                       /* LW_UFLUSH 0x00070FFF:0x00070000 */
} Lw90e3MapTypedef, GF100FLUSHMap;
#define  LW90e3_TYPEDEF                                            GF100FLUSHMap

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl90e3_h_ */
