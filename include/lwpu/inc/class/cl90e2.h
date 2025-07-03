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

#ifndef _cl90e2_h_
#define _cl90e2_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  GF100_SUBDEVICE_FIFO                                     (0x000090e2)

typedef struct {
    LwU32 Reserved00[0x800];                                       /* LW_PFIFO 0x00003FFF:0x00002000 */
} Lw90e2MapTypedef, GF100FIFOMap;
#define  LW90e2_TYPEDEF                                            GF100FIFOMap

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl90e2_h_ */
