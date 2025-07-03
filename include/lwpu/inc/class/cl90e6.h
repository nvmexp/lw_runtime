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

#ifndef _cl90e6_h_
#define _cl90e6_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  GF100_SUBDEVICE_MASTER                                     (0x000090e6)

typedef struct {
    LwU32 Reserved00[0x400];                                       /* LW_PMC 0x00000FFF:0x00000000 */
} Lw90e6MapTypedef, GF100MASTERMap;
#define  LW90e6_TYPEDEF                                            GF100MASTERMap

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl90e6_h_ */
