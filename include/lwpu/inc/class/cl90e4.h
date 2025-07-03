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

#ifndef _cl90e4_h_
#define _cl90e4_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  GF100_SUBDEVICE_LTCG                                     (0x000090e4)

typedef struct {
    LwU32 Reserved00[0x40000];                                       /*  LW_PLTCG 0x0017ffff:0x00140000 */
} Lw90e4MapTypedef, GF100LTCGMap;
#define  LW90e4_TYPEDEF                                            GF100LTCGMap

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl90e4_h_ */
