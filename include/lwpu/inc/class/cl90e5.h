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

#ifndef _cl90e5_h_
#define _cl90e5_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  GF100_SUBDEVICE_TOP                                     (0x000090e5)

typedef struct {
    LwU32 Reserved00[0x400];                                       /* LW_PTOP 0x000227FF:0x00022400 */
} Lw90e5MapTypedef, GF100TOPMap;
#define  LW90e5_TYPEDEF                                            GF100TOPMap

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl90e5_h_ */
