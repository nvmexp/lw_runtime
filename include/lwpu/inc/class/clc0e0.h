/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2016 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef _clc0e0_h_
#define _clc0e0_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  GP100_SUBDEVICE_GRAPHICS                                  (0x0000c0e0)

typedef struct {
    LwU32 Reserved00[0x200000];                                    /* 0x000000 - 0x1fffff */                                   
} LWC0E0MapTypedef, GP100SubdeviceGRMap;
#define  LWC0E0_TYPEDEF                                            GP100SubdeviceGRMap

/* GP10x have varying geometries */
/* Only GP100 has GR ECC*/
/* GP100 */
#define GR_GP100_ECC_GPC_COUNT               (0x00000006)
#define GR_GP100_ECC_TPC_COUNT               (0x00000005)
#define GR_GP100_ECC_TEX_COUNT               (0x00000002)

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cla0e0_h_ */
