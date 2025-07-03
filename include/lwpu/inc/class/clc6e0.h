/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef _clc6e0_h_
#define _clc6e0_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  GA100_SMC_GRAPHICS                                  (0x0000c6e0)

/* LwRmAlloc parameters */
typedef struct {
    LwU32 engineId;   // SMC Local GR Index that clients will provide
} LWC6E0_ALLOCATION_PARAMETERS;

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _clc6e0_h_ */
