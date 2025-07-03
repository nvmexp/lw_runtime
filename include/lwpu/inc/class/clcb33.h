/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef _clcb33_h_
#define _clcb33_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define LW_CONFIDENTIAL_COMPUTE                     (0x0000CB33)

typedef struct
{
    LwHandle hClient;
}LW_CONFIDENTIAL_COMPUTE_ALLOC_PARAMS;

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _clcb33_h_ */
