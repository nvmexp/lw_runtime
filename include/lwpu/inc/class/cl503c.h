/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _cl503c_h_
#define _cl503c_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define   LW50_THIRD_PARTY_P2P                          (0x0000503c)

/* LwRmAlloc parameters */
typedef struct {
    LwU32 flags;
} LW503C_ALLOC_PARAMETERS;

#define LW503C_ALLOC_PARAMETERS_FLAGS_TYPE                      1:0
#define LW503C_ALLOC_PARAMETERS_FLAGS_TYPE_PROPRIETARY  (0x00000000)
#define LW503C_ALLOC_PARAMETERS_FLAGS_TYPE_BAR1         (0x00000001)
#define LW503C_ALLOC_PARAMETERS_FLAGS_TYPE_LWLINK       (0x00000002)

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl503c_h_ */
