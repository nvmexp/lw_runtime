/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2020 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _cl0091_h_
#define _cl0091_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

/* LwRmAlloc parameters */
typedef struct {
    LwU32 hwClass;
} LW0091_ALLOC_PARAMETERS;

//
// PHYSICAL_GRAPHICS_OBJECT:
// SW object corresponding to physical implementation of 3d / compute classes
// such as {ARCH}_A and {ARCH}_COMPUTE_A.
// The HW class being allocated is required to be passed through alloc
// parameters. 
//
#define PHYSICAL_GRAPHICS_OBJECT    (0x00000091)

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl0091_h_ */

