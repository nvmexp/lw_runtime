/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2002-2002 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _cl5080_h_
#define _cl5080_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define LW50_DEFERRED_API_CLASS                                    (0x00005080)

/* LwRmAlloc parameters */
typedef struct {
    // Should the deferred api completion trigger an event
    LwBool notifyCompletion;
} LW5080_ALLOC_PARAMS;

/* dma method offsets, fields, and values */
#define LW5080_SET_OBJECT                                          (0x00000000)
#define LW5080_NO_OPERATION                                        (0x00000100)
#define LW5080_DEFERRED_API                                        (0x00000200)
#define LW5080_DEFERRED_API_HANDLE                                 31:0

// Class-specific allocation capabilities

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl5080_h_ */

