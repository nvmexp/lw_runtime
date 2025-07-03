/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2010-2014 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/


#include "lwtypes.h"

#ifndef _cle26d_h_
#define _cle26d_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LWE2_SYNCPOINT                                     (0x0000E26D)

/*
 * Allocation parameters for syncpoint objects
 */
typedef struct
{
    LwHandle hMemory;
    LwU32    offset;
    LwU32    flags;
    LwU32    index;
    LwU32    initialValue;
    LwU32    initialIndex;
} LW_SYNCPOINT_ALLOCATION_PARAMETERS;

#define LW_SYNCPOINT_ALLOC_INDEX_NONE                                   ((LwU32)(~0))
#define LW_SYNCPOINT_ALLOC_FLAGS_AUTO_INCREMENT                         0:0
#define LW_SYNCPOINT_ALLOC_FLAGS_AUTO_INCREMENT_DISABLE                 (0x00000000)
#define LW_SYNCPOINT_ALLOC_FLAGS_AUTO_INCREMENT_ENABLE                  (0x00000001)
#define LW_SYNCPOINT_ALLOC_FLAGS_SKIP_MEMORY_UPDATE_ON_SERVICE          1:1
#define LW_SYNCPOINT_ALLOC_FLAGS_SKIP_MEMORY_UPDATE_ON_SERVICE_DISABLE  (0x00000000)
#define LW_SYNCPOINT_ALLOC_FLAGS_SKIP_MEMORY_UPDATE_ON_SERVICE_ENABLE   (0x00000001)

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cle26d_h

