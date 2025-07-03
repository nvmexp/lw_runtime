/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2012-2014 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/


#include "lwtypes.h"

#ifndef _cle2ad_h_
#define _cle2ad_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LWE2_SYNCPOINT_BASE                                (0x0000e2ad)

/*
 * Allocation parameters for syncpoint base objects
 */
typedef struct
{
    LwHandle hMemory;
    LwU32    offset;
    LwU32    index;
    LwU32    initialValue;
    LwU32    initialIndex;
} LW_SYNCPOINT_BASE_ALLOCATION_PARAMETERS;

#define LW_SYNCPOINT_BASE_ALLOC_INDEX_NONE             ((LwU32)(~0))

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cle2ad_h

