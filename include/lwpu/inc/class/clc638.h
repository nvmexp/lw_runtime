/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2019 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _clc638_h_
#define _clc638_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define AMPERE_SMC_EXEC_PARTITION_REF    (0x0000c638)

/* LwRmAlloc parameters */
typedef struct {
    //
    // capDescriptor is a file descriptor for unix RM clients, but a void
    // pointer for windows RM clients.
    //
    // capDescriptor is transparent to RM clients i.e. RM's user-mode shim
    // populates this field on behalf of clients.
    //
    LW_DECLARE_ALIGNED(LwU64 capDescriptor, 8);

    LwU32 execPartitionId;
} LWC638_ALLOCATION_PARAMETERS;

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _clc638_h_ */

