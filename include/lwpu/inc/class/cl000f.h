/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _cl000f_h_
#define _cl000f_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define FABRIC_MANAGER_SESSION  (0x0000000F)

#define LW000F_NOTIFIERS_FABRIC_EVENT (0)

#define LW000F_FLAGS_CHANNEL_RECOVERY              0:0
#define LW000F_FLAGS_CHANNEL_RECOVERY_ENABLED      0x0
#define LW000F_FLAGS_CHANNEL_RECOVERY_DISABLED     0x1

typedef struct
{
    //
    // capDescriptor is a file descriptor for unix RM clients, but a void
    // pointer for windows RM clients.
    //
    // capDescriptor is transparent to RM clients i.e. RM's user-mode shim
    // populates this field on behalf of clients.
    //
    LW_DECLARE_ALIGNED(LwU64 capDescriptor, 8);

    LwU32 flags;
} LW000F_ALLOCATION_PARAMETERS;

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl000f_h_ */

