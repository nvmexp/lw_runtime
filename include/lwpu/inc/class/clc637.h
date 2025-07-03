/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2018 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _clc637_h_
#define _clc637_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define AMPERE_SMC_PARTITION_REF    (0x0000c637)

//
// This swizzId can be used by root clients like tools for device level
// profiling
//
#define LWC637_DEVICE_PROFILING_SWIZZID (0xFFFFFFFE)

//
// TODO: Deprecate LWC637_DEVICE_LEVEL_SWIZZID once all the clients are moved to
//       LWC637_DEVICE_PROFILING_SWIZZID
//
#define LWC637_DEVICE_LEVEL_SWIZZID LWC637_DEVICE_PROFILING_SWIZZID

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

    LwU32 swizzId;
} LWC637_ALLOCATION_PARAMETERS;

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _clc637_h_ */

