/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*
 * Class definition for allocating a contiguous or discontiguous Multicast FLA.
 */

#ifndef _cl00fd_h_
#define _cl00fd_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define LW_MEMORY_MULTICAST_FABRIC (0x000000fd)

/*
 *  alignment [IN]
 *    Alignment for the allocation.
 *    Should be at least the requested page size.
 *
 *  allocSize [IN]
 *    Size of the Multicast FLA VA.
 *
 *  pageSize [IN]
 *    Requested page size. Can be any of the LW_MEMORY_MULTICAST_FABRIC_PAGE_SIZE_*
 *
 *  allocFlags [IN]
 *    Reserved for future use
 *    Clients should pass 0 as of now.
 *
 *  numGpus [IN]
 *    Number of unique GPUs to be attached.
 *
 *  pOsEvent [IN]
 *    Optional OS event handle created with LwRmAllocOsEvent().
 */

#define LW_MEMORY_MULTICAST_FABRIC_PAGE_SIZE_512M      0x20000000

typedef struct {

    LW_DECLARE_ALIGNED(LwU64 alignment, 8);
    LW_DECLARE_ALIGNED(LwU64 allocSize, 8);
    LwU32    pageSize;
    LwU32    allocFlags;
    LwU32    numGpus;
    LW_DECLARE_ALIGNED(LwP64 *pOsEvent, 8);

} LW00FD_ALLOCATION_PARAMETERS;

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl00fd_h_ */
