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

#include "lwtypes.h"

/*
 * Class definition for allocating a contiguous or discontiguous FLA.
 */

#ifndef _cl00f8_h_
#define _cl00f8_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LW_MEMORY_FABRIC (0x000000f8)

/*
 *  alignment [IN]
 *    Alignment for the allocation.
 *    Should be at least the requested page size.
 *
 *  allocSize [IN]
 *    Size of the FLA VA.
 *
 *  pageSize [IN]
 *    Requested page size. Can be any of the LW_MEMORY_FABRIC_PAGE_SIZE_*
 *
 *  allocFlags [IN]
 *    Can be any of the LW00F8_ALLOC_FLAGS_*
 *        DEFAULT (sticky)
 *            The FLA -> PA mappings will be stuck to the object, i.e, once the mapping is created
 *            there is no way to unmap it explicitly.
 *            The FLA object must be destroyed to release the mappings.
 *            The FLA object can't be duped or exported until it has a mapping associated with it.
 *            Partial FLA->PA mappings will NOT be allowed.
 *        FLEXIBLE_FLA
 *            The FLA -> PA mappings can be modified anytime irrespective of the FLA object is duped
 *            or exported.
 *            Partial FLA mappings are allowed.
 *        FORCE_NONCONTIGUOUS
 *            The allocator may pick contiguous memory whenever possible. This flag forces the
 *            allocator to always allocate noncontiguous memory. This flag is mainly used for
 *            testing purpose. So, use with caution.
 *        FORCE_CONTIGUOUS
 *            This flag forces the allocator to always allocate contiguous memory.
 *        READ_ONLY
 *            The FLA -> PA mappings will be created read-only. This option is only available on
 *            debug/develop builds due to security concerns. The security concerns are due to the
 *            fact that FLA access errors (a.k.a PRIV errors) are not aways context attributable.
 *
 *  map.offset [IN]
 *    Offset into the physical memory descriptor.
 *    Must be physical memory page size aligned.
 *
 *  map.hVidMem [IN]
 *    Handle to the physical video memory. Must be passed when the sticky flag is set so that the
 *    FLA -> PA mapping can happen during object creation.
 *    Phys memory with 2MB pages is supported.
 *    Phys memory handle can be LW01_NULL_OBJECT if FLEXIBLE_FLA flag is passed.
 *    hVidMem should belong the same device and client which is allocating FLA.
 *
 *  map.flags [IN]
 *    Reserved for future use.
 *    Clients should pass 0 as of now.
 */

#define LW_MEMORY_FABRIC_PAGE_SIZE_2M        0x200000
#define LW_MEMORY_FABRIC_PAGE_SIZE_512M      0x20000000

#define LW00F8_ALLOC_FLAGS_DEFAULT             0
#define LW00F8_ALLOC_FLAGS_FLEXIBLE_FLA        LWBIT(0)
#define LW00F8_ALLOC_FLAGS_FORCE_NONCONTIGUOUS LWBIT(1)
#define LW00F8_ALLOC_FLAGS_FORCE_CONTIGUOUS    LWBIT(2)
#define LW00F8_ALLOC_FLAGS_READ_ONLY           LWBIT(3)

typedef struct {

    LW_DECLARE_ALIGNED(LwU64 alignment, 8);
    LW_DECLARE_ALIGNED(LwU64 allocSize, 8);

    LwU32 pageSize;
    LwU32 allocFlags;

    struct {
        LW_DECLARE_ALIGNED(LwU64 offset, 8);

        LwHandle hVidMem;
        LwU32    flags;
    } map;

} LW00F8_ALLOCATION_PARAMETERS;

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif /* _cl00f8_h_ */
