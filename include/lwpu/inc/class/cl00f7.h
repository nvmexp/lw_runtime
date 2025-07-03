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
 * Class definition for exporting memory handles to a different RM client on the
 * same or another node (OS).
 *
 * No memory is allocated or mapped using this class.
 *
 * Exported memory will be ref-counted based on the flags provided.
 *
 * Only supports exporting FLA memory (LW_MEMORY_FABRIC).
 */

#ifndef _cl00f7_h_
#define _cl00f7_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LW_MEMORY_FABRIC_EXPORT_V2 (0x000000f7)

#define LW_FABRIC_PACKET_LEN 32

/*
 *  maxHandles [IN]
 *    Max number of memory handles to be exported using the export object.
 *
 *  flags [IN]
 *    For future use.
 *    Set to zero for default behavior.
 *
 *  attach.index [IN]
 *    Index into the export object at which to start attaching the provided
 *    memory handles.
 *    attach.index + attach.numHandles must be <= maxHandles.
 *
 *  attach.numHandles [IN]
 *    Number of memory handles to be attached during the export object allocation.
 *    Must be <= MIN(LW00F7_ALLOC_MAX_ATTACHABLE_HANDLES, maxHandles).
 *    Can be zero if not required.
 *
 *  attach.handles [IN]
 *    Array of valid memory handles.
 *
 *  packet [OUT]
 *    Bag of bits which uniquely identifies this object universally.
 */

#define LW00F7_ALLOC_MAX_ATTACHABLE_HANDLES 2

typedef struct {
    LwU16 maxHandles;
    LwU32 flags;

    struct {
        LwU16 index;
        LwU16 numHandles;
        LwHandle handles[LW00F7_ALLOC_MAX_ATTACHABLE_HANDLES];
    } attach;

    LwU8 packet[LW_FABRIC_PACKET_LEN];
} LW00F7_ALLOCATION_PARAMETERS;

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif /* _cl00f7_h_ */
