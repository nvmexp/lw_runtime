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
#include "cl00f7.h"

/*
 * Class definition for importing memory handles from a different RM client on
 * the same or another node (OS).
 */

#ifndef _cl00f9_h_
#define _cl00f9_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LW_MEMORY_FABRIC_IMPORT_V2 (0x000000f9)

/*
 *  packet [IN]
 *    Bag of bits which uniquely identifies the export object universally.
 *
 *  index [IN]
 *    Index into the FLA handle array associated with the export UUID.
 *
 *  flags [IN]
 *    For future use.
 *    Set to zero for default behavior.
 *
 *  pOsEvent [IN]
 *    Optional OS event handle created with LwRmAllocOsEvent().
 */

typedef struct {
    LwU8   packet[LW_FABRIC_PACKET_LEN];
    LwU16  index;
    LwU32  flags;
    LW_DECLARE_ALIGNED(LwP64 *pOsEvent, 8);
} LW00F9_ALLOCATION_PARAMETERS;

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif /* _cl00f9_h_ */
