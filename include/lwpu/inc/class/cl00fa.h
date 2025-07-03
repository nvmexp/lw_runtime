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
 * Class definition for referencing the exported fabric memory, LW_MEMORY_FABRIC,
 * using the export object UUID.
 *
 * The class allocation may fail if there is no memory attached to the
 * the export object, LW_MEMORY_FABRIC_EXPORT_V2, at the provided index.
 *
 * A privileged fabric manager only class, parented by subdevice, which must
 * match with the owner GPU of the memory attached at the provided index.
 *
 * In future, the class may be relaxed for non-fabric manager  client usage
 * to optimize the single node IMEX sequence. The objects of this class can
 * be then mapped on GPU just like LW_MEMORY_FABRIC.
 */
#ifndef _cl00fa_h_
#define _cl00fa_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LW_MEMORY_FABRIC_EXPORTED_REF (0x000000fa)

#define LW_FABRIC_UUID_LEN 16

/*
 *  exportUuid [IN]
 *    Universally unique identifier of the export object. This is extracted
 *    from a fabric packet.
 *
 *  index [IN]
 *    Index of the export object to which the memory object is attached.
 *
 *  flags [IN]
 *     Lwrrently unused. Must be zero for now.
 */

typedef struct {
    LwU8  exportUuid[LW_FABRIC_UUID_LEN];
    LwU16 index;
    LwU32 flags;
} LW00FA_ALLOCATION_PARAMETERS;

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif /* _cl00fa_h_ */
