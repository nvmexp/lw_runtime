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
#include "cl00fa.h"

/*
 * Class definition for referencing the imported fabric memory,
 * LW_MEMORY_FABRIC_IMPORT_V2, using the export object UUID.
 *
 * A privileged fabric manager only class, parented by client.
 *
 * No fabric events will be generated during creation or destruction of
 * this class.
 */

#ifndef _cl00fb_h_
#define _cl00fb_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LW_MEMORY_FABRIC_IMPORTED_REF (0x000000fb)

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
} LW00FB_ALLOCATION_PARAMETERS;

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif /* _cl00fb_h_ */
