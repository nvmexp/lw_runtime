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

#include "lwtypes.h"

/*
 * Class definition for importing memory from a remote node
 *
 */

#ifndef _cl00f5_h_
#define _cl00f5_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LW01_MEMORY_FABRIC_IMPORT (0x000000f5)

#define LW01_MEMORY_FABRIC_IMPORT_BUFFER_SIZE 128

typedef struct LW_MEMORY_FABRIC_IMPORT_ALLOCATION_PARAMS
{
    LwU8 buffer[LW01_MEMORY_FABRIC_IMPORT_BUFFER_SIZE]; // opaque data buffer
} LW_MEMORY_FABRIC_IMPORT_ALLOCATION_PARAMS;

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif /* _cl00f5_h_ */
