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
 * Class definition for exporting memory to a different RM client.
 * No memory is allocated, only a class containing metadata
 * for the exporting memory is created for use in other calls.
 * Allocating this object increments a reference counter on the
 * parent memory object.
 *
 */

#ifndef _cl00f4_h_
#define _cl00f4_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LW01_MEMORY_FABRIC_EXPORT       (0x000000f4)

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif /* _cl00f4_h_ */
