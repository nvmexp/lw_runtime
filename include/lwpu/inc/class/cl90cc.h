/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2013 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _cl90cc_h_
#define _cl90cc_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  GF100_PROFILER                                            (0x000090CC)

/*
 * Creating the GF100_PROFILER object:
 * - The profiler object is instantiated as a child of either the subdevice or
 *   a channel group or channel, depending on whether reservations
 *   should be global to the subdevice or per-context. When the profiler
 *   requests a reservation or information about outstanding reservations, the
 *   scope of the request is determined by the profiler object's parent class.
 */

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl90cc_h_ */
