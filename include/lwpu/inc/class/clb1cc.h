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

#ifndef _clb1cc_h_
#define _clb1cc_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "clb0cc.h"

#define  MAXWELL_PROFILER_CONTEXT                                    (0x0000B1CC)

/*
 * Creating the MAXWELL_PROFILER_CONTEXT object:
 * - The profiler object is instantiated as a child of either a bc channel
 *   group or bc channel.
 */
typedef struct {
    /*
     * Handle of a specific subdevice of a broadcast device.
     */
    LwHandle hSubDevice;
} LWB1CC_ALLOC_PARAMETERS;

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _clb1cc_h_ */
