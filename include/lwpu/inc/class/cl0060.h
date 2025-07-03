/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2015 by LWPU Corporation.  All rights reserved.
* All information contained herein is proprietary and
* confidential to LWPU Corporation.  Any use, reproduction, or
* disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/


#include "lwtypes.h"

#ifndef _cl0060_h_
#define _cl0060_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LW0060_SYNC_GPU_BOOST                                       (0x00000060)

/*! 
 */
typedef struct {
    LwU32 gpuBoostGroupId;
} LW0060_ALLOC_PARAMETERS;

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif // _cl0060_h

