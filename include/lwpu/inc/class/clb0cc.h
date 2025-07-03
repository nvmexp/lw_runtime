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

#ifndef _clb0cc_h_
#define _clb0cc_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  MAXWELL_PROFILER                                            (0x0000B0CC)

/*
 * This is an interface definition for MAXWELL_PROFILER class and cannot
 * be instantiated by clients.
 * MAXWELL_PROFILER_CONTEXT and MAXWELL_PROFILER_DEVICE extends this
 * interface to define interface for context and device level profiling
 * respectively.
 */

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _clb0cc_h_ */
