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

#ifndef _clb2cc_h_
#define _clb2cc_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "clb0cc.h"

#define  MAXWELL_PROFILER_DEVICE                                    (0x0000B2CC)

/*
 * Creating the MAXWELL_PROFILER_DEVICE object:
 * - The profiler object is instantiated as a child of subdevice.
 */
typedef struct {
    /*
     * This parameter specifies the handle of the client that owns the context
     * specified by hContextTarget. This can set it to 0 where a context
     * specific operation is not needed. For context level operations see:
     * @ref LWB0CC_CTRL_CMD_RESERVE_HWPM_LEGACY, @ref LWB0CC_CTRL_CMD_RESERVE_PM_AREA_SMPC,
     * @ref LWB0CC_CTRL_CMD_ALLOC_PMA_STREAM.
     */
    LwHandle hClientTarget;

    /*
     * This parameter specifies the handle of the BC channel (or BC channel
     * group) object instance to which context-specific operations are to be
     * directed. If hClientTarget is set to 0 then this parameter is ignored.
     */
    LwHandle hContextTarget;
} LWB2CC_ALLOC_PARAMETERS;

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _clb2cc_h_ */
