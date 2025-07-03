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

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrlb2cc.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#include "ctrl/ctrlb0cc.h"
/* MAXWELL_PROFILER_DEVICE control commands and parameters */

#define LWB2CC_CTRL_CMD(cat,idx)          LWXXXX_CTRL_CMD(0xB2CC, LWB2CC_CTRL_##cat, idx)

/* MAXWELL_PROFILER_DEVICE command categories (6 bits) */
#define LWB2CC_CTRL_RESERVED (0x00)
#define LWB2CC_CTRL_PROFILER (0x01)

/*
 * LWB2CC_CTRL_CMD_NULL
 *
 *    This command does nothing.
 *    This command does not take any parameters.
 *
 * Possible status values returned are:
 *    LW_OK
 */
#define LWB2CC_CTRL_CMD_NULL (0xb2cc0000) /* finn: Evaluated from "(FINN_MAXWELL_PROFILER_DEVICE_RESERVED_INTERFACE_ID << 8) | 0x0" */






/*!
 * LWB2CC_CTRL_CMD_ENABLE_INBAND_PM_PROGRAMMING_METHOD
 *
 * Enables firmware method for inband programming of PM systems
 * reserved through @ref LWB0CC_CTRL_CMD_RESERVE_* 
 * 
 */
#define LWB2CC_CTRL_CMD_ENABLE_PM_METHOD (0xb2cc0101) /* finn: Evaluated from "(FINN_MAXWELL_PROFILER_DEVICE_PROFILER_INTERFACE_ID << 8) | 0x1" */

typedef struct LWB2CC_CTRL_ENABLE_ENABLE_PM_METHOD_PARAMS {
    /*!
     * [in] The handle of the client that owns the channel specified by hContext.
     */
    LwHandle hClient;
    /*!
     * [in] The handle of the target context (channel or channel group) object 
     *  for which user wants to enable this method.
     */
    LwHandle hContext;
    /*!
     * [in] Flag to enable or disable this method.
     */
    LwBool   bEnable;
} LWB2CC_CTRL_ENABLE_ENABLE_PM_METHOD_PARAMS;

/* _ctrlb2cc_h_ */
