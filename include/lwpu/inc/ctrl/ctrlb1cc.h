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
// Source file: ctrl/ctrlb1cc.finn
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
/* MAXWELL_PROFILER control commands and parameters */

#define LWB1CC_CTRL_CMD(cat,idx)          LWXXXX_CTRL_CMD(0xB1CC, LWB1CC_CTRL_##cat, idx)

/* MAXWELL_PROFILER_CONTEXT command categories (6 bits) */
#define LWB1CC_CTRL_RESERVED (0x00)
#define LWB1CC_CTRL_PROFILER (0x01)

/*
 * LWB1CC_CTRL_CMD_NULL
 *
 *    This command does nothing.
 *    This command does not take any parameters.
 *
 * Possible status values returned are:
 *    LW_OK
 */
#define LWB1CC_CTRL_CMD_NULL (0xb1cc0000) /* finn: Evaluated from "(FINN_MAXWELL_PROFILER_CONTEXT_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*!
 * LWB1CC_CTRL_CMD_ENABLE_PMA_PULSE_CTXSW
 *
 * Enables context switch of PMA pulse. 
 * 
 */
#define LWB1CC_CTRL_CMD_ENABLE_PMA_PULSE_CTXSW (0xb1cc0101) /* finn: Evaluated from "(FINN_MAXWELL_PROFILER_CONTEXT_PROFILER_INTERFACE_ID << 8) | 0x1" */

typedef struct LWB1CC_CTRL_ENABLE_PMA_PULSE_CTXSW_PARAMS {
    /*!
     * [in] Flag to enable or disable context switch of PMA pulse 
     */
    LwBool bEnable;
} LWB1CC_CTRL_ENABLE_PMA_PULSE_CTXSW_PARAMS;


/*!
 * LWB1CC_CTRL_CMD_ENABLE_INBAND_PM_PROGRAMMING_METHOD
 *
 * Enables firmware method for inband programming of PM systems
 * reserved through @ref LWB0CC_CTRL_CMD_RESERVE_* 
 * 
 */
#define LWB1CC_CTRL_CMD_ENABLE_INBAND_PM_PROGRAMMING_METHOD (0xb1cc0102) /* finn: Evaluated from "(FINN_MAXWELL_PROFILER_CONTEXT_PROFILER_INTERFACE_ID << 8) | 0x2" */

typedef struct LWB1CC_CTRL_ENABLE_PM_METHOD_PARAMS {
    /*!
     * [in] Flag to enable or disable the firmware method.
     */
    LwBool bEnable;
} LWB1CC_CTRL_ENABLE_PM_METHOD_PARAMS;

/* _ctrlb1cc_h_ */
