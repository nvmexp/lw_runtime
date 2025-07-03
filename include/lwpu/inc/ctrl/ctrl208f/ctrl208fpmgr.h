/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2011-2015 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl208f/ctrl208fpmgr.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl208f/ctrl208fbase.h"

/* LW20_SUBDEVICE_DIAG pmgr control commands and parameters */

/* 
 * LW208F_CTRL_CMD_PMGR_SET_GPIO_INTERRUPT_NOTIFICATION 
 *  
 * Enables/didables interrupt notification on specified GPIO pin 
 *  
 *    gpioPin
 *     GPIO pin to enable interrupt notification on
 *    isEnable
 *     'true' to enable notification, 'false' to disable
 *    direction
 *     1 for rising, 0 for falling transition interrupt
 *  
 * Possible status values returned are: 
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
*/

#define LW208F_CTRL_CMD_PMGR_SET_GPIO_INTERRUPT_NOTIFICATION (0x208f1301) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_PMGR_INTERFACE_ID << 8) | LW208F_CTRL_PMGR_GPIO_INTERRUPT_NOTIFICATION_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_PMGR_GPIO_INTERRUPT_NOTIFICATION_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW208F_CTRL_PMGR_GPIO_INTERRUPT_NOTIFICATION_PARAMS {
    LwU32  gpioPin;
    LwBool isEnable;
    LwU32  direction;
} LW208F_CTRL_PMGR_GPIO_INTERRUPT_NOTIFICATION_PARAMS;

/* _ctrl208fpmgr_h_ */
