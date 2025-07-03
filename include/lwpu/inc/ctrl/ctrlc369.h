/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2015-2020 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrlc369.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* MMU_FAULT_BUFFER control commands and parameters */

#define LWC369_CTRL_CMD(cat,idx)          LWXXXX_CTRL_CMD(0xC369, LWC369_CTRL_##cat, idx)//sw/dev/gpu_drv/chips_a/sdk/lwpu/inc/ctrl/ctrlc365.h

/* MMU_FAULT_BUFFER command categories (6bits) */
#define LWC369_CTRL_RESERVED         (0x00)
#define LWC369_CTRL_MMU_FAULT_BUFFER (0x01)

/*
 * LWC369_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LWC369_CTRL_CMD_NULL         (0xc3690000) /* finn: Evaluated from "(FINN_MMU_FAULT_BUFFER_RESERVED_INTERFACE_ID << 8) | 0x0" */






/*
 * LWC369_CTRL_CMD_MMU_FAULT_BUFFER_REGISTER_NON_REPLAY_BUFFER
 *
 * This call creates and registers a client buffer for the non replayable faults
 *
 *    pShadowBuffer [OUT]
 *       This parameter represents the pointer to the shadow buffer
 *
 *    pShadowBufferContext [OUT]
 *       Exelwtion context for pShadowBuffer queue
 *
 *    bufferSize [OUT]
 *       Size in bytes of the shadow buffer for non replayable faults
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LWC369_CTRL_CMD_MMU_FAULT_BUFFER_REGISTER_NON_REPLAY_BUF (0xc3690101) /* finn: Evaluated from "(FINN_MMU_FAULT_BUFFER_MMU_FAULT_BUFFER_INTERFACE_ID << 8) | LWC369_CTRL_MMU_FAULT_BUFFER_REGISTER_NON_REPLAY_BUF_PARAMS_MESSAGE_ID" */

#define LWC369_CTRL_MMU_FAULT_BUFFER_REGISTER_NON_REPLAY_BUF_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWC369_CTRL_MMU_FAULT_BUFFER_REGISTER_NON_REPLAY_BUF_PARAMS {
    LW_DECLARE_ALIGNED(LwP64 pShadowBuffer, 8);
    LW_DECLARE_ALIGNED(LwP64 pShadowBufferContext, 8);
    LwU32 bufferSize;
} LWC369_CTRL_MMU_FAULT_BUFFER_REGISTER_NON_REPLAY_BUF_PARAMS;


/*
 * LWC369_CTRL_CMD_MMU_FAULT_BUFFER_UNREGISTER_NON_REPLAY_BUFFER
 *
 * This call unregisters and destroys a client buffer for the non replayable
 * faults
 * 
 *    pShadowBuffer [IN]
 *       This parameter represents the pointer to the shadow buffer
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LWC369_CTRL_CMD_MMU_FAULT_BUFFER_UNREGISTER_NON_REPLAY_BUF (0xc3690102) /* finn: Evaluated from "(FINN_MMU_FAULT_BUFFER_MMU_FAULT_BUFFER_INTERFACE_ID << 8) | LWC369_CTRL_MMU_FAULT_BUFFER_UNREGISTER_NON_REPLAY_BUF_PARAMS_MESSAGE_ID" */

#define LWC369_CTRL_MMU_FAULT_BUFFER_UNREGISTER_NON_REPLAY_BUF_PARAMS_MESSAGE_ID (0x2U)

typedef struct LWC369_CTRL_MMU_FAULT_BUFFER_UNREGISTER_NON_REPLAY_BUF_PARAMS {
    LW_DECLARE_ALIGNED(LwP64 pShadowBuffer, 8);
} LWC369_CTRL_MMU_FAULT_BUFFER_UNREGISTER_NON_REPLAY_BUF_PARAMS;

/* _ctrlc369_h_ */
