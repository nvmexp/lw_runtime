/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2021-2022 by LWPU Corporation.  All rights reserved.  All 
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
// Source file: ctrl/ctrlcbca.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#include "ctrl/ctrlxxxx.h"
/* LW_COUNTER_COLLECTION_UNIT control commands and parameters */

#define LW_COUNTER_COLLECTION_UNIT_CTRL_CMD(cat,idx)                   LWXXXX_CTRL_CMD(0xCBCA, LWCBCA_CTRL_##cat, idx)

#define LWCBCA_CTRL_RESERVED                     (0x00)
#define LWCBCA_CTRL_COUNTER_COLLECTION_UNIT      (0x01)

/*
 * LW_COUNTER_COLLECTION_UNIT_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible return values:
 *   LW_OK
 */
#define LW_COUNTER_COLLECTION_UNIT_CTRL_CMD_NULL (0xcbca0000) /* finn: Evaluated from "(FINN_LW_COUNTER_COLLECTION_UNIT_RESERVED_INTERFACE_ID << 8) | 0x0" */



/*
 * LW_COUNTER_COLLECTION_UNIT_CTRL_CMD_SUBSCRIBE
 *
 * This command is used to subscribe performance counter data
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW_COUNTER_COLLECTION_UNIT_CTRL_CMD_SUBSCRIBE (0xcbca0101) /* finn: Evaluated from "(FINN_LW_COUNTER_COLLECTION_UNIT_CLW_INTERFACE_ID << 8) | 0x1" */

typedef struct LW_COUNTER_COLLECTION_UNIT_SUBSCRIBE_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 bufferSize, 8);
    LwU32 counterBlockSize;
} LW_COUNTER_COLLECTION_UNIT_SUBSCRIBE_PARAMS;

/*
 * LW_COUNTER_COLLECTION_UNIT_CTRL_CMD_UNSUBSCRIBE
 *
 * This command is used to unsubscribe performance counter data
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW_COUNTER_COLLECTION_UNIT_CTRL_CMD_UNSUBSCRIBE (0xcbca0102) /* finn: Evaluated from "(FINN_LW_COUNTER_COLLECTION_UNIT_CLW_INTERFACE_ID << 8) | 0x2" */

/* _ctrlcbca_h_ */

#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

