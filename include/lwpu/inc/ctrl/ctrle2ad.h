/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2012-2015 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrle2ad.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




/* LWE2_SYNCPOINT control commands and parameters */

#include "ctrl/ctrlxxxx.h"
#define LWE2AD_CTRL_CMD(cat,idx)  \
    LWXXXX_CTRL_CMD(0xE2AD, LWE2AD_CTRL_##cat, idx)

/* LWE2_SYNCPOINT command categories (6bits) */
#define LWE2AD_CTRL_RESERVED       (0x00)
#define LWE2AD_CTRL_SYNCPOINT_BASE (0x01)

/*
 * LWE2AD_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LWE2AD_CTRL_CMD_NULL       (0xe2ad0000) /* finn: Evaluated from "(FINN_LWE2_SYNCPOINT_BASE_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*
 * LWE2AD_CTRL_SET_VALUE
 *
 * This command sets the current value of the associated syncpoint base register.
 *
 *   value
 *     This parameter specifies the desired value of the syncpoint base register.
 *
 * Possible return values
 *    LW_OK
 *    LW_ERR_ILWALID_OBJECT_HANDLE
 */
#define LWE2AD_CTRL_CMD_SET_VALUE (0xe2ad0101) /* finn: Evaluated from "(FINN_LWE2_SYNCPOINT_BASE_SYNCPOINT_BASE_INTERFACE_ID << 8) | 0x1" */

typedef struct LWE2AD_CTRL_SET_VALUE_PARAMS {
    LwU32 value;
} LWE2AD_CTRL_SET_VALUE_PARAMS;

/*
 * LWE2AD_CTRL_RESET
 *
 * This command issues a reset operation on the associated syncpoint base
 * register to 0.  This operation can be issued at both passive and elevated
 * processor levels.
 *
 * Possible return values
 *    LW_OK
 *    LW_ERR_ILWALID_OBJECT_HANDLE
 */
#define LWE2AD_CTRL_CMD_RESET     (0xe2ad0102) /* finn: Evaluated from "(FINN_LWE2_SYNCPOINT_BASE_SYNCPOINT_BASE_INTERFACE_ID << 8) | 0x2" */

/*
 * LWE2AD_CTRL_GET_VALUE
 *
 * This command returns the current value of a syncpoint base register.  This
 * operation can be issued at both passive and elevated processor levels.
 *
 *   value
 *     This parameter returns the current value of the syncpoint base.
 *
 * Possible return values
 *    LW_OK
 *    LW_ERR_ILWALID_OBJECT_HANDLE
 */
#define LWE2AD_CTRL_CMD_GET_VALUE (0xe2ad0103) /* finn: Evaluated from "(FINN_LWE2_SYNCPOINT_BASE_SYNCPOINT_BASE_INTERFACE_ID << 8) | 0x3" */

typedef struct LWE2AD_CTRL_GET_VALUE_PARAMS {
    LwU32 value;
} LWE2AD_CTRL_GET_VALUE_PARAMS;

// _ctrle2ad_h_
