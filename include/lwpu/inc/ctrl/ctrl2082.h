/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//
// This file should be NEVER published.
//

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl2082.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#define LW2082_CTRL_CMD(cat,idx)  \
    LWXXXX_CTRL_CMD(0x2082, LW2082_CTRL_##cat, idx)

/* Command categories (6bits) */
#define LW2082_CTRL_RESERVED            (0x00)
#define LW2082_CTRL_BINAPI              (0x01)


/*
 * LW2082_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *    LW_OK
 */
#define LW2082_CTRL_CMD_NULL            (0x20820000) /* finn: Evaluated from "(FINN_LW2082_BINAPI_RESERVED_INTERFACE_ID << 8) | 0x0" */


/*
 * LW2082_CTRL_CMD_SET_BINAPI_TEST
 *
 * This command is used to test the functioning of LW2081_BINAPI_PRIV class
 * dummy variable used as, in: 0xdeadbeef, out: 0xf0f0f0f0
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 */

#define LW2082_CTRL_CMD_SET_BINAPI_TEST (0x20820101) /* finn: Evaluated from "(FINN_LW2082_BINAPI_INTERFACE_ID << 8) | LW2082_CTRL_CMD_SET_BINAPI_TEST_PARAMS_MESSAGE_ID" */

#define LW2082_CTRL_CMD_SET_BINAPI_TEST_PARAMS_MESSAGE_ID (0x01U)

typedef struct LW2082_CTRL_CMD_SET_BINAPI_TEST_PARAMS {
    LwU32 dummy;
} LW2082_CTRL_CMD_SET_BINAPI_TEST_PARAMS;

/* _ctrl2082_h_ */
