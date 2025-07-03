/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2019 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl00f5.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/*
 * See KT design doc for detailed control call descriptions
 * https://docs.google.com/document/d/10w_VCjVepxFB4Yyzqpghe-XYn6nllYeVkj050BuKDX4/edit#heading=h.kpi3mj9tdxdp
 *
 */


#define LW00F5_CTRL_CMD(cat,idx)  \
    LWXXXX_CTRL_CMD(0x00f5, LW00F5_CTRL_##cat, idx)

/* LW00F5 command categories (6bits) */
#define LW00F5_CTRL_RESERVED (0x00U)
#define LW00F5_CTRL_IMPORT   (0x01U)

/*
 * LW00F5_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW00F5_CTRL_CMD_NULL (0xf50000U) /* finn: Evaluated from "(FINN_LW01_MEMORY_FABRIC_IMPORT_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*
 * LW00F5_CTRL_CMD_IMPORT_VALIDATE
 *
 * Populate the imported memory,
 * making it ready to map locally
 *
 *   offset [in]
 *      Offset into allocation to populate
 *
 *   pfnArray [in]
 *      Array of PFNs to populate the handle with
 *
 *   numPfns [in]
 *      Number of valid entries in pfnArray
 *
 *   done [out]
 *      Whether the RM is expecting additional
 *      calls to _VALIDATE
 *
 * Possible status values returned are:
 *    TODO add possible status values
 *    LW_OK
 */
#define LW00F5_CTRL_CMD_IMPORT_VALIDATE            (0xf50101U) /* finn: Evaluated from "(FINN_LW01_MEMORY_FABRIC_IMPORT_IMPORT_INTERFACE_ID << 8) | LW00F5_CTRL_IMPORT_VALIDATE_PARAMS_MESSAGE_ID" */

#define LW00F5_CTRL_IMPORT_VALIDATE_PFN_ARRAY_SIZE 1000U

#define LW00F5_CTRL_IMPORT_VALIDATE_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW00F5_CTRL_IMPORT_VALIDATE_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 offset, 8);
    LwU32  pfnArray[LW00F5_CTRL_IMPORT_VALIDATE_PFN_ARRAY_SIZE];
    LwU32  numPfns;
    LwBool done;
} LW00F5_CTRL_IMPORT_VALIDATE_PARAMS;

/* _ctrl00f5_h_ */
