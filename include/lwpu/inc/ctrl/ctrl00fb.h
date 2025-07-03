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

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl00fb.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#include "lwtypes.h"
#include "ctrl/ctrl00fa.h"


#include "ctrl/ctrlxxxx.h"

#define LW00FB_CTRL_CMD(cat, idx) \
    LWXXXX_CTRL_CMD(0x00fb, LW00FB_CTRL_##cat, idx)

/* LW00FB command categories (6bits) */
#define LW00FB_CTRL_RESERVED   (0x00U)
#define LW00FB_CTRL_IMPORT_REF (0x01U)

/*
 * LW00FB_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW00FB_CTRL_CMD_NULL   (0xfb0000U) /* finn: Evaluated from "(FINN_LW_MEMORY_FABRIC_IMPORTED_REF_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*
 * LW00FB_CTRL_CMD_VALIDATE
 *
 * Populate the imported memory, making it ready to map locally.
 *
 *  attrs [IN]
 *   Attributes associated with memory allocation. 
 *
 *  offset [IN]
 *    Offset into memory allocation to query physical addresses for.
 *
 *  totalPfns [IN]
 *    Number of PFNs in memory allocation.
 *
 *  pfnArray [IN]
 *    Array of PFNs in memory allocation (2MB page size shifted).
 *
 *  numPfns [IN]
 *    Number of valid entries in pfnArray.
 *
 *  memFlags [IN]
 *    Flags associated with memory allocation.
 *
 *  flags [IN]
 *    Flags to notify RM about errors during import.
 *
 *  bDone [out]
 *    Whether the RM is expecting additional calls to _VALIDATE.
 */
#define LW00FB_CTRL_CMD_VALIDATE            (0xfb0101U) /* finn: Evaluated from "(FINN_LW_MEMORY_FABRIC_IMPORTED_REF_IMPORT_REF_INTERFACE_ID << 8) | LW00FB_CTRL_VALIDATE_PARAMS_MESSAGE_ID" */

#define LW00FB_CTRL_VALIDATE_PFN_ARRAY_SIZE 512U

#define LW00FB_CTRL_FLAGS_NONE              0U
#define LW00FB_CTRL_FLAGS_IMPORT_FAILED     1U

#define LW00FB_CTRL_VALIDATE_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW00FB_CTRL_VALIDATE_PARAMS {
    LW_DECLARE_ALIGNED(LW_FABRIC_MEMORY_ATTRS attrs, 8);
    LW_DECLARE_ALIGNED(LwU64 offset, 8);
    LW_DECLARE_ALIGNED(LwU64 totalPfns, 8);
    LwU32  pfnArray[LW00FB_CTRL_VALIDATE_PFN_ARRAY_SIZE];
    LwU32  numPfns;
    LwU32  memFlags;
    LwU32  flags;
    LwBool bDone;
} LW00FB_CTRL_VALIDATE_PARAMS;

/* _ctrl00fb_h_ */
