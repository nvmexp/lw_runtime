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
// Source file: ctrl/ctrl00fa.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"

#define LW00FA_CTRL_CMD(cat, idx) \
    LWXXXX_CTRL_CMD(0x00fa, LW00FA_CTRL_##cat, idx)

/* LW00FA command categories (6bits) */
#define LW00FA_CTRL_RESERVED   (0x00U)
#define LW00FA_CTRL_EXPORT_REF (0x01U)

/*
 * LW00FA_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW00FA_CTRL_CMD_NULL   (0xfa0000U) /* finn: Evaluated from "(FINN_LW_MEMORY_FABRIC_EXPORTED_REF_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*
 * LW00FA_CTRL_CMD_DESCRIBE
 *
 * Queries the physical attributes of the ref-counted exported memory allocation.
 *
 *  offset [IN]
 *    Offset into memory allocation to query physical addresses for.
 *
 *  totalPfns [OUT]
 *    Number of PFNs in memory allocation.
 *
 *  pfnArray [OUT]
 *    Array of PFNs in memory allocation (2MB page size shifted).
 *
 *  numPfns [OUT]
 *    Number of valid entries in pfnArray.
 *
 *  attrs [OUT]
 *    Attributes associated with memory allocation.
 *
 *  memFlags [OUT]
 *    Flags associated with memory allocation.
 */

#define LW00FA_CTRL_CMD_DESCRIBE            (0xfa0101U) /* finn: Evaluated from "(FINN_LW_MEMORY_FABRIC_EXPORTED_REF_EXPORT_REF_INTERFACE_ID << 8) | LW00FA_CTRL_DESCRIBE_PARAMS_MESSAGE_ID" */

#define LW00FA_CTRL_DESCRIBE_PFN_ARRAY_SIZE 512U

/*
 *  kind
 *    Kind of memory allocation.
 *
 *  pageSize
 *    Page size of memory allocation.
 */
typedef struct LW_FABRIC_MEMORY_ATTRS {
    LwU32 kind;
    LwU32 pageSize;
    LW_DECLARE_ALIGNED(LwU64 size, 8);
} LW_FABRIC_MEMORY_ATTRS;

#define LW00FA_CTRL_DESCRIBE_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW00FA_CTRL_DESCRIBE_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 offset, 8);
    LW_DECLARE_ALIGNED(LwU64 totalPfns, 8);
    LwU32 pfnArray[LW00FA_CTRL_DESCRIBE_PFN_ARRAY_SIZE];
    LwU32 numPfns;
    LW_DECLARE_ALIGNED(LW_FABRIC_MEMORY_ATTRS attrs, 8);
    LwU32 memFlags;
} LW00FA_CTRL_DESCRIBE_PARAMS;

/* _ctrl00fa_h_ */
