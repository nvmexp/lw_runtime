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
// Source file: ctrl/ctrl00f8.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"

#define LW00F8_CTRL_CMD(cat,idx)       LWXXXX_CTRL_CMD(0x00f8, LW00F8_CTRL_##cat, idx)

/* LW00F8 command categories (6bits) */
#define LW00F8_CTRL_RESERVED (0x00U)
#define LW00F8_CTRL_FABRIC   (0x01U)

/*
 * LW00F8_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW00F8_CTRL_CMD_NULL (0xf80000U) /* finn: Evaluated from "(FINN_LW_MEMORY_FABRIC_RESERVED_INTERFACE_ID << 8) | 0x0" */



/*
 * LW00F8_CTRL_CMD_GET_INFO
 *
 * Queries memory allocation attributes.
 *
 *  size [OUT]
 *    Size of the allocation.
 *
 *  pageSize [OUT]
 *    Page size of the allocation.
 *
 *  allocFlags [OUT]
 *    Flags passed during the allocation.
 */
#define LW00F8_CTRL_CMD_GET_INFO (0xf80101U) /* finn: Evaluated from "(FINN_LW_MEMORY_FABRIC_FABRIC_INTERFACE_ID << 8) | LW00F8_CTRL_GET_INFO_PARAMS_MESSAGE_ID" */

#define LW00F8_CTRL_GET_INFO_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW00F8_CTRL_GET_INFO_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 size, 8);
    LwU32 pageSize;
    LwU32 allocFlags;
} LW00F8_CTRL_GET_INFO_PARAMS;

/*
 * LW00F8_CTRL_CMD_DESCRIBE
 *
 * Queries the physical attributes of the fabric memory allocation.
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
 * Note: This ctrl call is only available for kerenl mode client in vGPU platforms.
 */

#define LW00F8_CTRL_CMD_DESCRIBE            (0xf80102) /* finn: Evaluated from "(FINN_LW_MEMORY_FABRIC_FABRIC_INTERFACE_ID << 8) | LW00F8_CTRL_DESCRIBE_PARAMS_MESSAGE_ID" */

#define LW00F8_CTRL_DESCRIBE_PFN_ARRAY_SIZE 512

#define LW00F8_CTRL_DESCRIBE_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW00F8_CTRL_DESCRIBE_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 offset, 8);
    LW_DECLARE_ALIGNED(LwU64 totalPfns, 8);
    LwU32 pfnArray[LW00F8_CTRL_DESCRIBE_PFN_ARRAY_SIZE];
    LwU32 numPfns;
} LW00F8_CTRL_DESCRIBE_PARAMS;

/* _ctrl00f8_h_ */
