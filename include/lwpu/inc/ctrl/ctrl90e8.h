/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2016 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl90e8.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* PHYS_MEMORY_SUBALLOCATOR control commands and parameters */

#define LW90E8_CTRL_CMD(cat,idx) LWXXXX_CTRL_CMD(0x90E8, LW90E8_CTRL_##cat, idx)

/* Command categories (6 bits) */
#define LW90E8_CTRL_RESERVED              (0x00)
#define LW90E8_CTRL_PHYS_MEM_SUBALLOCATOR (0x01)

/*
 * LW90E8_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LW90E8_CTRL_CMD_NULL              (0x90e80000) /* finn: Evaluated from "(FINN_LW_PHYS_MEM_SUBALLOCATOR_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*!
 * LW90E8_CTRL_CMD_PHYS_MEM_SUBALLOCATOR_GET_INFO
 * 
 * This command can be used to fetch information of given physical memory suballocator
 *
 * freeBytes
 *   [out] Total available free in bytes
 * totalBytes
 *   [out] Total size in bytes
 * base
 *   [out] Physical base address suballocator
 * largestFreeOffset
 *   [out] Largest available free block's offset
 * largestFreeBytes
 *   [out] Largest available free block's size in bytes
 *
 * Possible status values returned are:
 *
 * LW_OK
 * LW_ERR_ILWALID_OPERATION - Suballocator not found
 * LW_ERR_NOT_SUPPORTED
 */

#define LW90E8_CTRL_CMD_PHYS_MEM_SUBALLOCATOR_GET_INFO (0x90e80102) /* finn: Evaluated from "(FINN_LW_PHYS_MEM_SUBALLOCATOR_PHYS_MEM_SUBALLOCATOR_INTERFACE_ID << 8) | LW90E8_CTRL_PHYS_MEM_SUBALLOCATOR_GET_INFO_PARAMS_MESSAGE_ID" */

#define LW90E8_CTRL_PHYS_MEM_SUBALLOCATOR_GET_INFO_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW90E8_CTRL_PHYS_MEM_SUBALLOCATOR_GET_INFO_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 freeBytes, 8);
    LW_DECLARE_ALIGNED(LwU64 totalBytes, 8);
    LW_DECLARE_ALIGNED(LwU64 base, 8);
    LW_DECLARE_ALIGNED(LwU64 largestFreeOffset, 8);
    LW_DECLARE_ALIGNED(LwU64 largestFreeBytes, 8);
} LW90E8_CTRL_PHYS_MEM_SUBALLOCATOR_GET_INFO_PARAMS;

/*!
 * LW90E8_CTRL_CMD_PHYS_MEM_SUBALLOCATOR_RESIZE
 *
 * This command can be used to resize the suballocator
 *
 * grow
 *   [in] Grow the suballocator or shrink it?
 *        true = grow, false = shrink
 * resizeBy
 *   [in] Amount to resize by
 *
 * Possible status values returned are:
 *
 * LW_OK
 * LW_ERR_ILWALID_OPERATION - Suballocator not found for this process
 * LW_ERR_ILWALID_LIMIT     - New boundaries specified for suballocator are invalid
 * LW_ERR_NO_MEMORY         - No free memory available to shrink the suballocator
 *
 * LW_ERR_NOT_SUPPORTED
 */

#define LW90E8_CTRL_CMD_PHYS_MEM_SUBALLOCATOR_RESIZE (0x90e80103) /* finn: Evaluated from "(FINN_LW_PHYS_MEM_SUBALLOCATOR_PHYS_MEM_SUBALLOCATOR_INTERFACE_ID << 8) | LW90E8_CTRL_PHYS_MEM_SUBALLOCATOR_RESIZE_PARAMS_MESSAGE_ID" */

#define LW90E8_CTRL_PHYS_MEM_SUBALLOCATOR_RESIZE_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW90E8_CTRL_PHYS_MEM_SUBALLOCATOR_RESIZE_PARAMS {
    LwBool grow;
    LW_DECLARE_ALIGNED(LwU64 resizeBy, 8);
} LW90E8_CTRL_PHYS_MEM_SUBALLOCATOR_RESIZE_PARAMS;

/* _ctrl90e8_h_ */

