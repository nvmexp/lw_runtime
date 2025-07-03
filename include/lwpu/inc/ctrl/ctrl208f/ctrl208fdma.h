/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2013-2014 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl208f/ctrl208fdma.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl208f/ctrl208fbase.h"

/*
 * LW208F_CTRL_CMD_DMA_IS_SUPPORTED_SPARSE_VIRTUAL
 *
 * This command checks whether or not "sparse" virtual address ranges are
 * supported for a given chip. This API is intended for debug-use only.
 *
 *   bIsSupported
 *     Whether or not "sparse" virtual address ranges are supported.
 */
#define LW208F_CTRL_CMD_DMA_IS_SUPPORTED_SPARSE_VIRTUAL (0x208f1401) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_DMA_INTERFACE_ID << 8) | LW208F_CTRL_DMA_IS_SUPPORTED_SPARSE_VIRTUAL_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_DMA_IS_SUPPORTED_SPARSE_VIRTUAL_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW208F_CTRL_DMA_IS_SUPPORTED_SPARSE_VIRTUAL_PARAMS {
    LwBool bIsSupported;
} LW208F_CTRL_DMA_IS_SUPPORTED_SPARSE_VIRTUAL_PARAMS;

/*
 * LW208F_CTRL_CMD_DMA_GET_VAS_BLOCK_DETAILS
 *
 * This command retrieves various details of the virtual address space block
 * allocated from the virtual address space heap for the given virtual address.
 *
 *   virtualAddress
 *     Virtual address to get information about.
 *
 *   beginAddress
 *     Start address of the corresponding virtual address space block.
 *
 *   endAddress
 *     End address (inclusive) of the corresponding virtual address space
 *     block.
 *
 *   alignedAddress
 *     Aligned address of the corresponding virtual address space block.
 *
 *   pageSize
 *     Page size of the virtual address space block.
 *
 *   hVASpace
 *     Handle to an allocated VA space. If 0, it is assumed that the device's
 *     VA space should be used.
 */
#define LW208F_CTRL_CMD_DMA_GET_VAS_BLOCK_DETAILS (0x208f1402) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_DMA_INTERFACE_ID << 8) | LW208F_CTRL_DMA_GET_VAS_BLOCK_DETAILS_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_DMA_GET_VAS_BLOCK_DETAILS_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW208F_CTRL_DMA_GET_VAS_BLOCK_DETAILS_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 virtualAddress, 8);
    LW_DECLARE_ALIGNED(LwU64 beginAddress, 8);
    LW_DECLARE_ALIGNED(LwU64 endAddress, 8);
    LW_DECLARE_ALIGNED(LwU64 alignedAddress, 8);
    LwU32    pageSize;
    LwHandle hVASpace;
} LW208F_CTRL_DMA_GET_VAS_BLOCK_DETAILS_PARAMS;

/* _ctrl208fdma_h_ */

