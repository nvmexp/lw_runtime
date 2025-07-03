
/*
 * Copyright (c) 2001-2021, LWPU CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in 
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl0002.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#define LW0002_CTRL_CMD(cat,idx)          LWXXXX_CTRL_CMD(0x0002, LW0002_CTRL_##cat, idx)

/* Client command categories (6bits) */
#define LW0002_CTRL_RESERVED (0x00)
#define LW0002_CTRL_DMA      (0x01)


/*
 * LW0002_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *    LW_OK
 */

#define LW0002_CTRL_CMD_NULL (0x20000) /* finn: Evaluated from "(FINN_LW01_CONTEXT_DMA_RESERVED_INTERFACE_ID << 8) | 0x0" */



/*
 * LW0002_CTRL_CMD_UPDATE_CONTEXTDMA
 *
 * This command will update the parameters of the specified context dma.  The
 * context dma must be bound to a display channel.  The update is limited
 * to the display view of the context dma. Other use cases will continue to
 * use the original allocation parameters.
 *
 * This is used on platforms where memory may be moved by the operating
 * system after allocation.
 *
 * This control call supports the LWOS54_FLAGS_LOCK_BYPASS flag.
 *
 *   baseAddress
 *     This parameter, if selected by flags, indicates the new baseAddress for
 *     the ctxdma
 *   limit
 *     This parameter, if selected by flags, indicates the new limit of the
 *     ctxdma.
 *   hCtxDma
 *     ContextDma handle on which to operate. Must match the handle given to the control
 *     call.
 *   hChannel
 *     Display channel handle.  This field is ignored.
 *   hintHandle
 *     Hint value returned from HeapAllocHint which encodes information about
 *     the surface.  This is used by chips without generic kind.  Newer chips
 *     use the COMPR_INFO flag and the hintHandle must be zero.
 *   flags
 *     This parameter specifies flags which indicate which other parameters are
 *     valid.
 *       FLAGS_PAGESIZE updates the context DMA pagesize field, if not _DEFAULT
 *       FLAGS_USE_COMPR_INFO uses the surface format specified in the params, instead of hintHandle.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_OBJECT
 *    LW_ERR_ILWALID_ARGUMENT
 *    LWOS_STATUS_NOT_SUPPORTED
 */
#define LW0002_CTRL_CMD_UPDATE_CONTEXTDMA (0x20101) /* finn: Evaluated from "(FINN_LW01_CONTEXT_DMA_DMA_INTERFACE_ID << 8) | LW0002_CTRL_UPDATE_CONTEXTDMA_PARAMS_MESSAGE_ID" */

#define LW0002_CTRL_UPDATE_CONTEXTDMA_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0002_CTRL_UPDATE_CONTEXTDMA_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 baseAddress, 8);
    LW_DECLARE_ALIGNED(LwU64 limit, 8);
    LwHandle hSubDevice;
    LwHandle hCtxDma;
    LwHandle hChannel;
    LwHandle hintHandle;
    LwU32    flags;
} LW0002_CTRL_UPDATE_CONTEXTDMA_PARAMS;

#define LW0002_CTRL_CMD_UPDATE_CONTEXTDMA_FLAGS_BASEADDRESS                        0:0
#define LW0002_CTRL_CMD_UPDATE_CONTEXTDMA_FLAGS_BASEADDRESS_ILWALID                (0x00000000)
#define LW0002_CTRL_CMD_UPDATE_CONTEXTDMA_FLAGS_BASEADDRESS_VALID                  (0x00000001)
#define LW0002_CTRL_CMD_UPDATE_CONTEXTDMA_FLAGS_LIMIT                              1:1
#define LW0002_CTRL_CMD_UPDATE_CONTEXTDMA_FLAGS_LIMIT_ILWALID                      (0x00000000)
#define LW0002_CTRL_CMD_UPDATE_CONTEXTDMA_FLAGS_LIMIT_VALID                        (0x00000001)
#define LW0002_CTRL_CMD_UPDATE_CONTEXTDMA_FLAGS_HINT                               2:2
#define LW0002_CTRL_CMD_UPDATE_CONTEXTDMA_FLAGS_HINT_ILWALID                       (0x00000000)
#define LW0002_CTRL_CMD_UPDATE_CONTEXTDMA_FLAGS_HINT_VALID                         (0x00000001)
#define LW0002_CTRL_CMD_UPDATE_CONTEXTDMA_FLAGS_PAGESIZE                           4:3
#define LW0002_CTRL_CMD_UPDATE_CONTEXTDMA_FLAGS_PAGESIZE_DEFAULT                   (0x00000000)
#define LW0002_CTRL_CMD_UPDATE_CONTEXTDMA_FLAGS_PAGESIZE_4K                        (0x00000001)
#define LW0002_CTRL_CMD_UPDATE_CONTEXTDMA_FLAGS_PAGESIZE_BIG                       (0x00000002)
#define LW0002_CTRL_CMD_UPDATE_CONTEXTDMA_FLAGS_USE_COMPR_INFO                     6:5
#define LW0002_CTRL_CMD_UPDATE_CONTEXTDMA_FLAGS_USE_COMPR_INFO_NONE                (0x00000000)
#define LW0002_CTRL_CMD_UPDATE_CONTEXTDMA_FLAGS_USE_COMPR_INFO_FORMAT_PITCH        (0x00000001)
#define LW0002_CTRL_CMD_UPDATE_CONTEXTDMA_FLAGS_USE_COMPR_INFO_FORMAT_BLOCK_LINEAR (0x00000002)

/*
 * LW0002_CTRL_CMD_BIND_CONTEXTDMA
 *
 * Bind a context dma to a display channel.  Binding is no longer required for
 * Host channels, but does silently succeed.
 *
 * This control call supports the LWOS54_FLAGS_LOCK_BYPASS flag.
 *
 * This control replaces the obsolete RmBindContextDma() API.
 *
 * hChannel
 *     The channel for ctxdma bind
 *
 * Possible error codes include
 *   LW_OK
 *   LW_ERR_TOO_MANY_PRIMARIES          hash table is full
 *   LW_ERR_NO_MEMORY                   instance memory is full
 *   LW_ERR_ILWALID_OFFSET              surface is not correctly aligned
 *   LW_ERR_STATE_IN_USE                context dma was already bound given channel
 */
#define LW0002_CTRL_CMD_BIND_CONTEXTDMA                                            (0x20102) /* finn: Evaluated from "(FINN_LW01_CONTEXT_DMA_DMA_INTERFACE_ID << 8) | LW0002_CTRL_BIND_CONTEXTDMA_PARAMS_MESSAGE_ID" */

#define LW0002_CTRL_BIND_CONTEXTDMA_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW0002_CTRL_BIND_CONTEXTDMA_PARAMS {
    LwHandle hChannel;
} LW0002_CTRL_BIND_CONTEXTDMA_PARAMS;

/*
 * LW0002_CTRL_CMD_UNBIND_CONTEXTDMA
 *
 * Unbind a context dma from a display channel.
 *
 * This control call supports the LWOS54_FLAGS_LOCK_BYPASS flag.
 *
 * hChannel
 *     The display channel to unbind from
 *
 * Possible error codes include
 *   LW_OK
 *   LW_ERR_ILWALID_STATE               channel was not bound
 */
#define LW0002_CTRL_CMD_UNBIND_CONTEXTDMA (0x20103) /* finn: Evaluated from "(FINN_LW01_CONTEXT_DMA_DMA_INTERFACE_ID << 8) | LW0002_CTRL_UNBIND_CONTEXTDMA_PARAMS_MESSAGE_ID" */

#define LW0002_CTRL_UNBIND_CONTEXTDMA_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW0002_CTRL_UNBIND_CONTEXTDMA_PARAMS {
    LwHandle hChannel;
} LW0002_CTRL_UNBIND_CONTEXTDMA_PARAMS;

/* _ctrl0002.h_ */

