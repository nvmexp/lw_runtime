/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2006-2019 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl5080.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#include "ctrl/ctrl0080/ctrl0080dma.h"  /* LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS */
#include "ctrl/ctrl2080/ctrl2080dma.h"  /* LW2080_CTRL_DMA_* */
#include "ctrl/ctrl2080/ctrl2080fb.h"   /* LW2080_CTRL_FB_* */
#include "ctrl/ctrl2080/ctrl2080fifo.h" /* LW2080_CTRL_FIFO_* */
#include "ctrl/ctrl2080/ctrl2080gpu.h"  /* LW2080_CTRL_GPU_* */
#include "ctrl/ctrl2080/ctrl2080gr.h"   /* LW2080_CTRL_GR_* */
/* LW5080_DEFERRED_API - deferred RmControl commands */


#define LW5080_CTRL_CMD(cat,idx) LWXXXX_CTRL_CMD(0x5080, LW5080_CTRL_##cat,idx)

/* Command categories (6bits) */
#define LW5080_CTRL_RESERVED (0x00)
#define LW5080_CTRL_DEFERRED (0x01)

/*
 * LW5080_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *    LW_OK
 */
#define LW5080_CTRL_CMD_NULL (0x50800000) /* finn: Evaluated from "(FINN_LW50_DEFERRED_API_CLASS_RESERVED_INTERFACE_ID << 8) | 0x0" */






/*
 * LW5080_CTRL_CMD_DEFERRED_API (deprecated; use LW5080_CTRL_CMD_DEFERRED_API_V2 instead)
 *
 * This command sets up a deferred api call.
 *    hApiHandle
 *      Client Unique Handle which is used as the data of a sw method to ilwoke
 *      the api in the future
 *    cmd
 *      The rmcontrol cmd to ilwoke as a deferred api.
 *    flags_delete
 *      Indicates if an explicit delete is needed (default behavior is to
 *      auto delete after SW method has exelwted/completed).
 *    flags_wait_for_tlb_flush
 *      Indicates if the API should wait for an IlwalidateTlb to also occur
 *      (not just that it's exelwted) before being considered completed and
 *      works in conjunction with flags_delete.
 *    hClientVA, hDeviceVA
 *      Client/Device handles of the owner of the virtual address space to
 *      to be updated (used with the FillPteMem API bundle)
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */

#define LW5080_CTRL_CMD_DEFERRED_API (0x50800101) /* finn: Evaluated from "(FINN_LW50_DEFERRED_API_CLASS_DEFERRED_INTERFACE_ID << 8) | LW5080_CTRL_DEFERRED_API_PARAMS_MESSAGE_ID" */

#define LW5080_CTRL_DEFERRED_API_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW5080_CTRL_DEFERRED_API_PARAMS {
    LwHandle hApiHandle;
    LwU32    cmd;
    LwU32    flags;
    LwHandle hClientVA;
    LwHandle hDeviceVA;

    union {
        LW_DECLARE_ALIGNED(LW2080_CTRL_GPU_INITIALIZE_CTX_PARAMS InitCtx, 8);

        LW_DECLARE_ALIGNED(LW2080_CTRL_GPU_PROMOTE_CTX_PARAMS PromoteCtx, 8);

        LW2080_CTRL_GPU_EVICT_CTX_PARAMS                 EvictCtx;

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
        LW2080_CTRL_DMA_ILWALIDATE_TLB_PARAMS            IlwalidateTlb;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
        LW_DECLARE_ALIGNED(LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS FillPteMem, 8);
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
        LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_PARAMS     CacheAllocPolicy;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


        LW_DECLARE_ALIGNED(LW2080_CTRL_GR_CTXSW_ZLWLL_BIND_PARAMS ZlwllCtxsw, 8);

        LW_DECLARE_ALIGNED(LW2080_CTRL_GR_CTXSW_PM_BIND_PARAMS PmCtxsw, 8);

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
        LW2080_CTRL_FB_GPU_CACHE_PROMOTION_POLICY_PARAMS CachePromotePolicy;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


        LW_DECLARE_ALIGNED(LW2080_CTRL_GR_CTXSW_PREEMPTION_BIND_PARAMS PreemptionCtxsw, 8);
    } api_bundle;
} LW5080_CTRL_DEFERRED_API_PARAMS;

#define LW5080_CTRL_CMD_DEFERRED_API_FLAGS_DELETE                       0:0
#define LW5080_CTRL_CMD_DEFERRED_API_FLAGS_DELETE_EXPLICIT          (0x00000001)
#define LW5080_CTRL_CMD_DEFERRED_API_FLAGS_DELETE_IMPLICIT          (0x00000000)

#define LW5080_CTRL_CMD_DEFERRED_API_FLAGS_WAIT_FOR_TLB_FLUSH           1:1
#define LW5080_CTRL_CMD_DEFERRED_API_FLAGS_WAIT_FOR_TLB_FLUSH_FALSE (0x00000000)
#define LW5080_CTRL_CMD_DEFERRED_API_FLAGS_WAIT_FOR_TLB_FLUSH_TRUE  (0x00000001)
/*
 * LW5080_CTRL_CMD_DEFERRED_API_V2
 *
 * This command sets up a deferred api call.
 *    hApiHandle
 *      Client Unique Handle which is used as the data of a sw method to ilwoke
 *      the api in the future
 *    cmd
 *      The rmcontrol cmd to ilwoke as a deferred api.
 *    flags_delete
 *      Indicates if an explicit delete is needed (default behavior is to
 *      auto delete after SW method has exelwted/completed).
 *    flags_wait_for_tlb_flush
 *      Indicates if the API should wait for an IlwalidateTlb to also occur
 *      (not just that it's exelwted) before being considered completed and
 *      works in conjunction with flags_delete.
 *    hClientVA, hDeviceVA
 *      Client/Device handles of the owner of the virtual address space to
 *      to be updated (used with the FillPteMem API bundle)
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */

#define LW5080_CTRL_CMD_DEFERRED_API_V2                             (0x50800103) /* finn: Evaluated from "(FINN_LW50_DEFERRED_API_CLASS_DEFERRED_INTERFACE_ID << 8) | LW5080_CTRL_DEFERRED_API_V2_PARAMS_MESSAGE_ID" */

#define LW5080_CTRL_DEFERRED_API_V2_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW5080_CTRL_DEFERRED_API_V2_PARAMS {
    LwHandle hApiHandle;
    LwU32    cmd;
    LwU32    flags;
    LwHandle hClientVA;
    LwHandle hDeviceVA;

    union {
        LW_DECLARE_ALIGNED(LW2080_CTRL_GPU_INITIALIZE_CTX_PARAMS InitCtx, 8);

        LW_DECLARE_ALIGNED(LW2080_CTRL_GPU_PROMOTE_CTX_PARAMS PromoteCtx, 8);

        LW2080_CTRL_GPU_EVICT_CTX_PARAMS                 EvictCtx;

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
        LW2080_CTRL_DMA_ILWALIDATE_TLB_PARAMS            IlwalidateTlb;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
        LW_DECLARE_ALIGNED(LW0080_CTRL_DMA_FILL_PTE_MEM_PARAMS FillPteMem, 8);
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
        LW2080_CTRL_FB_GPU_CACHE_ALLOC_POLICY_V2_PARAMS  CacheAllocPolicy;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


        LW_DECLARE_ALIGNED(LW2080_CTRL_GR_CTXSW_ZLWLL_BIND_PARAMS ZlwllCtxsw, 8);

        LW_DECLARE_ALIGNED(LW2080_CTRL_GR_CTXSW_PM_BIND_PARAMS PmCtxsw, 8);

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
        LW2080_CTRL_FB_GPU_CACHE_PROMOTION_POLICY_PARAMS CachePromotePolicy;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
        LW_DECLARE_ALIGNED(LW2080_CTRL_FIFO_DISABLE_CHANNELS_PARAMS DisableChannels, 8);
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


        LW_DECLARE_ALIGNED(LW2080_CTRL_GR_CTXSW_PREEMPTION_BIND_PARAMS PreemptionCtxsw, 8);

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
        LW_DECLARE_ALIGNED(LW2080_CTRL_FIFO_UPDATE_CHANNEL_INFO_PARAMS ChannelInfoUpdate, 8);
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

    } api_bundle;
} LW5080_CTRL_DEFERRED_API_V2_PARAMS;

/*
 * LW5080_CTRL_CMD_REMOVE_API
 *
 * This command removes an explicit deferred api call.
 *    hApiHandle
 *      Client Unique Handle which is used as the data of a sw method to ilwoke
 *      the api in the future
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */

#define LW5080_CTRL_CMD_REMOVE_API (0x50800102) /* finn: Evaluated from "(FINN_LW50_DEFERRED_API_CLASS_DEFERRED_INTERFACE_ID << 8) | LW5080_CTRL_REMOVE_API_PARAMS_MESSAGE_ID" */

#define LW5080_CTRL_REMOVE_API_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW5080_CTRL_REMOVE_API_PARAMS {
    LwHandle hApiHandle;
} LW5080_CTRL_REMOVE_API_PARAMS;

/* _ctrl5080_h_ */

