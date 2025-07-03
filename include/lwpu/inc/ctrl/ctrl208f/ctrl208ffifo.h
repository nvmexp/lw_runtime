/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2009-2021 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl208f/ctrl208ffifo.finn
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
 * LW208F_CTRL_CMD_FIFO_CHECK_ENGINE_CONTEXT
 *
 * This command checks whether or not engine context exists for a given
 * engine for the channel with a given channel ID. This API is intended
 * for testing virtual context. For debug only.
 *
 *   hChannel
 *     The handle to the channel.
 *   engine
 *     The engine ID.
 *     Valid values are:
 *        LW2080_ENGINE_TYPE_GRAPHICS
 *   exists
 *     The output are TRUE or FALSE.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_DEVICE
 *   LW_ERR_ILWALID_CHANNEL
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW208F_CTRL_CMD_FIFO_CHECK_ENGINE_CONTEXT (0x208f0401) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_FIFO_INTERFACE_ID << 8) | LW208F_CTRL_FIFO_CHECK_ENGINE_CONTEXT_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_FIFO_CHECK_ENGINE_CONTEXT_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW208F_CTRL_FIFO_CHECK_ENGINE_CONTEXT_PARAMS {
    LwHandle hChannel;
    LwU32    engine;
    LwBool   exists;
} LW208F_CTRL_FIFO_CHECK_ENGINE_CONTEXT_PARAMS;

/*
 * LW208F_CTRL_CMD_FIFO_ENABLE_VIRTUAL_CONTEXT
 *
 * This command enables virtual context for a given channel (for all engines).
 * This API is intended for testing virtual context. For debug only.
 *
 *   hChannel
 *     The handle to the channel.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_DEVICE
 *   LW_ERR_ILWALID_CHANNEL
 */
#define LW208F_CTRL_CMD_FIFO_ENABLE_VIRTUAL_CONTEXT (0x208f0402) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_FIFO_INTERFACE_ID << 8) | LW208F_CTRL_FIFO_ENABLE_VIRTUAL_CONTEXT_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_FIFO_ENABLE_VIRTUAL_CONTEXT_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW208F_CTRL_FIFO_ENABLE_VIRTUAL_CONTEXT_PARAMS {
    LwHandle hChannel;
} LW208F_CTRL_FIFO_ENABLE_VIRTUAL_CONTEXT_PARAMS;

/*
 * LW208F_CTRL_CMD_FIFO_GET_CHANNEL_STATE
 *
 * This command returns the fifo channel state for the given channel.
 * This is for testing channel behavior.  For debug only.
 *
 *   hChannel 
 *    The handle to the channel
 *   hClient 
 *    The handle to the client
 *   bound
 *      The channel has been bound to channel RAM
 *   enabled
 *      The channel is able to run.
 *   scheduled
 *      The channel has been scheduled to run.
 *   cpuMap
 *      There is a cpu mapping available to this channel.
 *   contention
 *      The virtual channel is under contention
 *   runlistSet
 *      A runlist has been chosen for this channel
 *   deferRC
 *      An RC error has oclwrred, but recovery will occur at channel teardown.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_DEVICE
 *   LW_ERR_ILWALID_CHANNEL
 */
#define LW208F_CTRL_CMD_FIFO_GET_CHANNEL_STATE (0x208f0403) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_FIFO_INTERFACE_ID << 8) | LW208F_CTRL_FIFO_GET_CHANNEL_STATE_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_FIFO_GET_CHANNEL_STATE_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW208F_CTRL_FIFO_GET_CHANNEL_STATE_PARAMS {
    LwHandle hChannel;
    LwHandle hClient;
    LwBool   bBound;
    LwBool   bEnabled;
    LwBool   bScheduled;
    LwBool   bCpuMap;
    LwBool   bContention;
    LwBool   bRunlistSet;
    LwBool   bDeferRC;
} LW208F_CTRL_FIFO_GET_CHANNEL_STATE_PARAMS;

/*
 * LW208F_CTRL_CMD_FIFO_GET_CONTIG_RUNLIST_POOL
 *
 * This command returns the location of the pool runlists are allocated from for
 * WPR testing.
 * For debug only.
 *
 * physAddr [out]
 *    Physical address of the pool
 *
 * size [out]
 *    Size in bytes of the pool
 *
 */
#define LW208F_CTRL_CMD_FIFO_GET_CONTIG_RUNLIST_POOL (0x208f0404) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_FIFO_INTERFACE_ID << 8) | LW208F_CTRL_FIFO_GET_CONTIG_RUNLIST_POOL_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_FIFO_GET_CONTIG_RUNLIST_POOL_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW208F_CTRL_FIFO_GET_CONTIG_RUNLIST_POOL_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 physAddr, 8);
    LW_DECLARE_ALIGNED(LwU64 size, 8);
} LW208F_CTRL_FIFO_GET_CONTIG_RUNLIST_POOL_PARAMS;

/* _ctrl208ffifo_h_ */

