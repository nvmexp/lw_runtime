/* 
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2007-2017 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrla0bc.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* LWENC_SW_SESSION control commands and parameters */
#define LWA0BC_CTRL_CMD(cat,idx)                            LWXXXX_CTRL_CMD(0xA0BC, LWA0BC_CTRL_##cat, idx)

/* Command categories (6bits) */
#define LWA0BC_CTRL_RESERVED                         (0x00)
#define LWA0BC_CTRL_LWENC_SW_SESSION                 (0x01)

#define LWA0BC_CTRL_CMD_LWENC_MAX_BUFFER_ENTRIES     60

/*
 * LWA0BC_CTRL_CMD_LWENC_SW_SESSION_UPDATE_INFO
 *
 * This command is used to let RM know about runtime information about
 * running LWENC session on given GPU.
 *  *
 *   hResolution
 *     This parameter specifies the current horizontal resolution of LWENC session.
 *   vResolution
 *     This parameter specifies the current vertical resolution of LWENC session.
 *   averageEncodeLatency
 *     This field specifies the average encode latency over last 1 second.
 *   averageEncodeFps
 *     This field specifies the average encode FPS over last 1 second.
 *   timestampBufferSize
 *     This field specifies the number of entries in the caller's timestampBuffer.
 *     It should not be greater than LWA0BC_CTRL_CMD_LWENC_MAX_BUFFER_ENTRIES.
 *     When this field is zero, RM will assume that client has callwlated averageEncodeFps
 *     and averageEncodeLatency, thus ignore timestampBuffer.
 *   timestampBuffer
 *     This field specifies a pointer in the caller's address space
 *     to the buffer holding encode timestamps in microseconds.
 *     This buffer must be at least as big as timestampBufferSize multiplied
 *     by the size of the LWA0BC_CTRL_LWENC_TIMESTAMP structure.
 *     e.g. if there are 10 fps, buffer will contain only 10 entries and rest of
 *     entries should be 0x00. However if there are more than 60 fps, buffer will
 *     contain last/latest 60 entries of frame encoding start-end timestamps. Caller
 *     should make sure timestamps won't wrap around. RM assume that for each
 *     frame timestamp value endTime would be greater than startTime.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LWA0BC_CTRL_CMD_LWENC_SW_SESSION_UPDATE_INFO (0xa0bc0101) /* finn: Evaluated from "(FINN_LWENC_SW_SESSION_LWENC_SW_SESSION_INTERFACE_ID << 8) | LWA0BC_CTRL_LWENC_SW_SESSION_UPDATE_INFO_PARAMS_MESSAGE_ID" */

typedef struct LWA0BC_CTRL_LWENC_TIMESTAMP {
    LW_DECLARE_ALIGNED(LwU64 startTime, 8);
    LW_DECLARE_ALIGNED(LwU64 endTime, 8);
} LWA0BC_CTRL_LWENC_TIMESTAMP;

#define LWA0BC_CTRL_LWENC_SW_SESSION_UPDATE_INFO_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWA0BC_CTRL_LWENC_SW_SESSION_UPDATE_INFO_PARAMS {
    LwU32 hResolution;
    LwU32 vResolution;
    LwU32 averageEncodeLatency;
    LwU32 averageEncodeFps;
    LwU32 timestampBufferSize;
    LW_DECLARE_ALIGNED(LwP64 timestampBuffer, 8);
} LWA0BC_CTRL_LWENC_SW_SESSION_UPDATE_INFO_PARAMS;

/* _ctrla0bc_h_ */
