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
// Source file: ctrl/ctrla0bd.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* LWFBC_SW_SESSION control commands and parameters */
#define LWA0BD_CTRL_CMD(cat,idx)                            LWXXXX_CTRL_CMD(0xA0BD, LWA0BD_CTRL_##cat, idx)

/* Command categories (6bits) */
#define LWA0BD_CTRL_RESERVED                         (0x00)
#define LWA0BD_CTRL_LWFBC_SW_SESSION                 (0x01)

#define LWA0BD_CTRL_CMD_LWFBC_MAX_TIMESTAMP_ENTRIES  60

/*
 * LWA0BD_CTRL_CMD_LWFBC_SW_SESSION_UPDATE_INFO
 *
 * This command is used to let RM know about runtime information about
 * LWFBC session on given GPU.
 *  *
  *   hResolution
 *     This parameter specifies the current horizontal resolution of LWFBC session.
 *   vResolution
 *     This parameter specifies the current vertical resolution of LWFBC session.
 *   captureCallFlags
 *     This field specifies the flags associated with the capture call and the session.
 *     One of the flags specifies whether the user made the capture with wait or not.
 *   totalGrabCalls
 *     This field specifies the total number of grab calls made by the user.
 *   averageLatency
 *     This field specifies the average capture latency over last 1 second.
 *   averageFPS
 *     This field specifies the average frames captured.
  *   timestampEntryCount
 *     This field specifies the number of entries in the timestampEntry array.
 *     It should not be greater than LWA0BD_CTRL_CMD_LWFBC_MAX_TIMESTAMP_ENTRIES.
 *     When this field is zero, RM will assume that client has callwlated averageFBCFps
 *     and averageFBCLatency, thus ignore timestampEntry array.
 *   timestampEntry
 *     This field specifies a array holding capture timestamps in microseconds.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LWA0BD_CTRL_CMD_LWFBC_SW_SESSION_UPDATE_INFO (0xa0bd0101) /* finn: Evaluated from "(FINN_LWFBC_SW_SESSION_LWFBC_SW_SESSION_INTERFACE_ID << 8) | LWA0BD_CTRL_LWFBC_SW_SESSION_UPDATE_INFO_PARAMS_MESSAGE_ID" */

typedef struct LWA0BD_CTRL_LWFBC_TIMESTAMP {
    LW_DECLARE_ALIGNED(LwU64 startTime, 8);
    LW_DECLARE_ALIGNED(LwU64 endTime, 8);
} LWA0BD_CTRL_LWFBC_TIMESTAMP;

#define LWA0BD_CTRL_LWFBC_SW_SESSION_UPDATE_INFO_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWA0BD_CTRL_LWFBC_SW_SESSION_UPDATE_INFO_PARAMS {
    LwU32 hResolution;
    LwU32 vResolution;
    LwU32 captureCallFlags;
    LwU32 totalGrabCalls;
    LwU32 averageLatency;
    LwU32 averageFPS;
    LwU32 timestampEntryCount;
    LW_DECLARE_ALIGNED(LWA0BD_CTRL_LWFBC_TIMESTAMP timestampEntry[LWA0BD_CTRL_CMD_LWFBC_MAX_TIMESTAMP_ENTRIES], 8);
} LWA0BD_CTRL_LWFBC_SW_SESSION_UPDATE_INFO_PARAMS;

#define LWA0BD_LWFBC_WITH_WAIT                    1:0
#define LWA0BD_LWFBC_WITH_WAIT_FALSE    (0x00000000)
#define LWA0BD_LWFBC_WITH_WAIT_INFINITE (0x00000001)
#define LWA0BD_LWFBC_WITH_WAIT_TIMEOUT  (0x00000010)

/* _ctrla0bd_h_ */
