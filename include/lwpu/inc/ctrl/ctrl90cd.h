/*
 * SPDX-FileCopyrightText: Copyright (c) 2017-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
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
// Source file: ctrl/ctrl90cd.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* LW_EVENT_BUFFER control commands and parameters */

#define LW_EVENT_BUFFER_CTRL_CMD(cat,idx)           LWXXXX_CTRL_CMD(0x90CD, LW90CD_CTRL_##cat, idx)

#define LW90CD_CTRL_RESERVED          (0x00)
#define LW90CD_CTRL_EVENT             (0x01)

/*
* LW_EVENT_BUFFER_CTRL_CMD_NULL
*
* This command does nothing.
* This command does not take any parameters.
*
* Possible status values returned are:
*   LW_OK
*/
#define LW_EVENT_BUFFER_CTRL_CMD_NULL (0x90cd0000) /* finn: Evaluated from "(FINN_LW_EVENT_BUFFER_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*
* LW_EVENT_BUFFER_CTRL_CMD_ENABLE_EVENTS
*   This interface enables all the events that are associated to the event buffer
*/
#define LW_EVENT_BUFFER_CTRL_CMD_ENABLE_EVENTS           (0x90cd0101) /* finn: Evaluated from "(FINN_LW_EVENT_BUFFER_EVENT_INTERFACE_ID << 8) | 0x1" */

#define LW_EVENT_BUFFER_FLAG                                    0:32

/*
* LW_EVENT_BUFFER_FLAG_OVERFLOW_POLICY
*   This flag defines the kernel behavior when the buffer is full
*
*   DEFAULT/DISABLED: By default kernel doesn't assume any policy. To enable events
*                     an overflow policy has to be set to retain older or newer events
*
*   KEEP_OLDEST: kernel would retain older events and drop newer events if the buffer is full
*
*   KEEP_NEWEST: kernel would retain newer events and drop older events if the buffer is full
*
*/
#define LW_EVENT_BUFFER_FLAG_OVERFLOW_POLICY                    0:1
#define LW_EVENT_BUFFER_FLAG_OVERFLOW_POLICY_DISABLED    0
#define LW_EVENT_BUFFER_FLAG_OVERFLOW_POLICY_KEEP_OLDEST 1
#define LW_EVENT_BUFFER_FLAG_OVERFLOW_POLICY_KEEP_NEWEST 2
#define LW_EVENT_BUFFER_FLAG_OVERFLOW_POLICY_DEFAULT     LW_EVENT_BUFFER_FLAG_OVERFLOW_POLICY_DISABLED

/*
* LW_EVENT_BUFFER_CTRL_CMD_ENABLE_EVENTS_PARAMS
*
*   enable [IN]
*       This field is used to enable or disable events
*
*   flags[IN]
*       This field sets LW_EVENT_BUFFER_FLAG parameter used to configure event buffer overflow options
*
*   Possible status values returned are:
*       LW_OK
*       LW_ERR_ILWALID_ARGUMENT
*/
typedef struct LW_EVENT_BUFFER_CTRL_CMD_ENABLE_EVENTS_PARAMS {
    LwBool enable;
    LwU32  flags;
} LW_EVENT_BUFFER_CTRL_CMD_ENABLE_EVENTS_PARAMS;

/*
* LW_EVENT_BUFFER_CTRL_CMD_UPDATE_GET
*   This interface allows the user to update get pointers.
*   This call is useful in the KEEP_OLDEST policy to update free space available in the buffer.
*   In keep oldest policy, kernel adds new entries in the buffer only if there is free space.
*   The full/empty decision is made as follows:
*   -   when GET==PUT, the fifo is empty
*   -   when GET==PUT+1, the fifo is full
*/
#define LW_EVENT_BUFFER_CTRL_CMD_UPDATE_GET (0x90cd0102) /* finn: Evaluated from "(FINN_LW_EVENT_BUFFER_EVENT_INTERFACE_ID << 8) | 0x2" */

/*
* LW_EVENT_BUFFER_CTRL_CMD_UPDATE_GET_PARAMS
*
*   recordBufferGet [IN]
*       Value to be used to update the get offset of record buffer
*
*   varDataBufferGet[IN]
*       This is the buffer offset up to which user has consumed the vardataBuffer
*
*   Possible status values returned are:
*       LW_OK
*       LW_ERR_ILWALID_ARGUMENT: if any of the get offsets is greater than respective bufferSize.
*/
typedef struct LW_EVENT_BUFFER_CTRL_CMD_UPDATE_GET_PARAMS {
    LwU32 recordBufferGet;
    LwU32 varDataBufferGet;
} LW_EVENT_BUFFER_CTRL_CMD_UPDATE_GET_PARAMS;

/*
 * Send a test event-buffer notification (verification-only)
 */
#define LW_EVENT_BUFFER_CTRL_CMD_VERIF_NOTIFY         (0x90cd0103) /* finn: Evaluated from "(FINN_LW_EVENT_BUFFER_EVENT_INTERFACE_ID << 8) | 0x3" */

/*
 * Synchronous flush
 */
#define LW_EVENT_BUFFER_CTRL_CMD_FLUSH                (0x90cd0104) /* finn: Evaluated from "(FINN_LW_EVENT_BUFFER_EVENT_INTERFACE_ID << 8) | 0x4" */

/*
 * post event
 */
#define LW_EVENT_BUFFER_CTRL_CMD_POST_TELEMETRY_EVENT (0x90cd0105) /* finn: Evaluated from "(FINN_LW_EVENT_BUFFER_EVENT_INTERFACE_ID << 8) | 0x5" */

 /*
  *  LW_EVENT_BUFFER_CTRL_CMD_POST_TELEMETRY_EVENT_PARAMS
  *
  * eventType [IN]
  *     the LwTelemetry event type.
  * typeVersion [IN]
  *     the version of the event structure
  * eventData [IN]
  *     an array of 256 bytes used to hold the event data.
  * eventDataSz [IN]
  *     the amount of valid data in the eventData buffer.
  * varData [IN]
  *     an array of 256 bytes used to hold the var data.
  * varDataSz [IN]
  *     the amount of valid data in the varData buffer.
  *
  *   Possible status values returned are:
  *       LW_OK
  *       LW_ERR_ILWALID_ARGUMENT
  */
typedef struct LW_EVENT_BUFFER_CTRL_CMD_POST_TELEMETRY_EVENT_PARAMS {
    LwU32 eventType;
    LwU16 typeVersion;
    LwU8  eventData[256];
    LwU16 eventDataSz;
    LwU8  varData[256];
    LwU16 varDataSz;
} LW_EVENT_BUFFER_CTRL_CMD_POST_TELEMETRY_EVENT_PARAMS;

/* _ctr l90cd_h_ */

