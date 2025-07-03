/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2006 LWPU CORPORATION & AFFILIATES. All rights reserved.
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


 /***************************************************************************\
|*                                                                           *|
|*                         LW GPU Types                                      *|
|*                                                                           *|
|*  This header contains definitions describing LWPU's GPU hardware state. *|
|*                                                                           *|
 \***************************************************************************/


#ifndef LWGPUTYPES_INCLUDED
#define LWGPUTYPES_INCLUDED
#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

 /***************************************************************************\
|*                              LwNotification                               *|
 \***************************************************************************/

/***** LwNotification Structure *****/
/*
 * LW objects return information about method completion to clients via an
 * array of notification structures in main memory.
 *
 * The client sets the status field to LW???_NOTIFICATION_STATUS_IN_PROGRESS.
 * LW fills in the LwNotification[] data structure in the following order:
 * timeStamp, otherInfo32, otherInfo16, and then status.
 */

/* memory data structures */
typedef volatile struct LwNotificationRec {
 struct {                      /*                                   0000-    */
  LwU32 nanoseconds[2];        /* nanoseconds since Jan. 1, 1970       0-   7*/
 } timeStamp;                  /*                                       -0007*/
 LwV32 info32;                 /* info returned depends on method   0008-000b*/
 LwV16 info16;                 /* info returned depends on method   000c-000d*/
 LwV16 status;                 /* user sets bit 15, LW sets status  000e-000f*/
} LwNotification;

 /***************************************************************************\
|*                              LwGpuSemaphore                               *|
 \***************************************************************************/

/***** LwGpuSemaphore Structure *****/
/*
 * LwGpuSemaphore objects are used by the GPU to synchronize multiple
 * command-streams.
 *
 * Please refer to class documentation for details regarding the content of
 * the data[] field.
 */

/* memory data structures */
typedef volatile struct LwGpuSemaphoreRec {
 LwV32 data[2];                /* Payload/Report data               0000-0007*/
 struct {                      /*                                   0008-    */
  LwV32 nanoseconds[2];        /* nanoseconds since Jan. 1, 1970       8-   f*/
 } timeStamp;                  /*                                       -000f*/
} LwGpuSemaphore;

 /***************************************************************************\
|*                            LwGetReport                                    *|
 \***************************************************************************/

/*
 * LW objects, starting with Kelvin, return information such as pixel counts to
 * the user via the LW*_GET_REPORT method.
 *
 * The client fills in the "zero" field to any nonzero value and waits until it
 * becomes zero.  LW fills in the timeStamp, value, and zero fields.
 */
typedef volatile struct LWGetReportRec {
    struct  {                  /*                                   0000-    */
        LwU32 nanoseconds[2];  /* nanoseconds since Jan. 1, 1970       0-   7*/
    } timeStamp;               /*                                       -0007*/
    LwU32 value;               /* info returned depends on method   0008-000b*/
    LwU32 zero;                /* always written to zero            000c-000f*/
} LwGetReport;

 /***************************************************************************\
|*                           LwRcNotification                                *|
 \***************************************************************************/

/*
 * LW robust channel notification information is reported to clients via
 * standard LW01_EVENT objects bound to instance of the LW*_CHANNEL_DMA and
 * LW*_CHANNEL_GPFIFO objects.
 */
typedef struct LwRcNotificationRec {
    struct {
        LwU32 nanoseconds[2];  /* nanoseconds since Jan. 1, 1970       0-   7*/
    } timeStamp;               /*                                       -0007*/
    LwU32 exceptLevel;         /* exception level                   000c-000f*/
    LwU32 exceptType;          /* exception type                    0010-0013*/
} LwRcNotification;

 /***************************************************************************\
|*                              LwSyncPointFence                             *|
 \***************************************************************************/

/***** LwSyncPointFence Structure *****/
/*
 * LwSyncPointFence objects represent a syncpoint event.  The syncPointID
 * identifies the syncpoint register and the value is the value that the
 * register will contain right after the event oclwrs.
 *
 * If syncPointID contains LW_ILWALID_SYNCPOINT_ID then this is an invalid
 * event.  This is often used to indicate an event in the past (i.e. no need to
 * wait).
 *
 * For more info on syncpoints refer to Mobile channel and syncpoint
 * documentation.
 */
typedef struct LwSyncPointFenceRec {
    LwU32   syncPointID;
    LwU32   value;
} LwSyncPointFence;

#define LW_ILWALID_SYNCPOINT_ID ((LwU32)-1)

 /***************************************************************************\
|*                                                                           *|
|*  64 bit type definitions for use in interface structures.                 *|
|*                                                                           *|
 \***************************************************************************/

#if !defined(XAPIGEN)   /* LwOffset is XAPIGEN builtin type, so skip typedef */
typedef LwU64           LwOffset; /* GPU address                             */
#endif

#define LwOffset_HI32(n)  ((LwU32)(((LwU64)(n)) >> 32))
#define LwOffset_LO32(n)  ((LwU32)((LwU64)(n)))

/*
* There are two types of GPU-UUIDs available:
*
*  (1) a SHA-256 based 32 byte ID, formatted as a 64 character
*      hexadecimal string as "GPU-%16x-%08x-%08x-%08x-%024x"; this is
*      deprecated.
*
*  (2) a SHA-1 based 16 byte ID, formatted as a 32 character
*      hexadecimal string as "GPU-%08x-%04x-%04x-%04x-%012x" (the
*      canonical format of a UUID); this is the default.
*/
#define LW_GPU_UUID_SHA1_LEN            (16)
#define LW_GPU_UUID_SHA256_LEN          (32)
#define LW_GPU_UUID_LEN                 LW_GPU_UUID_SHA1_LEN

#ifdef __cplusplus
};
#endif

#endif /* LWGPUTYPES_INCLUDED */
