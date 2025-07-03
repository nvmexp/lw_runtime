/*
 * SPDX-FileCopyrightText: Copyright (c) 2004-2015 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080lwd.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)




#include "ctrl/ctrl2080/ctrl2080base.h"

#include "ctrl/ctrlxxxx.h"
/*
 * LW2080_CTRL_CMD_LWD_GET_DUMP_SIZE
 *
 * This command gets the expected dump size of a particular GPU dump component.
 * Note that events that occur between this command and a later
 * LW2080_CTRL_CMD_LWD_GET_DUMP command could alter the size of
 * the buffer required.
 *
 *   component
 *     One of LWDUMP_COMPONENT < 0x400 defined in lwdump.h to estimate
 *     the size of.
 *   size
 *     This parameter returns the expected size.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT if component is invalid.
 *
 */

#define LW2080_CTRL_CMD_LWD_GET_DUMP_SIZE (0x20802401) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWD_INTERFACE_ID << 8) | LW2080_CTRL_LWD_GET_DUMP_SIZE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_LWD_GET_DUMP_SIZE_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_LWD_GET_DUMP_SIZE_PARAMS {
    LwU32 component;
    LwU32 size;
} LW2080_CTRL_LWD_GET_DUMP_SIZE_PARAMS;

/*
 * LW2080_CTRL_CMD_LWD_GET_DUMP
 *
 * This command gets a dump of a particular GPU dump component. If triggers
 * is non-zero, the command waits for the trigger to occur
 * before it returns.
 *
 *   pBuffer
 *     This parameter points to the buffer for the data.
 *  component
 *     One of LWDUMP_COMPONENT < 0x400 defined in lwdump.h to select
 *     for dumping.
 *  size
 *     On entry, this parameter specifies the maximum length for
 *     the returned data. On exit, it specifies the number of bytes
 *     returned.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LWOS_ERROR_ILWALID_ARGUMENT if component is invalid.
 *   LWOS_ERROR_ILWALID_ADDRESS if pBuffer is invalid
 *   LWOS_ERROR_ILWALID_???? if the buffer was too small
 *
 *
 */
#define LW2080_CTRL_CMD_LWD_GET_DUMP (0x20802402) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWD_INTERFACE_ID << 8) | LW2080_CTRL_LWD_GET_DUMP_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_LWD_GET_DUMP_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW2080_CTRL_LWD_GET_DUMP_PARAMS {
    LW_DECLARE_ALIGNED(LwP64 pBuffer, 8);
    LwU32 component;
    LwU32 size;
} LW2080_CTRL_LWD_GET_DUMP_PARAMS;

/*
 * LW2080_CTRL_CMD_LWD_GET_NOCAT_JOURNAL
 *
 * This command returns the contents of the Journal used by  NOCAT, and
 * optionally clears the data
 *
 *   clear:
 *     [IN] indicates if should the data be cleared after reporting
 *
 *   JournalRecords :
 *     [OUT] an array of Journal records reported.
 *
 *   outstandingAssertCount:
 *     [OUT] number of asserts that remain to be reported on.
 *
 *   reportedAssertCount:
 *     [OUT] the number of asserts contained in the report
 *
 *   asserts:
 *     [OUT] an array of up to LW2080_NOCAT_JOURNAL_MAX_ASSERT_RECORDS assert reports
 */


#define LW2080_CTRL_CMD_LWD_GET_NOCAT_JOURNAL    (0x20802409) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWD_INTERFACE_ID << 8) | LW2080_CTRL_LWD_GET_NOCAT_JOURNAL_PARAMS_MESSAGE_ID" */

#define LW2080_NOCAT_JOURNAL_MAX_DIAG_BUFFER     1024
#define LW2080_NOCAT_JOURNAL_MAX_STR_LEN         65
#define LW2080_NOCAT_JOURNAL_MAX_JOURNAL_RECORDS 10
#define LW2080_NOCAT_JOURNAL_MAX_ASSERT_RECORDS  32

// structure to hold clock details.
typedef struct LW2080_NOCAT_JOURNAL_OVERCLOCK_DETAILS {
    LwS32 userMinOffset;
    LwS32 userMaxOffset;
    LwU32 factoryMinOffset;
    LwU32 factoryMaxOffset;
    LwU32 lastActiveClock;
    LwU32 lastActiveVolt;
    LwU32 lastActivePoint;
    LwU32 kappa;
} LW2080_NOCAT_JOURNAL_OVERCLOCK_DETAILS;


// structure to hold clock configuration & state.
typedef struct LW2080_NOCAT_JOURNAL_OVERCLOCK_CFG {
    LwU32                                  pstateVer;
    LW2080_NOCAT_JOURNAL_OVERCLOCK_DETAILS gpcOverclock;
    LW2080_NOCAT_JOURNAL_OVERCLOCK_DETAILS mclkOverclock;
    LwBool                                 bUserOverclocked;
    LwBool                                 bFactoryOverclocked;
} LW2080_NOCAT_JOURNAL_OVERCLOCK_CFG;

// structure to hold the GPU context at the time of the report.
typedef struct LW2080_NOCAT_JOURNAL_GPU_STATE {
    LwBool                             bValid;
    LwU32                              strap;
    LwU16                              deviceId;
    LwU16                              vendorId;
    LwU16                              subsystemVendor;
    LwU16                              subsystemId;
    LwU16                              revision;
    LwU16                              type;
    LwU32                              vbiosVersion;
    LwBool                             bOptimus;
    LwBool                             bMsHybrid;
    LwBool                             bFullPower;
    LwU32                              vbiosOemVersion;
    LwU16                              memoryType;
    LwU8                               tag[LW2080_NOCAT_JOURNAL_MAX_STR_LEN];
    LwU8                               vbiosProject[LW2080_NOCAT_JOURNAL_MAX_STR_LEN];
    LwBool                             bInFullchipReset;
    LwBool                             bInSecBusReset;
    LwBool                             bInGc6Reset;
    LW2080_NOCAT_JOURNAL_OVERCLOCK_CFG overclockCfg;
} LW2080_NOCAT_JOURNAL_GPU_STATE;

#define LW2080_NOCAT_JOURNAL_REC_TYPE_UNKNOWN  0
#define LW2080_NOCAT_JOURNAL_REC_TYPE_BUGCHECK 1
#define LW2080_NOCAT_JOURNAL_REC_TYPE_ENGINE   2
#define LW2080_NOCAT_JOURNAL_REC_TYPE_TDR      3
#define LW2080_NOCAT_JOURNAL_REC_TYPE_RC       4
#define LW2080_NOCAT_JOURNAL_REC_TYPE_ASSERT   5
#define LW2080_NOCAT_JOURNAL_REC_TYPE_ANY      6

// this should be relative to the highest type value
#define LW2080_NOCAT_JOURNAL_REC_TYPE_COUNT    (0x7) /* finn: Evaluated from "LW2080_NOCAT_JOURNAL_REC_TYPE_ANY + 1" */
typedef struct LW2080_NOCAT_JOURNAL_ENTRY {
    LwU8  recType;
    LwU32 bugcheck;
    LwU32 tdrBucketId;
    LwU8  source[LW2080_NOCAT_JOURNAL_MAX_STR_LEN];
    LwU32 subsystem;
    LW_DECLARE_ALIGNED(LwU64 errorCode, 8);
    LwU32 diagBufferLen;
    LwU8  diagBuffer[LW2080_NOCAT_JOURNAL_MAX_DIAG_BUFFER];
    LwU8  faultingEngine[LW2080_NOCAT_JOURNAL_MAX_STR_LEN];
    LwU32 mmuFaultType;
    LwU32 mmuErrorSrc;
    LwU8  tdrReason[LW2080_NOCAT_JOURNAL_MAX_STR_LEN];
} LW2080_NOCAT_JOURNAL_ENTRY;

typedef struct LW2080_NOCAT_JOURNAL_RECORD {
    LwU32                          GPUTag;
    LW_DECLARE_ALIGNED(LwU64 loadAddress, 8);
    LW_DECLARE_ALIGNED(LwU64 timeStamp, 8);
    LW_DECLARE_ALIGNED(LwU64 stateMask, 8);
    LW2080_NOCAT_JOURNAL_GPU_STATE nocatGpuState;
    LW_DECLARE_ALIGNED(LW2080_NOCAT_JOURNAL_ENTRY nocatJournalEntry, 8);
} LW2080_NOCAT_JOURNAL_RECORD;

// NOCAT activity counter indexes
// collection activity
#define LW2080_NOCAT_JOURNAL_REPORT_ACTIVITY_COLLECT_REQ_IDX         0
#define LW2080_NOCAT_JOURNAL_REPORT_ACTIVITY_ALLOCATED_IDX           1
#define LW2080_NOCAT_JOURNAL_REPORT_ACTIVITY_COLLECTED_IDX           2
#define LW2080_NOCAT_JOURNAL_REPORT_ACTIVITY_ALLOC_FAILED_IDX        3
#define LW2080_NOCAT_JOURNAL_REPORT_ACTIVITY_COLLECT_FAILED_IDX      4
#define LW2080_NOCAT_JOURNAL_REPORT_ACTIVITY_COLLECT_REQ_DROPPED_IDX 5

// reporting activity
#define LW2080_NOCAT_JOURNAL_REPORT_ACTIVITY_REQUESTED_IDX           6
#define LW2080_NOCAT_JOURNAL_REPORT_ACTIVITY_REPORTED_IDX            7
#define LW2080_NOCAT_JOURNAL_REPORT_ACTIVITY_DROPPED_IDX             8
#define LW2080_NOCAT_JOURNAL_REPORT_ACTIVITY_MISSED_IDX              9

// update activity
#define LW2080_NOCAT_JOURNAL_REPORT_ACTIVITY_UPDATE_REQ_IDX          10
#define LW2080_NOCAT_JOURNAL_REPORT_ACTIVITY_UPDATED_IDX             11
#define LW2080_NOCAT_JOURNAL_REPORT_ACTIVITY_UPDATE_FAILED_IDX       12

// general errors
#define LW2080_NOCAT_JOURNAL_REPORT_ACTIVITY_BUSY_IDX                13
#define LW2080_NOCAT_JOURNAL_REPORT_ACTIVITY_BAD_PARAM_IDX           14
#define LW2080_NOCAT_JOURNAL_REPORT_ACTIVITY_BAD_TYPE_IDX            15

#define LW2080_NOCAT_JOURNAL_REPORT_ACTIVITY_RES1_IDX                16
#define LW2080_NOCAT_JOURNAL_REPORT_ACTIVITY_RES2_IDX                17
#define LW2080_NOCAT_JOURNAL_REPORT_ACTIVITY_CACHE_UPDATE_IDX        18
#define LW2080_NOCAT_JOURNAL_REPORT_ACTIVITY_INSERT_RECORDS_IDX      19

// this should be relative to the highest counter index
#define LW2080_NOCAT_JOURNAL_REPORT_ACTIVITY_COUNTER_COUNT           (0x14) /* finn: Evaluated from "LW2080_NOCAT_JOURNAL_REPORT_ACTIVITY_INSERT_RECORDS_IDX + 1" */

#define LW2080_CTRL_NOCAT_GET_COUNTERS_ONLY         0:0
#define LW2080_CTRL_NOCAT_GET_COUNTERS_ONLY_YES                      1
#define LW2080_CTRL_NOCAT_GET_COUNTERS_ONLY_NO                       0

#define LW2080_CTRL_NOCAT_GET_RESET_COUNTERS        1:1
#define LW2080_CTRL_NOCAT_GET_RESET_COUNTERS_YES                     1
#define LW2080_CTRL_NOCAT_GET_RESET_COUNTERS_NO                      0


#define LW2080_CTRL_LWD_GET_NOCAT_JOURNAL_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW2080_CTRL_LWD_GET_NOCAT_JOURNAL_PARAMS {
    LwU32 flags;
    LwU32 nocatRecordCount;
    LwU32 nocatOutstandingRecordCount;
    LW_DECLARE_ALIGNED(LW2080_NOCAT_JOURNAL_RECORD journalRecords[LW2080_NOCAT_JOURNAL_MAX_JOURNAL_RECORDS], 8);
    LwU32 activityCounters[LW2080_NOCAT_JOURNAL_REPORT_ACTIVITY_COUNTER_COUNT];
    LwU8  reserved[LW2080_NOCAT_JOURNAL_MAX_STR_LEN];
} LW2080_CTRL_LWD_GET_NOCAT_JOURNAL_PARAMS;

 /*
 * LW2080_CTRL_CMD_LWD_SET_NOCAT_JOURNAL_DATA
 *
 * This command reports the TDR data collected by KMD to be added to the
 * nocat record
 *
 *   dataType:
 *     [IN] specifies the type of data provided.
 *  targetRecordType
 *     [IN] specifies record type the data is intended for.
 *  nocatJournalData
 *     [IN] specifies the data to be added.
 */

#define LW2080_CTRL_CMD_LWD_SET_NOCAT_JOURNAL_DATA        (0x2080240b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWD_INTERFACE_ID << 8) | LW2080_CTRL_LWD_SET_NOCAT_JOURNAL_DATA_PARAMS_MESSAGE_ID" */

// data types & structures
#define LW2080_CTRL_NOCAT_JOURNAL_DATA_TYPE_EMPTY         0
#define LW2080_CTRL_NOCAT_JOURNAL_DATA_TYPE_TDR_REASON    1
#define LW2080_CTRL_NOCAT_JOURNAL_DATA_TYPE_INSERT_RECORD 2
#define LW2080_CTRL_NOCAT_JOURNAL_DATA_TYPE_SET_TAG       3

#define LW2080_CTRL_NOCAT_TDR_TYPE_NONE                   0
#define LW2080_CTRL_NOCAT_TDR_TYPE_LEGACY                 1
#define LW2080_CTRL_NOCAT_TDR_TYPE_FULLCHIP               2
#define LW2080_CTRL_NOCAT_TDR_TYPE_BUSRESET               3
#define LW2080_CTRL_NOCAT_TDR_TYPE_GC6_RESET              4
#define LW2080_CTRL_NOCAT_TDR_TYPE_SURPRISE_REMOVAL       5
#define LW2080_CTRL_NOCAT_TDR_TYPE_UCODE_RESET            6
#define LW2080_CTRL_NOCAT_TDR_TYPE_TEST                   7

typedef struct LW2080CtrlNocatJournalDataTdrReason {
    LwU32 flags;
    LwU8  source[LW2080_NOCAT_JOURNAL_MAX_STR_LEN];
    LwU32 subsystem;
    LW_DECLARE_ALIGNED(LwU64 errorCode, 8);
    LwU32 reasonCode;
} LW2080CtrlNocatJournalDataTdrReason;

#define LW2080_CTRL_NOCAT_INSERT_ALLOW_NULL_STR         0:0
#define LW2080_CTRL_NOCAT_INSERT_ALLOW_NULL_STR_YES     1
#define LW2080_CTRL_NOCAT_INSERT_ALLOW_NULL_STR_NO      0
#define LW2080_CTRL_NOCAT_INSERT_ALLOW_0_LEN_BUFFER     1:1
#define LW2080_CTRL_NOCAT_INSERT_ALLOW_0_LEN_BUFFER_YES 1
#define LW2080_CTRL_NOCAT_INSERT_ALLOW_0_LEN_BUFFER_NO  0
typedef struct LW2080CtrlNocatJournalInsertRecord {
    LwU32 flags;
    LwU8  recType;
    LwU32 bugcheck;
    LwU8  source[LW2080_NOCAT_JOURNAL_MAX_STR_LEN];
    LwU32 subsystem;
    LW_DECLARE_ALIGNED(LwU64 errorCode, 8);
    LwU8  faultingEngine[LW2080_NOCAT_JOURNAL_MAX_STR_LEN];
    LwU32 tdrReason;
} LW2080CtrlNocatJournalInsertRecord;

#define LW2080_CTRL_NOCAT_TAG_CLEAR                 0:0
#define LW2080_CTRL_NOCAT_TAG_CLEAR_YES 1
#define LW2080_CTRL_NOCAT_TAG_CLEAR_NO  0
typedef struct LW2080CtrlNocatJournalSetTag {
    LwU32 flags;
    LwU8  tag[LW2080_NOCAT_JOURNAL_MAX_STR_LEN];
} LW2080CtrlNocatJournalSetTag;

#define LW2080_CTRL_LWD_SET_NOCAT_JOURNAL_DATA_PARAMS_MESSAGE_ID (0xBU)

typedef struct LW2080_CTRL_LWD_SET_NOCAT_JOURNAL_DATA_PARAMS {
    LwU32 dataType;
    LwU32 targetRecordType;
    union {
        LW_DECLARE_ALIGNED(LW2080CtrlNocatJournalDataTdrReason tdrReason, 8);
        LW_DECLARE_ALIGNED(LW2080CtrlNocatJournalInsertRecord insertData, 8);
        LW2080CtrlNocatJournalSetTag tagData;
    } nocatJournalData;
} LW2080_CTRL_LWD_SET_NOCAT_JOURNAL_DATA_PARAMS;
/* _ctr2080lwd_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

