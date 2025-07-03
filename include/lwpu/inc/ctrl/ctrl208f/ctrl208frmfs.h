/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl208f/ctrl208frmfs.finn
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
 * These control calls are for LWPU internal use only.  They define the queue
 * used for streaming disk files from the kernel mode resman driver.
 */

/*
 * Resman File Streaming Operations
 *
 * This opcode defines what operation each queue subrecord performs.
 */
#define LW208F_CTRL_RMFS_OP_NEXT_REC    0x00   /* end of subrecord                  */
#define LW208F_CTRL_RMFS_OP_OPEN        0x01   /* fopen, fopen_s                    */
#define LW208F_CTRL_RMFS_OP_CLOSE       0x02   /* fclose                            */
#define LW208F_CTRL_RMFS_OP_WRITE       0x03   /* fputc, fputs, fwrite              */
#define LW208F_CTRL_RMFS_OP_FLUSH       0x04   /* fflush, _flushall                 */
#define LW208F_CTRL_RMFS_OP_SEEK        0x05   /* fseek, _fseeki64, fsetpos, rewind */
#define LW208F_CTRL_RMFS_OP_CLOSE_QUEUE 0xff   /* Client side of queue is done      */

/*
 * Resman File Streaming Predefined file IDs
 *
 * The fileId for disk files are assigned incrfemntally during client-side
 * fopen, starting with _DISK.
 *
 * _ALL, _STDOUT, and _STDERR are special predefined file IDs that can be used
 * without opening.
 */
#define LW208F_CTRL_RMFS_FILEID_ALL     0   /* Applies to all open files.  */
#define LW208F_CTRL_RMFS_FILEID_STDOUT  1   /* stdout                      */
#define LW208F_CTRL_RMFS_FILEID_STDERR  2   /* stderr                      */
#define LW208F_CTRL_RMFS_FILEID_DISK    3   /* First disk file ID          */

/*
 * Resman File Streaming Command queue sub-record definition
 *
 * Contents of param vary according to operation, as follows:
 *
 * operation    param[]
 * _NEXT_REC    not used
 * _OPEN        0 terminated strings for filename and mode
 * _CLOSE       not used
 * _WRITE       data.  (data length is derived from size)
 * _FLUSH       not used
 * _SEEK        LwU8 origin, LwS64 offset
 * _STOP        not used
 */
typedef struct LW208F_CTRL_RMFS_CMD_QUEUE_SUBREC {
    LwU32 fileId;       /* Unique file ID.    LW208F_CTRL_RMFS_FILEID_<xyz>   */
    LwU16 size;         /* Total size of subrecord, including all parameters. */
    LwU8  operation;    /* LW208F_CTRL_RMFS_OP_<xyz>                          */
    LwU8  param[1];     /* Generic parameters.                                */
} LW208F_CTRL_RMFS_CMD_QUEUE_SUBREC;

/*
 * Resman File Streaming Command Queue Parameters
 */
#define LW208F_CTRL_RMFS_CMD_QUEUE_SIZE             0x400000    /* 4MB (1024 pages)             */
#define LW208F_CTRL_RMFS_CMD_QUEUE_ELEMENT_SIZE     4096
#define LW208F_CTRL_RMFS_CMD_QUEUE_ELEMENT_ALIGN    12    /* 2 ^ 12 = 4096                */
#define LW208F_CTRL_RMFS_CMD_QUEUE_HDR_ALIGN        4    /* 2 ^  4 = 16  (DMA alignment) */
#define LW208F_CTRL_RMFS_CMD_QUEUE_SUBREC_ALIGN     3    /* 2 ^  3 = 8                   */

/*
 * Resman File Streaming Status Queue Parameters
 */
#define LW208F_CTRL_RMFS_STATUS_QUEUE_SIZE          0x01000    /* 4KB (1 page) */
#define LW208F_CTRL_RMFS_STATUS_QUEUE_ELEMENT_SIZE  16
#define LW208F_CTRL_RMFS_STATUS_QUEUE_ELEMENT_ALIGN 4    /* 2 ^ 4 = 16   */
#define LW208F_CTRL_RMFS_STATUS_QUEUE_HDR_ALIGN     4    /* 2 ^ 4 = 16   */

/*
 * Client location
 *
 * Location of client queue endpoint.  Are files being streamed from the resman
 * driver on the CPU?  Or from the GSP ucode?
 */
#define LW208F_CTRL_RMFS_CLIENT_LOCATION_CPU        0  /* Resman driver on the CPU */
#define LW208F_CTRL_RMFS_CLIENT_LOCATION_GSP        1  /* Resman ucode on the GSP  */

/*
 * Resman File Streaming Init
 *
 * Sets up client (driver) side queue.
 */
#define LW208F_CTRL_CMD_RMFS_INIT                   (0x208f1600) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_RMFS_INTERFACE_ID << 8) | LW208F_CTRL_RMFS_INIT_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_RMFS_INIT_PARAMS_MESSAGE_ID (0x0U)

typedef struct LW208F_CTRL_RMFS_INIT_PARAMS {
    LwU32    dwClientLocation;                   /* [in]  LW208F_CTRL_RMFS_CLIENT_LOCATION_<xyz>  */
    LwHandle hCmdQueueMemory;                    /* [in]  Memory handle for command queue         */
    LwHandle hStatusQueueMemory;                 /* [in]  Memory handle for status queue          */
} LW208F_CTRL_RMFS_INIT_PARAMS;

/*
 * Resman File Streaming Close Queue
 *
 * Client side submits one final LW208F_CTRL_RMFS_OP_CLOSE_QUEUE subrecord and
 * then flushes any pending records.
 *
 * The client side is guaranteed not to touch the queue after this call.
 */
#define LW208F_CTRL_CMD_RMFS_CLOSE_QUEUE (0x208f1601) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_RMFS_INTERFACE_ID << 8) | LW208F_CTRL_RMFS_CLOSE_QUEUE_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_RMFS_CLOSE_QUEUE_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW208F_CTRL_RMFS_CLOSE_QUEUE_PARAMS {
    LwU32 dwClientLocation;                   /* [in]  LW208F_CTRL_RMFS_CLIENT_LOCATION_<xyz>  */
} LW208F_CTRL_RMFS_CLOSE_QUEUE_PARAMS;

/*
 * Resman File Streaming Cleanup
 *
 * Resman sets the client to the same state as it was before init.
 */
#define LW208F_CTRL_CMD_RMFS_CLEANUP (0x208f1602) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_RMFS_INTERFACE_ID << 8) | LW208F_CTRL_RMFS_CLEANUP_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_RMFS_CLEANUP_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW208F_CTRL_RMFS_CLEANUP_PARAMS {
    LwU32 dwClientLocation;                   /* [in]  LW208F_CTRL_RMFS_CLIENT_LOCATION_<xyz>  */
} LW208F_CTRL_RMFS_CLEANUP_PARAMS;

/*
 * Resman File Streaming Test
 *
 * Runs a file test on the client side (in CPU-RM or GSP-RM).
 */
#define LW208F_CTRL_CMD_RMFS_TEST               (0x208f1603) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_RMFS_INTERFACE_ID << 8) | LW208F_CTRL_RMFS_TEST_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_RMFS_TEST_FLAGS_FLUSH       0x00000001 /* Flush the queue after test. */
#define LW208F_CTRL_RMFS_TEST_FLAGS_CLOSE_QUEUE 0x00000002 /* Close the queue after test. */

#define LW208F_CTRL_RMFS_TEST_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW208F_CTRL_RMFS_TEST_PARAMS {
    LwU32 dwClientLocation;                   /* [in]  LW208F_CTRL_RMFS_CLIENT_LOCATION_<xyz>  */
    LwU32 dwReps;                             /* [in]  Number of loops                         */
    LwU32 flags;                              /* [in]  LW208F_CTRL_RMFS_TEST_FLAGS_<xyz>       */
    LwU32 dwTestData1;                        /* [in]  Reserved                                */
    LwU32 dwTestData2;                        /* [in]  Reserved                                */
} LW208F_CTRL_RMFS_TEST_PARAMS;

/* _ctrl208frmfs_h_ */
