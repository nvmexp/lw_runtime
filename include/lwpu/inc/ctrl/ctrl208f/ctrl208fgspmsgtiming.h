/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl208f/ctrl208fgspmsgtiming.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "lwmisc.h"
#include "ctrl/ctrl208f/ctrl208fbase.h"

/*
 * Resman GSP Message Timing
 */

/*
 * Value set index.
 */
#define LW208F_CTRL_GSP_MT_VALUE_INDEX_START 0U
#define LW208F_CTRL_GSP_MT_VALUE_INDEX_END   1U
#define LW208F_CTRL_GSP_MT_VALUE_INDEX_USER0 2U
#define LW208F_CTRL_GSP_MT_VALUE_INDEX_USER1 3U
/*
 * LW208F_CTRL_GSP_MT_VALUE_INDEX_NUM can be increased to add user defined
 * timing value sets, but LW208F_CTRL_GSP_MT_BUFFER must stay below 4K.
 */
#define LW208F_CTRL_GSP_MT_VALUE_INDEX_NUM   4U

/*
 * State variable - used to check internal integrity.
 */
#define LW208F_CTRL_GSP_MT_STATE_EMPTY       0U
#define LW208F_CTRL_GSP_MT_STATE_CPU         1U
#define LW208F_CTRL_GSP_MT_STATE_GSP_START   2U
#define LW208F_CTRL_GSP_MT_STATE_GSP_ADDINFO 3U
#define LW208F_CTRL_GSP_MT_STATE_GSP_DONE    4U

typedef struct LW208F_CTRL_GSP_MT_HEADER {
    LW_DECLARE_ALIGNED(LwU64 seqNum, 8);
    LwU32 hClient;
    LwU32 hObject;
    LwU32 cntrlCmd;
    LwU32 state;
} LW208F_CTRL_GSP_MT_HEADER;

typedef struct LW208F_CTRL_GSP_MT_VALUES {
    LW_DECLARE_ALIGNED(LwU64 timeNs, 8);
    LW_DECLARE_ALIGNED(LwU64 cycle, 8);
    LW_DECLARE_ALIGNED(LwU64 instRet, 8);
    LW_DECLARE_ALIGNED(LwU64 ioctl, 8);
    LW_DECLARE_ALIGNED(LwU64 ioRead, 8);
    LW_DECLARE_ALIGNED(LwU64 ioWrite, 8);
    LW_DECLARE_ALIGNED(LwU64 hubRead, 8);
    LW_DECLARE_ALIGNED(LwU64 hubWrite, 8);
    LW_DECLARE_ALIGNED(LwU64 icacheMiss, 8);
    LW_DECLARE_ALIGNED(LwU64 dcacheMiss, 8);
    LW_DECLARE_ALIGNED(LwU64 branch, 8);
    LW_DECLARE_ALIGNED(LwU64 branchMiss, 8);
    LW_DECLARE_ALIGNED(LwU64 extraInfo, 8);
} LW208F_CTRL_GSP_MT_VALUES;

#define LW208F_CTRL_GSP_MT_MEM_SIZE_UNALIGNED                                  \
   (sizeof(LW208F_CTRL_GSP_MT_HEADER) +                                        \
   (LW208F_CTRL_GSP_MT_VALUE_INDEX_NUM * sizeof(LW208F_CTRL_GSP_MT_VALUES)))

#define LW208F_CTRL_GSP_MT_MEM_SIZE_COPY                                       \
    LW_ALIGN_UP(LW208F_CTRL_GSP_MT_MEM_SIZE_UNALIGNED, 256U)

#define LW208F_CTRL_GSP_MT_MEM_SIZE_ALLOC 4096

typedef struct LW208F_CTRL_GSP_MT_BUFFER {
    LW_DECLARE_ALIGNED(LW208F_CTRL_GSP_MT_HEADER hdr, 8);
    LW_DECLARE_ALIGNED(LW208F_CTRL_GSP_MT_VALUES values[LW208F_CTRL_GSP_MT_VALUE_INDEX_NUM], 8);
} LW208F_CTRL_GSP_MT_BUFFER;

/*
 * Resman GSP Message Timing Init
 */
#define LW208F_CTRL_CMD_GSP_MT_INIT (0x208f1700) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_GSPMSGTIMING_INTERFACE_ID << 8) | LW208F_CTRL_GSP_MT_INIT_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_GSP_MT_INIT_PARAMS_MESSAGE_ID (0x0U)

typedef struct LW208F_CTRL_GSP_MT_INIT_PARAMS {
    LwHandle hMemory;                    /* [in, kern] Memory handle to shared buffer area */
    LW_DECLARE_ALIGNED(LwU64 physAddr, 8); /* [in, phys] Phys address of shared buffer area  */
} LW208F_CTRL_GSP_MT_INIT_PARAMS;

/*
 * Resman GSP Message Timing Cleanup
 */
#define LW208F_CTRL_CMD_GSP_MT_CLEANUP (0x208f1702) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_GSPMSGTIMING_INTERFACE_ID << 8) | LW208F_CTRL_GSP_MT_CLEANUP_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_GSP_MT_CLEANUP_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW208F_CTRL_GSP_MT_CLEANUP_PARAMS {
    LwU32 not_used;   // Structure cannot be empty.
} LW208F_CTRL_GSP_MT_CLEANUP_PARAMS;


/* ctrl208fgspmsgtiming_h */
