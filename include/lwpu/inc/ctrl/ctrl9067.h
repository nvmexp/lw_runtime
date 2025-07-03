/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2015 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl9067.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* Subcontext control commands and parameters */
#define LW9067_CTRL_CMD(cat,idx)          LWXXXX_CTRL_CMD(0x9067, LW9067_CTRL_##cat, idx)

/* Command categories (6bits) */
#define LW9067_CTRL_RESERVED      (0x00)
#define LW9067_CTRL_TPC_PARTITION (0x01)
#define LW9067_CTRL_CWD_WATERMARK (0x02)

/*!
 * Does nothing.
 */
#define LW9067_CTRL_CMD_NULL      (0x90670000) /* finn: Evaluated from "(FINN_FERMI_CONTEXT_SHARE_A_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*!
 * LW9067_CTRL_CMD_GET_TPC_PARTITION_TABLE
 *    This command gets the current partition table configuration of a subcontext
 *
 * LW9067_CTRL_CMD_SET_TPC_PARTITION_TABLE
 *    This command sets the partition table of a subcontext
 *
 * LW9067_CTRL_TPC_PARTITION_TABLE_PARAMS
 *     This structure defines the parameters used for SET/GET per-subcontext TPC partitioning table configuration
 *
 *       numUsedTpc [in/out]
 *           Specifies the number of TPCs used by the subcontext
 *           While querying the enabled TPCs, this is an output paramter
 *           While configuring the TPCs, this is an input parameter
 *
 *       tpcList [in/out]
 *           Array containing the TPCs enabled for the subcontext.
 *           The first numUsedTpc in the array interpreted as the valid entries.
 *
 *           Only applicable for STATIC and DYNAMIC modes.
 *
 * LW9067_CTRL_TPC_PARTITION_TABLE_MAX_TPC_COUNT
 *     Max TPC count supported by this ctrl call
 *     
 * LW9067_CTRL_TPC_PARTITION_TABLE_TPC_INFO
 *   This structure defines the parameters for a TPC
 *
 *       globalTpcIndex
 *          Global logical index of the enabled TPC
 *
 *       lmemBlockIndex
 *          Block index of the Local memory backing store for the enabled TPC.
 *          For GET command, we will return the current lmem block assigment for STATIC & DYNAMIC modes.
 *          For SET command, this index is relevant only for STATIC mode.
 *          HW automatically assign it for other modes. So should be zeroed out for other modes.
 *
 */
#define LW9067_CTRL_CMD_GET_TPC_PARTITION_TABLE       (0x90670101) /* finn: Evaluated from "(FINN_FERMI_CONTEXT_SHARE_A_TPC_PARTITION_INTERFACE_ID << 8) | 0x1" */

#define LW9067_CTRL_CMD_SET_TPC_PARTITION_TABLE       (0x90670102) /* finn: Evaluated from "(FINN_FERMI_CONTEXT_SHARE_A_TPC_PARTITION_INTERFACE_ID << 8) | LW9067_CTRL_TPC_PARTITION_TABLE_PARAMS_MESSAGE_ID" */

#define LW9067_CTRL_TPC_PARTITION_TABLE_TPC_COUNT_MAX 256

typedef struct LW9067_CTRL_TPC_PARTITION_TABLE_TPC_INFO {
    LwU16 globalTpcIndex;
    LwU16 lmemBlockIndex;
} LW9067_CTRL_TPC_PARTITION_TABLE_TPC_INFO;

#define LW9067_CTRL_TPC_PARTITION_TABLE_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW9067_CTRL_TPC_PARTITION_TABLE_PARAMS {
    LwU16                                    numUsedTpc;
    LW9067_CTRL_TPC_PARTITION_TABLE_TPC_INFO tpcList[LW9067_CTRL_TPC_PARTITION_TABLE_TPC_COUNT_MAX];
} LW9067_CTRL_TPC_PARTITION_TABLE_PARAMS;


/*!
 * LW9067_CTRL_CMD_GET_CWD_WATERMARK
 *    This command gets the cached watermark value for a subcontext
 *
 * LW9067_CTRL_CMD_SET_CWD_WATERMARK
 *    This command sets the watermark value for a subcontexts
 *
 * LW9067_CTRL_CWD_WATERMARK_PARAMS
 *     This structure defines the parameters used to SET/GET watermark value per-subcontext.
 *
 *        watermarkValue [in/out]
 *            Value of watermark per-subcontext
 *            Acts as a output parameter to get the current value of watermark for a subcontext.
 *            Acts as a input parameter to set the current value of watermark for a subcontexts.
 *
 * LW9067_CTRL_CWD_WATERMARK_VALUE_MIN
 *     Minimum value of watermark for any subcontext
 *     RM will throw an error if any value less than this value is specified
 *
 * LW9067_CTRL_CWD_WATERMARK_VALUE_DEFAULT
 *     Default value of watermark for any subcontext
 *     RM will set watermark of a subcontext to this value when the subcontext
 *     is created/initialized for the first time
 *
 * LW9067_CTRL_CWD_WATERMARK_VALUE_MAX
 *     Maximum value of watermark for any subcontext
 *     RM will throw an error if any value more than this value is specified
 *
 */



#define LW9067_CTRL_CMD_GET_CWD_WATERMARK       (0x90670201) /* finn: Evaluated from "(FINN_FERMI_CONTEXT_SHARE_A_CWD_WATERMARK_INTERFACE_ID << 8) | 0x1" */

#define LW9067_CTRL_CMD_SET_CWD_WATERMARK       (0x90670202) /* finn: Evaluated from "(FINN_FERMI_CONTEXT_SHARE_A_CWD_WATERMARK_INTERFACE_ID << 8) | 0x2" */

#define LW9067_CTRL_CWD_WATERMARK_VALUE_MIN     1
#define LW9067_CTRL_CWD_WATERMARK_VALUE_DEFAULT 2
#define LW9067_CTRL_CWD_WATERMARK_VALUE_MAX     256

typedef struct LW9067_CTRL_CWD_WATERMARK_PARAMS {
    LwU32 watermarkValue;
} LW9067_CTRL_CWD_WATERMARK_PARAMS;


/* _ctrl9067_h_ */
