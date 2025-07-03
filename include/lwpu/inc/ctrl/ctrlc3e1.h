/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrlc3e1.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#include "ctrl/ctrla0e1.h"          // C3E1 is partially derived from A0E1
#include "ctrl/ctrlc0e1.h"          // C3E1 is partially derived from C0E1
#define LWC3E1_CTRL_CMD(cat,idx)  \
    LWXXXX_CTRL_CMD(0xC3E1, LWC3E1_CTRL_##cat, idx)

/* LWC3E1 command categories (6bits) */
#define LWC3E1_CTRL_RESERVED (0x00)
#define LWC3E1_CTRL_FB       (0x01)

/*
 * LWC3E1_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LWC3E1_CTRL_CMD_NULL (0xc3e10000) /* finn: Evaluated from "(FINN_GV100_SUBDEVICE_FB_RESERVED_INTERFACE_ID << 8) | 0x0" */





#define LWC3E1_CTRL_CMD_FB_GET_ECC_COUNTS     (0xc3e10101) /* finn: Evaluated from "(FINN_GV100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | LWC3E1_CTRL_FB_GET_ECC_COUNTS_PARAMS_MESSAGE_ID" */

#define LWC3E1_CTRL_FB_ECC_PARTITION_COUNT    (0x00000010)
#define LWC3E1_CTRL_FB_ECC_SUBPARTITION_COUNT (0x00000002)
#define LWC3E1_CTRL_FB_ECC_SLICE_COUNT        (0x00000004)

/*
 * LWC3E1_CTRL_FB_QUERY_ECC_EXCEPTION_STATUS
 *
 * This structure represents the exception status of a class of per-unit
 * exceptions
 *
 *   count
 *     This parameter contains the number of exceptions that have oclwrred 
 *     since boot.
 *
 */
typedef LWC0E1_CTRL_FB_QUERY_ECC_EXCEPTION_STATUS LWC3E1_CTRL_FB_QUERY_ECC_EXCEPTION_STATUS;

typedef struct LWC3E1_CTRL_FB_GET_ECC_SLICE_COUNTS {

    LW_DECLARE_ALIGNED(LWC3E1_CTRL_FB_QUERY_ECC_EXCEPTION_STATUS dbe, 8);
    LW_DECLARE_ALIGNED(LWC3E1_CTRL_FB_QUERY_ECC_EXCEPTION_STATUS sbe, 8);
} LWC3E1_CTRL_FB_GET_ECC_SLICE_COUNTS;

typedef LWC3E1_CTRL_FB_GET_ECC_SLICE_COUNTS LWC3E1_CTRL_FB_GET_ECC_SUBPARTITION_COUNTS;

/*
 * LWC3E1_CTRL_FB_GET_ECC_PARTITION_COUNTS
 *
 * This structure contains the number of sbe and dbe exceptions that have oclwrred
 * in a subPartition/Slice.
 *
 *      ltc*
 *          Holds error counts for the L2 cache Data memory.
 *
 *      fb*
 *          Holds error counts for the frame buffer.
 *
 */

typedef struct LWC3E1_CTRL_FB_GET_ECC_PARTITION_COUNTS {

    LW_DECLARE_ALIGNED(LWC3E1_CTRL_FB_GET_ECC_SLICE_COUNTS ltc[LWC3E1_CTRL_FB_ECC_SLICE_COUNT], 8);
    LW_DECLARE_ALIGNED(LWC3E1_CTRL_FB_GET_ECC_SUBPARTITION_COUNTS fb[LWC3E1_CTRL_FB_ECC_SUBPARTITION_COUNT], 8);
} LWC3E1_CTRL_FB_GET_ECC_PARTITION_COUNTS;

/*
 * LWC3E1_CTRL_FB_GET_ECC_COUNTS_PARAMS
 *
 * This structure contains the number of sbe and dbe exceptions that have oclwrred
 * in a partition.
 *
 *      sliceCount[out]
 *         Holds the number of valid entries in the slice array.
 *
 *      subPartitionCount[out]
 *         Holds the number of valid entries in the subPartition array.
 *
 *      partitionCount[out]
 *         Holds the number of valid entries in the partition array.
 *
 *      partitionEcc[out]
 *         Refer LWC3E1_CTRL_FB_GET_ECC_PARTITION_COUNTS.
 *
 *      flags[in/out]
 *          Allow users to request filtered or raw counts.
 *
 */

#define LWC3E1_CTRL_FB_GET_ECC_COUNTS_FLAGS_TYPE            LWC0E1_CTRL_FB_GET_ECC_COUNTS_FLAGS_TYPE
#define LWC3E1_CTRL_FB_GET_ECC_COUNTS_FLAGS_TYPE_FILTERED  LWC0E1_CTRL_FB_GET_ECC_COUNTS_FLAGS_TYPE_FILTERED
#define LWC3E1_CTRL_FB_GET_ECC_COUNTS_FLAGS_TYPE_RAW       LWC0E1_CTRL_FB_GET_ECC_COUNTS_FLAGS_TYPE_RAW

#define LWC3E1_CTRL_FB_GET_ECC_COUNTS_FLAGS_OVERFLOW         LWC0E1_CTRL_FB_GET_ECC_COUNTS_FLAGS_OVERFLOW
#define LWC3E1_CTRL_FB_GET_ECC_COUNTS_FLAGS_OVERFLOW_FALSE LWC0E1_CTRL_FB_GET_ECC_COUNTS_FLAGS_OVERFLOW_FALSE
#define LWC3E1_CTRL_FB_GET_ECC_COUNTS_FLAGS_OVERFLOW_TRUE  LWC0E1_CTRL_FB_GET_ECC_COUNTS_FLAGS_OVERFLOW_TRUE

#define LWC3E1_CTRL_FB_GET_ECC_COUNTS_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWC3E1_CTRL_FB_GET_ECC_COUNTS_PARAMS {
    LwU32 sliceCount;
    LwU32 subPartitionCount;
    LwU32 partitionCount;
    LwU32 flags;

    LW_DECLARE_ALIGNED(LWC3E1_CTRL_FB_GET_ECC_PARTITION_COUNTS partitionEcc[LWC3E1_CTRL_FB_ECC_PARTITION_COUNT], 8);
} LWC3E1_CTRL_FB_GET_ECC_COUNTS_PARAMS;

/*
 * LWC3E1_CTRL_CMD_FB_GET_LATEST_ECC_ADDRESSES
 *
 * Refer LWA0E1_CTRL_CMD_FB_GET_LATEST_ECC_ADDRESSES documentation
 *
 */

#define LWC3E1_CTRL_CMD_FB_GET_LATEST_ECC_ADDRESSES            (0xc3e10102) /* finn: Evaluated from "(FINN_GV100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | 0x2" */

#define LWC3E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_MAX_COUNT      LWA0E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_MAX_COUNT

#define LWC3E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_ERROR_TYPE LWA0E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_ERROR_TYPE
#define LWC3E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_ERROR_TYPE_SBE LWA0E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_ERROR_TYPE_SBE
#define LWC3E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_ERROR_TYPE_DBE LWA0E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_ERROR_TYPE_DBE

/*
 * LWC3E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_ADDRESS
 *
 * Refer LWA0E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_ADDRESS documentation
 *
 */
typedef LWA0E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_ADDRESS LWC3E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_ADDRESS;

typedef LWA0E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_PARAMS LWC3E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LWC3E1_CTRL_CMD_FB_GET_LATEST_ECC_ADDRESSES_FINN_PARAMS_MESSAGE_ID (0x2U)

typedef struct LWC3E1_CTRL_CMD_FB_GET_LATEST_ECC_ADDRESSES_FINN_PARAMS {
    LW_DECLARE_ALIGNED(LWC3E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_PARAMS params, 8);
} LWC3E1_CTRL_CMD_FB_GET_LATEST_ECC_ADDRESSES_FINN_PARAMS;



/* _ctrlc3e1_h_ */
