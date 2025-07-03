/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2010-2015 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl90e1.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#define LW90E1_CTRL_CMD(cat,idx)  \
    LWXXXX_CTRL_CMD(0x90E1, LW90E1_CTRL_##cat, idx)


/* LW90E1 command categories (6bits) */
#define LW90E1_CTRL_RESERVED (0x00)
#define LW90E1_CTRL_FB       (0x01)


/*
 * LW90E1_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LW90E1_CTRL_CMD_NULL (0x90e10000) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_FB_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*
 * LW90E1_CTRL_CMD_FB_GET_ECC_COUNTS
 *
 * This command is used to query detailed ECC counts for each of the
 * units in FB, it covers LTC (level 2 cache) and FB memory errors
 *
 *   sliceCount (in/out)
 *     Input:  this parameter specifies how many slices per partition the
 *             passed in structure can support.
 *     Output: this parameter specifies how many slices of data
 *             was stored to the structure
 *
 *   partitionCount (in/out)
 *     Input:  this parameter specifies how many partitions the
 *             passed in structure can support.
 *     Output: this parameter specifies how many partitions of data
 *             was stored to the structure
 *
 *   partitionEcc
 *     This parameter stores the FB and LTC ECC single- and double-bit
 *     error counts.
 *
 *   status
 *     This parameter returns whether or not the passed in structure was big
 *     enough to store the counts. Bit-encoded overflow values:
 *
 *          LW2080_CTRL_GET_ECC_STATUS_SLICE_OVERFLOW             (0x00000001)
 *          LW2080_CTRL_GET_ECC_STATUS_PARTITION_OVERFLOW         (0x00000002)
 *
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 *
 */
#define LW90E1_CTRL_CMD_FB_GET_ECC_COUNTS                 (0x90e10101) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | LW90E1_CTRL_FB_GET_ECC_COUNTS_PARAMS_MESSAGE_ID" */

#define LW90E1_CTRL_FB_ECC_PARTITION_COUNT                (0x00000016)
#define LW90E1_CTRL_FB_ECC_SLICE_COUNT                    (0x00000004)

/*
 * Due to limitations of early ECC hardware, the RM may need to limit
 * the number of errors reported; e.g. it may be forced to report
 * no more than a single double-bit error, or to omit reporting of
 * single-bit errors completely.
 *
 * For RM some clients, such as MODS, this may not be sufficient. In
 * those cases, the RM can be instructed to return the errors as
 * they are obtained from the hardware itself, unfiltered.
 */
#define LW90E1_CTRL_FB_GET_ECC_COUNTS_FLAGS_TYPE             0:0
#define LW90E1_CTRL_FB_GET_ECC_COUNTS_FLAGS_TYPE_FILTERED (0x00000000)
#define LW90E1_CTRL_FB_GET_ECC_COUNTS_FLAGS_TYPE_RAW      (0x00000001)

/*
 * LW90E1_CTRL_FB_QUERY_ECC_EXCEPTION_STATUS
 *
 * This structure represents the exception status of a class of per-unit
 * exceptions
 *
 *   count
 *     number of exceptions that have oclwrred since boot
 */
typedef struct LW90E1_CTRL_FB_QUERY_ECC_EXCEPTION_STATUS {
    LW_DECLARE_ALIGNED(LwU64 count, 8);
} LW90E1_CTRL_FB_QUERY_ECC_EXCEPTION_STATUS;


typedef struct LW90E1_CTRL_FB_GET_ECC_SLICE_COUNTS {
    LW_DECLARE_ALIGNED(LW90E1_CTRL_FB_QUERY_ECC_EXCEPTION_STATUS ltcSbe, 8);
    LW_DECLARE_ALIGNED(LW90E1_CTRL_FB_QUERY_ECC_EXCEPTION_STATUS ltcDbe, 8);
    LW_DECLARE_ALIGNED(LW90E1_CTRL_FB_QUERY_ECC_EXCEPTION_STATUS fbSbe, 8);
    LW_DECLARE_ALIGNED(LW90E1_CTRL_FB_QUERY_ECC_EXCEPTION_STATUS fbDbe, 8);
} LW90E1_CTRL_FB_GET_ECC_SLICE_COUNTS;

typedef LW90E1_CTRL_FB_GET_ECC_SLICE_COUNTS LW90E1_CTRL_FB_GET_ECC_PARTITION_COUNTS[LW90E1_CTRL_FB_ECC_SLICE_COUNT];


#define LW90E1_CTRL_FB_GET_ECC_COUNTS_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW90E1_CTRL_FB_GET_ECC_COUNTS_PARAMS {
    LwU32 sliceCount;
    LwU32 partitionCount;
    LwU32 flags;

    LW_DECLARE_ALIGNED(LW90E1_CTRL_FB_GET_ECC_PARTITION_COUNTS partitionEcc[LW90E1_CTRL_FB_ECC_PARTITION_COUNT], 8);
} LW90E1_CTRL_FB_GET_ECC_COUNTS_PARAMS;

/*
 * LW90E1_CTRL_CMD_FB_GET_EDC_COUNTS
 *
 * This command is used to query detailed EDC counts for each of the
 * partitions on FB.
 *
 *   edcCounts
 *     This output parameter stores the EDC error counts per partition.
 *     Floor-swept partitions are compacted and extra partition data is set to 0.
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 *
 */
#define LW90E1_CTRL_CMD_FB_GET_EDC_COUNTS (0x90e10102) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | LW90E1_CTRL_FB_GET_EDC_COUNTS_PARAMS_MESSAGE_ID" */

#define LW90E1_CTRL_FB_GET_EDC_COUNTS_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW90E1_CTRL_FB_GET_EDC_COUNTS_PARAMS {
    LW_DECLARE_ALIGNED(LW90E1_CTRL_FB_QUERY_ECC_EXCEPTION_STATUS edcCounts[LW90E1_CTRL_FB_ECC_PARTITION_COUNT], 8);
} LW90E1_CTRL_FB_GET_EDC_COUNTS_PARAMS;

/*
 * LW90E1_CTRL_CMD_FB_GET_SBE_ADDRESSES
 *
 * This command is used to query count and value of addresses which have
 * generated at least one single-bit ECC error for a given partition/
 * subpartition. Note: this shares the LW90E1_CTRL_FB_GET_ECC_ADDRESSES_PARAMS
 * with LW90E1_CTRL_CMD_FB_GET_SBE_ADDRESSES.
 *    Clients that want double-bit errors should refer to the
 * LW90E1_CTRL_CMD_FB_GET_DBE_ADDRESSES command
 *
 *   partition (in)
 *        requested partition
 *
 *   subpartition (in)
 *        requested subpartition
 *
 *   addressCount (out)
 *        number of addresses being reported.
 *
 *   addresses (out)
 *        The actual Row/Bank/Column addresses.
 *
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 *
 */

/* Note: LW90E1_CTRL_CMD_FB_GET_ECC_ADDRESSES is deprecated. All new code should
         use LW90E1_CTRL_CMD_FB_GET_SBE_ADDRESSES and existing code should be
         colwerted to use LW90E1_CTRL_CMD_FB_GET_SBE_ADDRESSES */
#define LW90E1_CTRL_CMD_FB_GET_SBE_ADDRESSES                   (0x90e10103) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | 0x3" */

#define LW90E1_CTRL_CMD_FB_GET_ECC_ADDRESSES                   (0x90e10103) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | 0x3" */

#define LW90E1_CTRL_CMD_FB_GET_ECC_ADDRESSES_MAX_NUM_ADDRESSES (16)

typedef struct LW90E1_CTRL_FB_GET_ECC_ADDRESSES_PARAMS {
    LwU32 partition;
    LwU32 subpartition;

    LwU32 addressCount;
    LwU32 addresses[LW90E1_CTRL_CMD_FB_GET_ECC_ADDRESSES_MAX_NUM_ADDRESSES];
} LW90E1_CTRL_FB_GET_ECC_ADDRESSES_PARAMS;

typedef LW90E1_CTRL_FB_GET_ECC_ADDRESSES_PARAMS LW90E1_CTRL_FB_GET_SBE_ADDRESSES_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LW90E1_CTRL_CMD_FB_GET_SBE_ADDRESSES_FINN_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW90E1_CTRL_CMD_FB_GET_SBE_ADDRESSES_FINN_PARAMS {
    LW90E1_CTRL_FB_GET_SBE_ADDRESSES_PARAMS params;
} LW90E1_CTRL_CMD_FB_GET_SBE_ADDRESSES_FINN_PARAMS;



/*
 * LW90E1_CTRL_CMD_FB_GET_DBE_ADDRESSES
 *
 * This command is used to query count and value of addresses which have
 * generated at least one single-bit ECC error for a given partition/
 * subpartition. Note: this shares the LW90E1_CTRL_FB_GET_ECC_ADDRESSES_PARAMS
 * with LW90E1_CTRL_CMD_FB_GET_SBE_ADDRESSES
 *    Clients that want single-bit errors should refer to the
 * LW90E1_CTRL_CMD_FB_GET_SBE_ADDRESSES command
 *
 *   partition (in)
 *        requested partition
 *
 *   subpartition (in)
 *        requested subpartition
 *
 *   addressCount (out)
 *        number of addresses being reported.
 *
 *   addresses (out)
 *        The actual Row/Bank/Column addresses.
 *
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 *
 */
#define LW90E1_CTRL_CMD_FB_GET_DBE_ADDRESSES (0x90e10104) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | 0x4" */

typedef LW90E1_CTRL_FB_GET_ECC_ADDRESSES_PARAMS LW90E1_CTRL_FB_GET_DBE_ADDRESSES_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LW90E1_CTRL_CMD_FB_GET_DBE_ADDRESSES_FINN_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW90E1_CTRL_CMD_FB_GET_DBE_ADDRESSES_FINN_PARAMS {
    LW90E1_CTRL_FB_GET_DBE_ADDRESSES_PARAMS params;
} LW90E1_CTRL_CMD_FB_GET_DBE_ADDRESSES_FINN_PARAMS;



/*
 * LW90E1_CTRL_CMD_FB_GET_AGGREGATE_ECC_COUNTS
 *
 * This command is used to query detailed ECC counts for each of the
 * units in FB for the lifetime of a given GPU, to the extent the RM
 * can determine these values. This covers LTC (level 2 cache) and FB
 * memory errors
 *
 *   partitionEcc
 *     This parameter stores the FB and LTC ECC single- and double-bit
 *     error counts.
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_NOT_SUPPORTED
 *
 */
#define LW90E1_CTRL_CMD_FB_GET_AGGREGATE_ECC_COUNTS (0x90e10105) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | LW90E1_CTRL_FB_GET_AGGREGATE_ECC_COUNTS_PARAMS_MESSAGE_ID" */

#define LW90E1_CTRL_FB_GET_AGGREGATE_ECC_COUNTS_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW90E1_CTRL_FB_GET_AGGREGATE_ECC_COUNTS_PARAMS {
    LW_DECLARE_ALIGNED(LW90E1_CTRL_FB_GET_ECC_PARTITION_COUNTS partitionEcc[LW90E1_CTRL_FB_ECC_PARTITION_COUNT], 8);
} LW90E1_CTRL_FB_GET_AGGREGATE_ECC_COUNTS_PARAMS;

/*
 * LW90E1_CTRL_CMD_FB_GET_EDC_CRC_DATA
 *
 * This command is used to query EDC expected/actual CRC data for each of the
 * partitions on FB.
 *
 * Floor-swept partitions are compacted and extra partition data is set to 0.
 *
 *   expectedCrcs
 *     This output parameter stores LW_PFB_FBPA_CRC_DATA2_EXPECTED_EDC per partition.
 *   actualCrcs
 *     This output parameter stores LW_PFB_FBPA_CRC_DATA2_ACTUAL_EDC per partition.
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 *
 */
#define LW90E1_CTRL_CMD_FB_GET_EDC_CRC_DATA (0x90e10106) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | LW90E1_CTRL_FB_GET_EDC_CRC_DATA_PARAMS_MESSAGE_ID" */

#define LW90E1_CTRL_FB_GET_EDC_CRC_DATA_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW90E1_CTRL_FB_GET_EDC_CRC_DATA_PARAMS {
    LwU32 expectedCrcs[LW90E1_CTRL_FB_ECC_PARTITION_COUNT];
    LwU32 actualCrcs[LW90E1_CTRL_FB_ECC_PARTITION_COUNT];
} LW90E1_CTRL_FB_GET_EDC_CRC_DATA_PARAMS;

/*
 * LW90E1_CTRL_CMD_FB_GET_EDC_MAX_DELTAS
 *
 * This command is used to query EDC max error deltas for each of the
 * partitions on FB.
 *
 *   maxDeltas
 *     This output parameter stores LW_PFB_FBPA_CRC_ERROR_MAX_DELTA_VALUE per partition.
 *     Floor-swept partitions are compacted and extra partition data is set to 0.
 *   crcTickValue
 *     This output parameter stores LW_PFB_FBPA_CRC_TICK_VALUE
 *     which is needed to callwlate the actual error rate.
 *   flags
 *     Input paremeter to specify additional actions.
 *     Legal values for this parameter include:
 *
 *       LW90E1_CTRL_FB_GET_EDC_MAX_DELTAS_FLAGS_RESET
 *           Indicates that RM should reset the max error deltas to 0 after
 *           reading the current values.
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 *
 */
#define LW90E1_CTRL_CMD_FB_GET_EDC_MAX_DELTAS (0x90e10107) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | LW90E1_CTRL_FB_GET_EDC_MAX_DELTAS_PARAMS_MESSAGE_ID" */

#define LW90E1_CTRL_FB_GET_EDC_MAX_DELTAS_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW90E1_CTRL_FB_GET_EDC_MAX_DELTAS_PARAMS {
    LwU32 maxDeltas[LW90E1_CTRL_FB_ECC_PARTITION_COUNT];
    LwU32 crcTickValue;
    LwU32 flags;
} LW90E1_CTRL_FB_GET_EDC_MAX_DELTAS_PARAMS;

/* Legal flags parameter values */
#define LW90E1_CTRL_FB_GET_EDC_MAX_DELTAS_FLAGS_RESET         0:0
#define LW90E1_CTRL_FB_GET_EDC_MAX_DELTAS_FLAGS_RESET_NO  (0x00000000)
#define LW90E1_CTRL_FB_GET_EDC_MAX_DELTAS_FLAGS_RESET_YES (0x00000001)

/*
 * LW90E1_CTRL_CMD_FB_QUERY_ECC_CAPABILITIES
 *
 * This command is used to query the ECC capabilities of the GPU. 
 *
 *   flags
 *     Output parameter to specify what ECC capabilities are supported.
 *     Legal values for this parameter include:
 *
 *       LW90E1_CTRL_FB_QUERY_ECC_CAPABILITIES_FLAGS_FB
 *           Indicates that ECC on the framebuffer is supported.
 *
 *       LW90E1_CTRL_FB_QUERY_ECC_CAPABILITIES_FLAGS_L2
 *           Indicates that ECC on the L2 cache is supported
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 *
 */
#define LW90E1_CTRL_CMD_FB_QUERY_ECC_CAPABILITIES         (0x90e10108) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | LW90E1_CTRL_FB_QUERY_ECC_CAPABILITIES_PARAMS_MESSAGE_ID" */

#define LW90E1_CTRL_FB_QUERY_ECC_CAPABILITIES_PARAMS_MESSAGE_ID (0x8U)

typedef struct LW90E1_CTRL_FB_QUERY_ECC_CAPABILITIES_PARAMS {
    LwU32 flags;
} LW90E1_CTRL_FB_QUERY_ECC_CAPABILITIES_PARAMS;

/* Legal flag parameter values */
#define LW90E1_CTRL_FB_QUERY_ECC_CAPABILITIES_FLAGS_FB         0:0
#define LW90E1_CTRL_FB_QUERY_ECC_CAPABILITIES_FLAGS_FB_NOECC  (0x00000000)
#define LW90E1_CTRL_FB_QUERY_ECC_CAPABILITIES_FLAGS_FB_ECC    (0x00000001)

#define LW90E1_CTRL_FB_QUERY_ECC_CAPABILITIES_FLAGS_LTC        1:1
#define LW90E1_CTRL_FB_QUERY_ECC_CAPABILITIES_FLAGS_LTC_NOECC (0x00000000)
#define LW90E1_CTRL_FB_QUERY_ECC_CAPABILITIES_FLAGS_LTC_ECC   (0x00000001)

/* _ctrl90e1_h_ */

