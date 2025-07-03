/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012-2015 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrlc0e1.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#include "ctrl/ctrla0e1.h"          // C0E1 is partially derived from A0E1
#define LWC0E1_CTRL_CMD(cat,idx)  \
    LWXXXX_CTRL_CMD(0xC0E1, LWC0E1_CTRL_##cat, idx)


/* LWC0E1 command categories (6bits) */
#define LWC0E1_CTRL_RESERVED (0x00)
#define LWC0E1_CTRL_FB       (0x01)


/*
 * LWC0E1_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LWC0E1_CTRL_CMD_NULL (0xc0e10000) /* finn: Evaluated from "(FINN_GP100_SUBDEVICE_FB_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*
 * LWC0E1_CTRL_CMD_FB_GET_ECC_COUNTS
 *
 * This command is used to query detailed ECC counts for each of the
 * units in FB, it covers LTC (level 2 cache) and FB memory errors
 *
 *   sliceCount (in/out)
 *     On input, this parameter specifies how many slices per partition the
 *     passed in structure can support. On output, this parameter specifies 
 *     how many slices of data was stored to the structure
 *
 *   subpartitionCount (out)
 *     On output, this parameter specifies how many subpartitions are 
 *     supported on this GPU. For each partition only this many of fbSbe, 
 *     fbDbe, fbScrubSbe, and fbScrubDbe are valid.
 *
 *   partitionCount (in/out)
 *     On input, this parameter specifies how many partitions the
 *     passed in structure can support. On output, this parameter 
 *     specifies how many partitions of data was stored to the structure.
 *
 *   flags (in)
 *     This parameter specifies whether or not raw error counts are 
 *     requested.
 *
 *   partitionEcc (out)
 *     This parameter stores the FB and LTC ECC single- and double-bit
 *     error counts.
 *
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 *
 */
#define LWC0E1_CTRL_CMD_FB_GET_ECC_COUNTS                 (0xc0e10101) /* finn: Evaluated from "(FINN_GP100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | LWC0E1_CTRL_FB_GET_ECC_COUNTS_PARAMS_MESSAGE_ID" */

#define LWC0E1_CTRL_FB_ECC_PARTITION_COUNT                (0x00000010)
#define LWC0E1_CTRL_FB_ECC_SLICE_COUNT                    (0x00000002)
#define LWC0E1_CTRL_FB_ECC_SUBPARTITION_COUNT             (0x00000002)

/*
 * Due to limitations of early ECC hardware, the RM may need to limit
 * the number of errors reported; e.g. it may be forced to report
 * no more than a single double-bit error, or to omit reporting of
 * single-bit errors completely.
 *
 * For RM some clients, such as MODS, this may not be sufficient. In
 * those cases, the RM can be instructed to return the errors as
 * they are obtained from the hardware itself, unfiltered.
 *
 */
#define LWC0E1_CTRL_FB_GET_ECC_COUNTS_FLAGS_TYPE             0:0
#define LWC0E1_CTRL_FB_GET_ECC_COUNTS_FLAGS_TYPE_FILTERED (0x00000000)
#define LWC0E1_CTRL_FB_GET_ECC_COUNTS_FLAGS_TYPE_RAW      (0x00000001)

/*
 * LWC0E1_CTRL_FB_QUERY_ECC_EXCEPTION_STATUS
 *
 * This structure represents the exception status of a class of per-unit
 * exceptions
 *
 *   count
 *     This parameter contains the number of exceptions that have oclwrred 
 *     since boot.
 *
 */
typedef LWA0E1_CTRL_FB_QUERY_ECC_EXCEPTION_STATUS LWC0E1_CTRL_FB_QUERY_ECC_EXCEPTION_STATUS;

typedef LWA0E1_CTRL_FB_GET_ECC_SUBPARTITION_COUNTS LWC0E1_CTRL_FB_GET_ECC_SUBPARTITION_COUNTS;

typedef LWA0E1_CTRL_FB_GET_ECC_SLICE_COUNTS LWC0E1_CTRL_FB_GET_ECC_SLICE_COUNTS;

typedef struct LWC0E1_CTRL_FB_GET_ECC_PARTITION_COUNTS {
    LW_DECLARE_ALIGNED(LWC0E1_CTRL_FB_GET_ECC_SLICE_COUNTS ltc[LWC0E1_CTRL_FB_ECC_SLICE_COUNT], 8);
    LW_DECLARE_ALIGNED(LWC0E1_CTRL_FB_GET_ECC_SUBPARTITION_COUNTS fb[LWC0E1_CTRL_FB_ECC_SUBPARTITION_COUNT], 8);
} LWC0E1_CTRL_FB_GET_ECC_PARTITION_COUNTS;

#define LWC0E1_CTRL_FB_GET_ECC_COUNTS_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWC0E1_CTRL_FB_GET_ECC_COUNTS_PARAMS {
    LwU32 sliceCount;
    LwU32 subPartitionCount;
    LwU32 partitionCount;
    LwU32 flags;

    LW_DECLARE_ALIGNED(LWC0E1_CTRL_FB_GET_ECC_PARTITION_COUNTS partitionEcc[LWC0E1_CTRL_FB_ECC_PARTITION_COUNT], 8);
} LWC0E1_CTRL_FB_GET_ECC_COUNTS_PARAMS;

/*
 * LWC0E1_CTRL_CMD_FB_GET_EDC_COUNTS
 *
 * Plase see description of LWA0E1_CTRL_CMD_FB_GET_EDC_COUNTS for more 
 * information.
 *
 */
#define LWC0E1_CTRL_CMD_FB_GET_EDC_COUNTS (0xc0e10102) /* finn: Evaluated from "(FINN_GP100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | 0x2" */

typedef LWA0E1_CTRL_FB_GET_EDC_COUNTS_PARAMS LWC0E1_CTRL_FB_GET_EDC_COUNTS_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LWC0E1_CTRL_CMD_FB_GET_EDC_COUNTS_FINN_PARAMS_MESSAGE_ID (0x2U)

typedef struct LWC0E1_CTRL_CMD_FB_GET_EDC_COUNTS_FINN_PARAMS {
    LW_DECLARE_ALIGNED(LWC0E1_CTRL_FB_GET_EDC_COUNTS_PARAMS params, 8);
} LWC0E1_CTRL_CMD_FB_GET_EDC_COUNTS_FINN_PARAMS;



/*
 * LW90E1_CTRL_CMD_FB_GET_SBE_ADDRESSES
 *
 * Plase see description of LWA0E1_CTRL_CMD_FB_GET_SBE_ADDRESSES for more 
 * information.
 *
 */

#define LWC0E1_CTRL_CMD_FB_GET_SBE_ADDRESSES                   (0xc0e10103) /* finn: Evaluated from "(FINN_GP100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | 0x3" */

#define LWC0E1_CTRL_CMD_FB_GET_ECC_ADDRESSES_MAX_NUM_ADDRESSES LWA0E1_CTRL_CMD_FB_GET_ECC_ADDRESSES_MAX_NUM_ADDRESSES

typedef LWA0E1_CTRL_FB_GET_SBE_ADDRESSES_PARAMS LWC0E1_CTRL_FB_GET_SBE_ADDRESSES_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LWC0E1_CTRL_CMD_FB_GET_SBE_ADDRESSES_FINN_PARAMS_MESSAGE_ID (0x3U)

typedef struct LWC0E1_CTRL_CMD_FB_GET_SBE_ADDRESSES_FINN_PARAMS {
    LWC0E1_CTRL_FB_GET_SBE_ADDRESSES_PARAMS params;
} LWC0E1_CTRL_CMD_FB_GET_SBE_ADDRESSES_FINN_PARAMS;



/*
 * LW90E1_CTRL_CMD_FB_GET_DBE_ADDRESSES
 *
 * Plase see description of LWA0E1_CTRL_CMD_FB_GET_DBE_ADDRESSES for more 
 * information.
 *
 */
#define LWC0E1_CTRL_CMD_FB_GET_DBE_ADDRESSES (0xc0e10104) /* finn: Evaluated from "(FINN_GP100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | 0x4" */

typedef LWA0E1_CTRL_FB_GET_DBE_ADDRESSES_PARAMS LWC0E1_CTRL_FB_GET_DBE_ADDRESSES_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LWC0E1_CTRL_CMD_FB_GET_DBE_ADDRESSES_FINN_PARAMS_MESSAGE_ID (0x4U)

typedef struct LWC0E1_CTRL_CMD_FB_GET_DBE_ADDRESSES_FINN_PARAMS {
    LWC0E1_CTRL_FB_GET_DBE_ADDRESSES_PARAMS params;
} LWC0E1_CTRL_CMD_FB_GET_DBE_ADDRESSES_FINN_PARAMS;



/*
 * LWC0E1_CTRL_CMD_FB_GET_AGGREGATE_ECC_COUNTS
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
#define LWC0E1_CTRL_CMD_FB_GET_AGGREGATE_ECC_COUNTS (0xc0e10105) /* finn: Evaluated from "(FINN_GP100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | 0x5" */

typedef LWA0E1_CTRL_FB_GET_AGGREGATE_ECC_COUNTS_PARAMS LWC0E1_CTRL_FB_GET_AGGREGATE_ECC_COUNTS_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LWC0E1_CTRL_CMD_FB_GET_AGGREGATE_ECC_COUNTS_FINN_PARAMS_MESSAGE_ID (0x5U)

typedef struct LWC0E1_CTRL_CMD_FB_GET_AGGREGATE_ECC_COUNTS_FINN_PARAMS {
    LW_DECLARE_ALIGNED(LWC0E1_CTRL_FB_GET_AGGREGATE_ECC_COUNTS_PARAMS params, 8);
} LWC0E1_CTRL_CMD_FB_GET_AGGREGATE_ECC_COUNTS_FINN_PARAMS;



/*
 * LWC0E1_CTRL_CMD_FB_GET_EDC_CRC_DATA
 *
 * Plase see description of LWA0E1_CTRL_CMD_FB_GET_EDC_CRC_DATA for more 
 * information.
 *
 */
#define LWC0E1_CTRL_CMD_FB_GET_EDC_CRC_DATA (0xc0e10106) /* finn: Evaluated from "(FINN_GP100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | 0x6" */

typedef LWA0E1_CTRL_FB_GET_EDC_CRC_DATA_PARAMS LWC0E1_CTRL_FB_GET_EDC_CRC_DATA_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LWC0E1_CTRL_CMD_FB_GET_EDC_CRC_DATA_FINN_PARAMS_MESSAGE_ID (0x6U)

typedef struct LWC0E1_CTRL_CMD_FB_GET_EDC_CRC_DATA_FINN_PARAMS {
    LWC0E1_CTRL_FB_GET_EDC_CRC_DATA_PARAMS params;
} LWC0E1_CTRL_CMD_FB_GET_EDC_CRC_DATA_FINN_PARAMS;



/*
 * LWC0E1_CTRL_CMD_FB_GET_EDC_MAX_DELTAS
 *
 * Plase see description of LWA0E1_CTRL_CMD_FB_GET_EDC_MAX_DELTAS for more 
 * information.
 *
 */
#define LWC0E1_CTRL_CMD_FB_GET_EDC_MAX_DELTAS (0xc0e10107) /* finn: Evaluated from "(FINN_GP100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | 0x7" */

typedef LWA0E1_CTRL_FB_GET_EDC_MAX_DELTAS_PARAMS LWC0E1_CTRL_FB_GET_EDC_MAX_DELTAS_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LWC0E1_CTRL_CMD_FB_GET_EDC_MAX_DELTAS_FINN_PARAMS_MESSAGE_ID (0x7U)

typedef struct LWC0E1_CTRL_CMD_FB_GET_EDC_MAX_DELTAS_FINN_PARAMS {
    LWC0E1_CTRL_FB_GET_EDC_MAX_DELTAS_PARAMS params;
} LWC0E1_CTRL_CMD_FB_GET_EDC_MAX_DELTAS_FINN_PARAMS;



/* Legal flags parameter values */
#define LWC0E1_CTRL_FB_GET_EDC_MAX_DELTAS_FLAGS_RESET      LWA0E1_CTRL_FB_GET_EDC_MAX_DELTAS_FLAGS_RESET
#define LWC0E1_CTRL_FB_GET_EDC_MAX_DELTAS_FLAGS_RESET_NO  LWA0E1_CTRL_FB_GET_EDC_MAX_DELTAS_FLAGS_RESET_NO
#define LWC0E1_CTRL_FB_GET_EDC_MAX_DELTAS_FLAGS_RESET_YES LWA0E1_CTRL_FB_GET_EDC_MAX_DELTAS_FLAGS_RESET_YES

/*
 * LWC0E1_CTRL_CMD_FB_QUERY_ECC_CAPABILITIES
 *
 * Plase see description of LWA0E1_CTRL_CMD_FB_QUERY_ECC_CAPABILITIES for more 
 * information.
 *
 */
#define LWC0E1_CTRL_CMD_FB_QUERY_ECC_CAPABILITIES         (0xc0e10108) /* finn: Evaluated from "(FINN_GP100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | 0x8" */

typedef LWA0E1_CTRL_FB_QUERY_ECC_CAPABILITIES_PARAMS LWC0E1_CTRL_FB_QUERY_ECC_CAPABILITIES_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LWC0E1_CTRL_CMD_FB_QUERY_ECC_CAPABILITIES_FINN_PARAMS_MESSAGE_ID (0x8U)

typedef struct LWC0E1_CTRL_CMD_FB_QUERY_ECC_CAPABILITIES_FINN_PARAMS {
    LWC0E1_CTRL_FB_QUERY_ECC_CAPABILITIES_PARAMS params;
} LWC0E1_CTRL_CMD_FB_QUERY_ECC_CAPABILITIES_FINN_PARAMS;



/* Legal flag parameter values */
#define LWC0E1_CTRL_FB_QUERY_ECC_CAPABILITIES_FLAGS_FB         LW90E1_CTRL_FB_QUERY_ECC_CAPABILITIES_FLAGS_FB
#define LWC0E1_CTRL_FB_QUERY_ECC_CAPABILITIES_FLAGS_FB_NOECC  LW90E1_CTRL_FB_QUERY_ECC_CAPABILITIES_FLAGS_FB_NOECC
#define LWC0E1_CTRL_FB_QUERY_ECC_CAPABILITIES_FLAGS_FB_ECC    LW90E1_CTRL_FB_QUERY_ECC_CAPABILITIES_FLAGS_FB_ECC

#define LWC0E1_CTRL_FB_QUERY_ECC_CAPABILITIES_FLAGS_LTC        LW90E1_CTRL_FB_QUERY_ECC_CAPABILITIES_FLAGS_LTC
#define LWC0E1_CTRL_FB_QUERY_ECC_CAPABILITIES_FLAGS_LTC_NOECC LW90E1_CTRL_FB_QUERY_ECC_CAPABILITIES_FLAGS_LTC_NOECC
#define LWC0E1_CTRL_FB_QUERY_ECC_CAPABILITIES_FLAGS_LTC_ECC   LW90E1_CTRL_FB_QUERY_ECC_CAPABILITIES_FLAGS_LTC_ECC

/*
 * LWC0E1_CTRL_CMD_FB_GET_DRAM_ECC_COUNTS
 *
 * This command is used to query detailed volatile ECC counts for the DRAM
 * units in FB since the last reboot or driver load.  A simple array of 
 * errors is returned including the partition and subpartition number of 
 * the affected unit.
 *
 *   flags (in, out)
 *     This parameter specifies whether or not raw error counts are 
 *     requested. See LWC0E1_CTRL_FB_GET_ECC_COUNTS_FLAGS_TYPE defined
 *     above for valid flag values on input.
 *     On output, this parameter specifies whether or not all units with ECC
 *     errors fit into the dramEcc array.
 * 
 *   entryCount (out)
 *     This parameter specifies the number of entries in the list which
 *     contain counts. The range will be between 0 and 
 *     LWC0E1_CTRL_FB_ECC_DRAM_COUNT.
 *
 *   dramEcc (out)
 *     This parameter stores the DRAM ECC single- and double-bit
 *     error counts.
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_NOT_SUPPORTED
 *
 */
#define LWC0E1_CTRL_CMD_FB_GET_DRAM_ECC_COUNTS                (0xc0e10109) /* finn: Evaluated from "(FINN_GP100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | 0x9" */

#define LWC0E1_CTRL_FB_DRAM_ECC_MAX_COUNT                     (LWA0E1_CTRL_FB_DRAM_ECC_MAX_COUNT)

typedef LWA0E1_CTRL_FB_GET_DRAM_ECC_COUNTS LWC0E1_CTRL_FB_GET_DRAM_ECC_COUNTS;

#define LWC0E1_CTRL_FB_GET_ECC_COUNTS_FLAGS_OVERFLOW         1:1
#define LWC0E1_CTRL_FB_GET_ECC_COUNTS_FLAGS_OVERFLOW_FALSE (0x00000000)
#define LWC0E1_CTRL_FB_GET_ECC_COUNTS_FLAGS_OVERFLOW_TRUE  (0x00000001)

typedef LWA0E1_CTRL_FB_GET_DRAM_ECC_COUNTS_PARAMS LWC0E1_CTRL_FB_GET_DRAM_ECC_COUNTS_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LWC0E1_CTRL_CMD_FB_GET_DRAM_ECC_COUNTS_FINN_PARAMS_MESSAGE_ID (0x9U)

typedef struct LWC0E1_CTRL_CMD_FB_GET_DRAM_ECC_COUNTS_FINN_PARAMS {
    LW_DECLARE_ALIGNED(LWC0E1_CTRL_FB_GET_DRAM_ECC_COUNTS_PARAMS params, 8);
} LWC0E1_CTRL_CMD_FB_GET_DRAM_ECC_COUNTS_FINN_PARAMS;



/*
 * LWC0E1_CTRL_CMD_FB_GET_LTC_ECC_COUNTS
 *
 * This command is used to query detailed volatile ECC counts for the LTC 
 * (level 2 cache) units in FB since the last reboot or driver load. A 
 * simple array of errors is returned including the partition and slice 
 * number of the affected unit.
 *
 *   flags (out)
 *     This parameter specifies whether or not all units with ECC errors
 *     fit into the ltcEcc array.
 * 
 *   entryCount (out)
 *     This parameter specifies the number of entries in the list which
 *     contain counts. The range will be between 0 and 
 *     LWC0E1_CTRL_FB_LTC_ECC_COUNT.
 *
 *   ltcEcc (out)
 *     This parameter stores the DRAM ECC single- and double-bit
 *     error counts.
 *
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_NOT_SUPPORTED
 *
 */
#define LWC0E1_CTRL_CMD_FB_GET_LTC_ECC_COUNTS (0xc0e1010a) /* finn: Evaluated from "(FINN_GP100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | 0xA" */

#define LWC0E1_CTRL_FB_LTC_ECC_MAX_COUNT      (32)

typedef LWA0E1_CTRL_FB_GET_LTC_ECC_COUNTS LWC0E1_CTRL_FB_GET_LTC_ECC_COUNTS;

typedef LWA0E1_CTRL_FB_GET_LTC_ECC_COUNTS_PARAMS LWC0E1_CTRL_FB_GET_LTC_ECC_COUNTS_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LWC0E1_CTRL_CMD_FB_GET_LTC_ECC_COUNTS_FINN_PARAMS_MESSAGE_ID (0xAU)

typedef struct LWC0E1_CTRL_CMD_FB_GET_LTC_ECC_COUNTS_FINN_PARAMS {
    LWC0E1_CTRL_FB_GET_LTC_ECC_COUNTS_PARAMS params;
} LWC0E1_CTRL_CMD_FB_GET_LTC_ECC_COUNTS_FINN_PARAMS;



/*
 * LWC0E1_CTRL_CMD_FB_GET_AGGREGATE_DRAM_ECC_COUNTS
 *
 * This command is used to query detailed aggregate ECC counts for the DRAM
 * units in FB since the last reboot or driver load.  A simple array of 
 * errors is returned including the partition and subpartition number of 
 * the affected unit.
 *
 * See LWC0E1_CTRL_FB_GET_DRAM_ECC_COUNTS_PARAMS for details on the parameters.
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_NOT_SUPPORTED
 *
 */
#define LWC0E1_CTRL_CMD_FB_GET_AGGREGATE_DRAM_ECC_COUNTS (0xc0e1010b) /* finn: Evaluated from "(FINN_GP100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | 0xB" */

typedef LWC0E1_CTRL_FB_GET_DRAM_ECC_COUNTS_PARAMS LWC0E1_CTRL_FB_GET_AGGREGATE_DRAM_ECC_COUNTS_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LWC0E1_CTRL_CMD_FB_GET_AGGREGATE_DRAM_ECC_COUNTS_FINN_PARAMS_MESSAGE_ID (0xBU)

typedef struct LWC0E1_CTRL_CMD_FB_GET_AGGREGATE_DRAM_ECC_COUNTS_FINN_PARAMS {
    LW_DECLARE_ALIGNED(LWC0E1_CTRL_FB_GET_AGGREGATE_DRAM_ECC_COUNTS_PARAMS params, 8);
} LWC0E1_CTRL_CMD_FB_GET_AGGREGATE_DRAM_ECC_COUNTS_FINN_PARAMS;



/*
 * LWC0E1_CTRL_CMD_FB_GET_AGGREGATE_LTC_ECC_COUNTS
 *
 * This command is used to query detailed aggregate ECC counts for the LTC 
 * (level 2 cache) units in FB since the last reboot or driver load. A 
 * simple array of errors is returned including the partition and slice 
 * number of the affected unit.
 *
 * See LWA0E1_CTRL_FB_GET_LTC_ECC_COUNTS_PARAMS for details on the parameters.
 *
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_NOT_SUPPORTED
 *
 */
#define LWC0E1_CTRL_CMD_FB_GET_AGGREGATE_LTC_ECC_COUNTS (0xc0e1010c) /* finn: Evaluated from "(FINN_GP100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | 0xC" */

typedef LWC0E1_CTRL_FB_GET_LTC_ECC_COUNTS_PARAMS LWC0E1_CTRL_FB_GET_AGGREGATE_LTC_ECC_COUNTS_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LWC0E1_CTRL_CMD_FB_GET_AGGREGATE_LTC_ECC_COUNTS_FINN_PARAMS_MESSAGE_ID (0xLW)

typedef struct LWC0E1_CTRL_CMD_FB_GET_AGGREGATE_LTC_ECC_COUNTS_FINN_PARAMS {
    LWC0E1_CTRL_FB_GET_AGGREGATE_LTC_ECC_COUNTS_PARAMS params;
} LWC0E1_CTRL_CMD_FB_GET_AGGREGATE_LTC_ECC_COUNTS_FINN_PARAMS;



/*
 * LWC0E1_CTRL_CMD_FB_GET_LATEST_ECC_ADDRESSES
 *
 * This command is used to query FB addresses of the latest ECC memory events
 * since last driver restart.
 *
 *   addressCount
 *     This parameter is the number of valid addresses in the list
 *     returned by the physAddresses parameter.
 *
 *   totalAddressCount
 *     This parameter is the total number of addresses that has been
 *     added to the list since driver initialization. 
 *
 *   physAddresses
 *     This parameter stores the physical address.
 *
 *   addresses
 *     This parameter stores the RBC (Row/Bank/Column) address.
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_NOT_SUPPORTED
 *
 */

#define LWC0E1_CTRL_CMD_FB_GET_LATEST_ECC_ADDRESSES            (0xc0e1010d) /* finn: Evaluated from "(FINN_GP100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | 0xD" */

#define LWC0E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_MAX_COUNT      (32)

#define LWC0E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_ERROR_TYPE        0:0
#define LWC0E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_ERROR_TYPE_SBE (0x00000000)
#define LWC0E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_ERROR_TYPE_DBE (0x00000001)

/*
 * LWC0E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_ADDRESS
 *
 *   physAddress
 *     This parameter is the physical address of the error, as callwlated
 *     by the reverse mapping of the Row/Bank/Column address identified by
 *     the GPU.
 *
 *   rbcAddress
 *     This parameter is the row, bank, column address as directly reported
 *     by the GPU.
 *
 *   rbcAddressExt
 *     This parameter is the extended row, bank, column address as directly
 *     reported by the GPU.
 *
 *   partitionNumber
 *     This parameter is partition number in which the ECC error oclwred.
 *
 *   subPartitionNumber
 *     This parameter is the subpartition number in which the ECC error 
 *     oclwred.
 *
 *   errorType
 *     This parameter is the type of ECC error that oclwred, either single
 *     bit error (SBE) or double bit error (DBE).
 *
 */
typedef LWA0E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_ADDRESS LWC0E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_ADDRESS;

typedef LWA0E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_PARAMS LWC0E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LWC0E1_CTRL_CMD_FB_GET_LATEST_ECC_ADDRESSES_FINN_PARAMS_MESSAGE_ID (0xDU)

typedef struct LWC0E1_CTRL_CMD_FB_GET_LATEST_ECC_ADDRESSES_FINN_PARAMS {
    LW_DECLARE_ALIGNED(LWC0E1_CTRL_FB_GET_LATEST_ECC_ADDRESSES_PARAMS params, 8);
} LWC0E1_CTRL_CMD_FB_GET_LATEST_ECC_ADDRESSES_FINN_PARAMS;



/*!
 * This structure contains the address of DRAM single- and double-bit error.
 *
 *   address0 (out)
 *     This parameter returns the following address fields
 *       TYPE
 *         This denotes the type of ECC error.
 *           NONE - It entry is empty.
 *           SBE  - It is a single bit ECC error.
 *           DBE  - It is a double bit ECC error.
 *       PARTITON
 *         This denotes the partition number.
 *       SUBPARTITON
 *         This denotes the subpartition number.
 *       COLUMN
 *         This denotes the column address.
 *
 *   address1 (out)
 *     This parameter returns the following address fields
 *       ROW
 *         This denotes the row address.
 *       BANK
 *         This denotes the memory bank.
 *       EXTBANK
 *         This denotes the external memory bank.
 */
typedef struct LWC0E1_CTRL_FB_AGGREGATE_DRAM_ECC_ADDRESS {
    LwU32 address0;
    LwU32 address1;
} LWC0E1_CTRL_FB_AGGREGATE_DRAM_ECC_ADDRESS;

#define LWC0E1_CTRL_FB_AGGREGATE_DRAM_ECC_ADDRESS_0_TYPE                     1:0
#define LWC0E1_CTRL_FB_AGGREGATE_DRAM_ECC_ADDRESS_0_TYPE_NONE 0x00000000
#define LWC0E1_CTRL_FB_AGGREGATE_DRAM_ECC_ADDRESS_0_TYPE_SBE  0x00000001
#define LWC0E1_CTRL_FB_AGGREGATE_DRAM_ECC_ADDRESS_0_TYPE_DBE  0x00000002
#define LWC0E1_CTRL_FB_AGGREGATE_DRAM_ECC_ADDRESS_0_PARTITION                5:2
#define LWC0E1_CTRL_FB_AGGREGATE_DRAM_ECC_ADDRESS_0_SUBPARTITION             6:6
#define LWC0E1_CTRL_FB_AGGREGATE_DRAM_ECC_ADDRESS_0_COLUMN                 31:12
#define LWC0E1_CTRL_FB_AGGREGATE_DRAM_ECC_ADDRESS_1_ROW                     19:0
#define LWC0E1_CTRL_FB_AGGREGATE_DRAM_ECC_ADDRESS_1_BANK                   30:27
#define LWC0E1_CTRL_FB_AGGREGATE_DRAM_ECC_ADDRESS_1_EXTBANK                31:31

/*
 * LWC0E1_CTRL_CMD_FB_GET_AGGREGATE_DRAM_ECC_ADDRESSES
 *
 * This command is used to query aggregate ECC addresses for the DRAM units in
 * FB for the lifetime of GPU. An array of error addresses.
 *
 *   flags (out)
 *     This parameter returns the following flags.
 *       LWC0E1_CTRL_FB_GET_AGGREGATE_DRAM_ECC_ADDRESSES_FLAGS_OVERFLOW
 *         When set this denotes that number of aggregate ECC addresses are more
 *         than maximum number of entries that can be returned.
 * 
 *   addressCount (out)
 *     This parameter returns the number of address entries. The range will be
 *     between 0 and LWC0E1_CTRL_FB_AGGREGATE_DRAM_ECC_ADDRESS_ENTRIES.
 *
 *   address (out)
 *     This parameter is an array of addresses. It returns the DRAM ECC single-
 *     and double-bit error addresses. For more details please see description
 *     of LWC0E1_CTRL_FB_AGGREGATE_DRAM_ECC_ADDRESS.
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_NOT_SUPPORTED
 */

#define LWC0E1_CTRL_CMD_FB_GET_AGGREGATE_DRAM_ECC_ADDRESSES   (0xc0e10110) /* finn: Evaluated from "(FINN_GP100_SUBDEVICE_FB_FB_INTERFACE_ID << 8) | 0x10" */

#define LWC0E1_CTRL_FB_AGGREGATE_DRAM_ECC_ADDRESS_ENTRIES     256

typedef struct LWC0E1_CTRL_FB_GET_AGGREGATE_DRAM_ECC_ADDRESSES_PARAMS {
    LwU32                                     flags;
    LwU32                                     addressCount;
    LWC0E1_CTRL_FB_AGGREGATE_DRAM_ECC_ADDRESS address[LWC0E1_CTRL_FB_AGGREGATE_DRAM_ECC_ADDRESS_ENTRIES];
} LWC0E1_CTRL_FB_GET_AGGREGATE_DRAM_ECC_ADDRESSES_PARAMS;

/* _ctrlc0e1_h_ */


#define LWC0E1_CTRL_FB_GET_AGGREGATE_DRAM_ECC_ADDRESSES_FLAGS_OVERFLOW    LWBIT(0)
