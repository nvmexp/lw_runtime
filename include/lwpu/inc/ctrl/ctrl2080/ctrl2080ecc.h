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
// Source file: ctrl/ctrl2080/ctrl2080ecc.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl2080/ctrl2080base.h"

/*
 * LW2080_CTRL_CMD_ECC_GET_AGGREGATE_ERROR_COUNTS
 *
 * eccCounts [out]
 *      Reports the error counters for all the units.
 */

#define LW2080_CTRL_CMD_ECC_GET_AGGREGATE_ERROR_COUNTS (0x20803401) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_ECC_INTERFACE_ID << 8) | LW2080_CTRL_ECC_GET_AGGREGATE_ERROR_COUNTS_PARAMS_MESSAGE_ID" */

typedef enum LW2080_ECC_UNITS {
    ECC_UNIT_LRF = 0,
    ECC_UNIT_CBU = 1,
    ECC_UNIT_L1 = 2,
    ECC_UNIT_L1DATA = 3,
    ECC_UNIT_L1TAG = 4,
    ECC_UNIT_SHM = 5,
    ECC_UNIT_TEX = 6,
    ECC_UNIT_SM_ICACHE = 7,
    ECC_UNIT_LTC = 8,
    ECC_UNIT_DRAM = 9,
    ECC_UNIT_GCC_L15 = 10,
    ECC_UNIT_GPCMMU = 11,
    ECC_UNIT_HUBMMU_L2TLB = 12,
    ECC_UNIT_HUBMMU_HUBTLB = 13,
    ECC_UNIT_HUBMMU_FILLUNIT = 14,
    ECC_UNIT_GPCCS = 15,
    ECC_UNIT_FECS = 16,
    ECC_UNIT_PMU = 17,
    ECC_UNIT_SM_RAMS = 18,
    ECC_UNIT_HSHUB = 19,
    ECC_UNIT_PCIE_REORDER = 20,
    ECC_UNIT_PCIE_P2PREQ = 21,
    ECC_UNIT_MAX = 22,
    ECC_UNIT_GR = 23, // ECC_UNIT_GR isn't an actual unit (see comment below)
} LW2080_ECC_UNITS;
/*
 * Because most GR units use the grEccLocation struct, it's helpful to expose
 * this struct directly so clients have a generic way to refer to any unit
 * location that uses grEccLocation.
 */

typedef struct LW2080_ECC_COUNTS {
    //
    // Indicates whether this entry is valid. This is required as ECC support
    // for units can change between architectures.
    // Deprecated in favor of the flags field.
    //
    LwU8 flags;

    LW_DECLARE_ALIGNED(LwU64 correctedTotalCounts, 8);
    LW_DECLARE_ALIGNED(LwU64 correctedUniqueCounts, 8);
    LW_DECLARE_ALIGNED(LwU64 uncorrectedTotalCounts, 8);
    LW_DECLARE_ALIGNED(LwU64 uncorrectedUniqueCounts, 8);
} LW2080_ECC_COUNTS;

// Legal flag values
#define LW2080_ECC_COUNTS_FLAGS_VALID               0:0
#define LW2080_ECC_COUNTS_FLAGS_VALID_TRUE         1
#define LW2080_ECC_COUNTS_FLAGS_VALID_FALSE        0

#define LW2080_ECC_COUNTS_FLAGS_UNIQUE_VALID        1:1
#define LW2080_ECC_COUNTS_FLAGS_UNIQUE_VALID_TRUE  1
#define LW2080_ECC_COUNTS_FLAGS_UNIQUE_VALID_FALSE 0

#define LW2080_CTRL_ECC_GET_AGGREGATE_ERROR_COUNTS_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_ECC_GET_AGGREGATE_ERROR_COUNTS_PARAMS {
    LW_DECLARE_ALIGNED(LW2080_ECC_COUNTS eccCounts[ECC_UNIT_MAX], 8);
} LW2080_CTRL_ECC_GET_AGGREGATE_ERROR_COUNTS_PARAMS;

/*
 * LW2080_CTRL_CMD_ECC_GET_DETAILED_AGGREGATE_ERROR_COUNTS
 *
 * unit [in]:
 *      The ECC protected unit for which the counts are being requested.
 *
 * location [in]:
 *      The detailed location information for which the counters are requested.
 *      The unit field will determine if the locations are gpc-tpc or
 *      partition-subpartition.
 *
 * eccCounts [out]:
 *      The counters for the errors seen in the requested location.
 *
 * Return values:
 *      LW_OK on success
 *      LW_ERR_ILWALID_ARGUMENT if the requested unit or the location is invalid.
 *      LW_ERR_NOT_SUPPORTED otherwise
 */

#define LW2080_CTRL_CMD_ECC_GET_DETAILED_AGGREGATE_ERROR_COUNTS (0x20803402) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_ECC_INTERFACE_ID << 8) | LW2080_CTRL_ECC_GET_DETAILED_AGGREGATE_ERROR_COUNTS_PARAMS_MESSAGE_ID" */

typedef struct grEccLocation {
    LwU32 location;
    LwU32 sublocation;
} grEccLocation;

typedef struct ltcEccLocation {
    LwU32 partition;
    LwU32 slice;
} ltcEccLocation;

typedef struct dramEccLocation {
    LwU32 partition;
    LwU32 sublocation;
} dramEccLocation;

typedef struct texEccLocation {
    grEccLocation grLocation;
    LwU32         texId;
} texEccLocation;

typedef struct falconEccLocation {
    //
    // Lwrrently this is unused and RM does not check the contents
    // Added for future proofing and makes integration with eccInjection SRTs easier
    //
    LwU8 falconLocation;
} falconEccLocation;

typedef struct busEccLocation {
    //
    // Lwrrently this is unused and RM does not check the contents
    // Added for future proofing and makes integration with eccInjection SRTs easier
    //
    LwU8 busLocation;
} busEccLocation;

typedef union LW2080_ECC_LOCATION_INFO {
    grEccLocation     lrf;

    grEccLocation     cbu;

    grEccLocation     l1;

    grEccLocation     l1Data;

    grEccLocation     l1Tag;

    grEccLocation     shm;

    texEccLocation    tex;

    grEccLocation     smIcache;

    ltcEccLocation    ltc;

    dramEccLocation   dram;

    grEccLocation     gr;

    grEccLocation     gccl15;

    grEccLocation     gpcmmu;

    grEccLocation     l2tlb;

    grEccLocation     hubtlb;

    grEccLocation     fillunit;

    grEccLocation     gpccs;

    grEccLocation     fecs;

    falconEccLocation pmu;

    grEccLocation     smRams;

    grEccLocation     hshub;

    busEccLocation    pcie_reorder;

    busEccLocation    pcie_p2preq;
} LW2080_ECC_LOCATION_INFO;


#define LW2080_CTRL_ECC_GET_DETAILED_AGGREGATE_ERROR_COUNTS_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW2080_CTRL_ECC_GET_DETAILED_AGGREGATE_ERROR_COUNTS_PARAMS {
    LW2080_ECC_UNITS         unit;
    LW2080_ECC_LOCATION_INFO location;
    LW_DECLARE_ALIGNED(LW2080_ECC_COUNTS eccCounts, 8);
} LW2080_CTRL_ECC_GET_DETAILED_AGGREGATE_ERROR_COUNTS_PARAMS;

/*
 * LW2080_CTRL_CMD_ECC_GET_DETAILED_ERROR_COUNTS
 *
 * Similar to LW2080_CTRL_CMD_ECC_GET_DETAILED_AGGREGATE_ERROR_COUNTS except used
 * for getting ECC counters in the current driver run.
 *
 * Refer to LW2080_CTRL_ECC_GET_DETAILED_AGGREGATE_ERROR_COUNTS_PARAMS
 * details for usage.
 */

#define LW2080_CTRL_CMD_ECC_GET_DETAILED_ERROR_COUNTS (0x20803403) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_ECC_INTERFACE_ID << 8) | LW2080_CTRL_ECC_GET_DETAILED_ERROR_COUNTS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_ECC_GET_DETAILED_ERROR_COUNTS_PARAMS_MESSAGE_ID (0x3U)

typedef LW2080_CTRL_ECC_GET_DETAILED_AGGREGATE_ERROR_COUNTS_PARAMS LW2080_CTRL_ECC_GET_DETAILED_ERROR_COUNTS_PARAMS;

/*
 * LW2080_CTRL_CMD_ECC_GET_AGGREGATE_DRAM_ERROR_ADDRESSES
 *
 * Reports the ECC DRAM addresses stored on the InfoROM.
 *
 * flags [out]:
 *
 * addressCount [out]:
 *      Number of valid entries in the LW2080_ECC_AGGREGATE_DRAM_ADDRESS array.
 *
 * LW2080_ECC_AGGREGATE_DRAM_ADDRESS [out]:
 *      Contains the address information for the ECC errors that have oclwrred
 *      over the life of the board.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_POINTER
 *      LW_ERR_ILWALID_STATE
 *      LW_ERR_NOT_SUPPORTED
 */

#define LW2080_CTRL_CMD_ECC_GET_AGGREGATE_DRAM_ERROR_ADDRESSES                  (0x20803404) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_ECC_INTERFACE_ID << 8) | LW2080_CTRL_ECC_GET_AGGREGATE_DRAM_ERROR_ADDRESSES_PARAMS_MESSAGE_ID" */

#define LW2080_ECC_AGGREGATE_DRAM_ADDRESS_MAX_COUNT                             600

#define LW2080_CTRL_ECC_GET_AGGREGATE_DRAM_ERROR_ADDRESSES_FLAGS_OVERFLOW        0:0
#define LW2080_CTRL_ECC_GET_AGGREGATE_DRAM_ERROR_ADDRESSES_FLAGS_OVERFLOW_FLASE 0
#define LW2080_CTRL_ECC_GET_AGGREGATE_DRAM_ERROR_ADDRESSES_FLAGS_OVERFLOW_TRUE  1

#define LW2080_ECC_AGGREGATE_DRAM_ADDRESS_ERROR_TYPE_SBE        0:0
#define LW2080_ECC_AGGREGATE_DRAM_ADDRESS_ERROR_TYPE_SBE_FALSE                  0
#define LW2080_ECC_AGGREGATE_DRAM_ADDRESS_ERROR_TYPE_SBE_TRUE                   1

#define LW2080_ECC_AGGREGATE_DRAM_ADDRESS_ERROR_TYPE_DBE        1:1
#define LW2080_ECC_AGGREGATE_DRAM_ADDRESS_ERROR_TYPE_DBE_FALSE                  0
#define LW2080_ECC_AGGREGATE_DRAM_ADDRESS_ERROR_TYPE_DBE_TRUE                   1

typedef struct ECC_DRAM_ADDRESS_RBC {
    LwU32 row;
    LwU32 bank;
    LwU32 column;
    LwU32 extBank;
} ECC_DRAM_ADDRESS_RBC;

typedef struct LW2080_ECC_AGGREGATE_DRAM_ADDRESS {
    LwU8                 errorType;
    dramEccLocation      location;
    ECC_DRAM_ADDRESS_RBC rbc;
} LW2080_ECC_AGGREGATE_DRAM_ADDRESS;

#define LW2080_CTRL_ECC_GET_AGGREGATE_DRAM_ERROR_ADDRESSES_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW2080_CTRL_ECC_GET_AGGREGATE_DRAM_ERROR_ADDRESSES_PARAMS {
    LwU32                             flags;
    LwU32                             addressCount;
    LW2080_ECC_AGGREGATE_DRAM_ADDRESS address[LW2080_ECC_AGGREGATE_DRAM_ADDRESS_MAX_COUNT];
} LW2080_CTRL_ECC_GET_AGGREGATE_DRAM_ERROR_ADDRESSES_PARAMS;

/*
 * LW2080_CTRL_CMD_ECC_GET_SRAM_ERROR_BUFFER
 *
 * Reports the SRAM round robin buffer stored on the InfoROM.
 *
 * LW2080_CTRL_ECC_SRAM_ERROR_BUFFER_ENTRY [out]:
 *      Contains the SRAM error buffer stored on the InfoROM.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_STATE
 *      LW_ERR_NOT_SUPPORTED
 */

#define LW2080_CTRL_CMD_ECC_GET_SRAM_ERROR_BUFFER                        (0x20803405) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_ECC_INTERFACE_ID << 8) | LW2080_CTRL_ECC_GET_SRAM_ERROR_BUFFER_PARAMS_MESSAGE_ID" */

#define LW2080_ECC_SRAM_ERROR_MAX_COUNT                                  16

#define LW2080_CTRL_ECC_SRAM_ERROR_BUFFER_ENTRY_ERROR_TYPE_UNCORRECTABLE 0x0
#define LW2080_CTRL_ECC_SRAM_ERROR_BUFFER_ENTRY_ERROR_TYPE_CORRECTABLE   0x1

typedef struct LW2080_CTRL_ECC_SRAM_ERROR_BUFFER_ENTRY {
    LW2080_ECC_UNITS         unit;
    LW2080_ECC_LOCATION_INFO location;

    LwU8                     errorType;
    LwU32                    timestamp;
    LwU32                    address;
} LW2080_CTRL_ECC_SRAM_ERROR_BUFFER_ENTRY;

#define LW2080_CTRL_ECC_GET_SRAM_ERROR_BUFFER_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW2080_CTRL_ECC_GET_SRAM_ERROR_BUFFER_PARAMS {
    LwU32                                   entryCount;
    LW2080_CTRL_ECC_SRAM_ERROR_BUFFER_ENTRY sramError[LW2080_ECC_SRAM_ERROR_MAX_COUNT];
} LW2080_CTRL_ECC_GET_SRAM_ERROR_BUFFER_PARAMS;

/*
 * LW2080_CTRL_CMD_ECC_GET_AGGREGATE_ERROR_ADDRESSES
 *
 * Reports the ECC addresses stored on the InfoROM.
 *
 * addressCount [out]:
 *      Number of valid entries in the LW2080_ECC_AGGREGATE_ADDRESS array.
 *
 * LW2080_ECC_AGGREGATE_ADDRESS [out]:
 *      Contains the address information for the ECC errors that have oclwrred
 *      over the life of the board.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_POINTER
 *      LW_ERR_ILWALID_STATE
 *      LW_ERR_NOT_SUPPORTED
 */

#define LW2080_CTRL_CMD_ECC_GET_AGGREGATE_ERROR_ADDRESSES                  (0x20803406) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_ECC_INTERFACE_ID << 8) | LW2080_CTRL_ECC_GET_AGGREGATE_ERROR_ADDRESSES_PARAMS_MESSAGE_ID" */

#define LW2080_ECC_AGGREGATE_ADDRESS_MAX_COUNT                             600

#define LW2080_CTRL_ECC_GET_AGGREGATE_ERROR_ADDRESSES_FLAGS_OVERFLOW        0:0
#define LW2080_CTRL_ECC_GET_AGGREGATE_ERROR_ADDRESSES_FLAGS_OVERFLOW_FLASE 0
#define LW2080_CTRL_ECC_GET_AGGREGATE_ERROR_ADDRESSES_FLAGS_OVERFLOW_TRUE  1

#define LW2080_ECC_AGGREGATE_ADDRESS_ERROR_TYPE_CORR                    0:0
#define LW2080_ECC_AGGREGATE_ADDRESS_ERROR_TYPE_CORR_FALSE                 0
#define LW2080_ECC_AGGREGATE_ADDRESS_ERROR_TYPE_CORR_TRUE                  1

#define LW2080_ECC_AGGREGATE_ADDRESS_ERROR_TYPE_UNCORR                  1:1
#define LW2080_ECC_AGGREGATE_ADDRESS_ERROR_TYPE_UNCORR_FALSE               0
#define LW2080_ECC_AGGREGATE_ADDRESS_ERROR_TYPE_UNCORR_TRUE                1



typedef struct LW2080_ECC_AGGREGATE_ADDRESS {
    LwU8                     errorType;
    LW2080_ECC_UNITS         unit;
    LW2080_ECC_LOCATION_INFO location;
    union {
        LwU32                rawAddress;

        ECC_DRAM_ADDRESS_RBC rbc;
    } address;
} LW2080_ECC_AGGREGATE_ADDRESS;

#define LW2080_CTRL_ECC_GET_AGGREGATE_ERROR_ADDRESSES_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW2080_CTRL_ECC_GET_AGGREGATE_ERROR_ADDRESSES_PARAMS {
    LwU8                         flags;
    LwU32                        addressCount;
    LW2080_ECC_AGGREGATE_ADDRESS entry[LW2080_ECC_AGGREGATE_ADDRESS_MAX_COUNT];
} LW2080_CTRL_ECC_GET_AGGREGATE_ERROR_ADDRESSES_PARAMS;

/*
 * LW2080_CTRL_CMD_ECC_GET_LATEST_ECC_ADDRESSES
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
 *     added to the list since driver initialization. If the total
 *     number of addresses is greater than
 *     LW2080_CTRL_ECC_GET_LATEST_ECC_ADDRESSES_MAX_COUNT,
 *     the earliest addresses are discarded from the list.
 *
 *   addresses
 *     This parameter stores the RBC (Row/Bank/Column) address. See
 *     documentation for LW2080_CTRL_ECC_GET_LATEST_ECC_ADDRESSES_ADDRESS.
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_NOT_SUPPORTED
 *
 */

#define LW2080_CTRL_CMD_ECC_GET_LATEST_ECC_ADDRESSES               (0x20803407) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_ECC_INTERFACE_ID << 8) | LW2080_CTRL_ECC_GET_LATEST_ECC_ADDRESSES_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_ECC_GET_LATEST_ECC_ADDRESSES_MAX_COUNT         (32)

#define LW2080_CTRL_ECC_GET_LATEST_ECC_ADDRESSES_ERROR_TYPE        0:0
#define LW2080_CTRL_ECC_GET_LATEST_ECC_ADDRESSES_ERROR_TYPE_SBE    (0x00000000)
#define LW2080_CTRL_ECC_GET_LATEST_ECC_ADDRESSES_ERROR_TYPE_DBE    (0x00000001)

#define LW2080_CTRL_ECC_GET_LATEST_ECC_ADDRESSES_ERROR_SRC         0:0
#define LW2080_CTRL_ECC_GET_LATEST_ECC_ADDRESSES_ERROR_SRC_DEFAULT (0x00000000)
#define LW2080_CTRL_ECC_GET_LATEST_ECC_ADDRESSES_ERROR_SRC_DATA    (0x00000000)
#define LW2080_CTRL_ECC_GET_LATEST_ECC_ADDRESSES_ERROR_SRC_CHECK   (0x00000001)

/*
 * LW2080_CTRL_ECC_GET_LATEST_ECC_ADDRESSES_ADDRESS
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
 *   rbcAddressExt2
 *     This parameter is additional information about the address reported in
 *     rbcAddress and rbcAddressExt
 *
 *   partitionNumber
 *     This parameter is partition number in which the ECC error oclwrred.
 *
 *   subPartitionNumber
 *     This parameter is the subpartition number in which the ECC error
 *     oclwrred.
 *
 *   errorType
 *     This parameter is the type of ECC error that oclwrred, either single
 *     bit error (SBE) or double bit error (DBE).
 *
 *   sbeBitPosition
 *     This parameter is the corrected bit position for SBE
 *     Supported on VOLTA+, do not use otherwise
 *
 *   sbeErrorSrc
 *     NOTE: Not supported yet, do not use
 *     This parameter indicates whether the SBE is from the ECC check bits or
 *     the actual data.
 *     0 - ECC error from data
 *     1 - ECC error from check-bits
 *
 *   eccDiag
 *      This is a multibit field. The caller is responsible for extracting
 *      the bits of interest.
 */
typedef struct LW2080_CTRL_ECC_GET_LATEST_ECC_ADDRESSES_ADDRESS {
    LW_DECLARE_ALIGNED(LwU64 physAddress, 8);
    LwU32 rbcAddress;
    LwU32 rbcAddressExt;
    LwU32 rbcAddressExt2;
    LwU32 partitionNumber;
    LwU32 subPartitionNumber;
    LwU32 errorType;
    LwU32 sbeBitPosition;
    LwU32 sbeErrorSrc;
    LwU32 eccDiag;
} LW2080_CTRL_ECC_GET_LATEST_ECC_ADDRESSES_ADDRESS;

#define LW2080_CTRL_ECC_GET_LATEST_ECC_ADDRESSES_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW2080_CTRL_ECC_GET_LATEST_ECC_ADDRESSES_PARAMS {
    LwU32 addressCount;
    LwU32 totalAddressCount;
    LW_DECLARE_ALIGNED(LW2080_CTRL_ECC_GET_LATEST_ECC_ADDRESSES_ADDRESS addresses[LW2080_CTRL_ECC_GET_LATEST_ECC_ADDRESSES_MAX_COUNT], 8);
} LW2080_CTRL_ECC_GET_LATEST_ECC_ADDRESSES_PARAMS;

#define LW2080_CTRL_CMD_ECC_GET_CLIENT_EXPOSED_COUNTERS (0x20803408) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_ECC_INTERFACE_ID << 8) | LW2080_CTRL_ECC_GET_CLIENT_EXPOSED_COUNTERS_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_ECC_GET_CLIENT_EXPOSED_COUNTERS_PARAMS
 *
 * sramLastClearedTimestamp [out]
 * dramLastClearedTimestamp [out]
 *      unix-epoch based timestamp. These fields indicate when the error counters
 *      were last cleared by the user.
 *
 * sramErrorCounts [out]
 * dramErrorCounts [out]
 *      Aggregate error counts for SRAM and DRAM
 */

#define LW2080_CTRL_ECC_GET_CLIENT_EXPOSED_COUNTERS_PARAMS_MESSAGE_ID (0x8U)

typedef struct LW2080_CTRL_ECC_GET_CLIENT_EXPOSED_COUNTERS_PARAMS {
    LwU32 sramLastClearedTimestamp;
    LwU32 dramLastClearedTimestamp;

    LW_DECLARE_ALIGNED(LwU64 sramCorrectedTotalCounts, 8);
    LW_DECLARE_ALIGNED(LwU64 sramUncorrectedTotalCounts, 8);
    LW_DECLARE_ALIGNED(LwU64 dramCorrectedTotalCounts, 8);
    LW_DECLARE_ALIGNED(LwU64 dramUncorrectedTotalCounts, 8);
} LW2080_CTRL_ECC_GET_CLIENT_EXPOSED_COUNTERS_PARAMS;
/* _ctrl2080ecc_h_ */
