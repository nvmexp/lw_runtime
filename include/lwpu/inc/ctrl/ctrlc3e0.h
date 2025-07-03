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
// Source file: ctrl/ctrlc3e0.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#include "ctrl/ctrlc0e0.h"          // C3E0 is partially derived from C0E0
#define LWC3E0_CTRL_CMD(cat,idx)  \
    LWXXXX_CTRL_CMD(0xC3E0, LWC3E0_CTRL_##cat, idx)

/* LWC3E0 command categories (6bits) */
#define LWC3E0_CTRL_RESERVED (0x00)
#define LWC3E0_CTRL_GRAPHICS (0x01)

/*
 * The following define, enum and associated structure will be moved to the new ECC 2080 class files
 * once it is created
 */
#define ECC_ERRORS_MAX_COUNT 168

typedef enum eccErrorLoc {
    eccLocilwalid = 0,
    eccLoclrf = 1,
    eccLoccbu = 2,
    eccLocl1Data = 3,
    eccLocl1Tag = 4,
    eccLocltc = 5,
    eccLocdram = 6,
} eccErrorLoc;

typedef struct ECC_ERROR_ENTRY {
    eccErrorLoc errorLocation;

    //
    // In case of LTC or DRAM errors
    // Location -> partition number, subLocation -> slice/sub-partition number
    // For other error types
    // Location -> GPC number, sub-location -> TPC number
    //
    LwU32       location;
    LwU32       subLocation;

    //
    // For Volta DRAM and LTC uncorrected corresponds to DED and corrected corresponds
    // to SEC errors
    //
    LW_DECLARE_ALIGNED(LwU64 uncorrectedTotal, 8);
    LW_DECLARE_ALIGNED(LwU64 uncorrectedUnique, 8);
    LW_DECLARE_ALIGNED(LwU64 correctedTotal, 8);
    LW_DECLARE_ALIGNED(LwU64 correctedUnique, 8);
} ECC_ERROR_ENTRY;

typedef struct ECC_ERROR_PARAMS {
    LwU32 entryCount;
    LW_DECLARE_ALIGNED(ECC_ERROR_ENTRY eccErrors[ECC_ERRORS_MAX_COUNT], 8);
} ECC_ERROR_PARAMS;

/*
 * LWC3E0_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LWC3E0_CTRL_CMD_NULL (0xc3e00000) /* finn: Evaluated from "(FINN_GV100_SUBDEVICE_GRAPHICS_RESERVED_INTERFACE_ID << 8) | 0x0" */





#define LWC3E0_CTRL_CMD_GR_GET_ECC_COUNTS (0xc3e00101) /* finn: Evaluated from "(FINN_GV100_SUBDEVICE_GRAPHICS_GRAPHICS_INTERFACE_ID << 8) | LWC3E0_CTRL_GR_GET_ECC_COUNTS_PARAMS_MESSAGE_ID" */

#define LWC3E0_CTRL_GR_ECC_GPC_COUNT      (0x00000006)
#define LWC3E0_CTRL_GR_ECC_TPC_COUNT      (0x00000007)

typedef LWC0E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS LWC3E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS;

/*
 * LWC3E0_CTRL_GR_GET_ECC_TPC_COUNTS
 *
 * This structure contains the number of corrected and uncorrected
 * exceptions that have oclwrred in a TPC.
 *
 *      Total vs Unique
 *          HW reports ECC counts, but due to multiple errors in the same location
 *          repeatedly the total counter would log all those errors whereas the
 *          unique counter would only increment by 1.
 *
 *      rf*
 *          Holds error counts for the register file.
 *
 *      cbu*
 *          Holds error counts for the CBU unit which has memory protection for Volta.
 *
 *      l1Data*
 *          Holds error counts for the L1 Data Cache Data memory.
 *
 *      l1Tag*
 *          Holds error counts for the L1 Data Cache Tag memory.
 *
 */

typedef struct LWC3E0_CTRL_GR_GET_ECC_TPC_COUNTS {

    LW_DECLARE_ALIGNED(LWC3E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS rfUncorrectedTotal, 8);
    LW_DECLARE_ALIGNED(LWC3E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS rfUncorrectedUnique, 8);
    LW_DECLARE_ALIGNED(LWC3E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS rfCorrectedTotal, 8);
    LW_DECLARE_ALIGNED(LWC3E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS rfCorrectedUnique, 8);

    LW_DECLARE_ALIGNED(LWC3E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS cbuUncorrectedTotal, 8);
    LW_DECLARE_ALIGNED(LWC3E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS cbuUncorrectedUnique, 8);

    LW_DECLARE_ALIGNED(LWC3E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS l1DataUncorrectedTotal, 8);
    LW_DECLARE_ALIGNED(LWC3E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS l1DataUncorrectedUnique, 8);
    LW_DECLARE_ALIGNED(LWC3E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS l1DataCorrectedTotal, 8);
    LW_DECLARE_ALIGNED(LWC3E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS l1DataCorrectedUnique, 8);

    LW_DECLARE_ALIGNED(LWC3E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS l1TagUncorrectedTotal, 8);
    LW_DECLARE_ALIGNED(LWC3E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS l1TagUncorrectedUnique, 8);
    LW_DECLARE_ALIGNED(LWC3E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS l1TagCorrectedTotal, 8);
    LW_DECLARE_ALIGNED(LWC3E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS l1TagCorrectedUnique, 8);
} LWC3E0_CTRL_GR_GET_ECC_TPC_COUNTS;

typedef LWC3E0_CTRL_GR_GET_ECC_TPC_COUNTS LWC3E0_CTRL_GR_GET_ECC_GPC_COUNTS[LWC3E0_CTRL_GR_ECC_TPC_COUNT];

/*
 * LWC3E0_CTRL_GR_GET_ECC_COUNTS_PARAMS
 *
 * This structure contains the number of corrected and uncorrected
 * exceptions that have oclwrred in a GPC.
 *
 *      tpcCount[out]
 *         Holds the number of valid entries in the tpc array.
 *
 *      gpcCount[out]
 *         Holds the number of valid entries in the gpc array.
 *
 *      flags[out]
 *
 *      gpcEccCounts[out]
 *         Refer LWC3E0_CTRL_GR_GET_ECC_TPC_COUNTS.
 *
 */

#define LWC3E0_CTRL_GR_GET_ECC_COUNTS_FLAGS_OVERFLOW         0:0
#define LWC3E0_CTRL_GR_GET_ECC_COUNTS_FLAGS_OVERFLOW_FALSE 0
#define LWC3E0_CTRL_GR_GET_ECC_COUNTS_FLAGS_OVERFLOW_TRUE  1

#define LWC3E0_CTRL_GR_GET_ECC_COUNTS_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWC3E0_CTRL_GR_GET_ECC_COUNTS_PARAMS {
    LwU32 tpcCount;
    LwU32 gpcCount;

    LwU8  flags;

    LW_DECLARE_ALIGNED(LWC3E0_CTRL_GR_GET_ECC_GPC_COUNTS gpcEccCounts[LWC3E0_CTRL_GR_ECC_GPC_COUNT], 8);
} LWC3E0_CTRL_GR_GET_ECC_COUNTS_PARAMS;

/* _ctrlc3e0_h_ */
