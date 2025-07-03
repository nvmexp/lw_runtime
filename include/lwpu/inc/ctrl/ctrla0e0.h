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
// Source file: ctrl/ctrla0e0.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#include "ctrl/ctrl90e0.h"          // A0E0 is partially derived from 90E0
#define LWA0E0_CTRL_CMD(cat,idx)  \
    LWXXXX_CTRL_CMD(0xA0E0, LWA0E0_CTRL_##cat, idx)


/* LWA0E0 command categories (6bits) */
#define LWA0E0_CTRL_RESERVED (0x00)
#define LWA0E0_CTRL_GRAPHICS (0x01)


/*
 * LWA0E0_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LWA0E0_CTRL_CMD_NULL (0xa0e00000) /* finn: Evaluated from "(FINN_GK110_SUBDEVICE_GRAPHICS_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*
 * LWA0E0_CTRL_CMD_GR_GET_ECC_COUNTS
 *
 * This command is used to query detailed ECC and parity counts for each of the
 * units in GRAPHICS since the last reboot. It covers L1C (level 1 cache)
 * and SM RF (register file) errors.
 *
 *   gpcCount (in/out)
 *     On input, this parameter specifies how many GPCs the
 *     passed in structure can support. On output, this parameter
 *     specifies how many GPCs of data was stored to the structure.
 *
 *   tpcCount (in/out)
 *     On input, this parameter specifies how many TPCs per GPC the
 *     passed in structure can support. On output, this parameter
 *     specifies how many TPCs per GPC of data was stored to the
 *     structure.
 *
 *   texCount (in/out)
 *     On input, this parameter specifies how many Texture units per
 *     TPC the passed in structure can support. On output, this parameter
 *     specifies how many Texture units per TPC of data was stored to the
 *     structure.
 *
 *   gpcEcc (out)
 *     This parameter stores the L1C and RF ECC single- and double-bit
 *     error counts and the Texture correctable and uncorrectable counts.
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 *
 */
#define LWA0E0_CTRL_CMD_GR_GET_ECC_COUNTS (0xa0e00101) /* finn: Evaluated from "(FINN_GK110_SUBDEVICE_GRAPHICS_GRAPHICS_INTERFACE_ID << 8) | LWA0E0_CTRL_GR_GET_ECC_COUNTS_PARAMS_MESSAGE_ID" */

#define LWA0E0_CTRL_GR_ECC_GPC_COUNT      (0x00000005)
#define LWA0E0_CTRL_GR_ECC_TPC_COUNT      (0x00000003)
#define LWA0E0_CTRL_GR_ECC_TEX_COUNT      (0x00000004)

/*
 * LWA0E0_CTRL_GPU_QUERY_ECC_EXCEPTION_STATUS
 *
 * This structure represents the exception status of a class of per-unit
 * exceptions
 *
 *   count
 *     This parameter constains the number of exceptions that have oclwrred
 *     since boot.
 *
 */
typedef LW90E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS LWA0E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS;


typedef struct LWA0E0_CTRL_GR_GET_ECC_TEX_COUNTS {
    LW_DECLARE_ALIGNED(LWA0E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS correctable, 8);
    LW_DECLARE_ALIGNED(LWA0E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS unCorrectable, 8);
} LWA0E0_CTRL_GR_GET_ECC_TEX_COUNTS;

/*
 * LWA0E0_CTRL_GR_GET_ECC_TPC_COUNTS
 *
 * This structure contains the number of single-bit and double-bit
 * exceptions that have oclwrred in a TPC.
 *
 *   l1Sbe
 *     This parameter contains the number of l1 single-bit exceptions.
 *
 *   l1Dbe
 *     This parameter contains the number of l1 double-bit exceptions.
 *
 *   rfSbe
 *     This parameter contains the number of register file single-bit
 *     exceptions.
 *
 *   rfDbe
 *     This parameter contains the number of register file double-bit
 *     exceptions.
 *
 *   tex
 *     This parameter contains the number of correctable and uncorrectable
 *     parity errors for each texture unit.
 *
 */
typedef struct LWA0E0_CTRL_GR_GET_ECC_TPC_COUNTS {
    LW_DECLARE_ALIGNED(LWA0E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS l1Sbe, 8);
    LW_DECLARE_ALIGNED(LWA0E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS l1Dbe, 8);
    LW_DECLARE_ALIGNED(LWA0E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS rfSbe, 8);
    LW_DECLARE_ALIGNED(LWA0E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS rfDbe, 8);
    LW_DECLARE_ALIGNED(LWA0E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS shmSbe, 8);
    LW_DECLARE_ALIGNED(LWA0E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS shmDbe, 8);
    LW_DECLARE_ALIGNED(LWA0E0_CTRL_GR_GET_ECC_TEX_COUNTS tex[LWA0E0_CTRL_GR_ECC_TEX_COUNT], 8);
} LWA0E0_CTRL_GR_GET_ECC_TPC_COUNTS;

typedef LWA0E0_CTRL_GR_GET_ECC_TPC_COUNTS LWA0E0_CTRL_GR_GET_ECC_GPC_COUNTS[LWA0E0_CTRL_GR_ECC_TPC_COUNT];


#define LWA0E0_CTRL_GR_GET_ECC_COUNTS_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWA0E0_CTRL_GR_GET_ECC_COUNTS_PARAMS {
    LwU32 tpcCount;
    LwU32 gpcCount;
    LwU32 texCount;

    LW_DECLARE_ALIGNED(LWA0E0_CTRL_GR_GET_ECC_GPC_COUNTS ecc[LWA0E0_CTRL_GR_ECC_GPC_COUNT], 8);
} LWA0E0_CTRL_GR_GET_ECC_COUNTS_PARAMS;

/*
 * LWA0E0_CTRL_CMD_GR_GET_AGGREGATE_ECC_COUNTS
 *
 * This command is used to query detailed ECC and parity counts for each of the
 * units in GRAPHICS for the lifetime of a given GPU, to the extent the RM
 * can determine these values. It covers L1C (level 1 cache) and SM RF
 * (register file) errors.
 *
 *   gpcEcc
 *     This parameter stores the L1C and RF ECC single- and double-bit
 *     error counts.
 *
 * Possible return status values are
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *
 */
#define LWA0E0_CTRL_CMD_GR_GET_AGGREGATE_ECC_COUNTS (0xa0e00102) /* finn: Evaluated from "(FINN_GK110_SUBDEVICE_GRAPHICS_GRAPHICS_INTERFACE_ID << 8) | LWA0E0_CTRL_GR_GET_AGGREGATE_ECC_COUNTS_PARAMS_MESSAGE_ID" */

#define LWA0E0_CTRL_GR_GET_AGGREGATE_ECC_COUNTS_PARAMS_MESSAGE_ID (0x2U)

typedef struct LWA0E0_CTRL_GR_GET_AGGREGATE_ECC_COUNTS_PARAMS {
    LW_DECLARE_ALIGNED(LWA0E0_CTRL_GR_GET_ECC_GPC_COUNTS ecc[LWA0E0_CTRL_GR_ECC_GPC_COUNT], 8);
} LWA0E0_CTRL_GR_GET_AGGREGATE_ECC_COUNTS_PARAMS;

/*
 * LWA0E0_CTRL_CMD_GR_QUERY_ECC_CAPABILITIES
 *
 * Please see description of LW90E0_CTRL_CMD_GR_QUERY_ECC_CAPABILITIES for
 * more information.
 *
 */
#define LWA0E0_CTRL_CMD_GR_QUERY_ECC_CAPABILITIES (0xa0e00103) /* finn: Evaluated from "(FINN_GK110_SUBDEVICE_GRAPHICS_GRAPHICS_INTERFACE_ID << 8) | 0x3" */

typedef LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_PARAMS LWA0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LWA0E0_CTRL_CMD_GR_QUERY_ECC_CAPABILITIES_FINN_PARAMS_MESSAGE_ID (0x3U)

typedef struct LWA0E0_CTRL_CMD_GR_QUERY_ECC_CAPABILITIES_FINN_PARAMS {
    LWA0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_PARAMS params;
} LWA0E0_CTRL_CMD_GR_QUERY_ECC_CAPABILITIES_FINN_PARAMS;



/* Legal flag parameter values */
#define LWA0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_L1C             LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_L1C
#define LWA0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_L1C_NOECC    LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_L1C_NOECC
#define LWA0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_L1C_ECC      LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_L1C_ECC

#define LWA0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SM              LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SM
#define LWA0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SM_NOECC     LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SM_NOECC
#define LWA0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SM_ECC       LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SM_ECC

#define LWA0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_TEX             LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_TEX
#define LWA0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_TEX_NOPARITY LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_TEX_NOPARITY
#define LWA0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_TEX_PARITY   LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_TEX_PARITY
#define LWA0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_TEX_NOECC    LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_TEX_NOECC
#define LWA0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_TEX_ECC      LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_TEX_ECC

#define LWA0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SHM             LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SHM
#define LWA0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SHM_NOECC    LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SHM_NOECC
#define LWA0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SHM_ECC      LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SHM_ECC

/*
 * LWA0E0_CTRL_CMD_GR_GET_L1C_ECC_COUNTS
 *
 * This command is used to query detailed L1C (level 1 Cache) ECC counts for
 * each of unit in GRAPHICS since the last reboot or driver load.  A
 * simple array of errors is returned including the GPC and TPC number
 * of the affected unit.
 *
 *   flags (out)
 *     This parameter specifies whether or not all units with ECC errors
 *     fit into the l1cEcc array.
 * 
 *   entryCount (out)
 *     This parameter specifies the number of entries in the list which
 *     contain counts. The range will be between 0 and
 *     LWA0E0_CTRL_GR_LTC_ECC_COUNT.
 *
 *   l1cEcc (out)
 *     This parameter stores the L1C ECC single- and double-bit
 *     error counts.
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 *
 */
#define LWA0E0_CTRL_CMD_GR_GET_L1C_ECC_COUNTS                    (0xa0e00104) /* finn: Evaluated from "(FINN_GK110_SUBDEVICE_GRAPHICS_GRAPHICS_INTERFACE_ID << 8) | LWA0E0_CTRL_GR_GET_L1C_ECC_COUNTS_PARAMS_MESSAGE_ID" */

#define LWA0E0_CTRL_GR_L1C_ECC_MAX_COUNT                         (256)

typedef struct LWA0E0_CTRL_GR_GET_L1C_ECC_COUNTS {
    LwU16 gpc;
    LwU16 tpc;
    LwU32 sbe;
    LwU32 dbe;
} LWA0E0_CTRL_GR_GET_L1C_ECC_COUNTS;

#define LWA0E0_CTRL_GR_GET_ECC_COUNTS_FLAGS_OVERFLOW         0:0
#define LWA0E0_CTRL_GR_GET_ECC_COUNTS_FLAGS_OVERFLOW_FALSE (0x00000000)
#define LWA0E0_CTRL_GR_GET_ECC_COUNTS_FLAGS_OVERFLOW_TRUE  (0x00000001)

#define LWA0E0_CTRL_GR_GET_L1C_ECC_COUNTS_PARAMS_MESSAGE_ID (0x4U)

typedef struct LWA0E0_CTRL_GR_GET_L1C_ECC_COUNTS_PARAMS {
    LwU32                             flags;
    LwU32                             entryCount;
    LWA0E0_CTRL_GR_GET_L1C_ECC_COUNTS l1cEcc[LWA0E0_CTRL_GR_L1C_ECC_MAX_COUNT];
} LWA0E0_CTRL_GR_GET_L1C_ECC_COUNTS_PARAMS;

/*
 * LWA0E0_CTRL_CMD_GR_GET_RF_ECC_COUNTS
 *
 * This command is used to query detailed RF (register file) ECC for
 * each of unit in GRAPHICS since the last reboot or driver load.  A
 * simple array of errors is returned including the GPC and TPC number
 * of the affected unit.
 *
 *   flags (out)
 *     This parameter specifies whether or not all units with ECC errors
 *     fit into the rfEcc array.
 *
 *   entryCount (out)
 *     This parameter specifies the number of entries in the list which
 *     contain counts. The range will be between 0 and
 *     LWA0E0_CTRL_GR_RF_ECC_COUNTS.
 *
 *   rfEcc (out)
 *     This parameter stores the RF ECC single- and double-bit
 *     error counts.
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 *
 */
#define LWA0E0_CTRL_CMD_GR_GET_RF_ECC_COUNTS (0xa0e00105) /* finn: Evaluated from "(FINN_GK110_SUBDEVICE_GRAPHICS_GRAPHICS_INTERFACE_ID << 8) | LWA0E0_CTRL_GR_GET_RF_ECC_COUNTS_PARAMS_MESSAGE_ID" */

#define LWA0E0_CTRL_GR_RF_ECC_MAX_COUNT      (256)

typedef struct LWA0E0_CTRL_GR_GET_RF_ECC_COUNTS {
    LwU16 gpc;
    LwU16 tpc;
    LW_DECLARE_ALIGNED(LwU64 sbe, 8);
    LW_DECLARE_ALIGNED(LwU64 dbe, 8);
} LWA0E0_CTRL_GR_GET_RF_ECC_COUNTS;

#define LWA0E0_CTRL_GR_GET_RF_ECC_COUNTS_PARAMS_MESSAGE_ID (0x5U)

typedef struct LWA0E0_CTRL_GR_GET_RF_ECC_COUNTS_PARAMS {
    LwU32 flags;
    LwU32 entryCount;
    LW_DECLARE_ALIGNED(LWA0E0_CTRL_GR_GET_RF_ECC_COUNTS rfEcc[LWA0E0_CTRL_GR_RF_ECC_MAX_COUNT], 8);
} LWA0E0_CTRL_GR_GET_RF_ECC_COUNTS_PARAMS;

/*
 * LWA0E0_CTRL_CMD_GR_GET_TEX_ERROR_COUNTS
 *
 * This command is used to query detailed TEX (texture) parity for
 * each of unit in GRAPHICS since the last reboot or driver load.  A
 * simple array of errors is returned including the GPC, TPC, and TEX number
 * of the affected unit.
 *
 *   flags (out)
 *     This parameter specifies whether or not all units with ECC errors
 *     fit into the l1cEcc array.
 *
 *   entryCount (out)
 *     This parameter specifies the number of entries in the list which
 *     contain counts. The range will be between 0 and
 *     LWA0E0_CTRL_GR_LTC_ECC_COUNT.
 *
 *   texEcc (out)
 *     This parameter stores the TEX correctable and uncorrectable parity
 *     error counts.
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 *
 */
#define LWA0E0_CTRL_CMD_GR_GET_TEX_ERROR_COUNTS (0xa0e00106) /* finn: Evaluated from "(FINN_GK110_SUBDEVICE_GRAPHICS_GRAPHICS_INTERFACE_ID << 8) | LWA0E0_CTRL_GR_GET_TEX_ERROR_COUNTS_PARAMS_MESSAGE_ID" */

#define LWA0E0_CTRL_GR_TEX_MAX_ERROR_COUNT      (200)

typedef struct LWA0E0_CTRL_GR_GET_TEX_ERROR_COUNTS {
    LwU16 gpc;
    LwU16 tpc;
    LwU16 tex;
    LwU32 correctable;
    LwU32 unCorrectable;
} LWA0E0_CTRL_GR_GET_TEX_ERROR_COUNTS;

#define LWA0E0_CTRL_GR_GET_TEX_ERROR_COUNTS_PARAMS_MESSAGE_ID (0x6U)

typedef struct LWA0E0_CTRL_GR_GET_TEX_ERROR_COUNTS_PARAMS {
    LwU32                               flags;
    LwU32                               entryCount;
    LWA0E0_CTRL_GR_GET_TEX_ERROR_COUNTS texEcc[LWA0E0_CTRL_GR_TEX_MAX_ERROR_COUNT];
} LWA0E0_CTRL_GR_GET_TEX_ERROR_COUNTS_PARAMS;

/*
 * LWA0E0_CTRL_CMD_GR_GET_AGGREGATE_L1C_ECC_COUNTS
 *
 * This command is used to query detailed aggregate L1C (level 1 Cache) ECC
 * counts for each of the units in GRAPHICS. Due to geometry changes
 * between different generations of chips, the LWA0E0_CTRL_CMD_GR_GET_ECC_COUNTS
 * would not be able to include all the counts from new chips. By changing the
 * array, from a fixed two dimensional array, to a single dimensional array and
 * explicitly including the GPC and TPC number of the affected unit, most
 * changes in geometry will be handled properly.
 *
 *   flags (out)
 *     This parameter specifies whether or not all units with ECC errors
 *     fit into the l1cEcc array.
 *
 *   entryCount (out)
 *     This parameter specifies the number of entries in the list which
 *     contain counts. The range will be between 0 and
 *     LWA0E0_CTRL_GR_LTC_ECC_COUNT.
 *
 *   l1cEcc (out)
 *     This parameter stores the L1C ECC single- and double-bit
 *     error counts.
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 *
 */
#define LWA0E0_CTRL_CMD_GR_GET_AGGREGATE_L1C_ECC_COUNTS (0xa0e00107) /* finn: Evaluated from "(FINN_GK110_SUBDEVICE_GRAPHICS_GRAPHICS_INTERFACE_ID << 8) | 0x7" */


typedef LWA0E0_CTRL_GR_GET_L1C_ECC_COUNTS_PARAMS LWA0E0_CTRL_GR_GET_AGGREGATE_L1C_ECC_COUNTS_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LWA0E0_CTRL_CMD_GR_GET_AGGREGATE_L1C_ECC_COUNTS_FINN_PARAMS_MESSAGE_ID (0x7U)

typedef struct LWA0E0_CTRL_CMD_GR_GET_AGGREGATE_L1C_ECC_COUNTS_FINN_PARAMS {
    LWA0E0_CTRL_GR_GET_AGGREGATE_L1C_ECC_COUNTS_PARAMS params;
} LWA0E0_CTRL_CMD_GR_GET_AGGREGATE_L1C_ECC_COUNTS_FINN_PARAMS;



/*
 * LWA0E0_CTRL_CMD_GR_GET_AGGREGATE_RF_ECC_COUNTS
 *
 * This command is used to query detailed aggregate RF (register file) ECC
 * counts for each of each of the units in GRAPHICS. Due to geometry changes
 * between different generations of chips, the LWA0E0_CTRL_CMD_GR_GET_ECC_COUNTS
 * would not be able to include all the counts from new chips. By changing the
 * array, from a fixed two dimensional array, to a single dimensional array and
 * explicitly including the GPC and TPC number of the affected unit.
 *
 *   flags (out)
 *     This parameter specifies whether or not all units with ECC errors
 *     fit into the l1cEcc array.
 *
 *   entryCount (out)
 *     This parameter specifies the number of entries in the list which
 *     contain counts. The range will be between 0 and
 *     LWA0E0_CTRL_GR_LTC_ECC_COUNT.
 *
 *   rfEcc (out)
 *     This parameter stores the RF ECC single- and double-bit
 *     error counts.
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 *
 */
#define LWA0E0_CTRL_CMD_GR_GET_AGGREGATE_RF_ECC_COUNTS (0xa0e00108) /* finn: Evaluated from "(FINN_GK110_SUBDEVICE_GRAPHICS_GRAPHICS_INTERFACE_ID << 8) | 0x8" */

typedef LWA0E0_CTRL_GR_GET_RF_ECC_COUNTS_PARAMS LWA0E0_CTRL_GR_GET_AGGREGATE_RF_ECC_COUNTS_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LWA0E0_CTRL_CMD_GR_GET_AGGREGATE_RF_ECC_COUNTS_FINN_PARAMS_MESSAGE_ID (0x8U)

typedef struct LWA0E0_CTRL_CMD_GR_GET_AGGREGATE_RF_ECC_COUNTS_FINN_PARAMS {
    LW_DECLARE_ALIGNED(LWA0E0_CTRL_GR_GET_AGGREGATE_RF_ECC_COUNTS_PARAMS params, 8);
} LWA0E0_CTRL_CMD_GR_GET_AGGREGATE_RF_ECC_COUNTS_FINN_PARAMS;



/*
 * LWA0E0_CTRL_CMD_GR_GET_AGGREGATE_TEX_ERROR_COUNTS
 *
 * This command is used to query detailed aggregate TEX (texture)
 * counts for each of each of the units in GRAPHICS. Due to geometry changes
 * between different generations of chips, the LWA0E0_CTRL_CMD_GR_GET_ECC_COUNTS
 * would not be able to include all the counts from new chips. By changing the
 * array, from a fixed two dimensional array, to a single dimensional array and
 * explicitly including the GPC, TPC, and TEX number of the affected unit.
 *
 *   flags (out)
 *     This parameter specifies whether or not all units with ECC errors
 *     fit into the l1cEcc array.
 *
 *   entryCount (out)
 *     This parameter specifies the number of entries in the list which
 *     contain counts. The range will be between 0 and
 *     LWA0E0_CTRL_GR_LTC_ECC_COUNT.
 *
 *   texEcc (out)
 *     This parameter stores the TEX correctable and uncorrectable
 *     error counts.
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 *
 */
#define LWA0E0_CTRL_CMD_GR_GET_AGGREGATE_TEX_ERROR_COUNTS (0xa0e00109) /* finn: Evaluated from "(FINN_GK110_SUBDEVICE_GRAPHICS_GRAPHICS_INTERFACE_ID << 8) | 0x9" */

typedef LWA0E0_CTRL_GR_GET_TEX_ERROR_COUNTS_PARAMS LWA0E0_CTRL_GR_GET_AGGREGATE_TEX_ERROR_COUNTS_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LWA0E0_CTRL_CMD_GR_GET_AGGREGATE_TEX_ERROR_COUNTS_FINN_PARAMS_MESSAGE_ID (0x9U)

typedef struct LWA0E0_CTRL_CMD_GR_GET_AGGREGATE_TEX_ERROR_COUNTS_FINN_PARAMS {
    LWA0E0_CTRL_GR_GET_AGGREGATE_TEX_ERROR_COUNTS_PARAMS params;
} LWA0E0_CTRL_CMD_GR_GET_AGGREGATE_TEX_ERROR_COUNTS_FINN_PARAMS;



/*
 * LWA0E0_CTRL_CMD_GR_GET_SHM_ECC_COUNTS
 *
 * This command is used to query detailed SHM (shared memory) ECC for
 * each of unit in GRAPHICS since the last reboot or driver load.  A
 * simple array of errors is returned including the GPC and TPC number
 * of the affected unit.
 *
 *   flags (out)
 *     This parameter specifies whether or not all units with ECC errors
 *     fit into the shmEcc array.
 *
 *   entryCount (out)
 *     This parameter specifies the number of entries in the list which
 *     contain counts. The range will be between 0 and
 *     LWA0E0_CTRL_GR_SHM_ECC_COUNTS.
 *
 *   shmEcc (out)
 *     This parameter stores the SHM ECC single- and double-bit
 *     error counts.
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 *
 */
#define LWA0E0_CTRL_CMD_GR_GET_SHM_ECC_COUNTS (0xa0e0010a) /* finn: Evaluated from "(FINN_GK110_SUBDEVICE_GRAPHICS_GRAPHICS_INTERFACE_ID << 8) | LWA0E0_CTRL_GR_GET_SHM_ECC_COUNTS_PARAMS_MESSAGE_ID" */

#define LWA0E0_CTRL_GR_SHM_ECC_MAX_COUNT      (256)

typedef struct LWA0E0_CTRL_GR_GET_SHM_ECC_COUNTS {
    LwU16 gpc;
    LwU16 tpc;
    LW_DECLARE_ALIGNED(LwU64 sbe, 8);
    LW_DECLARE_ALIGNED(LwU64 dbe, 8);
} LWA0E0_CTRL_GR_GET_SHM_ECC_COUNTS;

#define LWA0E0_CTRL_GR_GET_SHM_ECC_COUNTS_PARAMS_MESSAGE_ID (0xAU)

typedef struct LWA0E0_CTRL_GR_GET_SHM_ECC_COUNTS_PARAMS {
    LwU32 flags;
    LwU32 entryCount;
    LW_DECLARE_ALIGNED(LWA0E0_CTRL_GR_GET_SHM_ECC_COUNTS shmEcc[LWA0E0_CTRL_GR_SHM_ECC_MAX_COUNT], 8);
} LWA0E0_CTRL_GR_GET_SHM_ECC_COUNTS_PARAMS;

/*
 * LWA0E0_CTRL_CMD_GR_GET_AGGREGATE_SHM_ECC_COUNTS
 *
 * This command is used to query detailed aggregate SHM (shared memory) ECC
 * counts for each of each of the units in GRAPHICS. Due to geometry changes
 * between different generations of chips, the LWA0E0_CTRL_CMD_GR_GET_ECC_COUNTS
 * would not be able to include all the counts from new chips. By changing the
 * array, from a fixed two dimensional array, to a single dimensional array and
 * explicitly including the GPC and TPC number of the affected unit.
 *
 *   flags (out)
 *     This parameter specifies whether or not all units with ECC errors
 *     fit into the shmEcc array.
 *
 *   entryCount (out)
 *     This parameter specifies the number of entries in the list which
 *     contain counts. The range will be between 0 and
 *     LWA0E0_CTRL_GR_SHM_ECC_COUNT.
 *
 *   rfEcc (out)
 *     This parameter stores the SHM ECC single- and double-bit
 *     error counts.
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 *
 */
#define LWA0E0_CTRL_CMD_GR_GET_AGGREGATE_SHM_ECC_COUNTS (0xa0e0010b) /* finn: Evaluated from "(FINN_GK110_SUBDEVICE_GRAPHICS_GRAPHICS_INTERFACE_ID << 8) | 0xB" */

typedef LWA0E0_CTRL_GR_GET_SHM_ECC_COUNTS_PARAMS LWA0E0_CTRL_GR_GET_AGGREGATE_SHM_ECC_COUNTS_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LWA0E0_CTRL_CMD_GR_GET_AGGREGATE_SHM_ECC_COUNTS_FINN_PARAMS_MESSAGE_ID (0xBU)

typedef struct LWA0E0_CTRL_CMD_GR_GET_AGGREGATE_SHM_ECC_COUNTS_FINN_PARAMS {
    LW_DECLARE_ALIGNED(LWA0E0_CTRL_GR_GET_AGGREGATE_SHM_ECC_COUNTS_PARAMS params, 8);
} LWA0E0_CTRL_CMD_GR_GET_AGGREGATE_SHM_ECC_COUNTS_FINN_PARAMS;



/* _ctrla0e0_h_ */

