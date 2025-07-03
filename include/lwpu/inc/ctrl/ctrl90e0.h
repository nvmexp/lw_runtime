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
// Source file: ctrl/ctrl90e0.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#define LW90E0_CTRL_CMD(cat,idx)  \
    LWXXXX_CTRL_CMD(0x90E0, LW90E0_CTRL_##cat, idx)


/* LW90E0 command categories (6bits) */
#define LW90E0_CTRL_RESERVED (0x00)
#define LW90E0_CTRL_GRAPHICS (0x01)


/*
 * LW90E0_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LW90E0_CTRL_CMD_NULL (0x90e00000) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_GRAPHICS_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*
 * LW90E0_CTRL_CMD_GR_GET_ECC_COUNTS
 *
 * This command is used to query detailed ECC and EDC counts for each of the
 * units in GRAPHICS since the last reboot. It covers L1C (level 1 cache)
 * and SM RF (register file) errors.
 *
 * smCount (in/out)
 *   Input:  this parameter specifies how many SMs per TPC the
 *           passed in structure can support.
 *   Output: this parameter specifies how many SMs of data
 *           was stored to the structure
 *
 * tpcCount (in/out)
 *   Input:  this parameter specifies how many TPC the
 *           passed in structure can support.
 *   Output: this parameter specifies how many TPCs of data
 *           was stored to the structure
 *
 * gpcEcc
 *   This parameter stores the L1C and RF ECC single- and double-bit
 *   error counts.
 *
 * status
 *   This paramater signifies if the passed in structure was too small to store
 *   all the values.  Bit-wise status flags:
 *        LW90E0_CTRL_GET_ECC_STATUS_SM_OVERFLOW                (0x00000001)
 *        LW90E0_CTRL_GET_ECC_STATUS_TPC_OVERFLOW               (0x00000002)
 *
 * Possible return status values are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *
 */
#define LW90E0_CTRL_CMD_GR_GET_ECC_COUNTS (0x90e00101) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_GRAPHICS_GRAPHICS_INTERFACE_ID << 8) | LW90E0_CTRL_GR_GET_ECC_COUNTS_PARAMS_MESSAGE_ID" */

#define LW90E0_CTRL_GR_ECC_GPC_COUNT      (0x00000006)
#define LW90E0_CTRL_GR_ECC_TPC_COUNT      (0x00000005)


/*
 * LW90E0_CTRL_GPU_QUERY_ECC_EXCEPTION_STATUS
 *
 * This structure represents the exception status of a class of per-unit
 * exceptions
 *
 *   count
 *     number of exceptions that have oclwrred since boot
 */
typedef struct LW90E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS {
    LW_DECLARE_ALIGNED(LwU64 count, 8);
} LW90E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS;


/*
 * LW90E0_CTRL_GR_GET_ECC_TPC_COUNTS
 *
 * This structure contains the number of single-bit and double-bit
 * exceptions that have oclwrred in a TPC.
 *
 *   l1Sbe
 *     number of l1 single-bit exceptions
 *
 *   l1Dbe
 *     number of l1 double-bit exceptions
 *
 *   rfSbe
 *     number of register file single-bit exceptions
 *
 *   rfDbe
 *     number of register file double-bit exceptions
 *
 *   shmSbe
 *     number of shared memory single-bit exceptions
 *
 *   shmDbe
 *     number of shared memory double-bit exceptions
 */
typedef struct LW90E0_CTRL_GR_GET_ECC_TPC_COUNTS {
    LW_DECLARE_ALIGNED(LW90E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS l1Sbe, 8);
    LW_DECLARE_ALIGNED(LW90E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS l1Dbe, 8);
    LW_DECLARE_ALIGNED(LW90E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS rfSbe, 8);
    LW_DECLARE_ALIGNED(LW90E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS rfDbe, 8);
    LW_DECLARE_ALIGNED(LW90E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS shmSbe, 8);
    LW_DECLARE_ALIGNED(LW90E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS shmDbe, 8);
} LW90E0_CTRL_GR_GET_ECC_TPC_COUNTS;

typedef LW90E0_CTRL_GR_GET_ECC_TPC_COUNTS LW90E0_CTRL_GR_GET_ECC_GPC_COUNTS[LW90E0_CTRL_GR_ECC_TPC_COUNT];


#define LW90E0_CTRL_GR_GET_ECC_COUNTS_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW90E0_CTRL_GR_GET_ECC_COUNTS_PARAMS {
    LwU32 tpcCount;
    LwU32 gpcCount;

    LW_DECLARE_ALIGNED(LW90E0_CTRL_GR_GET_ECC_GPC_COUNTS gpcEcc[LW90E0_CTRL_GR_ECC_GPC_COUNT], 8);
} LW90E0_CTRL_GR_GET_ECC_COUNTS_PARAMS;

/*
 * LW90E0_CTRL_CMD_GR_GET_AGGREGATE_ECC_COUNTS
 *
 * This command is used to query detailed ECC and EDC counts for each of the
 * units in GRAPHICS for the lifetime of a given GPU, to the extent the RM
 * can determine these values. It covers L1C (level 1 cache) and SM RF
 * (register file) errors.
 *
 * gpcEcc
 *   This parameter stores the L1C and RF ECC single- and double-bit
 *   error counts.
 *
 * Possible return status values are
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *
 */
#define LW90E0_CTRL_CMD_GR_GET_AGGREGATE_ECC_COUNTS (0x90e00102) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_GRAPHICS_GRAPHICS_INTERFACE_ID << 8) | LW90E0_CTRL_GR_GET_AGGREGATE_ECC_COUNTS_PARAMS_MESSAGE_ID" */

#define LW90E0_CTRL_GR_GET_AGGREGATE_ECC_COUNTS_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW90E0_CTRL_GR_GET_AGGREGATE_ECC_COUNTS_PARAMS {
    LW_DECLARE_ALIGNED(LW90E0_CTRL_GR_GET_ECC_GPC_COUNTS gpcEcc[LW90E0_CTRL_GR_ECC_GPC_COUNT], 8);
} LW90E0_CTRL_GR_GET_AGGREGATE_ECC_COUNTS_PARAMS;

/*
 * LW90E0_CTRL_CMD_GR_QUERY_ECC_CAPABILITIES
 *
 * This command is used to query the ECC capabilities of the GPU.
 *
 *   flags
 *     Output parameter to specify what ECC capabilities are supported.
 *     Legal values for this parameter include:
 *
 *       LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_L1C
 *           Indicates that ECC on the L1 cache is supported.
 *
 *       LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SM
 *           Indicates that ECC on the SM Register File is supported
 *
 *       LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_TEX
 *           Indicates that parity protection for Texture is supported
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 *
 */
#define LW90E0_CTRL_CMD_GR_QUERY_ECC_CAPABILITIES (0x90e00103) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_GRAPHICS_GRAPHICS_INTERFACE_ID << 8) | LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_PARAMS_MESSAGE_ID" */

#define LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_PARAMS {
    LwU32 flags;
} LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_PARAMS;

/* Legal flag parameter values */
#define LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_L1C             0:0
#define LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_L1C_NOECC    (0x00000000)
#define LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_L1C_ECC      (0x00000001)

#define LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SM              1:1
#define LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SM_NOECC     (0x00000000)
#define LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SM_ECC       (0x00000001)

#define LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_TEX             3:2
#define LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_TEX_NOPARITY (0x00000000)
#define LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_TEX_PARITY   (0x00000001)
#define LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_TEX_NOECC    (0x00000002)
#define LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_TEX_ECC      (0x00000003)

#define LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SHM             4:4
#define LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SHM_NOECC    (0x00000000)
#define LW90E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SHM_ECC      (0x00000001)

/* _ctrl90e0_h_ */

