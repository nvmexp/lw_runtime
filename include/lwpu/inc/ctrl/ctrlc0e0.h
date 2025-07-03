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
// Source file: ctrl/ctrlc0e0.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#include "ctrl/ctrla0e0.h"          // C0E0 is partially derived from A0E0
#define LWC0E0_CTRL_CMD(cat,idx)  \
    LWXXXX_CTRL_CMD(0xC0E0, LWC0E0_CTRL_##cat, idx)


/* LWC0E0 command categories (6bits) */
#define LWC0E0_CTRL_RESERVED (0x00)
#define LWC0E0_CTRL_GRAPHICS (0x01)


/*
 * LWC0E0_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LWC0E0_CTRL_CMD_NULL (0xc0e00000) /* finn: Evaluated from "(FINN_GP100_SUBDEVICE_GRAPHICS_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*
 * LWC0E0_CTRL_CMD_GR_GET_ECC_COUNTS
 *
 * This command is used to query detailed ECC and EDC counts for each of the
 * units in GRAPHICS since the last reboot. It covers SHM (shared memory),
 * SM RF (register file), and TEX (texture unit) errors.
 *
 * gpcCount (in/out)
 *   Input:  this parameter specifies how many GPCs the
 *           passed in structure can support.
 *   Output: this parameter specifies how many GPCs of data
 *           was stored to the structure
 *
 * tpcCount (in/out)
 *   Input:  this parameter specifies how many TPC per GPC the
 *           passed in structure can support.
 *   Output: this parameter specifies how many TPCs per GPC of data
 *           was stored to the structure
 *
 * texCount (in/out)
 *   Input:  this parameter specifies how many texture units per TPC the
 *           passed in structure can support.
 *   Output: this parameter specifies how many texture units per TPC of data
 *           was stored to the structure
 *
 * gpcEcc (out)
 *   This parameter stores the SHM and RF ECC  single- and double-bit
 *   error counts and TEX parity error counts.
 *
 * Possible return status values are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *
 */
#define LWC0E0_CTRL_CMD_GR_GET_ECC_COUNTS (0xc0e00101) /* finn: Evaluated from "(FINN_GP100_SUBDEVICE_GRAPHICS_GRAPHICS_INTERFACE_ID << 8) | LWC0E0_CTRL_GR_GET_ECC_COUNTS_PARAMS_MESSAGE_ID" */

#define LWC0E0_CTRL_GR_ECC_GPC_COUNT      (0x00000006)
#define LWC0E0_CTRL_GR_ECC_TPC_COUNT      (0x00000005)
#define LWC0E0_CTRL_GR_ECC_TEX_COUNT      (0x00000002)

/*
 * LWC0E0_CTRL_GPU_QUERY_ECC_EXCEPTION_STATUS
 *
 * This structure represents the exception status of a class of per-unit
 * exceptions
 *
 *   count
 *     This parameter constains the number of exceptions that have oclwrred
 *     since boot.
 *
 */
typedef LWA0E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS LWC0E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS;


typedef LWA0E0_CTRL_GR_GET_ECC_TEX_COUNTS LWC0E0_CTRL_GR_GET_ECC_TEX_COUNTS;

/*
 * LWC0E0_CTRL_GR_GET_ECC_TPC_COUNTS
 *
 * This structure contains the number of single-bit and double-bit
 * exceptions that have oclwrred in a TPC.
 *
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
 *   tex
 *     This parameter contains the number of correctable and uncorrectable
 *     parity errors for each texture unit.
 */
typedef struct LWC0E0_CTRL_GR_GET_ECC_TPC_COUNTS {
    LW_DECLARE_ALIGNED(LWC0E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS rfSbe, 8);
    LW_DECLARE_ALIGNED(LWC0E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS rfDbe, 8);
    LW_DECLARE_ALIGNED(LWC0E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS shmSbe, 8);
    LW_DECLARE_ALIGNED(LWC0E0_CTRL_GR_QUERY_ECC_EXCEPTION_STATUS shmDbe, 8);
    LW_DECLARE_ALIGNED(LWC0E0_CTRL_GR_GET_ECC_TEX_COUNTS tex[LWC0E0_CTRL_GR_ECC_TEX_COUNT], 8);
} LWC0E0_CTRL_GR_GET_ECC_TPC_COUNTS;

typedef LWC0E0_CTRL_GR_GET_ECC_TPC_COUNTS LWC0E0_CTRL_GR_GET_ECC_GPC_COUNTS[LWC0E0_CTRL_GR_ECC_TPC_COUNT];


#define LWC0E0_CTRL_GR_GET_ECC_COUNTS_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWC0E0_CTRL_GR_GET_ECC_COUNTS_PARAMS {
    LwU32 tpcCount;
    LwU32 gpcCount;
    LwU32 texCount;

    LW_DECLARE_ALIGNED(LWC0E0_CTRL_GR_GET_ECC_GPC_COUNTS gpcEcc[LWC0E0_CTRL_GR_ECC_GPC_COUNT], 8);
} LWC0E0_CTRL_GR_GET_ECC_COUNTS_PARAMS;

/*
 * LWC0E0_CTRL_CMD_GR_GET_AGGREGATE_ECC_COUNTS
 *
 * This command is used to query detailed ECC and EDC counts for each of the
 * units in GRAPHICS for the lifetime of a given GPU, to the extent the RM
 * can determine these values. It covers SHM (shared memory), SM RF
 * (register file), and TEX (texture) errors.
 *
 * gpcEcc
 *   This parameter stores the SHM and RF ECC  single- and double-bit
 *   error counts and TEX parity error counts.
 *
 * Possible return status values are
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *
 */
#define LWC0E0_CTRL_CMD_GR_GET_AGGREGATE_ECC_COUNTS (0xc0e00102) /* finn: Evaluated from "(FINN_GP100_SUBDEVICE_GRAPHICS_GRAPHICS_INTERFACE_ID << 8) | 0x2" */

typedef LWC0E0_CTRL_GR_GET_ECC_COUNTS_PARAMS LWC0E0_CTRL_GR_GET_AGGREGATE_ECC_COUNTS_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LWC0E0_CTRL_CMD_GR_GET_AGGREGATE_ECC_COUNTS_FINN_PARAMS_MESSAGE_ID (0x2U)

typedef struct LWC0E0_CTRL_CMD_GR_GET_AGGREGATE_ECC_COUNTS_FINN_PARAMS {
    LW_DECLARE_ALIGNED(LWC0E0_CTRL_GR_GET_AGGREGATE_ECC_COUNTS_PARAMS params, 8);
} LWC0E0_CTRL_CMD_GR_GET_AGGREGATE_ECC_COUNTS_FINN_PARAMS;



/*
 * LWC0E0_CTRL_CMD_GR_QUERY_ECC_CAPABILITIES
 *
 * This command is used to query the ECC capabilities of the GPU.
 *
 *   flags
 *     Output parameter to specify what ECC capabilities are supported.
 *     Legal values for this parameter include:
 *
 *       LWC0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_L1C
 *           Indicates that ECC on the L1 cache is supported.
 *
 *       LWC0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SM
 *           Indicates that ECC on the SM Register File is supported
 *
 *       LWC0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_TEX
 *           Indicates that parity protection and/or ECC for Texture is supported
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 *
 */
#define LWC0E0_CTRL_CMD_GR_QUERY_ECC_CAPABILITIES (0xc0e00103) /* finn: Evaluated from "(FINN_GP100_SUBDEVICE_GRAPHICS_GRAPHICS_INTERFACE_ID << 8) | LWC0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_PARAMS_MESSAGE_ID" */

#define LWC0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_PARAMS_MESSAGE_ID (0x3U)

typedef struct LWC0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_PARAMS {
    LwU32 flags;
} LWC0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_PARAMS;

/* Legal flag parameter values */
#define LWC0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_L1C             0:0
#define LWC0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_L1C_NOECC    (0x00000000)
#define LWC0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_L1C_ECC      (0x00000001)

#define LWC0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SM              1:1
#define LWC0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SM_NOECC     (0x00000000)
#define LWC0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SM_ECC       (0x00000001)

#define LWC0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_TEX             3:2
#define LWC0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_TEX_NOPARITY (0x00000000)
#define LWC0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_TEX_PARITY   (0x00000001)
#define LWC0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_TEX_NOECC    (0x00000002)
#define LWC0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_TEX_ECC      (0x00000003)

#define LWC0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SHM             4:4
#define LWC0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SHM_NOECC    (0x00000000)
#define LWC0E0_CTRL_GR_QUERY_ECC_CAPABILITIES_FLAGS_SHM_ECC      (0x00000001)

/* _ctrlc0e0_h_ */

