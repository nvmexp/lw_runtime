/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2019-2020 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrlc763.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* Vidmem Access bit buffer control commands and parameters */

#define LWC763_CTRL_CMD(cat,idx)          LWXXXX_CTRL_CMD(0xC763, LWC763_CTRL_##cat, idx)//sw/dev/gpu_drv/chips_a/sdk/lwpu/inc/ctrl/ctrlc763.h

/* MMU_VIDMEM_ACCESS_BIT_BUFFER command categories (6bits) */
#define LWC763_CTRL_RESERVED                     (0x00)
#define LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER     (0x01)

/*
 * SW def for number of range checkers. Current value taken from 
 * LW_PFB_PRI_MMU_VIDMEM_ACCESS_BIT_START_ADDR_LO__SIZE_1
 * on GA102. Compile time assert to check that the below
 * definition is consistent with HW manuals is included in
 * each gmmu HAL where this is relevant.
 */
#define LW_VIDMEM_ACCESS_BIT_BUFFER_NUM_CHECKERS 8

/*
 * LWC763_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LWC763_CTRL_CMD_NULL                     (0xc7630000) /* finn: Evaluated from "(FINN_MMU_VIDMEM_ACCESS_BIT_BUFFER_RESERVED_INTERFACE_ID << 8) | 0x0" */





#define LWC763_CTRL_CMD_VIDMEM_ACCESS_BIT_ENABLE_LOGGING (0xc7630101) /* finn: Evaluated from "(FINN_MMU_VIDMEM_ACCESS_BIT_BUFFER_VIDMEM_ACCESS_BIT_BUFFER_INTERFACE_ID << 8) | LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_ENABLE_LOGGING_PARAMS_MESSAGE_ID" */

// Supported granularities for the vidmem access bit buffer logging
typedef enum LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_GRANULARITY {
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_GRANULARITY_64KB = 0,
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_GRANULARITY_128KB = 1,
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_GRANULARITY_256KB = 2,
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_GRANULARITY_512KB = 3,
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_GRANULARITY_1MB = 4,
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_GRANULARITY_2MB = 5,
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_GRANULARITY_4MB = 6,
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_GRANULARITY_8MB = 7,
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_GRANULARITY_16MB = 8,
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_GRANULARITY_32MB = 9,
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_GRANULARITY_64MB = 10,
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_GRANULARITY_128MB = 11,
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_GRANULARITY_256MB = 12,
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_GRANULARITY_512MB = 13,
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_GRANULARITY_1GB = 14,
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_GRANULARITY_2GB = 15,
} LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_GRANULARITY;

/**
 * enum of disable mode to be used when the MMU enters protected mode
 */
typedef enum LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_DISABLE_MODE {
    /*!
     * Disable mode will set all the access/dirty bits as '0'
     */
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_DISABLE_MODE_CLEAR = 0,
    /*!
     * Disable mode will set all the access/dirty bits as '1'
     */
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_DISABLE_MODE_SET = 1,
} LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_DISABLE_MODE;


// 
// If clients want to enable logging specifically for some MMU, clients need to
// do it in a loop
//
typedef enum LW_VIDMEM_ACCESS_BIT_BUFFER_MMU_TYPE {
    /*!
     * Read/Write Attrs only for HUBMMU registers
     */
    LW_VIDMEM_ACCESS_BIT_BUFFER_HUBMMU = 0,
    /*!
     * Read/Write Attrs only for GPCMMU registers
     */
    LW_VIDMEM_ACCESS_BIT_BUFFER_GPCMMU = 1,
    /*!
     * Read/Write Attrs only for HSHUBMMU registers
     */
    LW_VIDMEM_ACCESS_BIT_BUFFER_HSHUBMMU = 2,
    /*!
     * Default will enable for all MMU possible
     */
    LW_VIDMEM_ACCESS_BIT_BUFFER_DEFAULT = 3,
} LW_VIDMEM_ACCESS_BIT_BUFFER_MMU_TYPE;

typedef enum LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_TRACK_MODE {
    /*!
     Mode to track access bits
     */
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_TRACK_MODE_ACCESS = 0,
    /*!
     Mode to track dirty bits
     */
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_TRACK_MODE_DIRTY = 1,
} LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_TRACK_MODE;

/*
 * LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_ENABLE_LOGGING_PARAMS
 *
 * This structure is used to enable logging of the VAB and specifies
 * the requested configuration for the 8 independent range checkers.
 * The tracking mode and disable mode are the same for all range checkers.
 */
#define LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_ENABLE_LOGGING_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_ENABLE_LOGGING_PARAMS {
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_GRANULARITY  granularity[LW_VIDMEM_ACCESS_BIT_BUFFER_NUM_CHECKERS];
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_TRACK_MODE   trackMode;
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_DISABLE_MODE disableMode;
    LW_DECLARE_ALIGNED(LwU64 startAddress[LW_VIDMEM_ACCESS_BIT_BUFFER_NUM_CHECKERS], 8);
    LwU8                                              rangeCount;
    LW_VIDMEM_ACCESS_BIT_BUFFER_MMU_TYPE              mmuType;
} LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_ENABLE_LOGGING_PARAMS;


#define LWC763_CTRL_CMD_VIDMEM_ACCESS_BIT_DISABLE_LOGGING (0xc7630102) /* finn: Evaluated from "(FINN_MMU_VIDMEM_ACCESS_BIT_BUFFER_VIDMEM_ACCESS_BIT_BUFFER_INTERFACE_ID << 8) | 0x2" */

/*
 * LWC763_CTRL_CMD_VIDMEM_ACCESS_BIT_BUFFER_DUMP
 *
 * This call initiates the dump request with the properties set using enable
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LWC763_CTRL_CMD_VIDMEM_ACCESS_BIT_DUMP            (0xc7630103) /* finn: Evaluated from "(FINN_MMU_VIDMEM_ACCESS_BIT_BUFFER_VIDMEM_ACCESS_BIT_BUFFER_INTERFACE_ID << 8) | LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_DUMP_PARAMS_MESSAGE_ID" */

/*
 * LW_VAB_OP enumerates the types of dumps supported
 *
 * The options are:
 *
 *     AGGREGATE
 *     Collects access buffer bits over multiple dumps using a bitwise OR.
 *
 *     DIFF
 *     Sets a bit to 1 if it changed from 0 to 1 with this dump. If a bit was
 *     cleared since the last dump it will be 0. If a bit does not change
 *     with this dump it will be 0.
 *
 *     CURRENT
 *     Copies the current access bit buffer state as is from HW. This operation
 *     clears any underlying aggregation from previous dumps with the other
 *     two operations.
 *
 *     INVALID
 *     Should be unused and otherwise indicates error
 */
typedef enum LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_OP {
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_OP_AGGREGATE = 0,
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_OP_DIFF = 1,
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_OP_LWRRENT = 2,
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_OP_ILWALID = 3,
} LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_OP;

/*
 * LWC763_CTRL_VIDMEM_ACCESS_BIT_DUMP_PARAMS
 *
 * This structure records the dumped bits for the client
 * masked by the client's access bit mask determined
 * during VidmemAccessBitBuffer construction.
 *
 * bMetadata [IN]
 *      Whether or not clients want disable data.
 *
 * op_enum [IN]
 *      A member of LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_OP controlling the type of dump.
 *
 * accessBits [OUT]
 *      The client's access bits masked according to the client's access bit mask.
 *
 * gpcDisable [OUT]
 *      The GPC disable data from the VAB dump. See GPC_DISABLE in the Ampere-801 FD.
 *      
 * hubDisable [OUT]
 *      The HUB disable data from the VAB dump. See HUB_DISABLE in the Ampere-801 FD.
 *
 * hsceDisable [OUT]
 *      The HSCE disable data from the VAB dump. See HSCE_DIS in the Ampere-801 FD.
 *
 * linkDisable [OUT]
 *      The LINK disable data from the VAB dump. See LINK_DIS in the Ampere-801 FD.
 */
#define LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_DUMP_PARAMS_MESSAGE_ID (0x3U)

typedef struct LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_DUMP_PARAMS {
    LwBool                                  bMetadata;
    LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_OP op_enum;
    LW_DECLARE_ALIGNED(LwU64 accessBits[64], 8);
    LW_DECLARE_ALIGNED(LwU64 gpcDisable, 8);
    LwU32                                   hubDisable;
    LwU16                                   hsceDisable;
    LwU8                                    linkDisable;
} LWC763_CTRL_VIDMEM_ACCESS_BIT_BUFFER_DUMP_PARAMS;


#define LWC763_CTRL_CMD_VIDMEM_ACCESS_BIT_PUT_OFFSET (0xc7630104) /* finn: Evaluated from "(FINN_MMU_VIDMEM_ACCESS_BIT_BUFFER_VIDMEM_ACCESS_BIT_BUFFER_INTERFACE_ID << 8) | LWC763_CTRL_VIDMEM_ACCESS_BIT_PUT_OFFSET_PARAMS_MESSAGE_ID" */

#define LWC763_CTRL_VIDMEM_ACCESS_BIT_PUT_OFFSET_PARAMS_MESSAGE_ID (0x4U)

typedef struct LWC763_CTRL_VIDMEM_ACCESS_BIT_PUT_OFFSET_PARAMS {
    LwU32 vidmemAccessBitPutOffset;
} LWC763_CTRL_VIDMEM_ACCESS_BIT_PUT_OFFSET_PARAMS;
/* _ctrlc763_h_ */
