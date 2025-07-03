/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2019 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl00f4.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/*
 * See KT design doc for detailed control call descriptions
 * https://docs.google.com/document/d/10w_VCjVepxFB4Yyzqpghe-XYn6nllYeVkj050BuKDX4/edit#heading=h.kpi3mj9tdxdp
 *
 */


#define LW00F4_CTRL_CMD(cat,idx)  \
    LWXXXX_CTRL_CMD(0x00f4, LW00F4_CTRL_##cat, idx)

/* LW00F4 command categories (6bits) */
#define LW00F4_CTRL_RESERVED (0x00U)
#define LW00F4_CTRL_EXPORT   (0x01U)

/*
 * LW00F4_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW00F4_CTRL_CMD_NULL (0xf40000U) /* finn: Evaluated from "(FINN_LW01_MEMORY_FABRIC_EXPORT_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*
 * LW00F4_CTRL_CMD_EXPORT_SERIALIZE
 *
 * Outputs a serialized export object as a 128 byte
 * data buffer.
 *
 *   buffer [out]
 *     The 128 byte data buffer
 *
 * Possible status values returned are:
 *    LW_ERR_NOT_SUPPORTED
 *    LW_ERR_ILWALID_STATE
 *    LW_OK
 */
#define LW00F4_CTRL_CMD_EXPORT_SERIALIZE     (0xf40101U) /* finn: Evaluated from "(FINN_LW01_MEMORY_FABRIC_EXPORT_EXPORT_INTERFACE_ID << 8) | LW00F4_CTRL_EXPORT_SERIALIZE_PARAMS_MESSAGE_ID" */

#define LW00F4_EXPORT_SERIALIZED_BUFFER_SIZE 128U

#define LW00F4_CTRL_EXPORT_SERIALIZE_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW00F4_CTRL_EXPORT_SERIALIZE_PARAMS {
    LwU8 buffer[LW00F4_EXPORT_SERIALIZED_BUFFER_SIZE];
} LW00F4_CTRL_EXPORT_SERIALIZE_PARAMS;

/*
 * LW00F4_CTRL_CMD_EXPORT_DESCRIBE
 *
 * Queries the physical parameters of the underlying memory allocation.
 *
 *   offset [in]
 *     Offset (PFN index) into allocation to query physical parameters for
 *
 *   totalPfns [out]
 *     Number of PFNs in the memory object
 *
 *   pfnArray [out]
 *     Array of page frame numbers in memory allocation
 *
 *   numPfns [out]
 *     Number of valid entries in pfnArray
 *
 *  Possible status values returned are:
 *    LW_ERR_INSUFFICIENT_PERMISSIONS
 *    LW_ERR_ILWALID_ARGUMENT
 *    LW_OK
 */
#define LW00F4_CTRL_CMD_EXPORT_DESCRIBE            (0xf40102U) /* finn: Evaluated from "(FINN_LW01_MEMORY_FABRIC_EXPORT_EXPORT_INTERFACE_ID << 8) | LW00F4_CTRL_EXPORT_DESCRIBE_PARAMS_MESSAGE_ID" */

#define LW00F4_CTRL_EXPORT_DESCRIBE_PFN_ARRAY_SIZE 1000U

#define LW00F4_CTRL_EXPORT_DESCRIBE_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW00F4_CTRL_EXPORT_DESCRIBE_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 offset, 8);
    LW_DECLARE_ALIGNED(LwU64 totalPfns, 8);
    LwU32 pfnArray[LW00F4_CTRL_EXPORT_DESCRIBE_PFN_ARRAY_SIZE];
    LwU32 numPfns;
} LW00F4_CTRL_EXPORT_DESCRIBE_PARAMS;

/* _ctrl00f4_h_ */
