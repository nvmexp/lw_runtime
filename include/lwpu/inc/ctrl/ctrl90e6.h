/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012-2019 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl90e6.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#define LW90E6_CTRL_CMD(cat,idx)  \
    LWXXXX_CTRL_CMD(0x90E6, LW90E6_CTRL_##cat, idx)


/* LW90E6 command categories (6bits) */
#define LW90E6_CTRL_RESERVED (0x00)
#define LW90E6_CTRL_MASTER   (0x01)


/*
 * LW90E6_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LW90E6_CTRL_CMD_NULL (0x90e60000) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_MASTER_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*
 * LW90E6_CTRL_CMD_MASTER_GET_ERROR_INTR_OFFSET_MASK
 *
 * This command is used to query the offset and mask within the object mapping
 * that can be used to query for ECC and LWLINK interrupts.
 *
 * If a read of the given offset+mask is non-zero then it is possible an ECC or
 * an LWLINK error has been reported and not yet handled. If this is true then
 * the caller must either wait until the read returns zero or call into the
 * corresponding count reporting APIs to get updated counts.
 *
 * offset
 *   The offset into a GF100_SUBDEVICE_MASTSER's mapping where the top level
 *   interrupt register can be found.
 * mask
 *   Compatibility field that contains the same bits as eccMask. This field is
 *   deprecated and will be removed.
 * eccMask
 *   The mask to AND with the value found at offset to determine if any ECC
 *   interrupts are pending.
 * lwlinkMask
 *   The mask to AND with the value found at offset to determine if any LWLINK
 *   interrupts are pending.
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 */
#define LW90E6_CTRL_CMD_MASTER_GET_ERROR_INTR_OFFSET_MASK (0x90e60101) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_MASTER_MASTER_INTERFACE_ID << 8) | LW90E6_CTRL_MASTER_GET_ERROR_INTR_OFFSET_MASK_PARAMS_MESSAGE_ID" */

#define LW90E6_CTRL_MASTER_GET_ERROR_INTR_OFFSET_MASK_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW90E6_CTRL_MASTER_GET_ERROR_INTR_OFFSET_MASK_PARAMS {
    LwU32 offset;
    LwU32 mask;     // TODO: remove after all users have switched to use eccMask
    LwU32 eccMask;
    LwU32 lwlinkMask;
} LW90E6_CTRL_MASTER_GET_ERROR_INTR_OFFSET_MASK_PARAMS;

// TODO: remove once users of this interface have switched to the new name.
#define LW90E6_CTRL_CMD_MASTER_GET_ECC_INTR_OFFSET_MASK LW90E6_CTRL_CMD_MASTER_GET_ERROR_INTR_OFFSET_MASK

typedef LW90E6_CTRL_MASTER_GET_ERROR_INTR_OFFSET_MASK_PARAMS LW90E6_CTRL_MASTER_GET_ECC_INTR_OFFSET_MASK_PARAMS;

/*
 * LW90E6_CTRL_CMD_MASTER_GET_VIRTUAL_FUNCTION_ERROR_CONT_INTR_MASK
 *
 * This command is used to query the mask within the fastpath register
 * (VIRTUAL_FUNCTION_ERR_CONT) that can be used to query for ECC and LWLINK interrupts.
 *
 * If a read of the given mask is non-zero then it is possible an ECC or
 * an LWLINK error has been reported and not yet handled. If this is true then
 * the caller must either wait until the read returns zero or call into the
 * corresponding count reporting APIs to get updated counts.
 *
 * [out] eccMask
 *   The mask to AND with the value found at offset to determine if any ECC
 *   interrupts are possibly pending.
 * [out] lwlinkMask
 *   The mask to AND with the value found at offset to determine if any LWLINK
 *   interrupts are possibly pending.
 *
 *   Possible return status values are
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 */
#define LW90E6_CTRL_CMD_MASTER_GET_VIRTUAL_FUNCTION_ERROR_CONT_INTR_MASK (0x90e60102) /* finn: Evaluated from "(FINN_GF100_SUBDEVICE_MASTER_MASTER_INTERFACE_ID << 8) | LW90E6_CTRL_MASTER_GET_VIRTUAL_FUNCTION_ERROR_CONT_INTR_MASK_PARAMS_MESSAGE_ID" */

#define LW90E6_CTRL_MASTER_GET_VIRTUAL_FUNCTION_ERROR_CONT_INTR_MASK_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW90E6_CTRL_MASTER_GET_VIRTUAL_FUNCTION_ERROR_CONT_INTR_MASK_PARAMS {
    LwU32 eccMask;
    LwU32 lwlinkMask;
} LW90E6_CTRL_MASTER_GET_VIRTUAL_FUNCTION_ERROR_CONT_INTR_MASK_PARAMS;

/* _ctrl90e6_h_ */

