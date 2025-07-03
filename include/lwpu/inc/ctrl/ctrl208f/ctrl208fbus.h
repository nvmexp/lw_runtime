/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2020-2021 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl208f/ctrl208fbus.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl208f/ctrl208fbase.h"

/*
 * LW208F_CTRL_CMD_BUS_IS_BAR1_VIRTUAL
 *
 * This command checks whether or not BAR1 is in virtual mode.
 * This API is intended for internal testing only.
 *
 *   bIsVirtual
 *     Whether or not Bar1 is in virtual mode.
 */
#define LW208F_CTRL_CMD_BUS_IS_BAR1_VIRTUAL (0x208f1801) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_BUS_INTERFACE_ID << 8) | LW208F_CTRL_BUS_IS_BAR1_VIRTUAL_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_BUS_IS_BAR1_VIRTUAL_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW208F_CTRL_BUS_IS_BAR1_VIRTUAL_PARAMS {
    LwBool bIsVirtual;
} LW208F_CTRL_BUS_IS_BAR1_VIRTUAL_PARAMS;

/*
 * LW208F_CTRL_CMD_BUS_ECC_INJECT_ERROR
 *
 * This ctrl call injects PCI-E XAL-EP ECC errors.  Please see the ECC
 * overview page for more information on ECC and ECC injection:
 *
 * https://confluence.lwpu.com/display/CSSRM/ECC
 *
 * Parameters:
 *
 * location
 *   Specifies the XAL-EP HW unit where the injection will occur.
 *
 * errorType
 *   Specifies whether the injected error will be correctable or uncorrectable.
 *   Correctable errors have no effect on running programs while uncorrectable
 *   errors will cause all channels to be torn down.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW208F_CTRL_CMD_BUS_ECC_INJECT_ERROR (0x208f1802) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_BUS_INTERFACE_ID << 8) | LW208F_CTRL_BUS_ECC_INJECT_ERROR_PARAMS_MESSAGE_ID" */

typedef enum LW208F_CTRL_BUS_UNIT_TYPE {
    LW208F_CTRL_BUS_PCIE_REORDER_BUFFER = 0,
    LW208F_CTRL_BUS_PCIE_P2PREQ_BUFFER = 1,
} LW208F_CTRL_BUS_UNIT_TYPE;

typedef enum LW208F_CTRL_BUS_ERROR_TYPE {
    LW208F_CTRL_BUS_ERROR_TYPE_CORRECTABLE = 0,
    LW208F_CTRL_BUS_ERROR_TYPE_UNCORRECTABLE = 1,
    LW208F_CTRL_BUS_ERROR_TYPE_NONE = 2,
} LW208F_CTRL_BUS_ERROR_TYPE;

#define LW208F_CTRL_BUS_ECC_INJECT_ERROR_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW208F_CTRL_BUS_ECC_INJECT_ERROR_PARAMS {
    LW208F_CTRL_BUS_ERROR_TYPE errorType;
    LW208F_CTRL_BUS_UNIT_TYPE  errorUnit;
} LW208F_CTRL_BUS_ECC_INJECT_ERROR_PARAMS;

/*
 * LW208F_CTRL_CMD_BUS_ECC_INJECTION_SUPPORTED
 *
 * Reports if error injection is supported for XAL-EP
 *
 * bCorrectableSupported [out]:
 *      Boolean value that shows if correcatable errors can be injected.
 *
 * bUncorrectableSupported [out]:
 *      Boolean value that shows if uncorrecatable errors can be injected.
 *
 * Return values:
 *      LW_OK on success
 *      LW_ERR_INSUFFICIENT_PERMISSIONS if priv write not enabled.
 *      LW_ERR_NOT_SUPPORTED otherwise
 *
 *
 */
#define LW208F_CTRL_CMD_BUS_ECC_INJECTION_SUPPORTED (0x208f1803) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_BUS_INTERFACE_ID << 8) | LW208F_CTRL_BUS_ECC_INJECTION_SUPPORTED_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_BUS_ECC_INJECTION_SUPPORTED_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW208F_CTRL_BUS_ECC_INJECTION_SUPPORTED_PARAMS {
    LW208F_CTRL_BUS_UNIT_TYPE errorUnit;
    LwBool                    bCorrectableSupported;
    LwBool                    bUncorrectableSupported;
} LW208F_CTRL_BUS_ECC_INJECTION_SUPPORTED_PARAMS;

/* _ctrl208fbus_h_ */

