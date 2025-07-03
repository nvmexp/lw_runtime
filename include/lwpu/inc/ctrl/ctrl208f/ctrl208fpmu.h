/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2018 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl208f/ctrl208fpmu.finn
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
 * LW208F_CTRL_CMD_PMU_ECC_INJECT_ERROR
 *
 * This ctrl call injects PMU ECC errors.  Please see the ECC
 * overview page for more information on ECC and ECC injection:
 *
 * https://confluence.lwpu.com/display/CSSRM/ECC
 *
 * Parameters:
 *
 * location
 *   Specifies the PMU HW unit where the injection will occur.
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
#define LW208F_CTRL_CMD_PMU_ECC_INJECT_ERROR (0x208f0c01) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_PMU_INTERFACE_ID << 8) | LW208F_CTRL_PMU_ECC_INJECT_ERROR_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_PMU_ECC_INJECT_ERROR_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW208F_CTRL_PMU_ECC_INJECT_ERROR_PARAMS {
    LwU8 errorType;
} LW208F_CTRL_PMU_ECC_INJECT_ERROR_PARAMS;

#define LW208F_CTRL_PMU_ECC_INJECT_ERROR_TYPE                 0:0
#define LW208F_CTRL_PMU_ECC_INJECT_ERROR_TYPE_CORRECTABLE   (0x00000000)
#define LW208F_CTRL_PMU_ECC_INJECT_ERROR_TYPE_UNCORRECTABLE (0x00000001)

/*
 * LW208F_CTRL_CMD_PMU_ECC_INJECTION_SUPPORTED
 *
 * Reports if error injection is supported for the PMU
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
#define LW208F_CTRL_CMD_PMU_ECC_INJECTION_SUPPORTED         (0x208f0c02) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_PMU_INTERFACE_ID << 8) | LW208F_CTRL_PMU_ECC_INJECTION_SUPPORTED_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_PMU_ECC_INJECTION_SUPPORTED_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW208F_CTRL_PMU_ECC_INJECTION_SUPPORTED_PARAMS {
    LwBool bCorrectableSupported;
    LwBool bUncorrectableSupported;
} LW208F_CTRL_PMU_ECC_INJECTION_SUPPORTED_PARAMS;

/* _ctrl208fpmu_h_ */
