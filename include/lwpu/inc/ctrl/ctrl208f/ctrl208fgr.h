/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2010-2019 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl208f/ctrl208fgr.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl208f/ctrl208fbase.h"

#include "ctrl/ctrl2080/ctrl2080gr.h"
/*
 * LW208F_CTRL_CMD_GR_GET_GOLDEN_CTX_IMAGE
 *
 * Control command to copy out the RM copy of the "golden context" backing
 * store to higher level drivers.
 *
 * Parameters:
 *
 * pImage
 *   Buffer to store the copied out context data.
 *
 * imageSize
 *   If pImage is NULL, this is an output returning the size of the image
 *   in bytes.
 *   If pImage is non-NULL, this is an input indicating the size of the image
 *   to be copied out.
 *
 *   It is expected that users of the call will first ascertain the size,
 *   allocate a buffer, and then call again with a properly sized buffer.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LWOS_STATUS_ERROR_ILWALID_PARAM_STATE
 */
#define LW208F_CTRL_CMD_GR_GET_GOLDEN_CTX_IMAGE (0x208f1201) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_GR_INTERFACE_ID << 8) | LW208F_CTRL_GR_GET_GOLDEN_CTX_IMAGE_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_GR_GET_GOLDEN_CTX_IMAGE_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW208F_CTRL_GR_GET_GOLDEN_CTX_IMAGE_PARAMS {
    LW_DECLARE_ALIGNED(LwP64 pImage, 8);
    LwU32 imageSize;
} LW208F_CTRL_GR_GET_GOLDEN_CTX_IMAGE_PARAMS;

typedef struct LW208F_CTRL_GR_SET_GOLDEN_CTX_IMAGE_PARAMS {
    LwHandle hChannel;
    LW_DECLARE_ALIGNED(LwP64 pImage, 8);
    LwU32    imageSize;
} LW208F_CTRL_GR_SET_GOLDEN_CTX_IMAGE_PARAMS;

/*
 * LW208F_CTRL_CMD_GR_ECC_INJECT_ERROR
 *
 * Control command to inject a gr ecc error
 *
 * Parameters:
 *
 * location
 *   location index
 * sublocation
 *   sublocation index
 * location
 *   LW208F_CTRL_GR_ECC_INJECT_ERROR_LOC
 * errorType
 *   SBE or DBE
 * grRouteInfo
 *   Routing info for SMC
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW208F_CTRL_CMD_GR_ECC_INJECT_ERROR (0x208f1203) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_GR_INTERFACE_ID << 8) | LW208F_CTRL_GR_ECC_INJECT_ERROR_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_GR_ECC_INJECT_ERROR_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW208F_CTRL_GR_ECC_INJECT_ERROR_PARAMS {
    LwU32 location;
    LwU32 sublocation;
    LwU8  unit;
    LwU8  errorType;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
} LW208F_CTRL_GR_ECC_INJECT_ERROR_PARAMS;

#define LW208F_CTRL_GR_ECC_INJECT_ERROR_UNIT              3:0
#define LW208F_CTRL_GR_ECC_INJECT_ERROR_UNIT_LRF     (0x00000000)
#define LW208F_CTRL_GR_ECC_INJECT_ERROR_UNIT_SHM     (0x00000001)
#define LW208F_CTRL_GR_ECC_INJECT_ERROR_UNIT_L1DATA  (0x00000002)
#define LW208F_CTRL_GR_ECC_INJECT_ERROR_UNIT_L1TAG   (0x00000003)
#define LW208F_CTRL_GR_ECC_INJECT_ERROR_UNIT_CBU     (0x00000004)
#define LW208F_CTRL_GR_ECC_INJECT_ERROR_UNIT_ICACHE  (0x00000005)
#define LW208F_CTRL_GR_ECC_INJECT_ERROR_UNIT_GCC_L15 (0x00000006)
#define LW208F_CTRL_GR_ECC_INJECT_ERROR_UNIT_GPCMMU  (0x00000007)
#define LW208F_CTRL_GR_ECC_INJECT_ERROR_UNIT_GPCCS   (0x00000008)
#define LW208F_CTRL_GR_ECC_INJECT_ERROR_UNIT_FECS    (0x00000009)
#define LW208F_CTRL_GR_ECC_INJECT_ERROR_UNIT_SM_RAMS (0x0000000A)

#define LW208F_CTRL_GR_ECC_INJECT_ERROR_TYPE              0:0
#define LW208F_CTRL_GR_ECC_INJECT_ERROR_TYPE_SBE     (0x00000000)
#define LW208F_CTRL_GR_ECC_INJECT_ERROR_TYPE_DBE     (0x00000001)

/*
 * LW208F_CTRL_CMD_GR_ECC_INJECTION_SUPPORTED
 *
 * Reports if error injection is supported for a given HW unit
 *
 * unit [in]:
 *      The ECC protected unit for which ECC injection support is being checked.
 *      The unit type is defined by LW208F_CTRL_GR_ECC_INJECT_ERROR_UNIT.
 *
 * bCorrectableSupported [out]:
 *      Boolean value that shows if correcatable errors can be injected.
 *
 * bUncorrectableSupported [out]:
 *      Boolean value that shows if uncorrecatable errors can be injected.
 *
 * Return values:
 *      LW_OK on success
 *      LW_ERR_ILWALID_ARGUMENT if the requested unit is invalid.
 *      LW_ERR_INSUFFICIENT_PERMISSIONS if priv write not enabled.
 *      LW_ERR_NOT_SUPPORTED otherwise
 *
 *
 */
#define LW208F_CTRL_CMD_GR_ECC_INJECTION_SUPPORTED   (0x208f1204) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_GR_INTERFACE_ID << 8) | LW208F_CTRL_GR_ECC_INJECTION_SUPPORTED_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_GR_ECC_INJECTION_SUPPORTED_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW208F_CTRL_GR_ECC_INJECTION_SUPPORTED_PARAMS {
    LwU8   unit;
    LwBool bCorrectableSupported;
    LwBool bUncorrectableSupported;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
} LW208F_CTRL_GR_ECC_INJECTION_SUPPORTED_PARAMS;

/* _ctrl208fgr_h_ */
