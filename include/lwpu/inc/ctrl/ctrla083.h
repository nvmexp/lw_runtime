/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2015 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrla083.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* LWA083_GRID_DISPLAYLESS control commands and parameters */

#define LWA083_CTRL_CMD(cat,idx)             LWXXXX_CTRL_CMD(0xA083, LWA083_CTRL_##cat, idx)

/* Command categories (6bits) */
#define LWA083_CTRL_RESERVED        (0x00)
#define LWA083_CTRL_VIRTUAL_DISPLAY (0x01)

/*
 * LWA083_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LWA083_CTRL_CMD_NULL        (0xa0830000) /* finn: Evaluated from "(FINN_LWA083_GRID_DISPLAYLESS_RESERVED_INTERFACE_ID << 8) | 0x0" */





/*
 * LWA083_CTRL_CMD_VIRTUAL_DISPLAY_GET_NUM_HEADS
 *
 * This command returns the number of heads supported on a GPU with
 * Displayless path enabled OR on a virtual GPU.
 *
 *   numHeads
 *     This parameter returns total number of heads supported under
 *     a particular GRID SW license.
 *
 *   maxNumHeads
 *     This parameter returns the MAX number of heads supported on this
 *     Displayless GRID GPU without any overrides of GPU capabilities
 *     due to GRID Software Licensing.
 *     On Windows/Linux, we cannot dynamically change number of display
 *     head/source exposed to OS. Hence DD is required to report total
 *     number of display heads during init.
 *     DD registers during initialization the MAX number of heads supported and
 *     post that ensure only fixed allowed number of heads (reported via
 *     numHeads) can be made active (usable) for a particular license.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */

#define LWA083_CTRL_CMD_VIRTUAL_DISPLAY_GET_NUM_HEADS (0xa0830101) /* finn: Evaluated from "(FINN_LWA083_GRID_DISPLAYLESS_VIRTUAL_DISPLAY_INTERFACE_ID << 8) | LWA083_CTRL_VIRTUAL_DISPLAY_GET_NUM_HEADS_PARAMS_MESSAGE_ID" */

#define LWA083_CTRL_VIRTUAL_DISPLAY_GET_NUM_HEADS_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWA083_CTRL_VIRTUAL_DISPLAY_GET_NUM_HEADS_PARAMS {
    LwU32 numHeads;
    LwU32 maxNumHeads;
} LWA083_CTRL_VIRTUAL_DISPLAY_GET_NUM_HEADS_PARAMS;

/*
 * LWA083_CTRL_CMD_VIRTUAL_DISPLAY_GET_MAX_RESOLUTION
 * 
 * This command returns the maximum resolution supported on a GPU with
 * Displayless path enabled OR on a virtual GPU.
 *
 *   headIndex
 *     This parameter specifies the head for which the maximum resolution is to be 
 *     retrieved.
 *
 *   maxHResolution
 *     This parameter returns the maximum horizontal resolution.
 *
 *   maxVResolution
 *      This parameter returns the maximum vertical resolution.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LWA083_CTRL_CMD_VIRTUAL_DISPLAY_GET_MAX_RESOLUTION (0xa0830102) /* finn: Evaluated from "(FINN_LWA083_GRID_DISPLAYLESS_VIRTUAL_DISPLAY_INTERFACE_ID << 8) | LWA083_CTRL_VIRTUAL_DISPLAY_GET_MAX_RESOLUTION_PARAMS_MESSAGE_ID" */

#define LWA083_CTRL_VIRTUAL_DISPLAY_GET_MAX_RESOLUTION_PARAMS_MESSAGE_ID (0x2U)

typedef struct LWA083_CTRL_VIRTUAL_DISPLAY_GET_MAX_RESOLUTION_PARAMS {
    LwU32 headIndex;
    LwU32 maxHResolution;
    LwU32 maxVResolution;
} LWA083_CTRL_VIRTUAL_DISPLAY_GET_MAX_RESOLUTION_PARAMS;

/*
 * LWA083_CTRL_CMD_VIRTUAL_DISPLAY_GET_DEFAULT_EDID
 * 
 * This command returns the default EDID for GRID
 * displayless path.
 *
 *   pEdidBuffer
 *     This field provides a pointer to a buffer into which the
 *     default EDID for GRID displayless is retrieved.
 *   edidSize
 *     This field is used as an input/output. It specifies the maximum size of
 *     pEdidBuffer. If edidSize is set to zero on input, the actual required
 *     size of buffer will be updated in this field before returning.
 *     If however, edidSize is non-zero and smaller than that required, an 
 *     error will be returned indicating the buffer was too small.
 *   connectorType
 *     This field is used as an input. It takes the connector information
 *     for which EDID for GRID displayless is queried.
 *     for DVI_D, Digital EDID is returned
 *     for VGA_15_PIN, Analog EDID is returned
 *     Both Digital and Analog EDIDs have same set of resolutions
 * 
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LWOS_ERR_BUFFER_TOO_SMALL
 */

#define LWA083_CTRL_CMD_VIRTUAL_DISPLAY_GET_DEFAULT_EDID (0xa0830103) /* finn: Evaluated from "(FINN_LWA083_GRID_DISPLAYLESS_VIRTUAL_DISPLAY_INTERFACE_ID << 8) | LWA083_CTRL_VIRTUAL_DISPLAY_GET_DEFAULT_EDID_PARAMS_MESSAGE_ID" */

#define LWA083_CTRL_VIRTUAL_DISPLAY_GET_DEFAULT_EDID_PARAMS_MESSAGE_ID (0x3U)

typedef struct LWA083_CTRL_VIRTUAL_DISPLAY_GET_DEFAULT_EDID_PARAMS {
    LW_DECLARE_ALIGNED(LwP64 pEdidBuffer, 8);
    LwU32 edidSize;
    LwU8  connectorType;
} LWA083_CTRL_VIRTUAL_DISPLAY_GET_DEFAULT_EDID_PARAMS;

/* valid connector type values. Keeping DVI as the default one for displayless */
#define LWA083_CTRL_CMD_CONNECTOR_INFO_TYPE_DVI_D      (0x00)
#define LWA083_CTRL_CMD_CONNECTOR_INFO_TYPE_VGA_15_PIN (0x01)

/*
 * LWA083_CTRL_CMD_VIRTUAL_DISPLAY_IS_ACTIVE
 *
 * This command to query if any display is active or not
 *
 * Parameters:
 *
 *   isDisplayActive
 *     Indicates if display is lwrrently active or not.
 *
 *   Possible status values returned are:
 *     LWOS_STATUS_SUCCESS
 */
#define LWA083_CTRL_CMD_VIRTUAL_DISPLAY_IS_ACTIVE      (0xa0830104) /* finn: Evaluated from "(FINN_LWA083_GRID_DISPLAYLESS_VIRTUAL_DISPLAY_INTERFACE_ID << 8) | LWA083_CTRL_VIRTUAL_DISPLAY_IS_ACTIVE_PARAMS_MESSAGE_ID" */

#define LWA083_CTRL_VIRTUAL_DISPLAY_IS_ACTIVE_PARAMS_MESSAGE_ID (0x4U)

typedef struct LWA083_CTRL_VIRTUAL_DISPLAY_IS_ACTIVE_PARAMS {
    LwBool isDisplayActive;
} LWA083_CTRL_VIRTUAL_DISPLAY_IS_ACTIVE_PARAMS;

/*
 * LWA083_CTRL_CMD_VIRTUAL_DISPLAY_IS_CONNECTED
 *
 * This command to query if any display is connected or not
 *
 * Parameters:
 *
 *   isDisplayConnected
 *     Indicates if display is lwrrently connected or not.
 *
 *   Possible status values returned are:
 *     LWOS_STATUS_SUCCESS
 */
#define LWA083_CTRL_CMD_VIRTUAL_DISPLAY_IS_CONNECTED (0xa0830105) /* finn: Evaluated from "(FINN_LWA083_GRID_DISPLAYLESS_VIRTUAL_DISPLAY_INTERFACE_ID << 8) | LWA083_CTRL_VIRTUAL_DISPLAY_IS_CONNECTED_PARAMS_MESSAGE_ID" */

#define LWA083_CTRL_VIRTUAL_DISPLAY_IS_CONNECTED_PARAMS_MESSAGE_ID (0x5U)

typedef struct LWA083_CTRL_VIRTUAL_DISPLAY_IS_CONNECTED_PARAMS {
    LwU32 isDisplayConnected;
} LWA083_CTRL_VIRTUAL_DISPLAY_IS_CONNECTED_PARAMS;


/*
 * LWA083_CTRL_CMD_VIRTUAL_DISPLAY_GET_MAX_PIXELS
 *
 * This command returns the maximum pixel count for on a GPU with
 * Displayless path enabled OR on a virtual GPU.
 *
 *   maxPixels
 *     This parameter returns the max pixel limit for a displayless GPU
 *     OR a vGPU
 *
 * Possible status values returned are:
 *   LW_OK
 */

#define LWA083_CTRL_CMD_VIRTUAL_DISPLAY_GET_MAX_PIXELS (0xa0830106) /* finn: Evaluated from "(FINN_LWA083_GRID_DISPLAYLESS_VIRTUAL_DISPLAY_INTERFACE_ID << 8) | LWA083_CTRL_VIRTUAL_DISPLAY_GET_MAX_PIXELS_PARAMS_MESSAGE_ID" */

#define LWA083_CTRL_VIRTUAL_DISPLAY_GET_MAX_PIXELS_PARAMS_MESSAGE_ID (0x6U)

typedef struct LWA083_CTRL_VIRTUAL_DISPLAY_GET_MAX_PIXELS_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 maxPixels, 8);
} LWA083_CTRL_VIRTUAL_DISPLAY_GET_MAX_PIXELS_PARAMS;

/* _ctrla083_h_ */
