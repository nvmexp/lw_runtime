/*
 * SPDX-FileCopyrightText: Copyright (c) 2005-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl0073/ctrl0073system.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl0073/ctrl0073base.h"

/* LW04_DISPLAY_COMMON system-level control commands and parameters */

/* extract cap bit setting from tbl */
#define LW0073_CTRL_SYSTEM_GET_CAP(tbl,c)         (((LwU8)tbl[(1?c)]) & (0?c))

/* Caps format is byte_index:bit_mask.
 * Important: keep the number of bytes needed for these fields in sync with
 * LW0073_CTRL_SYSTEM_CAPS_TBL_SIZE
 */
#define LW0073_CTRL_SYSTEM_CAPS_AA_FOS_GAMMA_COMP_SUPPORTED        0:0x01
#define LW0073_CTRL_SYSTEM_CAPS_TV_LOWRES_BUG_85919                0:0x02
#define LW0073_CTRL_SYSTEM_CAPS_DFP_GPU_SCALING_BUG_154102         0:0x04
#define LW0073_CTRL_SYSTEM_CAPS_SLI_INTERLACED_MODE_BUG_235218     0:0x08 // Deprecated
#define LW0073_CTRL_SYSTEM_CAPS_STEREO_DIN_AVAILABLE               0:0x10
#define LW0073_CTRL_SYSTEM_CAPS_OFFSET_PCLK_DFP_FOR_EMI_BUG_443891 0:0x20
#define LW0073_CTRL_SYSTEM_CAPS_GET_DMI_SCANLINE_SUPPORTED         0:0x40
/*
 * Indicates support for HDCP Key Selection Vector (KSV) list and System
 * Renewability Message (SRM) validation
*/
#define LW0073_CTRL_SYSTEM_CAPS_KSV_SRM_VALIDATION_SUPPORTED       0:0x80

#define LW0073_CTRL_SYSTEM_CAPS_SINGLE_HEAD_MST_SUPPORTED          1:0x01
#define LW0073_CTRL_SYSTEM_CAPS_SINGLE_HEAD_DUAL_SST_SUPPORTED     1:0x02
#define LW0073_CTRL_SYSTEM_CAPS_HDMI_2_0_SUPPORTED                 1:0x04
#define LW0073_CTRL_SYSTEM_CAPS_CROSS_BAR_SUPPORTED                1:0x08
#define LW0073_CTRL_SYSTEM_CAPS_RASTER_LOCK_NEEDS_MIO_POWER        1:0x10
/*
 * Indicates that modesets where no heads are increasing resource requirements,
 * or no heads are decreasing resource requirements, can be done glitchlessly.
 */
#define LW0073_CTRL_SYSTEM_CAPS_GLITCHLESS_MODESET_SUPPORTED       1:0x20
/* Indicates the SW ACR is enabled for HDMI 2.1 due to Bug 3275257. */
#define LW0073_CTRL_SYSTEM_CAPS_HDMI21_SW_ACR_BUG_3275257          1:0x40

/* Size in bytes of display caps table. Keep in sync with # of fields above. */
#define LW0073_CTRL_SYSTEM_CAPS_TBL_SIZE   2U

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_CAPS_V2
 *
 * This command returns the set of display capabilities for the parent device
 * in the form of an array of unsigned bytes.  Display capabilities
 * include supported features and required workarounds for the display
 * engine(s) within the device, each represented by a byte offset into the
 * table and a bit position within that byte.  The set of display capabilities
 * will be normalized across all GPUs within the device (a feature capability
 * will be set only if it's supported on all GPUs while a required workaround
 * capability will be set if any of the GPUs require it).
 *
 *   [out] capsTbl
 *     The display caps bits will be transferred by the RM into this array of
 *     unsigned bytes.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_CAPS_V2 (0x730138U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_CAPS_V2_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_GET_CAPS_V2_PARAMS_MESSAGE_ID (0x38U)

typedef struct LW0073_CTRL_SYSTEM_GET_CAPS_V2_PARAMS {
    LwU8 capsTbl[LW0073_CTRL_SYSTEM_CAPS_TBL_SIZE];
} LW0073_CTRL_SYSTEM_GET_CAPS_V2_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_NUM_HEADS
 *
 * This commands returns the number of heads supported by the specified
 * subdevice and available for use by displays.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   flags
 *     This parameter specifies optional flags to be used to while retrieving
 *     the number of heads.
 *     Possible valid flags are:
 *       LW0073_CTRL_SYSTEM_GET_NUM_HEADS_CLIENT
 *         This flag is used to request the number of heads that are
 *         lwrrently in use by an LW client using a user display class
 *         instance (see LW15_VIDEO_LUT_LWRSOR_DAC for an examle).  If this
 *         flag is disabled then the total number of heads supported is
 *         returned.
 *   numHeads
 *     This parameter returns the number of usable heads for the specified
 *     subdevice.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_NUM_HEADS (0x730102U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_NUM_HEADS_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_GET_NUM_HEADS_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW0073_CTRL_SYSTEM_GET_NUM_HEADS_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 flags;
    LwU32 numHeads;
} LW0073_CTRL_SYSTEM_GET_NUM_HEADS_PARAMS;

/* valid get num heads flags */
#define LW0073_CTRL_SYSTEM_GET_NUM_HEADS_FLAGS_CLIENT              0:0
#define LW0073_CTRL_SYSTEM_GET_NUM_HEADS_FLAGS_CLIENT_DISABLE (0x00000000U)
#define LW0073_CTRL_SYSTEM_GET_NUM_HEADS_FLAGS_CLIENT_ENABLE  (0x00000001U)


#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_DMI_SCANLINE
 *
 * This command returns the current DMI scanline of the specified head on the
 * specified subdevice. This is not supported on all GPUs.  Iff it is
 * supported, the LW0073_CTRL_SYSTEM_CAPS_GET_DMI_SCANLINE_SUPPORTED capability
 * should be exported.
 * This interface should not be used when in vga mode, so the client has to
 * ensure that a driver has loaded and the platform does not have fsdos or
 * other cases where vga can overtake the display.
 *
 *   subDeviceInstance (in)
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   head (in)
 *     This parameter specifies the head for which the active display
 *     should be retrieved.  This value must be between zero and the
 *     maximum number of heads supported by the subdevice. This ctrl call
 *     returns error if the head does not have a valid mode enabled.
 *   lwrrentDmiScanline (out)
 *     This parameter returns the current DMI scanline value for the specified
 *     head. If bIlwblank is TRUE then this parameter indicates the time in
 *     microseconds which is left until the scanout of the next frame starts.
 *     If this control is not available (for example when the head is in vga mode)
 *     LW0073_CTRL_CMD_SYSTEM_GET_DMI_SCANLINE_ILWALID_VALUE is returned.
 *   bIlwblank (out)
 *     This parameter indicates whether dmi is in blank or not. This also
 *     defines the meaning of lwrrentDmiScanline.
 *   bIsoHubLastRequestLineSupported (out)
 *     This parameter specifies whether isoHubLastRequestLine and
 *     dmiVblankDurationMicroSeconds are supported (this is hw dependent).
 *   isoHubLastRequestLine (out)
 *     This parameter returns the last scanline which was requested by the iso
 *     to be read out from the framebuffer. It doesn't tell whether the
 *     scanlines has already been read from the framebuffer. When the last
 *     scanline of a frame has been fetched this values stays at this value
 *     until the fetching for the next frame starts.
 *     If this control is not available (for example when the head is in vga mode)
 *     LW0073_CTRL_CMD_SYSTEM_GET_DMI_SCANLINE_ILWALID_VALUE is returned.
 *   dmiVblankDurationMicroSeconds (out)
 *     This parameter returns the vblank duration taked into account by dmi.
 *     This value influences when isoHubLastRequestLine starts with requesting
 *     framebuffer reads for the next frame.
 *     If this control is not available (for example when the head is in vga mode)
 *     LW0073_CTRL_CMD_SYSTEM_GET_DMI_SCANLINE_ILWALID_VALUE is returned.
 *   bStereoEyeSupported (out)
 *     This parameter specifies whether stereoEye reporting is supported (this
 *     is hw dependent). Note that this value doesn't actually reflect whether
 *     given head is really in stereo mode.
 *   stereoEye (out)
 *     If supported (ie bStereoEyeSupported is TRUE), this parameter returns
 *     either LW0073_CTRL_CMD_SYSTEM_GET_DMI_SCANLINE_RIGHT_EYE or
 *     LW0073_CTRL_CMD_SYSTEM_GET_DMI_SCANLINE_LEFT_EYE, reflecting the
 *     stereo eye that is lwrrently scanned out. Although this value typically
 *     changes at the beginning of vblank, the exact guarantee isn't more
 *     accurate than "somewhere in vblank".
 *     NOTE that stereoEye is RG state which typically lags behind DMI state,
 *     ie while RG state reflects frame N, DMI state might already reflect
 *     frame N+1!
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_DMI_SCANLINE               (0x730107U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_DMI_SCANLINE_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_SYSTEM_GET_DMI_SCANLINE_ILWALID_VALUE 0xFFFFFFFFU
#define LW0073_CTRL_CMD_SYSTEM_GET_DMI_SCANLINE_RIGHT_EYE     0x00000000U
#define LW0073_CTRL_CMD_SYSTEM_GET_DMI_SCANLINE_LEFT_EYE      0x00000001U

#define LW0073_CTRL_SYSTEM_GET_DMI_SCANLINE_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW0073_CTRL_SYSTEM_GET_DMI_SCANLINE_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  head;
    LwU32  lwrrentDmiScanline;
    LwBool bIlwblank;
    LwBool bIsoHubLastRequestLineSupported;
    LwU32  isoHubLastRequestLine;
    LwU32  dmiVblankDurationMicroSeconds;
    LwBool bStereoEyeSupported;
    LwU32  stereoEye;
} LW0073_CTRL_SYSTEM_GET_DMI_SCANLINE_PARAMS;

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




/*
 * LW0073_CTRL_CMD_SYSTEM_GET_SCANLINE
 *
 * This command returns the current RG scanline of the specified head on the
 * specified subdevice. To get the DMI scanline on supported chips, use
 * LW0073_CTRL_CMD_SYSTEM_GET_DMI_SCANLINE
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   head
 *     This parameter specifies the head for which the active display
 *     should be retrieved.  This value must be between zero and the
 *     maximum number of heads supported by the subdevice.
 *   lwrrentScanline
 *     This parameter returns the current RG scanline value for the specified
 *     head.  If the head does not have a valid mode enabled then a scanline
 *     value of 0xffffffff is returned.
 *   bStereoEyeSupported (out)
 *     This parameter specifies whether stereoEye reporting is supported (this
 *     is hw dependent). Note that this value doesn't actually reflect whether
 *     given head is really in stereo mode.
 *   stereoEye (out)
 *     If supported (ie bStereoEyeSupported is TRUE), this parameter returns
 *     either LW0073_CTRL_SYSTEM_GET_SCANLINE_PARAMS_RIGHT_EYE or
 *     LW0073_CTRL_SYSTEM_GET_SCANLINE_PARAMS_LEFT_EYE, reflecting the
 *     stereo eye that is lwrrently scanned out. Although this value typically
 *     changes at the beginning of vblank, the exact guarantee isn't more
 *     accurate than "somewhere in vblank".
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_SCANLINE           (0x730108U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_SCANLINE_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_SYSTEM_GET_SCANLINE_RIGHT_EYE 0x00000000U
#define LW0073_CTRL_CMD_SYSTEM_GET_SCANLINE_LEFT_EYE  0x00000001U

#define LW0073_CTRL_SYSTEM_GET_SCANLINE_PARAMS_MESSAGE_ID (0x8U)

typedef struct LW0073_CTRL_SYSTEM_GET_SCANLINE_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  head;
    LwU32  lwrrentScanline;
    LwBool bStereoEyeSupported;
    LwU32  stereoEye;
} LW0073_CTRL_SYSTEM_GET_SCANLINE_PARAMS;


#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_VBLANK_COUNTER
 *
 * This command returns the current VBlank counter of the specified head on the
 * specified subdevice.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   head
 *     This parameter specifies the head for which the vblank counter
 *     should be retrieved.  This value must be between zero and the
 *     maximum number of heads supported by the subdevice.
 *   verticalBlankCounter
 *     This parameter returns the vblank counter value for the specified
 *     head. If the display mode is not valid or vblank not active then
 *     the verticalBlankCounter value is undefined.
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_VBLANK_COUNTER (0x730109U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_VBLANK_COUNTER_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_GET_VBLANK_COUNTER_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW0073_CTRL_SYSTEM_GET_VBLANK_COUNTER_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 head;
    LwU32 verticalBlankCounter;
} LW0073_CTRL_SYSTEM_GET_VBLANK_COUNTER_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_VBLANK_ENABLE
 *
 * This command returns the current VBlank enable status for the specified
 * head.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   head
 *     This parameter specifies the head for which the vblank status
 *     should be retrieved.  This value must be between zero and the
 *     maximum number of heads supported by the subdevice.
 *   bEnabled
 *     This parameter returns the vblank enable status for the specified head.
 *     A value of LW_FALSE indicates that vblank interrupts are not lwrrently
 *     enabled while a value of LW_TRUE indicates that vblank are lwrrently
 *     enabled.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_VBLANK_ENABLE (0x73010aU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_VBLANK_ENABLE_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_GET_VBLANK_ENABLE_PARAMS_MESSAGE_ID (0xAU)

typedef struct LW0073_CTRL_SYSTEM_GET_VBLANK_ENABLE_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  head;
    LwBool bEnabled;
} LW0073_CTRL_SYSTEM_GET_VBLANK_ENABLE_PARAMS;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW0073_CTRL_CMD_SYSTEM_GET_SUPPORTED
 *
 * This command returns the set of supported display IDs for the specified
 * subdevice in the form of a 32bit display mask.  State from internal
 * display connectivity tables is used to determine the set of possible
 * display connections for the GPU.  The presence of a display in the
 * display mask only indicates the display is supported.  The connectivity
 * status of the display should be determined using the
 * LW0073_CTRL_CMD_SYSTEM_GET_CONNECT_STATE command.  The displayMask
 * value returned by LW0073_CTRL_CMD_SYSTEM_GET_SUPPORTED is static
 * and will remain consistent across boots of the system.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayMask
 *     This parameter returns a LW0073_DISPLAY_MASK value describing the set
 *     of displays supported by the subdevice.  An enabled bit in displayMask
 *     indicates the support of a display device with that displayId.
 *   displayMaskDDC
 *     This parameter returns a LW0073_DISPLAY_MASK value, indicating the
 *     subset of displayMask that supports DDC.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_SUPPORTED (0x730120U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_SUPPORTED_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_GET_SUPPORTED_PARAMS_MESSAGE_ID (0x20U)

typedef struct LW0073_CTRL_SYSTEM_GET_SUPPORTED_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayMask;
    LwU32 displayMaskDDC;
} LW0073_CTRL_SYSTEM_GET_SUPPORTED_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_CONNECT_STATE
 *
 * This command can be used to check the presence of a mask of display
 * devices on the specified subdevice.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   flags
 *     This parameter specifies optional flags to be used while retrieving
 *     the connection state information.
 *     Here are the current defined fields:
 *       LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_FLAGS_METHOD
 *         A client uses this field to indicate what method it wishes the
 *         system to use when determining the presence of attached displays.
 *         Possible values are:
 *            LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_FLAGS_METHOD_DEFAULT
 *              The system decides what method to use.
 *            LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_FLAGS_METHOD_CACHED
 *              Return the last full detection state for the display mask.
 *                   safety.)
 *            LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_FLAGS_METHOD_ECONODDC
 *              Ping the DDC address of the given display mask to check for
 *              a connected device. This is a lightweight method to check
 *              for a present device.
 *       LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_FLAGS_DDC
 *         A client uses this field to indicate whether to allow DDC during
 *         this detection or to not use it.
 *         Possible values are:
 *            LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_FLAGS_DDC_DEFAULT
 *              The system will use DDC as needed for each display.
 *            LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_FLAGS_DDC_DISABLE
 *              The system will not use DDC for any display. If DDC is
 *              disabled, this detection state will not be cached.
 *       LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_FLAGS_LOAD
 *         A client uses this field to indicate whether to detect loads
 *         during this detection or to not use it.
 *         Possible values are:
 *            LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_FLAGS_LOAD_DEFAULT
 *              The system will use load detection as needed for each display.
 *            LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_FLAGS_LOAD_DISABLE
 *              The system will not use load detection for any display. If
 *              load detection is disabled, this detection state will not
 *              be cached.
 *   displayMask
 *     This parameter specifies an LW0073_DISPLAY_MASK value describing
 *     the set of displays for which connectivity status is to be checked.
 *     If a display is present then the corresponding bit in the display
 *     mask is left enabled.  If the display is not present then the
 *     corresponding bit in the display mask is disabled.  Upon return this
 *     parameter contains the subset of displays in the mask that are
 *     connected.
 *
 *     If displayMask includes bit(s) that correspond to a TV encoder, the
 *     result will be simply 'yes' or 'no' without any indication of which
 *     connector(s) are actually attached.  For fine-grained TV attachment
 *     detection, please see LW0073_CTRL_CMD_TV_GET_ATTACHMENT_STATUS.
 *   retryTimeMs
 *     This parameter is an output to this command.  In case of
 *     LWOS_STATUS_ERROR_RETRY return status, this parameter returns the time
 *     duration in milli-seconds after which client should retry this command.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LWOS_STATUS_ERROR_RETRY
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_CONNECT_STATE (0x730122U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_PARAMS_MESSAGE_ID (0x22U)

typedef struct LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 flags;
    LwU32 displayMask;
    LwU32 retryTimeMs;
} LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_PARAMS;

/* valid get connect state flags */
#define LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_FLAGS_METHOD                  1:0
#define LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_FLAGS_METHOD_DEFAULT  (0x00000000U)
#define LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_FLAGS_METHOD_CACHED   (0x00000001U)
#define LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_FLAGS_METHOD_ECONODDC (0x00000002U)
#define LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_FLAGS_DDC                     4:4
#define LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_FLAGS_DDC_DEFAULT     (0x00000000U)
#define LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_FLAGS_DDC_DISABLE     (0x00000001U)
#define LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_FLAGS_LOAD                    5:5
#define LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_FLAGS_LOAD_DEFAULT    (0x00000000U)
#define LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_FLAGS_LOAD_DISABLE    (0x00000001U)
#define LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_FLAGS_VBLANK                  6:6
#define LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_FLAGS_VBLANK_DEFAULT  (0x00000000U)
#define LW0073_CTRL_SYSTEM_GET_CONNECT_STATE_FLAGS_VBLANK_SAFE     (0x00000001U)


/*
 * LW0073_CTRL_CMD_SYSTEM_GET_HOTPLUG_CONFIG
 *
 * This command can be used to retrieve dynamic hotplug state information that
 * are lwrrently recorded by the RM. This information can be used by the client
 * to determine which displays to detect after a hotplug event oclwrs.  Or if
 * the client knows that this device generates a hot plug/unplug signal on all
 * connectors, then this can be used to lwll displays from detection.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   flags
 *     This parameter specifies optional flags to be used while retrieving
 *     or changing the hotplug configuration.
 *       No flags are lwrrently defined.
 *   hotplugEventMask
 *     For _GET_HOTPLUG_CONFIG, this returns which connectors the client
 *     has asked for notifications for, when a hotplug event is detected.
 *     Events can only be provided for connectors whose displayID is set
 *     by the system in the hotplugInterruptible field.
 *   hotplugPollable
 *     For _GET_HOTPLUG_CONFIG, this returns which connectors are pollable
 *     in some non-destructive fashion.
 *   hotplugInterruptible
 *     For _GET_HOTPLUG_CONFIG, this returns which connectors are capable
 *     of generating interrupts.
 *
 *     This display mask specifies an LW0073_DISPLAY_MASK value describing
 *     the set of displays that have seen a hotplug or hotunplug event
 *     sometime after the last valid EDID read.  If the device never has
 *     a valid EDID read, then it will always be listed here.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */


#define LW0073_CTRL_CMD_SYSTEM_GET_HOTPLUG_CONFIG                  (0x730123U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_SET_HOTPLUG_CONFIG_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_GET_SET_HOTPLUG_CONFIG_PARAMS_MESSAGE_ID (0x23U)

typedef struct LW0073_CTRL_SYSTEM_GET_SET_HOTPLUG_CONFIG_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 flags;
    LwU32 hotplugEventMask;
    LwU32 hotplugPollable;
    LwU32 hotplugInterruptible;
    LwU32 hotplugAlwaysAttached;
} LW0073_CTRL_SYSTEM_GET_SET_HOTPLUG_CONFIG_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_HOTPLUG_STATE
 *
 * This command can be used to retrieve dynamic hotplug state information that
 * are lwrrently recorded by the RM. This information can be used by the client
 * to determine which displays to detect after a hotplug event oclwrs.  Or if
 * the client knows that this device generates a hot plug/unplug signal on all
 * connectors, then this can be used to lwll displays from detection.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   flags
 *     This parameter specifies optional flags to be used while retrieving
 *     the hotplug state information.
 *     Here are the current defined fields:
 *       LW0073_CTRL_SYSTEM_GET_HOTPLUG_STATE_FLAGS_LID
 *         A client uses this field to determine the lid state.
 *         Possible values are:
 *            LW0073_CTRL_SYSTEM_GET_HOTPLUG_STATE_FLAGS_LID_OPEN
 *              The lid is open.
 *            LW0073_CTRL_SYSTEM_GET_HOTPLUG_STATE_FLAGS_LID_CLOSED
 *              The lid is closed.  The client should remove devices as
 *              reported inside the
 *              LW0073_CTRL_SYSTEM_GET_CONNECT_POLICY_PARAMS.lidClosedMask.
 *   hotplugAfterEdidMask
 *     This display mask specifies an LW0073_DISPLAY_MASK value describing
 *     the set of displays that have seen a hotplug or hotunplug event
 *     sometime after the last valid EDID read.  If the device never has
 *     a valid EDID read, then it will always be listed here.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */


#define LW0073_CTRL_CMD_SYSTEM_GET_HOTPLUG_STATE (0x730124U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_HOTPLUG_STATE_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_GET_HOTPLUG_STATE_PARAMS_MESSAGE_ID (0x24U)

typedef struct LW0073_CTRL_SYSTEM_GET_HOTPLUG_STATE_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 flags;
    LwU32 hotplugAfterEdidMask;
} LW0073_CTRL_SYSTEM_GET_HOTPLUG_STATE_PARAMS;

/* valid get hoplug state flags */
#define LW0073_CTRL_SYSTEM_GET_HOTPLUG_STATE_FLAGS_LID                   0:0
#define LW0073_CTRL_SYSTEM_GET_HOTPLUG_STATE_FLAGS_LID_OPEN   (0x00000000U)
#define LW0073_CTRL_SYSTEM_GET_HOTPLUG_STATE_FLAGS_LID_CLOSED (0x00000001U)

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW0073_CTRL_CMD_SYSTEM_GET_HEAD_ROUTING_MAP
 *
 * This command can be used to retrieve the suggested head routing map
 * for the specified display mask.  A head routing map describes the
 * suggested crtc (or head) assignments for each display in the specified
 * mask.
 *
 * Up to MAX_DISPLAYS displays may be specified in the display mask.  Displays
 * are numbered from zero beginning with the lowest bit position set in the
 * mask.  The corresponding head assignment for each of specified displays can
 * then be found in the respective per-device field in the routing map.
 *
 * If a particular display cannot be successfully assigned a position in the
 * head routing map then it is removed from the display mask.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayMask
 *     This parameter specifies the LW0073_DISPLAY_MASK value for which
 *     the head routing map is desired.  Each enabled bit indicates
 *     a display device to include in the routing map. Enabled bits
 *     must represent supported displays as indicated by the
 *     LW0073_CTRL_CMD_SYSTEM_GET_SUPPORTED command.  If a particular
 *     display cannot be included in the routing map then it's corresponding
 *     bit in the displayMask will be disabled.  A return value of 0 in
 *     displayMask indicates that a head routing map could not be constructed
 *     with the given display devices.
 *   oldDisplayMask
 *     This optional parameter specifies a prior display mask to be
 *     used when generating the head routing map to be returned in
 *     headRoutingMap.  Displays set in oldDisplayMask are retained
 *     if possible in the new routing map.
 *   oldHeadRoutingMap
 *     This optional parameter specifies a prior head routing map to be
 *     used when generating the new routing map to be returned in
 *     headRoutingMap.  Head assignments in oldHeadRoutingMap are
 *     retained if possible in the new routing map.
 *   headRoutingMap
 *     This parameter returns the new head routing map.  This parameter
 *     is organized into eight distinct fields, each containing the head
 *     assignment for the corresponding display in display mask.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_HEAD_ROUTING_MAP           (0x730125U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_HEAD_ROUTING_MAP_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_GET_HEAD_ROUTING_MAP_PARAMS_MESSAGE_ID (0x25U)

typedef struct LW0073_CTRL_SYSTEM_GET_HEAD_ROUTING_MAP_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayMask;
    LwU32 oldDisplayMask;
    LwU32 oldHeadRoutingMap;
    LwU32 headRoutingMap;
} LW0073_CTRL_SYSTEM_GET_HEAD_ROUTING_MAP_PARAMS;

/* maximum number of allowed displays in a routing map */
#define LW0073_CTRL_SYSTEM_HEAD_ROUTING_MAP_MAX_DISPLAYS (8U)

/* per-display head assignments in a routing map */
#define LW0073_CTRL_SYSTEM_HEAD_ROUTING_MAP_DISPLAY0               3:0
#define LW0073_CTRL_SYSTEM_HEAD_ROUTING_MAP_DISPLAY1               7:4
#define LW0073_CTRL_SYSTEM_HEAD_ROUTING_MAP_DISPLAY2               11:8
#define LW0073_CTRL_SYSTEM_HEAD_ROUTING_MAP_DISPLAY3               15:12
#define LW0073_CTRL_SYSTEM_HEAD_ROUTING_MAP_DISPLAY4               19:16
#define LW0073_CTRL_SYSTEM_HEAD_ROUTING_MAP_DISPLAY5               23:20
#define LW0073_CTRL_SYSTEM_HEAD_ROUTING_MAP_DISPLAY6               27:24
#define LW0073_CTRL_SYSTEM_HEAD_ROUTING_MAP_DISPLAY7               31:28


/*
 * LW0073_CTRL_CMD_SYSTEM_GET_ACTIVE
 *
 * This command returns the active display ID for the specified head
 * on the specified subdevice.  The active display may be established
 * at system boot by low-level software and can then be later modified
 * by an LW client using a user display class instance (see
 * LW15_VIDEO_LUT_LWRSOR_DAC).
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   head
 *     This parameter specifies the head for which the active display
 *     should be retrieved.  This value must be between zero and the
 *     maximum number of heads supported by the subdevice.
 *   flags
 *     This parameter specifies optional flags to be used to while retrieving
 *     the active display information.
 *     Possible valid flags are:
 *       LW0073_CTRL_SYSTEM_GET_ACTIVE_FLAGS_CLIENT
 *         This flag is used to limit the search for the active display to
 *         that established by an LW client.  If this flag is not specified,
 *         then any active display is returned (setup at system boot by
 *         low-level software or later by an LW client).
 *   displayId
 *     This parameter returns the displayId of the active display.  A value
 *     of zero indicates no display is active.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_ACTIVE                (0x730126U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_ACTIVE_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_GET_ACTIVE_PARAMS_MESSAGE_ID (0x26U)

typedef struct LW0073_CTRL_SYSTEM_GET_ACTIVE_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 head;
    LwU32 flags;
    LwU32 displayId;
} LW0073_CTRL_SYSTEM_GET_ACTIVE_PARAMS;

/* valid get active flags */
#define LW0073_CTRL_SYSTEM_GET_ACTIVE_FLAGS_CLIENT                 0:0
#define LW0073_CTRL_SYSTEM_GET_ACTIVE_FLAGS_CLIENT_DISABLE (0x00000000U)
#define LW0073_CTRL_SYSTEM_GET_ACTIVE_FLAGS_CLIENT_ENABLE  (0x00000001U)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

/*
 * LW0073_CTRL_CMD_SYSTEM_SET_CONNECTORS_DIRTY
 *
 * This command can be used to check the presence of a mask of display
 * devices using full detection on the specified subdevice.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayMask
 *     This parameter specifies an LW0073_DISPLAY_MASK value describing
 *     the set of displays for which connectivity status to be marked as
 *     dirty. So, when the call comes for GET_CONNECT_STATE, it will do a
 *     full detection of these deviecs.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_CMD_SYSTEM_SET_CONNECTORS_DIRTY        (0x730127U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_SET_CONNECTORS_DIRTY_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_SET_CONNECTORS_DIRTY_PARAMS_MESSAGE_ID (0x27U)

typedef struct LW0073_CTRL_SYSTEM_SET_CONNECTORS_DIRTY_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayMask;
} LW0073_CTRL_SYSTEM_SET_CONNECTORS_DIRTY_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_LWR_CONNECTOR_MASK
 *
 * This command returns the mask of all connectors.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   connectorMask
 *     This parameter returns the union of the display mask of all connectors.
 *     It indicates we only need to detect the device from these masks.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_LWR_CONNECTOR_MASK (0x730128U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_LWR_CONNECTOR_MASK_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_GET_LWR_CONNECTOR_MASK_PARAMS_MESSAGE_ID (0x28U)

typedef struct LW0073_CTRL_SYSTEM_GET_LWR_CONNECTOR_MASK_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 connectorMask;
} LW0073_CTRL_SYSTEM_GET_LWR_CONNECTOR_MASK_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_BOOT_DISPLAY_MASK
 *
 * This command returns the initial boot display(s) in the form of
 * displayMask value.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *     This parameter must specify a value between zero and the total number
 *     of subdevices within the parent device.  This parameter should be set
 *     to zero for default behavior.
 *   displayMask
 *     This parameter returns display mask of LW0073_DISPLAY_MASK value.  It
 *     indicates the display device enabled during the boot.
 *   head
 *     This parameter specifies the head for which the active display mask
 *     should be retrieved, if the head is booted with display(s).  This value
 *     must be between zero and the maximum number of heads supported by the
 *     subdevice.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0073_CTRL_CMD_SYSTEM_GET_BOOT_DISPLAY_MASK (0x730140U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_BOOT_DISPLAY_MASK_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_GET_BOOT_DISPLAY_MASK_PARAMS_MESSAGE_ID (0x40U)

typedef struct LW0073_CTRL_SYSTEM_GET_BOOT_DISPLAY_MASK_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayMask;
    LwU32 head;
} LW0073_CTRL_SYSTEM_GET_BOOT_DISPLAY_MASK_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_STEREO_SUPPORT
 *
 * This command returns the support status of the LW stereo emitter
 * (also known as the stereo dongle). It reports if the stereo dongle
 * is present in terms of the USB interface initialized in Resman.
 * This provides a RmControl interface to the STEREO_DONGLE_SUPPORTED
 * command in stereoDongleControl.
 *
 * Parameters:
 * [OUT] support    the control word returned by stereoDongleControl
 * [IN]  head       (optional) head to be passed to stereoDongleControl
 *
 * Possible status values returned are:
 *   LW_ERR_NOT_SUPPORTED - stereo is not initialized on the GPU
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_STEREO_SUPPORT (0x730150U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_STEREO_SUPPORT_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_GET_STEREO_SUPPORT_PARAMS_MESSAGE_ID (0x50U)

typedef struct LW0073_CTRL_SYSTEM_GET_STEREO_SUPPORT_PARAMS {
    LwU32 support;
    LwU32 head;
} LW0073_CTRL_SYSTEM_GET_STEREO_SUPPORT_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_SET_STEREO_INDICATOR
 *
 * This command can be used to inform the RM that a client would like to
 * enable the system's stereo indicator LED, if present.
 *
 * Parameters:
 * [IN]  enable     Boolean value to enable (non-zero) or disable (zero)
 *                      the LED.
 * [OUT] status     The actual status of the operation. Will be one of
 *                      LW0073_CTRL_SYSTEM_STEREO_INDICATOR_STATUS
 *
 * Possible status values returned are:
 *   LW_OK - for all cases
 */
#define LW0073_CTRL_CMD_SYSTEM_SET_STEREO_INDICATOR (0x730151U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_CMD_SYSTEM_SET_STEREO_INDICATOR_PARAMS_MESSAGE_ID" */

typedef enum LW0073_CTRL_SYSTEM_STEREO_INDICATOR_STATUS {
    LW0073_CTRL_SYSTEM_STEREO_INDICATOR_STATUS_OK = 0,          /* Indicator set successfully */
    LW0073_CTRL_SYSTEM_STEREO_INDICATOR_STATUS_FAIL = 1,            /* Unable to change the indicator */
    LW0073_CTRL_SYSTEM_STEREO_INDICATOR_STATUS_NOT_SUPPORTED = 2,   /* System does not support indicator */
} LW0073_CTRL_SYSTEM_STEREO_INDICATOR_STATUS;

#define LW0073_CTRL_CMD_SYSTEM_SET_STEREO_INDICATOR_PARAMS_MESSAGE_ID (0x51U)

typedef struct LW0073_CTRL_CMD_SYSTEM_SET_STEREO_INDICATOR_PARAMS {
    LwU32 enable;
    LwU32 status;
} LW0073_CTRL_CMD_SYSTEM_SET_STEREO_INDICATOR_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_DISPLAY_POWER_SAVING_DIAG
 *
 * This command is used to do LwDPS diagnostics.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and
 *     the total number of subdevices within the parent device.  It should
 *     be set to zero for default behavior.
 *   displayId
 *     the displayId for which display configuration is desired
 *   bEnabled
 *     This parameter sets whether the diagnostic mode is enabled or disabled.
 *   hBpExt
 *     This parameter sets the value (in pixels) of the HBlank Back-Porch Extension.
 *     When bEnabled is false, the maximum HBlank Back-Porch Extension is returned.
 *   vFpExt
 *     This parameter sets the value (in pixels) of the VBlank Front-Porch Extension.
 *     When bEnabled is false, the maximum VBlank Front-Porch Extension is returned.
 *   vBpExt
 *     This parameter sets the value (in pixels) of the VBlank Back-Porch Extension.
 *     When bEnabled is false, the maximum VBlank Back-Porch Extension is returned.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0073_CTRL_CMD_SYSTEM_DISPLAY_POWER_SAVING_DIAG (0x730157U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_CMD_SYSTEM_DISPLAY_POWER_SAVING_DIAG_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_SYSTEM_DISPLAY_POWER_SAVING_DIAG_PARAMS_MESSAGE_ID (0x57U)

typedef struct LW0073_CTRL_CMD_SYSTEM_DISPLAY_POWER_SAVING_DIAG_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  displayId;
    LwBool bEnable;
    LwU32  hBpExt;
    LwU32  vFpExt;
    LwU32  vBpExt;
} LW0073_CTRL_CMD_SYSTEM_DISPLAY_POWER_SAVING_DIAG_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_DISPLAY_POWER_SAVING_STATE
 *
 * This command is used to determine whether lwDPS enabled or disabled
 *
 *   bEnabled
 *     This parameter returns whether lwDPS enabled or disabled
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_DISPLAY_POWER_SAVING_STATE (0x730158U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_CMD_SYSTEM_GET_DISPLAY_POWER_SAVING_STATE_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_SYSTEM_GET_DISPLAY_POWER_SAVING_STATE_PARAMS_MESSAGE_ID (0x58U)

typedef struct LW0073_CTRL_CMD_SYSTEM_GET_DISPLAY_POWER_SAVING_STATE_PARAMS {
    LwBool bEnabled;
} LW0073_CTRL_CMD_SYSTEM_GET_DISPLAY_POWER_SAVING_STATE_PARAMS;

#define LW0073_CTRL_CMD_SYSTEM_GET_DISPLAY_POWER_SAVING_STATE_ENABLE_OFF (0x00000000U)
#define LW0073_CTRL_CMD_SYSTEM_GET_DISPLAY_POWER_SAVING_STATE_ENABLE_ON  (0x00000001U)

/*
 * LW0073_CTRL_CMD_SYSTEM_SET_DISPLAY_POWER_SAVING_STATE
 *
 * This command is used to enable or disable lwDPS
 *
 *   bEnable
 *     This parameter indicates whether to enable or disable lwDPS
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0073_CTRL_CMD_SYSTEM_SET_DISPLAY_POWER_SAVING_STATE            (0x730159U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_CMD_SYSTEM_SET_DISPLAY_POWER_SAVING_STATE_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_SYSTEM_SET_DISPLAY_POWER_SAVING_STATE_PARAMS_MESSAGE_ID (0x59U)

typedef struct LW0073_CTRL_CMD_SYSTEM_SET_DISPLAY_POWER_SAVING_STATE_PARAMS {
    LwBool bEnable;
} LW0073_CTRL_CMD_SYSTEM_SET_DISPLAY_POWER_SAVING_STATE_PARAMS;

#define LW0073_CTRL_CMD_SYSTEM_SET_DISPLAY_POWER_SAVING_STATE_ENABLE_OFF (0x00000000U)
#define LW0073_CTRL_CMD_SYSTEM_SET_DISPLAY_POWER_SAVING_STATE_ENABLE_ON  (0x00000001U)

/*
 * LW0073_CTRL_SYSTEM_ACPI_ID_MAP
 *
 * This structure defines the mapping between the RM's displayId and the
 * defined ACPI IDs for each display.
 *   displayId
 *     This parameter is a handle to a single display output path from the
 *     GPU pins to the display connector.  Each display ID is defined by one bit.
 *     A zero in this parameter indicates a skip entry.
 *   acpiId
 *     This parameter defines the corresponding ACPI ID of the displayId.
 *   flags
 *     This parameter specifies optional flags that describe the association
 *     between the display ID and the ACPI ID.
 *       LW0073_CTRL_SYSTEM_ACPI_ID_MAP_ORIGIN
 *         This field describes where the ACPI was found.
 *           LW0073_CTRL_SYSTEM_ACPI_ID_MAP_ORIGIN_RM
 *             The ACPI ID was generated by RM code.
 *           LW0073_CTRL_SYSTEM_ACPI_ID_MAP_ORIGIN_DOD
 *             The ACPI ID was found via the ACPI _DOD call.
 *           LW0073_CTRL_SYSTEM_ACPI_ID_MAP_ORIGIN_CLIENT
 *             The ACPI ID was generated by RM Client and sent to RM. Note this
 *             must be set on a LW0073_CTRL_CMD_SYSTEM_SET_ACPI_ID_MAP call.
 *       LW0073_CTRL_SYSTEM_ACPI_ID_MAP_SNAG_UNDOCKED
 *         This flag explains that the ACPI ID is only valid when the system
 *         is undocked.  If this flag is not set, the ACPI ID is valid always.
 *       LW0073_CTRL_SYSTEM_ACPI_ID_MAP_SNAG_DOCKED
 *         This flag explains that the ACPI ID is only valid when the system
 *         is docked.  If this flag is not set, the ACPI ID is valid always.
 *       LW0073_CTRL_SYSTEM_ACPI_ID_MAP_DOD_BIOS_DETECT
 *         This flag is set only if the _DOD returns that the device can be
 *         detected by the system BIOS.  This flag is copied directly from
 *         the ACPI spec.
 *       LW0073_CTRL_SYSTEM_ACPI_ID_MAP_DOD_NON_VGA_OUTPUT
 *         This flag is set only if the _DOD returns that the device is
 *         a non-VGA device whose power is related to the VGA device.
 *         i.e. TV tuner, DVD decoder, Video capture. This flag is copied
 *         directly from the ACPI spec.
 *       LW0073_CTRL_SYSTEM_ACPI_ID_MAP_DOD_MULTIHEAD_ID
 *         This value is set only if the _DOD returns it.  The number
 *         indicates the head output of a multi-head device. This has no
 *         relation to the term, Head, lwrrently used in the RM today.
 *         This is strictly a copy of the value directly from the ACPI spec.
 *       LW0073_CTRL_SYSTEM_ACPI_ID_MAP_DOD_SCHEME
 *         This flag is set only if the _DOD returns that the acpiID follows
 *         the ACPI 3.0 spec.  This flag is copied directly from
 *         the ACPI spec.
 *
 */

typedef struct LW0073_CTRL_SYSTEM_ACPI_ID_MAP_PARAMS {
    LwU32 displayId;
    LwU32 acpiId;
    LwU32 flags;
} LW0073_CTRL_SYSTEM_ACPI_ID_MAP_PARAMS;

#define LW0073_CTRL_SYSTEM_ACPI_ID_MAP_ORIGIN                            1:0
#define LW0073_CTRL_SYSTEM_ACPI_ID_MAP_ORIGIN_RM                0x00000000U
#define LW0073_CTRL_SYSTEM_ACPI_ID_MAP_ORIGIN_DOD               0x00000001U
#define LW0073_CTRL_SYSTEM_ACPI_ID_MAP_ORIGIN_CLIENT            0x00000002U

#define LW0073_CTRL_SYSTEM_ACPI_ID_MAP_SNAG_UNDOCKED                     2:2
#define LW0073_CTRL_SYSTEM_ACPI_ID_MAP_SNAG_UNDOCKED_FALSE      0x00000000U
#define LW0073_CTRL_SYSTEM_ACPI_ID_MAP_SNAG_UNDOCKED_TRUE       0x00000001U

#define LW0073_CTRL_SYSTEM_ACPI_ID_MAP_SNAG_DOCKED                       3:3
#define LW0073_CTRL_SYSTEM_ACPI_ID_MAP_SNAG_DOCKED_FALSE        0x00000000U
#define LW0073_CTRL_SYSTEM_ACPI_ID_MAP_SNAG_DOCKED_TRUE         0x00000001U

#define LW0073_CTRL_SYSTEM_ACPI_ID_MAP_DOD_BIOS_DETECT                 16:16
#define LW0073_CTRL_SYSTEM_ACPI_ID_MAP_DOD_BIOS_DETECT_FALSE    0x00000000U
#define LW0073_CTRL_SYSTEM_ACPI_ID_MAP_DOD_BIOS_DETECT_TRUE     0x00000001U

#define LW0073_CTRL_SYSTEM_ACPI_ID_MAP_DOD_NON_VGA_OUTPUT              17:17
#define LW0073_CTRL_SYSTEM_ACPI_ID_MAP_DOD_NON_VGA_OUTPUT_FALSE 0x00000000U
#define LW0073_CTRL_SYSTEM_ACPI_ID_MAP_DOD_NON_VGA_OUTPUT_TRUE  0x00000001U

#define LW0073_CTRL_SYSTEM_ACPI_ID_MAP_DOD_MULTIHEAD_ID                20:18

#define LW0073_CTRL_SYSTEM_ACPI_ID_MAP_DOD_SCHEME                      31:31
#define LW0073_CTRL_SYSTEM_ACPI_ID_MAP_DOD_SCHEME_VENDOR        0x00000000U
#define LW0073_CTRL_SYSTEM_ACPI_ID_MAP_DOD_SCHEME_30            0x00000001U

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_ACPI_ID_MAP
 *
 * This command retrieves the mapping between the RM's displayId and the
 * defined ACPI IDs for each display.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and
 *     the total number of subdevices within the parent device.  It should
 *     be set to zero for default behavior.
 *   LW0073_CTRL_SYSTEM_ACPI_ID_MAP_PARAMS
 *     An array of display ID to ACPI ids with flags for each description.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *     Only returned if subdeviceInstance was not valid.
 */

#define LW0073_CTRL_CMD_SYSTEM_GET_ACPI_ID_MAP                  (0x73015aU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_ACPI_ID_MAP_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_ACPI_ID_MAP_MAX_DISPLAYS             (16U)

#define LW0073_CTRL_SYSTEM_GET_ACPI_ID_MAP_PARAMS_MESSAGE_ID (0x5AU)

typedef struct LW0073_CTRL_SYSTEM_GET_ACPI_ID_MAP_PARAMS {
    LwU32                                 subDeviceInstance;
    LW0073_CTRL_SYSTEM_ACPI_ID_MAP_PARAMS acpiIdMap[LW0073_CTRL_SYSTEM_ACPI_ID_MAP_MAX_DISPLAYS];
} LW0073_CTRL_SYSTEM_GET_ACPI_ID_MAP_PARAMS;
/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW0073_CTRL_CMD_SYSTEM_GET_INTERNAL_DISPLAYS
 *
 * This command returns the set of internal (safe) display IDs for the specified
 * subdevice in the form of a 32bit display mask. Safe means the displays do
 * not require copy protection as they are on the motherboard.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   internalDisplaysMask
 *     This parameter returns a LW0073_DISPLAY_MASK value describing the set
 *     of displays that are internal (safe) and which do not require copy
 *     protection schemes.
 *   availableInternalDisplaysMask
 *     This parameter returns a LW0073_DISPLAY_MASK value describing the set
 *     of displays that are internal and available for use.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_INTERNAL_DISPLAYS (0x73015bU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_INTERNAL_DISPLAYS_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_GET_INTERNAL_DISPLAYS_PARAMS_MESSAGE_ID (0x5BU)

typedef struct LW0073_CTRL_SYSTEM_GET_INTERNAL_DISPLAYS_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 internalDisplaysMask;
    LwU32 availableInternalDisplaysMask;
} LW0073_CTRL_SYSTEM_GET_INTERNAL_DISPLAYS_PARAMS;

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

/*
 * LW0073_CTRL_CMD_SYSTEM_ACPI_SUBSYSTEM_ACTIVATED
 *
 * This command is used to notify RM that all subdevices are ready for ACPI
 * calls. The caller must make sure that the OS is ready to handle the ACPI
 * calls for each ACPI ID. So, this call must be done after the OS has
 * initialized all the display ACPI IDs to this subdevice.
 * Besides, the ACPI spec provides a function for the display drivers to read
 * the EDID directly from the SBIOS for each display's ACPI ID. This function
 * is used to override the EDID found from a I2C or DPAux based transaction.
 * This command will also attempt to call the ACPI _DDC function to read the
 * EDID from the SBIOS for all displayIDs. If an EDID is found from this call,
 * the RM will store that new EDID in the EDID buffer of that OD.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *
 * Possible status values returned are:
 * LW_OK
 * LW_ERR_ILWALID_PARAM_STRUCT
 * LW_ERR_NOT_SUPPORTED
 *
 */

#define LW0073_CTRL_SYSTEM_ACPI_SUBSYSTEM_ACTIVATED_PARAMS_MESSAGE_ID (0x5LW)

typedef struct LW0073_CTRL_SYSTEM_ACPI_SUBSYSTEM_ACTIVATED_PARAMS {
    LwU32 subDeviceInstance;
} LW0073_CTRL_SYSTEM_ACPI_SUBSYSTEM_ACTIVATED_PARAMS;

#define LW0073_CTRL_CMD_SYSTEM_ACPI_SUBSYSTEM_ACTIVATED (0x73015lw) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_ACPI_SUBSYSTEM_ACTIVATED_PARAMS_MESSAGE_ID" */



/*
 * LW0073_CTRL_CMD_SYSTEM_FAST_LVDS_SWITCH
 *
 * This will control the fast LVDS switch on a Hybrid system where the LVDS
 * data mux is connected to the dGPU GPIO. This call can request the switch
 * or it can request whether this board supports the switch.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   method
 *     This parameter indicates the client's request.  Possible values are:
 *       LW0073_FAST_LVDS_SWITCH_METHOD_SUPPORT
 *         The client can use this to determine if this functionality is
 *         supported or not.  If supported, the return value will be
 *         LW_OK.  Otherwise, LW_ERR_NOT_SUPPORTED.
 *       LW0073_FAST_LVDS_SWITCH_METHOD_SWITCH_TO_IGPU
 *         The client sends this method to request a switch from dGPU to iGPU.
 *       LW0073_FAST_LVDS_SWITCH_METHOD_SWITCH_TO_DGPU
 *         The client sends this method to request a switch from iGPU to dGPU.
 *       LW0073_FAST_LVDS_SWITCH_METHOD_FORCE_ENABLE_SUPPORT
 *         The client sends this method to forcibly enable the support for
 *         this feature.  On some MXM systems, the GPIO function of the MXM
 *         card is unknown until we get information from the SBIOS.  The Hybrid
 *         spec allows for a CAP bit to tell us that the dGPU GPIO does control
 *         the fast LVDS Mux switch.  If that cap bit is set, this function
 *         should be called to enable the support.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_CMD_SYSTEM_FAST_LVDS_SWITCH         (0x73015dU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_FAST_LVDS_SWITCH_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_FAST_LVDS_SWITCH_PARAMS_MESSAGE_ID (0x5DU)

typedef struct LW0073_CTRL_SYSTEM_FAST_LVDS_SWITCH_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 method;
} LW0073_CTRL_SYSTEM_FAST_LVDS_SWITCH_PARAMS;

/* valid method values */
#define LW0073_FAST_LVDS_SWITCH_METHOD_SUPPORT              (0x00000000U)
#define LW0073_FAST_LVDS_SWITCH_METHOD_SWITCH_TO_IGPU       (0x00000001U)
#define LW0073_FAST_LVDS_SWITCH_METHOD_SWITCH_TO_DGPU       (0x00000002U)
#define LW0073_FAST_LVDS_SWITCH_METHOD_FORCE_ENABLE_SUPPORT (0x00000003U)

/*
 * To support RMCTRLs for BOARDOBJGRP_E255, we were required to increase the
 * XAPI limit to 16K. It was observed that XP does NOT allow the static array
 * size greater then 10K and this was causing the DVS failure. So we are using
 * the OLD XAPI value i.e. 4K for LW0073_CTRL_SYSTEM_SRM_BUFFER_MAX while
 * internally we are using the new updated XAPI value i.e. 16K.
 */
#define XAPI_ELWELOPE_MAX_PAYLOAD_SIZE_OLD                  4096U

/*
 * LW0073_CTRL_SYSTEM_SRM_CHUNK
 *
 * Several control commands require an SRM, which may be larger than the
 * available buffer. Therefore, this structure is used to transfer the needed
 * data.
 *
 *   startByte
 *     Index of the byte in the SRM buffer at which the current chunk of data
 *     starts. If this value is 0, it indicates the start of a new SRM. A
 *     value other than 0 indicates additional data for an SRM.
 *   numBytes
 *     Size in bytes of the current chunk of data.
 *   totalBytes
 *     Size in bytes of the entire SRM.
 *   srmBuffer
 *     Buffer containing the current chunk of SRM data.
 */
/* Set max SRM size to the XAPI max, minus some space for other fields */
#define LW0073_CTRL_SYSTEM_SRM_BUFFER_MAX                   (0xe00U) /* finn: Evaluated from "(XAPI_ELWELOPE_MAX_PAYLOAD_SIZE_OLD - 512)" */

typedef struct LW0073_CTRL_SYSTEM_SRM_CHUNK {
    LwU32 startByte;
    LwU32 numBytes;
    LwU32 totalBytes;

    /* C form: LwU8    srmBuffer[LW0073_CTRL_SYSTEM_SRM_BUFFER_MAX]; */
    LwU8  srmBuffer[LW0073_CTRL_SYSTEM_SRM_BUFFER_MAX];
} LW0073_CTRL_SYSTEM_SRM_CHUNK;

/*
 * LW0073_CTRL_CMD_SYSTEM_VALIDATE_SRM
 *
 * Instructs the RM to validate the SRM for use by HDCP revocation. The SRM
 * may be larger than the buffer provided by the API. In that case, the SRM is
 * sent in chunks no larger than LW0073_CTRL_SYSTEM_SRM_BUFFER_MAX bytes.
 *
 * Upon completion of the validation, which is an asynchronous operation, the
 * client will receive a <PLACE_HOLDER_EVENT> event. Alternatively, the client
 * may poll for completion of SRM validation via
 * LW0073_CTRL_CMD_SYSTEM_GET_SRM_STATUS.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   srm
 *     A chunk of the SRM.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_NOT_READY
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_WARN_MORE_PROCESSING_REQUIRED
 *   LW_ERR_INSUFFICIENT_RESOURCES
 */
#define LW0073_CTRL_CMD_SYSTEM_VALIDATE_SRM (0x73015eU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_VALIDATE_SRM_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_VALIDATE_SRM_PARAMS_MESSAGE_ID (0x5EU)

typedef struct LW0073_CTRL_SYSTEM_VALIDATE_SRM_PARAMS {
    LwU32                        subDeviceInstance;
    LW0073_CTRL_SYSTEM_SRM_CHUNK srm;
} LW0073_CTRL_SYSTEM_VALIDATE_SRM_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_SRM_STATUS
 *
 * Retrieves the status of the request to validate the SRM. If a request to
 * validate an SRM is still pending, LW_ERR_NOT_READY will be
 * returned and the status will not be updated.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   status
 *     Result of the last SRM validation request.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_NOT_READY
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_SRM_STATUS (0x73015fU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_SRM_STATUS_PARAMS_MESSAGE_ID" */

typedef enum LW0073_CTRL_SYSTEM_SRM_STATUS {
    LW0073_CTRL_SYSTEM_SRM_STATUS_OK = 0,      // Validation succeeded
    LW0073_CTRL_SYSTEM_SRM_STATUS_FAIL = 1,        // Validation request failed
    LW0073_CTRL_SYSTEM_SRM_STATUS_BAD_FORMAT = 2,  // Bad SRM format
    LW0073_CTRL_SYSTEM_SRM_STATUS_ILWALID = 3,      // Bad SRM signature
} LW0073_CTRL_SYSTEM_SRM_STATUS;

#define LW0073_CTRL_SYSTEM_GET_SRM_STATUS_PARAMS_MESSAGE_ID (0x5FU)

typedef struct LW0073_CTRL_SYSTEM_GET_SRM_STATUS_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 status;
} LW0073_CTRL_SYSTEM_GET_SRM_STATUS_PARAMS;

/*
 * LW0073_CTRL_CMD_GET_DEVICE_DISPLAY_MODE
 * This command is used for retrieving information about the
 * current display mode on the specified head. This information is
 * used by headfifolog tool as it helps to analyse possible reasons
 * of underflow. Underflow is a condition when head FB
 * lacks bandwidth and it rans out of pixel data for display device fetch.
 * Underflowing causes visible glitches in the display scanout.
 *
 *  displayId
 *    the displayId for which display configuration is desired
 *  activeWidth
 *    Visible screen width(horizontal resolution)
 *  activeHeight
 *    Visible screen height(vertical resolution)
 *  totalWidth
 *    Total width of entire raster size, it is more than activeWidth
 *    because of blanking interval
 *  totalHeight
 *    Total height of entire raster size, it is more than activeHeight
 *    because of blanking interval
 *  depth
 *    Bits per pixel (color quality)
 *  frequency
 *    Display refresh rate
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_DEVICE_DISPLAY_MODE (0x730160U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_DEVICE_DISPLAY_MODE_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_GET_DEVICE_DISPLAY_MODE_PARAMS_MESSAGE_ID (0x60U)

typedef struct LW0073_CTRL_SYSTEM_GET_DEVICE_DISPLAY_MODE_PARAMS {
    LwU32 displayId;
    LwU32 activeWidth;
    LwU32 activeHeight;
    LwU32 totalWidth;
    LwU32 totalHeight;
    LwU32 depth;
    LwU32 frequency;
} LW0073_CTRL_SYSTEM_GET_DEVICE_DISPLAY_MODE_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_HDCP_REVOCATION_CHECK
 *
 * Performs the HDCP revocation process. Given the supplied SRM, all attached
 * devices will be checked to see if they are on the revocation list or not.
 *
 *   srm
 *     The SRM to do the revocation check against. For SRMs larger than
 *     LW0073_CTRL_SYSTEM_SRM_BUFFER_MAX, the caller will need to break up the
 *     SRM into chunks and make multiple calls.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_NOT_READY
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_WARN_MORE_PROCESSING_REQUIRED
 *   LW_ERR_INSUFFICIENT_RESOURCES
 */
#define LW0073_CTRL_CMD_SYSTEM_HDCP_REVOCATION_CHECK (0x730161U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_HDCP_REVOCATE_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_HDCP_REVOCATE_PARAMS_MESSAGE_ID (0x61U)

typedef struct LW0073_CTRL_SYSTEM_HDCP_REVOCATE_PARAMS {
    LW0073_CTRL_SYSTEM_SRM_CHUNK srm;
} LW0073_CTRL_SYSTEM_HDCP_REVOCATE_PARAMS;

/*
 * LW0073_CTRL_CMD_UPDATE_SRM
 *
 * Updates the SRM used by RM for HDCP revocation checks. The SRM must have
 * been previously validated as authentic.
 *
 *   srm
 *     The SRM data. For SRMs larger than LW0073_CTRL_SYSTEM_SRM_BUFFER_MAX,
 *     the caller will need to break up the SRM into chunks and make multiple
 *     calls.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_NOT_READY
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_WARN_MORE_PROCESSING_REQUIRED
 *   LW_ERR_INSUFFICIENT_RESOURCES
 */
#define LW0073_CTRL_CMD_SYSTEM_UPDATE_SRM (0x730162U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_UPDATE_SRM_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_UPDATE_SRM_PARAMS_MESSAGE_ID (0x62U)

typedef struct LW0073_CTRL_SYSTEM_UPDATE_SRM_PARAMS {
    LW0073_CTRL_SYSTEM_SRM_CHUNK srm;
} LW0073_CTRL_SYSTEM_UPDATE_SRM_PARAMS;

/*
 * LW0073_CTRL_SYSTEM_CONNECTOR_INFO
 *
 * This structure describes a single connector table entry.
 *
 *   type
 *     This field specifies the connector type.
 *   displayMask
 *     This field specifies the the displayMask to which the connector belongs.
 *   location
 *     This field specifies the placement of the connector on the platform.
 *   hotplug
 *     This field specifies hotplug capabilities (if any) for the connector.
 */
typedef struct LW0073_CTRL_SYSTEM_CONNECTOR_INFO {
    LwU32 type;
    LwU32 displayMask;
    LwU32 location;
    LwU32 hotplug;
} LW0073_CTRL_SYSTEM_CONNECTOR_INFO;

/* valid type values */
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_VGA_15_PIN               (0x00000000U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_DVI_A                    (0x00000001U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_POD_VGA_15_PIN           (0x00000002U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_TV_COMPOSITE             (0x00000010U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_TV_SVIDEO                (0x00000011U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_TV_SVIDEO_BO_COMPOSITE   (0x00000012U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_TV_COMPONENT             (0x00000013U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_TV_SCART                 (0x00000014U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_TV_SCART_EIAJ4120        (0x00000014U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_TV_EIAJ4120              (0x00000017U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_PC_POD_HDTV_YPRPB        (0x00000018U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_PC_POD_SVIDEO            (0x00000019U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_PC_POD_COMPOSITE         (0x0000001AU)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_DVI_I_TV_SVIDEO          (0x00000020U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_DVI_I_TV_COMPOSITE       (0x00000021U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_DVI_I_TV_SV_BO_COMPOSITE (0x00000022U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_DVI_I                    (0x00000030U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_DVI_D                    (0x00000031U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_ADC                      (0x00000032U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_LFH_DVI_I_1              (0x00000038U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_LFH_DVI_I_2              (0x00000039U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_LFH_SVIDEO               (0x0000003AU)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_BNC                      (0x0000003LW)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_LVDS_SPWG                (0x00000040U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_LVDS_OEM                 (0x00000041U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_LVDS_SPWG_DET            (0x00000042U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_LVDS_OEM_DET             (0x00000043U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_TVDS_OEM_ATT             (0x00000045U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_VGA_15_PIN_UNDOCKED      (0x00000050U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_VGA_15_PIN_DOCKED        (0x00000051U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_DVI_I_UNDOCKED           (0x00000052U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_DVI_I_DOCKED             (0x00000053U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_DVI_D_UNDOCKED           (0x00000052U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_DVI_D_DOCKED             (0x00000053U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_DP_EXT                   (0x00000056U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_DP_INT                   (0x00000057U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_DP_EXT_UNDOCKED          (0x00000058U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_DP_EXT_DOCKED            (0x00000059U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_3PIN_DIN_STEREO          (0x00000060U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_HDMI_A                   (0x00000061U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_AUDIO_SPDIF              (0x00000062U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_HDMI_C_MINI              (0x00000063U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_LFH_DP_1                 (0x00000064U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_LFH_DP_2                 (0x00000065U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_TYPE_VIRTUAL_WFD              (0x00000070U)

/* valid hotplug values */
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_HOTPLUG_A_SUPPORTED           (0x00000001U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_HOTPLUG_B_SUPPORTED           (0x00000002U)

/*
 * Lw0073_CTRL_CMD_SYSTEM_GET_CONNECTOR_TABLE
 *
 * This command can be used to retrieve display connector information.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   version
 *     This parameter returns the version of the connector table.
 *   platform
 *     This parameter returns the type of platform of the associated subdevice.
 *   connectorTableEntries
 *     This parameter returns the number of valid entries in the connector
 *     table.
 *   connectorTable
 *     This parameter returns the connector information in the form of an
 *     array of LW0073_CTRL_SYSTEM_CONNECTOR_INFO structures.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_CONNECTOR_TABLE                      (0x730165U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_CONNECTOR_TABLE_PARAMS_MESSAGE_ID" */

/* maximum number of connector table entries */
#define LW0073_CTRL_SYSTEM_GET_CONNECTOR_TABLE_MAX_ENTRIES              (16U)

#define LW0073_CTRL_SYSTEM_GET_CONNECTOR_TABLE_PARAMS_MESSAGE_ID (0x65U)

typedef struct LW0073_CTRL_SYSTEM_GET_CONNECTOR_TABLE_PARAMS {
    LwU32                             subDeviceInstance;
    LwU32                             version;
    LwU32                             platform;
    LwU32                             connectorTableEntries;
    /*
     * C form:
     * LW0073_CTRL_SYSTEM_CONNECTOR_INFO connectorTable[LW0073_CTRL_SYSTEM_CONNECTOR_TABLE_MAX_ENTRIES];
     */
    LW0073_CTRL_SYSTEM_CONNECTOR_INFO connectorTable[LW0073_CTRL_SYSTEM_GET_CONNECTOR_TABLE_MAX_ENTRIES];
} LW0073_CTRL_SYSTEM_GET_CONNECTOR_TABLE_PARAMS;

/* valid version values */
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_VERSION_30                     (0x00000030U)

/* valid platform values */
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_PLATFORM_DEFAULT_ADD_IN_CARD   (0x00000000U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_PLATFORM_TWO_PLATE_ADD_IN_CARD (0x00000001U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_PLATFORM_MOBILE_ADD_IN_CARD    (0x00000008U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_PLATFORM_MXM_MODULE            (0x00000009U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_PLATFORM_MOBILE_BACK           (0x00000010U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_PLATFORM_MOBILE_BACK_LEFT      (0x00000011U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_PLATFORM_MOBILE_BACK_DOCK      (0x00000018U)
#define LW0073_CTRL_SYSTEM_CONNECTOR_INFO_PLATFORM_CRUSH_DEFAULT         (0x00000020U)

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW0073_CTRL_CMD_SYSTEM_GET_BOOT_DISPLAYS
 *
 * This command returns a mask of boot display IDs.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   bootDisplayMask
 *     This parameter returns the mask of boot display IDs.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_BOOT_DISPLAYS                         (0x730166U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_BOOT_DISPLAYS_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_GET_BOOT_DISPLAYS_PARAMS_MESSAGE_ID (0x66U)

typedef struct LW0073_CTRL_SYSTEM_GET_BOOT_DISPLAYS_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 bootDisplayMask;
} LW0073_CTRL_SYSTEM_GET_BOOT_DISPLAYS_PARAMS;

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_ACTIVE_HEAD
 *
 * This command returns the head (display controller) being used by the
 * specified display.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayMask
 *     This parameter is the mask of display in question.
 *   activeHead
 *     This parameter returns the head in use by the specified display.
 *     If error then this number is undefined.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_PARAM_STRUCT
 *    LW_ERR_GENERIC
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_ACTIVE_HEAD (0x730167U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_ACTIVE_HEAD_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_GET_ACTIVE_HEAD_PARAMS_MESSAGE_ID (0x67U)

typedef struct LW0073_CTRL_SYSTEM_GET_ACTIVE_HEAD_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayMask;
    LwU32 activeHead;
} LW0073_CTRL_SYSTEM_GET_ACTIVE_HEAD_PARAMS;


/*
 * LW0073_CTRL_CMD_SYSTEM_EXELWTE_ACPI_METHOD
 *
 * This command is used to execute general MXM ACPI methods.
 *
 * method
 *   This parameter identifies the MXM ACPI API to be ilwoked.
 *   Valid values for this parameter are:
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_MXMI
 *       This value specifies that the MXMI API is to ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_MXMS
 *       This value specifies that the MXMS API is to ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_MXMX
 *       This value specifies that the MXMX API is to ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_GPUON
 *       This value specifies that the Hybrid GPU ON API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_GPUOFF
 *       This value specifies that the Hybrid GPU OFF API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_GPUSTA
 *       This value specifies that the Hybrid GPU STA API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_MXDS
 *       This value specifies that the Hybrid GPU MXDS API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_LWHG_MXMX
 *       This value specifies that the Hybrid GPU MXMX API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DOS
 *       This value specifies that the Hybrid GPU DOS API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_ROM
 *       This value specifies that the Hybrid GPU ROM API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DOD
 *       This value specifies that the Hybrid GPU DOD API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_SUPPORT
 *       This value specifies that the Hybrid GPU DSM subfunction SUPPORT
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_HYBRIDCAPS
 *       This value specifies that the Hybrid GPU DSM subfunction SUPPORT
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_POLICYSELECT
 *       This value specifies that the Hybrid GPU DSM subfunction POLICYSELECT
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_POWERCONTROL
 *       This value specifies that the Hybrid GPU DSM subfunction POWERCONTROL
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_PLATPOLICY
 *       This value specifies that the Hybrid GPU DSM subfunction PLATPOLICY
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_DISPLAYSTATUS
 *       This value specifies that the Hybrid GPU DSM subfunction DISPLAYSTATUS
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_MDTL
 *       This value specifies that the Hybrid GPU DSM subfunction MDTL
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_HCSMBLIST
 *       This value specifies that the Hybrid GPU DSM subfunction HCSMBLIST
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_HCSMBADDR
 *       This value specifies that the Hybrid GPU DSM subfunction HCSMBADDR
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_HCREADBYTE
 *       This value specifies that the Hybrid GPU DSM subfunction HCREADBYTE
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_HCSENDBYTE
 *       This value specifies that the Hybrid GPU DSM subfunction HCSENDBYTES
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_HCGETSTATUS
 *       This value specifies that the Hybrid GPU DSM subfunction HCGETSTATUS
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_HCTRIGDDC
 *       This value specifies that the Hybrid GPU DSM subfunction HCTRIGDDC
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_HCGETDDC
 *       This value specifies that the Hybrid GPU DSM subfunction HCGETDDC
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DCS
 *       This value specifies that the Hybrid GPU DCS API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_MXM_MXSS
 *       This value specifies that the DSM MXM subfunction MXSS
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_MXM_MXMI
 *       This value specifies that the DSM MXM subfunction MXMI
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_MXM_MXMS
 *       This value specifies that the DSM MXM subfunction MXMS
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_MXM_MXPP
 *       This value specifies that the DSM MXM subfunction MXPP
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_MXM_MXDP
 *       This value specifies that the DSM MXM subfunction MXDP
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_MXM_MDTL
 *       This value specifies that the DSM MXM subfunction MDTL
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_MXM_MXCB
 *       This value specifies that the DSM MXM subfunction MXCB
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GENERIC_CTL_REMAPFUNC
 *       This value specifies the DSM generic remapping should return function
 *       and subfunction when this API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GENERIC_HYBRIDCAPS
 *       This value specifies that the generic DSM subfunction HYBRIDCAPS
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GENERIC_POLICYSELECT
 *       This value specifies that the generic DSM subfunction POLICYSELECT
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GENERIC_PLATPOLICY
 *       This value specifies that the generic DSM subfunction PLATPOLICY
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GENERIC_DISPLAYSTATUS
 *       This value specifies that the generic DSM subfunction DISPLAYSTATUS
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GENERIC_MDTL
 *       This value specifies that the generic DSM subfunction MDTL
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GENERIC_GETOBJBYTYPE
 *       This value specifies that the generic DSM subfunction GETOBJBYTYPE
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GENERIC_GETALLOBJS
 *       This value specifies that the generic DSM subfunction GETALLOBJS
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GENERIC_GETEVENTLIST
 *       This value specifies that the generic DSM subfunction GETEVENTLIST
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GENERIC_GETBACKLIGHT
 *       This value specifies that the generic DSM subfunction GETBACKLIGHT
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GENERIC_CTL_TESTSUBFUNCENABLED
 *       This value specifies the testIfDsmSubFunctionEnabled test should
 *       be done for the func/subfunction when this API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GENERIC_CTL_GETSUPPORTEDFUNC
 *       This value specifies the list of supported generic dsm functions
 *       should be returned.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_LWOP_OPTIMUSCAPS
 *       This value specifies that the DSM LWOP subfunction OPTIMUSCAPS
 *       API is to be ilwoked.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_LWOP_OPTIMUSFLAG
 *       This value specifies that the DSM LWOP subfunction OPTIMUSFLAG
 *       API is to be ilwoked. This API will set a Flag in sbios to Indicate
 *       that HD Audio Controller is disable/Enabled from GPU Config space.
 *       This flag will be used by sbios to restore Audio state after resuming
 *       from s3/s4.
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_WMMX_LWOP_GPUON
 *      This value specifies that the WMMX (WMI-ACPI) GPON methods has to be ilwoked
 *      this call should happen below DPC level from any client.
 * inData
 *   This parameter specifies the method-specific input buffer.  Data is
 *   passed to the specified API using this buffer.  For display related
 *   APIs the associated display mask can be found at a byte offset within
 *   the inData buffer using the following method-specific values:
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_MXMX_DISP_MASK_OFFSET
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_MXDS_DISP_MASK_OFFSET
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_LWHG_MXMX_DISP_MASK_OFFSET
 *     LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DOS_DISP_MASK_OFFSET
 * inDataSize
 *   This parameter specifies the size of the inData buffer in bytes.
 * outStatus
 *   This parameter returns the status code from the associated ACPI call.
 * outData
 *   This parameter specifies the method-specific output buffer.  Data
 *   is returned by the specified API using this buffer.
 * outDataSize
 *   This parameter specifies the size of the outData buffer in bytes.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0073_CTRL_CMD_SYSTEM_EXELWTE_ACPI_METHOD (0x730168U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_PARAMS_MESSAGE_ID (0x68U)

typedef struct LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_PARAMS {
    LwU32 method;
    LW_DECLARE_ALIGNED(LwP64 inData, 8);
    LwU16 inDataSize;
    LwU32 outStatus;
    LW_DECLARE_ALIGNED(LwP64 outData, 8);
    LwU16 outDataSize;
} LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_PARAMS;


/* valid method parameter values */
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_MXMX                               (0x00000002U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_MXMX_DISP_MASK_OFFSET              (0x00000001U)

#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_GPUON                              (0x00000003U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_GPUOFF                             (0x00000004U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_GPUSTA                             (0x00000005U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_MXDS                               (0x00000006U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_LWHG_MXMX                          (0x00000007U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DOS                                (0x00000008U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_ROM                                (0x00000009U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DOD                                (0x0000000aU)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_SUPPORT                        (0x0000000bU)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_HYBRIDCAPS                     (0x0000000lw)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_POLICYSELECT                   (0x0000000dU)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_POWERCONTROL                   (0x0000000eU)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_PLATPOLICY                     (0x0000000fU)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_DISPLAYSTATUS                  (0x00000010U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_MDTL                           (0x00000011U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_HCSMBLIST                      (0x00000012U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_HCSMBADDR                      (0x00000013U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_HCREADBYTE                     (0x00000014U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_HCSENDBYTE                     (0x00000015U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_HCGETSTATUS                    (0x00000016U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_HCTRIGDDC                      (0x00000017U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_HCGETDDC                       (0x00000018U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DCS                                (0x00000019U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_MXM_MXSS                       (0x0000001aU)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_MXM_MXMI                       (0x0000001bU)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_MXM_MXMS                       (0x0000001lw)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_MXM_MXPP                       (0x0000001dU)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_MXM_MXDP                       (0x0000001eU)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_MXM_MDTL                       (0x0000001fU)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_MXM_MXCB                       (0x00000020U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_MXM_GETEVENTLIST               (0x00000021U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GETMEMTABLE                    (0x00000022U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GETMEMCFG                      (0x00000023U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GETOBJBYTYPE                   (0x00000024U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GETALLOBJS                     (0x00000025U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GENERIC_CTL_REMAPFUNC          (0x00000026U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GENERIC_DISPLAYSTATUS          (0x0000002aU)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GENERIC_MDTL                   (0x0000002bU)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GENERIC_GETOBJBYTYPE           (0x0000002lw)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GENERIC_GETALLOBJS             (0x0000002dU)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GENERIC_GETEVENTLIST           (0x0000002eU)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GENERIC_GETBACKLIGHT           (0x0000002fU)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GENERIC_CTL_TESTSUBFUNCENABLED (0x00000030U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GENERIC_CTL_GETSUPPORTEDFUNC   (0x00000031U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_LWOP_OPTIMUSCAPS               (0x00000032U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_WMMX_LWOP_GPUON                    (0x00000033U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_LWOP_OPTIMUSFLAG               (0x00000034U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GENERIC_GETLICENSE             (0x00000035U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GENERIC_GETCALLBACKS           (0x00000036U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_NBCI_SUPPORTFUNCS              (0x00000037U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_NBCI_PLATCAPS                  (0x00000038U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_NBCI_PLATPOLICY                (0x00000039U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_GENERIC_MSTL                   (0x0000003aU)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DSM_LWGPS_FUNC_SUPPORT             (0x0000003bU)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_NBCI_MXDS                          (0x0000003lw)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_NBCI_MXDM                          (0x0000003dU)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_NBCI_MXID                          (0x0000003eU)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_NBCI_LRST                          (0x0000003fU)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DDC_EDID                           (0x00000040U)

/* valid input buffer offset values */
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_MXDS_DISP_MASK_OFFSET              (0x00000004U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_LWHG_MXMX_DISP_MASK_OFFSET         (0x00000004U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DOS_DISP_MASK_OFFSET               (0x00000004U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_NBCI_MXDS_DISP_MASK_OFFSET         (0x00000004U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_NBCI_MXDM_DISP_MASK_OFFSET         (0x00000004U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_NBCI_MXID_DISP_MASK_OFFSET         (0x00000004U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_NBCI_LRST_DISP_MASK_OFFSET         (0x00000004U)
#define LW0073_CTRL_SYSTEM_EXELWTE_ACPI_METHOD_DDC_EDID_DISP_MASK_OFFSET          (0x00000004U)

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_DISPLAY_CHANGE_INHIBIT
 *
 * This command returns the current display change inhibit flags.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayChangeInhibit
 *     This parameter returns the current display change inhibit mask.  A value
 *     of zero indicates that display changes are not presently inhibited.
 *     Legal values for this parameter include a mask of the following:
 *       LW0073_CTRL_SYSTEM_DISPLAY_CHANGE_INHIBIT_SPAN
 *         This value indicates that display changes are inhibited due to
 *         to spanning mode being enabled.
 *       LW0073_CTRL_SYSTEM_DISPLAY_CHANGE_INHIBIT_VIDEO
 *         This value indicates that display changes are inhibited due to
 *         video playback being enabled.
 *       LW0073_CTRL_SYSTEM_DISPLAY_CHANGE_INHIBIT_3D_FULLSCREEN
 *         This value indicates that display changes are inhibited due to
 *         fullscreen 3D mode being enabled.
 *       LW0073_CTRL_SYSTEM_DISPLAY_CHANGE_INHIBIT_DOS_FULLSCREEN
 *         This value indicates that display changes are inhibited due to
 *         fullscreen DOS mode being enabled.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_DISPLAY_CHANGE_INHIBIT                         (0x730169U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_DISPLAY_CHANGE_INHIBIT_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_GET_DISPLAY_CHANGE_INHIBIT_PARAMS_MESSAGE_ID (0x69U)

typedef struct LW0073_CTRL_SYSTEM_GET_DISPLAY_CHANGE_INHIBIT_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayChangeInhibit;
} LW0073_CTRL_SYSTEM_GET_DISPLAY_CHANGE_INHIBIT_PARAMS;

/* valid displayChangeInhibit mask values */
#define LW0073_CTRL_SYSTEM_DISPLAY_CHANGE_INHIBIT_SPAN           (0x00000001U)
#define LW0073_CTRL_SYSTEM_DISPLAY_CHANGE_INHIBIT_VIDEO          (0x00000002U)
#define LW0073_CTRL_SYSTEM_DISPLAY_CHANGE_INHIBIT_3D_FULLSCREEN  (0x00000004U)
#define LW0073_CTRL_SYSTEM_DISPLAY_CHANGE_INHIBIT_DOS_FULLSCREEN (0x00000008U)

/*
 * LW0073_CTRL_CMD_SYSTEM_SET_DESKTOP_PRIMARY_DISPLAY
 *
 * This command allows the client to select the Desktop Primary Display.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     The public ID of the Output Display which has been selected to be the
 *     Desktop Primary Display.
 *
 */
#define LW0073_CTRL_CMD_SYSTEM_SET_DESKTOP_PRIMARY_DISPLAY       (0x73016aU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_SET_DESKTOP_PRIMARY_DISPLAY_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_SET_DESKTOP_PRIMARY_DISPLAY_PARAMS_MESSAGE_ID (0x6AU)

typedef struct LW0073_CTRL_SYSTEM_SET_DESKTOP_PRIMARY_DISPLAY_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
} LW0073_CTRL_SYSTEM_SET_DESKTOP_PRIMARY_DISPLAY_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_VBIOS_MODESET_DISPLAY_MASK
 *
 * This command can be used to return a displayMask that indicates
 * which displayIds have had a vbios modeset since boot or resume
 * up until we do a driver level modeset on those displays. At least on
 * EVO display HW, the RM knows when the VBIOS has grabbed the display
 * away from the driver.  And we can model this on pre-EVO GPUs as well
 * since we know when the VBIOS has posted or not.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and
 *     the total number of subdevices within the parent device.  It should
 *     be set to zero for default behavior.
 *   displayMask
 *     This parameter returns a LW0073_DISPLAY_MASK value describing the set
 *     of displays supported by the subdevice that last had a vbios modeset
 *     on the display HW.  An enabled bit in displayMask indicates a display
 *     device with that displayId that meets this criteria.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_VBIOS_MODESET_DISPLAY_MASK (0x730170U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_VBIOS_MODESET_DISPLAY_MASK_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_GET_VBIOS_MODESET_DISPLAY_MASK_PARAMS_MESSAGE_ID (0x70U)

typedef struct LW0073_CTRL_SYSTEM_GET_VBIOS_MODESET_DISPLAY_MASK_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayMask;
} LW0073_CTRL_SYSTEM_GET_VBIOS_MODESET_DISPLAY_MASK_PARAMS;

/*
 * LW0073_CTRL_SYSTEM_DISPLAY_PROPS
 *
 * The following macros describe the display properties that can be
 * set with LW0073_CTRL_CMD_SYSTEM_SET_DISPLAY_PROPS and retrieved
 * with LW0073_CTLR_CMD_SYSTEM_GET_DISPLAY_PROPS.
 *
 *   LW0073_CTRL_SYSTEM_DISPLAY_PROPS_FULLSCREEN_DXG
 *     This property indicates if a full-screen DXG application has control
 *     of the display.
 *
 */
#define LW0073_CTRL_SYSTEM_DISPLAY_PROPS_FULLSCREEN_DXG           0:0
#define LW0073_CTRL_SYSTEM_DISPLAY_PROPS_FULLSCREEN_DXG_DISABLED (0x000000000ULL)
#define LW0073_CTRL_SYSTEM_DISPLAY_PROPS_FULLSCREEN_DXG_ENABLED  (0x000000001ULL)

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_DISPLAY_PROPS
 *
 * This command returns a mask of enabled display properties.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayProps
 *     LW0073_CTRL_SYSTEM_DISPLAY_PROPS_FULL_SCREEN_DXG
 *       When enabled indicates that a full-screen DXG application is
 *       presently running.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_DISPLAY_PROPS                 (0x730171U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_DISPLAY_PROPS_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_GET_DISPLAY_PROPS_PARAMS_MESSAGE_ID (0x71U)

typedef struct LW0073_CTRL_SYSTEM_GET_DISPLAY_PROPS_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayProps;
} LW0073_CTRL_SYSTEM_GET_DISPLAY_PROPS_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_SET_DISPLAY_PROPS
 *
 * This command sets display properties.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayProps
 *     LW0073_CTRL_SYSTEM_DISPLAY_PROPS_FULL_SCREEN_DXG
 *       Set to indicate a full-screen DXG application owns the display.
 *       Clear to indicate a full-screen DXG application no longer owns the
 *       display.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW0073_CTRL_CMD_SYSTEM_SET_DISPLAY_PROPS (0x730172U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_SET_DISPLAY_PROPS_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_SET_DISPLAY_PROPS_PARAMS_MESSAGE_ID (0x72U)

typedef struct LW0073_CTRL_SYSTEM_SET_DISPLAY_PROPS_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayProps;
} LW0073_CTRL_SYSTEM_SET_DISPLAY_PROPS_PARAMS;



/*
 * LW0073_CTRL_CMD_SYSTEM_SET_SMOOTH_BRIGHTNESS
 *
 * This command instructs the RM to enable or disable smooth brightness.
 * Smooth brightness forces all brightness calls to happen over a
 * preset transition time to improve the user visual experience.
 * This is a global enable that will affect all brightness contexts
 * on the GPU.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   bEnable
 *     This parameter enables or disables smooth brightness
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW0073_CTRL_CMD_SYSTEM_SET_SMOOTH_BRIGHTNESS (0x730174U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_SET_SMOOTH_BRIGHTNESS_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_SET_SMOOTH_BRIGHTNESS_PARAMS_MESSAGE_ID (0x74U)

typedef struct LW0073_CTRL_SYSTEM_SET_SMOOTH_BRIGHTNESS_PARAMS {
    LwU32  subDeviceInstance;
    LwBool bEnable;
} LW0073_CTRL_SYSTEM_SET_SMOOTH_BRIGHTNESS_PARAMS;
/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW0073_CTRL_CMD_SYSTEM_SET_VRR
 *
 * This command instructs the RM to configure Variable Refresh Rate (VRR).
 * VRR uses framelock & extended Vblank to provide a faster scanout for frames
 * that take longer to render than the time period for a frame at a given
 * refresh rate (I.E. 16 MS @ 60Hz) rather than resend the current frame, the
 * scanout is held off until either the new frame is rendered, or the monitor
 * must have its image refreshed.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *
 *   displayId
 *     This is no longer used and only provided to prevent breaking client
 *     builds.  Once all of the clients have cleaned up, this will be
 *     removed.
 *
 *   bEnable
 *     This parameter enables or disables VRR.
 *
 *   bEnableFrameSmoothing
 *     Enables the frame smoothing algorithm when releasing the frames in
 *     VRR mode.
 *
 *   bEnableDCBalancing
 *     Enables the DC balance algorithm when releasing the frames in
 *     VRR mode.  This is a special mode of frame smoothing and requires
 *     bEnableFrameSmoothing to be set to LW_TRUE.
 *
 *   bEnableLowFreqFrameTimeEstimation
 *     Enables improved scheduling for redundant frames when the framerate
 *     approaches the lower bound.
 *
 *   frameSmoothingTC
 *     A fixed point value with LW0073_CTRL_SYSTEM_SET_VRR_FS_TC_PRECISION
 *     fractional bits. Used as the alpha in an exponential averaging filter
 *     for the frame smoothing and DC balance computations:
 *
 *     y(n) = (alpha * y(n-1)) + ((1-alpha) * x(n))
 *
 *     This value is required to be less than LW0073_CTRL_SYSTEM_SET_VRR_FS_TC_UNIT_VALUE.
 *     If this value is 0, the time constant is set to LW0073_CTRL_SYSTEM_SET_VRR_FS_TC_DEFAULT.
 *
 *   bUsingMethodBased
 *      This field's purpose is to notify RM that client is intending to use method based
 *      VRR and does not need RM handling. When its value is set to LW_TRUE, client is
 *      expected to manage the ELV/RG release and RM will not enable display method traps
 *      to handle this cases. Only applicable for Volta_and_later.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 *    LW_ERR_ILWALID_STATE
 */
#define LW0073_CTRL_CMD_SYSTEM_SET_VRR (0x730175U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_SET_VRR_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_SET_VRR_PARAMS_MESSAGE_ID (0x75U)

typedef struct LW0073_CTRL_SYSTEM_SET_VRR_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  displayId;
    LwBool bEnable;
    LwBool bEnableFrameSmoothing;
    LwBool bEnableDCBalancing;
    LwBool bEnableLowFreqFrameTimeEstimation;
    LW_DECLARE_ALIGNED(LwU64 frameSmoothingTC, 8);
    LwBool bEnableVrrOnSli;
    LwBool bUsingMethodBased;
} LW0073_CTRL_SYSTEM_SET_VRR_PARAMS;

#define LW0073_CTRL_SYSTEM_SET_VRR_FS_TC_PRECISION  (12U)
#define LW0073_CTRL_SYSTEM_SET_VRR_FS_TC_UNIT_VALUE (0x1000U) /* finn: Evaluated from "(1 << LW0073_CTRL_SYSTEM_SET_VRR_FS_TC_PRECISION)" */
#define LW0073_CTRL_SYSTEM_SET_VRR_FS_TC_DEFAULT    (0xd99U) /* finn: Evaluated from "((850 << LW0073_CTRL_SYSTEM_SET_VRR_FS_TC_PRECISION) / 1000)" */

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_VRR_CONFIG
 *
 * This command provides the caller with the relevant info on VRR.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *
 *   bCapable
 *     This specifies if the current card is capable of VRR.
 *
 *   bGC6Capable
 *     This specifies if the GPU is capable of GC6.
 *
 *   bFLSupported
 *     This specifies if the LWSR framelock pin is supported.
 *
 *   frameLockPin
 *     This specifies the frame lock pin that will be used for VRR.
 *     Will return LW0073_CTRL_SYSTEM_GET_VRR_CONFIG_NO_VALID_LOCK_PIN if no pin
 *     can be found.
 *
 *   bSLI
 *     Indicates if the pin allocated was selected based on SLI config.
 *
 *   bVrrEnableByHulk
 *     Indicates that VRR has been enabled by Hulk.
 *
 *   Possible status values returned are:
 *    LWOS_STATUS_SUCCESS
 *    LWOS_STATUS_ERROR_ILWALID_ARGUMENT
 *    LWOS_STATUS_ERROR_ILWALID_STATE
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_VRR_CONFIG       (0x730176U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_VRR_CONFIG_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_GET_VRR_CONFIG_PARAMS_MESSAGE_ID (0x76U)

typedef struct LW0073_CTRL_SYSTEM_GET_VRR_CONFIG_PARAMS {
    LwU32  subDeviceInstance;
    LwBool bCapable;
    LwBool bGC6Capable;
    LwBool bFLSupported;
    LwU32  frameLockPin;
    LwU32  bSli;
    LwBool bVrrEnableByHulk;
} LW0073_CTRL_SYSTEM_GET_VRR_CONFIG_PARAMS;

#define LW0073_CTRL_SYSTEM_GET_VRR_CONFIG_NO_VALID_LOCK_PIN (0xFFU)


#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW0073_CTRL_SYSTEM_DISPLAY_ID_PROPS
 *
 * This structure describes the properties for each of the display paths on
 * on a subdevice.
 *
 *   displayId
 *     This parameter is a handle to a single display output path from the GPU
 *     pins to the display connector.
 *   props
 *     This parameter defines the corresponding properties of the displayId.
 *     Possible values for this field are:
 *       LW0073_CTRL_DISPLAY_ID_PROP_IS_VIRTUAL
 *         This value indicates whether the display is a virtual device (when
 *         set to TRUE) or a physical device (when set to FALSE).
 */

typedef struct LW0073_CTRL_SYSTEM_DISPLAY_ID_PROPS {
    LwU32 displayId;
    LwU32 props;
} LW0073_CTRL_SYSTEM_DISPLAY_ID_PROPS;

/* legal values for prop parameter */
#define LW0073_CTRL_DISPLAY_ID_PROP_IS_VIRTUAL                  0:0
#define LW0073_CTRL_DISPLAY_ID_PROP_IS_VIRTUAL_FALSE    (0x00000000U)
#define LW0073_CTRL_DISPLAY_ID_PROP_IS_VIRTUAL_TRUE     (0x00000001U)

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_DISPLAY_ID_PROPS
 *
 * This command returns the properties for each display device on the
 * specified subdevice.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   numDisplays
 *     This parameter will return the number of valid entries in displayDeviceProps.
 *   displayDeviceProps
 *     This parameter is an array that returns the display properties for each
 *     valid display devices on this subDeviceInstance. See the description
 *     of LW0073_CTRL_SYSTEM_DISPLAY_ID_PROPS for more details.
 *
 * Possible status values returned include:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_GPU_NOT_FULL_POWER
 */

#define LW0073_CTRL_CMD_SYSTEM_GET_DISPLAY_ID_PROPS     (0x730178U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_CMD_SYSTEM_GET_DISPLAY_ID_PROPS_PARAMS_MESSAGE_ID" */

/* maximum number of display prop entries */
#define LW0073_CTRL_SYSTEM_DISPLAY_ID_PROPS_MAX_ENTRIES 16U

#define LW0073_CTRL_CMD_SYSTEM_GET_DISPLAY_ID_PROPS_PARAMS_MESSAGE_ID (0x78U)

typedef struct LW0073_CTRL_CMD_SYSTEM_GET_DISPLAY_ID_PROPS_PARAMS {
    LwU32                               subDeviceInstance;
    LwU32                               numDisplays;
    LW0073_CTRL_SYSTEM_DISPLAY_ID_PROPS displayDeviceProps[LW0073_CTRL_SYSTEM_DISPLAY_ID_PROPS_MAX_ENTRIES];
} LW0073_CTRL_CMD_SYSTEM_GET_DISPLAY_ID_PROPS_PARAMS;
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW0073_CTRL_CMD_SYSTEM_ADD_VRR_HEAD
 *
 * This command instructs the RM to configure the specified head for use as a VRR head,
 * but it does NOT enable VRR. VRR uses framelock & extended Vblank to provide a faster scanout
 * for frames that take longer to render than the time period for a frame at a given
 * refresh rate (I.E. 16 MS @ 60Hz) rather than resend the current frame, the
 * scanout is held off until either the new frame is rendered, or the monitor
 * must have its image refreshed.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *
 *   displayId
 *     The public ID of the Output Display which is to be used for VRR.
 *
 *   headGroupId
 *     This indicates the head-group that this head will be part of.  All
 *     heads in a head group flip together.  This must be less than
 *     LW0073_CTRL_SYSTEM_ADD_VRR_HEAD_MAX_HEADGROUPS
 *
 *   frontPorchExtension
 *     This parameter specifies the value to extend the front porch by
 *     to extend Vblank for VRR.
 *
 *   backPorchExtension
 *     This parameter specifies the value to extend the back porch by
 *     to extend Vblank for VRR.
 *
 *   bBlockChannel
 *     This boolean indicates if the base channel associated with this head should be blocked as
 *     part of VRR processing. setting this to TRUE will cause the channel to be blocked when the
 *     NOOP method is processed by the base channel. the channel will remain blocked until the
 *     scan out is released.
 *
 *   bInitStateSuspended
 *     Setting this to LW_TRUE will force the initial state of the head to VRR_SUSPENDED
 *     Setting this to LW_FALSE will force the initial state of the head to VRR_ACTIVE
 *
 *   frameSemaphore
 *     This parameter is a pointer to an array of semaphores which
 *     will be monitored after each Vblank to determine when frames have
 *     finished rendering so that that framelock can be released & scanout
 *     resumed.
 *
 *   frameSemaphoreCount
 *     the number of semaphores in the framesemaphore array for indicating render complete.
 *     if the count is set to 0, then semaphores will not be checked for this head.
 *
 *   timeout
 *     This parameter specifies the amount of time in milliseconds after the
 *     most recent flip to wait for the render complete. If render complete is
 *     not indicated via the frame semaphore within this
 *     time period, scanout will be released with the current frame.
 *     Specifying 0 for this value will disable forced release being triggered
 *     by this head. Note that even if the frameSemaphore is set to NULL, a
 *     timeout may be specified, so that when in clone or surround mode, VRR
 *     can still maintain the monitor image.
 *
 *   timeoutMicroseconds
 *     This parameter specifies the amount of time in microseconds after the
 *     most recent flip to wait for the render complete. If render complete is
 *     not indicated via the frame semaphore within this
 *     time period, scanout will be released with the current frame.
 *     Specifying 0 for this value will disable forced release being triggered
 *     by this head. Note that even if the frameSemaphore is set to NULL, a
 *     timeout may be specified, so that when in clone or surround mode, VRR
 *     can still maintain the monitor image.
 *     If this is set to a non-zero value, it will override the timeout
 *     parameter
 *
 *   bOneShotMode
 *     Indicates whether to use the one shot mode available in the Maxwell display (class 947d).
 *     This flag is ignored if not supported.
 *
 *   bOneShotPinReleaseEnable
 *     If we are using one-shot mode, this indicates whether we should use the register
 *     or pin release method.
 *
 *   bExternalFramelockToggle
 *     Indicates whether the framelock pin needs to be triggered by the RM
 *     or an external device like LWSR will do so.
 *
 *   externalFramelockTimeout
 *     The time in milliseconds to wait for a framelock response from an external
 *     self-refresh TCON.
 *
 *   externalDeviceWAR
 *     A bitfield that specifies compatibility workarounds that should be enabled
 *     to properly interface with the self-refresh TCON in the system
 *
 *        LW0073_CTRL_SYSTEM_ADD_VRR_HEAD_EXDEV_FORCE_FIRST_3_UNSTALL
 *           Setting this bit will cause the first 3 frames to be manually unstalled
 *           by the RM via privreg rather than waiting for a framelock
 *        LW0073_CTRL_SYSTEM_ADD_VRR_HEAD_EXDEV_DELAY_SR_ENTRY
 *           Setting this bit will cause the RM to delay sending SR entry for 4 ms.
 *        LW0073_CTRL_SYSTEM_ADD_VRR_HEAD_EXDEV_FORCE_DOUBLE_FRAME
 *           Setting this will cause the RM to automatically unstall the RG
 *           after the legitimate frame has been sent, causing 2 frames to be scanned
 *           for every flip.
 *
 *  bEnableCrashSync
 *     Indicates whether crash-sync should be enabled on this head
 *
 *  crashSyncMethod
 *     Defines how LWSR crash-sync will be performed if crash-sync is enabled:
 *
 *        LW0073_CTRL_SYSTEM_ADD_VRR_HEAD_CRASH_SYNC_NORMAL
 *           Normal timings
 *        LW0073_CTRL_SYSTEM_ADD_VRR_HEAD_CRASH_SYNC_DELAY_RESET
 *           Reset when outside of the Active/Fetching region (delayed reset)
 *        LW0073_CTRL_SYSTEM_ADD_VRR_HEAD_CRASH_SYNC_IMM_RESET
 *           Reset immediately
 *        LW0073_CTRL_SYSTEM_ADD_VRR_HEAD_CRASH_SYNC_DELAY_RESET_AND_UPDATE
 *           Combined : Reset outside Active, and initiate an SR Update
 *        LW0073_CTRL_SYSTEM_ADD_VRR_HEAD_CRASH_SYNC_IMM_RESET_AND_UPDATE
 *           Reset immediately, and initiate an SR Update
 *
 *   bEnableTearingMode
 *     Indicates that this head should be treated like a VRR head in terms of
 *     base channel blocking and semaphore activity, but the RM will assume
 *     that it does not extend the VBLANK or stall the RG (ie. it is a
 *     continuous mode head) and therefore will not trigger framelocks or
 *     ELV unblocks and will not consider its raster position when deciding
 *     when to flip.  This should be used for non-VRR monitors that are part
 *     of a VRR display group.
 *
 *  winChInstance
 *      Window Channel instance on which traps would be generated by client. Only applicable
 *      for Volta_and_later
 *
 * Possible status values returned are:

 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 *    LW_ERR_ILWALID_STATE
 */
#define LW0073_CTRL_CMD_SYSTEM_ADD_VRR_HEAD (0x730179U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_ADD_VRR_HEAD_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_ADD_VRR_HEAD_PARAMS_MESSAGE_ID (0x79U)

typedef struct LW0073_CTRL_SYSTEM_ADD_VRR_HEAD_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  displayId;
    LwU32  headGroupId;
    LwBool bBlockChannel;
    LwBool bInitStateSuspended;
    LwU32  frontPorchExtension;
    LwU32  backPorchExtension;
    LW_DECLARE_ALIGNED(LwP64 frameSemaphore, 8);
    LwU32  frameSemaphoreCount;
    LwU32  timeout;
    LwU32  timeoutMicroseconds;
    LwBool bOneShotMode;
    LwBool bOneShotPinReleaseEnable;
    LwBool bExternalFramelockToggle;
    LwU32  externalFramelockTimeout;
    LwU32  externalDeviceWAR;
    LwBool bEnableCrashSync;
    LwU32  crashSyncMethod;
    LwBool bEnableTearingMode;
    LwBool bEnableLWSRLiteMode;
    LwU32  rblWriteDelayUs;
    LwU32  winChInstance;
} LW0073_CTRL_SYSTEM_ADD_VRR_HEAD_PARAMS;

#define LW0073_CTRL_SYSTEM_ADD_VRR_HEAD_EXDEV_FORCE_FIRST_3_UNSTALL       (0x00000001U)
#define LW0073_CTRL_SYSTEM_ADD_VRR_HEAD_EXDEV_DELAY_SR_ENTRY              (0x00000002U)
#define LW0073_CTRL_SYSTEM_ADD_VRR_HEAD_EXDEV_FORCE_DOUBLE_FRAME          (0x00000004U)


#define LW0073_CTRL_SYSTEM_ADD_VRR_HEAD_CRASH_SYNC_NORMAL                 (0x00000000U)
#define LW0073_CTRL_SYSTEM_ADD_VRR_HEAD_CRASH_SYNC_DELAY_RESET            (0x00000001U)
#define LW0073_CTRL_SYSTEM_ADD_VRR_HEAD_CRASH_SYNC_IMM_RESET              (0x00000003U)
#define LW0073_CTRL_SYSTEM_ADD_VRR_HEAD_CRASH_SYNC_DELAY_RESET_AND_UPDATE (0x00000005U)
#define LW0073_CTRL_SYSTEM_ADD_VRR_HEAD_CRASH_SYNC_IMM_RESET_AND_UPDATE   (0x00000007U)

#define LW0073_CTRL_SYSTEM_ADD_VRR_HEAD_MAX_HEADGROUPS                    (16U)

/*
 * LW0073_CTRL_CMD_SYSTEM_FORCE_VRR_RELEASE
 *
 * This command instructs the RM to start a new frame on the RG after waitForFlipMs
 * milliseconds have passed without a normal flip oclwrring.  This effectively
 * overrides the "timeout" value set in LW0073_CTRL_SYSTEM_ADD_VRR_HEAD_PARAMS
 * for the current frame.  This function will only shorten the timeout on the
 * current frame, it will not push it out.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   headGroupId
 *     This indicates the head-group that will be forced to release
 *   waitForFlipMs
 *     This parameter specifies the amount of time in milliseconds to wait for
 *     a flip before forcing the RGs on all VRR heads to start a new frame.
 *     The timer delay is relative to the beginning of the last frame that
 *     started scanning.  If the current time t is beyond this delay,
 *     the RG will be released immediately.  If this value is less than the
 *     expected frame time, the release will be scheduled for when the current scan
 *     concludes.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 *    LW_ERR_ILWALID_STATE
 */
#define LW0073_CTRL_CMD_SYSTEM_FORCE_VRR_RELEASE                          (0x73017aU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_FORCE_VRR_RELEASE_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_FORCE_VRR_RELEASE_PARAMS_MESSAGE_ID (0x7AU)

typedef struct LW0073_CTRL_SYSTEM_FORCE_VRR_RELEASE_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 headGroupId;
    LwU32 waitForFlipMs;
} LW0073_CTRL_SYSTEM_FORCE_VRR_RELEASE_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_HOTPLUG_UNPLUG_STATE
 *
 * This command can be used to retrieve hotplug and unplug state
 * information that are lwrrently recorded by the RM. This information is
 * used by the client to determine which displays to detect after a
 * hotplug event oclwrs. Or if  the client knows that this device generates
 * a hot plug/unplug signal on all connectors, then this can be used to call
 * displays from detection. The displayIds on which hotplug/unplug has
 * happened will be reported only ONCE to the client. That is if the call
 * is done multiple times for the same event update, then for consequent
 * calls the display mask will be reported as 0.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   flags
 *     This parameter specifies optional flags to be used while retrieving
 *     the hotplug state information.
 *     Here are the current defined fields:
 *       LW0073_CTRL_SYSTEM_GET_HOTPLUG_STATE_FLAGS_LID
 *         A client uses this field to determine the lid state.
 *         Possible values are:
 *       LW0073_CTRL_SYSTEM_GET_HOTPLUG_STATE_FLAGS_LID_OPEN
 *              The lid is open.
 *       LW0073_CTRL_SYSTEM_GET_HOTPLUG_STATE_FLAGS_LID_CLOSED
 *              The lid is closed.  The client should remove devices a
 *              reported inside the
 *              LW0073_CTRL_SYSTEM_GET_CONNECT_POLICY_PARAMS.lidClosedMask.
 *   hotPlugMask
 *     This display mask specifies an LW0073_DISPLAY_MASK value describing
 *     the set of displays that have seen a hotplug.
 *   hotUnplugMask
 *     This display mask specifies an LW0073_DISPLAY_MASK value describing
 *     the set of displays that have seen a hot unplug
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0073_CTRL_CMD_SYSTEM_GET_HOTPLUG_UNPLUG_STATE (0x73017bU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | 0x7B" */

typedef struct LW0073_CTRL_SYSTEM_GET_HOTPLUG_UNPLUG_STATE_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 flags;
    LwU32 hotPlugMask;
    LwU32 hotUnplugMask;
} LW0073_CTRL_SYSTEM_GET_HOTPLUG_UNPLUG_STATE_PARAMS;

/* valid get hoplug state flags */
#define LW0073_CTRL_SYSTEM_GET_HOTPLUG_UNPLUG_STATE_FLAGS_LID                   0:0
#define LW0073_CTRL_SYSTEM_GET_HOTPLUG_UNPLUG_STATE_FLAGS_LID_OPEN   (0x00000000U)
#define LW0073_CTRL_SYSTEM_GET_HOTPLUG_UNPLUG_STATE_FLAGS_LID_CLOSED (0x00000001U)

/*
 * LW0073_CTRL_SYSTEM_MODIFY_VRR_HEAD_PARAMS
 *
 * This structure defines the instructions for RM to modify the VRR configuration
 * of the specified head.  VRR must be ENABLED for this to be successful
 *
 * Note this is obsolete for use in its own command, but it is used by the
 * LW0073_CTRL_CMD_SYSTEM_MODIFY_VRR_HEAD_GROUP command.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *
 *   displayId
 *     The public ID of the Output Display which is to be used for VRR.
 *
 *   frontPorchExtension
 *     This parameter specifies the value to extend the front porch by
 *     to extend Vblank for VRR.
 *
 *   backPorchExtension
 *     This parameter specifies the value to extend the back porch by
 *     to extend Vblank for VRR.
 *
 *   bBlockChannel
 *     This boolean indicates if the base channel associated with this head should be blocked as
 *     part of VRR processing. setting this to TRUE will cause the channel to be blocked when the
 *     NOOP method is processed by the base channel. the channel will remain blocked until the
 *     scan out is released.
 *
 *   bClearSemaphores
 *     Setting this to TRUE will clear all of the current semaphores assigned to this head
 *
 *   bSetSuspended
 *     Setting this to TRUE will put the specified head into a VRR_HEAD_SUSPENDED state
 *     Setting this to FALSE will put the head into a VRR_HEAD_ACTIVE state
 *     Note: Suspended heads will ignore any frame-ready semaphores specified in
 *     ADD_VRR_HEAD
 *
 *   bIsAdaptiveSync
 *     Setting this to True will tell that it is Adaptive-Sync head.
 *
 */

typedef struct LW0073_CTRL_SYSTEM_MODIFY_VRR_HEAD_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  displayId;
    LwBool bBlockChannel;
    LwU32  frontPorchExtension;
    LwU32  backPorchExtension;
    LwBool bClearSemaphores;
    LwBool bSetSuspended;
    LwBool bIsAdaptiveSync;
} LW0073_CTRL_SYSTEM_MODIFY_VRR_HEAD_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_CLEAR_ELV_BLOCK
 *
 * This command instructs the RM to explicitly clear any
 * ELV block.  Clients should call this before attempting core-channel
 * updates when in VRR one-shot mode.  ELV block mode will be
 * properly restored to its appropriate setting based on the stall-lock
 * in Supervisor3 after the core channel update
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *
 *   displayId
 *     The public ID of the Output Display which is to be used for VRR.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 *    LW_ERR_ILWALID_STATE
 */

#define LW0073_CTRL_CMD_SYSTEM_CLEAR_ELV_BLOCK (0x73017dU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_CLEAR_ELV_BLOCK_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_CLEAR_ELV_BLOCK_PARAMS_MESSAGE_ID (0x7DU)

typedef struct LW0073_CTRL_SYSTEM_CLEAR_ELV_BLOCK_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
} LW0073_CTRL_SYSTEM_CLEAR_ELV_BLOCK_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_ARM_LIGHTWEIGHT_SUPERVISOR
 *
 * This command arms the display modeset supervisor to operate in
 * a lightweight mode.  By calling this, the client is implicitly
 * promising not to make any changes in the next modeset that require
 * the full supervisor.  After SV3, the LWSV will disarm and any subsequent
 * modesets will revert to full supervisors.  This must be called separately
 * for every display that will be part of the modeset.
 * It is recommended that the client explicitly disarm the lightweight
 * supervisor after every modeset as null modesets will not trigger the
 * supervisor interrupts and the RM will not be able to disarm automatically
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *
 *   displayId
 *     The public ID of the Output Display which is to be used for VRR.
 *
 *   bArmLWSV
 *     If this is set to LW_TRUE, the RM will arm the lightweight supervisor
 *     for the next modeset.
 *     If this is set to LW_FALSE, the RM will disarm the lightweight supervisor
 *
 *   bVrrState
 *     VRR state to be changed.
 *
 *   vActive
 *      GPU-SRC vertical active value
 *
 *   vfp
 *      GPU-SRC vertical front porch
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 *    LW_ERR_ILWALID_STATE
 */

#define LW0073_CTRL_CMD_SYSTEM_ARM_LIGHTWEIGHT_SUPERVISOR (0x73017eU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_ARM_LIGHTWEIGHT_SUPERVISOR_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_ARM_LIGHTWEIGHT_SUPERVISOR_PARAMS_MESSAGE_ID (0x7EU)

typedef struct LW0073_CTRL_SYSTEM_ARM_LIGHTWEIGHT_SUPERVISOR_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  displayId;
    LwBool bArmLWSV;
    LwBool bVrrState;
    LwU32  vActive;
    LwU32  vfp;
} LW0073_CTRL_SYSTEM_ARM_LIGHTWEIGHT_SUPERVISOR_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW0073_CTRL_CMD_SYSTEM_SCANOUT_LOGGING
 *
 * This command is used for enable/disable/get scanout logging.
 * In enable command client will request for buffer allocation in RM as per
 * the size parameter passed. In this buffer DPU will log scanout done timestamp,
 * ctxdma address and CRC as requested. These parameters will be used by client to
 * analyze stutter.
 * In the disable command the logging is stopped and allocated buffer is freed if
 * requested.
 * The get command is used to fetch the logged entries. RM will copy the logged
 * data into the buffer passed by client.
 *
 *   subDeviceInstance
 *     Client will give a subdevice to get right pGpu/pDisp for it
 *
 *   displayId
 *     DisplayId of the display for which the client needs the statistics
 *
 *   cmd
 *     LW0073_CTRL_SYSTEM_SCANOUT_LOGGING_DISABLE
 *        Send disable command to DPU, disable DMI interrupt and free allocated
 *        RM buffer based on bFreeBuffer flag
 *     LW0073_CTRL_SYSTEM_SCANOUT_LOGGING_ENABLE
 *        Allocate RM buffer, send enable command to DPU for scanout logging
 *        and enable DMI or RG scanline interrupt
 *     LW0073_CTRL_SYSTEM_SCANOUT_LOGGING_FETCH
 *        Copy logged data requested by client
 *
 *   size
 *     Number of logging enteries to be done in RM.
 *
 *   verticalScanline
 *     verticalScanline number to configure DMI/RG line interrupt for logging.
 *
 *   bFreeBuffer
 *     Flag to deallocate RM buffer.
 *
 *   bUseRasterLineIntr
 *     Flag to indicate DMI or RG line interrupt.
 *     Value 1 is RG line interrupt
 *     Value 0 is DMI line interrupt
 *
 *   scanoutLogFlag
 *     LW_RM_DPU_SCANOUTLOGGING_FLAGS_LOG_CTXDMA1:
 *       Flag to log ctxdma1 for stereo.
 *     LW_RM_DPU_SCANOUTLOGGING_FLAGS_LOG_CRC:
 *       Flag to capture CRC for frame's first slice. Where one slice means vactive/32.
 *       User needs to make sure that the verticalScanline is sufficiently larger than
 *       this otherwise the CRCs may not match.
 *     LW_RM_DPU_SCANOUTLOGGING_FLAGS_LOG_VRR_STATE:
 *       Flag to capture vrr state is active or suspended.
 *
 *   loggingAddr
 *     User allocated buffer address in which RM logged data will be copied.
 *
 * Possible status values returned include:
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_CMD_SYSTEM_SCANOUT_LOGGING (0x73017fU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_CMD_SYSTEM_SCANOUT_LOGGING_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_SYSTEM_SCANOUT_LOGGING_PARAMS_MESSAGE_ID (0x7FU)

typedef struct LW0073_CTRL_CMD_SYSTEM_SCANOUT_LOGGING_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  displayId;
    LwU32  cmd;
    LwU32  size;
    LwU32  verticalScanline;
    LwBool bFreeBuffer;
    LwBool bUseRasterLineIntr;
    LwBool bCaptureCRC;
    LwU8   scanoutLogFlag;
    LW_DECLARE_ALIGNED(LwP64 loggingAddr, 8);
} LW0073_CTRL_CMD_SYSTEM_SCANOUT_LOGGING_PARAMS;

#define LW0073_CTRL_SYSTEM_SCANOUT_LOGGING_DISABLE (0x00000000U)
#define LW0073_CTRL_SYSTEM_SCANOUT_LOGGING_ENABLE  (0x00000001U)
#define LW0073_CTRL_SYSTEM_SCANOUT_LOGGING_FETCH   (0x00000002U)

/*
 * LW0073_CTRL_SYSTEM_USE_TEST_PIOR_SETTINGS
 *
 *  This call disables the setting of the driver callwlated PIOR directions in favor of the values
  * setup for testing.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *
 *
 *  Possible status values returned are:
 *   LWOS_STATUS_SUCCESS
 *   LWOS_STATUS_ERROR_ILWALID_ARGUMENT
 *   LWOS_STATUS_ERROR_NOT_SUPPORTED
 *
 */
#define LW0073_CTRL_SYSTEM_USE_TEST_PIOR_SETTINGS  (0x730180U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_USE_TEST_PIOR_SETTINGS_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_USE_TEST_PIOR_SETTINGS_PARAMS_MESSAGE_ID (0x80U)

typedef struct LW0073_CTRL_SYSTEM_USE_TEST_PIOR_SETTINGS_PARAMS {
    LwU32  subDeviceInstance;
    LwBool bEnable;
} LW0073_CTRL_SYSTEM_USE_TEST_PIOR_SETTINGS_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_SET_VRR_STUTTER_TIMING
 *
 * This command calc/setup timing params for stutter management in LWSR.
 * All callwlation is done with timing info from EDID
 *
 *   subDeviceInstance
 *     Client will give a subdevice to get right pGpu/pDisp for it
 *
 *   displayId
 *     DisplayId of the panel for which client wants to set the timings
 *
 *   cmd
 *     LW0073_CTRL_CMD_SYSTEM_SET_STUTTER_TIMING
 *        Update RM struct with timing params for stutter management in LWSR
 *     LW0073_CTRL_CMD_SYSTEM_SET_MAX_REFRESH_RATE
 *        Configure SRC-Panel timing for max refresh rate
 *     LW0073_CTRL_CMD_SYSTEM_SET_MIN_REFRESH_RATE
 *        Configure SRC-Panel timing for min refresh rate
 *     LW0073_CTRL_CMD_SYSTEM_SET_DEFAULT_REFRESH_RATE
 *        Configure SRC-Panel timing for default refresh rate
 *
 *   src2PanelPclk
 *     src-panel pclk, read from EDID
 *
 *   hBlank
 *     src-panel horizontal blank, read from EDID
 *
 *   hActive
 *     src-panel horizontal active, read from EDID
 *
 *   vSync
 *     src-panel vSync, read from EDID
 *
 *   vActive
 *     src-panel vertical active, read from EDID
 *
 *   milwfp
 *     src-panel min vertical fp, read from EDID
 *
 *   milwB
 *     src-panel min vertical blank, read from EDID
 *
 *   maxVB
 *     src-panel max vertical blank, read from EDID
 *
 *   vBorder
 *     src-panel vertical border, read from EDID
 *
 *   vSyncPol
 *     src-panel vertical Sync Polarity, read from EDID
 *
 *   srcMinHz
 *     min refresh rate between SRC-Panel, read from EDID or regkey.
 *     The value from regkey can only higher than EDID specific min RR.
 *
 * Possible status values returned include:
 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_CMD_SYSTEM_SET_VRR_STUTTER_TIMING (0x730182U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_SET_VRR_STUTTER_TIMING_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_SET_VRR_STUTTER_TIMING_PARAMS_MESSAGE_ID (0x82U)

typedef struct LW0073_CTRL_SYSTEM_SET_VRR_STUTTER_TIMING_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 cmd;
    LwU32 src2PanelPclk;
    LwU32 hBlank;
    LwU32 hActive;
    LwU32 vSync;
    LwU32 vActive;
    LwU32 milwfp;
    LwU32 milwB;
    LwU32 maxVB;
    LwU32 vBorder;
    LwU32 vSyncPol;
    LwU32 srcMinHz;
    LwU32 tconSrcMinHz;
} LW0073_CTRL_SYSTEM_SET_VRR_STUTTER_TIMING_PARAMS;

#define LW0073_CTRL_CMD_SYSTEM_SET_STUTTER_TIMING       (0x00000000U)
#define LW0073_CTRL_CMD_SYSTEM_SET_MAX_REFRESH_RATE     (0x00000001U)
#define LW0073_CTRL_CMD_SYSTEM_SET_MIN_REFRESH_RATE     (0x00000002U)
#define LW0073_CTRL_CMD_SYSTEM_SET_DEFAULT_REFRESH_RATE (0x00000003U)
/*
* LW0073_CTRL_SYSTEM_VRR_LOG_INTERVAL_LOG
* This structure is used to log in all the VRR flip data per frame
*
*    frameNumber
*        Represents the index to the frame in the app.
*
*    releaseIntervalUs
*        Represents the interval between 2 frames. In this case it is the time since the last flip oclwrred.
*
*    timeSinceTrapUS
*        Represents the time since the last trap has oclwrred.
*
*    avgFlipInterval
*        Represents the average of the interval between flips until the last frame.
*
*    avgReleaseIntervalUs
*        Represents the average of the time between the vrr releases.
*
*    avgFrameUpdateIntervalUs
*        Represents the average time before the last fresh frame was scanned out.
*
*    forcedReleaseIntervalUs
*        Represents the time after which the last forced VRR release has oclwrred.
*
*    deferralTimeUs
*        Represents the time intentionally introduced in the flip so as to avoid screen stuttering.
*
*    accDeltaTimeUs
*        Represents the time acclwmulated in the odd/even frame aclwmulator buffers during DC balancing.
*
*    deltaPolarity
*        Polarity of the time difference in odd/even frame aclwmulators.
*        values:  1 - if time aclwmulated in even frame aclwmulator is greater than the odd frame aclwmulators.
*                -1 - if time aclwmulated in odd frame aclwmulator is greater than the even frame aclwmulators.
*
*    bForcedRelease
*        When set to LW_TRUE, signifies that the current vrr release was forced.
*
*    forcedReleaseDeltaUs
*        Time since the last forced vrr release.
*
*    pollCounter
*        Represents a polling counter for the flips.
*
*    tearingFlipCounter
*        Number of rearing flips.
*
*    tearingFlipRelTimeUs
*        Time since the last tear flip oclwrrence.
*
*  cmd:
*    LW0073_CTRL_SYSTEM_VRR_LOG_MAX_TEARING_FLIPS
*        Represents the maximum number of tearing flips in the log.
*
*    LW0073_CTRL_SYSTEM_VRR_LOG_ACTUAL_SIZE
*        Represents the size of the actual vrr log maintained in the headgroup structure.
*/

#define LW0073_CTRL_CMD_SYSTEM_VRR_LOG                  (0x730183U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_VRR_LOG_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_VRR_LOG_MAX_TEARING_FLIPS    (10U)
#define LW0073_CTRL_SYSTEM_VRR_LOG_SIZE                 (128U)
#define LW0073_CTRL_VRR_LOG_REPORT_SIZE                 (16U)

typedef struct LW0073_CTRL_SYSTEM_VRR_LOG_INTERVAL_LOG {
    LwU32  frameNumber;
    LwU32  flipToken;
    LwU32  releaseIntervalUs;
    LwU32  timeSinceTrapUs;
    LwU32  avgFlipIntervalUs;
    LwU32  avgReleaseIntervalUs;
    LwU32  avgFrameUpdateIntervalUs;
    LwU32  forcedReleaseIntervalUs;
    LwU32  deferralTimeUs;
    LwU32  accDeltaTimeUs;
    LwS32  deltaPolarity;
    LwBool bForcedRelease;
    LwS32  forcedReleaseDeltaUs;
    LwU32  pollCounter;
    LwU32  tearingFlipCounter;
    LW_DECLARE_ALIGNED(LwU64 tearingFlipRelTimeUs[LW0073_CTRL_SYSTEM_VRR_LOG_MAX_TEARING_FLIPS], 8);
} LW0073_CTRL_SYSTEM_VRR_LOG_INTERVAL_LOG;

/*
* LW0073_CTRL_SYSTEM_VRR_LOG_PARAMS
*
* This command is used to log one chunk of VRR flip logs from the RM into the KMD
*   subDeviceInstance
*     This parameter specifies the subdevice instance within the
*     LW04_DISPLAY_COMMON parent device to which the operation should be
*     directed.This parameter must specify a value between zero and the
*     total number of subdevices within the parent device.This parameter
*     should be set to zero for default behavior.
*
*   displayId
*     This is no longer used and only provided to prevent breaking client
*     builds.Once all of the clients have cleaned up, this will be
*     removed.swinme
*
*   vidPnSourceId
*     Represents a source id of the video pin.
*
*   lastQueriedVRRIndex
*     Represents the index of the the loast queried entry in the flipLog.
*
*   Output Parameters:-
*   flipLog
*     Represents the log maintained in the command ctrl params passed onto the KMD.
*     It holds the flip log values for the flip token from the vrr log
*
*   flipLogIndex
*     Represents the current index of the flipLog. Initially set to zero.
*
*   numberOfEntries
*     Represents the number of entries being transferred from RM to KMD for that escape call.
*/

#define LW0073_CTRL_SYSTEM_VRR_LOG_PARAMS_MESSAGE_ID (0x83U)

typedef struct LW0073_CTRL_SYSTEM_VRR_LOG_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 vidPnSourceId;
    LwS32 lastQueriedVRRIndex;
    LwU32 numberOfEntries;
    LW_DECLARE_ALIGNED(LW0073_CTRL_SYSTEM_VRR_LOG_INTERVAL_LOG flipLog[LW0073_CTRL_VRR_LOG_REPORT_SIZE], 8);
    LwS32 flipLogIndex;
} LW0073_CTRL_SYSTEM_VRR_LOG_PARAMS;
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
* LW0073_CTRL_SYSTEM_CONFIG_VRR_PSTATE_SWITCH_PARAMS
*
* This command is used to configure pstate switch parameters on VRR monitors
*   subDeviceInstance
*     This parameter specifies the subdevice instance within the
*     LW04_DISPLAY_COMMON parent device to which the operation should be
*     directed.This parameter must specify a value between zero and the
*     total number of subdevices within the parent device.This parameter
*     should be set to zero for default behavior.
*
*   displayId
*     DisplayId of the monitor being vrr configured
*
*   bVrrState
*     When set to LW_TRUE, signifies that the vrr is about to become active.
*     When set to LW_FALSE, signifies that the vrr is about to become suspended.
*
*   bVrrDirty
*     When set to LW_TRUE, indicates that vrr configuration has been changed
*     When set to LW_FALSE, this will indicate transitions from One shot mode to 
*     Continuous mode and vice versa 
*
*   bVrrEnabled
*     When set to LW_TRUE, indicates that vrr has been enabled, i.e. vBp extended by 2 lines
*
*   maxVblankExtension
*     When VRR is enabled, this is the maximum amount of lines that the vblank can be extended.
*     Only updated when bVrrDirty = true
*
*   internalVRRHeadVblankStretch
*     When VRR is enabled, this is the maximum amount of lines that the vblank can be extended.
*     On LWSR and DD panels . Only updated when bVrrDirty = true
*
*   milwblankExtension
*     When VRR is enabled, this is the minimum amount of lines that should be present in the Vblank. The purpose is to cap the maximum refresh (lwrrently only for HDMI 2.1 VRR compliance)
*/
#define LW0073_CTRL_CMD_SYSTEM_CONFIG_VRR_PSTATE_SWITCH (0x730184U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_CONFIG_VRR_PSTATE_SWITCH_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_CONFIG_VRR_PSTATE_SWITCH_PARAMS_MESSAGE_ID (0x84U)

typedef struct LW0073_CTRL_SYSTEM_CONFIG_VRR_PSTATE_SWITCH_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  displayId;
    LwBool bVrrState;
    LwBool bVrrDirty;
    LwBool bVrrEnabled;
    LwU32  maxVblankExtension;
    LwU32  internalVRRHeadVblankStretch;
    LwU32  milwblankExtension;
} LW0073_CTRL_SYSTEM_CONFIG_VRR_PSTATE_SWITCH_PARAMS;

/*
* LW0073_CTRL_SYSTEM_VRR_DISPLAY_INFO_PARAMS
*
* This command is used to update information about VRR capable monitors
*   subDeviceInstance
*     This parameter specifies the subdevice instance within the
*     LW04_DISPLAY_COMMON parent device to which the operation should be
*     directed.This parameter must specify a value between zero and the
*     total number of subdevices within the parent device.This parameter
*     should be set to zero for default behavior.
*
*   displayId
*     DisplayId of the panel for which client wants to add or remove from VRR
*     capable monitor list
*
*   bAddition
*     When set to LW_TRUE, signifies that the vrr monitor is to be added.
*     When set to LW_FALSE, signifies that the vrr monitor is to be removed.
*
*/
#define LW0073_CTRL_CMD_SYSTEM_VRR_DISPLAY_INFO (0x730185U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_VRR_DISPLAY_INFO_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_VRR_DISPLAY_INFO_PARAMS_MESSAGE_ID (0x85U)

typedef struct LW0073_CTRL_SYSTEM_VRR_DISPLAY_INFO_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  displayId;
    LwBool bAddition;
} LW0073_CTRL_SYSTEM_VRR_DISPLAY_INFO_PARAMS;

/*
* LW0073_CTRL_CMD_SYSTEM_MODIFY_VRR_HEAD_GROUP
*
* This command is used so that RM can coherently put all of the heads in
* a headgroup into the specified VRR modes as a batch.
*
* This is a kernel privileged control call
*
* modifyVrrHeadParams
*     This parameter contains the desired state of each VRR head as specified
*     in LW0073_CTRL_CMD_SYSTEM_MODIFY_VRR_HEAD_PARAMS.  Note that all of the heads
*     must be part of the same headgroup and all of the heads in a headgroup
*     must be part of the same call
*
* numHeads
*     This specifies how many entries in modifyVrrHeadParams are used
*
* Possible status values returned are:
*    LW_OK
*    LW_ERR_ILWALID_ARGUMENT
*    LW_ERR_ILWALID_STATE
*/

#define LW0073_CTRL_CMD_SYSTEM_MODIFY_VRR_HEAD_GROUP           (0x730186U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_MODIFY_VRR_HEAD_GROUP_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_SYSTEM_MODIFY_VRR_HEAD_GROUP_MAX_HEADS (16U)

#define LW0073_CTRL_SYSTEM_MODIFY_VRR_HEAD_GROUP_PARAMS_MESSAGE_ID (0x86U)

typedef struct LW0073_CTRL_SYSTEM_MODIFY_VRR_HEAD_GROUP_PARAMS {
    LW0073_CTRL_SYSTEM_MODIFY_VRR_HEAD_PARAMS modifyVrrHeadParams[LW0073_CTRL_CMD_SYSTEM_MODIFY_VRR_HEAD_GROUP_MAX_HEADS];
    LwU32                                     numHeads;
    LwBool                                    bVideoAdaptiveRefreshMode;
} LW0073_CTRL_SYSTEM_MODIFY_VRR_HEAD_GROUP_PARAMS;


#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW0073_CTRL_CMD_SYSTEM_INLINE_DISP_INTR_SERVICE_WAR_FOR_VR
 *
 * This command engages the WAR when VR devices are connected,
 * where the Pstate switching can cause delay in Vblank callbacks
 * reported to KMD, by servicing disp interrupts inline and reporting the
 * callbacks to KMD. Without the WAR, there can be stutters during pstate switch.
 * Bug#1778552
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *   bEngageWAR
 *     Indicates if inline disp interrupt serving WAR has to be engaged or
 *     disengaged.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0073_CTRL_CMD_SYSTEM_INLINE_DISP_INTR_SERVICE_WAR_FOR_VR (0x730187U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_INLINE_DISP_INTR_SERVICE_WAR_FOR_VR_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_INLINE_DISP_INTR_SERVICE_WAR_FOR_VR_PARAMS_MESSAGE_ID (0x87U)

typedef struct LW0073_CTRL_SYSTEM_INLINE_DISP_INTR_SERVICE_WAR_FOR_VR_PARAMS {
    LwU32  subDeviceInstance;
    LwBool bEngageWAR;
} LW0073_CTRL_SYSTEM_INLINE_DISP_INTR_SERVICE_WAR_FOR_VR_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_EDID_OVERRIDE
 *
 * Fetches the override edid buffer's data
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *
 *   displayId
 *     DisplayId of the panel for which we need to the override edid buffer's data
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_EDID_OVERRIDE (0x730188U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_CMD_SYSTEM_GET_EDID_OVERRIDE_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_EDID_OVERRIDE_SIZE           (512U)

#define LW0073_CTRL_CMD_SYSTEM_GET_EDID_OVERRIDE_PARAMS_MESSAGE_ID (0x88U)

typedef struct LW0073_CTRL_CMD_SYSTEM_GET_EDID_OVERRIDE_PARAMS {

    LwU8  pEorAddr[LW0073_CTRL_EDID_OVERRIDE_SIZE];
    LwU32 subDeviceInstance;
    LwU32 displayId;
} LW0073_CTRL_CMD_SYSTEM_GET_EDID_OVERRIDE_PARAMS;


/*
 * LW0073_CTRL_CMD_SYSTEM_CONFIGURE_BL_GPIO_CONTROL
 *
 * This command enable or disable eDP backlight GPIO pin SW-Control.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *
 *   displayId
 *     DisplayId of the panel for which client wants to add or remove from VRR
 *     capable monitor list
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_GENERIC
 *
 */

#define LW0073_CTRL_CMD_SYSTEM_CONFIGURE_BL_GPIO_CONTROL (0x730189U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_CMD_SYSTEM_CONFIGURE_BL_GPIO_CONTROL_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_SYSTEM_CONFIGURE_BL_GPIO_CONTROL_PARAMS_MESSAGE_ID (0x89U)

typedef struct LW0073_CTRL_CMD_SYSTEM_CONFIGURE_BL_GPIO_CONTROL_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  displayId;

    LwBool bEnableBacklightGPIOSWControl;
} LW0073_CTRL_CMD_SYSTEM_CONFIGURE_BL_GPIO_CONTROL_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_DUAL_MODE_DP_DONGLE_INFO
 *
 * This command can be used to get a mask of display IDs that have a
 * passive Dual Mode Dongle attached (these dongles need not necessarily have
 * monitors connected to them)
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   dongleMask
 *     This parameter provides an OR'ed value of the display IDs that
 *     have a passive Dual Mode dongle attached at the time of calling
 */

#define LW0073_CTRL_CMD_SYSTEM_GET_DUAL_MODE_DP_DONGLE_INFO (0x73018lw) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_CMD_SYSTEM_GET_DUAL_MODE_DP_DONGLE_INFO_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_SYSTEM_GET_DUAL_MODE_DP_DONGLE_INFO_PARAMS_MESSAGE_ID (0x8LW)

typedef struct LW0073_CTRL_CMD_SYSTEM_GET_DUAL_MODE_DP_DONGLE_INFO_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 dongleMask;
} LW0073_CTRL_CMD_SYSTEM_GET_DUAL_MODE_DP_DONGLE_INFO_PARAMS;

/*
* LW0073_CTRL_CMD_SYSTEM_FORCE_VRR_RELEASE_METHOD_MODE
*
* This command instructs the RM to start a new frame on the RG
* and release the ELV when the pTimer exceeds releaseTimeAbsNs
*
*   subDeviceInstance
*     This parameter specifies the subdevice instance within the
*     LW04_DISPLAY_COMMON parent device to which the operation should be
*     directed.  This parameter must specify a value between zero and the
*     total number of subdevices within the parent device.  This parameter
*     should be set to zero for default behavior.
*   head
*     This indicates the head that will be forced to release
*   releaseTimeAbsNs
*     This parameter specifies the absolute ptimer time when the
*     release should be triggered.  The RM will schedule this using
*     the timer interrupt, so there will be additional interrupt latency
*
* Possible status values returned are:
*    LW_OK
*    LW_ERR_ILWALID_ARGUMENT
*    LW_ERR_ILWALID_STATE
*/
#define LW0073_CTRL_CMD_SYSTEM_FORCE_VRR_RELEASE_METHOD_MODE (0x73018dU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_FORCE_VRR_RELEASE_METHOD_MODE_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_FORCE_VRR_RELEASE_METHOD_MODE_PARAMS_MESSAGE_ID (0x8DU)

typedef struct LW0073_CTRL_SYSTEM_FORCE_VRR_RELEASE_METHOD_MODE_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 head;                 // IN
    LwU32 releaseTimeAbsNs;     // IN
} LW0073_CTRL_SYSTEM_FORCE_VRR_RELEASE_METHOD_MODE_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_BACKLIGHT_TYPE
 *
 * Fetches the backlight type of panel
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *   displayId
 *     DisplayId of the panel for which we need to the backight type
 *   blSupported
 *     Represents the backlight Supported status returned by panel
 *   blEnabled
 *     Represents the backlight Enabled status returned by panel
 *   panelCapSupportedbyAux
 *     Represents the panel capability is supported by aux
 *   panelCapSupportedbyPin
 *     Represents the panel capability is supported by pin
 *   bIsNonPwm
 *     pointer to check nbci control type
 *   bIsBackLightEnabledbyPin
 *     pointer to check if backlight enabled by pin or not
 *   bIsBackLightEnabledbyAux
 *     pointer to check if backlight enabled by aux or not
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0073_CTRL_CMD_SYSTEM_GET_BACKLIGHT_TYPE                                    (0x73018eU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_CMD_SYSTEM_GET_BACKLIGHT_TYPE_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_SYSTEM_GET_NON_PWM_BACKLIGHT_EN_BY_AUX_SUPPORTED             0x00000001U
#define LW0073_CTRL_CMD_SYSTEM_GET_NON_PWM_BACKLIGHT_EN_BY_AUX_NOT_SUPPORTED         0x00000002U
#define LW0073_CTRL_CMD_SYSTEM_GET_NON_PWM_BACKLIGHT_BRIGHTNESS_OVER_AUX_ENABLED     0x00000003U
#define LW0073_CTRL_CMD_SYSTEM_GET_NON_PWM_BACKLIGHT_BRIGHTNESS_OVER_AUX_NOT_ENABLED 0x00000004U
#define LW0073_CTRL_CMD_SYSTEM_GET_PWM_BACKLIGHT_EN_BY_PIN_SET_SUPPORTED             0x00000005U
#define LW0073_CTRL_CMD_SYSTEM_GET_PWM_BACKLIGHT_EN_BY_AUX_SUPPORTED                 0x00000006U
#define LW0073_CTRL_CMD_SYSTEM_GET_PWM_BRIGHTNESS_OVER_AUX_ENABLED                   0x00000007U
#define LW0073_CTRL_CMD_SYSTEM_GET_PWM_BACKLIGHT_OVER_AUX_NOT_ENABLED                0x00000008U
#define LW0073_CTRL_CMD_SYSTEM_GET_PANEL_CAP_BY_AUX                                  0x00000001U
#define LW0073_CTRL_CMD_SYSTEM_GET_PANEL_CAP_BY_PIN                                  0x00000001U
#define LW0073_CTRL_CMD_SYSTEM_GET_BACKLIGHT_TYPE_PARAMS_MESSAGE_ID (0x8EU)

typedef struct LW0073_CTRL_CMD_SYSTEM_GET_BACKLIGHT_TYPE_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  displayId;
    LwU8   blSupported;
    LwU8   blEnabled;
    LwU8   panelCapSupportedbyAux;
    LwU8   panelCapSupportedbyPin;
    LwBool bIsNonPwm;
    LwBool bIsBackLightEnabledbyPin;
    LwBool bIsBackLightEnabledbyAux;
} LW0073_CTRL_CMD_SYSTEM_GET_BACKLIGHT_TYPE_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_NOTIFY_FRL_ENGAGED
 *
 * This command informs RM that FRL is engaged.
 * RM can use this information to try and go in a low power state.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.  This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behaviour.
 *
 *   displayId
 *     This is public ID of the Output Display on which flip will be delayed by
 *     FRL application.
 *
 *   params
 *     This parameter contains flags/info that client may want to pass to RM.
 *     Lwrrently a placeholder.
 *
 * Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 *    LW_ERR_ILWALID_STATE
 *    LW_ERR_ILWALID_NOT_SUPPORTED
 *    LW_ERR_ILWALID_POINTER
 *    LW_ERR_NO_MEMORY
 */

#define LW0073_CTRL_CMD_SYSTEM_NOTIFY_FRL_ENGAGED   (0x73018fU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_CMD_SYSTEM_NOTIFY_FRL_ENGAGED_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_MINIMUM_SECONDS_FRL_ENGAGED 1U

#define LW0073_CTRL_CMD_SYSTEM_NOTIFY_FRL_ENGAGED_PARAMS_MESSAGE_ID (0x8FU)

typedef struct LW0073_CTRL_CMD_SYSTEM_NOTIFY_FRL_ENGAGED_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 params;
} LW0073_CTRL_CMD_SYSTEM_NOTIFY_FRL_ENGAGED_PARAMS;
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW0073_CTRL_CMD_SYSTEM_QUERY_DISPLAY_IDS_WITH_MUX
 *
 * This command is used to query the display mask of all displays
 * that support dynamic display MUX.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *   displayMask (out)
 *     Mask of all displays that support dynamic display MUX
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW0073_CTRL_CMD_SYSTEM_QUERY_DISPLAY_IDS_WITH_MUX (0x730190U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_CMD_SYSTEM_QUERY_DISPLAY_IDS_WITH_MUX_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_SYSTEM_QUERY_DISPLAY_IDS_WITH_MUX_PARAMS_MESSAGE_ID (0x90U)

typedef struct LW0073_CTRL_CMD_SYSTEM_QUERY_DISPLAY_IDS_WITH_MUX_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 muxDisplayMask;
} LW0073_CTRL_CMD_SYSTEM_QUERY_DISPLAY_IDS_WITH_MUX_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW0073_CTRL_CMD_SYSTEM_SET_LWSR_FLAGS
 *
 * This command is used to set the system to distinguish between LWSR and force DD
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *   bIsLwsrEnabled (in)
 *     This flag tells whether its LWSR config or not
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */

#define LW0073_CTRL_CMD_SYSTEM_SET_LWSR_FLAGS (0x730191U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_CMD_SYSTEM_SET_LWSR_FLAGS_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_SYSTEM_SET_LWSR_FLAGS_PARAMS_MESSAGE_ID (0x91U)

typedef struct LW0073_CTRL_CMD_SYSTEM_SET_LWSR_FLAGS_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 bIsLwsrEnabled;
} LW0073_CTRL_CMD_SYSTEM_SET_LWSR_FLAGS_PARAMS;


/*
 * LW0073_CTRL_CMD_SYSTEM_GET_FLAT_PANEL_BRIGHTNESS
 *
 * This command returns several fields about brightness from a specified display.
 *
 * Parameters:
 * [IN]  dispId  The display ID to get values from, use 0 for default
 * [OUT] numLevels
 * [OUT] maxBrightness
 * [OUT] minBrightness
 * [OUT] lwrBrightness
 * [OUT] bAvailable
 *
 * Possible status values returned are:
 *   LW_ERR_ILWALID_OBJECT - Invalid displayId provided
 */
#define LW0073_CTRL_CMD_SYSTEM_GET_FLAT_PANEL_BRIGHTNESS (0x730192U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_GET_FLAT_PANEL_BRIGHTNESS_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_GET_FLAT_PANEL_BRIGHTNESS_PARAMS_MESSAGE_ID (0x92U)

typedef struct LW0073_CTRL_SYSTEM_GET_FLAT_PANEL_BRIGHTNESS_PARAMS {
    LwU32  dispId;
    LwU32  numLevels;
    LwU32  maxBrightness;
    LwU32  minBrightness;
    LwU32  lwrBrightness;
    LwBool bAvailable;
} LW0073_CTRL_SYSTEM_GET_FLAT_PANEL_BRIGHTNESS_PARAMS;


/*
 * LW0073_CTRL_CMD_SYSTEM_SET_FLAT_PANEL_BRIGHTNESS
 *
 * This command sets the brightness for a specified display.
 *
 * Parameters:
 * [IN]  dispId  The display ID to get values from, use 0 for default
 * [IN]  head    The index of the head, only used on mac
 * [IN]  brightness
 * [IN]  transitionRate
 *
 * Possible status values returned are:
 *   LW_ERR_ILWALID_ARGUMENT - Invalid head index provided (mac), or failed to find valid head
 */
#define LW0073_CTRL_CMD_SYSTEM_SET_FLAT_PANEL_BRIGHTNESS (0x730193U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_SET_FLAT_PANEL_BRIGHTNESS_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_SET_FLAT_PANEL_BRIGHTNESS_PARAMS_MESSAGE_ID (0x93U)

typedef struct LW0073_CTRL_SYSTEM_SET_FLAT_PANEL_BRIGHTNESS_PARAMS {
    LwU32 dispId;
    LwU32 head;
    LwU32 brightness;
    LwU32 transitionRate;
} LW0073_CTRL_SYSTEM_SET_FLAT_PANEL_BRIGHTNESS_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_UPDATE_MIN_REFRESH_RATE
 *
 * RM may in some cases reduce the nominal display refresh rate in order
 * to save power, but it will not reduce the refresh rate below the minimum
 * specified by this API. This control call will be called for each modeset
 * when head is active on Windows platform.
 * 
 * Sequence of events: 
 * 1) DD sends methods for a new mode.
 * 2) DD calls lw0073CtrlCmdSystemUpdateMinRefreshRate only for 
 *    internal panel which is VRR capable. 
 * 3) DD sends update method to trigger modeset which will trigger HW 
 *    supervisor interrupts.
 * 4) RM disables Vblank extensions in Sv1.
 * 5) RM callwlates extension for current mode and enables STRETCH on SV3.
 * 
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *   headIdx
 *     Head Index of the panel for which client wants to Update Min Refresh Rate.
 *   minRefreshRateHz
 *     Value of the Min refresh rate in Hz.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0073_CTRL_CMD_SYSTEM_UPDATE_MIN_REFRESH_RATE (0x730194U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_CMD_SYSTEM_UPDATE_MIN_REFRESH_RATE_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_SYSTEM_UPDATE_MIN_REFRESH_RATE_PARAMS_MESSAGE_ID (0x94U)

typedef struct LW0073_CTRL_CMD_SYSTEM_UPDATE_MIN_REFRESH_RATE_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 headIdx;
    LwU32 minRefreshRateHz;
} LW0073_CTRL_CMD_SYSTEM_UPDATE_MIN_REFRESH_RATE_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_CONFIG_OSM_MCLK_SWITCH
 *
 * Client will issue this control Call to RM. RM will configure Pstate 
 * settings on game launch or on game close based on whether one 
 * shot mode mclk is possible or not for a display mode. 
 * 
 * Sequence of events: 
 * 1) On app launch DD calls LW0073_CTRL_CMD_SYSTEM_CONFIG_VRR_PSTATE_SWITCH
 *    to tell RM to disable VRR mclk switch.
 * 2) As part of that call, RM locks pstate to min pstate which supports
 *    max mclk Frequency by setting display glitch limit.
 * 3) DD sends stall lock/display rate and update method which will trigger
 *    light weight HW supervisor interrupts.
 * 4) After Modeset DD will call  LW0073_CTRL_CMD_SYSTEM_CONFIG_OSM_MCLK_SWITCH
 *    to enable OSM based mclk switch (if possible).
 * 5) If OSM mclk switch is possible, RM clears display limit.
 * 6) On game close, DD will call LW0073_CTRL_CMD_SYSTEM_CONFIG_OSM_MCLK_SWITCH
 *    to lock pstate to maximum mclk frequency.
 * 7) DD sends stall lock/display rate and update method to trigger modeset
 *    which will trigger light weight HW supervisor interrupts.
 * 8) DD calls LW0073_CTRL_CMD_SYSTEM_CONFIG_VRR_PSTATE_SWITCH (true) to enable
 *    VRR Mclk switch and lower pstate when legacy mclk switch is not possible.
 * 9) If legacy mclk switch mechanism is possible, clear glitch limit.
 * 
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *   bisPreRunModeChange
 *     This flag tells about HW run mode method and is set to
 *     LW_TRUE if we are about to transition from OSM to continuous display mode.
 *     LW_FALSE if we have just completed a transition from continuous display 
 *     mode to OSM.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0073_CTRL_CMD_SYSTEM_CONFIG_OSM_MCLK_SWITCH (0x730195U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_CONFIG_OSM_MCLK_SWITCH_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_CONFIG_OSM_MCLK_SWITCH_PARAMS_MESSAGE_ID (0x95U)

typedef struct LW0073_CTRL_SYSTEM_CONFIG_OSM_MCLK_SWITCH_PARAMS {
    LwU32  subDeviceInstance;
    LwBool bIsPreRunModeChange;
} LW0073_CTRL_SYSTEM_CONFIG_OSM_MCLK_SWITCH_PARAMS;
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW0073_CTRL_CMD_SYSTEM_ALLOCATE_DISPLAY_BANDWIDTH
 *
 * This command allocates a specified amount of ISO memory bandwidth for
 * display.  If the requested amount of bandwidth cannot be allocated (either
 * because it exceeds the total bandwidth available to the system, or because
 * too much bandwidth is already allocated to other clients), the call will
 * fail and LW_ERR_INSUFFICIENT_RESOURCES will be returned.
 * 
 * If bandwidth has already been allocated via a prior call, and a new
 * allocation is requested, the new allocation will replace the old one.  (If
 * the new allocation fails, the old allocation remains in effect.)
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *   averageBandwidthKBPS
 *     This parameter specifies the amount of ISO memory bandwidth requested.
 *   floorBandwidthKBPS
 *     This parameter specifies the minimum required (i.e., floor) dramclk
 *     frequency, multiplied by the width of the pipe over which the display
 *     data will travel.  (It is understood that the bandwidth callwlated by
 *     multiplying the clock frequency by the pipe width will not be
 *     realistically achievable, due to overhead in the memory subsystem.  The
 *     API will not actually use the bandwidth value, except to reverse the
 *     callwlation to get the required dramclk frequency.)
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_INSUFFICIENT_RESOURCES
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_GENERIC
 */

#define LW0073_CTRL_CMD_SYSTEM_ALLOCATE_DISPLAY_BANDWIDTH (0x730196U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_SYSTEM_ALLOCATE_DISPLAY_BANDWIDTH_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_SYSTEM_ALLOCATE_DISPLAY_BANDWIDTH_PARAMS_MESSAGE_ID (0x96U)

typedef struct LW0073_CTRL_SYSTEM_ALLOCATE_DISPLAY_BANDWIDTH_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 averageBandwidthKBPS;
    LwU32 floorBandwidthKBPS;
} LW0073_CTRL_SYSTEM_ALLOCATE_DISPLAY_BANDWIDTH_PARAMS;

/*
 * LW0073_CTRL_SYSTEM_HOTPLUG_EVENT_CONFIG_PARAMS
 *
 * This structure represents the hotplug event config control parameters.
 *
 *   subDeviceInstance
 *     This parameter should specify the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *
 *   deviceMapFilter
 *     This parameter returns (in GET) or should specify (in SET) a device map
 *     indicating device(s) to sense.
 *
 *   hotPluginSense
 *     This parameter returns (in GET) or should specify (in SET) a device map
 *     indicating device(s) plugged in that caused the most recent hotplug
 *     event.
 *
 *   hotUnplugSense
 *     This parameter returns (in GET) or should specify (in SET) a device map
 *     indicating device(s) un plugged that caused the most recent hotplug
 *     event.
 */

typedef struct LW0073_CTRL_SYSTEM_HOTPLUG_EVENT_CONFIG_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 deviceMapFilter;
    LwU32 hotPluginSense;
    LwU32 hotUnplugSense;
} LW0073_CTRL_SYSTEM_HOTPLUG_EVENT_CONFIG_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_HOTPLUG_EVENT_CONFIG
 *
 * This command fetches the hotplug event configuration.
 *
 * See @ref LW0073_CTRL_SYSTEM_HOTPLUG_EVENT_CONFIG_PARAMS for documentation on
 * the parameters.
 */

#define LW0073_CTRL_CMD_SYSTEM_GET_HOTPLUG_EVENT_CONFIG  (0x730197U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | 0x97" */

/*
 * LW0073_CTRL_CMD_SYSTEM_SET_HOTPLUG_EVENT_CONFIG
 *
 * This command sets the hotplug event configuration.
 *
 * See @ref LW0073_CTRL_SYSTEM_HOTPLUG_EVENT_CONFIG_PARAMS for documentation on
 * the parameters.
 */

#define LW0073_CTRL_CMD_SYSTEM_SET_HOTPLUG_EVENT_CONFIG  (0x730198U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | 0x98" */


/*
 * LW0073_CTRL_CMD_SYSTEM_GET_PANEL_BRIGHTNESS_INFO
 *
 * Fetches the brightness info of the panel from DPCD Register values
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *   displayId
 *     DisplayId of the panel for which we need to the backight type
 *   dpcd_data[16]  
 *     array containing all the values stored in the DPCD Registers
 *   TargetFreq
 *     target frequency in PWM.
 *   ActualFreq
 *     actual frequency in PWM.
 *   DutyCycle
 *     Duty cycle in PWM.
 *   Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0073_CTRL_CMD_SYSTEM_GET_PANEL_BRIGHTNESS_INFO (0x730199U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_CMD_SYSTEM_GET_PANEL_BRIGHTNESS_INFO_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_SYSTEM_GET_PANEL_BRIGHTNESS_INFO_PARAMS_MESSAGE_ID (0x99U)

typedef struct LW0073_CTRL_CMD_SYSTEM_GET_PANEL_BRIGHTNESS_INFO_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU8  dpcd_data[16];
    LwU32 targetFreq;
    LwU32 actualFreq;
    LwU32 dutyCycle;
} LW0073_CTRL_CMD_SYSTEM_GET_PANEL_BRIGHTNESS_INFO_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_GET_SBIOS_BRIGHTNESS_INFO
 *
 * Fetches the Sbios information from corresponding registers.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *   displayId
 *     DisplayId of the panel for which we need to the backight type
 *   backlightType  
 *     array containing all the values stored in the DPCD Registers
 *   pwmInfoProvider
 *     target frequency in PWM.
 *   pwmInfoEntries
 *     actual frequency in PWM.
 *   nbciControlType
 *     Control type.
 *   minDutyCyclePer1000
 *     minimum duty cycle.
 *   maxDutyCyclePer1000
 *     maximum duty cycle.
 *   input_lwrve  
 *     points of the input pwm lwrve.
 *   output_lwrve
 *     points of the pwm output lwrve.
 *   Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0073_CTRL_CMD_SYSTEM_GET_SBIOS_BRIGHTNESS_INFO (0x73019aU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_CMD_SYSTEM_GET_SBIOS_BRIGHTNESS_INFO_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_SYSTEM_GET_SBIOS_BRIGHTNESS_INFO_PARAMS_MESSAGE_ID (0x9AU)

typedef struct LW0073_CTRL_CMD_SYSTEM_GET_SBIOS_BRIGHTNESS_INFO_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU8  backlightType;
    LwU8  pwmInfoProvider;
    LwU8  pwmInfoEntries;
    LwU16 minDutyCyclePer1000[20];
    LwU16 maxDutyCyclePer1000[20];
    LwU32 nbciControlType;
    LwU32 pwmFrequency[20];
    LwU16 input_lwrve[20];
    LwU16 output_lwrve[20];
    LwU32 pwmInfoEntriesTotal;
    LwU32 nEntries;
} LW0073_CTRL_CMD_SYSTEM_GET_SBIOS_BRIGHTNESS_INFO_PARAMS;

/*
*  LW0073_CTRL_CMD_SYSTEM_RECORD_CHANNEL_REGS
*
*  This command is used to read Core channel, Cursor channel, Window channel, and Head register values and encode these values with ProtoDmp.
*
*  subDeviceInstance (in)
*    This parameter specifies the subdevice instance within the
*    LW04_DISPLAY_COMMON parent device to which the operation should be
*    directed.
*  headMask (in)
*    Head mask representing which register values should be encoded
*  windowMask (in)
*    Window channel mask whose register values should be encoded
*  bRecordCoreChannel (in)
*    Indicates whether or not to encode core channel register values
*  bRecordLwrsorChannel (in)
*    Indicates whether or not to encode cursor channel register values
*
*  Possible status values returned are:
*    LW_OK
*    LW_ERR_ILWALID_ARGUMENT
*    LW_ERR_NOT_SUPPORTED
*/
#define LW0073_CTRL_CMD_SYSTEM_RECORD_CHANNEL_REGS (0x73019bU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_CMD_SYSTEM_RECORD_CHANNEL_REGS_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_SYSTEM_RECORD_CHANNEL_REGS_PARAMS_MESSAGE_ID (0x9BU)

typedef struct LW0073_CTRL_CMD_SYSTEM_RECORD_CHANNEL_REGS_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  headMask;
    LwU32  windowMask;
    LwBool bRecordCoreChannel;
    LwBool bRecordLwrsorChannel;
} LW0073_CTRL_CMD_SYSTEM_RECORD_CHANNEL_REGS_PARAMS;

/*
 * LW0073_CTRL_CMD_SYSTEM_CHECK_SIDEBAND_I2C_SUPPORT
 *
 * This command is used to query the display mux status for the given
 * display device
 *
 *   subDeviceInstance (in)
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0073_CTRL_CMD_SYSTEM_CHECK_SIDEBAND_I2C_SUPPORT (0x73019lw) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SYSTEM_INTERFACE_ID << 8) | LW0073_CTRL_CMD_SYSTEM_CHECK_SIDEBAND_I2C_SUPPORT_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_SYSTEM_CHECK_SIDEBAND_I2C_SUPPORT_PARAMS_MESSAGE_ID (0x9LW)

typedef struct LW0073_CTRL_CMD_SYSTEM_CHECK_SIDEBAND_I2C_SUPPORT_PARAMS {
    LwU32  subDeviceInstance;
    LwBool bIsSidebandI2cSupported;
} LW0073_CTRL_CMD_SYSTEM_CHECK_SIDEBAND_I2C_SUPPORT_PARAMS;

/* _ctrl0073system_h_ */

