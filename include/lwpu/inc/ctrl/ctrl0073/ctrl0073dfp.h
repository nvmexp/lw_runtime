/*
 * SPDX-FileCopyrightText: Copyright (c) 2005-2022 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0073/ctrl0073dfp.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl0073/ctrl0073base.h"

/* LW04_DISPLAY_COMMON dfp-display-specific control commands and parameters */

/*
 * LW0073_CTRL_CMD_DFP_GET_INFO
 *
 * This command can be used to determine the associated display type for
 * the specified displayId.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the display for which the dfp
 *     caps should be returned.  The display ID must be a dfp display
 *     as determined with the LW0073_CTRL_CMD_SPECIFIC_GET_TYPE command.
 *     If more than one displayId bit is set or the displayId is not a dfp,
 *     this call will return LW_ERR_ILWALID_ARGUMENT.
 *   flags
 *     This parameter returns the information specific to this dfp.  Here are
 *     the possible fields:
 *       LW0073_CTRL_DFP_FLAGS_SIGNAL
 *         This specifies the type of signal used for this dfp.
 *       LW0073_CTRL_DFP_FLAGS_LANES
 *         This specifies whether the board supports 1, 2, or 4 lanes
 *         for DISPLAYPORT signals.
 *       LW0073_CTRL_DFP_FLAGS_LIMIT
 *         Some GPUs were not qualified to run internal TMDS except at 60 HZ
 *         refresh rates.  So, if LIMIT_60HZ_RR is set, then the client must
 *         make sure to only allow 60 HZ refresh rate modes to the OS/User.
 *       LW0073_CTRL_DFP_FLAGS_SLI_SCALER
 *         While running in SLI, if SLI_SCALER_DISABLE is set, the GPU cannot
 *         scale any resolutions.  So, the output timing must match the
 *         memory footprint.
 *       LW0073_CTRL_DFP_FLAGS_HDMI_CAPABLE
 *         This specifies whether the DFP displayId is capable of
 *         transmitting HDMI.
 *       LW0073_CTRL_DFP_FLAGS_RANGE_LIMITED_CAPABLE
 *         This specifies whether the displayId is capable of sending a
 *         limited color range out from the board.
 *       LW0073_CTRL_DFP_FLAGS_RANGE_AUTO_CAPABLE
 *         This specifies whether the displayId is capable of auto-configuring
 *         the color range.
 *       LW0073_CTRL_DFP_FLAGS_FORMAT_YCBCR422_CAPABLE
 *         This specifies whether the displayId is capable of sending the
 *         YCBCR422 color format out from the board.
 *       LW0073_CTRL_DFP_FLAGS_FORMAT_YCBCR444_CAPABLE
 *         This specifies whether the displayId is capable of sending
 *         YCBCR444 color format out from the board.
 *       LW0073_CTRL_DFP_FLAGS_DP_LINK_BANDWIDTH
 *         This specifies whether the displayId is capable of doing high
 *         bit-rate (2.7Gbps) or low bit-rate (1.62Gbps) if the DFP is
 *         display port.
 *       LW0073_CTRL_DFP_FLAGS_HDMI_ALLOWED
 *         This specifies whether the DFP displayId is allowed to transmit HDMI
 *         based on the VBIOS settings.
 *       LW0073_CTRL_DFP_FLAGS_EMBEDDED_DISPLAYPORT
 *         This specifies whether the DFP displayId is actually an embedded display
 *         port based on VBIOS connector information AND ASSR cap.
 *       LW0073_CTRL_DFP_FLAGS_DP_LINK_CONSTRAINT
 *         This specifies whether the DFP displayId must be trained to RBR mode
 *         (if it is using DP protocol) whenever possible.
 *       LW0073_CTRL_DFP_FLAGS_LINK
 *         This specifies whether the board supports single or dual links
 *         for TMDS, LVDS, and SDI signals.
 *       LW0073_CTRL_DFP_FLAGS_DP_POST_LWRSOR2_DISABLED
 *         This specifies if PostLwrsor2 is disabled in the VBIOS
 *       LW0073_CTRL_DFP_FLAGS_DSI_DEVICE_ID
 *         This indicates whether this SOR uses DSI-A, DSI-B or both (ganged mode).
 *       LW0073_CTRL_DFP_FLAGS_DYNAMIC_MUX_CAPABLE
 *         This indicates whether this DFP supports Dynamic MUX
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_CMD_DFP_GET_INFO (0x731140U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_DFP_GET_INFO_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DFP_GET_INFO_PARAMS_MESSAGE_ID (0x40U)

typedef struct LW0073_CTRL_DFP_GET_INFO_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 flags;
} LW0073_CTRL_DFP_GET_INFO_PARAMS;

/* valid display types */
#define LW0073_CTRL_DFP_FLAGS_SIGNAL                                       2:0
#define LW0073_CTRL_DFP_FLAGS_SIGNAL_TMDS                    (0x00000000U)
#define LW0073_CTRL_DFP_FLAGS_SIGNAL_LVDS                    (0x00000001U)
#define LW0073_CTRL_DFP_FLAGS_SIGNAL_SDI                     (0x00000002U)
#define LW0073_CTRL_DFP_FLAGS_SIGNAL_DISPLAYPORT             (0x00000003U)
#define LW0073_CTRL_DFP_FLAGS_SIGNAL_DSI                     (0x00000004U)
#define LW0073_CTRL_DFP_FLAGS_SIGNAL_WRBK                    (0x00000005U)
#define LW0073_CTRL_DFP_FLAGS_LANE                                         5:3
#define LW0073_CTRL_DFP_FLAGS_LANE_NONE                      (0x00000000U)
#define LW0073_CTRL_DFP_FLAGS_LANE_SINGLE                    (0x00000001U)
#define LW0073_CTRL_DFP_FLAGS_LANE_DUAL                      (0x00000002U)
#define LW0073_CTRL_DFP_FLAGS_LANE_QUAD                      (0x00000003U)
#define LW0073_CTRL_DFP_FLAGS_LANE_OCT                       (0x00000004U)
#define LW0073_CTRL_DFP_FLAGS_LIMIT                                        6:6
#define LW0073_CTRL_DFP_FLAGS_LIMIT_DISABLE                  (0x00000000U)
#define LW0073_CTRL_DFP_FLAGS_LIMIT_60HZ_RR                  (0x00000001U)
#define LW0073_CTRL_DFP_FLAGS_SLI_SCALER                                   7:7
#define LW0073_CTRL_DFP_FLAGS_SLI_SCALER_NORMAL              (0x00000000U)
#define LW0073_CTRL_DFP_FLAGS_SLI_SCALER_DISABLE             (0x00000001U)
#define LW0073_CTRL_DFP_FLAGS_HDMI_CAPABLE                                 8:8
#define LW0073_CTRL_DFP_FLAGS_HDMI_CAPABLE_FALSE             (0x00000000U)
#define LW0073_CTRL_DFP_FLAGS_HDMI_CAPABLE_TRUE              (0x00000001U)
#define LW0073_CTRL_DFP_FLAGS_RANGE_LIMITED_CAPABLE                        9:9
#define LW0073_CTRL_DFP_FLAGS_RANGE_LIMITED_CAPABLE_FALSE    (0x00000000U)
#define LW0073_CTRL_DFP_FLAGS_RANGE_LIMITED_CAPABLE_TRUE     (0x00000001U)
#define LW0073_CTRL_DFP_FLAGS_RANGE_AUTO_CAPABLE                         10:10
#define LW0073_CTRL_DFP_FLAGS_RANGE_AUTO_CAPABLE_FALSE       (0x00000000U)
#define LW0073_CTRL_DFP_FLAGS_RANGE_AUTO_CAPABLE_TRUE        (0x00000001U)
#define LW0073_CTRL_DFP_FLAGS_FORMAT_YCBCR422_CAPABLE                    11:11
#define LW0073_CTRL_DFP_FLAGS_FORMAT_YCBCR422_CAPABLE_FALSE  (0x00000000U)
#define LW0073_CTRL_DFP_FLAGS_FORMAT_YCBCR422_CAPABLE_TRUE   (0x00000001U)
#define LW0073_CTRL_DFP_FLAGS_FORMAT_YCBCR444_CAPABLE                    12:12
#define LW0073_CTRL_DFP_FLAGS_FORMAT_YCBCR444_CAPABLE_FALSE  (0x00000000U)
#define LW0073_CTRL_DFP_FLAGS_FORMAT_YCBCR444_CAPABLE_TRUE   (0x00000001U)
#define LW0073_CTRL_DFP_FLAGS_HDMI_ALLOWED                               14:14
#define LW0073_CTRL_DFP_FLAGS_HDMI_ALLOWED_FALSE             (0x00000000U)
#define LW0073_CTRL_DFP_FLAGS_HDMI_ALLOWED_TRUE              (0x00000001U)
#define LW0073_CTRL_DFP_FLAGS_EMBEDDED_DISPLAYPORT                       15:15
#define LW0073_CTRL_DFP_FLAGS_EMBEDDED_DISPLAYPORT_FALSE     (0x00000000U)
#define LW0073_CTRL_DFP_FLAGS_EMBEDDED_DISPLAYPORT_TRUE      (0x00000001U)
#define LW0073_CTRL_DFP_FLAGS_DP_LINK_CONSTRAINT                         16:16
#define LW0073_CTRL_DFP_FLAGS_DP_LINK_CONSTRAINT_NONE        (0x00000000U)
#define LW0073_CTRL_DFP_FLAGS_DP_LINK_CONSTRAINT_PREFER_RBR  (0x00000001U)
#define LW0073_CTRL_DFP_FLAGS_DP_LINK_BW                                 19:17
#define LW0073_CTRL_DFP_FLAGS_DP_LINK_BW_1_62GBPS            (0x00000001U)
#define LW0073_CTRL_DFP_FLAGS_DP_LINK_BW_2_70GBPS            (0x00000002U)
#define LW0073_CTRL_DFP_FLAGS_DP_LINK_BW_5_40GBPS            (0x00000003U)
#define LW0073_CTRL_DFP_FLAGS_DP_LINK_BW_8_10GBPS            (0x00000004U)
#define LW0073_CTRL_DFP_FLAGS_LINK                                       21:20
#define LW0073_CTRL_DFP_FLAGS_LINK_NONE                      (0x00000000U)
#define LW0073_CTRL_DFP_FLAGS_LINK_SINGLE                    (0x00000001U)
#define LW0073_CTRL_DFP_FLAGS_LINK_DUAL                      (0x00000002U)
#define LW0073_CTRL_DFP_FLAGS_DP_FORCE_RM_EDID                           22:22
#define LW0073_CTRL_DFP_FLAGS_DP_FORCE_RM_EDID_FALSE         (0x00000000U)
#define LW0073_CTRL_DFP_FLAGS_DP_FORCE_RM_EDID_TRUE          (0x00000001U)
#define LW0073_CTRL_DFP_FLAGS_DSI_DEVICE_ID                              24:23
#define LW0073_CTRL_DFP_FLAGS_DSI_DEVICE_ID_DSI_NONE         (0x00000000U)
#define LW0073_CTRL_DFP_FLAGS_DSI_DEVICE_ID_DSI_A            (0x00000001U)
#define LW0073_CTRL_DFP_FLAGS_DSI_DEVICE_ID_DSI_B            (0x00000002U)
#define LW0073_CTRL_DFP_FLAGS_DSI_DEVICE_ID_DSI_GANGED       (0x00000003U)
#define LW0073_CTRL_DFP_FLAGS_DP_POST_LWRSOR2_DISABLED                   25:25
#define LW0073_CTRL_DFP_FLAGS_DP_POST_LWRSOR2_DISABLED_FALSE (0x00000000U)
#define LW0073_CTRL_DFP_FLAGS_DP_POST_LWRSOR2_DISABLED_TRUE  (0x00000001U)
#define LW0073_CTRL_DFP_FLAGS_DP_PHY_REPEATER_COUNT                      29:26
#define LW0073_CTRL_DFP_FLAGS_DYNAMIC_MUX_CAPABLE                        30:30
#define LW0073_CTRL_DFP_FLAGS_DYNAMIC_MUX_CAPABLE_FALSE      (0x00000000U)
#define LW0073_CTRL_DFP_FLAGS_DYNAMIC_MUX_CAPABLE_TRUE       (0x00000001U)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW0073_CTRL_CMD_DFP_SET_PANEL_POWER
 *
 * This command can be used to enable or disable the panel power for mobile
 * displays.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the dfp display which owns the
 *     panel power to adjust.  The display ID must be a dfp display
 *     as determined with the LW0073_CTRL_CMD_SPECIFIC_GET_TYPE command.
 *     If more than one displayId bit is set or the displayId is not a dfp,
 *     this call will return LW_ERR_ILWALID_ARGUMENT.
 *   enable
 *     This parameter sets whether to enable or disable the
 *     panel power.  Current values are:
 *       LW0073_CTRL_DFP_SET_PANEL_POWER_ENABLE_OFF
 *         This value will disable the panel.
 *       LW0073_CTRL_DFP_SET_PANEL_POWER_ENABLE_ON
 *         This value will enable the panel.
 *   retryTimeMs
 *     This parameter is an output to this command.  In case of
 *     LWOS_STATUS_ERROR_RETRY return status, this parameter returns the time
 *     duration in milli-seconds after which client should retry this command.
 *
*/
#define LW0073_CTRL_CMD_DFP_SET_PANEL_POWER                  (0x731141U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_DFP_SET_PANEL_POWER_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DFP_SET_PANEL_POWER_PARAMS_MESSAGE_ID (0x41U)

typedef struct LW0073_CTRL_DFP_SET_PANEL_POWER_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 enable;
    LwU32 retryTimeMs;
} LW0073_CTRL_DFP_SET_PANEL_POWER_PARAMS;

#define LW0073_CTRL_DFP_SET_PANEL_POWER_ENABLE_OFF      (0x00000000U)
#define LW0073_CTRL_DFP_SET_PANEL_POWER_ENABLE_ON       (0x00000001U)

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW0073_CTRL_CMD_DFP_GET_DP2TMDS_DONGLE_INFO
 *
 * This command can be used to determine information about dongles attached
 * to a displayport connection.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the dfp display which owns the
 *     panel power to adjust.  The display ID must be a dfp display
 *     as determined with the LW0073_CTRL_CMD_SPECIFIC_GET_TYPE command.
 *     If more than one displayId bit is set or the displayId is not a dfp,
 *     this call will return LW_ERR_ILWALID_ARGUMENT.
 *   flags
 *     This parameter provide state information about the dongle attachments.
 *       LW0073_CTRL_DFP_GET_DP2TMDS_DONGLE_INFO_FLAGS_CAPABLE
 *         Specifies if the connection is capable of a dongle.  This field
 *         returns false in all cases of signal types except for those capable
 *         of outputting TMDS.  Even then the if a gpio is not defined, the
 *         the a statement of false will also be returned.
 *       LW0073_CTRL_DFP_GET_DP2TMDS_DONGLE_INFO_FLAGS_ATTACHED
 *         When attached, this value specifies that a dongle is detected and
 *         attached.  The client should read the _TYPE field to determine
 *         if it is a dp2hdmi or dp2dvi dongle.
 *      LW0073_CTRL_DFP_GET_DP2TMDS_DONGLE_INFO_FLAGS_TYPE
 *         _DP2DVI: no response to i2cAddr 0x80 per DP interop guidelines.
 *                  clients MUST avoid outputting HDMI even if capable.
 *         _DP2HDMI: dongle responds to i2cAddr 0x80 per DP interop guidelines.
 *                   client is allowed to output HDMI when possible.
 *         _LFH_DVI: DMS59-DVI breakout dongle is in use.
 *         _LFH_VGA: DMS59-VGA breakout dongle is in use.
 *      LW0073_CTRL_DFP_GET_DISPLAYPORT_DONGLE_INFO_FLAGS_DP2TMDS_DONGLE_TYPE
 *         _1: Max TMDS Clock rate is 165 MHz for both DVI and HDMI.
 *         _2: Max TMDS Clock rate will be specified in the dongle
 *              address space at device address 0x80.
 *              DVI  is up to 165 MHz
 *              HDMI is up to 300 MHz
 *              There are type 2 devices that support beyond 600 MHz
 *              though not defined in the spec.
 *   maxTmdsClkRateHz
 *     This defines the max TMDS clock rate for dual mode adaptor in Hz.
 */
#define LW0073_CTRL_CMD_DFP_GET_DISPLAYPORT_DONGLE_INFO (0x731142U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_DFP_GET_DISPLAYPORT_DONGLE_INFO_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DFP_GET_DISPLAYPORT_DONGLE_INFO_PARAMS_MESSAGE_ID (0x42U)

typedef struct LW0073_CTRL_DFP_GET_DISPLAYPORT_DONGLE_INFO_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 flags;
    LwU32 maxTmdsClkRateHz;
} LW0073_CTRL_DFP_GET_DISPLAYPORT_DONGLE_INFO_PARAMS;

#define  LW0073_CTRL_DFP_GET_DISPLAYPORT_DONGLE_INFO_FLAGS_CAPABLE                  0:0
#define LW0073_CTRL_DFP_GET_DISPLAYPORT_DONGLE_INFO_FLAGS_CAPABLE_FALSE         (0x00000000U)
#define LW0073_CTRL_DFP_GET_DISPLAYPORT_DONGLE_INFO_FLAGS_CAPABLE_TRUE          (0x00000001U)
#define  LW0073_CTRL_DFP_GET_DISPLAYPORT_DONGLE_INFO_FLAGS_ATTACHED                 1:1
#define LW0073_CTRL_DFP_GET_DISPLAYPORT_DONGLE_INFO_FLAGS_ATTACHED_FALSE        (0x00000000U)
#define LW0073_CTRL_DFP_GET_DISPLAYPORT_DONGLE_INFO_FLAGS_ATTACHED_TRUE         (0x00000001U)
#define  LW0073_CTRL_DFP_GET_DISPLAYPORT_DONGLE_INFO_FLAGS_TYPE                     7:4
#define LW0073_CTRL_DFP_GET_DISPLAYPORT_DONGLE_INFO_FLAGS_TYPE_DP2DVI           (0x00000000U)
#define LW0073_CTRL_DFP_GET_DISPLAYPORT_DONGLE_INFO_FLAGS_TYPE_DP2HDMI          (0x00000001U)
#define LW0073_CTRL_DFP_GET_DISPLAYPORT_DONGLE_INFO_FLAGS_TYPE_LFH_DVI          (0x00000002U)
#define LW0073_CTRL_DFP_GET_DISPLAYPORT_DONGLE_INFO_FLAGS_TYPE_LFH_VGA          (0x00000003U)
#define  LW0073_CTRL_DFP_GET_DISPLAYPORT_DONGLE_INFO_FLAGS_DP2TMDS_DONGLE_TYPE      8:8
#define LW0073_CTRL_DFP_GET_DISPLAYPORT_DONGLE_INFO_FLAGS_DP2TMDS_DONGLE_TYPE_1 (0x00000000U)
#define LW0073_CTRL_DFP_GET_DISPLAYPORT_DONGLE_INFO_FLAGS_DP2TMDS_DONGLE_TYPE_2 (0x00000001U)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW0073_CTRL_DFP_EDID_AUDIO_CAPS
 *
 * This structure describes audioCaps information as outlined in the
 * CEA861-X specs.
 *
 *   caps
 *     This parameter specifies the valid defines to be used when sending this
 *     call.
 *     Here are the current defined fields:
 *       LW0073_CTRL_DFP_EDID_AUDIO_CAPS_FORMAT
 *         A audio format based on CEA861-X.
 *       LW0073_CTRL_CMD_DFP_SET_EDID_AUDIO_CAPS_SAMPLE_SIZE
 *         A client specifies the sample size capabilities as defined by the
 *         HDMI display receiver's EDID.  These are only applicable to LPCM.
 *       LW0073_CTRL_CMD_DFP_SET_EDID_AUDIO_CAPS_SAMPLE_FREQ
 *         A client specifies the sample frequencies capabilities as defined
 *         by the HDMI display receiver's EDID.
 *       LW0073_CTRL_CMD_DFP_EDID_AUDIO_CAPS_CHN_COUNT
 *         A client specifies the audio channel count as defined by the
 *         HDMI display receiver's EDID.  The default is always 2 channels.
 *       LW0073_CTRL_DFP_EDID_AUDIO_CAPS_MBR_DIV_BY_8
 *         A client specifies the maximum bit rate divided by 8Khz per CEA861-X
 *         spec for certain audio formats.
 *       LW0073_CTRL_DFP_EDID_AUDIO_CAPS_VALID
 *         Whether the format is valid or invalid.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

typedef struct LW0073_CTRL_DFP_EDID_AUDIO_CAPS {
    LwU32 caps;
} LW0073_CTRL_DFP_EDID_AUDIO_CAPS;

#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_FORMAT                     3:0
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_FORMAT_LPCM         (0x00000001U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_FORMAT_AC3          (0x00000002U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_FORMAT_MPEG1        (0x00000003U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_FORMAT_MP3          (0x00000004U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_FORMAT_MPEG2        (0x00000005U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_FORMAT_AAC          (0x00000006U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_FORMAT_DTS          (0x00000007U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_FORMAT_ATRAC        (0x00000008U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_FORMAT_ONEBIT       (0x00000009U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_FORMAT_DDP          (0x0000000aU)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_FORMAT_DTSHD        (0x0000000bU)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_FORMAT_MAT          (0x0000000lw)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_FORMAT_DST          (0x0000000dU)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_FORMAT_WMA          (0x0000000eU)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_MBR_DIV_BY_8               11:4
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_SAMPLE_SIZE                16:14
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_SAMPLE_SIZE_16BIT   (0x00000001U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_SAMPLE_SIZE_20BIT   (0x00000002U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_SAMPLE_SIZE_24BIT   (0x00000004U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_SAMPLE_FREQ                24:17
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_SAMPLE_FREQ_0320KHZ (0x00000001U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_SAMPLE_FREQ_0441KHZ (0x00000002U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_SAMPLE_FREQ_0480KHZ (0x00000004U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_SAMPLE_FREQ_0882KHZ (0x00000008U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_SAMPLE_FREQ_0960KHZ (0x00000010U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_SAMPLE_FREQ_1764KHZ (0x00000020U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_SAMPLE_FREQ_1920KHZ (0x00000040U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_COUNT                  28:25
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_COUNT_2         (0x00000001U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_COUNT_3         (0x00000002U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_COUNT_4         (0x00000003U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_COUNT_5         (0x00000004U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_COUNT_6         (0x00000005U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_COUNT_7         (0x00000006U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_COUNT_8         (0x00000007U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_VALID                      31:31
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_VALID_NO            (0x00000000U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_VALID_YES           (0x00000001U)

/* TODO: bug #1975866: LW0073_CTRL_DFP_SET_EDID_AUDIO_CAPS_PARAMS parameters are used elsewhere in code
 * Examine the other usages of this parameter and perhaps remove those use cases
 *
 * LW0073_CTRL_CMD_DFP_SET_EDID_AUDIO_CAPS
 *
 * This command is used to inform hardware the HDMI receiver's audio
 * capabilities.  The client shall parse the HDMI receiver and specify the
 * sample frequency, sample size, and audio formats that are supported based on
 * CEA861B.  The client should inform hardware at initial boot, a modeset, and
 * whenever a hotplug event oclwrs.
 *
 *   deviceMask
 *     This parameter indicates the digital display device's
 *     mask. This comes as input to this command.
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *     This parameter must specify a value between zero and the total number
 *     of subdevices within the parent device.  This parameter should be set
 *     to zero for default behavior.
 *   numEntries
 *     This parameter specifies how many entries RM should loop and which
 *     contain entries that RM should write to the hardware.
 *   audioCaps
 *     This parameter specifies the valid defines to be used when sending
 *     this call. See the description of LW0073_CTRL_DFP_EDID_AUDIO_CAPS for
 *     details on audioCaps information.
 *   maxFreqSupported
 *     Supply the maximum frequency supported for the overall audio caps.
 *     This value should match CEA861-X defines for sample freq.
 *   chnAllocation
 *     A bitmask of the speaker allocation capabilities.
 *       LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_FL_FR
 *         Specify _PRESENT for front left and front right speakers.
 *     LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_LFE
 *         Specify _PRESENT for low frequency effect.
 *     LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_FC
 *         Specify _PRESENT for front center speakers.
 *     LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_RL_RR
 *         Specify _PRESENT for rear left, rear right location.
 *     LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_RC
 *         Specify _PRESENT for rear center speakers
 *     LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_RLC_RRC
 *         Specify _PRESENT for rear left center and rear right center.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *
 * Note: This command is DEPRECATED after Tesla!
 *
 */
#define LW0073_CTRL_CMD_DFP_SET_EDID_AUDIO_CAPS             (0x731143U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | 0x43" */

#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_MAX_NUM             16U

typedef struct LW0073_CTRL_DFP_SET_EDID_AUDIO_CAPS_PARAMS {
    LwU32                           subDeviceInstance;
    LwU32                           displayId;
    LwU32                           numEntries;
    LW0073_CTRL_DFP_EDID_AUDIO_CAPS audioCaps[LW0073_CTRL_DFP_EDID_AUDIO_CAPS_MAX_NUM];
    LwU32                           maxFreqSupported;
    LwU32                           chnAllocation;
} LW0073_CTRL_DFP_SET_EDID_AUDIO_CAPS_PARAMS;

#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_MAX_FREQ_SUPPORTED_0320KHZ     (0x00000001U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_MAX_FREQ_SUPPORTED_0441KHZ     (0x00000002U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_MAX_FREQ_SUPPORTED_0480KHZ     (0x00000003U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_MAX_FREQ_SUPPORTED_0882KHZ     (0x00000004U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_MAX_FREQ_SUPPORTED_0960KHZ     (0x00000005U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_MAX_FREQ_SUPPORTED_1764KHZ     (0x00000006U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_MAX_FREQ_SUPPORTED_1920KHZ     (0x00000007U)

#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_FL_FR                       0:0
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_FL_FR_ABSENT    (0x00000000U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_FL_FR_PRESENT   (0x00000001U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_LFE                         1:1
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_LFE_ABSENT      (0x00000000U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_LFE_PRESENT     (0x00000001U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_FC                          2:2
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_FC_ABSENT       (0x00000000U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_FC_PRESENT      (0x00000001U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_RL_RR                       3:3
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_RL_RR_ABSENT    (0x00000000U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_RL_RR_PRESENT   (0x00000001U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_RC                          4:4
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_RC_ABSENT       (0x00000000U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_RC_PRESENT      (0x00000001U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_FLC_FRC                     5:5
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_FLC_FRC_ABSENT  (0x00000000U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_FLC_FRC_PRESENT (0x00000001U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_RLC_RRC                     6:6
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_RLC_RRC_ABSENT  (0x00000000U)
#define LW0073_CTRL_DFP_EDID_AUDIO_CAPS_CHN_ALLOCATION_RLC_RRC_PRESENT (0x00000001U)

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW0073_CTRL_CMD_DFP_SET_ELD_AUDIO_CAPS
 *
 * This command is used to inform hardware the receiver's audio capabilities
 * using the new EDID Like Data (ELD) memory structure.  The ELD memory
 * structure is read by the audio driver by issuing the ELD Data command verb.
 * This mechanism is used for passing sink device' audio EDID information
 * from graphics software to audio software.  ELD contents will contain a
 * subset of the sink device's EDID information.
 * The client should inform hardware at initial boot, a modeset, and whenever
 * a hotplug event oclwrs.
 *
 *   displayId
 *     This parameter indicates the digital display device's
 *     mask. This comes as input to this command.
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be directed.
 *     This parameter must specify a value between zero and the total number
 *     of subdevices within the parent device.  This parameter should be set
 *     to zero for default behavior.
 *   numELDSize
 *     This parameter specifies how many bytes of data RM should write to the
 *     ELD buffer.  Section 7.3.3.36 of the ECN specifies that the ELD buffer
 *     size of zero based.  HDAudio driver will then use this information to
 *     determine how many bytes of the ELD buffer the HDAudio should read.
 *     The maximum size of the buffer is 96 bytes.
 *   bufferELD
 *     This buffer contains data as defined in the ECR HDMI ELD memory structure.
 *     Refer to the ELD Memory Structure Specification for more details.
 *     The format should be:
 *       - Header block is fixed at 4 bytes
 *         The header block contains the ELD version and baseline ELD len as
 *         well as some reserved fields.
 *       - Baseline block for audio descriptors is 76 bytes
 *         (15 SAD; each SAD=3 bytes requiring 45 bytes with 31 bytes to spare)
 *         As well as some other bits used to denote the CEA version,
 *         the speaker allocation data, monitor name, connector type, and
 *         hdcp capabilities.
 *       - Vendor specific block of 16 bytes
 *   maxFreqSupported
 *     Supply the maximum frequency supported for the overall audio caps.
 *     This value should match CEA861-X defines for sample freq.
 *   ctrl:
 *     LW0073_CTRL_DFP_SET_ELD_AUDIO_CAPS_CTRL_PD:
 *         Specifies the presence detect of the receiver.  On a hotplug
 *         or modeset client should set this bit to TRUE.
 *     LW0073_CTRL_DFP_SET_ELD_AUDIO_CAPS_CTRL_ELDV:
 *         Specifies whether the ELD buffer contents are valid.
 *         An intrinsic unsolicited response (UR) is generated whenever
 *         the ELDV bit changes in value and the PD=1. When _PD=1(hotplug),
 *         RM will set the ELDV bit after ELD buffer contents are written.
 *         If _ELDV bit is set to false such as during a unplug, then the
 *         contents of the ELD buffer will be cleared.
 *   deviceEntry:
 *     The deviceEntry number from which the SF should accept packets.
 *     _NONE if disabling audio.
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_CMD_DFP_SET_ELD_AUDIO_CAPS                         (0x731144U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_DFP_SET_ELD_AUDIO_CAP_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DFP_ELD_AUDIO_CAPS_ELD_BUFFER                      96U

#define LW0073_CTRL_DFP_SET_ELD_AUDIO_CAP_PARAMS_MESSAGE_ID (0x44U)

typedef struct LW0073_CTRL_DFP_SET_ELD_AUDIO_CAP_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 numELDSize;
    LwU8  bufferELD[LW0073_CTRL_DFP_ELD_AUDIO_CAPS_ELD_BUFFER];
    LwU32 maxFreqSupported;
    LwU32 ctrl;
    LwU32 deviceEntry;
} LW0073_CTRL_DFP_SET_ELD_AUDIO_CAP_PARAMS;

#define LW0073_CTRL_DFP_ELD_AUDIO_CAPS_MAX_FREQ_SUPPORTED_0320KHZ (0x00000001U)
#define LW0073_CTRL_DFP_ELD_AUDIO_CAPS_MAX_FREQ_SUPPORTED_0441KHZ (0x00000002U)
#define LW0073_CTRL_DFP_ELD_AUDIO_CAPS_MAX_FREQ_SUPPORTED_0480KHZ (0x00000003U)
#define LW0073_CTRL_DFP_ELD_AUDIO_CAPS_MAX_FREQ_SUPPORTED_0882KHZ (0x00000004U)
#define LW0073_CTRL_DFP_ELD_AUDIO_CAPS_MAX_FREQ_SUPPORTED_0960KHZ (0x00000005U)
#define LW0073_CTRL_DFP_ELD_AUDIO_CAPS_MAX_FREQ_SUPPORTED_1764KHZ (0x00000006U)
#define LW0073_CTRL_DFP_ELD_AUDIO_CAPS_MAX_FREQ_SUPPORTED_1920KHZ (0x00000007U)

#define LW0073_CTRL_DFP_ELD_AUDIO_CAPS_CTRL_PD                                     0:0
#define LW0073_CTRL_DFP_ELD_AUDIO_CAPS_CTRL_PD_FALSE              (0x00000000U)
#define LW0073_CTRL_DFP_ELD_AUDIO_CAPS_CTRL_PD_TRUE               (0x00000001U)
#define LW0073_CTRL_DFP_ELD_AUDIO_CAPS_CTRL_ELDV                                   1:1
#define LW0073_CTRL_DFP_ELD_AUDIO_CAPS_CTRL_ELDV_FALSE            (0x00000000U)
#define LW0073_CTRL_DFP_ELD_AUDIO_CAPS_CTRL_ELDV_TRUE             (0x00000001U)

#define LW0073_CTRL_DFP_ELD_AUDIO_CAPS_DEVICE_ENTRY_0             (0x00000000U)
#define LW0073_CTRL_DFP_ELD_AUDIO_CAPS_DEVICE_ENTRY_1             (0x00000001U)
#define LW0073_CTRL_DFP_ELD_AUDIO_CAPS_DEVICE_ENTRY_2             (0x00000002U)
#define LW0073_CTRL_DFP_ELD_AUDIO_CAPS_DEVICE_ENTRY_3             (0x00000003U)
#define LW0073_CTRL_DFP_ELD_AUDIO_CAPS_DEVICE_ENTRY_NONE          (0x00000007U)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

/*
 * LW0073_CTRL_DFP_VIDEO_COMPONENT_INFO
 *
 * This structure describes various parameters associated with
 * a video state component. The parameters include the following:
 *
 *   videoComponent
 *     The index for a video state component
 *   value
 *     This parameter returns the value of the video state component.
 *   algorithm
 *     This parameter returns the algorithm of the video state component.
 *   enable
 *     This parameter returns the override for a video state component.
 *
 */
typedef struct LW0073_CTRL_DFP_VIDEO_COMPONENT_INFO {
    LwU32 videoComponent;
    LwU32 value;
    LW_DECLARE_ALIGNED(LwU64 algorithm, 8);
    LwU32 enable;
} LW0073_CTRL_DFP_VIDEO_COMPONENT_INFO;

/* valid videoComponent values */
#define LW0073_CTRL_DFP_VIDEO_COMPONENT_INFO_INDEX_BRIGHTNESS   (0x00000000U)
#define LW0073_CTRL_DFP_VIDEO_COMPONENT_INFO_INDEX_CONTRAST     (0x00000001U)
#define LW0073_CTRL_DFP_VIDEO_COMPONENT_INFO_INDEX_HUE          (0x00000002U)
#define LW0073_CTRL_DFP_VIDEO_COMPONENT_INFO_INDEX_SATURATION   (0x00000003U)
#define LW0073_CTRL_DFP_VIDEO_COMPONENT_INFO_INDEX_COLORTEMP    (0x00000004U)
#define LW0073_CTRL_DFP_VIDEO_COMPONENT_INFO_INDEX_Y_GAMMA      (0x00000005U)
#define LW0073_CTRL_DFP_VIDEO_COMPONENT_INFO_INDEX_RGB_GAMMA_R  (0x00000006U)
#define LW0073_CTRL_DFP_VIDEO_COMPONENT_INFO_INDEX_RGB_GAMMA_G  (0x00000007U)
#define LW0073_CTRL_DFP_VIDEO_COMPONENT_INFO_INDEX_RGB_GAMMA_B  (0x00000008U)
#define LW0073_CTRL_DFP_VIDEO_COMPONENT_INFO_INDEX_COLOR_SPACE  (0x00000009U)
#define LW0073_CTRL_DFP_VIDEO_COMPONENT_INFO_INDEX_COLOR_RANGE  (0x0000000aU)
#define LW0073_CTRL_DFP_VIDEO_COMPONENT_INFO_INDEX_DEINTERLACE  (0x0000000bU)
#define LW0073_CTRL_DFP_VIDEO_COMPONENT_INFO_INDEX_SCALING      (0x0000000lw)
#define LW0073_CTRL_DFP_VIDEO_COMPONENT_INFO_INDEX_CADENCE      (0x0000000dU)
#define LW0073_CTRL_DFP_VIDEO_COMPONENT_INFO_INDEX_NOISE_REDUCE (0x0000000eU)
#define LW0073_CTRL_DFP_VIDEO_COMPONENT_INFO_INDEX_EDGE_ENHANCE (0x0000000fU)
#define LW0073_CTRL_DFP_VIDEO_COMPONENT_INFO_INDEX_OVERDRIVE    (0x00000010U)
#define LW0073_CTRL_DFP_VIDEO_COMPONENT_INFO_INDEX_SPLITSCREEN  (0x00000011U)
#define LW0073_CTRL_DFP_VIDEO_COMPONENT_INFO_INDEX_MAX          (0x12U) /* finn: Evaluated from "LW0073_CTRL_DFP_VIDEO_COMPONENT_INFO_INDEX_SPLITSCREEN + 1" */

/* video-component-state data related define */
#define LW0073_CTRL_DFP_VIDEO_COMPONENT_INFO_DATA_ILWALID       (0xdeadf00dU)

/*
 * LW0073_CTRL_CMD_DFP_GET_VIDEO_COMPONENT_INFO
 *
 * This command is used to retrieve a display's video color and post-processing
 * control information. Clients can use this command to enumerate the video
 * control data associated with the input displayId. Requests to retrieve video
 * component information use a list of one ore more
 * LW0073_CTRL_VIDEO_COMPONENT_INFO structures.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *     This parameter must specify a value between zero and the total number
 *     of subdevices within the parent device.  This parameter should be set
 *     to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the display for which the video
 *     control data is requested. If more than one displayId bit is set or
 *     the displayId is not a DFP, this call will return
 *     LW_ERR_ILWALID_ARGUMENT.
 *   videoComponentListSize
 *     This field specifies the number of entries on the caller's
 *     videoComponentList.
 *   videoComponentList
 *     This field specifies a pointer in the caller's address space
 *     to the buffer into which the video component information is to be
 *     returned. This buffer must be at least as big as videoComponentListSize
 *     multiplied by the size of the LW0073_CTRL_VIDEO_COMPONENT_INFO
 *     structure.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_CMD_DFP_GET_VIDEO_COMPONENT_INFO            (0x731146U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_DFP_GET_VIDEO_COMPONENT_INFO_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DFP_GET_VIDEO_COMPONENT_INFO_PARAMS_MESSAGE_ID (0x46U)

typedef struct LW0073_CTRL_DFP_GET_VIDEO_COMPONENT_INFO_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 videoComponentListSize;
    LW_DECLARE_ALIGNED(LwP64 videoComponentList, 8);
} LW0073_CTRL_DFP_GET_VIDEO_COMPONENT_INFO_PARAMS;

/*
 * LW0073_CTRL_CMD_DFP_SET_LWDPS_CAPS
 *
 * This command is used to set the current video playback status for use
 * by the Display Power Saving (lwDPS) feature.  The playback status is
 * used to maximize power savings by altering the DFP refresh rate used for
 * video playback.
 *
 *   displayId
 *     This parameter specifies the ID of the device for which lwDPS
 *     capabilities are being reported.
 *
 *   baseHTotal
 *     This parameter specifies the base (original) raster width specified in
 *     the EDID (active + regular blanking). This value is retrieved from the
 *     first DTD.
 *   baseVTotal
 *     This parameter specifies the base (original) raster height specified in
 *     the EDID (active + regular blanking). This value is retrieved from the
 *     first DTD.
 *   basePClk
 *     This parameter specifies the base (original) pixel clock specified in
 *     the EDID. This value is retrieved from the first DTD.
 *   maxHTotal
 *     This parameter specifies the maximum raster width specified in the EDID
 *     (active + regular blanking + blanking extension) of which this panel is
 *     capable. This can also be specified using ACPI. A value of 0 insinuates
 *     this is not supported.
 *   maxVTotal
 *     Similar to maxHTotal, this specifies the maximum raster height specified
 *     in the EDID or other means (ACPI), active + regular blanking + extension.
 *     Again, a value of 0 indicates this is not supported.
 *   bUsesVfp
 *     Vertical blanking may be specified for either front porch or (preferably)
 *     back porch. If this boolean value is true, we will attempt to use a
 *     front porch extension. Otherwise we'll use the (default) back porch
 *     extension.
 *   videoModeSupported
 *     This parameter is an enum indicating supported video modes reported
 *     in the EDID of the given panel.
 *   HTotalMSOverride
 *     This parameter is the hTotal value passed by MS to be overridden
 *     if requested
 *   VTotalMSOverride
 *     This parameter is the vTotal value passed by MS to be overridden
 *     if requested
 *   bIsCallFromMS
 *     This parameter indicates if this is a direct request from MS to fall
 *     down to lower refresh rate
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0073_CTRL_CMD_DFP_SET_LWDPS_CAPS (0x731147U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DFP_SET_LWDPS_CAPS_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DFP_SET_LWDPS_CAPS_PARAMS_MESSAGE_ID (0x47U)

typedef struct LW0073_CTRL_CMD_DFP_SET_LWDPS_CAPS_PARAMS {
    LwU32  displayId;
    LwU32  baseHTotal;
    LwU32  baseVTotal;
    LwU32  basePClk;
    LwU32  maxHTotal;
    LwU32  maxVTotal;
    LwBool bUsesVfp;
    LwU32  videoModeSupported;
    LwU32  hTotalMSOverride;
    LwU32  vTotalMSOverride;
    LwBool bIsCallFromMS;
} LW0073_CTRL_CMD_DFP_SET_LWDPS_CAPS_PARAMS;

/* valid mode flags */
#define LW0073_CTRL_DFP_LWDPS_CAPS_VIDEO_MODE_SUPPORTED_NONE      (0x00000000U)
#define LW0073_CTRL_DFP_LWDPS_CAPS_VIDEO_MODE_SUPPORTED_48HZ      (0x00000001U)
#define LW0073_CTRL_DFP_LWDPS_CAPS_VIDEO_MODE_SUPPORTED_48HZ_50HZ (0x00000002U)

/*
 * LW0073_CTRL_CMD_DFP_GET_MAX_PACKET_PER_HBLANK
 *
 * This command can be used to determine information about the maximum
 * number of packets for HDMI or DP that will be sent during
 * the horizontal blanking period.  This is partilwlarly useful for
 * determining the audio restrictions by obtaining the # of audio
 * packets that will be sent.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the display for which the
 *     maximum number of packets per hblank information will be returned.
 *     If more than one displayId bit is set or  the displayId is not a DFP,
 *     this call will return LW_ERR_ILWALID_ARGUMENT.
 *   maxPacketPerHBlank
 *     This parameter provide the maximum number of 32-pixel packets that
 *     will fit in the horizontal blanking interval.
*/
#define LW0073_CTRL_CMD_DFP_GET_MAX_PACKET_PER_HBLANK             (0x731148U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_DFP_GET_MAX_PACKET_PER_HBLANK_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DFP_GET_MAX_PACKET_PER_HBLANK_PARAMS_MESSAGE_ID (0x48U)

typedef struct LW0073_CTRL_DFP_GET_MAX_PACKET_PER_HBLANK_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 maxPacketPerHBlank;
} LW0073_CTRL_DFP_GET_MAX_PACKET_PER_HBLANK_PARAMS;

/*
 * LW0073_CTRL_CMD_DFP_SET_SPREAD_SPECTRUM
 *
 * This command is used to control spread spectrum.  The specified
 * setting will be used on subsequent modesets for the
 * associated display device.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the display for which spread
 *     spectrum is to be controlled.  If more than one displayId bit is set or
 *     the displayId is not a DFP, this call will return
 *     LW_ERR_ILWALID_ARGUMENT.
 *   enable
 *     This parameter specifies the desired spread spectrum setting:
 *       LW0073_CTRL_DFP_SET_SPREAD_SPECTRUM_ENABLE_TRUE
 *         Spread spectrum will be enabled during subsequent modesets.
 *       LW0073_CTRL_DFP_SET_SPREAD_SPECTRUM_ENABLE_FALSE
 *         Spread spectrum will be disabled during subsequent modesets.
 *
 */
#define LW0073_CTRL_CMD_DFP_SET_SPREAD_SPECTRUM (0x731149U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | 0x49" */

typedef struct LW0073_CTRL_DFP_SET_SPREAD_SPECTRUM_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 enable;
} LW0073_CTRL_DFP_SET_SPREAD_SPECTRUM_PARAMS;

/* valid enable parameter values */
#define LW0073_CTRL_DFP_SET_SPREAD_SPECTRUM_ENABLE_FALSE (0x00000000U)
#define LW0073_CTRL_DFP_SET_SPREAD_SPECTRUM_ENABLE_TRUE  (0x00000001U)

/*
 * LW0073_CTRL_CMD_DFP_GET_HYBRID_PADS_INFO
 *
 * This command is used to get hybrid pad configuration for a particular display.
 *
 *   physicalPort
 *     It will reflect hybrid pad's physical port
 *   displayId
 *     This parameter specifies the ID of the display for which user wants
 *     to know hybrid pad state.
 *   bPadLwrrMode
 *     It will contain pad's current mode
 *     i.e. LW_PMGR_HYBRID_PADCTL_MODE_AUX or
 *          LW_PMGR_HYBRID_PADCTL_MODE_I2C.
 *   bpadLwrrState
 *     It will contain pad's current power state
 *     i.e. LW_PMGR_HYBRID_SPARE_PAD_PWR_POWERDOWN or
 *          LW_PMGR_HYBRID_SPARE_PAD_PWR_POWERUP .
 *   bUsesHybridPad
 *     It will contain TRUE if display uses hybrid pad, false otherwise.
 *   numAuxPorts
 *     max number of aux ports supported.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *
 */
#define LW0073_CTRL_CMD_DFP_GET_HYBRID_PADS_INFO         (0x73114aU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_DFP_GET_HYBRID_PADS_INFO_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DFP_GET_HYBRID_PADS_INFO_PARAMS_MESSAGE_ID (0x4AU)

typedef struct LW0073_CTRL_DFP_GET_HYBRID_PADS_INFO_PARAMS {
    LwU32  physicalPort;
    LwU32  displayId;
    LwBool bPadLwrrMode;
    LwBool bpadLwrrState;
    LwBool bUsesHybridPad;
    LwU32  numAuxPorts;
} LW0073_CTRL_DFP_GET_HYBRID_PADS_INFO_PARAMS;

/*
 * LW0073_CTRL_CMD_DFP_GET_ALL_HYBRID_PADS_INFO
 *
 * This command is used to get pad configuration information for all AUX physical ports.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   numAuxPorts
 *     This is an output parameter
 *     This parameter specifies number of aux ports in current GPU subdevice.
 *   displayMask
 *     This is an output parameter.
 *     This parameter specifies the display mask associated with the port number.
 *   bPadLwrrMode
 *     This is an output parameter.
 *     It will contain pad's current mode
 *     i.e. LW_PMGR_HYBRID_PADCTL_MODE_AUX or
 *          LW_PMGR_HYBRID_PADCTL_MODE_I2C.
 *   bPadPowerState
 *     This is an output parameter.
 *     It will contain pad's current power state
 *     i.e. LW_PMGR_HYBRID_SPARE_PAD_PWR_POWERDOWN or
 *          LW_PMGR_HYBRID_SPARE_PAD_PWR_POWERUP.
 *   bUsesHybridPad
 *     This is an output parameter.
 *     It will contain TRUE if display uses hybrid pad, false otherwise.
 *   cmd
 *     This parameter is an input to this command.
 *     Here is the current defined field:
 *       LW0073_CTRL_CMD_TYPE_HYBRID_PAD
 *         The request type specifies if we are need to send physical port
 *         number or hybrid pad information based on status request:
 *           LW0073_CTRL_CMD_TYPE_HYBRID_PAD_NUM_PHYSICAL_PORT
 *             Send number of physical port.
 *           LW0073_CTRL_CMD_TYPE_HYBRID_PAD_INFO
 *             Send hybrid pad information based on physical port number.
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_CMD_DFP_GET_ALL_HYBRID_PADS_INFO (0x73114bU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_DFP_GET_ALL_HYBRID_PADS_INFO_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DFP_PHYSICAL_MAX_PORTS           8U

#define LW0073_CTRL_DFP_GET_ALL_HYBRID_PADS_INFO_PARAMS_MESSAGE_ID (0x4BU)

typedef struct LW0073_CTRL_DFP_GET_ALL_HYBRID_PADS_INFO_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 numAuxPorts;
    struct {
        LwU32  displayMask;
        LwU32  padLwrrMode;
        LwBool bPadPowerState;
        LwBool bUsesHybridPad;
    } hybridPads[LW0073_CTRL_DFP_PHYSICAL_MAX_PORTS];
} LW0073_CTRL_DFP_GET_ALL_HYBRID_PADS_INFO_PARAMS;


/*
 * LW0073_CTRL_CMD_DFP_GET_SPREAD_SPECTRUM_STATUS
 *
 * This command is used to get spread spectrum status for a display device.
 *
 * displayId
 *    Display ID for which the spread spectrum status is needed.
 * checkRMSsState
 *    Default is to check in Vbios. This flag lets this control call to check in register.
 * status
 *    Return status value.
 */

#define LW0073_CTRL_CMD_DFP_GET_SPREAD_SPECTRUM (0x73114lw) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_DFP_GET_SPREAD_SPECTRUM_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DFP_GET_SPREAD_SPECTRUM_PARAMS_MESSAGE_ID (0x4LW)

typedef struct LW0073_CTRL_DFP_GET_SPREAD_SPECTRUM_PARAMS {
    LwU32  displayId;
    LwBool enabled;
} LW0073_CTRL_DFP_GET_SPREAD_SPECTRUM_PARAMS;
/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW0073_CTRL_CMD_DFP_UPDATE_DYNAMIC_DFP_CACHE
 *
 * Update the Dynamic DFP with Bcaps read from remote display.
 * Also updates hdcpFlags, gpu hdcp capable flags in DFP.
 * If bResetDfp is true, all the flags are reset before making changes.
 *
 *   Possible status values returned are:
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 */

#define LW0073_CTRL_CMD_DFP_UPDATE_DYNAMIC_DFP_CACHE (0x73114eU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_DFP_UPDATE_DYNAMIC_DFP_CACHE_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DFP_UPDATE_DYNAMIC_DFP_CACHE_PARAMS_MESSAGE_ID (0x4EU)

typedef struct LW0073_CTRL_DFP_UPDATE_DYNAMIC_DFP_CACHE_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  headIndex;
    LwU8   bcaps;
    LwU8   bksv[5];
    LwU32  hdcpFlags;
    LwBool bHdcpCapable;
    LwBool bResetDfp;
    LwU8   updateMask;
} LW0073_CTRL_DFP_UPDATE_DYNAMIC_DFP_CACHE_PARAMS;

#define LW0073_CTRL_DFP_UPDATE_DYNAMIC_DFP_CACHE_MASK_BCAPS 0x01U
#define LW0073_CTRL_DFP_UPDATE_DYNAMIC_DFP_CACHE_MASK_BKSV  0x02U
#define LW0073_CTRL_DFP_UPDATE_DYNAMIC_DFP_CACHE_MASK_FLAGS 0x03U

/*
 * LW0073_CTRL_CMD_DFP_SET_AUDIO_ENABLE
 *
 * This command sets the audio enable state of the DFP.  When disabled,
 * no audio stream packets or audio timestamp packets will be sent.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the display for which the dfp
 *     audio should be enabled or disabled.  The display ID must be a dfp display.
 *     If the displayId is not a dfp, this call will return
 *     LW_ERR_ILWALID_ARGUMENT.
 *   enable
 *     This parameter specifies whether to enable (LW_TRUE) or disable (LW_FALSE)
 *     audio to the display.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *
 *
 */
#define LW0073_CTRL_CMD_DFP_SET_AUDIO_ENABLE                (0x731150U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_DFP_SET_AUDIO_ENABLE_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DFP_SET_AUDIO_ENABLE_PARAMS_MESSAGE_ID (0x50U)

typedef struct LW0073_CTRL_DFP_SET_AUDIO_ENABLE_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  displayId;
    LwBool enable;
} LW0073_CTRL_DFP_SET_AUDIO_ENABLE_PARAMS;


#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
*  LW0073_CTRL_CMD_DFP_SET_DSI_INFO
*
*  This command can be used to setup DSI information.
*
*   subDeviceInstance
*     This parameter specifies the subdevice instance within the
*     LW04_DISPLAY_COMMON parent device to which the operation should be
*     directed. This parameter must specify a value between zero and the
*     total number of subdevices within the parent device.  This parameter
*     should be set to zero for default behavior.
*   displayId
*     This parameter specifies the ID of the display on which the DSI
*     info will be set.  The display ID must be a DSI-capable display.
*
*  Possible status values returned are:
*   LW_OK
*   LW_ERR_ILWALID_ARGUMENT
*   LW_ERR_NOT_SUPPORTED
*/


#define LW0073_CTRL_CMD_DFP_SET_DSI_INFO (0x731151U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_DFP_SET_DSI_INFO_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DFP_SET_DSI_INFO_PARAMS_MESSAGE_ID (0x51U)

typedef struct LW0073_CTRL_DFP_SET_DSI_INFO_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
} LW0073_CTRL_DFP_SET_DSI_INFO_PARAMS;

#define LW0073_CTRL_DFP_SET_DSI_INFO_PIXEL_FORMAT_RESERVED       (0x0U)
#define LW0073_CTRL_DFP_SET_DSI_INFO_PIXEL_FORMAT_16BPP          (0x1U)
#define LW0073_CTRL_DFP_SET_DSI_INFO_PIXEL_FORMAT_18BPP_UNPACKED (0x2U)
#define LW0073_CTRL_DFP_SET_DSI_INFO_PIXEL_FORMAT_18BPP_PACKED   (0x3U)
#define LW0073_CTRL_DFP_SET_DSI_INFO_PIXEL_FORMAT_24BPP          (0x4U)

#define LW0073_CTRL_DFP_SET_DSI_INFO_TRANSMIT_MODE_NON_BURST     (0x0U)
#define LW0073_CTRL_DFP_SET_DSI_INFO_TRANSMIT_MODE_NON_BURST_WOE (0x1U)
#define LW0073_CTRL_DFP_SET_DSI_INFO_TRANSMIT_MODE_BURST         (0x2U)
#define LW0073_CTRL_DFP_SET_DSI_INFO_TRANSMIT_MODE_COMMAND       (0x3U)

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW0073_CTRL_DFP_ASSIGN_SOR_LINKCONFIG
 *
 * This enum defines default/primary/secondary sor sublinks to be configured.
 * These access modes are:
 *
 *  LW0073_CTRL_DFP_ASSIGN_SOR_FORCE_NONE
 *    Default link config
 *  LW0073_CTRL_DFP_ASSIGN_SOR_FORCE_PRIMARY_SOR_LINK
 *    Primary sor sublink to be configured
 *  LW0073_CTRL_DFP_ASSIGN_SOR_FORCE_SECONDARY_SOR_LINK
 *    Secondary sor sublink to be configured
 */
typedef enum LW0073_CTRL_DFP_ASSIGN_SOR_LINKCONFIG {
    LW0073_CTRL_DFP_ASSIGN_SOR_FORCE_NONE = 0,
    LW0073_CTRL_DFP_ASSIGN_SOR_FORCE_PRIMARY_SOR_LINK = 1,
    LW0073_CTRL_DFP_ASSIGN_SOR_FORCE_SECONDARY_SOR_LINK = 2,
} LW0073_CTRL_DFP_ASSIGN_SOR_LINKCONFIG;

/*
 * LW0073_CTRL_DFP_ASSIGN_SOR_INFO
 *
 * This structure describes info about assigned SOR
 *
 *   displayMask
 *     The displayMask for the SOR corresponding to its HW routings
 *   sorType
 *     This parameter specifies the SOR type
 *          Here are the current defined fields:
 *          LW0073_CTRL_DFP_SOR_TYPE_NONE
 *              Unallocated SOR
 *          LW0073_CTRL_DFP_SOR_TYPE_2H1OR_PRIMARY
 *              Primary SOR for 2H1OR stream
 *          LW0073_CTRL_DFP_SOR_TYPE_2H1OR_SECONDARY
 *              Secondary SOR for 2H1OR stream
 *          LW0073_CTRL_DFP_SOR_TYPE_SINGLE
 *              Default Single SOR
 * Note - sorType should only be referred to identify 2H1OR Primary and Secondary SOR
 *
 */

typedef struct LW0073_CTRL_DFP_ASSIGN_SOR_INFO {
    LwU32 displayMask;
    LwU32 sorType;
} LW0073_CTRL_DFP_ASSIGN_SOR_INFO;

#define LW0073_CTRL_DFP_SOR_TYPE_NONE            (0x00000000U)
#define LW0073_CTRL_DFP_SOR_TYPE_SINGLE          (0x00000001U)
#define LW0073_CTRL_DFP_SOR_TYPE_2H1OR_PRIMARY   (0x00000002U)
#define LW0073_CTRL_DFP_SOR_TYPE_2H1OR_SECONDARY (0x00000003U)

/*
 *  LW0073_CTRL_CMD_DFP_ASSIGN_SOR
 *
 *  This command is used by the clients to assign SOR to DFP for CROSS-BAR
 *  when the default SOR-DFP routing that comes from vbios is not considered.
 *  SOR shall be assigned to a DFP at the runtime. This call should be called
 *  before a modeset is done on any dfp display and also before LinkTraining for DP displays.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     DisplayId of the primary display for which SOR is to be assigned. However, if
 *     displayId is 0 then RM shall return the XBAR config it has stored in it's structures.
 *   sorExcludeMask
 *     sorMask of the SORs which should not be used for assignment. If this is 0,
 *     then SW is free to allocate any available SOR.
 *   slaveDisplayId
 *      displayId of the slave device in case of dualSST mode. This ctrl call will
 *      allocate SORs to both slave and the master if slaveDisplayId is set.
 *   forceSublinkConfig
 *      forces RM to configure primary or secondary sor sublink on the given diaplayId.
 *      If not set, then RM will do the default configurations.
 *   bIs2Head1Or
 *      Specifies that SOR allocation is required for 2 head 1 OR. This will allocate
 *      2 SOR for same displayId - one Master and one Slave. Slave SOR would be disconnected
 *      from any padlink and get feedback clock from Master SOR's padlink.
 *   sorAssignList[LW0073_CTRL_CMD_DFP_ASSIGN_SOR_MAX_SORS]
 *       returns the displayMask for all SORs corresponding to their HW routings.
 *   sorAssignListWithTag[LW0073_CTRL_CMD_DFP_ASSIGN_SOR_MAX_SORS]
 *       returns the displayMask for all SORs corresponding to their HW routings along with
 *       SOR type to identify 2H1OR Primary and Secondary SORs. SOR type would be identified by
 *       LW0073_CTRL_DFP_SOR_TYPE. sorAssignList would look as below -
 *       sorAssignListWithTag[] = { DisplayMask, SOR Type
 *                                 {0x100,       SECONDARY_SOR}
 *                                 {0x200,       SINGLE_SOR}
 *                                 {0x100,       PRIMARY_SOR}
 *                                 {0,           NONE}}
 *                                }
 *       Here, for display id = 0x100, SOR2 is Primary and SOR0 is Secondary.
 *       Note - sorAssignList parameter would be removed after Bug 200749158 is resolved
 *   reservedSorMask
 *       returns the sorMask reserved for the internal panels.
 *   flags
 *       Other detail settings.
 *           _AUDIO_OPTIMAL: Client requests trying to get audio SOR if possible.
 *                           If there's no audio capable SOR and OD is HDMI/DP,
 *                           RM will fail the control call.
 *           _AUDIO_DEFAULT: RM does not check audio-capability of SOR.
 *
 *           _ACTIVE_SOR_NOT_AUDIO_CAPABLE_YES : RM returns Active SOR which is not Audio capable.
 *           _ACTIVE_SOR_NOT_AUDIO_CAPABLE_NO  : RM is not returning 'Active non-audio capable SOR'.
 *
 *  Possible status values returned are: 
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */


#define LW0073_CTRL_CMD_DFP_ASSIGN_SOR           (0x731152U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_DFP_ASSIGN_SOR_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DFP_ASSIGN_SOR_MAX_SORS  4U

#define LW0073_CTRL_DFP_ASSIGN_SOR_PARAMS_MESSAGE_ID (0x52U)

typedef struct LW0073_CTRL_DFP_ASSIGN_SOR_PARAMS {
    LwU32                                 subDeviceInstance;
    LwU32                                 displayId;
    LwU8                                  sorExcludeMask;
    LwU32                                 slaveDisplayId;
    LW0073_CTRL_DFP_ASSIGN_SOR_LINKCONFIG forceSublinkConfig;
    LwBool                                bIs2Head1Or;
    LwU32                                 sorAssignList[LW0073_CTRL_CMD_DFP_ASSIGN_SOR_MAX_SORS];
    LW0073_CTRL_DFP_ASSIGN_SOR_INFO       sorAssignListWithTag[LW0073_CTRL_CMD_DFP_ASSIGN_SOR_MAX_SORS];
    LwU8                                  reservedSorMask;
    LwU32                                 flags;
} LW0073_CTRL_DFP_ASSIGN_SOR_PARAMS;

#define LW0073_CTRL_DFP_ASSIGN_SOR_FLAGS_AUDIO                                      0:0
#define LW0073_CTRL_DFP_ASSIGN_SOR_FLAGS_AUDIO_OPTIMAL                    (0x00000001U)
#define LW0073_CTRL_DFP_ASSIGN_SOR_FLAGS_AUDIO_DEFAULT                    (0x00000000U)
#define LW0073_CTRL_DFP_ASSIGN_SOR_FLAGS_ACTIVE_SOR_NOT_AUDIO_CAPABLE               1:1
#define LW0073_CTRL_DFP_ASSIGN_SOR_FLAGS_ACTIVE_SOR_NOT_AUDIO_CAPABLE_NO  (0x00000000U)
#define LW0073_CTRL_DFP_ASSIGN_SOR_FLAGS_ACTIVE_SOR_NOT_AUDIO_CAPABLE_YES (0x00000001U)

/*
*  LW0073_CTRL_CMD_DFP_GET_PADLINK_MASK
*
*  This command will only be used by chipTB tests to get the padlinks corresponding
*  to the given displayId. RM gets this information from vbios. This control call is
*  only for verif purpose.
*
*   subDeviceInstance
*     This parameter specifies the subdevice instance within the
*     LW04_DISPLAY_COMMON parent device to which the operation should be
*     directed. This parameter must specify a value between zero and the
*     total number of subdevices within the parent device.  This parameter
*     should be set to zero for default behavior.
*   displayId
*     DisplayId of the display for which the client needs analog link Mask
*   padlinkMask
*     analogLinkMask for the given displayId. This value returned is 0xffffffff if
*     the given displayId is invalid else RM returns the corresponding padlinkMask.
*   LW_OK
*   LW_ERR_ILWALID_ARGUMENT
*   LW_ERR_NOT_SUPPORTED
*/


#define LW0073_CTRL_CMD_DFP_GET_PADLINK_MASK                              (0x731153U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_DFP_GET_PADLINK_MASK_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DFP_GET_PADLINK_MASK_PARAMS_MESSAGE_ID (0x53U)

typedef struct LW0073_CTRL_DFP_GET_PADLINK_MASK_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 padlinkMask;
} LW0073_CTRL_DFP_GET_PADLINK_MASK_PARAMS;


#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

/*
 * LW0073_CTRL_CMD_DFP_LCD_GPIO_FUNC_TYPE
 *      This enum defines the functions that are supported for which a
 *      corresponding GPIO pin number could be retrieved
 *      Values copied from objgpio.h GPIO_FUNC_TYPE_LCD_*. Please keep the
 *      values in sync between the 2 files
 */

typedef enum LW0073_CTRL_CMD_DFP_LCD_GPIO_FUNC_TYPE {
    // GPIO types of LCD GPIO functions common to all internal panels
    LW0073_CTRL_CMD_DFP_LCD_GPIO_FUNC_TYPE_LCD_BACKLIGHT = 268435456,
    LW0073_CTRL_CMD_DFP_LCD_GPIO_FUNC_TYPE_LCD_POWER = 285212672,
    LW0073_CTRL_CMD_DFP_LCD_GPIO_FUNC_TYPE_LCD_POWER_OK = 301989888,
    LW0073_CTRL_CMD_DFP_LCD_GPIO_FUNC_TYPE_LCD_SELF_TEST = 318767104,
    LW0073_CTRL_CMD_DFP_LCD_GPIO_FUNC_TYPE_LCD_LAMP_STATUS = 335544320,
    LW0073_CTRL_CMD_DFP_LCD_GPIO_FUNC_TYPE_LCD_BRIGHTNESS = 352321536,
} LW0073_CTRL_CMD_DFP_LCD_GPIO_FUNC_TYPE;

/*
 * LW0073_CTRL_CMD_DFP_GET_LCD_GPIO_PIN_NUM
 *
 * This command can be used to get the GPIO pin number that corresponds to one
 * of the LCD functions
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the dfp display.
 *     If more than one displayId bit is set or the displayId is not a dfp,
 *     this call will return LWOS_STATUS_ERROR_ILWALID_ARGUMENT.
 *   funcType
 *      The LDC function for which the GPIO pin number is needed
 *   lcdGpioPinNum
 *     The GPIO pin number that corresponds to the LCD function.
 *
*/
#define LW0073_CTRL_CMD_DFP_GET_LCD_GPIO_PIN_NUM (0x731154U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_DFP_GET_LCD_GPIO_PIN_NUM_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DFP_GET_LCD_GPIO_PIN_NUM_PARAMS_MESSAGE_ID (0x54U)

typedef struct LW0073_CTRL_DFP_GET_LCD_GPIO_PIN_NUM_PARAMS {
    LwU32                                  subDeviceInstance;
    LwU32                                  displayId;
    LW0073_CTRL_CMD_DFP_LCD_GPIO_FUNC_TYPE funcType;
    LwU32                                  lcdGpioPinNum;
} LW0073_CTRL_DFP_GET_LCD_GPIO_PIN_NUM_PARAMS;

/*
 *  LW0073_CTRL_CMD_DFP_CLEAR_SOR_XBAR
 *
 *  This command is used by the mods client to clear SOR XBAR
 *  for the given displayMask.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayMask
 *     DisplayMask of the displays whose SOR XBAR state has to be cleared.
 *   sorAssignList[LW0073_CTRL_CMD_DFP_ASSIGN_SOR_MAX_SORS]
 *       returns the displayMask for all SORs corresponding to their HW routings.
 *   sorAssignListWithTag[LW0073_CTRL_CMD_DFP_ASSIGN_SOR_MAX_SORS]
 *       returns the displayMask for all SORs corresponding to their HW routings along with
 *       SOR type to identify 2H1OR Primary and Secondary SORs. SOR type would be identified by
 *       LW0073_CTRL_DFP_SOR_TYPE. sorAssignList would look as below -
 *       sorAssignListWithTag[] = { DisplayMask, SOR Type
 *                                 {0x100,       SECONDARY_SOR}
 *                                 {0x200,       SINGLE_SOR}
 *                                 {0x100,       PRIMARY_SOR}
 *                                 {0,           NONE}}
 *                                }
 *       Here, for display id = 0x100, SOR2 is Primary and SOR0 is Secondary.
 *       Note - sorAssignList parameter would be removed after Bug 200749158 is resolved
 *  Possible status values returned are:
 *   LWOS_STATUS_SUCCESS
 *   LWOS_STATUS_ERROR_ILWALID_ARGUMENT
 *   LWOS_STATUS_ERROR_NOT_SUPPORTED
 */


#define LW0073_CTRL_CMD_DFP_CLEAR_SOR_XBAR (0x731155U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DFP_CLEAR_SOR_XBAR_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DFP_CLEAR_SOR_XBAR_PARAMS_MESSAGE_ID (0x55U)

typedef struct LW0073_CTRL_CMD_DFP_CLEAR_SOR_XBAR_PARAMS {
    LwU32                           subDeviceInstance;
    LwU32                           displayMask;
    LwU32                           sorAssignList[LW0073_CTRL_CMD_DFP_ASSIGN_SOR_MAX_SORS];
    LW0073_CTRL_DFP_ASSIGN_SOR_INFO sorAssignListWithTag[LW0073_CTRL_CMD_DFP_ASSIGN_SOR_MAX_SORS];
} LW0073_CTRL_CMD_DFP_CLEAR_SOR_XBAR_PARAMS;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 *  LW0073_CTRL_CMD_DFP_CONFIG_TWO_HEAD_ONE_OR
 *
 *  This command is used for configuration of 2 head 1 OR.
 *
 *   subDeviceInstance
 *      This parameter specifies the subdevice instance within the
 *      LW04_DISPLAY_COMMON parent device to which the operation should be
 *      directed. This parameter must specify a value between zero and the
 *      total number of subdevices within the parent device.  This parameter
 *      should be set to zero for default behavior.
 *   displayId
 *      Display Id of the panel for which Two Head One OR is going to be used
 *   bEnable
 *      Enable/Disable 2 Head 1 OR
 *   masterSorIdx
 *      Master SOR Index which will send pixels to panel
 *   slaveSorIdx
 *      Slave SOR Index which will take feedback clock from Master SOR's
 *      padlink
 *  Possible status values returned are:
 *   LWOS_STATUS_SUCCESS
 *   LWOS_STATUS_ERROR_ILWALID_ARGUMENT
 *   LWOS_STATUS_ERROR_NOT_SUPPORTED
 */


#define LW0073_CTRL_CMD_DFP_CONFIG_TWO_HEAD_ONE_OR (0x731156U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_DFP_CONFIG_TWO_HEAD_ONE_OR_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DFP_CONFIG_TWO_HEAD_ONE_OR_PARAMS_MESSAGE_ID (0x56U)

typedef struct LW0073_CTRL_DFP_CONFIG_TWO_HEAD_ONE_OR_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  displayId;
    LwBool bEnable;
    LwU32  masterSorIdx;
    LwU32  slaveSorIdx;
} LW0073_CTRL_DFP_CONFIG_TWO_HEAD_ONE_OR_PARAMS;

/*
 *  LW0073_CTRL_CMD_DFP_DSC_CRC_CONTROL
 *
 *  This command is used to enable/disable CRC on the GPU or query the registers
 *  related to it
 *
 *   subDeviceInstance
 *      This parameter specifies the subdevice instance within the
 *      LW04_DISPLAY_COMMON parent device to which the operation should be
 *      directed. This parameter must specify a value between zero and the
 *      total number of subdevices within the parent device.  This parameter
 *      should be set to zero for default behavior.
 *   headIndex
 *      index of the head
 *   cmd
 *      specifying if setup or querying is done
 *   bEnable
 *      enable or disable CRC on the GPU
 *   gpuCrc0
 *      0-indexed CRC register of the GPU
 *   gpuCrc1
 *      1-indexed CRC register of the GPU
 *   gpuCrc0
 *      2-indexed CRC register of the GPU
 *  Possible status values returned are:
 *   LWOS_STATUS_SUCCESS
 *   LWOS_STATUS_ERROR_NOT_SUPPORTED
 */


#define LW0073_CTRL_CMD_DFP_DSC_CRC_CONTROL (0x731157U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_DFP_DSC_CRC_CONTROL_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DFP_DSC_CRC_CONTROL_PARAMS_MESSAGE_ID (0x57U)

typedef struct LW0073_CTRL_DFP_DSC_CRC_CONTROL_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  headIndex;
    LwU32  cmd;
    LwBool bEnable;
    LwU16  gpuCrc0;
    LwU16  gpuCrc1;
    LwU16  gpuCrc2;
} LW0073_CTRL_DFP_DSC_CRC_CONTROL_PARAMS;

#define LW0073_CTRL_DP_CRC_CONTROL_CMD                                     0:0
#define LW0073_CTRL_DP_CRC_CONTROL_CMD_SETUP (0x00000000U)
#define LW0073_CTRL_DP_CRC_CONTROL_CMD_QUERY (0x00000001U)

/*
 * LW0073_CTRL_CMD_DFP_INIT_MUX_DATA
 *
 * This control call is used to configure the display MUX related data
 * for the given display device. Clients to RM are expected to call this
 * control call to initialize the data related to MUX before any MUX related
 * operations such mux switch or PSR entry/ exit are performed.
 *
 *   subDeviceInstance (in)
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *   displayId (in)
 *     ID of the display device for which the mux state has to be initialized
 *   manfId (in)
 *     Specifies the manufacturer ID of panel obtained from the EDID. This
 *     parameter is expected to be non-zero only in case of internal panel.
 *   productId (in)
 *     Specifies the product ID of panel obtained from the EDID. This
 *     parameter is expected to be non-zero only in case of internal panel.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW0073_CTRL_CMD_DFP_INIT_MUX_DATA    (0x731158U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DFP_INIT_MUX_DATA_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DFP_INIT_MUX_DATA_PARAMS_MESSAGE_ID (0x58U)

typedef struct LW0073_CTRL_CMD_DFP_INIT_MUX_DATA_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU16 manfId;
    LwU16 productId;
} LW0073_CTRL_CMD_DFP_INIT_MUX_DATA_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW0073_CTRL_CMD_DFP_SIDEBAND_PSR_CTRL
 *
 * This command is used to control PSR related functionalities using
 * i2c side band communications. This is lwrrently used only in case
 * of dynamic display mux feature.
 *
 *   subDeviceInstance (in)
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *   displayId (in)
 *     ID of the display device on which PSR tasks have to be performed
 *   cmd (in)
 *     Flags indicating the action to be performed. Here are
 *     the possible fields:
 *       LW0073_CTRL_DFP_SB_PSR_CTRL_CMD_SR_QUERY_CAPS
 *         Used to query PSR capabilities
 *       LW0073_CTRL_DFP_SB_PSR_CTRL_CMD_SR_ENTER_BLOCKING
 *         Used to enter PSR using sideband i2c transaction
 *         Wait until SR is entered successfully before returning.
 *       LW0073_CTRL_DFP_SB_PSR_CTRL_CMD_SR_ENTER_NON_BLOCKING
 *         Used to enter PSR using sideband i2c transaction.
 *         Return immediately after SR entry sequence is exelwted.
 *       LW0073_CTRL_DFP_SB_PSR_CTRL_CMD_SR_EXIT_BLOCKING
 *         Used to exit PSR using sideband i2c transaction.
 *         Wait until SR is exited successfully before returning.
 *       LW0073_CTRL_DFP_SB_PSR_CTRL_CMD_SR_EXIT_NON_BLOCKING
 *         Used to exit PSR using sideband i2c transaction.
 *         Return immediately after SR exit sequence is exelwted.
 *       LW0073_CTRL_DFP_SB_PSR_CTRL_CMD_SR_QUERY_STATUS
 *         Used to query the current SR status.
 *   srCaps (out)
 *     Returns SR capabilities of the panel. This will be populated
 *     when the caller requests LW0073_CTRL_DFP_SB_PSR_CTRL_CMD_QUERY_SR_CAPS.
 *     Here are the possible fields:
 *       LW0073_CTRL_DFP_SB_PSR_CTRL_CAPS_SR_SB_SUPPORT_YES
 *         Set when PSR entry and exit is supported via sideband
 *         I2C communication
 *       LW0073_CTRL_DFP_SB_PSR_CTRL_CAPS_SR_SB_SUPPORT_NO
 *         Set when PSR entry and exit is not supported via sideband
 *         I2C communication
 *   srStatus (out)
 *     Status indicating if SR was entered or exited successfully. This is populated
 *     when the caller requests LW0073_CTRL_DFP_SB_PSR_CTRL_CMD_QUERY_SR_STATUS.
 *     Here are the possible fields:
 *       LW0073_CTRL_DFP_SB_PSR_CTRL_STAT_SR_ENTERED_YES
 *         Set when panel is in self refresh. The panel is expected to be in SR as
 *         as long as the transition from PSR_INACTIVE to PSR_ACTIVE state is
 *         complete. If the panel is still transitioning from PSR_INACTIVE to
 *         PSR_ACTIVE, LW0073_CTRL_DFP_SB_PSR_CTRL_STAT_SR_ENTERED_NO is set.
 *       LW0073_CTRL_DFP_SB_PSR_CTRL_STAT_SR_ENTERED_NO
 *         Set when panel is out of self refresh
 *   srExitTransitionToInactiveLatencyMs (out)
 *     psr exit latency stats in milli-seconds, from state 2 (SR active) to state 4 (transition to inactive)
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_TIMEOUT
 */

#define LW0073_CTRL_CMD_DFP_SIDEBAND_PSR_CTRL (0x731159U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DFP_SIDEBAND_PSR_CTRL_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DFP_SIDEBAND_PSR_CTRL_PARAMS_MESSAGE_ID (0x59U)

typedef struct LW0073_CTRL_CMD_DFP_SIDEBAND_PSR_CTRL_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 cmd;
    LwU32 srCaps;
    LwU32 srStatus;
    LwU32 flags;
    LwU32 srExitTransitionToInactiveMs;
} LW0073_CTRL_CMD_DFP_SIDEBAND_PSR_CTRL_PARAMS;

/* valid commands */
#define LW0073_CTRL_DFP_SB_PSR_CTRL_CMD_SR_QUERY_CAPS         0x00000000U
#define LW0073_CTRL_DFP_SB_PSR_CTRL_CMD_SR_ENTER_BLOCKING     0x00000001U
#define LW0073_CTRL_DFP_SB_PSR_CTRL_CMD_SR_ENTER_NON_BLOCKING 0x00000002U
#define LW0073_CTRL_DFP_SB_PSR_CTRL_CMD_SR_EXIT_BLOCKING      0x00000003U
#define LW0073_CTRL_DFP_SB_PSR_CTRL_CMD_SR_EXIT_NON_BLOCKING  0x00000004U
#define LW0073_CTRL_DFP_SB_PSR_CTRL_CMD_SR_QUERY_STATUS       0x00000005U

/* valid capability flags */
#define LW0073_CTRL_DFP_SB_PSR_CTRL_CAPS_SR_SB_SUPPORT          0:0
#define LW0073_CTRL_DFP_SB_PSR_CTRL_CAPS_SR_SB_SUPPORT_NO     0x00000000U
#define LW0073_CTRL_DFP_SB_PSR_CTRL_CAPS_SR_SB_SUPPORT_YES    0x00000001U

/* valid status flags */
#define LW0073_CTRL_DFP_SB_PSR_CTRL_STAT_SR_ENTERED             0:0
#define LW0073_CTRL_DFP_SB_PSR_CTRL_STAT_SR_ENTERED_NO        0x00000000U
#define LW0073_CTRL_DFP_SB_PSR_CTRL_STAT_SR_ENTERED_YES       0x00000001U

/* special flags */
#define LW0073_CTRL_DFP_SB_PSR_CTRL_IN_DISCRETE_MODE             0:0
#define LW0073_CTRL_DFP_SB_PSR_CTRL_IN_DISCRETE_MODE_NO       0x00000000U
#define LW0073_CTRL_DFP_SB_PSR_CTRL_IN_DISCRETE_MODE_YES      0x00000001U

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW0073_CTRL_CMD_DFP_SWITCH_DISP_MUX
 *
 * This command is used to switch the dynamic display mux between
 * integrated GPU and discrete GPU.
 *
 *   subDeviceInstance (in)
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *   displayId (in)
 *     ID of the display device for which the display MUX has to be switched
 *   flags (in)
 *     Flags indicating the action to be performed. Here are the possible
 *     valid values-
 *       LW0073_CTRL_DFP_DISP_MUX_SWITCH_IGPU_TO_DGPU
 *         When set mux is switched from integrated to discrete GPU.
 *       LW0073_CTRL_DFP_DISP_MUX_SWITCH_DGPU_TO_IGPU
 *         When set mux is switched from discrete to integrated GPU.
 *       LW0073_CTRL_DFP_DISP_MUX_SWITCH_SKIP_SIDEBAND_ACCESS
 *         Set to true for PSR panels as we skip sideband access.
 *   auxSettleDelay (in)
 *     Time, in milliseconds, necessary for AUX channel to settle and become
 *     accessible after a mux switch. Set to zero to use the default delay.
 *   muxSwitchLatencyMs (out)
 *     mux switch latency stats in milli-seconds
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW0073_CTRL_CMD_DFP_SWITCH_DISP_MUX                   (0x731160U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DFP_SWITCH_DISP_MUX_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DFP_SWITCH_DISP_MUX_PARAMS_MESSAGE_ID (0x60U)

typedef struct LW0073_CTRL_CMD_DFP_SWITCH_DISP_MUX_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 flags;
    LwU32 auxSettleDelay;
    LwU32 muxSwitchLatencyMs;
} LW0073_CTRL_CMD_DFP_SWITCH_DISP_MUX_PARAMS;

/* valid flags*/
#define LW0073_CTRL_DFP_DISP_MUX_SWITCH                            0:0
#define LW0073_CTRL_DFP_DISP_MUX_SWITCH_IGPU_TO_DGPU               0x00000000
#define LW0073_CTRL_DFP_DISP_MUX_SWITCH_DGPU_TO_IGPU               0x00000001
#define LW0073_CTRL_DFP_DISP_MUX_SWITCH_SKIP_SIDEBAND_ACCESS       1:1
#define LW0073_CTRL_DFP_DISP_MUX_SWITCH_SKIP_SIDEBAND_ACCESS_YES   0x00000001
#define LW0073_CTRL_DFP_DISP_MUX_SWITCH_SKIP_SIDEBAND_ACCESS_NO    0x00000000

/*
 * LW0073_CTRL_CMD_DFP_RUN_PRE_DISP_MUX_OPERATIONS
 *
 * This command is used to perform all the operations that need to be
 * performed before a mux switch is started.
 *
 *   subDeviceInstance (in)
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *   displayId (in)
 *     ID of the display device for which the pre mux switch operations have
 *     to be performed.
 *   flags (in)
 *     Flags indicating the action to be performed. Here are the possible
 *     valid values -
 *       LW0073_CTRL_DFP_DISP_MUX_FLAGS_SWITCH_TYPE_IGPU_TO_DGPU
 *         Indicates a switch from i to d is initiated
 *       LW0073_CTRL_DFP_DISP_MUX_FLAGS_SWITCH_TYPE_DGPU_TO_IGPU
 *         Indicates a switch from d to i is initiated
 *       LW0073_CTRL_DFP_DISP_MUX_FLAGS_SR_ENTER_SKIP_NO
 *         When set RM will execute the PSR enter sequence. By default RM will
 *         not skip SR enter sequence
 *       LW0073_CTRL_DFP_DISP_MUX_FLAGS_SR_ENTER_SKIP_YES
 *         When set RM will skip the PSR enter sequence
 *   iGpuBrightness (in)
 *     iGPU brightness value (scale 0~100) before switching mux from I2D.
 *     This is used to match brightness after switching mux to dGPU
 *   preOpsLatencyMs (out)
 *     premux switch operations latency stats in milli-seconds. This includes -
 *      - disabling SOR sequencer and enable BL GPIO control
 *      - toggling LCD VDD, BL EN and PWM MUX GPIOs
 *      - PSR entry, if not skipped
 *   psrEntryLatencyMs (out)
 *     psr entry latency stats in milli-seconds
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW0073_CTRL_CMD_DFP_RUN_PRE_DISP_MUX_OPERATIONS (0x731161U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DFP_RUN_PRE_DISP_MUX_OPERATIONS_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DFP_RUN_PRE_DISP_MUX_OPERATIONS_PARAMS_MESSAGE_ID (0x61U)

typedef struct LW0073_CTRL_CMD_DFP_RUN_PRE_DISP_MUX_OPERATIONS_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 flags;
    LwU32 iGpuBrightness;
    LwU32 preOpsLatencyMs;
    LwU32 psrEntryLatencyMs;
} LW0073_CTRL_CMD_DFP_RUN_PRE_DISP_MUX_OPERATIONS_PARAMS;

/* valid flags*/
#define LW0073_CTRL_DFP_DISP_MUX_FLAGS_SWITCH_TYPE                   0:0
#define LW0073_CTRL_DFP_DISP_MUX_FLAGS_SWITCH_TYPE_IGPU_TO_DGPU             0x00000000U
#define LW0073_CTRL_DFP_DISP_MUX_FLAGS_SWITCH_TYPE_DGPU_TO_IGPU             0x00000001U
#define LW0073_CTRL_DFP_DISP_MUX_FLAGS_SR_ENTER_SKIP                 1:1
#define LW0073_CTRL_DFP_DISP_MUX_FLAGS_SR_ENTER_SKIP_NO                     0x00000000U
#define LW0073_CTRL_DFP_DISP_MUX_FLAGS_SR_ENTER_SKIP_YES                    0x00000001U
#define LW0073_CTRL_DFP_DISP_MUX_FLAGS_MUX_SWITCH_IGPU_POWER_TIMING  2:2
#define LW0073_CTRL_DFP_DISP_MUX_FLAGS_MUX_SWITCH_IGPU_POWER_TIMING_KNOWN   0x00000000
#define LW0073_CTRL_DFP_DISP_MUX_FLAGS_MUX_SWITCH_IGPU_POWER_TIMING_UNKNOWN 0x00000001

#define LW0073_CTRL_DISP_MUX_BACKLIGHT_BRIGHTNESS_MIN                       0U
#define LW0073_CTRL_DISP_MUX_BACKLIGHT_BRIGHTNESS_MAX                       100U

/*
 * LW0073_CTRL_CMD_DFP_RUN_POST_DISP_MUX_OPERATIONS
 *
 * This command is used to perform all the operations that need to be
 * performed after a successful mux switch is completed.
 *
 *   subDeviceInstance (in)
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *   displayId (in)
 *     ID of the display device for which the post mux switch operations have
 *     to be performed.
 *   flags (in)
 *     Flags indicating the action to be performed. Here are the possible
 *     valid values -
 *       LW0073_CTRL_DFP_DISP_MUX_FLAGS_SWITCH_TYPE_IGPU_TO_DGPU
 *         Indicates a switch from i to d is initiated
 *       LW0073_CTRL_DFP_DISP_MUX_FLAGS_SWITCH_TYPE_DGPU_TO_IGPU
 *         Indicates a switch from d to i is initiated
 *       LW0073_CTRL_DFP_DISP_MUX_FLAGS_SR_EXIT_SKIP_NO
 *         When set RM will execute the PSR exit sequence. By default RM will
 *         not skip SR exit sequence
 *       LW0073_CTRL_DFP_DISP_MUX_FLAGS_SR_EXIT_SKIP_YES
 *         When set RM will skip the PSR exit sequence
 *       LW0073_CTRL_DFP_DISP_MUX_FLAGS_MUX_SWITCH_IGPU_POWER_TIMING_KNOWN
 *         Indicates mux switches where we know when igpu powers up
 *       LW0073_CTRL_DFP_DISP_MUX_FLAGS_MUX_SWITCH_IGPU_POWER_TIMING_UNKNOWN
 *         Indicates mux switches where we don't know when igpu powers up
 *   postOpsLatencyMs (out)
 *     postmux switch operations latency stats in milli-seconds. This includes -
  *     - restoring SOR sequencer and BL GPIO control
 *      - toggling LCD VDD, BL EN and PWM MUX GPIOs
 *      - PSR exit, if not skipped
 *   psrExitLatencyMs (out)
 *     psr exit latency stats in milli-seconds
 *   psrExitTransitionToInactiveLatencyMs (out)
 *     psr exit latency stats in milli-seconds, from state 2 (SR active) to state 4 (transition to inactive)
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_TIMEOUT in case of SR exit failure
 */

#define LW0073_CTRL_CMD_DFP_RUN_POST_DISP_MUX_OPERATIONS                    (0x731162U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DFP_RUN_POST_DISP_MUX_OPERATIONS_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DFP_RUN_POST_DISP_MUX_OPERATIONS_PARAMS_MESSAGE_ID (0x62U)

typedef struct LW0073_CTRL_CMD_DFP_RUN_POST_DISP_MUX_OPERATIONS_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 flags;
    LwU32 postOpsLatencyMs;
    LwU32 psrExitLatencyMs;
    LwU32 psrExitTransitionToInactiveLatencyMs;
} LW0073_CTRL_CMD_DFP_RUN_POST_DISP_MUX_OPERATIONS_PARAMS;

/* valid flags*/
#define LW0073_CTRL_DFP_DISP_MUX_FLAGS_SWITCH_TYPE                   0:0
#define LW0073_CTRL_DFP_DISP_MUX_FLAGS_SR_EXIT_SKIP                  1:1
#define LW0073_CTRL_DFP_DISP_MUX_FLAGS_SR_EXIT_SKIP_NO  0x00000000U
#define LW0073_CTRL_DFP_DISP_MUX_FLAGS_SR_EXIT_SKIP_YES 0x00000001U

/*
 * LW0073_CTRL_CMD_DFP_GET_DISP_MUX_STATUS
 *
 * This command is used to query the display mux status for the given
 * display device
 *
 *   subDeviceInstance (in)
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *   displayId (in)
 *     ID of the display device for which the post mux switch operations have
 *     to be performed.
 *   muxStatus (out)
 *     status indicating the current state of the mux.
 *     valid values -
 *       LW0073_CTRL_DFP_DISP_MUX_STATE_INTEGRATED_GPU
 *         Indicates that the MUX is lwrrently switched to integrated GPU.
 *       LW0073_CTRL_DFP_DISP_MUX_STATE_DISCRETE_GPU
 *         Indicates that the MUX is lwrrently switched to discrete GPU.
 *       LW0073_CTRL_DFP_DISP_MUX_MODE_DISCRETE_ONLY
 *         Indicates that the MUX mode is set to discrete mode, where all displays
 *         are driven by discrete GPU.
 *       LW0073_CTRL_DFP_DISP_MUX_MODE_INTEGRATED_ONLY
 *         Indicates that the MUX mode is set to integrated mode, where all
 *         displays are driven by Integrated GPU.
 *       LW0073_CTRL_DFP_DISP_MUX_MODE_HYBRID
 *         Indicates that the MUX mode is set to hybrid, where internal panel is
 *         driven by integrated GPU, while external displays might be driven by
 *         discrete GPU.
 *       LW0073_CTRL_DFP_DISP_MUX_MODE_DYNAMIC
 *         Indicates that the MUX mode is dynamic. It is only in this mode, the
 *         display MUX can be toggled between discrete and hybrid dynamically.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW0073_CTRL_CMD_DFP_GET_DISP_MUX_STATUS         (0x731163U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DFP_GET_DISP_MUX_STATUS_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DFP_GET_DISP_MUX_STATUS_PARAMS_MESSAGE_ID (0x63U)

typedef struct LW0073_CTRL_CMD_DFP_GET_DISP_MUX_STATUS_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 muxStatus;
} LW0073_CTRL_CMD_DFP_GET_DISP_MUX_STATUS_PARAMS;

/* valid flags */
#define LW0073_CTRL_DFP_DISP_MUX_STATE                        1:0
#define LW0073_CTRL_DFP_DISP_MUX_STATE_ILWALID                  0x00000000U
#define LW0073_CTRL_DFP_DISP_MUX_STATE_INTEGRATED_GPU           0x00000001U
#define LW0073_CTRL_DFP_DISP_MUX_STATE_DISCRETE_GPU             0x00000002U
#define LW0073_CTRL_DFP_DISP_MUX_MODE                         4:2
#define LW0073_CTRL_DFP_DISP_MUX_MODE_ILWALID                   0x00000000U
#define LW0073_CTRL_DFP_DISP_MUX_MODE_INTEGRATED_ONLY           0x00000001U
#define LW0073_CTRL_DFP_DISP_MUX_MODE_DISCRETE_ONLY             0x00000002U
#define LW0073_CTRL_DFP_DISP_MUX_MODE_HYBRID                    0x00000003U
#define LW0073_CTRL_DFP_DISP_MUX_MODE_DYNAMIC                   0x00000004U

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW0073_CTRL_CMD_DFP_SIDEBAND_I2C_CTRL
 *
 * This command is used to perform sideband I2C communications with TCON
 * that support the feature. This is lwrrently used only in case of 
 * dynamic display mux feature.
 *
 *   subDeviceInstance (in)
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *   displayId (in)
 *     ID of the display device on which sideband I2C tasks have to be performed
 *   cmd (in)
 *     Flags indicating the action to be performed. Here are
 *     the possible fields:
 *       LW0073_CTRL_DFP_SB_I2C_CTRL_CMD_WRITE
 *         Used to perform I2C write using sideband communication channel
 *       LW0073_CTRL_DFP_SB_I2C_CTRL_CMD_READ
 *         Used to perform I2C write using sideband communication channel
 *   addr (in)
 *     The addr to which the read or write transactions are targeted
 *   data[] (in/out)
 *     In case of a read transaction, this parameter returns the data from
 *     transaction request.  In case of a write transaction, the client
 *     should write to this buffer for the data to send.  The max # of bytes
 *     allowed is LW0073_CTRL_DFP_DISPMUX_SIDEBAND_I2C_CTRL_MAX_DATA_SIZE.
 *   size (in)
 *     Specifies how many data bytes to read/write depending on the transaction type.
 *     Lwrrently, this is limited to LW0073_CTRL_DFP_DISPMUX_SIDEBAND_I2C_CTRL_MAX_DATA_SIZE.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_TIMEOUT
 */

#define LW0073_CTRL_CMD_DFP_DISPMUX_SIDEBAND_I2C_CTRL           (0x731164U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DFP_DISPMUX_SIDEBAND_I2C_CTRL_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DFP_DISPMUX_SIDEBAND_I2C_CTRL_MAX_DATA_SIZE 1U
#define LW0073_CTRL_CMD_DFP_DISPMUX_SIDEBAND_I2C_CTRL_PARAMS_MESSAGE_ID (0x64U)

typedef struct LW0073_CTRL_CMD_DFP_DISPMUX_SIDEBAND_I2C_CTRL_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 cmd;
    LwU16 addr;
    LwU8  data[LW0073_CTRL_DFP_DISPMUX_SIDEBAND_I2C_CTRL_MAX_DATA_SIZE];
    LwU32 size;
} LW0073_CTRL_CMD_DFP_DISPMUX_SIDEBAND_I2C_CTRL_PARAMS;

/* valid commands */
#define LW0073_CTRL_DFP_DISPMUX_SB_I2C_CTRL_CMD_ILWALID  0x00000000U
#define LW0073_CTRL_DFP_DISPMUX_SB_I2C_CTRL_CMD_WRITE    0x00000001U
#define LW0073_CTRL_DFP_DISPMUX_SB_I2C_CTRL_CMD_READ     0x00000002U

/*
 * LW0073_CTRL_CMD_DFP_DISPMUX_GET_IGPU_BLEN_STATUS
 *
 * This command is used to query the iGPU Backlight Enable GPIO status
 *
 *   subDeviceInstance (in)
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *   displayId (in)
 *     ID of the display device for which the post mux switch operations have
 *     to be performed.
 *   igpuBlState (out)
 *     status indicating the current state of the iGPU BLEN GPIO.
 *     valid values -
 *       LW0073_CTRL_DFP_DISPMUX_IGPU_BLEN_STATE_ON
 *         Indicates that iGPU has turned on its backlight control
 *       LW0073_CTRL_DFP_DISPMUX_IGPU_BLEN_STATE_OFF
 *         Indicates that iGPU has turned off its backlight control
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW0073_CTRL_CMD_DFP_DISPMUX_GET_IGPU_BLEN_STATUS (0x731165U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DFP_DISPMUX_IGPU_BLEN_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DFP_DISPMUX_IGPU_BLEN_PARAMS_MESSAGE_ID (0x65U)

typedef struct LW0073_CTRL_CMD_DFP_DISPMUX_IGPU_BLEN_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 igpuBlState;
} LW0073_CTRL_CMD_DFP_DISPMUX_IGPU_BLEN_PARAMS;

/* valid values */
#define LW0073_CTRL_DFP_DISPMUX_IGPU_BLEN_STATE_OFF     0x00000000U
#define LW0073_CTRL_DFP_DISPMUX_IGPU_BLEN_STATE_ON      0x00000001U
#define LW0073_CTRL_DFP_DISPMUX_IGPU_BLEN_STATE_ILWALID 0xFFFFFFFFU

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
*  LW0073_CTRL_CMD_DFP_GET_DSI_MODE_TIMING
*
*  This command can be used to get DSI mode timing parameters.
*
*   subDeviceInstance
*     This parameter specifies the subdevice instance within the
*     LW04_DISPLAY_COMMON parent device to which the operation should be
*     directed. This parameter must specify a value between zero and the
*     total number of subdevices within the parent device.  This parameter
*     should be set to zero for default behavior.
*   displayId
*     This parameter specifies the ID of the display on which the DSI
*     info will be set. The display ID must be a DSI-capable display.
*   hActive
*     This parameter specifies the horizontal length of the active pixel
*     data in the raster.
*   vActive
*     This parameter specifies the vertical lines of the active pixel
*     data in the raster.
*   bpp
*     This parameter specifies the depth (Bits per Pixel) of the output
*     display stream.
*   refresh
*     This parameter specifies the refresh rate of the panel (in Hz).
*
*  Possible status values returned are:
*   LW_OK
*   LW_ERR_ILWALID_ARGUMENT
*   LW_ERR_NOT_SUPPORTED
*/

#define LW0073_CTRL_CMD_DFP_GET_DSI_MODE_TIMING         (0x731166U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DFP_GET_DSI_MODE_TIMING_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DFP_GET_DSI_MODE_TIMING_PARAMS_MESSAGE_ID (0x66U)

typedef struct LW0073_CTRL_CMD_DFP_GET_DSI_MODE_TIMING_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 hActive;
    LwU32 vActive;
    LwU32 bpp;
    LwU32 refresh;
} LW0073_CTRL_CMD_DFP_GET_DSI_MODE_TIMING_PARAMS;

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

/*
*  LW0073_CTRL_CMD_DFP_DISPMUX_GET_TCON_INFO
*  
*  This command is used to get id information about TCON.
*  
*  subDeviceInstance (in)
*    This parameter specifies the subdevice instance within the
*    LW04_DISPLAY_COMMON parent device to which the operation should be
*    directed.
*  displayId (in)
*    ID of the display device for which the post mux switch operations have
*    to be performed.
*  vendorOui (out)
*    IEEE OUI of the TCON vendor.
*  ddsFirmwareRev (out)
*    DDS specific firmware revision #.
*  hardwareRev (out)
*    Hardware revision #.
*  softwareRev (out)
*    Software revision #. In most TCONs, this value is set to a 16 bit
*    panel FW checksum (different from DDS FW).
*  deviceName (out)
*    Name of the TCON HW part.
*
*  Possible status values returned are:
*    LW_OK
*    LW_ERR_ILWALID_ARGUMENT
*    LW_ERR_NOT_SUPPORTED
*/
#define LW0073_CTRL_CMD_DFP_DISPMUX_GET_TCON_INFO      (0x731167U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DFP_DISPMUX_GET_TCON_INFO_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DFP_DISPMUX_TCON_NAME_MAX_SIZE 20U

#define LW0073_CTRL_CMD_DFP_DISPMUX_GET_TCON_INFO_PARAMS_MESSAGE_ID (0x67U)

typedef struct LW0073_CTRL_CMD_DFP_DISPMUX_GET_TCON_INFO_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 vendorOui;
    LwU16 ddsFirmwareRev;
    LwU16 hardwareRev;
    LwU16 softwareRev;
    char  deviceName[LW0073_CTRL_CMD_DFP_DISPMUX_TCON_NAME_MAX_SIZE];
} LW0073_CTRL_CMD_DFP_DISPMUX_GET_TCON_INFO_PARAMS;

/*
*  LW0073_CTRL_CMD_DFP_DISPMUX_GET_MUX_DEVICE_ID
*  
*  This command is used to get ID information about MUX device.
*  
*  subDeviceInstance (in)
*    This parameter specifies the subdevice instance within the
*    LW04_DISPLAY_COMMON parent device to which the operation should be
*    directed.
*  displayId (in)
*    ID of the display device for which the post mux switch operations have
*    to be performed.
*  vendorId (out)
*    ID of the mux vendor.
*  deviceId (out)
*    ID of the mux part.
*  vendorName (out)
*    Name of the mux vendor.
*  deviceName (out)
*    Name of the mux device.
*  auxSettleDelayMs (out)
*    Time needed for AUX channel to settle after mux switch.
*  
*  Possible status values returned are:
*    LW_OK
*    LW_ERR_ILWALID_ARGUMENT
*    LW_ERR_NOT_SUPPORTED
*/
#define LW0073_CTRL_CMD_DFP_DISPMUX_GET_MUX_DEVICE_ID (0x731168U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DFP_DISPMUX_GET_MUX_DEVICE_ID_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DFP_DISPMUX_MUX_NAME_MAX_SIZE 20U

#define LW0073_CTRL_CMD_DFP_DISPMUX_GET_MUX_DEVICE_ID_PARAMS_MESSAGE_ID (0x68U)

typedef struct LW0073_CTRL_CMD_DFP_DISPMUX_GET_MUX_DEVICE_ID_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 auxSettleDelayMs;
    LwU16 vendorId;
    LwU16 deviceId;
    char  vendorName[LW0073_CTRL_CMD_DFP_DISPMUX_MUX_NAME_MAX_SIZE];
    char  deviceName[LW0073_CTRL_CMD_DFP_DISPMUX_MUX_NAME_MAX_SIZE];
} LW0073_CTRL_CMD_DFP_DISPMUX_GET_MUX_DEVICE_ID_PARAMS;

/*
 * LW0073_CTRL_CMD_DFP_DISPMUX_SET_PANEL_HDR_SETTINGS
 *
 * This command is used to set the panel HDR/brightness3 settings when the 
 * panel is using custom Aux based HDR settings. This call is used for
 * internal HDR capable panels when dynamic mux switching is enabled with
 * an Intel Integrated GPU.
 *
 *   subDeviceInstance (in)
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed.
 *   displayId (in)
 *     ID of the display device for which the post mux switch operations have
 *     to be performed.
 *   maxContentLuminanceNits (in)
 *     indicating the maximum luminance of the content being sent by source
 *   maxAvgContentLuminanceNits (in)
 *     indicating the maximum average luminance of the content being sent by source
 *   minPanelLumOverrideNits (in)
 *     Overrides the minimum luminance in nits that the panel should emulate. This
 *     should usually match the minimum luminance indicated in EDID's HDR metadata.
 *   maxPanelLumOverrideNits (in)
 *     Overrides the maximum luminance in nits that the panel should emulate. This
 *     should usually be less than or equal to the maximum luminance indicated in 
 *     EDID's HDR metadata.
 *   maxPanelFullFrameLumOverride (in)
 *     Overrides the maximum full-frame luminance in nits that the panel should emulate. 
 *     This value should usually be less-than-or-equal to the maximum full-frame luminance
 *     indicated in EDID's HDR metadata. This value is usually less-than-or-equal to the
 *     maxPanelLumOverrideNits
 *   lwrPanelBrightnessNits (in)
 *     Specifies what should be the current luminance when panel is controlling the 
 *     brightness. In case of LCD panels where brightness is controlled by GPU PWM in SDR
 *     mode, this setting is set to the same value as maxPanelLumOverrideNits and GPU is
 *     responsible for controlling the brightness via PWM.
 *   numFramesToApplyBrightnessIn (in)
 *     Indicating the number of frames overwhich lwrPanelBrightnessNits is applied (when 
 *     not using GPU PWM for brightness control)
 *   perFrameStepsBrightnessToApply (in)
 *     Indicates number of brightness steps to be applied within a frame. This is usually
 *     an optional value for TCONs to implement and can be left to 0.
 *   bHdrEnabled
 *     Indicates that panel should be configured to accept HDR10 input
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW0073_CTRL_CMD_DFP_DISPMUX_SET_PANEL_HDR_SETTINGS (0x731169U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DFP_DISPMUX_SET_PANEL_HDR_SETTINGS_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DFP_DISPMUX_SET_PANEL_HDR_SETTINGS_PARAMS_MESSAGE_ID (0x69U)

typedef struct LW0073_CTRL_CMD_DFP_DISPMUX_SET_PANEL_HDR_SETTINGS_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  displayId;
    LwU32  maxContentLuminanceNits;
    LwU32  maxAvgContentLuminanceNits;
    LwU32  minPanelLumOverrideNits;
    LwU32  maxPanelLumOverrideNits;
    LwU32  maxPanelFullFrameLumOverride;
    LwU32  lwrPanelBrightnessNits;
    LwU32  numFramesToApplyBrightnessIn;
    LwU32  perFrameStepsBrightnessToApply;
    LwBool bHdrEnabled;
} LW0073_CTRL_CMD_DFP_DISPMUX_SET_PANEL_HDR_SETTINGS_PARAMS;

/*
*  LW0073_CTRL_CMD_DFP_DISPMUX_ASSR_CTRL
*  
*  This command is used to query / set panel ASSR state via I2C after LRST
*  
*  subDeviceInstance (in)
*    This parameter specifies the subdevice instance within the
*    LW04_DISPLAY_COMMON parent device to which the operation should be
*    directed.
*  displayId (in)
*    ID of the display device for which LRST has been triggered.
*  cmd (in)
*    Query / set ASSR state
*  state (in/out)
*    Current ASSR state read / New ASSR state to be set
*
*  Possible status values returned are:
*    LW_OK
*    LW_ERR_ILWALID_ARGUMENT
*    LW_ERR_NOT_SUPPORTED
*/
#define LW0073_CTRL_CMD_DFP_DISPMUX_ASSR_CTRL (0x731170U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DFP_DISPMUX_ASSR_CTRL_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DFP_DISPMUX_ASSR_CTRL_PARAMS_MESSAGE_ID (0x70U)

typedef struct LW0073_CTRL_CMD_DFP_DISPMUX_ASSR_CTRL_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 cmd;
    LwU8  state;
} LW0073_CTRL_CMD_DFP_DISPMUX_ASSR_CTRL_PARAMS;

/* valid commands and state values */
#define LW0073_CTRL_CMD_DFP_DISPMUX_ASSR_CMD_QUERY         0x00000000U
#define LW0073_CTRL_CMD_DFP_DISPMUX_ASSR_CMD_SET           0x00000001U
#define LW0073_CTRL_CMD_DFP_DISPMUX_ASSR_STATE_DISABLED    0x00000000U
#define LW0073_CTRL_CMD_DFP_DISPMUX_ASSR_STATE_ENABLED     0x00000001U

/*
*  LW0073_CTRL_CMD_DFP_DISPMUX_GET_SIDEBAND_DSC_CAPS
*  
*  This command is used to read the DSC caps from a MUX capable eDP panel.
*  Uses sideband i2c to read DPCD offsets 0x60 to 0x6f
*  
*  subDeviceInstance (in)
*    This parameter specifies the subdevice instance within the
*    LW04_DISPLAY_COMMON parent device to which the operation should be
*    directed.
*  displayId (in)
*    ID of the display device for which the caps should be fetched
*  dscCaps (out)
*    The 16 byte caps read from DPCD offset 0x60 to 0x6f
*
*  Possible status values returned are:
*    LW_OK
*    LW_ERR_ILWALID_ARGUMENT
*    LW_ERR_NOT_SUPPORTED
*/
#define LW0073_CTRL_CMD_DFP_DISPMUX_GET_SIDEBAND_DSC_CAPS  (0x731171U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DFP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DFP_DISPMUX_GET_SIDEBAND_DSC_CAPS_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DFP_DISPMUX_SIDEBAND_DSC_CAPS_SIZE 16U

#define LW0073_CTRL_CMD_DFP_DISPMUX_GET_SIDEBAND_DSC_CAPS_PARAMS_MESSAGE_ID (0x71U)

typedef struct LW0073_CTRL_CMD_DFP_DISPMUX_GET_SIDEBAND_DSC_CAPS_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU8  dscCaps[LW0073_CTRL_CMD_DFP_DISPMUX_SIDEBAND_DSC_CAPS_SIZE];
} LW0073_CTRL_CMD_DFP_DISPMUX_GET_SIDEBAND_DSC_CAPS_PARAMS;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/* _ctrl0073dfp_h_ */
