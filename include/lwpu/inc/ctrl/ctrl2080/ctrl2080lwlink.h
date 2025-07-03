/*
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080lwlink.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#include "ctrl/ctrl2080/ctrl2080base.h"

/* LW20_SUBDEVICE_XX bus control commands and parameters */

/*
 * LW2080_CTRL_CMD_LWLINK_GET_LWLINK_CAPS
 *
 * This command returns the LWLink capabilities supported by the subdevice.
 *
 *   capsTbl
 *     This is bit field for getting different global caps. The individual bitfields are specified by LW2080_CTRL_LWLINK_CAPS_* 
 *   lowestLwlinkVersion
 *     This field specifies the lowest supported LWLink version for this subdevice.
 *   highestLwlinkVersion
 *     This field specifies the highest supported LWLink version for this subdevice.
 *   lowestNciVersion
 *     This field specifies the lowest supported NCI version for this subdevice.
 *   highestNciVersion
 *     This field specifies the highest supported NCI version for this subdevice.
 *   discoveredLinkMask
 *     This field provides a bitfield mask of LWLink links discovered on this subdevice.
 *   enabledLinkMask
 *     This field provides a bitfield mask of LWLink links enabled on this subdevice.
 *
 */
#define LW2080_CTRL_CMD_LWLINK_GET_LWLINK_CAPS_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_CMD_LWLINK_GET_LWLINK_CAPS_PARAMS {
    LwU32 capsTbl;

    LwU8  lowestLwlinkVersion;
    LwU8  highestLwlinkVersion;
    LwU8  lowestNciVersion;
    LwU8  highestNciVersion;

    LwU32 discoveredLinkMask;
    LwU32 enabledLinkMask;
} LW2080_CTRL_CMD_LWLINK_GET_LWLINK_CAPS_PARAMS;

/* extract cap bit setting from tbl */
#define LW2080_CTRL_LWLINK_GET_CAP(tbl,c)              (((LwU8)tbl[(1?c)]) & (0?c))

/*
 * LW2080_CTRL_LWLINK_CAPS
 *
 *   SUPPORTED
 *     Set if LWLink is present and supported on this subdevice, LW_FALSE otherwise. This field is used for *global* caps only and NOT for per-link caps
 *   P2P_SUPPORTED
 *     Set if P2P over LWLink is supported on this subdevice, LW_FALSE otherwise.
 *   SYSMEM_ACCESS
 *     Set if sysmem can be accessed over LWLink on this subdevice, LW_FALSE otherwise.
 *   PEER_ATOMICS
 *     Set if P2P atomics are supported over LWLink on this subdevice, LW_FALSE otherwise.
 *   SYSMEM_ATOMICS
 *     Set if sysmem atomic transcations are supported over LWLink on this subdevice, LW_FALSE otherwise.
 *   PEX_TUNNELING
 *     Set if PEX tunneling over LWLink is supported on this subdevice, LW_FALSE otherwise.
 *   SLI_BRIDGE
 *     GLOBAL: Set if SLI over LWLink is supported on this subdevice, LW_FALSE otherwise.
 *     LINK:   Set if SLI over LWLink is supported on a link, LW_FALSE otherwise.
 *   SLI_BRIDGE_SENSABLE
 *     GLOBAL: Set if the subdevice is capable of sensing SLI bridges, LW_FALSE otherwise.
 *     LINK:   Set if the link is capable of sensing an SLI bridge, LW_FALSE otherwise.
 *   POWER_STATE_L0
 *     Set if L0 is a supported power state on this subdevice/link, LW_FALSE otherwise.
 *   POWER_STATE_L1
 *     Set if L1 is a supported power state on this subdevice/link, LW_FALSE otherwise.
 *   POWER_STATE_L2
 *     Set if L2 is a supported power state on this subdevice/link, LW_FALSE otherwise.
 *   POWER_STATE_L3
 *     Set if L3 is a supported power state on this subdevice/link, LW_FALSE otherwise.
 *   VALID
 *     Set if this link is supported on this subdevice, LW_FALSE otherwise. This field is used for *per-link* caps only and NOT for global caps.
 *
 */

/* caps format is byte_index:bit_mask */
#define LW2080_CTRL_LWLINK_CAPS_SUPPORTED                          0:0x01
#define LW2080_CTRL_LWLINK_CAPS_P2P_SUPPORTED                      0:0x02
#define LW2080_CTRL_LWLINK_CAPS_SYSMEM_ACCESS                      0:0x04
#define LW2080_CTRL_LWLINK_CAPS_P2P_ATOMICS                        0:0x08
#define LW2080_CTRL_LWLINK_CAPS_SYSMEM_ATOMICS                     0:0x10
#define LW2080_CTRL_LWLINK_CAPS_PEX_TUNNELING                      0:0x20
#define LW2080_CTRL_LWLINK_CAPS_SLI_BRIDGE                         0:0x40
#define LW2080_CTRL_LWLINK_CAPS_SLI_BRIDGE_SENSABLE                0:0x80
#define LW2080_CTRL_LWLINK_CAPS_POWER_STATE_L0                     1:0x01
#define LW2080_CTRL_LWLINK_CAPS_POWER_STATE_L1                     1:0x02
#define LW2080_CTRL_LWLINK_CAPS_POWER_STATE_L2                     1:0x04
#define LW2080_CTRL_LWLINK_CAPS_POWER_STATE_L3                     1:0x08
#define LW2080_CTRL_LWLINK_CAPS_VALID                              1:0x10

/*
 * Size in bytes of lwlink caps table.  This value should be one greater
 * than the largest byte_index value above.
 */
#define LW2080_CTRL_LWLINK_CAPS_TBL_SIZE               2U

#define LW2080_CTRL_LWLINK_CAPS_LWLINK_VERSION_ILWALID (0x00000000U)
#define LW2080_CTRL_LWLINK_CAPS_LWLINK_VERSION_1_0     (0x00000001U)
#define LW2080_CTRL_LWLINK_CAPS_LWLINK_VERSION_2_0     (0x00000002U)
#define LW2080_CTRL_LWLINK_CAPS_LWLINK_VERSION_2_2     (0x00000004U)
#define LW2080_CTRL_LWLINK_CAPS_LWLINK_VERSION_3_0     (0x00000005U)
#define LW2080_CTRL_LWLINK_CAPS_LWLINK_VERSION_3_1     (0x00000006U)
#define LW2080_CTRL_LWLINK_CAPS_LWLINK_VERSION_4_0     (0x00000007U)

#define LW2080_CTRL_LWLINK_CAPS_NCI_VERSION_ILWALID    (0x00000000U)
#define LW2080_CTRL_LWLINK_CAPS_NCI_VERSION_1_0        (0x00000001U)
#define LW2080_CTRL_LWLINK_CAPS_NCI_VERSION_2_0        (0x00000002U)
#define LW2080_CTRL_LWLINK_CAPS_NCI_VERSION_2_2        (0x00000004U)
#define LW2080_CTRL_LWLINK_CAPS_NCI_VERSION_3_0        (0x00000005U)
#define LW2080_CTRL_LWLINK_CAPS_NCI_VERSION_3_1        (0x00000006U)
#define LW2080_CTRL_LWLINK_CAPS_NCI_VERSION_4_0        (0x00000007U)


/*
 * LW2080_CTRL_CMD_LWLINK_GET_LWLINK_CAPS
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW2080_CTRL_CMD_LWLINK_GET_LWLINK_CAPS         (0x20803001U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_CMD_LWLINK_GET_LWLINK_CAPS_PARAMS_MESSAGE_ID" */

/* 
 * LW2080_CTRL_LWLINK_DEVICE_INFO
 *
 * This structure stores information about the device to which this link is associated
 *
 *   deviceIdFlags
 *      Bitmask that specifies which IDs are valid for the device
 *      Refer LW2080_CTRL_LWLINK_DEVICE_INFO_DEVICE_ID_FLAGS_* for possible values
 *      If LW2080_CTRL_LWLINK_DEVICE_INFO_DEVICE_ID_FLAGS_PCI is set, PCI information is valid
 *      If LW2080_CTRL_LWLINK_DEVICE_INFO_DEVICE_ID_FLAGS_UUID is set, UUID is valid
 *   domain, bus, device, function, pciDeviceId
 *      PCI information for the device
 *   deviceType
 *      Type of the device
 *      See LW2080_CTRL_LWLINK_DEVICE_INFO_DEVICE_TYPE_* for possible values
 *   deviceUUID
 *      This field specifies the device UUID of the device. Useful for identifying the device (or version)
 */
typedef struct LW2080_CTRL_LWLINK_DEVICE_INFO {
    // ID Flags
    LwU32 deviceIdFlags;

    // PCI Information
    LwU32 domain;
    LwU16 bus;
    LwU16 device;
    LwU16 function;
    LwU32 pciDeviceId;

    // Device Type
    LW_DECLARE_ALIGNED(LwU64 deviceType, 8);

    // Device UUID
    LwU8  deviceUUID[16];
} LW2080_CTRL_LWLINK_DEVICE_INFO;

#define LW2080_CTRL_LWLINK_DEVICE_INFO_DEVICE_ID_FLAGS        31:0 
#define LW2080_CTRL_LWLINK_DEVICE_INFO_DEVICE_ID_FLAGS_NONE (0x00000000U)
#define LW2080_CTRL_LWLINK_DEVICE_INFO_DEVICE_ID_FLAGS_PCI  (0x00000001U)
#define LW2080_CTRL_LWLINK_DEVICE_INFO_DEVICE_ID_FLAGS_UUID (0x00000002U)

#define LW2080_CTRL_LWLINK_DEVICE_INFO_DEVICE_TYPE_EBRIDGE  (0x00000000U)
#define LW2080_CTRL_LWLINK_DEVICE_INFO_DEVICE_TYPE_NPU      (0x00000001U)
#define LW2080_CTRL_LWLINK_DEVICE_INFO_DEVICE_TYPE_GPU      (0x00000002U)
#define LW2080_CTRL_LWLINK_DEVICE_INFO_DEVICE_TYPE_SWITCH   (0x00000003U)
#define LW2080_CTRL_LWLINK_DEVICE_INFO_DEVICE_TYPE_TEGRA    (0x00000004U)
#define LW2080_CTRL_LWLINK_DEVICE_INFO_DEVICE_TYPE_NONE     (0x000000FFU)

#define LW2080_CTRL_LWLINK_DEVICE_INFO_DEVICE_UUID_ILWALID  (0xFFFFFFFFU)

/*
 * LW2080_CTRL_LWLINK_LWLINK_LINK_STATUS_INFO
 *
 * This structure stores the per-link status of different LWLink parameters.
 *
 *   capsTbl
 *     This is bit field for getting different global caps. The individual bitfields
 *     are specified by LW2080_CTRL_LWLINK_CAPS_*
 *   phyType
 *     This field specifies the type of PHY (LWHS or GRS) being used for this link.
 *   subLinkWidth
 *     This field specifies the no. of lanes per sublink.
 *   linkState
 *     This field specifies the current state of the link.
 *     See LW2080_CTRL_CMD_LWLINK_GET_LWLINK_STATUS_LINK_STATE_* for possible values.
 *   rxSublinkStatus
 *     This field specifies the current state of RX sublink.
 *     See LW2080_CTRL_CMD_LWLINK_GET_LWLINK_STATUS_SUBLINK_RX_STATE_* for possible values.
 *   txSublinkStatus
 *     This field specifies the current state of TX sublink.
 *     See LW2080_CTRL_CMD_LWLINK_GET_LWLINK_STATUS_SUBLINK_TX_STATE_* for possible values.
 *   bLaneReversal
 *     This field indicates that lane reversal is in effect on this link.
 *   lwlinkVersion
 *     This field specifies the LWLink version supported by the link.
 *   nciVersion
 *     This field specifies the NCI version supported by the link.
 *   phyVersion
 *     This field specifies the version of PHY being used by the link.
 *   lwlinkLineRateMbps
 *      Bit rate at which bits toggle on wires in megabits per second.
 *      NOTE: This value is the full speed line rate, not the instantaneous line rate of the link.
 *   lwlinkLinkClockMhz
 *      Clock corresponding to link logic in mega hertz
 *   lwlinkRefClkType
 *      This field specifies whether the link clock is taken from LWHS reflck
 *      or PEX refclk for the current GPU.
 *   lwlinkLinkDataRateKiBps
 *      Effective rate available for transactions after subtracting overhead,
 *      as seen at Data Layer in kibibytes (1024 bytes) per second.
 *      Only valid in GA100+, reported as 0 otherwise
 *      NOTE: Because minion callwlates these values, it will only be valid if
 *            links are in ACTIVE state
 *   lwlinkRefClkSpeedMhz
 *      The input reference frequency to the PLL
 *   connected
 *     This field specifies if any device is connected on the other end of the link
 *   loopProperty
 *     This field specifies if the link is a loopback/loopout link. See LW2080_CTRL_LWLINK_STATUS_LOOP_PROPERTY_* for possible values.
 *   remoteDeviceLinkNumber
 *     This field specifies the link number on the remote end of the link 
 *   remoteDeviceInfo
 *     This field stores the device information for the remote end of the link
 *
 */
typedef struct LW2080_CTRL_LWLINK_LINK_STATUS_INFO {
    // Top level capablilites
    LwU32  capsTbl;

    LwU8   phyType;
    LwU8   subLinkWidth;

    // Link and sublink states
    LwU32  linkState;
    LwU8   rxSublinkStatus;
    LwU8   txSublinkStatus;

    // Indicates that lane reversal is in effect on this link.
    LwBool bLaneReversal;

    LwU8   lwlinkVersion;
    LwU8   nciVersion;
    LwU8   phyVersion;

    // Legacy clock information (to be deprecated)
    LwU32  lwlinkLinkClockKHz;
    LwU32  lwlinkCommonClockSpeedKHz;
    LwU32  lwlinkRefClkSpeedKHz;

    LwU32  lwlinkCommonClockSpeedMhz;

    // Clock Speed and Data Rate Reporting
    LwU32  lwlinkLineRateMbps;
    LwU32  lwlinkLinkClockMhz;
    LwU8   lwlinkRefClkType;
    LwU32  lwlinkLinkDataRateKiBps;
    LwU32  lwlinkRefClkSpeedMhz;

    // Connection information
    LwBool connected;
    LwU8   loopProperty;
    LwU8   remoteDeviceLinkNumber;
    LwU8   localDeviceLinkNumber;

    //
    // Added as part of LwLink 3.0
    // Note: SID has link info appended to it when provided by minion
    //
    LW_DECLARE_ALIGNED(LwU64 remoteLinkSid, 8);
    LW_DECLARE_ALIGNED(LwU64 localLinkSid, 8);

    // Ampere+ only
    LwU32  laneRxdetStatusMask;

    LW_DECLARE_ALIGNED(LW2080_CTRL_LWLINK_DEVICE_INFO remoteDeviceInfo, 8);
    LW_DECLARE_ALIGNED(LW2080_CTRL_LWLINK_DEVICE_INFO localDeviceInfo, 8);
} LW2080_CTRL_LWLINK_LINK_STATUS_INFO;

// LWLink link states
#define LW2080_CTRL_LWLINK_STATUS_LINK_STATE_INIT               (0x00000000U)
#define LW2080_CTRL_LWLINK_STATUS_LINK_STATE_HWCFG              (0x00000001U)
#define LW2080_CTRL_LWLINK_STATUS_LINK_STATE_SWCFG              (0x00000002U)
#define LW2080_CTRL_LWLINK_STATUS_LINK_STATE_ACTIVE             (0x00000003U)
#define LW2080_CTRL_LWLINK_STATUS_LINK_STATE_FAULT              (0x00000004U)
#define LW2080_CTRL_LWLINK_STATUS_LINK_STATE_RECOVERY           (0x00000006U)
#define LW2080_CTRL_LWLINK_STATUS_LINK_STATE_RECOVERY_AC        (0x00000008U)
#define LW2080_CTRL_LWLINK_STATUS_LINK_STATE_RECOVERY_RX        (0x0000000aU)
#define LW2080_CTRL_LWLINK_STATUS_LINK_STATE_ILWALID            (0xFFFFFFFFU)

// LWLink Rx sublink states
#define LW2080_CTRL_LWLINK_STATUS_SUBLINK_RX_STATE_HIGH_SPEED_1 (0x00000000U)
#define LW2080_CTRL_LWLINK_STATUS_SUBLINK_RX_STATE_SINGLE_LANE  (0x00000004U)
#define LW2080_CTRL_LWLINK_STATUS_SUBLINK_RX_STATE_TRAINING     (0x00000005U)
#define LW2080_CTRL_LWLINK_STATUS_SUBLINK_RX_STATE_SAFE_MODE    (0x00000006U)
#define LW2080_CTRL_LWLINK_STATUS_SUBLINK_RX_STATE_OFF          (0x00000007U)
#define LW2080_CTRL_LWLINK_STATUS_SUBLINK_RX_STATE_TEST         (0x00000008U)
#define LW2080_CTRL_LWLINK_STATUS_SUBLINK_RX_STATE_FAULT        (0x0000000eU)
#define LW2080_CTRL_LWLINK_STATUS_SUBLINK_RX_STATE_ILWALID      (0x000000FFU)

// LWLink Tx sublink states
#define LW2080_CTRL_LWLINK_STATUS_SUBLINK_TX_STATE_HIGH_SPEED_1 (0x00000000U)
#define LW2080_CTRL_LWLINK_STATUS_SUBLINK_TX_STATE_SINGLE_LANE  (0x00000004U)
#define LW2080_CTRL_LWLINK_STATUS_SUBLINK_TX_STATE_TRAINING     (0x00000005U)
#define LW2080_CTRL_LWLINK_STATUS_SUBLINK_TX_STATE_SAFE_MODE    (0x00000006U)
#define LW2080_CTRL_LWLINK_STATUS_SUBLINK_TX_STATE_OFF          (0x00000007U)
#define LW2080_CTRL_LWLINK_STATUS_SUBLINK_TX_STATE_TEST         (0x00000008U)
#define LW2080_CTRL_LWLINK_STATUS_SUBLINK_TX_STATE_FAULT        (0x0000000eU)
#define LW2080_CTRL_LWLINK_STATUS_SUBLINK_TX_STATE_ILWALID      (0x000000FFU)

#define LW2080_CTRL_LWLINK_STATUS_PHY_LWHS                      (0x00000001U)
#define LW2080_CTRL_LWLINK_STATUS_PHY_GRS                       (0x00000002U)
#define LW2080_CTRL_LWLINK_STATUS_PHY_ILWALID                   (0x000000FFU)

// Version information
#define LW2080_CTRL_LWLINK_STATUS_LWLINK_VERSION_1_0            (0x00000001U)
#define LW2080_CTRL_LWLINK_STATUS_LWLINK_VERSION_2_0            (0x00000002U)
#define LW2080_CTRL_LWLINK_STATUS_LWLINK_VERSION_2_2            (0x00000004U)
#define LW2080_CTRL_LWLINK_STATUS_LWLINK_VERSION_3_0            (0x00000005U)
#define LW2080_CTRL_LWLINK_STATUS_LWLINK_VERSION_3_1            (0x00000006U)
#define LW2080_CTRL_LWLINK_STATUS_LWLINK_VERSION_4_0            (0x00000007U)
#define LW2080_CTRL_LWLINK_STATUS_LWLINK_VERSION_ILWALID        (0x000000FFU)

#define LW2080_CTRL_LWLINK_STATUS_NCI_VERSION_1_0               (0x00000001U)
#define LW2080_CTRL_LWLINK_STATUS_NCI_VERSION_2_0               (0x00000002U)
#define LW2080_CTRL_LWLINK_STATUS_NCI_VERSION_2_2               (0x00000004U)
#define LW2080_CTRL_LWLINK_STATUS_NCI_VERSION_3_0               (0x00000005U)
#define LW2080_CTRL_LWLINK_STATUS_NCI_VERSION_3_1               (0x00000006U)
#define LW2080_CTRL_LWLINK_STATUS_NCI_VERSION_4_0               (0x00000007U)
#define LW2080_CTRL_LWLINK_STATUS_NCI_VERSION_ILWALID           (0x000000FFU)

#define LW2080_CTRL_LWLINK_STATUS_LWHS_VERSION_1_0              (0x00000001U)
#define LW2080_CTRL_LWLINK_STATUS_LWHS_VERSION_ILWALID          (0x000000FFU)

#define LW2080_CTRL_LWLINK_STATUS_GRS_VERSION_1_0               (0x00000001U)
#define LW2080_CTRL_LWLINK_STATUS_GRS_VERSION_ILWALID           (0x000000FFU)

// Connection properties
#define LW2080_CTRL_LWLINK_STATUS_CONNECTED_TRUE                (0x00000001U)
#define LW2080_CTRL_LWLINK_STATUS_CONNECTED_FALSE               (0x00000000U)

#define LW2080_CTRL_LWLINK_STATUS_LOOP_PROPERTY_LOOPBACK        (0x00000001U)
#define LW2080_CTRL_LWLINK_STATUS_LOOP_PROPERTY_LOOPOUT         (0x00000002U)
#define LW2080_CTRL_LWLINK_STATUS_LOOP_PROPERTY_NONE            (0x00000000U)

#define LW2080_CTRL_LWLINK_STATUS_REMOTE_LINK_NUMBER_ILWALID    (0x000000FFU)

#define LW2080_CTRL_LWLINK_MAX_LINKS                            32

// LWLink REFCLK types
#define LW2080_CTRL_LWLINK_REFCLK_TYPE_ILWALID                  (0x00U)
#define LW2080_CTRL_LWLINK_REFCLK_TYPE_LWHS                     (0x01U)
#define LW2080_CTRL_LWLINK_REFCLK_TYPE_PEX                      (0x02U)

#define LW2080_CTRL_CMD_LWLINK_GET_LWLINK_STATUS_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW2080_CTRL_CMD_LWLINK_GET_LWLINK_STATUS_PARAMS {
    LwU32 enabledLinkMask;
    LW_DECLARE_ALIGNED(LW2080_CTRL_LWLINK_LINK_STATUS_INFO linkInfo[LW2080_CTRL_LWLINK_MAX_LINKS], 8);
} LW2080_CTRL_CMD_LWLINK_GET_LWLINK_STATUS_PARAMS;

/*
 * LW2080_CTRL_CMD_LWLINK_GET_LWLINK_STATUS
 *
 *   enabledLinkMask
 *     This field specifies the mask of available links on this subdevice.
 *   linkInfo
 *     This structure stores the per-link status of different LWLink parameters. The link is identified using an index.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW2080_CTRL_CMD_LWLINK_GET_LWLINK_STATUS (0x20803002U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_CMD_LWLINK_GET_LWLINK_STATUS_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_LWLINK_ERR_INFO
 *   Error information per link
 *
 *   TLErrlog
 *     Returns the error mask for LWLINK TL errors
 *     Used in Pascal
 *
 *   TLIntrEn
 *     Returns the intr enable mask for LWLINK TL errors
 *     Used in Pascal
 *
 *   TLCTxErrStatus0
 *     Returns the TLC Tx Error Mask 0
 *     Used in Volta and later
 *
 *   TLCTxErrStatus1
 *     Returns the TLC Tx Error Mask 1
 *     Used in Ampere and later
 *
 *   TLCTxSysErrStatus0
 *     Returns the TLC Tx Sys Error Mask 0
 *     Used in Ampere and later.
 *
 *   TLCRxErrStatus0
 *     Returns the TLC Rx Error Mask 0
 *     Used in Volta and later
 *
 *   TLCRxErrStatus1
 *     Returns the TLC Rx Error Mask 1
 *     Used in Volta and later
 *
 *   TLCRxSysErrStatus0
 *     Returns the TLC Rx Sys Error Mask 0
 *     Used in Ampere and later.
 *
 *   TLCTxErrLogEn0
 *     Returns the TLC Tx Error Log En 0
 *     Used in Volta and later
 *
 *   TLCTxErrLogEn1
 *     Returns the TLC Tx Error Log En 1
 *     Used in Ampere and later
 *
 *   TLCTxSysErrLogEn0
 *     Returns the TLC Tx Sys Error Log En 0
 *     Used in Ampere and later
 *
 *   TLCRxErrLogEn0
 *     Returns the TLC Rx Error Log En 0
 *     Used in Volta and later
 *
 *   TLCRxErrLogEn1
 *     Returns the TLC Rx Error Log En 1
 *     Used in Volta and later
 *
 *   TLCRxSysErrLogEn0
 *     Returns the TLC Rx Sys Error Log En 0
 *     Used in Ampere and later
 *
 *   MIFTxErrStatus0
 *     Returns the MIF Rx Error Mask 0
 *     Used in Volta and Turing
 *
 *   MIFRxErrStatus0
 *     Returns the MIF Tx Error Mask 0
 *     Used in Volta and Turing
 *
 *   LWLIPTLnkErrStatus0
 *     Returns the LWLIPT_LNK Error Mask 0
 *     Used in Ampere and later
 *
 *   LWLIPTLnkErrLogEn0
 *     Returns the LWLIPT_LNK Log En Mask 0
 *     Used in Ampere and later
 *
 *   DLSpeedStatusTx
 *     Returns the LWLINK DL speed status for sublink Tx
 *
 *   DLSpeedStatusRx
 *     Returns the LWLINK DL speed status for sublink Rx
 *
 *   bExcessErrorDL
 *     Returns true for excessive error rate interrupt from DL
 */
typedef struct LW2080_CTRL_LWLINK_ERR_INFO {
    LwU32  TLErrlog;
    LwU32  TLIntrEn;
    LwU32  TLCTxErrStatus0;
    LwU32  TLCTxErrStatus1;
    LwU32  TLCTxSysErrStatus0;
    LwU32  TLCRxErrStatus0;
    LwU32  TLCRxErrStatus1;
    LwU32  TLCRxSysErrStatus0;
    LwU32  TLCTxErrLogEn0;
    LwU32  TLCTxErrLogEn1;
    LwU32  TLCTxSysErrLogEn0;
    LwU32  TLCRxErrLogEn0;
    LwU32  TLCRxErrLogEn1;
    LwU32  TLCRxSysErrLogEn0;
    LwU32  MIFTxErrStatus0;
    LwU32  MIFRxErrStatus0;
    LwU32  LWLIPTLnkErrStatus0;
    LwU32  LWLIPTLnkErrLogEn0;
    LwU32  DLSpeedStatusTx;
    LwU32  DLSpeedStatusRx;
    LwBool bExcessErrorDL;
} LW2080_CTRL_LWLINK_ERR_INFO;

/*
 * LW2080_CTRL_LWLINK_COMMON_ERR_INFO
 *   Error information per IOCTRL
 *
 *   LWLIPTErrStatus0
 *     Returns the LWLIPT_COMMON Error Mask 0
 *     Used in Ampere and later
 *
 *   LWLIPTErrLogEn0
 *     Returns the LWLIPT_COMMON Log En Mask 0
 *     Used in Ampere and later
 */
typedef struct LW2080_CTRL_LWLINK_COMMON_ERR_INFO {
    LwU32 LWLIPTErrStatus0;
    LwU32 LWLIPTErrLogEn0;
} LW2080_CTRL_LWLINK_COMMON_ERR_INFO;

/* Extract the error status bit for a given TL error index i */
#define LW2080_CTRL_LWLINK_GET_TL_ERRLOG_BIT(intr, i)       (((1U << i) & (intr)) >> i)

/* Extract the intr enable bit for a given TL error index i */
#define LW2080_CTRL_LWLINK_GET_TL_INTEN_BIT(intr, i)        LW2080_CTRL_LWLINK_GET_TL_ERRLOG_BIT(intr, i)

/* Error status values for a given LWLINK TL error */
#define LW2080_CTRL_LWLINK_TL_ERRLOG_TRUE                               (0x00000001U)
#define LW2080_CTRL_LWLINK_TL_ERRLOG_FALSE                              (0x00000000U)

/* Intr enable/disable for a given LWLINK TL error */
#define LW2080_CTRL_LWLINK_TL_INTEN_TRUE                                (0x00000001U)
#define LW2080_CTRL_LWLINK_TL_INTEN_FALSE                               (0x00000000U)

/* LWLINK TL interrupt enable fields for errors */
#define LW2080_CTRL_LWLINK_TL_INTEN_IDX_RXDLDATAPARITYEN                0U
#define LW2080_CTRL_LWLINK_TL_INTEN_IDX_RXDLCTRLPARITYEN                1U
#define LW2080_CTRL_LWLINK_TL_INTEN_IDX_RXPROTOCOLEN                    2U
#define LW2080_CTRL_LWLINK_TL_INTEN_IDX_RXOVERFLOWEN                    3U
#define LW2080_CTRL_LWLINK_TL_INTEN_IDX_RXRAMDATAPARITYEN               4U
#define LW2080_CTRL_LWLINK_TL_INTEN_IDX_RXRAMHDRPARITYEN                5U
#define LW2080_CTRL_LWLINK_TL_INTEN_IDX_RXRESPEN                        6U
#define LW2080_CTRL_LWLINK_TL_INTEN_IDX_RXPOISONEN                      7U
#define LW2080_CTRL_LWLINK_TL_INTEN_IDX_TXRAMDATAPARITYEN               8U
#define LW2080_CTRL_LWLINK_TL_INTEN_IDX_TXRAMHDRPARITYEN                9U
#define LW2080_CTRL_LWLINK_TL_INTEN_IDX_DLFLOWPARITYEN                  10U
#define LW2080_CTRL_LWLINK_TL_INTEN_IDX_DLHDRPARITYEN                   12U
#define LW2080_CTRL_LWLINK_TL_INTEN_IDX_TXCREDITEN                      13U
#define LW2080_CTRL_LWLINK_TL_INTEN_IDX_MAX                             14U

/* LWLINK TL error fields */
#define LW2080_CTRL_LWLINK_TL_ERRLOG_IDX_RXDLDATAPARITYERR              0U
#define LW2080_CTRL_LWLINK_TL_ERRLOG_IDX_RXDLCTRLPARITYERR              1U
#define LW2080_CTRL_LWLINK_TL_ERRLOG_IDX_RXPROTOCOLERR                  2U
#define LW2080_CTRL_LWLINK_TL_ERRLOG_IDX_RXOVERFLOWERR                  3U
#define LW2080_CTRL_LWLINK_TL_ERRLOG_IDX_RXRAMDATAPARITYERR             4U
#define LW2080_CTRL_LWLINK_TL_ERRLOG_IDX_RXRAMHDRPARITYERR              5U
#define LW2080_CTRL_LWLINK_TL_ERRLOG_IDX_RXRESPERR                      6U
#define LW2080_CTRL_LWLINK_TL_ERRLOG_IDX_RXPOISONERR                    7U
#define LW2080_CTRL_LWLINK_TL_ERRLOG_IDX_TXRAMDATAPARITYERR             8U
#define LW2080_CTRL_LWLINK_TL_ERRLOG_IDX_TXRAMHDRPARITYERR              9U
#define LW2080_CTRL_LWLINK_TL_ERRLOG_IDX_DLFLOWPARITYERR                10U
#define LW2080_CTRL_LWLINK_TL_ERRLOG_IDX_DLHDRPARITYERR                 12U
#define LW2080_CTRL_LWLINK_TL_ERRLOG_IDX_TXCREDITERR                    13U
#define LW2080_CTRL_LWLINK_TL_ERRLOG_IDX_MAX                            14U

/* LWLINK DL speed status for sublink Tx*/
#define LW2080_CTRL_LWLINK_SL0_SLSM_STATUS_TX_PRIMARY_STATE_HS          (0x00000000U)
#define LW2080_CTRL_LWLINK_SL0_SLSM_STATUS_TX_PRIMARY_STATE_SINGLE_LANE (0x00000004U)
#define LW2080_CTRL_LWLINK_SL0_SLSM_STATUS_TX_PRIMARY_STATE_TRAIN       (0x00000005U)
#define LW2080_CTRL_LWLINK_SL0_SLSM_STATUS_TX_PRIMARY_STATE_SAFE        (0x00000006U)
#define LW2080_CTRL_LWLINK_SL0_SLSM_STATUS_TX_PRIMARY_STATE_OFF         (0x00000007U)

/* LWLINK DL speed status for sublink Rx*/
#define LW2080_CTRL_LWLINK_SL1_SLSM_STATUS_RX_PRIMARY_STATE_HS          (0x00000000U)
#define LW2080_CTRL_LWLINK_SL1_SLSM_STATUS_RX_PRIMARY_STATE_SINGLE_LANE (0x00000004U)
#define LW2080_CTRL_LWLINK_SL1_SLSM_STATUS_RX_PRIMARY_STATE_TRAIN       (0x00000005U)
#define LW2080_CTRL_LWLINK_SL1_SLSM_STATUS_RX_PRIMARY_STATE_SAFE        (0x00000006U)
#define LW2080_CTRL_LWLINK_SL1_SLSM_STATUS_RX_PRIMARY_STATE_OFF         (0x00000007U)

#define LW2080_CTRL_LWLINK_MAX_IOCTRLS                                  3U
/*
 *   LW2080_CTRL_LWLINK_GET_ERR_INFO_PARAMS
 *
 *   linkMask
 *     Returns the mask of links enabled
 *
 *   linkErrInfo
 *     Returns the error information for all the links
 *
 *   ioctrlMask
 *     Returns the mask of ioctrls
 *
 *   commonErrInfo
 *     Returns the error information common to each IOCTRL
 */
#define LW2080_CTRL_LWLINK_GET_ERR_INFO_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW2080_CTRL_LWLINK_GET_ERR_INFO_PARAMS {
    LwU32                              linkMask;
    LW2080_CTRL_LWLINK_ERR_INFO        linkErrInfo[LW2080_CTRL_LWLINK_MAX_LINKS];
    LwU32                              ioctrlMask;
    LW2080_CTRL_LWLINK_COMMON_ERR_INFO commonErrInfo[LW2080_CTRL_LWLINK_MAX_IOCTRLS];
} LW2080_CTRL_LWLINK_GET_ERR_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_LWLINK_GET_ERR_INFO
 *     This command is used to query the LWLINK error information
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_LWLINK_GET_ERR_INFO                 (0x20803003U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_GET_ERR_INFO_PARAMS_MESSAGE_ID" */

/*
 * APIs for getting LWLink counters
 */

// These are the bitmask definitions for different counter types

#define LW2080_CTRL_LWLINK_COUNTER_ILWALID                  0x00000000U

#define LW2080_CTRL_LWLINK_COUNTER_TL_TX0                   0x00000001U
#define LW2080_CTRL_LWLINK_COUNTER_TL_TX1                   0x00000002U
#define LW2080_CTRL_LWLINK_COUNTER_TL_RX0                   0x00000004U
#define LW2080_CTRL_LWLINK_COUNTER_TL_RX1                   0x00000008U

#define LW2080_CTRL_LWLINK_LP_COUNTERS_DL                   0x00000010U

#define LW2080_CTRL_LWLINK_COUNTER_DL_RX_ERR_ECC_LANE_L(i)      (1 << (i + 8))
#define LW2080_CTRL_LWLINK_COUNTER_DL_RX_ERR_ECC_LANE__SIZE 4U
#define LW2080_CTRL_LWLINK_COUNTER_DL_RX_ERR_ECC_LANE_L0    0x00000100U
#define LW2080_CTRL_LWLINK_COUNTER_DL_RX_ERR_ECC_LANE_L1    0x00000200U
#define LW2080_CTRL_LWLINK_COUNTER_DL_RX_ERR_ECC_LANE_L2    0x00000400U
#define LW2080_CTRL_LWLINK_COUNTER_DL_RX_ERR_ECC_LANE_L3    0x00000800U

#define LW2080_CTRL_LWLINK_COUNTER_DL_RX_ERR_CRC_FLIT       0x00010000U

#define LW2080_CTRL_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L(i)      (1 << (i + 17))
#define LW2080_CTRL_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE__SIZE 8U
#define LW2080_CTRL_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L0    0x00020000U
#define LW2080_CTRL_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L1    0x00040000U
#define LW2080_CTRL_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L2    0x00080000U
#define LW2080_CTRL_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L3    0x00100000U
#define LW2080_CTRL_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L4    0x00200000U
#define LW2080_CTRL_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L5    0x00400000U
#define LW2080_CTRL_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L6    0x00800000U
#define LW2080_CTRL_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L7    0x01000000U

#define LW2080_CTRL_LWLINK_COUNTER_DL_TX_ERR_REPLAY         0x02000000U
#define LW2080_CTRL_LWLINK_COUNTER_DL_TX_ERR_RECOVERY       0x04000000U

#define LW2080_CTRL_LWLINK_COUNTER_DL_RX_ERR_REPLAY         0x08000000U

#define LW2080_CTRL_LWLINK_COUNTER_DL_RX_ERR_CRC_MASKED     0x10000000U

/*
 * Note that COUNTER_MAX_TYPES will need to be updated each time
 * a new counter type gets added to the list above.
 *
 */
#define LW2080_CTRL_LWLINK_COUNTER_MAX_TYPES                32U

/*
 * LW2080_CTRL_CMD_LWLINK_GET_COUNTERS
 *  This command gets the counts for different counter types.
 *
 * [in] counterMask
 *     Mask of counter types to be queried
 *     One of LW2080_CTRL_LWLINK_COUNTERS_TYPE_* macros
 *
 * [in] linkMask
 *     Mask of links to be queried
 *
 * [out] counters
 *     Counter value returned
 *
 *     [out] bTx0TlCounterOverflow
 *      This boolean is set to LW_TRUE if TX Counter 0 has rolled over.
 *
 *     [out] bTx1TlCounterOverflow
 *      This boolean is set to LW_TRUE if TX Counter 1 has rolled over.
 *
 *     [out] bRx0TlCounterOverflow
 *      This boolean is set to LW_TRUE if RX Counter 0 has rolled over.
 *
 *     [out] bRx1TlCounterOverflow
 *      This boolean is set to LW_TRUE if RX Counter 1 has rolled over.
 *
 *     [out] value 
 *      This array contains the error counts for each error type as requested from
 *      the counterMask. The array indexes correspond to the mask bits one-to-one.
 */
typedef struct LW2080_CTRL_LWLINK_GET_COUNTERS_VALUES {
    LwBool bTx0TlCounterOverflow;
    LwBool bTx1TlCounterOverflow;
    LwBool bRx0TlCounterOverflow;
    LwBool bRx1TlCounterOverflow;
    LW_DECLARE_ALIGNED(LwU64 value[LW2080_CTRL_LWLINK_COUNTER_MAX_TYPES], 8);
} LW2080_CTRL_LWLINK_GET_COUNTERS_VALUES;

#define LW2080_CTRL_LWLINK_GET_COUNTERS_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW2080_CTRL_LWLINK_GET_COUNTERS_PARAMS {
    LwU32 counterMask;
    LW_DECLARE_ALIGNED(LwU64 linkMask, 8);
    LW_DECLARE_ALIGNED(LW2080_CTRL_LWLINK_GET_COUNTERS_VALUES counters[LW2080_CTRL_LWLINK_MAX_LINKS], 8);
} LW2080_CTRL_LWLINK_GET_COUNTERS_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_GET_COUNTERS   (0x20803004U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_GET_COUNTERS_PARAMS_MESSAGE_ID" */


/*
 * LW2080_CTRL_CMD_LWLINK_CLEAR_COUNTERS
 *  This command clears/resets the counters for the specified types.
 *
 * [in] linkMask
 *  This parameter specifies for which links we want to clear the 
 *  counters.
 *
 * [in] counterMask
 *  This parameter specifies the input mask for desired counters to be
 *  cleared. Note that all counters cannot be cleared.
 *
 *  NOTE: Bug# 2098529: On Turing all DL errors and LP counters are cleared
 *        together. They cannot be cleared individually per error type. RM
 *        would possibly move to a new API on Ampere and beyond
 */

#define LW2080_CTRL_CMD_LWLINK_CLEAR_COUNTERS (0x20803005U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_CLEAR_COUNTERS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_LWLINK_CLEAR_COUNTERS_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW2080_CTRL_LWLINK_CLEAR_COUNTERS_PARAMS {
    LwU32 counterMask;
    LW_DECLARE_ALIGNED(LwU64 linkMask, 8);
} LW2080_CTRL_LWLINK_CLEAR_COUNTERS_PARAMS;

/*
 * LW2080_CTRL_CMD_LWLINK_INJECT_ERROR
 *  This command causes all the same actions to occur as if the related
 *  error were to occur, either fatal or recoverable.
 *
 * [in] linkMask        size: 32 bits
 *  Controls which links to apply error injection to.
 * [in] bFatal
 *  This parameter specifies that the error should be fatal.
 *
 */
#define LW2080_CTRL_CMD_LWLINK_INJECT_ERROR (0x20803006U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_INJECT_ERROR_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_LWLINK_INJECT_ERROR_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW2080_CTRL_LWLINK_INJECT_ERROR_PARAMS {
    LwU32  linkMask;
    LwBool bFatalError;
} LW2080_CTRL_LWLINK_INJECT_ERROR_PARAMS;

/* LWLINK unit list - to be used with error notifiers */
#define LW2080_CTRL_LWLINK_UNIT_DL                  0x01U
#define LW2080_CTRL_LWLINK_UNIT_TL                  0x02U
#define LW2080_CTRL_LWLINK_UNIT_TLC_RX_0            0x03U
#define LW2080_CTRL_LWLINK_UNIT_TLC_RX_1            0x04U
#define LW2080_CTRL_LWLINK_UNIT_TLC_TX_0            0x05U
#define LW2080_CTRL_LWLINK_UNIT_MIF_RX_0            0x06U
#define LW2080_CTRL_LWLINK_UNIT_MIF_TX_0            0x07U

/*
 * LW2080_CTRL_CMD_LWLINK_GET_ERROR_RECOVERIES
 *  This command gets the number of successful error recoveries
 *
 * [in]  linkMask        size: 32 bits
 *    This parameter controls which links to get recoveries for.
 * [out] numRecoveries
 *    This parameter specifies the number of successful per link error recoveries
 */
#define LW2080_CTRL_CMD_LWLINK_GET_ERROR_RECOVERIES (0x20803007U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_CMD_LWLINK_GET_ERROR_RECOVERIES_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_LWLINK_GET_ERROR_RECOVERIES_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW2080_CTRL_CMD_LWLINK_GET_ERROR_RECOVERIES_PARAMS {
    LwU32 linkMask;
    LwU32 numRecoveries[LW2080_CTRL_LWLINK_MAX_LINKS];
} LW2080_CTRL_CMD_LWLINK_GET_ERROR_RECOVERIES_PARAMS;

/*
 * LW2080_CTRL_CMD_LWLINK_GET_LINK_LAST_ERROR_REMOTE_TYPE
 *
 * This command queries the remote endpoint type of the link recorded at the
 * time the last error oclwrred on the link.
 *
 *   [in] linkId
 *     This parameter specifies the link to get the last remote endpoint type
 *     recorded for.
 *
 *   [out] remoteType
 *     This parameter returns the remote endpoint type of the link recorded at
 *     the time the last error oclwrred on the link. Possible values are:
 *       LW2080_CTRL_LWLINK_DEVICE_INFO_DEVICE_TYPE_NONE
 *         The link is not connected to an active remote endpoint.
 *       LW2080_CTRL_LWLINK_DEVICE_INFO_DEVICE_TYPE_GPU
 *         The remote endpoint of the link is a peer GPU.
 *       LW2080_CTRL_LWLINK_DEVICE_INFO_DEVICE_TYPE_NPU
 *         The remote endpoint of the link is the host system (e.g., an NPU
 *         on IBM POWER platforms).
 *       LW2080_CTRL_LWLINK_DEVICE_INFO_DEVICE_TYPE_TEGRA
 *         The remote endpoint of the link a cheetah device
 *
 * Possible return status values are:
 *   LW_OK
 *     If the remoteType parameter value is valid upon return.
 *   LW_ERR_ILWALID_ARGUMENT
 *     If the linkId parameter does not specify a valid link.
 *   LW_ERR_NOT_SUPPORTED
 *     If LWLINK is not supported on this GPU or the remote endpoint type is
 *     not recorded in non-volatile storage.
 */
#define LW2080_CTRL_CMD_LWLINK_GET_LINK_LAST_ERROR_REMOTE_TYPE (0x20803008U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_GET_LINK_LAST_ERROR_REMOTE_TYPE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_LWLINK_GET_LINK_LAST_ERROR_REMOTE_TYPE_PARAMS_MESSAGE_ID (0x8U)

typedef struct LW2080_CTRL_LWLINK_GET_LINK_LAST_ERROR_REMOTE_TYPE_PARAMS {
    LwU32 linkId;
    LwU32 remoteType;
} LW2080_CTRL_LWLINK_GET_LINK_LAST_ERROR_REMOTE_TYPE_PARAMS;

/*
 * LW2080_CTRL_CMD_LWLINK_GET_LINK_FATAL_ERROR_COUNTS
 *
 * This command queries the number of each type of fatal errors that have
 * oclwrred on the given link.
 *
 *   [in] linkId
 *     This parameter specifies the link to get the fatal error information
 *     for.
 *
 *   [out] supportedCounts
 *     This parameter identifies which counts in the fatalErrorCounts array
 *     are valid for the given link. A bit set in this field means that the
 *     corresponding index is valid in the fatalErrorCounts array.
 *
 *   [out] fatalErrorCounts
 *     This parameter returns an array of 8-bit counts, one for each type of
 *     fatal error that can occur on the link. The valid indices of this array
 *     are:
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL(C)_RX_DL_DATA_PARITY
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL(C)_RX_DL_CTRL_PARITY
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL_RX_PROTOCOL
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL_RX_OVERFLOW
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL(C)_RX_RAM_DATA_PARITY
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL(C)_RX_RAM_HDR_PARITY
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL_RX_RESP
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL_RX_POISON
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_DATA_POISONED_PKT_RCVD
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL(C)_TX_RAM_DATA_PARITY
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL(C)_TX_RAM_HDR_PARITY
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL_TX_CREDIT
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL_DL_FLOW_CTRL_PARITY
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_TX_DL_FLOW_CTRL_PARITY
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL_DL_HDR_PARITY
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_DL_TX_RECOVERY_LONG
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_DL_TX_FAULT_RAM
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_DL_TX_FAULT_INTERFACE
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_DL_TX_FAULT_SUBLINK_CHANGE
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_DL_RX_FAULT_SUBLINK_CHANGE
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_DL_RX_FAULT_DL_PROTOCOL
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_DL_LTSSM_FAULT
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_DL_HDR_PARITY
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_ILWALID_AE_FLIT_RCVD
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_ILWALID_BE_FLIT_RCVD
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_ILWALID_ADDR_ALIGN
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_PKT_LEN
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_RSVD_CMD_ENC
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_RSVD_DAT_LEN_ENC
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_RSVD_ADDR_TYPE
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_RSVD_RSP_STATUS
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_RSVD_PKT_STATUS
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_RSVD_CACHE_ATTR_ENC_IN_PROBE_REQ
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_RSVD_CACHE_ATTR_ENC_IN_PROBE_RESP
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_DAT_LEN_GT_ATOMIC_REQ_MAX_SIZE
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_DAT_LEN_GT_RMW_REQ_MAX_SIZE
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_DAT_LEN_LT_ATR_RESP_MIN_SIZE
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_ILWALID_PO_FOR_CACHE_ATTR
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_ILWALID_COMPRESSED_RESP
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_RESP_STATUS_TARGET
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_RESP_STATUS_UNSUPPORTED_REQUEST
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_HDR_OVERFLOW
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_DATA_OVERFLOW
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_STOMPED_PKT_RCVD
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_CORRECTABLE_INTERNAL
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_UNSUPPORTED_VC_OVERFLOW
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_UNSUPPORTED_LWLINK_CREDIT_RELEASE
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_UNSUPPORTED_NCISOC_CREDIT_RELEASE
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_TX_HDR_CREDIT_OVERFLOW
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_TX_DATA_CREDIT_OVERFLOW
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_TX_DL_REPLAY_CREDIT_OVERFLOW
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_TX_UNSUPPORTED_VC_OVERFLOW
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_TX_STOMPED_PKT_SENT
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_TX_DATA_POISONED_PKT_SENT
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_TX_RESP_STATUS_TARGET
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_TX_RESP_STATUS_UNSUPPORTED_REQUEST
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_MIF_RX_RAM_DATA_PARITY
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_MIF_RX_RAM_HDR_PARITY
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_MIF_TX_RAM_DATA_PARITY
 *       LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_MIF_TX_RAM_HDR_PARITY
 *
 * Possible return status values are:
 *   LW_OK
 *     If the values in the fatalErrorCounts array are valid upon return.
 *   LW_ERR_ILWALID_ARGUMENT
 *     If the linkId parameter does not specify a valid link.
 *   LW_ERR_NOT_SUPPORTED
 *     If LWLINK is not supported on this GPU or aggregate LWLINK fatal error
 *     counts are not recorded in non-volatile storage.
 */
#define LW2080_CTRL_CMD_LWLINK_GET_LINK_FATAL_ERROR_COUNTS                           (0x20803009U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_GET_LINK_FATAL_ERROR_COUNTS_PARAMS_MESSAGE_ID" */

/*
 * LWLink 1 Fatal Error Types
 */
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL_RX_DL_DATA_PARITY                     0U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL_RX_DL_CTRL_PARITY                     1U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL_RX_PROTOCOL                           2U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL_RX_OVERFLOW                           3U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL_RX_RAM_DATA_PARITY                    4U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL_RX_RAM_HDR_PARITY                     5U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL_RX_RESP                               6U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL_RX_POISON                             7U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL_TX_RAM_DATA_PARITY                    8U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL_TX_RAM_HDR_PARITY                     9U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL_TX_CREDIT                             10U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL_DL_FLOW_CTRL_PARITY                   11U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TL_DL_HDR_PARITY                         12U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_DL_TX_RECOVERY_LONG                      13U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_DL_TX_FAULT_RAM                          14U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_DL_TX_FAULT_INTERFACE                    15U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_DL_TX_FAULT_SUBLINK_CHANGE               16U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_DL_RX_FAULT_SUBLINK_CHANGE               17U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_DL_RX_FAULT_DL_PROTOCOL                  18U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_DL_LTSSM_FAULT                           19U

/*
 * LWLink 2 Fatal Error Types
 */
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_DL_DATA_PARITY                    0U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_DL_CTRL_PARITY                    1U
// No direct equivalent to:                 TL_RX_PROTOCOL                            2
// No direct equivalent to:                 TL_RX_OVERFLOW                            3
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_RAM_DATA_PARITY                   4U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_RAM_HDR_PARITY                    5U
// No direct equivalent to:                 TL_RX_RESP                                6
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_DATA_POISONED_PKT_RCVD            7U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_TX_RAM_DATA_PARITY                   8U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_TX_RAM_HDR_PARITY                    9U
// No direct equivalent to:                 TL_TX_CREDIT                             10
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_TX_DL_FLOW_CONTROL_PARITY            11U
// No direct equivalent to:                 TL_DL_HDR_PARITY                         12
// Identical to LWLink 1:                   DL_TX_RECOVERY_LONG                      13
// Identical to LWLink 1:                   DL_TX_FAULT_RAM                          14
// Identical to LWLink 1:                   DL_TX_FAULT_INTERFACE                    15
// Identical to LWLink 1:                   DL_TX_FAULT_SUBLINK_CHANGE               16
// Identical to LWLink 1:                   DL_RX_FAULT_SUBLINK_CHANGE               17
// Identical to LWLink 1:                   DL_RX_FAULT_DL_PROTOCOL                  18
// Identical to LWLink 1:                   DL_LTSSM_FAULT                           19
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_DL_HDR_PARITY                     20U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_ILWALID_AE_FLIT_RCVD              21U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_ILWALID_BE_FLIT_RCVD              22U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_ILWALID_ADDR_ALIGN                23U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_PKT_LEN                           24U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_RSVD_CMD_ENC                      25U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_RSVD_DAT_LEN_ENC                  26U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_RSVD_ADDR_TYPE                    27U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_RSVD_RSP_STATUS                   28U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_RSVD_PKT_STATUS                   29U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_RSVD_CACHE_ATTR_ENC_IN_PROBE_REQ  30U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_RSVD_CACHE_ATTR_ENC_IN_PROBE_RESP 31U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_DAT_LEN_GT_ATOMIC_REQ_MAX_SIZE    32U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_DAT_LEN_GT_RMW_REQ_MAX_SIZE       33U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_DAT_LEN_LT_ATR_RESP_MIN_SIZE      34U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_ILWALID_PO_FOR_CACHE_ATTR         35U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_ILWALID_COMPRESSED_RESP           36U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_RESP_STATUS_TARGET                37U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_RESP_STATUS_UNSUPPORTED_REQUEST   38U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_HDR_OVERFLOW                      39U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_DATA_OVERFLOW                     40U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_STOMPED_PKT_RCVD                  41U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_CORRECTABLE_INTERNAL              42U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_UNSUPPORTED_VC_OVERFLOW           43U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_UNSUPPORTED_LWLINK_CREDIT_RELEASE 44U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_UNSUPPORTED_NCISOC_CREDIT_RELEASE 45U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_TX_HDR_CREDIT_OVERFLOW               46U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_TX_DATA_CREDIT_OVERFLOW              47U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_TX_DL_REPLAY_CREDIT_OVERFLOW         48U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_TX_UNSUPPORTED_VC_OVERFLOW           49U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_TX_STOMPED_PKT_SENT                  50U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_TX_DATA_POISONED_PKT_SENT            51U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_TX_RESP_STATUS_TARGET                52U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_TX_RESP_STATUS_UNSUPPORTED_REQUEST   53U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_MIF_RX_RAM_DATA_PARITY                   54U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_MIF_RX_RAM_HDR_PARITY                    55U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_MIF_TX_RAM_DATA_PARITY                   56U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_MIF_TX_RAM_HDR_PARITY                    57U

/*
 * LWLink 3 Fatal Error Types
 */
// Identical to LWLink 2:                   TLC_RX_DL_DATA_PARITY                     0
// Identical to LWLink 2:                   TLC_RX_DL_CTRL_PARITY                     1
// No direct equivalent to:                 TL_RX_PROTOCOL                            2
// No direct equivalent to:                 TL_RX_OVERFLOW                            3
// No direct equivalent to:                 TLC_RX_RAM_DATA_PARITY                    4
// No direct equivalent to:                 RX_RAM_HDR_PARITY                         5
// No direct equivalent to:                 TL_RX_RESP                                6
// No direct equivalent to:                 TLC_RX_DATA_POISONED_PKT_RCVD             7
// No direct equivalent to:                 TLC_TX_RAM_DATA_PARITY                    8
// No direct equivalent to:                 TLC_TX_RAM_HDR_PARITY                     9
// No direct equivalent to:                 TL_TX_CREDIT                             10
// Identical to LWLink 2:                   TLC_TX_DL_FLOW_CONTROL_PARITY            11
// No direct equivalent to:                 TL_DL_HDR_PARITY                         12
// No direct equivalent to:                 DL_TX_RECOVERY_LONG                      13
// Identical to LWLink 1:                   DL_TX_FAULT_RAM                          14
// Identical to LWLink 1:                   DL_TX_FAULT_INTERFACE                    15
// Identical to LWLink 1:                   DL_TX_FAULT_SUBLINK_CHANGE               16
// Identical to LWLink 1:                   DL_RX_FAULT_SUBLINK_CHANGE               17
// Identical to LWLink 1:                   DL_RX_FAULT_DL_PROTOCOL                  18
// No direct equivalent to:                 DL_LTSSM_FAULT                           19
// Identical to LWLink 2:                   TLC_RX_DL_HDR_PARITY                     20
// Identical to LWLink 2:                   TLC_RX_ILWALID_AE_FLIT_RCVD              21
// Identical to LWLink 2:                   TLC_RX_ILWALID_BE_FLIT_RCVD              22
// Identical to LWLink 2:                   TLC_RX_ILWALID_ADDR_ALIGN                23
// Identical to LWLink 2:                   TLC_RX_PKT_LEN                           24
// Identical to LWLink 2:                   TLC_RX_RSVD_CMD_ENC                      25
// Identical to LWLink 2:                   TLC_RX_RSVD_DAT_LEN_ENC                  26
// No direct equivalent to:                 TLC_RX_RSVD_ADDR_TYPE                    27
// No direct equivalent to:                 TLC_RX_RSVD_RSP_STATUS                   28
// Identical to LWLink 2:                   TLC_RX_RSVD_PKT_STATUS                   29
// Identical to LWLink 2:                   TLC_RX_RSVD_CACHE_ATTR_ENC_IN_PROBE_REQ  30
// Identical to LWLink 2:                   TLC_RX_RSVD_CACHE_ATTR_ENC_IN_PROBE_RESP 31
// No direct equivalent to:                 TLC_RX_DAT_LEN_GT_ATOMIC_REQ_MAX_SIZE    32
// Identical to LWLink 2:                   TLC_RX_DAT_LEN_GT_RMW_REQ_MAX_SIZE       33
// Identical to LWLink 2:                   TLC_RX_DAT_LEN_LT_ATR_RESP_MIN_SIZE      34
// Identical to LWLink 2:                   TLC_RX_ILWALID_PO_FOR_CACHE_ATTR         35
// Identical to LWLink 2:                   TLC_RX_ILWALID_COMPRESSED_RESP           36
// No direct equivalent to:                 TLC_RX_RESP_STATUS_TARGET                37
// No direct equivalent to:                 TLC_RX_RESP_STATUS_UNSUPPORTED_REQUEST   38
// Identical to LWLink 2:                   TLC_RX_HDR_OVERFLOW                      39
// Identical to LWLink 2:                   TLC_RX_DATA_OVERFLOW                     40
// Identical to LWLink 2:                   TLC_RX_STOMPED_PKT_RCVD                  41
// No direct equivalent to:                 TLC_RX_CORRECTABLE_INTERNAL              42
// No direct equivalent to:                 TLC_RX_UNSUPPORTED_VC_OVERFLOW           43
// No direct equivalent to:                 TLC_RX_UNSUPPORTED_LWLINK_CREDIT_RELEASE 44
// No direct equivalent to:                 TLC_RX_UNSUPPORTED_NCISOC_CREDIT_RELEASE 45
// No direct equivalent to:                 TLC_TX_HDR_CREDIT_OVERFLOW               46
// No direct equivalent to:                 TLC_TX_DATA_CREDIT_OVERFLOW              47
// No direct equivalent to:                 TLC_TX_DL_REPLAY_CREDIT_OVERFLOW         48
// No direct equivalent to:                 TLC_TX_UNSUPPORTED_VC_OVERFLOW           49
// No direct equivalent to:                 TLC_TX_STOMPED_PKT_SENT                  50
// No direct equivalent to:                 TLC_TX_DATA_POISONED_PKT_SENT            51
// No direct equivalent to:                 TLC_TX_RESP_STATUS_TARGET                52
// No direct equivalent to:                 TLC_TX_RESP_STATUS_UNSUPPORTED_REQUEST   53
// No direct equivalent to:                 MIF_RX_RAM_DATA_PARITY                   54
// No direct equivalent to:                 MIF_RX_RAM_HDR_PARITY                    55
// No direct equivalent to:                 MIF_TX_RAM_DATA_PARITY                   56
// No direct equivalent to:                 MIF_TX_RAM_HDR_PARITY                    57
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_RX_ILWALID_COLLAPSED_RESPONSE        58U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_TX_NCISOC_HDR_ECC_DBE                59U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_TLC_TX_NCISOC_PARITY                     60U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_DL_LTSSM_FAULT_UP                        61U
#define LW2080_CTRL_LWLINK_FATAL_ERROR_TYPE_DL_LTSSM_FAULT_DOWN                      62U

#define LW2080_CTRL_LWLINK_NUM_FATAL_ERROR_TYPES                                     63U

#define LW2080_CTRL_LWLINK_IS_FATAL_ERROR_COUNT_VALID(count, supportedCounts)    \
    (!!((supportedCounts) & LWBIT64(count)))

#define LW2080_CTRL_LWLINK_GET_LINK_FATAL_ERROR_COUNTS_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW2080_CTRL_LWLINK_GET_LINK_FATAL_ERROR_COUNTS_PARAMS {
    LwU32 linkId;
    LW_DECLARE_ALIGNED(LwU64 supportedCounts, 8);
    LwU8  fatalErrorCounts[LW2080_CTRL_LWLINK_NUM_FATAL_ERROR_TYPES];
} LW2080_CTRL_LWLINK_GET_LINK_FATAL_ERROR_COUNTS_PARAMS;

/*
 * LW2080_CTRL_CMD_LWLINK_GET_LINK_NONFATAL_ERROR_RATES
 *
 * This command queries recent non-fatal error rates for the given link.
 *
 * The error rates specify the maximum number of errors per minute recorded
 * for the given link within a 24-hour period for daily maximums or a 30-day
 * period for monthly maximums.
 *
 *   [in] linkId
 *     This parameter specifies the link to get the nonfatal error information
 *     for.
 *
 *   [out] numDailyMaxNonfatalErrorRates
 *     This parameter returns the number of valid nonfatal error rate entries
 *     in the dailyMaxNonfatalErrorRates parameter.
 *
 *   [out] dailyMaxNonfatalErrorRates
 *     This parameter returns maximum nonfatal error rate entries recorded
 *     over the last few 24-hour periods. For example, index 0 contains the
 *     maximum nonfatal error rate recorded in the current day, index 1
 *     contains the maximum nonfatal error rate recorded yesterday ago, etc.
 *
 *   [out] numMonthlyMaxNonfatalErrorRates
 *     This parameter returns the number of valid nonfatal error rate entries
 *     in the monthlyMaxNonfatalErrorRates parameter.
 *
 *   [out] monthlyMaxNonfatalErrorRates
 *     THis parameter returns maximum nonfatal error rate entries recorded
 *     over the last few 30-day periods. For example, index 0 contains the
 *     maximum nonfatal error rate recorded in the current month, index 1
 *     contains the maximum nonfatal error recorded last month, etc.
 *
 * Possible status values returned are:
 *   LW_OK
 *     If any nonfatal error rates are valid upon return.
 *   LW_ERR_ILWALID_ARGUMENT
 *     If the linkId parameter does not specify a valid link.
 *   LW_ERR_NOT_SUPPORTED
 *     If LWLINK is not supported on this GPU or LWLINK nonfatal error rates
 *     are not recorded in non-volatile storage.
 */
#define LW2080_CTRL_CMD_LWLINK_GET_LINK_NONFATAL_ERROR_RATES (0x2080300aU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_GET_LINK_NONFATAL_ERROR_RATES_PARAMS_MESSAGE_ID" */

typedef struct LW2080_CTRL_LWLINK_NONFATAL_ERROR_RATE {
    LwU32 errorsPerMinute;
    LwU32 timestamp;
} LW2080_CTRL_LWLINK_NONFATAL_ERROR_RATE;

#define LW2080_CTRL_LWLINK_NONFATAL_ERROR_RATE_ENTRIES 5U

#define LW2080_CTRL_LWLINK_GET_LINK_NONFATAL_ERROR_RATES_PARAMS_MESSAGE_ID (0xAU)

typedef struct LW2080_CTRL_LWLINK_GET_LINK_NONFATAL_ERROR_RATES_PARAMS {
    LwU32                                  linkId;
    LwU32                                  numDailyMaxNonfatalErrorRates;
    LW2080_CTRL_LWLINK_NONFATAL_ERROR_RATE dailyMaxNonfatalErrorRates[LW2080_CTRL_LWLINK_NONFATAL_ERROR_RATE_ENTRIES];
    LwU32                                  numMonthlyMaxNonfatalErrorRates;
    LW2080_CTRL_LWLINK_NONFATAL_ERROR_RATE monthlyMaxNonfatalErrorRates[LW2080_CTRL_LWLINK_NONFATAL_ERROR_RATE_ENTRIES];
} LW2080_CTRL_LWLINK_GET_LINK_NONFATAL_ERROR_RATES_PARAMS;

/*
 * LW2080_CTRL_CMD_LWLINK_SET_ERROR_INJECTION_MODE
 *
 * This command sets the injection mode so that error handling and error
 * logging software can be aware that errors cropping up on links are
 * intentional and not due to HW failures.
 *
 *   [in] bEnabled
 *     This parameter specifies whether injection mode should be enabled or
 *     disabled.
 *
 * Possible status values returned are:
 *   LW_OK
 *     If injection mode is enabled or disabled according to the parameters.
 *   LW_ERR_NOT_SUPPORTED
 *     If LWLINK is not supported on this GPU.
 */
#define LW2080_CTRL_CMD_LWLINK_SET_ERROR_INJECTION_MODE (0x2080300bU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_SET_ERROR_INJECTION_MODE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_LWLINK_SET_ERROR_INJECTION_MODE_PARAMS_MESSAGE_ID (0xBU)

typedef struct LW2080_CTRL_LWLINK_SET_ERROR_INJECTION_MODE_PARAMS {
    LwBool bEnabled;
} LW2080_CTRL_LWLINK_SET_ERROR_INJECTION_MODE_PARAMS;

/*
 * LW2080_CTRL_CMD_LWLINK_SETUP_EOM
 *
 * This command passes a packed 32bit params value to LW_PMINION_MISC_0_SCRATCH_SWRW_0
 * and then issues an EOM DLCMD to minion for the desired link. Only one DLCMD 
 * at a time can be issued to any given link.
 *
 * Params Packing is specified in Minion IAS
 */
#define LW2080_CTRL_CMD_LWLINK_SETUP_EOM (0x2080300lw) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_CMD_LWLINK_SETUP_EOM_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_LWLINK_SETUP_EOM_PARAMS_MESSAGE_ID (0xLW)

typedef struct LW2080_CTRL_CMD_LWLINK_SETUP_EOM_PARAMS {
    LwU8  linkId;
    LwU32 params;
} LW2080_CTRL_CMD_LWLINK_SETUP_EOM_PARAMS;

/*
 * LW2080_CTRL_CMD_LWLINK_SET_POWER_STATE
 *
 * This command sets the mask of links associated with the GPU
 * to a target power state
 *
 * [in] linkMask
 *     Mask of links that will be put to desired power state
 *     Note: In Turing RM supports only tansitions into/out of L2
 * [in] powerState
 *     Target power state to which the links will transition
 *     This can be any one of LW2080_CTRL_LWLINK_POWER_STATE_* states
 *
 * Possible status values returned are:
 *   LW_OK
 *     If all links transitioned successfully to the target state
 *   LW_ERR_NOT_SUPPORTED
 *     If LWLINK is not supported on the chip or if the power state
 *     is not enabled on the chip
 *   LW_ERR_ILWALID_ARGUMENT
 *     If the any of the links in the mask is not enabled
 *   LW_ERR_ILWALID_REQUEST
 *     If the power state transition is not supported
 *   LW_WARN_MORE_PROCESSING_REQUIRED
 *      Link has received the request for the power transition
 *      The transition will happen when the remote end also agrees
 *
 *  Note: Lwrrently only L0->L2 and L2->L0 is supported
 */
#define LW2080_CTRL_CMD_LWLINK_SET_POWER_STATE (0x2080300dU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_SET_POWER_STATE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_LWLINK_SET_POWER_STATE_PARAMS_MESSAGE_ID (0xDU)

typedef struct LW2080_CTRL_LWLINK_SET_POWER_STATE_PARAMS {
    LwU32 linkMask;
    LwU32 powerState;
} LW2080_CTRL_LWLINK_SET_POWER_STATE_PARAMS;

// LWLink Power States
#define LW2080_CTRL_LWLINK_POWER_STATE_L0      (0x00U)
#define LW2080_CTRL_LWLINK_POWER_STATE_L1      (0x01U)
#define LW2080_CTRL_LWLINK_POWER_STATE_L2      (0x02U)
#define LW2080_CTRL_LWLINK_POWER_STATE_L3      (0x03U)

/*
 * LW2080_CTRL_CMD_LWLINK_GET_POWER_STATE
 *
 * This command gets the power state of a link associated
 * with the GPU
 *
 * [in] linkId
 *     Link whose power state is being requested
 * [out] powerState
 *     Current power state of the link
 *     Is any one the LW2080_CTRL_LWLINK_POWER_STATE_* states
 *
 * Possible status values returned are:
 *   LW_OK
 *     If the power state is retrieved successfully
 *   LW_ERR_NOT_SUPPORTED
 *     If LWLINK is not supported on the chip
 *   LW_ERR_ILWALID_ARGUMENT
 *     If the link is not enabled on the GPU
 *   LW_ERR_ILWALID_STATE
 *     If the link is in an invalid state
 */
#define LW2080_CTRL_CMD_LWLINK_GET_POWER_STATE (0x2080300eU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_GET_POWER_STATE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_LWLINK_GET_POWER_STATE_PARAMS_MESSAGE_ID (0xEU)

typedef struct LW2080_CTRL_LWLINK_GET_POWER_STATE_PARAMS {
    LwU32 linkId;
    LwU32 powerState;
} LW2080_CTRL_LWLINK_GET_POWER_STATE_PARAMS;

/*
 * LW2080_CTRL_CMD_LWLINK_INJECT_TLC_ERROR
 *
 * This command injects TLC_*_REPORT_INJECT error. An RM interrupt
 * will be triggered after injection. Lwrrently the injection call
 * only deals with HW_ERR, UR_ERR, PRIV_ERR in TX_SYS and RX_LNK devices
 *
 * [in] linkId
 *     Link whose power state is being requested.
 * [in] errorType
 *     error type that needs to be injected.
 * [in] device
 *     The device this injection is intended for.
 * [in] bBroadcast
 *     Whether the link report error should be fired in multiple links.

 * Possible status values returned are:
 *   LW_OK
 *     If the injection succeeds.
 *   LW_ERR_NOT_SUPPORTED
 *     If the error type of LWLINK is not supported on the chip
 */
#define LW2080_CTRL_CMD_LWLINK_INJECT_TLC_ERROR (0x2080300fU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_INJECT_TLC_ERROR_PARAMS_MESSAGE_ID" */

typedef enum LW2080_CTRL_LWLINK_INJECT_TLC_ERROR_DEVICE {
    TLC_RX_LNK = 0,
    TLC_TX_SYS = 1,
} LW2080_CTRL_LWLINK_INJECT_TLC_ERROR_DEVICE;

typedef enum LW2080_CTRL_LWLINK_INJECT_TLC_TX_SYS_REPORT_ERROR_TYPE {
    TX_SYS_TX_RSP_STATUS_HW_ERR = 0,
    TX_SYS_TX_RSP_STATUS_UR_ERR = 1,
    TX_SYS_TX_RSP_STATUS_PRIV_ERR = 2,
} LW2080_CTRL_LWLINK_INJECT_TLC_TX_SYS_REPORT_ERROR_TYPE;

typedef enum LW2080_CTRL_LWLINK_INJECT_TLC_RX_LNK_REPORT_ERROR_TYPE {
    RX_LNK_RX_RSP_STATUS_HW_ERR = 0,
    RX_LNK_RX_RSP_STATUS_UR_ERR = 1,
    RX_LNK_RX_RSP_STATUS_PRIV_ERR = 2,
} LW2080_CTRL_LWLINK_INJECT_TLC_RX_LNK_REPORT_ERROR_TYPE;

typedef union LW2080_CTRL_LWLINK_INJECT_TLC_ERROR_TYPE {
    LW2080_CTRL_LWLINK_INJECT_TLC_TX_SYS_REPORT_ERROR_TYPE txSysErrorType;
    LW2080_CTRL_LWLINK_INJECT_TLC_RX_LNK_REPORT_ERROR_TYPE rxLnkErrorType;
} LW2080_CTRL_LWLINK_INJECT_TLC_ERROR_TYPE;


#define LW2080_CTRL_LWLINK_INJECT_TLC_ERROR_PARAMS_MESSAGE_ID (0xFU)

typedef struct LW2080_CTRL_LWLINK_INJECT_TLC_ERROR_PARAMS {
    LwU32                                      linkId;
    LW2080_CTRL_LWLINK_INJECT_TLC_ERROR_DEVICE device;
    LwBool                                     bBroadcast;
    LW2080_CTRL_LWLINK_INJECT_TLC_ERROR_TYPE   errorType;
} LW2080_CTRL_LWLINK_INJECT_TLC_ERROR_PARAMS;

/*
 * LW2080_CTRL_CMD_LWLINK_CHECK_BRIDGE
 *
 * This command returns the presence and data fields of an LWLink Bridge EEPROM.
 *
 * [in]  linkId
 *     The LWLink ID to check for a bridge EEPROM
 * [out] bPresent
 *     LW_TRUE if the EEPROM chip is detected.
 * [out] bValid
 *     LW_TRUE if the the data read passes validity checks. If so, the following
 *     fields are populated.
 * [out] firmwareVersion
 *     The firmware version formatted as PPPP.SSSS.BB, e.g. 4931.0200.01.01,
 *     padded with one or more 0x00
 * [out] bridgeVendor
 *     The bridge vendor name, padded with one or more 0x00
 * [out] boardPartNumber
 *     The board part number, formatted as CCC-FPPPP-SSSS-RRR
 *     (e.g. 699-24931-0200-000), padded with one or more 0x00
 * [out] boardRevision
 *     The board revision, e.g. A00, padded with one or more 0x00
 * [out] businessUnit
 *     Business unit identifier. See LW2080_CTRL_LWLINK_BRIDGE_BUSINESS_UNIT_*
 * [out] configuration
 *     Bridge form factor (2-way/3-way/4-way).
 *     See LW2080_CTRL_LWLINK_BRIDGE_CONFIGURATION_*
 * [out] spacing
 *     # of slots spacing identifier. See LW2080_CTRL_LWLINK_BRIDGE_SPACING_*
 * [out] interconnectType
 *     Type of interconnect. See LW2080_CTRL_LWLINK_BRIDGE_INTERCONNECT_TYPE_*
 * [out] interconnectWidth
 *     Width of interconnect LWHS lanes.
 *     See LW2080_CTRL_LWLINK_BRIDGE_INTERCONNECT_WIDTH_*
 * [out] maximumLandDataRate
 *     Maximum data transfer rate in T/s, as an IEEE-754 32-bit float
 * [out] featureIllumination
 *     Illumination feature supported.
 *     See LW2080_CTRL_LWLINK_BRIDGE_ILLUMINATION_FEATURE_*
 *
 */
#define LW2080_CTRL_CMD_LWLINK_CHECK_BRIDGE                         (0x20803010U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_CHECK_BRIDGE_PARAMS_MESSAGE_ID" */

// ASCII bytes plus space for null terminator
#define LW2080_CTRL_LWLINK_BRIDGE_FIRMWARE_VERSION_LENGTH           (0x11U) /* finn: Evaluated from "(16 + 1)" */
#define LW2080_CTRL_LWLINK_BRIDGE_VENDOR_LENGTH                     (0x15U) /* finn: Evaluated from "(20 + 1)" */
#define LW2080_CTRL_LWLINK_BRIDGE_BOARD_PART_NUMBER_LENGTH          (0x15U) /* finn: Evaluated from "(20 + 1)" */
#define LW2080_CTRL_LWLINK_BRIDGE_BOARD_REVISION_LENGTH             (0x4U) /* finn: Evaluated from "(3 + 1)" */

#define LW2080_CTRL_LWLINK_BRIDGE_BUSINESS_UNIT_UNDEFINED           (0x00U)
#define LW2080_CTRL_LWLINK_BRIDGE_BUSINESS_UNIT_GEFORCE             (0x01U)
#define LW2080_CTRL_LWLINK_BRIDGE_BUSINESS_UNIT_QUADRO              (0x02U)
#define LW2080_CTRL_LWLINK_BRIDGE_BUSINESS_UNIT_TESLA               (0x03U)

#define LW2080_CTRL_LWLINK_BRIDGE_CONFIGURATION_UNDEFINED           (0x00U)
#define LW2080_CTRL_LWLINK_BRIDGE_CONFIGURATION_2_WAY               (0x02U)
#define LW2080_CTRL_LWLINK_BRIDGE_CONFIGURATION_3_WAY               (0x03U)
#define LW2080_CTRL_LWLINK_BRIDGE_CONFIGURATION_4_WAY               (0x04U)

#define LW2080_CTRL_LWLINK_BRIDGE_SPACING_UNDEFINED                 (0x00U)
#define LW2080_CTRL_LWLINK_BRIDGE_SPACING_2_SLOT                    (0x02U)
#define LW2080_CTRL_LWLINK_BRIDGE_SPACING_3_SLOT                    (0x03U)
#define LW2080_CTRL_LWLINK_BRIDGE_SPACING_4_SLOT                    (0x04U)

#define LW2080_CTRL_LWLINK_BRIDGE_INTERCONNECT_TYPE_UNDEFINED       (0x00U)
#define LW2080_CTRL_LWLINK_BRIDGE_INTERCONNECT_TYPE_LWLINK_2        (0x02U)
#define LW2080_CTRL_LWLINK_BRIDGE_INTERCONNECT_TYPE_LWLINK_3        (0x03U)

#define LW2080_CTRL_LWLINK_BRIDGE_INTERCONNECT_WIDTH_UNDEFINED      (0x00U)
#define LW2080_CTRL_LWLINK_BRIDGE_INTERCONNECT_WIDTH_4_LANES        (0x02U)
#define LW2080_CTRL_LWLINK_BRIDGE_INTERCONNECT_WIDTH_8_LANES        (0x03U)
#define LW2080_CTRL_LWLINK_BRIDGE_INTERCONNECT_WIDTH_16_LANES       (0x04U)

#define LW2080_CTRL_LWLINK_BRIDGE_ILLUMINATION_FEATURE_NONE         (0x00U)
#define LW2080_CTRL_LWLINK_BRIDGE_ILLUMINATION_FEATURE_SINGLE_COLOR (0x01U)
#define LW2080_CTRL_LWLINK_BRIDGE_ILLUMINATION_FEATURE_RGB          (0x02U)

#define LW2080_CTRL_LWLINK_CHECK_BRIDGE_PARAMS_MESSAGE_ID (0x10U)

typedef struct LW2080_CTRL_LWLINK_CHECK_BRIDGE_PARAMS {
    LwU32  linkId;
    LwBool bPresent;
    LwBool bValid;
    char   firmwareVersion[LW2080_CTRL_LWLINK_BRIDGE_FIRMWARE_VERSION_LENGTH];
    char   bridgeVendor[LW2080_CTRL_LWLINK_BRIDGE_VENDOR_LENGTH];
    char   boardPartNumber[LW2080_CTRL_LWLINK_BRIDGE_BOARD_PART_NUMBER_LENGTH];
    char   boardRevision[LW2080_CTRL_LWLINK_BRIDGE_BOARD_REVISION_LENGTH];
    LwU8   businessUnit;
    LwU8   configuration;
    LwU8   spacing;
    LwU8   interconnectType;
    LwU8   interconnectWidth;
    LwF32  maximumLaneDataRate;
    LwU8   featureIllumination;
} LW2080_CTRL_LWLINK_CHECK_BRIDGE_PARAMS;

/*
 * LW2080_CTRL_CMD_LWLINK_GET_LINK_FOM_VALUES
 *
 * This command returns the per-lane Figure Of Merit (FOM) Values from a link
 *
 * [in]  linkId
 *     The LWLink link ID to report FOM values for
 * [out] numLanes
 *     This field specifies the no. of lanes per link
 * [out] figureOfMeritValues
 *     This field contains the FOM values per lane
 *
 */
#define LW2080_CTRL_CMD_LWLINK_GET_LINK_FOM_VALUES (0x20803011U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_CMD_LWLINK_GET_LINK_FOM_VALUES_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_LWLINK_MAX_LANES               4U

#define LW2080_CTRL_CMD_LWLINK_GET_LINK_FOM_VALUES_PARAMS_MESSAGE_ID (0x11U)

typedef struct LW2080_CTRL_CMD_LWLINK_GET_LINK_FOM_VALUES_PARAMS {
    LwU32 linkId;
    LwU8  numLanes;
    LwU16 figureOfMeritValues[LW2080_CTRL_LWLINK_MAX_LANES];
} LW2080_CTRL_CMD_LWLINK_GET_LINK_FOM_VALUES_PARAMS;

/*
 * LW2080_CTRL_LWLINK_SET_LWLINK_PEER
 *
 * This command sets/unsets the USE_LWLINK_PEER bit for a given
 *     mask of peers
 *
 * [in] peerMask
 *     Mask of Peer IDs for which USE_LWLINK_PEER needs to be updated
 * [in] bEnable
 *     Whether the bit needs to be set or unset
 *
 * Possible status values returned are:
 *   LW_OK
 *     If the USE_LWLINK_PEER bit was updated successfully
 *   LW_ERR_NOT_SUPPORTED
 *     If LWLINK is not supported on the chip, or
 *     If unsetting USE_LWLINK_PEER bit is not supported
 *
 * NOTE: This is only supported on Windows
 *
 */
#define LW2080_CTRL_CMD_LWLINK_SET_LWLINK_PEER (0x20803012U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_SET_LWLINK_PEER_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_LWLINK_SET_LWLINK_PEER_PARAMS_MESSAGE_ID (0x12U)

typedef struct LW2080_CTRL_LWLINK_SET_LWLINK_PEER_PARAMS {
    LwU32  peerMask;
    LwBool bEnable;
} LW2080_CTRL_LWLINK_SET_LWLINK_PEER_PARAMS;

/*
 * LW2080_CTRL_CMD_LWLINK_READ_UPHY_PAD_LANE_REG
 *
 * This command packs the lane and addr values into LW_PMINION_MISC_0_SCRATCH_SWRW_0
 * and then issues a READPADLANEREG DLCMD to minion for the desired link. Only one DLCMD 
 * at a time can be issued to any given link.
 * 
 * After this command completes it is necessary to read the appropriate
 * LW_PLWL_BR0_PAD_CTL_7_CFG_RDATA register to retrieve the results of the read
 * Only GV100 should read LW_PLWL_BR0_PAD_CTL_7_CFG_RDATA.
 * From TU102+ the ctrl the required data would be updated in phyConfigData.
 *
 * [in] linkId
 *     Link whose pad lane register is being read
 * [in] lane
 *     Lane whose pad lane register is being read
 * [in] addr
 *     Address of the pad lane register to read
 * [out] phyConfigData
 *     Provides phyconfigaddr and landid
 *
 * Possible status values returned are:
 *   LW_OK
 *     If the minion command completed successfully
 *   LW_ERR_NOT_SUPPORTED
 *     If LWLINK is not supported on the chip
 *   LW_ERR_ILWALID_ARGUMENT
 *     If the link is not enabled on the GPU or the lane is invalid
 *   LW_ERR_TIMEOUT
 *     If a timeout oclwrred waiting for minion response
 */
#define LW2080_CTRL_CMD_LWLINK_READ_UPHY_PAD_LANE_REG (0x20803013U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_READ_UPHY_PAD_LANE_REG_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_LWLINK_READ_UPHY_PAD_LANE_REG_PARAMS_MESSAGE_ID (0x13U)

typedef struct LW2080_CTRL_LWLINK_READ_UPHY_PAD_LANE_REG_PARAMS {
    LwU8  linkId;
    LwU8  lane;
    LwU16 addr;
    LwU32 phyConfigData;
} LW2080_CTRL_LWLINK_READ_UPHY_PAD_LANE_REG_PARAMS;

/*
 * Structure to store the ECC error data.
 * valid
 *     Is the lane valid or not
 * eccErrorValue
 *     Value of the Error.
 * overflowed
 *     If the error overflowed or not
 */
typedef struct LW2080_CTRL_LWLINK_LANE_ERROR {
    LwBool bValid;
    LwU32  eccErrorValue;
    LwBool overflowed;
} LW2080_CTRL_LWLINK_LANE_ERROR;

/*
 * Structure to store ECC error data for Links
 * errorLane array index corresponds to the lane number.
 *
 * errorLane[]
 *    Stores the ECC error data per lane.
 */
typedef struct LW2080_CTRL_LWLINK_LINK_ECC_ERROR {
    LW2080_CTRL_LWLINK_LANE_ERROR errorLane[LW2080_CTRL_LWLINK_MAX_LANES];
    LwU32                         eccDecFailed;
    LwBool                        eccDecFailedOverflowed;
} LW2080_CTRL_LWLINK_LINK_ECC_ERROR;

/*
 * LW2080_CTRL_LWLINK_GET_LWLINK_ECC_ERRORS
 *
 * Control to get the values of ECC ERRORS
 *
 * Parameters:
 *    linkMask [IN]
 *      Links on which the ECC error data requested
 *      A valid link/port mask returned by the port masks returned by
 *      LWSWITCH_GET_INFO
 *    errorLink[] [OUT]
 *      Stores the ECC error related information for each link.
 *      errorLink array index corresponds to the link Number.   
 */

#define LW2080_CTRL_LWLINK_GET_LWLINK_ECC_ERRORS_PARAMS_MESSAGE_ID (0x14U)

typedef struct LW2080_CTRL_LWLINK_GET_LWLINK_ECC_ERRORS_PARAMS {
    LwU32                             linkMask;
    LW2080_CTRL_LWLINK_LINK_ECC_ERROR errorLink[LW2080_CTRL_LWLINK_MAX_LINKS];
} LW2080_CTRL_LWLINK_GET_LWLINK_ECC_ERRORS_PARAMS;


#define LW2080_CTRL_CMD_LWLINK_GET_LWLINK_ECC_ERRORS     (0x20803014U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_GET_LWLINK_ECC_ERRORS_PARAMS_MESSAGE_ID" */

// Lwlink throughput counters reading data flits in TX
#define LW2080_CTRL_LWLINK_READ_TP_COUNTERS_TYPE_DATA_TX 0U

// Lwlink throughput counters reading data flits in RX
#define LW2080_CTRL_LWLINK_READ_TP_COUNTERS_TYPE_DATA_RX 1U

// Lwlink throughput counters reading all flits in TX
#define LW2080_CTRL_LWLINK_READ_TP_COUNTERS_TYPE_RAW_TX  2U

// Lwlink throughput counters reading all flits in RX
#define LW2080_CTRL_LWLINK_READ_TP_COUNTERS_TYPE_RAW_RX  3U

#define LW2080_CTRL_LWLINK_READ_TP_COUNTERS_TYPE_MAX     4U

/*
 * LW2080_CTRL_CMD_LWLINK_READ_TP_COUNTERS
 *
 * Reads reserved monotonically increasing LWLINK throughput counters for given linkIds
 *
 * [in] counterMask
 *     Mask of counter types to be queried
 *     One of LW2080_CTRL_LWLINK_READ_TP_COUNTERS_TYPE_* macros
 * [in] linkMask
 *     Mask of links to be queried
 * [out] value
 *     Throughput counter value returned
 *
 * Possible status values returned are:
 *   LW_OK
 *     If command completed successfully
 *   LW_ERR_NOT_SUPPORTED
 *     If LWLINK is not supported on the chip
 *   LW_ERR_ILWALID_ARGUMENT
 *     If numLinks is out-of-range or requested link is inactive
 *
 * Note:
 * The following commands will be deprecated in favor of LW2080_CTRL_CMD_LWLINK_READ_TP_COUNTERS:
 *     LW90CC_CTRL_CMD_LWLINK_GET_COUNTERS
 *     LW2080_CTRL_CMD_LWLINK_GET_COUNTERS
 * Other commands that will be deprecated due to the change in design:
 *     LW90CC_CTRL_CMD_LWLINK_RESERVE_COUNTERS
 *     LW90CC_CTRL_CMD_LWLINK_RELEASE_COUNTERS
 *     LW90CC_CTRL_CMD_LWLINK_SET_COUNTERS_FROZEN
 *     LW90CC_CTRL_CMD_LWLINK_GET_TL_COUNTER_CFG
 *     LW90CC_CTRL_CMD_LWLINK_SET_TL_COUNTER_CFG
 *     LW90CC_CTRL_CMD_LWLINK_CLEAR_COUNTERS
 *
 * Also, note that there is no counter overflow handling for these calls.
 * These counters would be counting in flits and assuming 25GB/s bandwidth per link,
 * with traffic flowing continuously, it would take 174 years for overflow to happen.
 * It is reasonable to assume an overflow will not occur within the GPU operation,
 * given that the counters get reset at system reboot or GPU reset. Counters are 63-bit.
 */

typedef struct LW2080_CTRL_LWLINK_READ_TP_COUNTERS_VALUES {
    LW_DECLARE_ALIGNED(LwU64 value[LW2080_CTRL_LWLINK_READ_TP_COUNTERS_TYPE_MAX], 8);
} LW2080_CTRL_LWLINK_READ_TP_COUNTERS_VALUES;

#define LW2080_CTRL_LWLINK_READ_TP_COUNTERS_PARAMS_MESSAGE_ID (0x15U)

typedef struct LW2080_CTRL_LWLINK_READ_TP_COUNTERS_PARAMS {
    LwU16 counterMask;
    LW_DECLARE_ALIGNED(LwU64 linkMask, 8);
    LW_DECLARE_ALIGNED(LW2080_CTRL_LWLINK_READ_TP_COUNTERS_VALUES counters[LW2080_CTRL_LWLINK_MAX_LINKS], 8);
} LW2080_CTRL_LWLINK_READ_TP_COUNTERS_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_READ_TP_COUNTERS      (0x20803015U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_READ_TP_COUNTERS_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_LWLINK_LOCK_LINK_POWER_STATE
 *
 * This command locks the link power state so that RM doesn't modify the state
 * of the link during pstate switch.
 *
 *   [in] linkMask        Links for which power mode needs to be locked.
 */
#define LW2080_CTRL_CMD_LWLINK_LOCK_LINK_POWER_STATE (0x20803016U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_CMD_LWLINK_LOCK_LINK_POWER_STATE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_LWLINK_LOCK_LINK_POWER_STATE_PARAMS_MESSAGE_ID (0x16U)

typedef struct LW2080_CTRL_CMD_LWLINK_LOCK_LINK_POWER_STATE_PARAMS {
    LwBool bLockPowerMode;
} LW2080_CTRL_CMD_LWLINK_LOCK_LINK_POWER_STATE_PARAMS;

/*
 * LW2080_CTRL_CMD_LWLINK_ENABLE_LWLINK_PEER
 *
 * This command is used to enable RM LWLink enabled peer state.
 * Note: This just updates the RM state. To reflect the state in the registers,
 *       use LW2080_CTRL_CMD_LWLINK_SET_LWLINK_PEER
 *
 * [in] peerMask
 *     Mask of Peer IDs for which USE_LWLINK_PEER needs to be enabled
 * [in] bEnable
 *     Whether the bit needs to be set or unset
 *
 * Possible status values returned are:
 *   LW_OK
 *     If the USE_LWLINK_PEER bit was enabled successfully
 *   LW_ERR_NOT_SUPPORTED
 *     If LWLINK is not supported on the chip, or
 *     If unsetting USE_LWLINK_PEER bit is not supported
 *
 */
#define LW2080_CTRL_CMD_LWLINK_ENABLE_LWLINK_PEER (0x20803017U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_ENABLE_LWLINK_PEER_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_LWLINK_ENABLE_LWLINK_PEER_PARAMS_MESSAGE_ID (0x17U)

typedef struct LW2080_CTRL_LWLINK_ENABLE_LWLINK_PEER_PARAMS {
    LwU32  peerMask;
    LwBool bEnable;
} LW2080_CTRL_LWLINK_ENABLE_LWLINK_PEER_PARAMS;

#define LW2080_CTRL_LWLINK_GET_LP_COUNTERS_COUNT_TX_LWHS   0U
#define LW2080_CTRL_LWLINK_GET_LP_COUNTERS_COUNT_TX_EIGHTH 1U
#define LW2080_CTRL_LWLINK_GET_LP_COUNTERS_COUNT_TX_OTHER  2U
#define LW2080_CTRL_LWLINK_GET_LP_COUNTERS_NUM_TX_LP_ENTER 3U
#define LW2080_CTRL_LWLINK_GET_LP_COUNTERS_NUM_TX_LP_EXIT  4U
#define LW2080_CTRL_LWLINK_GET_LP_COUNTERS_COUNT_TX_SLEEP  5U
#define LW2080_CTRL_LWLINK_GET_LP_COUNTERS_MAX_COUNTERS    6U

/*
 * LW2080_CTRL_CMD_LWLINK_GET_LP_COUNTERS
 *
 * Reads LWLINK low power counters for given linkId
 *
 * [in] linkId
 *     ID of the link to be queried
 * [in,out] counterValidMask
 *     Mask of valid counters
 * [out] counterValues
 *     Low power counter values returned
 *
 * Possible status values returned are:
 *   LW_OK
 *     If command completed successfully
 *   LW_ERR_NOT_SUPPORTED
 *     If LWLINK is not supported on the chip
 *   LW_ERR_ILWALID_ARGUMENT
 *     If linkId is out-of-range or requested link is inactive
 */

#define LW2080_CTRL_LWLINK_GET_LP_COUNTERS_PARAMS_MESSAGE_ID (0x18U)

typedef struct LW2080_CTRL_LWLINK_GET_LP_COUNTERS_PARAMS {
    LwU32 linkId;
    LwU32 counterValidMask;
    LwU32 counterValues[LW2080_CTRL_LWLINK_GET_LP_COUNTERS_MAX_COUNTERS];
} LW2080_CTRL_LWLINK_GET_LP_COUNTERS_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_GET_LP_COUNTERS                  (0x20803018U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_GET_LP_COUNTERS_PARAMS_MESSAGE_ID" */

/*
 * LWLINK Link states
 * These should ALWAYS match the lwlink core library defines in lwlink.h
 */
#define LW2080_LWLINK_CORE_LINK_STATE_OFF                       0x00U
#define LW2080_LWLINK_CORE_LINK_STATE_HS                        0x01U
#define LW2080_LWLINK_CORE_LINK_STATE_SAFE                      0x02U
#define LW2080_LWLINK_CORE_LINK_STATE_FAULT                     0x03U
#define LW2080_LWLINK_CORE_LINK_STATE_RECOVERY                  0x04U
#define LW2080_LWLINK_CORE_LINK_STATE_FAIL                      0x05U
#define LW2080_LWLINK_CORE_LINK_STATE_DETECT                    0x06U
#define LW2080_LWLINK_CORE_LINK_STATE_RESET                     0x07U
#define LW2080_LWLINK_CORE_LINK_STATE_ENABLE_PM                 0x08U
#define LW2080_LWLINK_CORE_LINK_STATE_DISABLE_PM                0x09U
#define LW2080_LWLINK_CORE_LINK_STATE_SLEEP                     0x0AU
#define LW2080_LWLINK_CORE_LINK_STATE_SAVE_STATE                0x0BU
#define LW2080_LWLINK_CORE_LINK_STATE_RESTORE_STATE             0x0LW
#define LW2080_LWLINK_CORE_LINK_STATE_PRE_HS                    0x0EU
#define LW2080_LWLINK_CORE_LINK_STATE_DISABLE_ERR_DETECT        0x0FU
#define LW2080_LWLINK_CORE_LINK_STATE_LANE_DISABLE              0x10U
#define LW2080_LWLINK_CORE_LINK_STATE_LANE_SHUTDOWN             0x11U
#define LW2080_LWLINK_CORE_LINK_STATE_TRAFFIC_SETUP             0x12U
#define LW2080_LWLINK_CORE_LINK_STATE_INITPHASE1                0x13U
#define LW2080_LWLINK_CORE_LINK_STATE_INITNEGOTIATE             0x14U
#define LW2080_LWLINK_CORE_LINK_STATE_POST_INITNEGOTIATE        0x15U
#define LW2080_LWLINK_CORE_LINK_STATE_INITOPTIMIZE              0x16U
#define LW2080_LWLINK_CORE_LINK_STATE_POST_INITOPTIMIZE         0x17U
#define LW2080_LWLINK_CORE_LINK_STATE_DISABLE_HEARTBEAT         0x18U
#define LW2080_LWLINK_CORE_LINK_STATE_CONTAIN                   0x19U
#define LW2080_LWLINK_CORE_LINK_STATE_INITTL                    0x1AU
#define LW2080_LWLINK_CORE_LINK_STATE_INITPHASE5                0x1BU
#define LW2080_LWLINK_CORE_LINK_STATE_ALI                       0x1LW
#define LW2080_LWLINK_CORE_LINK_STATE_ILWALID                   0xFFU

/*
 * LWLINK TX Sublink states
 * These should ALWAYS match the lwlink core library defines in lwlink.h
 */
#define LW2080_LWLINK_CORE_SUBLINK_STATE_TX_HS                  0x00U
#define LW2080_LWLINK_CORE_SUBLINK_STATE_TX_SINGLE_LANE         0x04U
#define LW2080_LWLINK_CORE_SUBLINK_STATE_TX_TRAIN               0x05U
#define LW2080_LWLINK_CORE_SUBLINK_STATE_TX_SAFE                0x06U
#define LW2080_LWLINK_CORE_SUBLINK_STATE_TX_OFF                 0x07U
#define LW2080_LWLINK_CORE_SUBLINK_STATE_TX_COMMON_MODE         0x08U
#define LW2080_LWLINK_CORE_SUBLINK_STATE_TX_COMMON_MODE_DISABLE 0x09U
#define LW2080_LWLINK_CORE_SUBLINK_STATE_TX_DATA_READY          0x0AU
#define LW2080_LWLINK_CORE_SUBLINK_STATE_TX_EQ                  0x0BU
#define LW2080_LWLINK_CORE_SUBLINK_STATE_TX_PRBS_EN             0x0LW
#define LW2080_LWLINK_CORE_SUBLINK_STATE_TX_POST_HS             0x0DU

/*
 * LWLINK RX Sublink states
 * These should ALWAYS match the lwlink core library defines in lwlink.h
 */
#define LW2080_LWLINK_CORE_SUBLINK_STATE_RX_HS                  0x00U
#define LW2080_LWLINK_CORE_SUBLINK_STATE_RX_SINGLE_LANE         0x04U
#define LW2080_LWLINK_CORE_SUBLINK_STATE_RX_TRAIN               0x05U
#define LW2080_LWLINK_CORE_SUBLINK_STATE_RX_SAFE                0x06U
#define LW2080_LWLINK_CORE_SUBLINK_STATE_RX_OFF                 0x07U
#define LW2080_LWLINK_CORE_SUBLINK_STATE_RX_RXCAL               0x08U
#define LW2080_LWLINK_CORE_SUBLINK_STATE_RX_INIT_TERM           0x09U

/*
 * Link training seed values
 * These should ALWAYS match the values defined in lwlink.h
 */
#define LW2080_CTRL_LWLINK_MAX_SEED_NUM                         6U
#define LW2080_CTRL_LWLINK_MAX_SEED_BUFFER_SIZE                 (0x7U) /* finn: Evaluated from "LW2080_CTRL_LWLINK_MAX_SEED_NUM + 1" */

// LWLINK callback types
#define LW2080_CTRL_LWLINK_CALLBACK_TYPE_GET_DL_LINK_MODE       0x00U
#define LW2080_CTRL_LWLINK_CALLBACK_TYPE_SET_DL_LINK_MODE       0x01U
#define LW2080_CTRL_LWLINK_CALLBACK_TYPE_GET_TL_LINK_MODE       0x02U
#define LW2080_CTRL_LWLINK_CALLBACK_TYPE_SET_TL_LINK_MODE       0x03U
#define LW2080_CTRL_LWLINK_CALLBACK_TYPE_GET_TX_SUBLINK_MODE    0x04U
#define LW2080_CTRL_LWLINK_CALLBACK_TYPE_SET_TX_SUBLINK_MODE    0x05U
#define LW2080_CTRL_LWLINK_CALLBACK_TYPE_GET_RX_SUBLINK_MODE    0x06U
#define LW2080_CTRL_LWLINK_CALLBACK_TYPE_SET_RX_SUBLINK_MODE    0x07U
#define LW2080_CTRL_LWLINK_CALLBACK_TYPE_GET_RX_SUBLINK_DETECT  0x08U
#define LW2080_CTRL_LWLINK_CALLBACK_TYPE_SET_RX_SUBLINK_DETECT  0x09U
#define LW2080_CTRL_LWLINK_CALLBACK_TYPE_WRITE_DISCOVERY_TOKEN  0x0AU
#define LW2080_CTRL_LWLINK_CALLBACK_TYPE_READ_DISCOVERY_TOKEN   0x0BU
#define LW2080_CTRL_LWLINK_CALLBACK_TYPE_TRAINING_COMPLETE      0x0LW
#define LW2080_CTRL_LWLINK_CALLBACK_TYPE_GET_UPHY_LOAD          0x0DU

/*
 * Structure to store the GET_DL_MODE callback params.
 * mode
 *     The current Lwlink DL mode
 */
typedef struct LW2080_CTRL_LWLINK_CALLBACK_GET_DL_LINK_MODE_PARAMS {
    LwU32 mode;
} LW2080_CTRL_LWLINK_CALLBACK_GET_DL_LINK_MODE_PARAMS;

/*
 * Structure to store the SET_DL_LINK_MODE callback OFF params
 * seedData
 *     The output seed data
 */
typedef struct LW2080_CTRL_LWLINK_SET_DL_LINK_MODE_OFF_PARAMS {
    LwU32 seedData[LW2080_CTRL_LWLINK_MAX_SEED_BUFFER_SIZE];
} LW2080_CTRL_LWLINK_SET_DL_LINK_MODE_OFF_PARAMS;

/*
 * Structure to store the SET_DL_LINK_MODE callback PRE_HS params
 * remoteDeviceType
 *     The input remote Device Type
 * ipVerDlPl
 *     The input DLPL version
 */
typedef struct LW2080_CTRL_LWLINK_SET_DL_LINK_MODE_PRE_HS_PARAMS {
    LwU32 remoteDeviceType;
    LwU32 ipVerDlPl;
} LW2080_CTRL_LWLINK_SET_DL_LINK_MODE_PRE_HS_PARAMS;

/*
 * Structure to store SET_DL_LINK_MODE callback INIT_PHASE1 params
 * seedData[]
 *     The input seed data
 */
typedef struct LW2080_CTRL_LWLINK_SET_DL_LINK_MODE_INIT_PHASE1_PARAMS {
    LwU32 seedData[LW2080_CTRL_LWLINK_MAX_SEED_BUFFER_SIZE];
} LW2080_CTRL_LWLINK_SET_DL_LINK_MODE_INIT_PHASE1_PARAMS;

/*
 * Structure to store the Lwlink Remote and Local SID info
 * remoteSid
 *     The output remote SID
 * remoteDeviceType
 *     The output remote Device Type
 * remoteLinkId
 *     The output remote link ID
 * localSid
 *     The output local SID
 */
typedef struct LW2080_CTRL_LWLINK_REMOTE_LOCAL_SID_INFO {
    LW_DECLARE_ALIGNED(LwU64 remoteSid, 8);
    LwU32 remoteDeviceType;
    LwU32 remoteLinkId;
    LW_DECLARE_ALIGNED(LwU64 localSid, 8);
} LW2080_CTRL_LWLINK_REMOTE_LOCAL_SID_INFO;

/*
 * Structure to store the SET_DL_LINK_MODE callback POST_INITNEGOTIATE params
 * bInitnegotiateConfigGood
 *     The output bool if the config is good
 * remoteLocalSidInfo
 *     The output structure containing the Lwlink Remote/Local SID info
 */
typedef struct LW2080_CTRL_LWLINK_SET_DL_LINK_MODE_POST_INITNEGOTIATE_PARAMS {
    LwBool bInitnegotiateConfigGood;
    LW_DECLARE_ALIGNED(LW2080_CTRL_LWLINK_REMOTE_LOCAL_SID_INFO remoteLocalSidInfo, 8);
} LW2080_CTRL_LWLINK_SET_DL_LINK_MODE_POST_INITNEGOTIATE_PARAMS;

/*
 * Structure to store the SET_DL_LINK_MODE callback POST_INITOPTIMIZE params
 * bPollDone
 *     The output bool if the polling has finished
 */
typedef struct LW2080_CTRLLWLINK_SET_DL_LINK_MODE_POST_INITOPTIMIZE_PARAMS {
    LwBool bPollDone;
} LW2080_CTRLLWLINK_SET_DL_LINK_MODE_POST_INITOPTIMIZE_PARAMS;

/*
 * Structure to store the SET_DL_LINK_MODE callback params
 * mode
 *     The input lwlink state to set
 * bSync
 *     The input sync boolean
 * linkMode
 *     The input link mode to be set for the callback
 */
typedef struct LW2080_CTRL_LWLINK_CALLBACK_SET_DL_LINK_MODE_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 mode, 8);
    LwBool bSync;
    LwU32  linkMode;
    union {
        LW2080_CTRL_LWLINK_SET_DL_LINK_MODE_OFF_PARAMS              linkModeOffParams;
        LW2080_CTRL_LWLINK_SET_DL_LINK_MODE_PRE_HS_PARAMS           linkModePreHsParams;
        LW2080_CTRL_LWLINK_SET_DL_LINK_MODE_INIT_PHASE1_PARAMS      linkModeInitPhase1Params;
        LW_DECLARE_ALIGNED(LW2080_CTRL_LWLINK_SET_DL_LINK_MODE_POST_INITNEGOTIATE_PARAMS linkModePostInitNegotiateParams, 8);
        LW2080_CTRLLWLINK_SET_DL_LINK_MODE_POST_INITOPTIMIZE_PARAMS linkModePostInitOptimizeParams;
    } linkModeParams;
} LW2080_CTRL_LWLINK_CALLBACK_SET_DL_LINK_MODE_PARAMS;

/*
 * Structure to store the GET_TL_MODE callback params.
 * mode
 *     The current Lwlink TL mode
 */
typedef struct LW2080_CTRL_LWLINK_CALLBACK_GET_TL_LINK_MODE_PARAMS {
    LwU32 mode;
} LW2080_CTRL_LWLINK_CALLBACK_GET_TL_LINK_MODE_PARAMS;

/*
 * Structure to store the SET_TL_LINK_MODE callback params
 * mode
 *     The input lwlink mode to set
 * bSync
 *     The input sync boolean
 */
typedef struct LW2080_CTRL_LWLINK_CALLBACK_SET_TL_LINK_MODE_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 mode, 8);
    LwBool bSync;
} LW2080_CTRL_LWLINK_CALLBACK_SET_TL_LINK_MODE_PARAMS;

/*
 * Structure to store the GET_RX/TX_SUBLINK_MODE callback params
 * sublinkMode
 *     The current Sublink mode
 * sublinkSubMode
 *     The current Sublink sub mode
 */
typedef struct LW2080_CTRL_LWLINK_CALLBACK_GET_SUBLINK_MODE_PARAMS {
    LwU32 sublinkMode;
    LwU32 sublinkSubMode;
} LW2080_CTRL_LWLINK_CALLBACK_GET_SUBLINK_MODE_PARAMS;

/*
 * Structure to store the SET_TL_LINK_MODE callback params
 * mode
 *     The input lwlink mode to set
 * bSync
 *     The input sync boolean
 */
typedef struct LW2080_CTRL_LWLINK_CALLBACK_SET_TX_SUBLINK_MODE_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 mode, 8);
    LwBool bSync;
} LW2080_CTRL_LWLINK_CALLBACK_SET_TX_SUBLINK_MODE_PARAMS;

/*
 * Structure to store the SET_RX_SUBLINK_MODE callback params
 * mode
 *     The input lwlink mode to set
 * bSync
 *     The input sync boolean
 */
typedef struct LW2080_CTRL_LWLINK_CALLBACK_SET_RX_SUBLINK_MODE_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 mode, 8);
    LwBool bSync;
} LW2080_CTRL_LWLINK_CALLBACK_SET_RX_SUBLINK_MODE_PARAMS;

/*
 * Structure to store the GET_RX_SUBLINK_DETECT callback params
 * laneRxdetStatusMask
 *     The output RXDET per-lane status mask 
 */
typedef struct LW2080_CTRL_LWLINK_CALLBACK_GET_RX_DETECT_PARAMS {
    LwU32 laneRxdetStatusMask;
} LW2080_CTRL_LWLINK_CALLBACK_GET_RX_DETECT_PARAMS;

/*
 * Structure to store the SET_RX_DETECT callback params
 * bSync
 *     The input bSync boolean
 */
typedef struct LW2080_CTRL_LWLINK_CALLBACK_SET_RX_DETECT_PARAMS {
    LwBool bSync;
} LW2080_CTRL_LWLINK_CALLBACK_SET_RX_DETECT_PARAMS;

/*
 * Structure to store the RD_WR_DISCOVERY_TOKEN callback params
 * ipVerDlPl
 *     The input DLPL version
 * token
 *     The output token
 */
typedef struct LW2080_CTRL_LWLINK_CALLBACK_RD_WR_DISCOVERY_TOKEN_PARAMS {
    LwU32 ipVerDlPl;
    LW_DECLARE_ALIGNED(LwU64 token, 8);
} LW2080_CTRL_LWLINK_CALLBACK_RD_WR_DISCOVERY_TOKEN_PARAMS;

/*
 * Structure to store the GET_UPHY_LOAD callback params
 * bUnlocked
 *     The output unlocked boolean
 */
typedef struct LW2080_CTRL_LWLINK_CALLBACK_GET_UPHY_LOAD_PARAMS {
    LwBool bUnlocked;
} LW2080_CTRL_LWLINK_CALLBACK_GET_UPHY_LOAD_PARAMS;

/*
 * Structure to store the Union of Callback params
 * type
 *     The input type of callback to be exelwted
 */
typedef struct LW2080_CTRL_LWLINK_CALLBACK_TYPE {
    LwU8 type;
    union {
        LW2080_CTRL_LWLINK_CALLBACK_GET_DL_LINK_MODE_PARAMS getDlLinkMode;
        LW_DECLARE_ALIGNED(LW2080_CTRL_LWLINK_CALLBACK_SET_DL_LINK_MODE_PARAMS setDlLinkMode, 8);
        LW2080_CTRL_LWLINK_CALLBACK_GET_TL_LINK_MODE_PARAMS getTlLinkMode;
        LW_DECLARE_ALIGNED(LW2080_CTRL_LWLINK_CALLBACK_SET_TL_LINK_MODE_PARAMS setTlLinkMode, 8);
        LW2080_CTRL_LWLINK_CALLBACK_GET_SUBLINK_MODE_PARAMS getTxSublinkMode;
        LW_DECLARE_ALIGNED(LW2080_CTRL_LWLINK_CALLBACK_SET_TX_SUBLINK_MODE_PARAMS setTxSublinkMode, 8);
        LW2080_CTRL_LWLINK_CALLBACK_GET_SUBLINK_MODE_PARAMS getRxSublinkMode;
        LW_DECLARE_ALIGNED(LW2080_CTRL_LWLINK_CALLBACK_SET_RX_SUBLINK_MODE_PARAMS setRxSublinkMode, 8);
        LW2080_CTRL_LWLINK_CALLBACK_GET_RX_DETECT_PARAMS    getRxSublinkDetect;
        LW2080_CTRL_LWLINK_CALLBACK_SET_RX_DETECT_PARAMS    setRxSublinkDetect;
        LW_DECLARE_ALIGNED(LW2080_CTRL_LWLINK_CALLBACK_RD_WR_DISCOVERY_TOKEN_PARAMS writeDiscoveryToken, 8);
        LW_DECLARE_ALIGNED(LW2080_CTRL_LWLINK_CALLBACK_RD_WR_DISCOVERY_TOKEN_PARAMS readDiscoveryToken, 8);
        LW2080_CTRL_LWLINK_CALLBACK_GET_UPHY_LOAD_PARAMS    getUphyLoad;
    } callbackParams;
} LW2080_CTRL_LWLINK_CALLBACK_TYPE;

/*
 * LW2080_CTRL_CMD_LWLINK_CORE_CALLBACK
 *
 * Generic LwLink callback RPC to route commands to GSP
 *
 * [In] linkdId
 *     ID of the link to be used
 * [In/Out] callBackType
 *     Callback params
 */
#define LW2080_CTRL_LWLINK_CORE_CALLBACK_PARAMS_MESSAGE_ID (0x19U)

typedef struct LW2080_CTRL_LWLINK_CORE_CALLBACK_PARAMS {
    LwU32 linkId;
    LW_DECLARE_ALIGNED(LW2080_CTRL_LWLINK_CALLBACK_TYPE callbackType, 8);
} LW2080_CTRL_LWLINK_CORE_CALLBACK_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_CORE_CALLBACK (0x20803019U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_CORE_CALLBACK_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_LWLINK_GET_ALI_ENABLED
 *
 * Returns if ALI is enabled
 *
 * [Out] bEnableAli
 *     Output boolean for ALI enablement
 */
#define LW2080_CTRL_LWLINK_GET_ALI_ENABLED_PARAMS_MESSAGE_ID (0x1aU)

typedef struct LW2080_CTRL_LWLINK_GET_ALI_ENABLED_PARAMS {
    LwBool bEnableAli;
} LW2080_CTRL_LWLINK_GET_ALI_ENABLED_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_GET_ALI_ENABLED (0x2080301aU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_GET_ALI_ENABLED_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_LWLINK_UPDATE_REMOTE_LOCAL_SID
 *
 * Update Remote and Local Sid info via GSP
 *
 * [In] linkId
 *     ID of the link to be used
 * [Out] remoteLocalSidInfo
 *     The output structure containing the Lwlink Remote/Local SID info
 */
#define LW2080_CTRL_LWLINK_UPDATE_REMOTE_LOCAL_SID_PARAMS_MESSAGE_ID (0x1bU)

typedef struct LW2080_CTRL_LWLINK_UPDATE_REMOTE_LOCAL_SID_PARAMS {
    LwU32 linkId;
    LW_DECLARE_ALIGNED(LW2080_CTRL_LWLINK_REMOTE_LOCAL_SID_INFO remoteLocalSidInfo, 8);
} LW2080_CTRL_LWLINK_UPDATE_REMOTE_LOCAL_SID_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_UPDATE_REMOTE_LOCAL_SID   (0x2080301bU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_UPDATE_REMOTE_LOCAL_SID_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_LWLINK_UPDATE_HSHUB_MUX_TYPE_PROGRAM 0x0U
#define LW2080_CTRL_LWLINK_UPDATE_HSHUB_MUX_TYPE_RESET   0x1U

/*
 * LW2080_CTRL_CMD_LWLINK_UPDATE_HSHUB_MUX
 *
 * Generic Hshub Mux Update RPC to route commands to GSP
 *
 * [In] updateType
 *     HSHUB Mux update type to program or reset Mux
 * [In] bSysMem
 *     Boolean to differentiate between sysmen and peer mem
 * [In] peerMask
 *     Mask of peer IDs. Only parsed when bSysMem is false
 */
#define LW2080_CTRL_LWLINK_UPDATE_HSHUB_MUX_PARAMS_MESSAGE_ID (0x1lw)

typedef struct LW2080_CTRL_LWLINK_UPDATE_HSHUB_MUX_PARAMS {
    LwBool updateType;
    LwBool bSysMem;
    LwU32  peerMask;
} LW2080_CTRL_LWLINK_UPDATE_HSHUB_MUX_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_UPDATE_HSHUB_MUX (0x2080301lw) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_UPDATE_HSHUB_MUX_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_LWLINK_PRE_SETUP_LWLINK_PEER
 *
 * Performs all the necessary actions required before setting a peer on LWLink
 *
 * [In] peerId
 *     Peer ID which will be set on LWLink
 * [In] peerLinkMask
 *     Mask of links that connects the given peer
 * [In] bLwswitchConn
 *     Is the GPU connected to LWSwitch
 */
#define LW2080_CTRL_LWLINK_PRE_SETUP_LWLINK_PEER_PARAMS_MESSAGE_ID (0x1dU)

typedef struct LW2080_CTRL_LWLINK_PRE_SETUP_LWLINK_PEER_PARAMS {
    LwU32  peerId;
    LwU32  peerLinkMask;
    LwBool bLwswitchConn;
} LW2080_CTRL_LWLINK_PRE_SETUP_LWLINK_PEER_PARAMS;
#define LW2080_CTRL_CMD_LWLINK_PRE_SETUP_LWLINK_PEER (0x2080301dU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_PRE_SETUP_LWLINK_PEER_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_LWLINK_POST_SETUP_LWLINK_PEER
 *
 * Performs all the necessary actions required after setting a peer on LWLink
 *
 * [In] peerMask
 *     Mask of Peer IDs which has been set on LWLink
 */
#define LW2080_CTRL_LWLINK_POST_SETUP_LWLINK_PEER_PARAMS_MESSAGE_ID (0x1eU)

typedef struct LW2080_CTRL_LWLINK_POST_SETUP_LWLINK_PEER_PARAMS {
    LwU32 peerMask;
} LW2080_CTRL_LWLINK_POST_SETUP_LWLINK_PEER_PARAMS;
#define LW2080_CTRL_CMD_LWLINK_POST_SETUP_LWLINK_PEER        (0x2080301eU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_POST_SETUP_LWLINK_PEER_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_LWLINK_REMOVE_LWLINK_MAPPING_TYPE_SYSMEM 0x1U
#define LW2080_CTRL_LWLINK_REMOVE_LWLINK_MAPPING_TYPE_PEER   0x2U

/*
 * LW2080_CTRL_CMD_LWLINK_REMOVE_LWLINK_MAPPING
 *
 * Performs all the necessary actions required to remove LWLink mapping (sysmem or peer or both)
 *
 * [In] mapTypeMask
 *     Remove LWLink mapping for the given map types (sysmem or peer or both)
 * [In] peerMask
 *     Mask of Peer IDs which needs to be removed on LWLink
 *     Only parsed if mapTypeMask accounts peer
 * [In] bL2Entry
 *     Is the peer removal happening because links are entering L2 low power state?
 *     Only parsed if mapTypeMask accounts peer
 */
#define LW2080_CTRL_LWLINK_REMOVE_LWLINK_MAPPING_PARAMS_MESSAGE_ID (0x1fU)

typedef struct LW2080_CTRL_LWLINK_REMOVE_LWLINK_MAPPING_PARAMS {
    LwU32  mapTypeMask;
    LwU32  peerMask;
    LwBool bL2Entry;
} LW2080_CTRL_LWLINK_REMOVE_LWLINK_MAPPING_PARAMS;
#define LW2080_CTRL_CMD_LWLINK_REMOVE_LWLINK_MAPPING (0x2080301fU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_REMOVE_LWLINK_MAPPING_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_LWLINK_SAVE_RESTORE_HSHUB_STATE
 *
 * Performs all the necessary actions required to save/restore HSHUB state during LWLink L2 entry/exit
 *
 * [In] bSave
 *     Whether this is a save/restore operation
 * [In] linkMask
 *     Mask of links for which HSHUB config registers need to be saved/restored
 */
#define LW2080_CTRL_LWLINK_SAVE_RESTORE_HSHUB_STATE_PARAMS_MESSAGE_ID (0x20U)

typedef struct LW2080_CTRL_LWLINK_SAVE_RESTORE_HSHUB_STATE_PARAMS {
    LwBool bSave;
    LwU32  linkMask;
} LW2080_CTRL_LWLINK_SAVE_RESTORE_HSHUB_STATE_PARAMS;
#define LW2080_CTRL_CMD_LWLINK_SAVE_RESTORE_HSHUB_STATE (0x20803020U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_SAVE_RESTORE_HSHUB_STATE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_LWLINK_PROGRAM_BUFFERREADY_FLAGS_SET        (0x00000000)
#define LW2080_CTRL_LWLINK_PROGRAM_BUFFERREADY_FLAGS_SAVE       (0x00000001)
#define LW2080_CTRL_LWLINK_PROGRAM_BUFFERREADY_FLAGS_RESTORE    (0x00000002)

/*
 * LW2080_CTRL_CMD_LWLINK_PROGRAM_BUFFERREADY
 *
 * Performs all the necessary actions required to save/restore bufferready state during LWLink L2 entry/exit
 *
 * [In] flags
 *     Whether to set, save or restore bufferready
 * [In] bSysmem
 *     Whether to perform the operation for sysmem links or peer links
 * [In] peerLinkMask
 *     Mask of peer links for which bufferready state need to be set/saved/restored
 */
#define LW2080_CTRL_LWLINK_PROGRAM_BUFFERREADY_PARAMS_MESSAGE_ID (0x21U)

typedef struct LW2080_CTRL_LWLINK_PROGRAM_BUFFERREADY_PARAMS {
    LwU32  flags;
    LwBool bSysmem;
    LwU32  peerLinkMask;
} LW2080_CTRL_LWLINK_PROGRAM_BUFFERREADY_PARAMS;
#define LW2080_CTRL_CMD_LWLINK_PROGRAM_BUFFERREADY (0x20803021U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_PROGRAM_BUFFERREADY_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_LWLINK_UPDATE_LWRRENT_CONFIG
 *
 * Performs all the necessary actions required to update the current Lwlink configuration 
 *
 * [out] bLwlinkSysmemEnabled
 *     Whether sysmem lwlink support was enabled
 */
#define LW2080_CTRL_LWLINK_UPDATE_LWRRENT_CONFIG_PARAMS_MESSAGE_ID (0x22U)

typedef struct LW2080_CTRL_LWLINK_UPDATE_LWRRENT_CONFIG_PARAMS {
    LwBool bLwlinkSysmemEnabled;
} LW2080_CTRL_LWLINK_UPDATE_LWRRENT_CONFIG_PARAMS;
#define LW2080_CTRL_CMD_LWLINK_UPDATE_LWRRENT_CONFIG (0x20803022U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_UPDATE_LWRRENT_CONFIG_PARAMS_MESSAGE_ID" */

//
// Set the near end loopback mode using the following
// Lwrrently, three modes - NEA, NEDR, NEW
//
#define LW2080_CTRL_LWLINK_SET_LOOPBACK_MODE_DEFAULT              (0x00000000)
#define LW2080_CTRL_LWLINK_SET_LOOPBACK_MODE_NEA                  (0x00000001)
#define LW2080_CTRL_LWLINK_SET_LOOPBACK_MODE_NEDR                 (0x00000002)
#define LW2080_CTRL_LWLINK_SET_LOOPBACK_MODE_NEDW                 (0x00000003)

/*
 * LW2080_CTRL_CMD_LWLINK_SET_LOOPBACK_MODE
 *
 * Generic LwLink callback for MODS
 *
 * [In] linkdId
 *     ID of the link to be used
 * [In] loopbackMode
 *     This value will decide which loopback mode need to
 *     set on the specified link.
 *     Modes are NEA / NEDR / NEDW
 */
#define LW2080_CTRL_LWLINK_SET_LOOPBACK_MODE_PARAMS_MESSAGE_ID (0x23U)

typedef struct LW2080_CTRL_LWLINK_SET_LOOPBACK_MODE_PARAMS {
    LwU32 linkId;
    LwU8  loopbackMode;
} LW2080_CTRL_LWLINK_SET_LOOPBACK_MODE_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_SET_LOOPBACK_MODE (0x20803023U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_SET_LOOPBACK_MODE_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_LWLINK_UPDATE_PEER_LINK_MASK
 *
 * Synchronizes the peerLinkMask between CPU-RM and GSP-RM
 *
 * [In] gpuInst
 *     Gpu instance
 * [In] peerLinkMask
 *     Mask of links to the given peer GPU
 */
#define LW2080_CTRL_LWLINK_UPDATE_PEER_LINK_MASK_PARAMS_MESSAGE_ID (0x24U)

typedef struct LW2080_CTRL_LWLINK_UPDATE_PEER_LINK_MASK_PARAMS {
    LwU32 gpuInst;
    LwU32 peerLinkMask;
} LW2080_CTRL_LWLINK_UPDATE_PEER_LINK_MASK_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_UPDATE_PEER_LINK_MASK (0x20803024U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_UPDATE_PEER_LINK_MASK_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_LWLINK_UPDATE_LINK_CONNECTION
 *
 * Updates the remote connection information for a link
 *
 * [In] linkId
 *     Id of the link to be used
 * [In] bConnected
 *     Boolean that tracks whether the link is connected
 * [In] remoteDeviceType
 *     Tracks whether the remote device is switch/gpu/ibmnpu/cheetah
 * [In] remoteLinkNumber
 *     Tracks the link number for the connected remote device
 */
#define LW2080_CTRL_LWLINK_UPDATE_LINK_CONNECTION_PARAMS_MESSAGE_ID (0x25U)

typedef struct LW2080_CTRL_LWLINK_UPDATE_LINK_CONNECTION_PARAMS {
    LwU32  linkId;
    LwBool bConnected;
    LW_DECLARE_ALIGNED(LwU64 remoteDeviceType, 8);
    LwU32  remoteLinkNumber;
} LW2080_CTRL_LWLINK_UPDATE_LINK_CONNECTION_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_UPDATE_LINK_CONNECTION (0x20803025U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_UPDATE_LINK_CONNECTION_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_LWLINK_ENABLE_LINKS_POST_TOPOLOGY
 *
 * Enable links post topology via GSP
 *
 * [In]  linkMask
 *     Mask of links to enable
 * [Out] initializedLinks
 *     Mask of links that were initialized
 */
#define LW2080_CTRL_LWLINK_ENABLE_LINKS_POST_TOPOLOGY_PARAMS_MESSAGE_ID (0x26U)

typedef struct LW2080_CTRL_LWLINK_ENABLE_LINKS_POST_TOPOLOGY_PARAMS {
    LwU32 linkMask;
    LwU32 initializedLinks;
} LW2080_CTRL_LWLINK_ENABLE_LINKS_POST_TOPOLOGY_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_ENABLE_LINKS_POST_TOPOLOGY (0x20803026U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_ENABLE_LINKS_POST_TOPOLOGY_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_LWLINK_PRE_LINK_TRAIN_ALI
 *
 * [In] linkMask
 *     Mask of enabled links to train
 * [In] bSync
 *     The input sync boolean
 */
#define LW2080_CTRL_LWLINK_PRE_LINK_TRAIN_ALI_PARAMS_MESSAGE_ID (0x27U)

typedef struct LW2080_CTRL_LWLINK_PRE_LINK_TRAIN_ALI_PARAMS {
    LwU32  linkMask;
    LwBool bSync;
} LW2080_CTRL_LWLINK_PRE_LINK_TRAIN_ALI_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_PRE_LINK_TRAIN_ALI (0x20803027U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_PRE_LINK_TRAIN_ALI_PARAMS_MESSAGE_ID" */

//
// Read Refresh counter - the pass/fail oclwrrences
//

typedef struct LW2080_CTRL_LWLINK_PHY_REFRESH_STATUS_INFO {
    // requested links or not
    LwBool bValid;

    // counters
    LwU16  passCount;
    LwU16  failCount;
} LW2080_CTRL_LWLINK_PHY_REFRESH_STATUS_INFO;

#define LW2080_CTRL_LWLINK_MAX_LINK_COUNT 32

/*
 * LW2080_CTRL_CMD_LWLINK_GET_REFRESH_COUNTERS
 *
 *
 * [In] linkMask
 *     Specifies for which links we want to read the counters
 * [Out] refreshCountPass
 *     Count of number of times PHY refresh pass
 * [Out] refreshCountFail
 *     Count of number of times PHY refresh fail
 */
#define LW2080_CTRL_LWLINK_GET_REFRESH_COUNTERS_PARAMS_MESSAGE_ID (0x28U)

typedef struct LW2080_CTRL_LWLINK_GET_REFRESH_COUNTERS_PARAMS {
    LwU32                                      linkMask;
    LW2080_CTRL_LWLINK_PHY_REFRESH_STATUS_INFO refreshCount[LW2080_CTRL_LWLINK_MAX_LINK_COUNT];
} LW2080_CTRL_LWLINK_GET_REFRESH_COUNTERS_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_GET_REFRESH_COUNTERS (0x20803028U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_GET_REFRESH_COUNTERS_PARAMS_MESSAGE_ID" */

//
// Clear Refresh counter - the pass/fail oclwrrences
//

/*
 * LW2080_CTRL_CMD_LWLINK_CLEAR_REFRESH_COUNTERS
 *
 *
 * [In] linkMask
 *     Specifies for which links we want to clear the counters
 */
#define LW2080_CTRL_LWLINK_CLEAR_REFRESH_COUNTERS_PARAMS_MESSAGE_ID (0x29U)

typedef struct LW2080_CTRL_LWLINK_CLEAR_REFRESH_COUNTERS_PARAMS {
    LwU32 linkMask;
} LW2080_CTRL_LWLINK_CLEAR_REFRESH_COUNTERS_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_CLEAR_REFRESH_COUNTERS (0x20803029U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_CLEAR_REFRESH_COUNTERS_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_LWLINK_GET_LINK_MASK_POST_RX_DET
 *
 * Get link mask post Rx detection
 *
 * [Out] postRxDetLinkMask
 *     Mask of links discovered
 */
#define LW2080_CTRL_LWLINK_GET_LINK_MASK_POST_RX_DET_PARAMS_MESSAGE_ID (0x2aU)

typedef struct LW2080_CTRL_LWLINK_GET_LINK_MASK_POST_RX_DET_PARAMS {
    LwU32 postRxDetLinkMask;
} LW2080_CTRL_LWLINK_GET_LINK_MASK_POST_RX_DET_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_GET_LINK_MASK_POST_RX_DET (0x2080302aU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_GET_LINK_MASK_POST_RX_DET_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_LWLINK_LINK_TRAIN_ALI
 *
 * [In] linkMask
 *     Mask of enabled links to train
 * [In] bSync
 *     The input sync boolean
 */
#define LW2080_CTRL_LWLINK_LINK_TRAIN_ALI_PARAMS_MESSAGE_ID (0x2bU)

typedef struct LW2080_CTRL_LWLINK_LINK_TRAIN_ALI_PARAMS {
    LwU32  linkMask;
    LwBool bSync;
} LW2080_CTRL_LWLINK_LINK_TRAIN_ALI_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_LINK_TRAIN_ALI (0x2080302bU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_LINK_TRAIN_ALI_PARAMS_MESSAGE_ID" */

typedef struct LW2080_CTRL_LWLINK_DEVICE_LINK_VALUES {
    LwBool bValid;
    LwU8   linkId;
    LwU32  ioctrlId;
    LwU8   pllMasterLinkId;
    LwU8   pllSlaveLinkId;
    LwU32  ipVerDlPl;
} LW2080_CTRL_LWLINK_DEVICE_LINK_VALUES;

/*
 * LW2080_CTRL_CMD_LWLINK_GET_LWLINK_DEVICE_INFO
 *
 * [Out] ioctrlMask
 *    Mask of IOCTRLs discovered from PTOP device info table
 * [Out] ioctrlNumEntries
 *    Number of IOCTRL entries in the PTOP device info table
 * [Out] ioctrlSize
 *    Maximum number of entries in the PTOP device info table
 * [Out] discoveredLinks
 *    Mask of links discovered from all the IOCTRLs
 * [Out] ipVerLwlink
 *    IP revision of the LWLink HW
 * [Out] linkInfo
 *    Per link information
 */

#define LW2080_CTRL_LWLINK_GET_LWLINK_DEVICE_INFO_PARAMS_MESSAGE_ID (0x2lw)

typedef struct LW2080_CTRL_LWLINK_GET_LWLINK_DEVICE_INFO_PARAMS {
    LwU32                                 ioctrlMask;
    LwU8                                  ioctrlNumEntries;
    LwU32                                 ioctrlSize;
    LwU32                                 discoveredLinks;
    LwU32                                 ipVerLwlink;
    LW2080_CTRL_LWLINK_DEVICE_LINK_VALUES linkInfo[LW2080_CTRL_LWLINK_MAX_LINKS];
} LW2080_CTRL_LWLINK_GET_LWLINK_DEVICE_INFO_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_GET_LWLINK_DEVICE_INFO (0x2080302lw) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_GET_LWLINK_DEVICE_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_LWLINK_MAX_LINKS_PER_IOCTRL_SW    6U

typedef struct LW2080_CTRL_LWLINK_DEVICE_IP_REVISION_VALUES {
    LwU32 ipVerIoctrl;
    LwU32 ipVerMinion;
} LW2080_CTRL_LWLINK_DEVICE_IP_REVISION_VALUES;

/*
 * LW2080_CTRL_CMD_LWLINK_GET_IOCTRL_DEVICE_INFO
 *
 * [In] ioctrlIdx
 *    IOCTRL index
 * [Out] PublicId
 *    PublicId of the IOCTRL discovered
 * [Out] localDiscoveredLinks
 *    Mask of discovered links local to the IOCTRL
 * [Out] localGlobalLinkOffset
 *    Global link offsets for the locally discovered links
 * [Out] ioctrlDiscoverySize
 *    IOCTRL table size
 * [Out] numDevices
 *    Number of devices discovered from the IOCTRL
 * [Out] deviceIpRevisions
 *    IP revisions for the devices discovered in the IOCTRL
 */

#define LW2080_CTRL_LWLINK_GET_IOCTRL_DEVICE_INFO_PARAMS_MESSAGE_ID (0x2dU)

typedef struct LW2080_CTRL_LWLINK_GET_IOCTRL_DEVICE_INFO_PARAMS {
    LwU32                                        ioctrlIdx;
    LwU32                                        PublicId;
    LwU32                                        localDiscoveredLinks;
    LwU32                                        localGlobalLinkOffset;
    LwU32                                        ioctrlDiscoverySize;
    LwU8                                         numDevices;
    LW2080_CTRL_LWLINK_DEVICE_IP_REVISION_VALUES ipRevisions;
} LW2080_CTRL_LWLINK_GET_IOCTRL_DEVICE_INFO_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_GET_IOCTRL_DEVICE_INFO (0x2080302dU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_GET_IOCTRL_DEVICE_INFO_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_LWLINK_PROGRAM_LINK_SPEED
 *
 * Program LWLink Speed from OS/VBIOS
 *
 * [In] bPlatformLinerateDefined
 *    Whether line rate is defined in the platform
 * [In] platformLineRate
 *    Platform defined line rate
 * [Out] lwlinkLinkSpeed
 *    The line rate that was programmed for the links
 */
#define LW2080_CTRL_LWLINK_PROGRAM_LINK_SPEED_PARAMS_MESSAGE_ID (0x2eU)

typedef struct LW2080_CTRL_LWLINK_PROGRAM_LINK_SPEED_PARAMS {
    LwBool bPlatformLinerateDefined;
    LwU32  platformLineRate;
    LwU32  lwlinkLinkSpeed;
} LW2080_CTRL_LWLINK_PROGRAM_LINK_SPEED_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_PROGRAM_LINK_SPEED (0x2080302eU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_PROGRAM_LINK_SPEED_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_LWLINK_ARE_LINKS_TRAINED
 *
 * [In] linkMask
 *     Mask of links whose state will be checked
 * [In] bActiveOnly
 *     The input boolean to check for Link Active state
 * [Out] bIsLinkActive
 *     Boolean array to track if the link is trained
 */
#define LW2080_CTRL_LWLINK_ARE_LINKS_TRAINED_PARAMS_MESSAGE_ID (0x2fU)

typedef struct LW2080_CTRL_LWLINK_ARE_LINKS_TRAINED_PARAMS {
    LwU32  linkMask;
    LwBool bActiveOnly;
    LwBool bIsLinkActive[LW2080_CTRL_LWLINK_MAX_LINKS];
} LW2080_CTRL_LWLINK_ARE_LINKS_TRAINED_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_ARE_LINKS_TRAINED (0x2080302fU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_ARE_LINKS_TRAINED_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_LWLINK_RESET_FLAGS_ASSERT      (0x00000000)
#define LW2080_CTRL_LWLINK_RESET_FLAGS_DEASSERT    (0x00000001)
#define LW2080_CTRL_LWLINK_RESET_FLAGS_TOGGLE      (0x00000002)

/*
 * LW2080_CTRL_CMD_LWLINK_RESET_LINKS
 *
 * [In] linkMask
 *     Mask of links which need to be reset
 * [In] flags
 *     Whether to assert, de-assert or toggle the Lwlink reset
 */

#define LW2080_CTRL_LWLINK_RESET_LINKS_PARAMS_MESSAGE_ID (0x30U)

typedef struct LW2080_CTRL_LWLINK_RESET_LINKS_PARAMS {
    LwU32 linkMask;
    LwU32 flags;
} LW2080_CTRL_LWLINK_RESET_LINKS_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_RESET_LINKS (0x20803030U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_RESET_LINKS_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_LWLINK_DISABLE_DL_INTERRUPTS
 *
 * [In] linkMask
 *     Mask of links for which DL interrrupts need to be disabled
 */
#define LW2080_CTRL_LWLINK_DISABLE_DL_INTERRUPTS_PARAMS_MESSAGE_ID (0x31U)

typedef struct LW2080_CTRL_LWLINK_DISABLE_DL_INTERRUPTS_PARAMS {
    LwU32 linkMask;
} LW2080_CTRL_LWLINK_DISABLE_DL_INTERRUPTS_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_DISABLE_DL_INTERRUPTS (0x20803031U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_DISABLE_DL_INTERRUPTS_PARAMS_MESSAGE_ID" */

/*
 * Structure to store the GET_LINK_AND_CLOCK__INFO params
 *
 * [Out] bLinkConnectedToSystem
 *     Boolean indicating sysmem connection of a link
 * [Out] bLinkConnectedToPeer
 *     Boolean indicating peer connection of a link
 * [Out] bLinkReset
 *     Whether the link is in reset
 * [Out] subLinkWidth
 *     Number of lanes per sublink
 * [Out] linkState
 *     Mode of the link
 * [Out] txSublinkState
 *     Tx sublink state
 * [Out] rxSublinkState
 *     Rx sublink state
 * [Out] bLaneReversal
 *     Boolean indicating if a link's lanes are reversed
 * [Out] lwlinkLinkClockKHz
 *     Link clock value in KHz
 * [Out] lwlinkLineRateMbps
 *     Link line rate in Mbps
 * [Out] lwlinkLinkClockMhz
 *     Link clock in MHz
 * [Out] lwlinkLinkDataRateKiBps
 *     Link Data rate in KiBps
 * [Out] lwlinkRefClkType
 *     Current Lwlink refclk source
 * [Out] lwlinkReqLinkClockMhz
 *     Requested link clock value
 */
typedef struct LW2080_CTRL_LWLINK_GET_LINK_AND_CLOCK_VALUES {
    LwBool bLinkConnectedToSystem;
    LwBool bLinkConnectedToPeer;
    LwBool bLinkReset;
    LwU8   subLinkWidth;
    LwU32  linkState;
    LwU32  txSublinkState;
    LwU32  rxSublinkState;
    LwBool bLaneReversal;
    LwU32  lwlinkLinkClockKHz;
    LwU32  lwlinkLineRateMbps;
    LwU32  lwlinkLinkClockMhz;
    LwU32  lwlinkLinkDataRateKiBps;
    LwU8   lwlinkRefClkType;
    LwU32  lwlinkReqLinkClockMhz;
} LW2080_CTRL_LWLINK_GET_LINK_AND_CLOCK_VALUES;

/*
 * LW2080_CTRL_CMD_LWLINK_GET_LINK_AND_CLOCK_INFO
 *
 * [In] linkMask
 *     Mask of enabled links to loop over
 * [Out] lwlinkRefClkSpeedKHz
 *     Ref clock value n KHz
 * [Out] linkInfo
 *     Per link information
 */
#define LW2080_CTRL_LWLINK_GET_LINK_AND_CLOCK_INFO_PARAMS_MESSAGE_ID (0x32U)

typedef struct LW2080_CTRL_LWLINK_GET_LINK_AND_CLOCK_INFO_PARAMS {
    LwU32                                        linkMask;
    LwU32                                        lwlinkRefClkSpeedKHz;
    LW2080_CTRL_LWLINK_GET_LINK_AND_CLOCK_VALUES linkInfo[LW2080_CTRL_LWLINK_MAX_LINKS];
} LW2080_CTRL_LWLINK_GET_LINK_AND_CLOCK_INFO_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_GET_LINK_AND_CLOCK_INFO (0x20803032U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_GET_LINK_AND_CLOCK_INFO_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_LWLINK_SETUP_LWLINK_SYSMEM
 *
 * Updates the HSHUB sysmem config resgister state to reflect sysmem LWLinks
 *
 * [In] sysmemLinkMask
 *     Mask of discovered sysmem LWLinks
 */
#define LW2080_CTRL_LWLINK_SETUP_LWLINK_SYSMEM_PARAMS_MESSAGE_ID (0x33U)

typedef struct LW2080_CTRL_LWLINK_SETUP_LWLINK_SYSMEM_PARAMS {
    LwU32 sysmemLinkMask;
} LW2080_CTRL_LWLINK_SETUP_LWLINK_SYSMEM_PARAMS;
#define LW2080_CTRL_CMD_LWLINK_SETUP_LWLINK_SYSMEM (0x20803033U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_SETUP_LWLINK_SYSMEM_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_LWLINK_PROCESS_FORCED_CONFIGS
 *
 * Process LWLink forced configurations which includes setting of HSHUB and memory system
 *
 * [In] bLegacyForcedConfig
 *     Tracks whether the forced config is legacy forced config or chiplib config
 * [Out] bOverrideComputePeerMode
 *     Whether compute peer mode was enabled
 * [In] phase
 *     Only applicable when bLegacyForcedConfig is true
 *     Tracks the set of registers to program from the LWLink table
 * [In] linkConnection
 *     Array of chiplib configurations
 */
#define LW2080_CTRL_LWLINK_PROCESS_FORCED_CONFIGS_PARAMS_MESSAGE_ID (0x34U)

typedef struct LW2080_CTRL_LWLINK_PROCESS_FORCED_CONFIGS_PARAMS {
    LwBool bLegacyForcedConfig;
    LwBool bOverrideComputePeerMode;
    LwU32  phase;
    LwU32  linkConnection[LW2080_CTRL_LWLINK_MAX_LINKS];
} LW2080_CTRL_LWLINK_PROCESS_FORCED_CONFIGS_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_PROCESS_FORCED_CONFIGS (0x20803034U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_PROCESS_FORCED_CONFIGS_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_LWLINK_SYNC_LWLINK_SHUTDOWN_PROPS
 *
 * Sync the LWLink lane shutdown properties with GSP-RM
 *
 * [In] bLaneShutdownEnabled
 *     Whether lwlink shutdown is enabled for the chip
 * [In] bLaneShutdownOnUnload
 *     Whether lwlink shutdown should be triggered on driver unload
 */
#define LW2080_CTRL_LWLINK_SYNC_LWLINK_SHUTDOWN_PROPS_PARAMS_MESSAGE_ID (0x35U)

typedef struct LW2080_CTRL_LWLINK_SYNC_LWLINK_SHUTDOWN_PROPS_PARAMS {
    LwBool bLaneShutdownEnabled;
    LwBool bLaneShutdownOnUnload;
} LW2080_CTRL_LWLINK_SYNC_LWLINK_SHUTDOWN_PROPS_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_SYNC_LWLINK_SHUTDOWN_PROPS (0x20803035U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_SYNC_LWLINK_SHUTDOWN_PROPS_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_LWLINK_ENABLE_SYSMEM_LWLINK_ATS
 *
 * Enable ATS functionality related to LWLink sysmem if hardware support is available
 *
 * [In] notUsed
 */
#define LW2080_CTRL_LWLINK_ENABLE_SYSMEM_LWLINK_ATS_PARAMS_MESSAGE_ID (0x36U)

typedef struct LW2080_CTRL_LWLINK_ENABLE_SYSMEM_LWLINK_ATS_PARAMS {
    LwU32 notUsed;
} LW2080_CTRL_LWLINK_ENABLE_SYSMEM_LWLINK_ATS_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_ENABLE_SYSMEM_LWLINK_ATS (0x20803036U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_ENABLE_SYSMEM_LWLINK_ATS_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_LWLINK_HSHUB_GET_SYSMEM_LWLINK_MASK
 *
 *  Get the mask of Lwlink links connected to system
 *
 * [Out] sysmemLinkMask
 *      Mask of Lwlink links connected to system
 */
#define LW2080_CTRL_LWLINK_HSHUB_GET_SYSMEM_LWLINK_MASK_PARAMS_MESSAGE_ID (0x37U)

typedef struct LW2080_CTRL_LWLINK_HSHUB_GET_SYSMEM_LWLINK_MASK_PARAMS {
    LwU32 sysmemLinkMask;
} LW2080_CTRL_LWLINK_HSHUB_GET_SYSMEM_LWLINK_MASK_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_HSHUB_GET_SYSMEM_LWLINK_MASK (0x20803037U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_HSHUB_GET_SYSMEM_LWLINK_MASK_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_LWLINK_GET_SET_LWSWITCH_FLA_ADDR
 *
 * Get/Set LWSwitch FLA address
 *
 * [In] bGet
 *     Whether to get or set the LWSwitch FLA address
 * [In/Out] addr
 *     Address that is to be set or retrieved.
 */
#define LW2080_CTRL_LWLINK_GET_SET_LWSWITCH_FLA_ADDR_PARAMS_MESSAGE_ID (0x38U)

typedef struct LW2080_CTRL_LWLINK_GET_SET_LWSWITCH_FLA_ADDR_PARAMS {
    LwBool bGet;
    LW_DECLARE_ALIGNED(LwU64 addr, 8);
} LW2080_CTRL_LWLINK_GET_SET_LWSWITCH_FLA_ADDR_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_GET_SET_LWSWITCH_FLA_ADDR (0x20803038) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_GET_SET_LWSWITCH_FLA_ADDR_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_LWLINK_SYNC_LINK_MASKS_AND_VBIOS_INFO
 *
 * Syncs the different link masks and vbios defined values between CPU-RM and GSP-RM
 *
 * [in]  discoveredLinks
 *     Mask of links discovered from IOCTRLs
 *
 * [in]  connectedLinksMask
 *     Mask of links which are connected (remote present)
 *
 * [in]  bridgeSensableLinks
 *     Mask of links whose remote endpoint presence can be sensed
 *
 * [in]  bridgedLinks
 *    Mask of links which are connected (remote present)
 *    Same as connectedLinksMask, but also tracks the case where link
 *    is connected but marginal and could not initialize
 *
 * [out] initDisabledLinksMask
 *      Mask of links for which initialization is disabled
 *
 * [out] vbiosDisabledLinkMask
 *      Mask of links disabled in the VBIOS
 *
 * [out] initializedLinks
 *      Mask of initialized links
 *
 * [out] bEnableTrainingAtLoad
 *      Whether the links should be trained to active during driver load
 *
 * [out] bEnableSafeModeAtLoad
 *      Whether the links should be initialized to swcfg during driver load
 */

#define LW2080_CTRL_LWLINK_SYNC_LINK_MASKS_AND_VBIOS_INFO_PARAMS_MESSAGE_ID (0x39U)

typedef struct LW2080_CTRL_LWLINK_SYNC_LINK_MASKS_AND_VBIOS_INFO_PARAMS {
    LwU32  discoveredLinks;
    LwU32  connectedLinksMask;
    LwU32  bridgeSensableLinks;
    LwU32  bridgedLinks;
    LwU32  initDisabledLinksMask;
    LwU32  vbiosDisabledLinkMask;
    LwU32  initializedLinks;
    LwBool bEnableTrainingAtLoad;
    LwBool bEnableSafeModeAtLoad;
} LW2080_CTRL_LWLINK_SYNC_LINK_MASKS_AND_VBIOS_INFO_PARAMS;

#define LW2080_CTRL_CMD_LWLINK_SYNC_LINK_MASKS_AND_VBIOS_INFO (0x20803039U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | LW2080_CTRL_LWLINK_SYNC_LINK_MASKS_AND_VBIOS_INFO_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_LWLINK_ENABLE_LINKS
 *
 * Enable pre-topology setup on the mask of enabled links
 * This command accepts no parameters.
 */

#define LW2080_CTRL_CMD_LWLINK_ENABLE_LINKS                   (0x2080303aU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWLINK_INTERFACE_ID << 8) | 0x3a" */

/* _ctrl2080lwlink_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

