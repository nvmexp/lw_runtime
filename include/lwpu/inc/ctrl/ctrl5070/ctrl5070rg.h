/*
 * SPDX-FileCopyrightText: Copyright (c) 2001-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl5070/ctrl5070rg.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl5070/ctrl5070base.h"

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW5070_CTRL_CMD_GET_INTERNAL_TV_RASTER_TIMINGS
 *
 * This 'get' command returns the raster timings that should be used when
 * programming the TV raster.
 *
 *      Protocol
 *          The protocol for which the raster timings are desired.
 *
 *      PClkFreqKHz...Blank2StartY
 *          Self explanatort raster timings.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 */
#define LW5070_CTRL_CMD_GET_INTERNAL_TV_RASTER_TIMINGS                    (0x50700201) /* finn: Evaluated from "(FINN_LW50_DISPLAY_RG_INTERFACE_ID << 8) | 0x1" */

#define LW5070_CTRL_CMD_GET_INTERNAL_TV_RASTER_TIMINGS_PROTOCOL           31:0
#define LW5070_CTRL_CMD_GET_INTERNAL_TV_RASTER_TIMINGS_PROTOCOL_NTSC_M    (0x00000000)
#define LW5070_CTRL_CMD_GET_INTERNAL_TV_RASTER_TIMINGS_PROTOCOL_NTSC_J    (0x00000001)
#define LW5070_CTRL_CMD_GET_INTERNAL_TV_RASTER_TIMINGS_PROTOCOL_PAL_BDGHI (0x00000002)
#define LW5070_CTRL_CMD_GET_INTERNAL_TV_RASTER_TIMINGS_PROTOCOL_PAL_M     (0x00000003)
#define LW5070_CTRL_CMD_GET_INTERNAL_TV_RASTER_TIMINGS_PROTOCOL_PAL_N     (0x00000004)
#define LW5070_CTRL_CMD_GET_INTERNAL_TV_RASTER_TIMINGS_PROTOCOL_PAL_CN    (0x00000005)
#define LW5070_CTRL_CMD_GET_INTERNAL_TV_RASTER_TIMINGS_PROTOCOL_480P_60   (0x00000006)
#define LW5070_CTRL_CMD_GET_INTERNAL_TV_RASTER_TIMINGS_PROTOCOL_576P_50   (0x00000007)
#define LW5070_CTRL_CMD_GET_INTERNAL_TV_RASTER_TIMINGS_PROTOCOL_720P_50   (0x00000008)
#define LW5070_CTRL_CMD_GET_INTERNAL_TV_RASTER_TIMINGS_PROTOCOL_720P_60   (0x00000009)
#define LW5070_CTRL_CMD_GET_INTERNAL_TV_RASTER_TIMINGS_PROTOCOL_1080I_50  (0x0000000A)
#define LW5070_CTRL_CMD_GET_INTERNAL_TV_RASTER_TIMINGS_PROTOCOL_1080I_60  (0x0000000B)
#define LW5070_CTRL_CMD_GET_INTERNAL_TV_RASTER_TIMINGS_PROTOCOL_1080P_24  (0x0000000C)

typedef struct LW5070_CTRL_CMD_GET_INTERNAL_TV_RASTER_TIMINGS_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       Protocol;         // [IN]
// The following #defines must be kept in sync with TVPROTOCOL in disptype.h
    LwU32                       PClkFreqKHz;      // [OUT]
    LwU32                       HActive;          // [OUT]
    LwU32                       VActive;          // [OUT]
    LwU32                       Width;            // [OUT]
    LwU32                       Height;           // [OUT]
    LwU32                       SyncEndX;         // [OUT]
    LwU32                       SyncEndY;         // [OUT]
    LwU32                       BlankEndX;        // [OUT]
    LwU32                       BlankEndY;        // [OUT]
    LwU32                       BlankStartX;      // [OUT]
    LwU32                       BlankStartY;      // [OUT]
    LwU32                       Blank2EndY;       // [OUT]
    LwU32                       Blank2StartY;     // [OUT]
} LW5070_CTRL_CMD_GET_INTERNAL_TV_RASTER_TIMINGS_PARAMS;

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW5070_CTRL_CMD_GET_RG_STATUS
 *
 * This 'get' command returns the status of raster generator
 *
 *      head
 *          The head for which RG status is desired.
 *
 *      scanLocked
 *          Whether or not RG is scan (raster or frame) locked.
 *      flipLocked
 *          Whether or not RG is flip locked.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_GET_RG_STATUS                (0x50700202) /* finn: Evaluated from "(FINN_LW50_DISPLAY_RG_INTERFACE_ID << 8) | LW5070_CTRL_CMD_GET_RG_STATUS_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_GET_RG_STATUS_SCANLOCKED_NO  (0x00000000)
#define LW5070_CTRL_CMD_GET_RG_STATUS_SCANLOCKED_YES (0x00000001)

#define LW5070_CTRL_CMD_GET_RG_STATUS_FLIPLOCKED_NO  (0x00000000)
#define LW5070_CTRL_CMD_GET_RG_STATUS_FLIPLOCKED_YES (0x00000001)

#define LW5070_CTRL_CMD_GET_RG_STATUS_STALLED_NO     (0x00000000)
#define LW5070_CTRL_CMD_GET_RG_STATUS_STALLED_YES    (0x00000001)

#define LW5070_CTRL_CMD_GET_RG_STATUS_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW5070_CTRL_CMD_GET_RG_STATUS_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       head;

    LwU32                       scanLocked;           // [OUT]
    LwU32                       flipLocked;           // [OUT]
    LwU32                       rgStalled;
} LW5070_CTRL_CMD_GET_RG_STATUS_PARAMS;

/*
 * LW5070_CTRL_CMD_UNDERFLOW_PARAMS
 *
 * This structure contains data for
 * LW5070_CTRL_CMD_SET_RG_UNDERFLOW_PROP and
 * LW5070_CTRL_CMD_GET_RG_UNDERFLOW_PROP CTRL calls
 *
 *   head
 *     The head for which RG underflow properties needed to be set/get.
 *     Valid values for this parameter are 0 to LW5070_CTRL_CMD_MAX_HEADS.
 *   enable
 *     _SET_RG_: Whether to enable or disable RG underflow reporting.
 *     _GET_RG_: Whether or not RG underflow reporting is enabled.
 *   underflow
 *     _SET_RG_: Clear underflow (TRUE) or leave it alone (FALSE).
 *     _GET_RG_: RG underflow underflowed (TRUE) or not underflowed (FALSE).
 *   mode
 *     _SET_RG_: What mode to use when underflow oclwrs. This is
 *               independent from enable field. This is always active.
 *     _GET_RG_: What mode is used when underflow oclwrs. This is
 *               independent from enable field. This is always active.
 */
typedef struct LW5070_CTRL_CMD_UNDERFLOW_PARAMS {
    LwU32 head;
    LwU32 enable;
    LwU32 underflow;
    LwU32 mode;
} LW5070_CTRL_CMD_UNDERFLOW_PARAMS;

#define LW5070_CTRL_CMD_UNDERFLOW_PROP_ENABLED_NO          (0x00000000)
#define LW5070_CTRL_CMD_UNDERFLOW_PROP_ENABLED_YES         (0x00000001)
#define LW5070_CTRL_CMD_UNDERFLOW_PROP_UNDERFLOWED_NO      (0x00000000)
#define LW5070_CTRL_CMD_UNDERFLOW_PROP_UNDERFLOWED_YES     (0x00000001)
#define LW5070_CTRL_CMD_UNDERFLOW_PROP_MODE_REPEAT         (0x00000000)
#define LW5070_CTRL_CMD_UNDERFLOW_PROP_MODE_RED            (0x00000001)
#define LW5070_CTRL_CMD_UNDERFLOW_PROP_ENABLE_NO           (0x00000000)
#define LW5070_CTRL_CMD_UNDERFLOW_PROP_ENABLE_YES          (0x00000001)
#define LW5070_CTRL_CMD_UNDERFLOW_PROP_CLEAR_UNDERFLOW_NO  (0x00000000)
#define LW5070_CTRL_CMD_UNDERFLOW_PROP_CLEAR_UNDERFLOW_YES (0x00000001)

/*
 * LW5070_CTRL_CMD_GET_RG_UNDERFLOW_PROP
 *
 * This command returns the underflow reporting parameters inside
 * LW5070_CTRL_CMD_UNDERFLOW_PARAMS structure
 *
 *   underflowParams
 *     Contains data for underflow logging.
 *     Check LW5070_CTRL_CMD_UNDERFLOW_PARAMS structure.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_GET_RG_UNDERFLOW_PROP              (0x50700203) /* finn: Evaluated from "(FINN_LW50_DISPLAY_RG_INTERFACE_ID << 8) | LW5070_CTRL_CMD_GET_RG_UNDERFLOW_PROP_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_GET_RG_UNDERFLOW_PROP_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW5070_CTRL_CMD_GET_RG_UNDERFLOW_PROP_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS      base;
    LW5070_CTRL_CMD_UNDERFLOW_PARAMS underflowParams;
} LW5070_CTRL_CMD_GET_RG_UNDERFLOW_PROP_PARAMS;


/*
 * LW5070_CTRL_CMD_SET_RG_UNDERFLOW_PROP
 *
 * This command sets up the underflow parameters using
 * LW5070_CTRL_CMD_UNDERFLOW_PARAMS structure
 *
 *   underflowParams
 *     Contains data for underflow logging.
 *     Check LW5070_CTRL_CMD_UNDERFLOW_PARAMS structure.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_SET_RG_UNDERFLOW_PROP (0x50700204) /* finn: Evaluated from "(FINN_LW50_DISPLAY_RG_INTERFACE_ID << 8) | LW5070_CTRL_CMD_SET_RG_UNDERFLOW_PROP_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_SET_RG_UNDERFLOW_PROP_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW5070_CTRL_CMD_SET_RG_UNDERFLOW_PROP_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS      base;
    LW5070_CTRL_CMD_UNDERFLOW_PARAMS underflowParams;
} LW5070_CTRL_CMD_SET_RG_UNDERFLOW_PROP_PARAMS;


/*
 * LW5070_CTRL_CMD_GET_RG_FLIPLOCK_PROP
 *
 * This command gets the timing parameters associated with the lockout period.
 *
 *      head
 *          The head for which RG fliplock properties are desired.
 *
 *      maxSwapLockoutSkew
 *          The maximum possible skew between the swap lockout signals for all
 *          heads which are fliplocked to this head.
 *
 *      swapLockoutStart
 *          Determines the start of the start lockout period, expressed as the
 *          number of lines before the end of the frame. The minimum allowed
 *          value is 1.

 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_GET_RG_FLIPLOCK_PROP (0x50700205) /* finn: Evaluated from "(FINN_LW50_DISPLAY_RG_INTERFACE_ID << 8) | LW5070_CTRL_CMD_GET_RG_FLIPLOCK_PROP_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_GET_RG_FLIPLOCK_PROP_MAX_SWAP_LOCKOUT_SKEW          9:0

#define LW5070_CTRL_CMD_GET_RG_FLIPLOCK_PROP_SWAP_LOCKOUT_START            15:0

#define LW5070_CTRL_CMD_GET_RG_FLIPLOCK_PROP_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW5070_CTRL_CMD_GET_RG_FLIPLOCK_PROP_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       head;

    LwU32                       maxSwapLockoutSkew;
    LwU32                       swapLockoutStart;
} LW5070_CTRL_CMD_GET_RG_FLIPLOCK_PROP_PARAMS;

/*
 * LW5070_CTRL_CMD_SET_RG_FLIPLOCK_PROP
 *
 * This command sets the timing parameters associated with the lockout period.
 *
 *      head
 *          The head for which RG fliplock properties are desired.
 *
 *      maxSwapLockoutSkew
 *          The maximum possible skew between the swap lockout signals for all
 *          heads which are fliplocked to this head.
 *
 *      swapLockoutStart
 *          Determines the start of the start lockout period, expressed as the
 *          number of lines before the end of the frame. The minimum allowed
 *          value is 1.

 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_SET_RG_FLIPLOCK_PROP                            (0x50700206) /* finn: Evaluated from "(FINN_LW50_DISPLAY_RG_INTERFACE_ID << 8) | LW5070_CTRL_CMD_SET_RG_FLIPLOCK_PROP_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_SET_RG_FLIPLOCK_PROP_MAX_SWAP_LOCKOUT_SKEW      9:0
#define LW5070_CTRL_CMD_SET_RG_FLIPLOCK_PROP_MAX_SWAP_LOCKOUT_SKEW_INIT (0x00000000)

#define LW5070_CTRL_CMD_SET_RG_FLIPLOCK_PROP_SWAP_LOCKOUT_START         15:0
#define LW5070_CTRL_CMD_SET_RG_FLIPLOCK_PROP_SWAP_LOCKOUT_START_INIT    (0x00000000)

#define LW5070_CTRL_CMD_SET_RG_FLIPLOCK_PROP_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW5070_CTRL_CMD_SET_RG_FLIPLOCK_PROP_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       head;

    LwU32                       maxSwapLockoutSkew;
    LwU32                       swapLockoutStart;
} LW5070_CTRL_CMD_SET_RG_FLIPLOCK_PROP_PARAMS;

/*
 * LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN
 *
 * This command returns which lockpin has been connected for the specified
 * subdevice in the current SLI and/or framelock configuration.
 *
 *      head
 *          The head for which the locking is associated with
 *
 *      masterScanLock
 *          Indicate the connection status and pin number of master scanlock
 *
 *      slaveScanLock
 *          Indicate the connection status and pin number of slave scanlock
 *
 *      flipLock
 *          Indicate the connection status and pin number of fliplock
 *
 *      stereoLock
 *          Indicate the connection status and pin number of stereo lock
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 */
#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN                                (0x50700207) /* finn: Evaluated from "(FINN_LW50_DISPLAY_RG_INTERFACE_ID << 8) | LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_MASTER_SCAN_LOCK_CONNECTED     0:0
#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_MASTER_SCAN_LOCK_CONNECTED_NO  (0x00000000)
#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_MASTER_SCAN_LOCK_CONNECTED_YES (0x00000001)
#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_MASTER_SCAN_LOCK_PIN           3:1

#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_SLAVE_SCAN_LOCK_CONNECTED      0:0
#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_SLAVE_SCAN_LOCK_CONNECTED_NO   (0x00000000)
#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_SLAVE_SCAN_LOCK_CONNECTED_YES  (0x00000001)
#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_SLAVE_SCAN_LOCK_PIN            3:1

#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_FLIP_LOCK_CONNECTED       0:0
#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_FLIP_LOCK_CONNECTED_NO         (0x00000000)
#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_FLIP_LOCK_CONNECTED_YES        (0x00000001)
#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_FLIP_LOCK_PIN             3:1

#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_STEREO_LOCK_CONNECTED     0:0
#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_STEREO_LOCK_CONNECTED_NO       (0x00000000)
#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_STEREO_LOCK_CONNECTED_YES      (0x00000001)
#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_STEREO_LOCK_PIN           3:1

#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       head;

    LwU32                       masterScanLock;
    LwU32                       slaveScanLock;
    LwU32                       flipLock;
    LwU32                       stereoLock;
} LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_PARAMS;

/*
 * LW5070_CTRL_CMD_SET_VIDEO_STATUS
 *
 * This command is used to set the current video playback status for use
 * by the Display Power Saving (lwDPS) feature.  The playback status is
 * used to maximize power savings by altering the DFP refresh rate used for
 * video playback.
 *
 *   displayId
 *     This parameter specifies the ID of the video playback display.
 *     Only one display may be indicated in this parameter.
 *   clientId
 *     This parameter specifies the opaque client ID associated with
 *     the video playback application.
 *   mode
 *     This parameter specifies the video playback mode.  Valid values
 *     for this parameter include:
 *       LW5070_CTRL_DFP_SET_VIDEO_STATUS_MODE_NON_FULLSCREEN
 *         This value indicates that there is either no video playback or
 *         that video playback is windowed.
 *       LW5070_CTRL_DFP_SET_VIDEO_STATUS_MODE_FULLSCREEN
 *         This value indicates that video playback is fullscreen.
 *       LW5070_CTRL_DFP_SET_VIDEO_STATUS_MODE_D3D
 *         This value indicates that there is a D3D app started.
 *   frameRate
 *     The parameter indicates the current video playback frame rate.
 *     The value is a 32 bit unsigned fixed point number, 24 bit unsigned
 *     integer (bits 31:7), and 8 fraction bits (bits 7:0), measured in
 *     number of frames per second.
 *     A value of 0 indicates that video playback is stopped or not playing.
 *   frameRateAlarmUpperLimit
 *     The parameter indicates the upper limit which will can be tolerated in
 *     notifying frame rate change. If the frame rate changed but is still
 *     below the limit. The newer frame rate doesn't have to be set till it's
 *     over the limit.
 *     The value is a 32 bit unsigned fixed point number, 24 bit unsigned
 *     integer (bits 31:7), and 8 fraction bits (bits 7:0), measured in
 *     number of frames per second.
 *     A value of 0 indicates no tolerance of frame rate notifying. Instant
 *     frame rate has to be set when it has changed.
 *   frameRateAlarmLowerLimit
 *     The parameter indicates the lower limit which will can be tolerated in
 *     notifying frame rate change. If the frame rate changed but is still
 *     above the limit. The newer frame rate doesn't have to be set till it's
 *     below the limit.
 *     The value is a 32 bit unsigned fixed point number, 24 bit unsigned
 *     integer (bits 31:7), and 8 fraction bits (bits 7:0), measured in
 *     number of frames per second.
 *     A value of 0 indicates no tolerance of frame rate notifying. Instant
 *     frame rate has to be set when it has changed.
 *
 *     The frameRateAlarm limit values can be used by the video client to
 *     indicate the the range in which frame rate changes do not require
 *     notification (i.e. frame rates outside these limits will result in
 *     notification).
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW5070_CTRL_CMD_SET_VIDEO_STATUS (0x50700209) /* finn: Evaluated from "(FINN_LW50_DISPLAY_RG_INTERFACE_ID << 8) | LW5070_CTRL_DFP_SET_VIDEO_STATUS_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_DFP_SET_VIDEO_STATUS_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW5070_CTRL_DFP_SET_VIDEO_STATUS_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;

    LwU32                       displayId;
    LwU32                       clientId;
    LwU32                       mode;
    LwU32                       frameRate;
    LwU32                       frameRateAlarmUpperLimit;
    LwU32                       frameRateAlarmLowerLimit;
} LW5070_CTRL_DFP_SET_VIDEO_STATUS_PARAMS;

/* valid mode flags */
#define LW5070_CTRL_DFP_SET_VIDEO_STATUS_MODE_NON_FULLSCREEN                              (0x00000000)
#define LW5070_CTRL_DFP_SET_VIDEO_STATUS_MODE_FULLSCREEEN                                 (0x00000001)
#define LW5070_CTRL_DFP_SET_VIDEO_STATUS_MODE_D3D                                         (0x00000002)

/*
 * LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_STATELESS
 *
 * This command returns which set of lockpins needs to be used in order to
 * successfully raster lock two heads on different GPUs together.  The
 * second GPU is not inferred from linked SLI state, if any, and needs to
 * be specified explicitly.
 *
 *   head
 *     The local head to be locked with the peer head.
 *
 *   peer.hDisplay
 *     The handle identifying a display object allocated on another
 *     GPU.  It specifies the peer of interest with a subdevice
 *     index (see below) and needs to be be distinct from the handle
 *     supplied directly to LwRmControl().
 *
 *   peer.subdeviceIndex
 *     The index of the peer subdevice of interest.
 *
 *   peer.head
 *     The peer head to be locked with the local head.
 *
 *   masterScanLockPin
 *   slaveScanLockPin
 *     Returns the master and slave scanlock pins that would need to
 *     be used to lock the specified heads together, if any.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_OBJECT_PARENT
 */
#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_STATELESS                                (0x5070020a) /* finn: Evaluated from "(FINN_LW50_DISPLAY_RG_INTERFACE_ID << 8) | LW5070_CTRL_GET_RG_CONNECTED_LOCKPIN_STATELESS_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_STATELESS_MASTER_SCAN_LOCK_CONNECTED     0:0
#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_STATELESS_MASTER_SCAN_LOCK_CONNECTED_NO  (0x00000000)
#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_STATELESS_MASTER_SCAN_LOCK_CONNECTED_YES (0x00000001)
#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_STATELESS_MASTER_SCAN_LOCK_PIN           2:1

#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_STATELESS_SLAVE_SCAN_LOCK_CONNECTED      0:0
#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_STATELESS_SLAVE_SCAN_LOCK_CONNECTED_NO   (0x00000000)
#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_STATELESS_SLAVE_SCAN_LOCK_CONNECTED_YES  (0x00000001)
#define LW5070_CTRL_CMD_GET_RG_CONNECTED_LOCKPIN_STATELESS_SLAVE_SCAN_LOCK_PIN            2:1

#define LW5070_CTRL_GET_RG_CONNECTED_LOCKPIN_STATELESS_PARAMS_MESSAGE_ID (0xAU)

typedef struct LW5070_CTRL_GET_RG_CONNECTED_LOCKPIN_STATELESS_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       head;

    struct {
        LwHandle hDisplay;
        LwU32    subdeviceIndex;
        LwU32    head;
    } peer;

    LwU32 masterScanLock;
    LwU32 slaveScanLock;
} LW5070_CTRL_GET_RG_CONNECTED_LOCKPIN_STATELESS_PARAMS;

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

/*
 * LW5070_CTRL_CMD_GET_PINSET_LOCKPINS
 *
 * Get the lockpins for the specified pinset.
 *
 *   pinset [in]
 *     The pinset whose corresponding lockpin numbers need to be determined
 *     must be specified with this parameter.
 *
 *   scanLockPin [out]
 *     The scanlock lockpin (rasterlock or framelock) index, which can be
 *     either master or slave, is returned in this parameter.
 *
 *   flipLockPin [out]
 *     The fliplock lockpin index, is returned in this parameter.
 *
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW5070_CTRL_CMD_GET_PINSET_LOCKPINS                (0x5070020b) /* finn: Evaluated from "(FINN_LW50_DISPLAY_RG_INTERFACE_ID << 8) | LW5070_CTRL_GET_PINSET_LOCKPINS_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_GET_PINSET_LOCKPINS_SCAN_LOCK_PIN_NONE 0xffffffff

#define LW5070_CTRL_GET_PINSET_LOCKPINS_FLIP_LOCK_PIN_NONE 0xffffffff

#define LW5070_CTRL_GET_PINSET_LOCKPINS_PARAMS_MESSAGE_ID (0xBU)

typedef struct LW5070_CTRL_GET_PINSET_LOCKPINS_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       pinset;
    LwU32                       scanLockPin;
    LwU32                       flipLockPin;
} LW5070_CTRL_GET_PINSET_LOCKPINS_PARAMS;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW5070_CTRL_CMD_GET_RG_SCAN_LINE
 *
 * This 'get' command returns the current scan line value from raster generator 
 *
 *      head
 *          The head for which current scan line number is desired.
 * 
 *      scanLine
 *          Current scan line number.
 *
 *      ilwblank
 *          Whether or not in vblank.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC
 */
#define LW5070_CTRL_CMD_GET_RG_SCAN_LINE               (0x5070020c) /* finn: Evaluated from "(FINN_LW50_DISPLAY_RG_INTERFACE_ID << 8) | LW5070_CTRL_CMD_GET_RG_SCAN_LINE_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_CMD_GET_RG_SCAN_LINE_IN_VBLANK_NO  (0x00000000)
#define LW5070_CTRL_CMD_GET_RG_SCAN_LINE_IN_VBLANK_YES (0x00000001)

#define LW5070_CTRL_CMD_GET_RG_SCAN_LINE_PARAMS_MESSAGE_ID (0xLW)

typedef struct LW5070_CTRL_CMD_GET_RG_SCAN_LINE_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       head;
    LwU32                       scanLine;             // [OUT]
    LwU32                       ilwblank;             // [OUT]
} LW5070_CTRL_CMD_GET_RG_SCAN_LINE_PARAMS;

/*
 * LW5070_CTRL_CMD_GET_FRAMELOCK_HEADER_LOCKPINS
 *
 * This command returns FrameLock header lock pin information.
 * Lock pin index returned by this command corresponds to the 
 * evo lock pin number. Example - lock pin index 0 means 
 * LOCKPIN_0. 
 *
 *   frameLockPin [out]
 *     This parameter returns the FrameLock pin index 
 *     connected to FrameLock header.
 *
 *   rasterLockPin [out]
 *     This parameter returns the RasterLock pin index 
 *     connected to FrameLock header.
 *    
 *   flipLockPin [out]
 *     This parameter returns the FlipLock pin index 
 *     connected to FrameLock header.
 *
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED   
 */

#define LW5070_CTRL_CMD_GET_FRAMELOCK_HEADER_LOCKPINS                  (0x5070020d) /* finn: Evaluated from "(FINN_LW50_DISPLAY_RG_INTERFACE_ID << 8) | LW5070_CTRL_GET_FRAMELOCK_HEADER_LOCKPINS_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_GET_FRAMELOCK_HEADER_LOCKPINS_FRAME_LOCK_PIN_NONE  (0xffffffff)
#define LW5070_CTRL_GET_FRAMELOCK_HEADER_LOCKPINS_RASTER_LOCK_PIN_NONE (0xffffffff)
#define LW5070_CTRL_GET_FRAMELOCK_HEADER_LOCKPINS_FLIP_LOCK_PIN_NONE   (0xffffffff)
#define LW5070_CTRL_GET_FRAMELOCK_HEADER_LOCKPINS_PARAMS_MESSAGE_ID (0xDU)

typedef struct LW5070_CTRL_GET_FRAMELOCK_HEADER_LOCKPINS_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       frameLockPin;
    LwU32                       rasterLockPin;
    LwU32                       flipLockPin;
} LW5070_CTRL_GET_FRAMELOCK_HEADER_LOCKPINS_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW5070_CTRL_CMD_GET_STEREO_PHASE
 *
 * This command returns the current stereo phase mask of a given
 * headmask, all fetched at the same vblank to make them comparable.
 * Bit set in the phaseMask means left eye is visible.
 * Bit not set in the phaseMask means right eye is visible.
 *
 *   headMask [in]
 *     This parameter defines the heads for which a stereo phase information
 *     should be returned. Prerequisite for that is that the heads are all    
 *     scanlocked among each other- Without that the stereo phase relationship
 *     is not defined.
 *
 *   phaseMask [out]
 *     This parameter returns one bit of information for every head set in the
 *     headMask parameter.
 *     LW5070_CTRL_GET_STEREO_PHASE_LEFT_EYE indicates the left eye is shown.
 *     LW5070_CTRL_GET_STEREO_PHASE_RIGHT_EYE indicates the right eye is shown.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_GENERIC
 *   LW_ERR_NOT_SUPPORTED   
 *   LW_ERR_ILWALID_LOCK_STATE
*/

#define LW5070_CTRL_CMD_GET_STEREO_PHASE       (0x5070020e) /* finn: Evaluated from "(FINN_LW50_DISPLAY_RG_INTERFACE_ID << 8) | LW5070_CTRL_GET_STEREO_PHASE_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_GET_STEREO_PHASE_LEFT_EYE  1
#define LW5070_CTRL_GET_STEREO_PHASE_RIGHT_EYE 0
#define LW5070_CTRL_GET_STEREO_PHASE_PARAMS_MESSAGE_ID (0xEU)

typedef struct LW5070_CTRL_GET_STEREO_PHASE_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       headMask;
    LwU32                       phaseMask;
} LW5070_CTRL_GET_STEREO_PHASE_PARAMS;

/*
* LW5070_CTRL_CMD_SET_RG_AND_ELV_BLOCK_UNSTALL
*
* This 'set' command triggers an RG unstall
*
*      head
*          The head for which RG status is desired.
*
*      bAllowOneElv
*          Allows one ELV to happen during the unstall.
*
*      allowElvSubdeviceMask
*          Specifies which subdevices to allow the ELV to occur on.
*
* Possible status values returned are:
*      LW_OK
*      LW_ERR_ILWALID_PARAM_STRUCT
*      LW_ERR_ILWALID_ARGUMENT
*      LW_ERR_GENERIC
*      LW_ERR_NOT_SUPPORTED
*/
#define LW5070_CTRL_CMD_SET_RG_UNSTALL_AND_ELV_BLOCK (0x5070020f) /* finn: Evaluated from "(FINN_LW50_DISPLAY_RG_INTERFACE_ID << 8) | LW5070_CTRL_SET_RG_UNSTALL_AND_ELV_BLOCK_PARAMS_MESSAGE_ID" */

#define LW5070_CTRL_SET_RG_UNSTALL_AND_ELV_BLOCK_PARAMS_MESSAGE_ID (0xFU)

typedef struct LW5070_CTRL_SET_RG_UNSTALL_AND_ELV_BLOCK_PARAMS {
    LW5070_CTRL_CMD_BASE_PARAMS base;
    LwU32                       head;
    LwBool                      bAllowOneElv;
    LwU32                       allowElvSubdeviceMask;
} LW5070_CTRL_SET_RG_UNSTALL_AND_ELV_BLOCK_PARAMS;

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/* _ctrl5070rg_h_ */
