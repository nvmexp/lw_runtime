/*
 * SPDX-FileCopyrightText: Copyright (c) 2015-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0073/ctrl0073stereo.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl0073/ctrl0073base.h"

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW0073_CTRL_CMD_STEREO_DONGLE_POLL
 *
 * This command returns the value of poll stereo function
 * This provides a RmControl interface to the STEREO_DONGLE_POLL 
 * command in stereoDongleControl.
 *
 * Parameters:
 * [IN]  subDeviceInstance - This parameter specifies the subdevice instance 
 *        within the LW04_DISPLAY_COMMON parent device to which the operation 
 *        should be directed.  This parameter must specify a value between 
 *        zero and the total number of subdevices within the parent device.  
 *        This parameter should be set to zero for default behavior.
 * [OUT] control  -  poll stereo control result
 *
 * Possible status values returned are:
 *   LW_ERR_NOT_SUPPORTED - stereo is not initialized on the GPU
 */
#define LW0073_CTRL_CMD_STEREO_DONGLE_POLL (0x731701U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_STEREO_INTERFACE_ID << 8) | LW0073_CTRL_STEREO_DONGLE_POLL_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_STEREO_DONGLE_POLL_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0073_CTRL_STEREO_DONGLE_POLL_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 control;
} LW0073_CTRL_STEREO_DONGLE_POLL_PARAMS;

/*
 * LW0073_CTRL_CMD_STEREO_DONGLE_SUPPORTED
 *
 * This command returns the support status of the LW stereo emitter
 * (also known as the stereo dongle). It reports if the stereo dongle
 * is present in terms of the USB interface initialized in Resman.
 * This provides a RmControl interface to the STEREO_DONGLE_SUPPORTED 
 * command in stereoDongleControl.
 *
 * Parameters:
 * [IN]  subDeviceInstance - This parameter specifies the subdevice instance 
 *        within the LW04_DISPLAY_COMMON parent device to which the operation 
 *        should be directed.  This parameter must specify a value between 
 *        zero and the total number of subdevices within the parent device.  
 *        This parameter should be set to zero for default behavior.
 * [IN]  head               - head to be passed to stereoDongleControl
 * [IN]  bI2cEmitter        - I2C driven DT embedded emitter
 * [IN]  bForcedSupported   - GPIO23 driven emitter
 * [OUT] support            - the control word returned by stereoDongleControl
 *
 * Possible status values returned are:
 *   LW_ERR_NOT_SUPPORTED - stereo is not initialized on the GPU
 */
#define LW0073_CTRL_CMD_STEREO_DONGLE_SUPPORTED (0x731702U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_STEREO_INTERFACE_ID << 8) | LW0073_CTRL_STEREO_DONGLE_SUPPORTED_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_STEREO_DONGLE_SUPPORTED_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW0073_CTRL_STEREO_DONGLE_SUPPORTED_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  head;
    LwBool bI2cEmitter;
    LwBool bForcedSupported;
    LwU32  support;
} LW0073_CTRL_STEREO_DONGLE_SUPPORTED_PARAMS;

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW0073_CTRL_CMD_STEREO_DONGLE_SET_TIMINGS
 *
 * Sets new video mode timings
 * E.g. from display driver on mode set
 *
 * Parameters:
 * [IN]  subDeviceInstance - This parameter specifies the subdevice instance 
 *        within the LW04_DISPLAY_COMMON parent device to which the operation 
 *        should be directed.  This parameter must specify a value between 
 *        zero and the total number of subdevices within the parent device.  
 *        This parameter should be set to zero for default behavior.
 * [IN]  head      - head to be passed to stereoDongleControl 
 * [IN]  timings   - new timings to be set
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED    - stereo is not initialized on the GPU
 */
#define LW0073_CTRL_CMD_STEREO_DONGLE_SET_TIMINGS (0x731703U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_STEREO_INTERFACE_ID << 8) | LW0073_CTRL_STEREO_DONGLE_SET_TIMINGS_PARAMS_MESSAGE_ID" */

typedef struct LW0073_CTRL_STEREO_VIDEO_MODE_TIMINGS {
    LwU32 PixelClock;
    LwU16 TotalWidth;
    LwU16 VisibleImageWidth;
    LwU16 HorizontalBlankStart;
    LwU16 HorizontalBlankWidth;
    LwU16 HorizontalSyncStart;
    LwU16 HorizontalSyncWidth;
    LwU16 TotalHeight;
    LwU16 VisibleImageHeight;
    LwU16 VerticalBlankStart;
    LwU16 VerticalBlankHeight;
    LwU16 VerticalSyncStart;
    LwU16 VerticalSyncHeight;
    LwU16 InterlacedMode;
    LwU16 DoubleScanMode;

    LwU16 MonitorVendorId;
    LwU16 MonitorProductId;
} LW0073_CTRL_STEREO_VIDEO_MODE_TIMINGS;

#define LW0073_CTRL_STEREO_DONGLE_SET_TIMINGS_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW0073_CTRL_STEREO_DONGLE_SET_TIMINGS_PARAMS {
    LwU32                                 subDeviceInstance;
    LwU32                                 head;
    LW0073_CTRL_STEREO_VIDEO_MODE_TIMINGS timings;
} LW0073_CTRL_STEREO_DONGLE_SET_TIMINGS_PARAMS;

/*
 * LW0073_CTRL_CMD_STEREO_DONGLE_ACTIVATE
 *
 * stereoDongleActivate wrapper / LW_STEREO_DONGLE_ACTIVATE_DATA_ACTIVE_YES
 * Updates sbios of 3D stereo state active
 *
 * Parameters:
 * [IN]  subDeviceInstance - This parameter specifies the subdevice instance 
 *        within the LW04_DISPLAY_COMMON parent device to which the operation 
 *        should be directed.  This parameter must specify a value between 
 *        zero and the total number of subdevices within the parent device.  
 *        This parameter should be set to zero for default behavior.
 * [IN]  head                   - head to be passed to stereoDongleActivate
 * [IN]  bSDA                   - enable stereo on DDC SDA
 * [IN]  bWorkStation           - is workstation stereo?
 * [IN]  bDLP                   - is checkerboard DLP Stereo?
 * [IN]  IRPower                - IR power value
 * [IN]  flywheel               - FlyWheel value
 * [IN]  bRegIgnore             - use reg?
 * [IN]  bI2cEmitter            - Sets LW_STEREO_DONGLE_ACTVATE_DATA_I2C_EMITTER_YES and pStereo->bAegisDT
 * [IN]  bForcedSupported       - Sets LW_STEREO_DONGLE_FORCED_SUPPORTED_YES and pStereo->GPIOControlledDongle
 * [IN]  bInfoFrame             - Aegis DT with DP InfoFrame
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT    - if (head > OBJ_MAX_HEADS)
 *   LW_ERR_NOT_SUPPORTED       - stereo is not initialized on the GPU
 */
#define LW0073_CTRL_CMD_STEREO_DONGLE_ACTIVATE (0x731704U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_STEREO_INTERFACE_ID << 8) | LW0073_CTRL_STEREO_DONGLE_ACTIVATE_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_STEREO_DONGLE_ACTIVATE_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW0073_CTRL_STEREO_DONGLE_ACTIVATE_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  head;
    LwBool bSDA;
    LwBool bWorkStation;
    LwBool bDLP;
    LwU8   IRPower;
    LwU8   flywheel;
    LwBool bRegIgnore;
    LwBool bI2cEmitter;
    LwBool bForcedSupported;
    LwBool bInfoFrame;
} LW0073_CTRL_STEREO_DONGLE_ACTIVATE_PARAMS;

/*
 * LW0073_CTRL_CMD_STEREO_DONGLE_DEACTIVATE
 *
 * stereoDongleActivate wrapper / LW_STEREO_DONGLE_ACTIVATE_DATA_ACTIVE_NO
 *
 * If active count<=0 then no 3D app is running which indicates 
 * that we have really deactivated the stereo, updates sbios of 3D stereo state NOT ACTIVE.
 *
 * Parameters:
 * [IN]  subDeviceInstance - This parameter specifies the subdevice instance 
 *        within the LW04_DISPLAY_COMMON parent device to which the operation 
 *        should be directed.  This parameter must specify a value between 
 *        zero and the total number of subdevices within the parent device.  
 *        This parameter should be set to zero for default behavior.
 * [IN]  head                   - head to be passed to stereoDongleActivate
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT    - if (head > OBJ_MAX_HEADS)
 *   LW_ERR_NOT_SUPPORTED       - stereo is not initialized on the GPU
 */
#define LW0073_CTRL_CMD_STEREO_DONGLE_DEACTIVATE (0x731705U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_STEREO_INTERFACE_ID << 8) | LW0073_CTRL_STEREO_DONGLE_DEACTIVATE_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_STEREO_DONGLE_DEACTIVATE_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW0073_CTRL_STEREO_DONGLE_DEACTIVATE_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 head;
} LW0073_CTRL_STEREO_DONGLE_DEACTIVATE_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW0073_CTRL_CMD_STEREO_SET_DONGLE_AEGIS_SURROUND
 *
 * Sets mew aegis surround info
 *
 * Parameters:
 * [IN]  subDeviceInstance - This parameter specifies the subdevice instance 
 *        within the LW04_DISPLAY_COMMON parent device to which the operation 
 *        should be directed.  This parameter must specify a value between 
 *        zero and the total number of subdevices within the parent device.  
 *        This parameter should be set to zero for default behavior.
 * [IN]  aegis
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED       - stereo is not initialized on the GPU
 */
#define LW0073_CTRL_CMD_STEREO_SET_DONGLE_AEGIS_SURROUND (0x731706U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_STEREO_INTERFACE_ID << 8) | LW0073_CTRL_STEREO_SET_DONGLE_AEGIS_SURROUND_PARAMS_MESSAGE_ID" */

typedef struct LW0073_CTRL_STEREO_AEGIS_SURROUND_INFO {
    LwU32 GpuID;
    LwU32 displayID;
} LW0073_CTRL_STEREO_AEGIS_SURROUND_INFO;

#define MAX_AEGIS_DISPLAY_IN_SURROUND 16

#define LW0073_CTRL_STEREO_SET_DONGLE_AEGIS_SURROUND_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW0073_CTRL_STEREO_SET_DONGLE_AEGIS_SURROUND_PARAMS {
    LwU32                                  subDeviceInstance;
    LW0073_CTRL_STEREO_AEGIS_SURROUND_INFO aegis[MAX_AEGIS_DISPLAY_IN_SURROUND];
} LW0073_CTRL_STEREO_SET_DONGLE_AEGIS_SURROUND_PARAMS;

/*
 * LW0073_CTRL_CMD_STEREO_DONGLE_GET_ACTIVE_COUNT
 *
 * Get stereo dongle active count.
 * getActiveCount(pStereo) wrapper
 *
 * Parameters:
 * [IN]  subDeviceInstance - This parameter specifies the subdevice instance 
 *        within the LW04_DISPLAY_COMMON parent device to which the operation 
 *        should be directed.  This parameter must specify a value between 
 *        zero and the total number of subdevices within the parent device.  
 *        This parameter should be set to zero for default behavior.
 * [OUT]  activeCount
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED       - stereo is not initialized on the GPU
 */
#define LW0073_CTRL_CMD_STEREO_DONGLE_GET_ACTIVE_COUNT (0x731707U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_STEREO_INTERFACE_ID << 8) | LW0073_CTRL_STEREO_DONGLE_GET_ACTIVE_COUNT_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_STEREO_DONGLE_GET_ACTIVE_COUNT_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW0073_CTRL_STEREO_DONGLE_GET_ACTIVE_COUNT_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 activeCount;
} LW0073_CTRL_STEREO_DONGLE_GET_ACTIVE_COUNT_PARAMS;

/*
 * LW0073_CTRL_CMD_STEREO_DONGLE_DEVICE_STATUS
 *
 * dongleStatus wrapper
 * 
 * Parameters:
 * [IN]  subDeviceInstance - This parameter specifies the subdevice instance 
 *        within the LW04_DISPLAY_COMMON parent device to which the operation 
 *        should be directed.  This parameter must specify a value between 
 *        zero and the total number of subdevices within the parent device.  
 *        This parameter should be set to zero for default behavior.
 * [IN/OUT]  control
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED       - stereo is not initialized on the GPU
 */
#define LW0073_CTRL_CMD_STEREO_DONGLE_DEVICE_STATUS (0x731708U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_STEREO_INTERFACE_ID << 8) | LW0073_CTRL_STEREO_DONGLE_DEVICE_STATUS_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_STEREO_DONGLE_DEVICE_STATUS_PARAMS_MESSAGE_ID (0x8U)

typedef struct LW0073_CTRL_STEREO_DONGLE_DEVICE_STATUS_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 control;
} LW0073_CTRL_STEREO_DONGLE_DEVICE_STATUS_PARAMS;

/*
 * LW0073_CTRL_CMD_STEREO_DONGLE_DEVICE_WAKEUP
 *
 * handleSelectiveSuspend wrapper (USB_ACTIVATE)
 *
 * Parameters:
 * [IN]  subDeviceInstance - This parameter specifies the subdevice instance 
 *        within the LW04_DISPLAY_COMMON parent device to which the operation 
 *        should be directed.  This parameter must specify a value between 
 *        zero and the total number of subdevices within the parent device.  
 *        This parameter should be set to zero for default behavior.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED       - stereo is not initialized on the GPU
 */
#define LW0073_CTRL_CMD_STEREO_DONGLE_DEVICE_WAKEUP (0x731709U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_STEREO_INTERFACE_ID << 8) | LW0073_CTRL_CMD_STEREO_DONGLE_DEVICE_WAKEUP_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_STEREO_DONGLE_DEVICE_WAKEUP_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW0073_CTRL_CMD_STEREO_DONGLE_DEVICE_WAKEUP_PARAMS {
    LwU32 subDeviceInstance;
} LW0073_CTRL_CMD_STEREO_DONGLE_DEVICE_WAKEUP_PARAMS;

/*
 * LW0073_CTRL_CMD_STEREO_DONGLE_DEVICE_RESUME_SLEEP
 *
 * handleSelectiveSuspend wrapper (USB_DEACTIVATE)
 *
 * Parameters:
 * [IN]  subDeviceInstance - This parameter specifies the subdevice instance 
 *        within the LW04_DISPLAY_COMMON parent device to which the operation 
 *        should be directed.  This parameter must specify a value between 
 *        zero and the total number of subdevices within the parent device.  
 *        This parameter should be set to zero for default behavior.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED       - stereo is not initialized on the GPU
 */
#define LW0073_CTRL_CMD_STEREO_DONGLE_DEVICE_RESUME_SLEEP (0x73170aU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_STEREO_INTERFACE_ID << 8) | LW0073_CTRL_CMD_STEREO_DONGLE_DEVICE_RESUME_SLEEP_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_STEREO_DONGLE_DEVICE_RESUME_SLEEP_PARAMS_MESSAGE_ID (0xAU)

typedef struct LW0073_CTRL_CMD_STEREO_DONGLE_DEVICE_RESUME_SLEEP_PARAMS {
    LwU32 subDeviceInstance;
} LW0073_CTRL_CMD_STEREO_DONGLE_DEVICE_RESUME_SLEEP_PARAMS;

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/* _ctrl0073stereo_h_ */
