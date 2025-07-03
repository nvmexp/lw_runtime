/*
 * SPDX-FileCopyrightText: Copyright (c) 2012-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0073/ctrl0073psr.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#include "ctrl/ctrl0073/ctrl0073base.h"

/* LW04_DISPLAY_COMMON psr-display-specific control commands and parameters */

/*
 * LW0073_CTRL_CMD_PSR_ACCESS_SR_PANEL_REG
 *
 * This command is used to read/write the Self Refresh Panel's control registers.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     To be filled with Display ID on which to perform the current rd/wr access.
 *   accessType
 *     One of the valid access types defiled below.
 *   i2cAddress
 *     Slave address of the I2C device.
 *   srAddress
 *     I2C/DPCD offset that will be rd/wr in the current transaction.
 *   messageLength
 *     Returns number of bytes read/written in this transaction.
 *   pMessage
 *     Pointer to the buffer holding the data read/written in this transaction.
 *
 *   Possible status values returned are:
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 */

#define LW0073_CTRL_CMD_PSR_ACCESS_SR_PANEL_REG (0x731601U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_PSR_INTERFACE_ID << 8) | LW0073_CTRL_PSR_ACCESS_SR_PANEL_REG_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_I2C_INDEX_LENGTH_MAX        4U
#define LW0073_CTRL_I2C_MESSAGE_LENGTH_MAX      32U

#define LW0073_CTRL_PSR_ACCESS_SR_PANEL_REG_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0073_CTRL_PSR_ACCESS_SR_PANEL_REG_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 accessType;

    LwU8  i2cAddress;
    LwU32 srAddress;

    LwU32 messageLength;
    /* C form: LwU8 message[LW0073_CTRL_I2C_MESSAGE_LENGTH_MAX]; */
    LwU8  message[LW0073_CTRL_I2C_MESSAGE_LENGTH_MAX];
} LW0073_CTRL_PSR_ACCESS_SR_PANEL_REG_PARAMS;

/* valid accessType parameter values */
#define LW0073_CTRL_PSR_READ_SELF_REFRESH_PANEL_REG  (0x00000000U)
#define LW0073_CTRL_PSR_WRITE_SELF_REFRESH_PANEL_REG (0x00000001U)

/*
 * LW0073_CTRL_CMD_PSR_GET_SR_PANEL_INFO
 *
 * displayId
 *    Display ID on which this information is being requested.
 * frameLockPin
 *    Returns the frame lock pin of the panel.
 * i2cAddress
 *    Returns the i2c address on which the SR panel is attached.
 *    NOTE: applies only to LVDS panels, otherwise this field
 *          should be ignored.
 * bSelfRefreshEnabled
 *    Returns whether SR is enabled in RM.
 *
 *   Possible status values returned are:
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 *     LW_ERR_NOT_SUPPORTED
 */

#define LW0073_CTRL_CMD_PSR_GET_SR_PANEL_INFO        (0x731602U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_PSR_INTERFACE_ID << 8) | LW0073_CTRL_PSR_GET_SR_PANEL_INFO_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_PSR_GET_SR_PANEL_INFO_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW0073_CTRL_PSR_GET_SR_PANEL_INFO_PARAMS {
    LwU32  displayId;
    LwU32  frameLockPin;
    LwU8   i2cAddress;
    LwBool bSelfRefreshEnabled;
} LW0073_CTRL_PSR_GET_SR_PANEL_INFO_PARAMS;

/*
 * LW0073_CTRL_CMD_PSR_PROCESS_LWSR_PANEL_PACD
 *
 * Command to process the LWSR Panel PACD for panel self refresh (PSR).
 *
 *   displayId (in)
 *     ID of panel on which the operation is to be performed.
 *   psrType (in)
 *     Specifies whether to initialize Lwpu PSR or VESA PSR.
 *   edid_manuf_id (in)
 *     Panel EDID manuf. ID.
 *   edid_product_id (in)
 *     Panel EDID product ID.
 *   edid_serial_number (in)
 *     Panel EDID serial number
 *   bLwsrPanelPacdMatched (out)
 *     Indicates if the Panel PACD is Matched.
 *
 * Possible status values returned are:
 *   LWOS_STATUS_SUCCESS
 *   LWOS_STATUS_ERROR_ILWALID_ARGUMENT
 *   LWOS_STATUS_ERROR_NOT_SUPPORTED
 */
#define LW0073_CTRL_CMD_PSR_PROCESS_LWSR_PANEL_PACD (0x731603U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_PSR_INTERFACE_ID << 8) | LW0073_CTRL_PSR_PROCESS_LWSR_PANEL_PACD_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_PSR_PROCESS_LWSR_PANEL_PACD_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW0073_CTRL_PSR_PROCESS_LWSR_PANEL_PACD_PARAMS {
    LwU32  displayId;
    LwU32  psrType;
    LwU16  edid_manuf_id;
    LwU16  edid_product_id;
    LwU32  edid_serial_number;
    LwBool bLwsrPanelPacdMatched;
} LW0073_CTRL_PSR_PROCESS_LWSR_PANEL_PACD_PARAMS;

/*
 * LW0073_CTRL_CMD_PSR_PROCESS_LWSR_MUTEX
 *
 * Command to process the LWSR mutex for panel self refresh (PSR).
 *
 *   displayId (in)
 *     ID of panel on which the operation is to be performed.
 *   psrType (in)
 *     Specifies whether to initialize Lwpu PSR or VESA PSR.
 *   psrTconFlags (out)
 *     Returns TCON flags mapped to LW0073_CTRL_PSR_TCON_FLAGS*
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *
 * psrTconFlags is a mask composed from these values:
 *   LW0073_CTRL_PSR_TCON_FLAGS_NONE:
 *      No flags
 *   LW0073_CTRL_PSR_TCON_FLAGS_DP_FPGA:
 *      GPU is connected with DP to the FPGA
 *   LW0073_CTRL_PSR_TCON_FLAGS_LWSR_FORCE_CRASH_SYNC_DELAY_RESET:
 *      Force crash sync delay reset
 *   LW0073_CTRL_PSR_TCON_FLAGS_LWSR_HIMAX_HX8880B_MUTEX_WAR:
 *      Apply workarounds for the HIMAX HX8880-B mutex
 *   LW0073_CTRL_PSR_TCON_FLAGS_LWSR_ALTERNATIVE_ALGORITHM:
 *      Apply workarounds for Mutex alternative algorithm.
  *   LW0073_CTRL_PSR_TCON_FLAGS_LWSR_TIMING_LATCH_ONLY_IMMEDIATE:
 *      Apply workarounds for TCON only supports the timing latch immediately.
 *   LW0073_CTRL_PSR_TCON_FLAGS_LWSR_CEREBREX_CRX1200_INIT_STATUS_WAR
 *     Do not reset TCON as idle at init.
 *   LW0073_CTRL_PSR_TCON_FLAGS_LWSR_CEREBREX_CRX1200_RESUME_TO_D0_WAR
 *      Skip setting Panel Power status to D0 during GC6 exit.
 *   LW0073_CTRL_PSR_TCON_FLAGS_LWSR_CEREBREX_CRX1200_OVERRIDE_IMMEDIATE_CRASH_SYNC_WAR
 *      Override the immediate crash sync and force a delay reset and update crash sync
 *   LW0073_CTRL_PSR_TCON_FLAGS_LWSR_HIMAX_HX8880B_LENGTHEN_SRC_VBP_WAR
 *      Apply workarounds for Himax HX8880B to lengthen SRC VBP
 *   LW0073_CTRL_PSR_TCON_FLAGS_LWSR_NOVATEK_NT71870_MUTEX_DELAY_35MS
 *      Apply workarounds for Novatek NT71870 Mutex delay 35ms
 *   LW0073_CTRL_PSR_TCON_FLAGS_LWSR_HIMAX_HX8880B_MODS_SR_STATE_WAR
 *      Apply workarounds for Himax HX8880B MODS SR state.
 *   LW0073_CTRL_PSR_TCON_FLAGS_LWSR_ANAPASS_ANA38404C_PANEL_POWER_TO_BL_EN_DELAY_WAR
 *      Apply delay between panel power to BL_EN by aux.
 */
#define LW0073_CTRL_CMD_PSR_PROCESS_LWSR_MUTEX (0x731604U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_PSR_INTERFACE_ID << 8) | LW0073_CTRL_PSR_PROCESS_LWSR_MUTEX_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_PSR_PROCESS_LWSR_MUTEX_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW0073_CTRL_PSR_PROCESS_LWSR_MUTEX_PARAMS {
    LwU32 displayId;
    LwU32 psrType;
    LwU32 psrTconFlags;
} LW0073_CTRL_PSR_PROCESS_LWSR_MUTEX_PARAMS;

/* valid psrType values */
#define LW0073_CTRL_PSR_INIT_PANEL_SR_PSR_TYPE_LWSR                                              (0x00000000U)
#define LW0073_CTRL_PSR_INIT_PANEL_SR_PSR_TYPE_VESA                                              (0x00000001U)

/*
 * valid psrTconFlags values
 * Please make sure the below flag valus are the same with lwsrMutexKeys.h
*/

#define LW0073_CTRL_PSR_TCON_FLAGS_NONE                                                          (0x00000000U)
#define LW0073_CTRL_PSR_TCON_FLAGS_DP_FPGA                                                       (0x00000001U)
#define LW0073_CTRL_PSR_TCON_FLAGS_LWSR_FORCE_CRASH_SYNC_DELAY_RESET                             (0x00000002U)
#define LW0073_CTRL_PSR_TCON_FLAGS_LWSR_HIMAX_HX8880B_MUTEX_WAR                                  (0x00000004U)
#define LW0073_CTRL_PSR_TCON_FLAGS_LWSR_ALTERNATIVE_ALGORITHM                                    (0x00000008U)
#define LW0073_CTRL_PSR_TCON_FLAGS_LWSR_TIMING_LATCH_ONLY_IMMEDIATE                              (0x00000010U)
#define LW0073_CTRL_PSR_TCON_FLAGS_LWSR_CEREBREX_CRX1200_INIT_STATUS_WAR                         (0x00000020U)
#define LW0073_CTRL_PSR_TCON_FLAGS_LWSR_CEREBREX_CRX1200_SKIP_RESUME_TO_D0_WAR                   (0x00000040U)
#define LW0073_CTRL_PSR_TCON_FLAGS_LWSR_CEREBREX_CRX1200_OVERRIDE_IMMEDIATE_CRASH_SYNC_WAR       (0x00000080U)
#define LW0073_CTRL_PSR_TCON_FLAGS_LWSR_HIMAX_HX8880B_LENGTHEN_SRC_VBP_WAR                       (0x00000100U)
#define LW0073_CTRL_PSR_TCON_FLAGS_LWSR_NOVATEK_NT71870_MUTEX_DELAY_35MS                         (0x00000200U)
#define LW0073_CTRL_PSR_TCON_FLAGS_LWSR_AUO_QHD_SKIP_SETTING_MAX_RR                              (0x00000400U)
#define LW0073_CTRL_PSR_TCON_FLAGS_LWSR_HIMAX_HX8887_MUTEX_DELAY_2MS                             (0x00000800U)
#define LW0073_CTRL_PSR_TCON_FLAGS_LWSR_NOVATEK_NT71872_MUTEX_DELAY_20MS                         (0x00001000U)
#define LW0073_CTRL_PSR_TCON_FLAGS_LWSR_HIMAX_HX8880B_MODS_SR_STATE_WAR                          (0x00002000U)
#define LW0073_CTRL_PSR_TCON_FLAGS_LWSR_AUO_FHD_B173HAN01_1_HX8880B_DELAY_100MS_BEFORE_MUTEX     (0x00004000U)
#define LW0073_CTRL_PSR_TCON_FLAGS_LWSR_CEREBREX_CRX1200_DPCD_PWM_BRIGHTNESS_BIT_ALIGN_WAR       (0x00008000U)
#define LW0073_CTRL_PSR_TCON_FLAGS_LWSR_AUO_FHD_B173HAN01_1_HX8880B_DISABLE_PWM_FREQ_CTRL_BY_AUX (0x00010000U)
#define LW0073_CTRL_PSR_TCON_FLAGS_LWSR_AUO_FHD_B173HAN03_1_SET_MAX_RR_120_BUG200443747          (0x00020000U)
#define LW0073_CTRL_PSR_TCON_FLAGS_LWSR_NOVATEK_NT71872_ENABLE_AUTO_BL_ON_HDR                    (0x00040000U)
#define LW0073_CTRL_PSR_TCON_FLAGS_LWSR_ANAPASS_ANA38404C_PANEL_POWER_TO_BL_EN_DELAY_WAR         (0x00080000U)

/*
 * LW0073_CTRL_CMD_PSR_CHECK_LWSR_REGS_UNLOCKED
 *
 * Command to check if the LWSR registers are unlocked for panel self refresh (PSR).
 *
 *   displayId (in)
 *     ID of panel on which the operation is to be performed.
 *   psrType (in)
 *     Specifies whether to initialize Lwpu PSR or VESA PSR.
 *   bLwsrRegsUnlocked (out)
 *     Indicates if the LWSR mutex has been unlocked.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0073_CTRL_CMD_PSR_CHECK_LWSR_REGS_UNLOCKED                                             (0x731605U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_PSR_INTERFACE_ID << 8) | LW0073_CTRL_PSR_CHECK_LWSR_REGS_UNLOCKED_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_PSR_CHECK_LWSR_REGS_UNLOCKED_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW0073_CTRL_PSR_CHECK_LWSR_REGS_UNLOCKED_PARAMS {
    LwU32  displayId;
    LwU32  psrType;
    LwBool bLwsrRegsUnlocked;
} LW0073_CTRL_PSR_CHECK_LWSR_REGS_UNLOCKED_PARAMS;

/*
 * LW0073_CTRL_CMD_PSR_INIT_PANEL_SR
 *
 * Command to initialize panel self refresh (PSR).
 *
 *   displayId (in)
 *     ID of panel on which the operation is to be performed.
 *   psrType (in)
 *     Specifies whether to initialize Lwpu PSR or VESA PSR.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0073_CTRL_CMD_PSR_INIT_PANEL_SR (0x731606U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_PSR_INTERFACE_ID << 8) | LW0073_CTRL_PSR_INIT_PANEL_SR_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_PSR_INIT_PANEL_SR_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW0073_CTRL_PSR_INIT_PANEL_SR_PARAMS {
    LwU32 displayId;
    LwU32 psrType;
    LwU32 psrTconFlags;
} LW0073_CTRL_PSR_INIT_PANEL_SR_PARAMS;

/*
 * LW0073_CTRL_CMD_PSR_GET_VENDOR_INFO
 *
 * Command to get the PSR vendor info
 *
 *   displayId (in)
 *     ID of panel on which the operation is to be performed.
 *   pTconId (in)
 *     Specifies the pointer to the TCON IDs struct.
 *   pEdidId (in)
 *     Specifies the pointer to the EDID IDs struct.
 *   pPsrVendorInfo (in)
 *     Specifies the pointer where the vendor info needs to be copied to.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_OBJECT_NOT_FOUND
 */
#define LW0073_CTRL_CMD_PSR_GET_VENDOR_INFO (0x731607U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_PSR_INTERFACE_ID << 8) | LW0073_CTRL_PSR_GET_VENDOR_INFO_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_PSR_GET_VENDOR_INFO_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW0073_CTRL_PSR_GET_VENDOR_INFO_PARAMS {
    LwU32 displayId;
    LW_DECLARE_ALIGNED(LwP64 pTconId, 8);
    LW_DECLARE_ALIGNED(LwP64 pEdidId, 8);
    LW_DECLARE_ALIGNED(LwP64 pPsrVendorInfo, 8);
} LW0073_CTRL_PSR_GET_VENDOR_INFO_PARAMS;

/* _ctrl0073psr_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

