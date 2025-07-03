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
// Source file: ctrl/ctrl0073/ctrl0073dp.finn
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
 * LW0073_CTRL_CMD_DP_AUXCH_CTRL
 *
 * This command can be used to perform an aux channel transaction to the
 * displayPort receiver.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the display for which the dfp
 *     caps should be returned.  The display ID must a dfp display.
 *     If more than one displayId bit is set or the displayId is not a dfp,
 *     this call will return LW_ERR_ILWALID_ARGUMENT.
 *   bAddrOnly
 *     If set to LW_TRUE, this parameter prompts an address-only
 *     i2c-over-AUX transaction to be issued, if supported.  Else the
 *     call fails with LWOS_STATUS_ERR_NOT_SUPPORTED.  The size parameter is
 *     expected to be 0 for address-only transactions.
 *   cmd
 *     This parameter is an input to this command.  The cmd parameter follows
 *     Section 2.4 AUX channel syntax in the DisplayPort spec.
 *     Here are the current defined fields:
 *       LW0073_CTRL_DP_AUXCH_CMD_TYPE
 *         This specifies the request command transaction
 *           LW0073_CTRL_DP_AUXCH_CMD_TYPE_I2C
 *             Set this value to indicate a I2C transaction.
 *           LW0073_CTRL_DP_AUXCH_CMD_TYPE_AUX
 *             Set this value to indicate a DisplayPort transaction.
 *       LW0073_CTRL_DP_AUXCH_CMD_I2C_MOT
 *         This field is dependent on LW0073_CTRL_DP_AUXCH_CMD_TYPE.
 *         It is only valid if LW0073_CTRL_DP_AUXCH_CMD_TYPE_I2C
 *         is specified above and indicates a middle of transaction.
 *         In the case of AUX, this field should be set to zero.  The valid
 *         values are:
 *           LW0073_CTRL_DP_AUXCH_CMD_I2C_MOT_FALSE
 *             The I2C transaction is not in the middle of a transaction.
 *           LW0073_CTRL_DP_AUXCH_CMD_I2C_MOT_TRUE
 *             The I2C transaction is in the middle of a transaction.
 *       LW0073_CTRL_DP_AUXCH_CMD_REQ_TYPE
 *         The request type specifies if we are doing a read/write or write
 *         status request:
 *           LW0073_CTRL_DP_AUXCH_CMD_REQ_TYPE_READ
 *             An I2C or AUX channel read is requested.
 *           LW0073_CTRL_DP_AUXCH_CMD_REQ_TYPE_WRITE
 *             An I2C or AUX channel write is requested.
 *           LW0073_CTRL_DP_AUXCH_CMD_REQ_TYPE_WRITE_STATUS
 *             An I2C write status request desired.  This value should
 *             not be set in the case of an AUX CH request and only applies
 *             to I2C write transaction command.
 *   addr
 *     This parameter is an input to this command.  The addr parameter follows
 *     Section 2.4 in DisplayPort spec and the client should refer to the valid
 *     address in DisplayPort spec.  Only the first 20 bits are valid.
 *   data[]
 *     In the case of a read transaction, this parameter returns the data from
 *     transaction request.  In the case of a write transaction, the client
 *     should write to this buffer for the data to send.  The max # of bytes
 *     allowed is LW0073_CTRL_DP_AUXCH_MAX_DATA_SIZE.
 *   size
 *     Specifies how many data bytes to read/write depending on the transaction type.
 *     The input size value should be indexed from 0.  That means if you want to read
 *     1 byte -> size = 0, 2 bytes -> size = 1, 3 bytes -> size = 2, up to 16 bytes
 *     where size = 15.  On return, this parameter returns total number of data bytes
 *     successfully read/written from/to the transaction (indexed from 1).  That is,
 *     if you successfully requested 1 byte, you would send down size = 0.  On return,
 *     you should expect size = 1 if all 1 byte were successfully read. (Note that
 *     it is valid for a display to reply with fewer than the requested number of
 *     bytes; in that case, it is up to the client to make a new request for the
 *     remaining bytes.)
 *   replyType
 *     This parameter is an output to this command.  It returns the auxChannel
 *     status after the end of the aux Ch transaction.  The valid values are
 *     based on the DisplayPort spec:
 *       LW0073_CTRL_DP_AUXCH_REPLYTYPE_ACK
 *         In the case of a write,
 *         AUX: write transaction completed and all data bytes written.
 *         I2C: return size bytes has been written to i2c slave.
 *         In the case of a read, return of ACK indicates ready to reply
 *         another read request.
 *       LW0073_CTRL_DP_AUXCH_REPLYTYPE_NACK
 *         In the case of a write, first return size bytes have been written.
 *         In the case of a read, implies that does not have requested data
 *         for the read request transaction.
 *       LW0073_CTRL_DP_AUXCH_REPLYTYPE_DEFER
 *         Not ready for the write/read request and client should retry later.
 *       LW0073_CTRL_DP_DISPLAYPORT_AUXCH_REPLYTYPE_I2CNACK
 *         Applies to I2C transactions only.  For I2C write transaction:
 *         has written the first return size bytes to I2C slave before getting
 *         NACK.  For a read I2C transaction, the I2C slave has NACKED the I2C
 *         address.
 *       LW0073_CTRL_DP_AUXCH_REPLYTYPE_I2CDEFER
 *         Applicable to I2C transactions.  For I2C write and read
 *         transactions, I2C slave has yet to ACK or NACK the I2C transaction.
 *       LW0073_CTRL_DP_AUXCH_REPLYTYPE_TIMEOUT
 *         The receiver did not respond within the timeout period defined in
 *         the DisplayPort 1.1a specification.
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
#define LW0073_CTRL_CMD_DP_AUXCH_CTRL      (0x731341U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_DP_AUXCH_CTRL_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DP_AUXCH_MAX_DATA_SIZE 16U
#define LW0073_CTRL_DP_AUXCH_CTRL_PARAMS_MESSAGE_ID (0x41U)

typedef struct LW0073_CTRL_DP_AUXCH_CTRL_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  displayId;
    LwBool bAddrOnly;
    LwU32  cmd;
    LwU32  addr;
    LwU8   data[LW0073_CTRL_DP_AUXCH_MAX_DATA_SIZE];
    LwU32  size;
    LwU32  replyType;
    LwU32  retryTimeMs;
} LW0073_CTRL_DP_AUXCH_CTRL_PARAMS;

#define LW0073_CTRL_DP_AUXCH_CMD_TYPE                          3:3
#define LW0073_CTRL_DP_AUXCH_CMD_TYPE_I2C               (0x00000000U)
#define LW0073_CTRL_DP_AUXCH_CMD_TYPE_AUX               (0x00000001U)
#define LW0073_CTRL_DP_AUXCH_CMD_I2C_MOT                       2:2
#define LW0073_CTRL_DP_AUXCH_CMD_I2C_MOT_FALSE          (0x00000000U)
#define LW0073_CTRL_DP_AUXCH_CMD_I2C_MOT_TRUE           (0x00000001U)
#define LW0073_CTRL_DP_AUXCH_CMD_REQ_TYPE                      1:0
#define LW0073_CTRL_DP_AUXCH_CMD_REQ_TYPE_WRITE         (0x00000000U)
#define LW0073_CTRL_DP_AUXCH_CMD_REQ_TYPE_READ          (0x00000001U)
#define LW0073_CTRL_DP_AUXCH_CMD_REQ_TYPE_WRITE_STATUS  (0x00000002U)

#define LW0073_CTRL_DP_AUXCH_ADDR                             20:0

#define LW0073_CTRL_DP_AUXCH_REPLYTYPE                         3:0
#define LW0073_CTRL_DP_AUXCH_REPLYTYPE_ACK              (0x00000000U)
#define LW0073_CTRL_DP_AUXCH_REPLYTYPE_NACK             (0x00000001U)
#define LW0073_CTRL_DP_AUXCH_REPLYTYPE_DEFER            (0x00000002U)
#define LW0073_CTRL_DP_AUXCH_REPLYTYPE_TIMEOUT          (0x00000003U)
#define LW0073_CTRL_DP_AUXCH_REPLYTYPE_I2CNACK          (0x00000004U)
#define LW0073_CTRL_DP_AUXCH_REPLYTYPE_I2CDEFER         (0x00000008U)

//This is not the register field, this is software failure case when we
//have invalid argument
#define LW0073_CTRL_DP_AUXCH_REPLYTYPE_ILWALID_ARGUMENT (0xffffffffU)

/*
 * LW0073_CTRL_CMD_DP_AUXCH_SET_SEMA
 *
 * This command can be used to set the semaphore in order to gain control of
 * the aux channel.  This control is only used in HW verification.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the display for which the dfp
 *     caps should be returned.  The display ID must a dfp display
 *     as determined with the LW0073_CTRL_CMD_SPECIFIC_GET_TYPE command.
 *     If more than one displayId bit is set or the displayId is not a dfp,
 *     this call will return LW_ERR_ILWALID_ARGUMENT.
 *   owner
 *     This parameter is an input to this command.
 *     Here are the current defined fields:
 *       LW0073_CTRL_DP_AUXCH_SET_SEMA_OWNER_RM
 *         Write the aux channel semaphore for resource manager to own the
 *         the aux channel.
 *       LW0073_CTRL_DP_AUXCH_SET_SEMA_OWNER_VBIOS
 *         Write the aux channel semaphore for vbios/efi to own the
 *         the aux channel.  This value is used only for HW verification
 *         and should not be used in normal driver operation.
 *       LW0073_CTRL_DP_AUXCH_SET_SEMA_OWNER_PMU
 *         Write the aux channel semaphore for pmu to own the
 *         the aux channel.  This value is used only by pmu
 *         and should not be used in normal driver operation.
 *       LW0073_CTRL_DP_AUXCH_SET_SEMA_OWNER_DPU
 *         Write the aux channel semaphore for dpu to own the
 *         the aux channel and should not be used in normal
 *         driver operation.
 *       LW0073_CTRL_DP_AUXCH_SET_SEMA_OWNER_SEC2
 *         Write the aux channel semaphore for sec2 to own the
 *         the aux channel and should not be used in normal
 *         driver operation.
 *       LW0073_CTRL_DP_AUXCH_SET_SEMA_OWNER_RELEASE
 *         Write the aux channel semaphore for hardware to own the
 *         the aux channel.  This value is used only for HW verification
 *         and should not be used in normal driver operation.
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0073_CTRL_CMD_DP_AUXCH_SET_SEMA               (0x731342U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_DP_AUXCH_SET_SEMA_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DP_AUXCH_SET_SEMA_PARAMS_MESSAGE_ID (0x42U)

typedef struct LW0073_CTRL_DP_AUXCH_SET_SEMA_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 owner;
} LW0073_CTRL_DP_AUXCH_SET_SEMA_PARAMS;

#define LW0073_CTRL_DP_AUXCH_SET_SEMA_OWNER                 2:0
#define LW0073_CTRL_DP_AUXCH_SET_SEMA_OWNER_RELEASE (0x00000000U)
#define LW0073_CTRL_DP_AUXCH_SET_SEMA_OWNER_RM      (0x00000001U)
#define LW0073_CTRL_DP_AUXCH_SET_SEMA_OWNER_VBIOS   (0x00000002U)
#define LW0073_CTRL_DP_AUXCH_SET_SEMA_OWNER_PMU     (0x00000003U)
#define LW0073_CTRL_DP_AUXCH_SET_SEMA_OWNER_DPU     (0x00000004U)
#define LW0073_CTRL_DP_AUXCH_SET_SEMA_OWNER_SEC2    (0x00000005U)

/*
 * LW0073_CTRL_CMD_DP_CTRL
 *
 * This command is used to set various displayPort configurations for
 * the specified displayId such a lane count and link bandwidth.  It
 * is assumed that link training has already oclwrred.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the display for which the dfp
 *     caps should be returned.  The display ID must a dfp display.
 *     If more than one displayId bit is set or the displayId is not a dfp,
 *     this call will return LW_ERR_ILWALID_ARGUMENT.
 *   cmd
 *     This parameter is an input to this command.
 *     Here are the current defined fields:
 *       LW0073_CTRL_DP_CMD_SET_LANE_COUNT
 *         Set to specify the number of displayPort lanes to configure.
 *           LW0073_CTRL_DP_CMD_SET_LANE_COUNT_FALSE
 *             No request to set the displayport lane count.
 *           LW0073_CTRL_DP_CMD_SET_LANE_COUNT_TRUE
 *             Set this value to indicate displayport lane count change.
 *       LW0073_CTRL_DP_CMD_SET_LINK_BANDWIDTH
 *         Set to specify a request to change the link bandwidth.
 *           LW0073_CTRL_DP_CMD_SET_LINK_BANDWIDTH_FALSE
 *             No request to set the displayport link bandwidth.
 *           LW0073_CTRL_DP_CMD_SET_LINK_BANDWIDTH_TRUE
 *             Set this value to indicate displayport link bandwidth change.
 *       LW0073_CTRL_DP_CMD_SET_LINK_BANDWIDTH
 *         Set to specify a request to change the link bandwidth.
 *           LW0073_CTRL_DP_CMD_SET_LINK_BANDWIDTH_FALSE
 *             No request to set the displayport link bandwidth.
 *           LW0073_CTRL_DP_CMD_SET_LINK_BANDWIDTH_TRUE
 *             Set this value to indicate displayport link bandwidth change.
 *       LW0073_CTRL_DP_CMD_DISABLE_DOWNSPREAD
 *         Set to disable downspread during link training.
 *           LW0073_CTRL_DP_CMD_DISABLE_DOWNSPREAD_FALSE
 *             Downspread will be enabled.
 *           LW0073_CTRL_DP_CMD_DISABLE_DOWNSPREAD_TRUE
 *             Downspread will be disabled (e.g. for compliance testing).
 *       LW0073_CTRL_DP_CMD_SET_FORMAT_MODE
 *         This field specifies the DP stream mode.
 *           LW0073_CTRL_DP_CMD_SET_FORMAT_MODE_SINGLE_STREAM
 *             This value indicates that single stream mode is specified.
 *           LW0073_CTRL_DP_CMD_SET_FORMAT_MODE_MULTI_STREAM
 *             This value indicates that multi stream mode is specified.
 *       LW0073_CTRL_DP_CMD_FAST_LINK_TRAINING
 *         Set to do Fast link training (avoid AUX transactions for link
 *         training). We need to restore all the previous trained link settings
 *         (e.g. the drive current/preemphasis settings) before doing FLT.
 *         During FLT, we send training pattern 1 followed by training pattern 2
 *         each for a period of 500us.
 *           LW0073_CTRL_DP_CMD_FAST_LINK_TRAINING_NO
 *             Not a fast link training scenario.
 *           LW0073_CTRL_DP_CMD_FAST_LINK_TRAINING_YES
 *             Do fast link training.
 *       LW0073_CTRL_DP_CMD_NO_LINK_TRAINING
 *         Set to do No link training. We need to restore all the previous
 *         trained link settings (e.g. the drive current/preemphasis settings)
 *         before doing NLT, but we don't need to do the Clock Recovery and
 *         Channel Equalization. (Please refer to LWPU PANEL SELFREFRESH
 *         CONTROLLER SPECIFICATION 3.1.6 for detail flow)
 *           LW0073_CTRL_DP_CMD_NO_LINK_TRAINING_NO
 *             Not a no link training scenario.
 *           LW0073_CTRL_DP_CMD_NO_LINK_TRAINING_YES
 *             Do no link training.
 *       LW0073_CTRL_DP_CMD_USE_DOWNSPREAD_SETTING
 *         Specifies whether RM should use the DP Downspread setting specified by
 *         LW0073_CTRL_DP_CMD_DISABLE_DOWNSPREAD command regardless of what the Display
 *         is capable of. This is used along with the Fake link training option so that
 *         we can configure the GPU to enable/disable spread when a real display is
 *         not connected.
 *           LW0073_CTRL_DP_CMD_USE_DOWNSPREAD_SETTING_FORCE
 *              RM Always use the DP Downspread setting specified.
 *           LW0073_CTRL_DP_CMD_USE_DOWNSPREAD_SETTING_DEFAULT
 *              RM will enable Downspread only if the display supports it. (default)
 *       LW0073_CTRL_DP_CMD_SKIP_HW_PROGRAMMING
 *         Specifies whether RM should skip HW training of the link.
 *         If this is the case then RM only updates its SW state without actually
 *         touching any HW registers. Clients should use this ONLY if it has determined -
 *         a. link is trained and not lost
 *         b. desired link config is same as current trained link config
 *         c. link is not in D3 (should be in D0)
 *           LW0073_CTRL_DP_CMD_SKIP_HW_PROGRAMMING_NO
 *              RM doesn't skip HW LT as the current Link Config is not the same as the
 *              requested Link Config.
 *           LW0073_CTRL_DP_CMD_SKIP_HW_PROGRAMMING_YES
 *              RM skips HW LT and only updates its SW state as client has determined that
 *              the current state of the link and the requested Link Config is the same.
 *       LW0073_CTRL_DP_CMD_DISABLE_LINK_CONFIG
 *         Set if the client does not want link training to happen.
 *         This should ONLY be used for HW verification.
 *           LW0073_CTRL_DP_CMD_DISABLE_LINK_CONFIG_FALSE
 *             This is normal production behaviour which shall perform
 *             link training or follow the normal procedure for lane count
 *             reduction.
 *           LW0073_CTRL_DP_CMD_DISABLE_LINK_CONFIG_TRUE
 *             Set this value to not perform link config steps, this should
 *             only be turned on for HW verif testing.  If _LINK_BANDWIDTH
 *             or _LANE_COUNT is set, RM will only write to the TX DP registers
 *             and perform no link training.
 *       LW0073_CTRL_DP_CMD_POST_LT_ADJ_REQ_GRANTED
 *         This field specifies if source grants Post Link training Adjustment request or not.
 *           LW0073_CTRL_DP_CMD_POST_LT_ADJ_REQ_GRANTED_NO
 *              Source does not grant Post Link training Adjustment request
 *           LW0073_CTRL_DP_CMD_POST_LT_ADJ_REQ_GRANTED_YES
 *              Source grants Post Link training Adjustment request
 *              Source wants to link train LT Tunable Repeaters
 *       LW0073_CTRL_DP_CMD_FAKE_LINK_TRAINING
 *         This field specifies if fake link training is to be done. This will
 *         program enough of the hardware to avoid any hardware hangs and
 *         depending upon option chosen by the client, OR will be enabled for
 *         transmisssion.
 *           LW0073_CTRL_DP_CMD_FAKE_LINK_TRAINING_NO
 *              No Fake LT will be performed
 *           LW0073_CTRL_DP_CMD_FAKE_LINK_TRAINING_DONOT_TOGGLE_TRANSMISSION
 *              SOR will be not powered up during Fake LT
 *           LW0073_CTRL_DP_CMD_FAKE_LINK_TRAINING_TOGGLE_TRANSMISSION_ON
 *              SOR will be powered up during Fake LT
 *       LW0073_CTRL_DP_CMD_TRAIN_PHY_REPEATER
 *         This field specifies if source wants to link train LT Tunable Repeaters or not.
 *           LW0073_CTRL_DP_CMD_TRAIN_PHY_REPEATER_NO
 *              Source does not want to link train LT Tunable Repeaters
 *           LW0073_CTRL_DP_CMD_TRAIN_PHY_REPEATER_YES
 *       LW0073_CTRL_DP_CMD_BANDWIDTH_TEST
 *         Set if the client wants to reset the link after the link
 *         training is done. As a part of uncommtting a DP display.
 *           LW0073_CTRL_DP_CMD_BANDWIDTH_TEST_NO
 *             This is for normal operation, if DD decided not to reset the link.
 *           LW0073_CTRL_DP_CMD_BANDWIDTH_TEST_YES
 *             This is to reset the link, if DD decided to uncommit the display because
 *             the link is no more required to be enabled, as in a DP compliance test.
 *       LW0073_CTRL_DP_CMD_LINK_CONFIG_CHECK_DISABLE
 *         Set if the client does not want link training to happen.
 *         This should ONLY be used for HW verification if necessary.
 *           LW0073_CTRL_DP_CMD_LINK_CONFIG_CHECK_DISABLE_FALSE
 *             This is normal production behaviour which shall perform
 *             pre link training checks such as if both rx and tx are capable
 *             of the requested config for lane and link bw.
 *           LW0073_CTRL_DP_CMD_LINK_CONFIG_CHECK_DISABLE_TRUE
 *             Set this value to bypass link config check, this should
 *             only be turned on for HW verif testing.  If _LINK_BANDWIDTH
 *             or _LANE_COUNT is set, RM will not check TX and DX caps.
 *       LW0073_CTRL_DP_CMD_FALLBACK_CONFIG
 *         Set if requested config by client fails and if link if being
 *         trained for the fallback config.
 *           LW0073_CTRL_DP_CMD_FALLBACK_CONFIG_FALSE
 *             This is the normal case when the link is being trained for a requested config.
 *           LW0073_CTRL_DP_CMD_LINK_CONFIG_CHECK_DISABLE_TRUE
 *             Set this value in case the link configuration for requested config fails
 *             and the link is being trained for a fallback config.
 *       LW0073_CTRL_DP_CMD_ENABLE_FEC
 *         Specifies whether RM should set LW_DPCD14_FEC_CONFIGURATION_FEC_READY
 *         before link training if client has determined that FEC is required(for DSC).
 *         If required to be enabled RM sets FEC enable bit in panel, start link training.
 *         Enabling/disabling FEC on GPU side is not done during Link training
 *         and RM Ctrl call LW0073_CTRL_CMD_DP_CONFIGURE_FEC has to be called
 *         explicitly to enable/disable FEC after LT(including PostLT LQA).
 *         If enabled, FEC would be disabled while powering down the link.
 *         Client has to make sure to account for 3% overhead of transmitting
 *         FEC symbols while callwlating DP bandwidth.
 *           LW0073_CTRL_DP_CMD_ENABLE_FEC_FALSE
 *             This is the normal case when FEC is not required
 *           LW0073_CTRL_DP_CMD_ENABLE_FEC_TRUE
 *             Set this value in case FEC needs to be enabled
 *   data
 *     This parameter is an input and output to this command.
 *     Here are the current defined fields:
 *       LW0073_CTRL_DP_DATA_SET_LANE_COUNT
 *         This field specifies the desired setting for lane count.  A client
 *         may choose any lane count as long as it does not exceed the
 *         capability of DisplayPort receiver as indicated in the
 *         receiver capability field.  The valid values for this field are:
 *           LW0073_CTRL_DP_DATA_SET_LANE_COUNT_0
 *             For zero-lane configurations, link training is shut down.
 *           LW0073_CTRL_DP_DATA_SET_LANE_COUNT_1
 *             For one-lane configurations, lane0 is used.
 *           LW0073_CTRL_DP_DATA_SET_LANE_COUNT_2
 *             For two-lane configurations, lane0 and lane1 is used.
 *           LW0073_CTRL_DP_DATA_SET_LANE_COUNT_4
 *             For four-lane configurations, all lanes are used.
 *           LW0073_CTRL_DP_DATA_SET_LANE_COUNT_8
 *             For devices that supports 8-lane DP.
 *         On return, the lane count setting is returned which may be
 *         different from the requested input setting.
 *       LW0073_CTRL_DP_DATA_SET_LINK_BW
 *         This field specifies the desired setting for link bandwidth.  There
 *         are only four supported main link bandwidth settings.  The
 *         valid values for this field are:
 *           LW0073_CTRL_DP_DATA_SET_LINK_BW_1_62GBPS
 *           LW0073_CTRL_DP_DATA_SET_LINK_BW_2_70GBPS
 *           LW0073_CTRL_DP_DATA_SET_LINK_BW_5_40GBPS
 *           LW0073_CTRL_DP_DATA_SET_LINK_BW_8_10GBPS
 *         On return, the link bandwidth setting is returned which may be
 *         different from the requested input setting.
 *       LW0073_CTRL_DP_DATA_TARGET
 *         This field specifies which physical repeater or sink to be trained.
 *         Client should make sure
 *             1. Physical repeater should be targeted in order, start from the one closest to GPU.
 *             2. All physical repeater is properly trained before targets sink.
 *         The valid values for this field are:
 *           LW0073_CTRL_DP_DATA_TARGET_PHY_REPEATER_X
 *               'X' denotes physical repeater index. It's a 1-based index to
 *                   reserve 0 for _SINK.
 *               'X' can't be more than 8.
 *           LW0073_CTRL_DP_DATA_TARGET_SINK
 *   err
 *     This parameter specifies provides info regarding the outcome
 *     of this calling control call.  If zero, no errors were found.
 *     Otherwise, this parameter will specify the error detected.
 *     The valid parameter is broken down as follows:
 *        LW0073_CTRL_DP_ERR_SET_LANE_COUNT
 *          If set to _ERR, set lane count failed.
 *        LW0073_CTRL_DP_ERR_SET_LINK_BANDWIDTH
 *          If set to _ERR, set link bandwidth failed.
 *        LW0073_CTRL_DP_ERR_DISABLE_DOWNSPREAD
 *          If set to _ERR, disable downspread failed.
 *        LW0073_CTRL_DP_ERR_ILWALID_PARAMETER
 *          If set to _ERR, at least one of the calling functions
 *          failed due to an invalid parameter.
 *        LW0073_CTRL_DP_ERR_SET_LINK_TRAINING
 *          If set to _ERR, link training failed.
 *        LW0073_CTRL_DP_ERR_TRAIN_PHY_REPEATER
 *          If set to _ERR, the operation to Link Train repeater is failed.
 *        LW0073_CTRL_DP_ERR_ENABLE_FEC
 *          If set to _ERR, the operation to enable FEC is failed.
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

#define LW0073_CTRL_CMD_DP_CTRL                     (0x731343U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_DP_CTRL_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DP_CTRL_PARAMS_MESSAGE_ID (0x43U)

typedef struct LW0073_CTRL_DP_CTRL_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 cmd;
    LwU32 data;
    LwU32 err;
    LwU32 retryTimeMs;
    LwU32 eightLaneDpcdBaseAddr;
} LW0073_CTRL_DP_CTRL_PARAMS;

#define LW0073_CTRL_DP_CMD_SET_LANE_COUNT                           0:0
#define LW0073_CTRL_DP_CMD_SET_LANE_COUNT_FALSE                         (0x00000000U)
#define LW0073_CTRL_DP_CMD_SET_LANE_COUNT_TRUE                          (0x00000001U)
#define LW0073_CTRL_DP_CMD_SET_LINK_BW                              1:1
#define LW0073_CTRL_DP_CMD_SET_LINK_BW_FALSE                            (0x00000000U)
#define LW0073_CTRL_DP_CMD_SET_LINK_BW_TRUE                             (0x00000001U)
#define LW0073_CTRL_DP_CMD_DISABLE_DOWNSPREAD                       2:2
#define LW0073_CTRL_DP_CMD_DISABLE_DOWNSPREAD_FALSE                     (0x00000000U)
#define LW0073_CTRL_DP_CMD_DISABLE_DOWNSPREAD_TRUE                      (0x00000001U)
#define LW0073_CTRL_DP_CMD_UNUSED                                   3:3
#define LW0073_CTRL_DP_CMD_SET_FORMAT_MODE                          4:4
#define LW0073_CTRL_DP_CMD_SET_FORMAT_MODE_SINGLE_STREAM                (0x00000000U)
#define LW0073_CTRL_DP_CMD_SET_FORMAT_MODE_MULTI_STREAM                 (0x00000001U)
#define LW0073_CTRL_DP_CMD_FAST_LINK_TRAINING                       5:5
#define LW0073_CTRL_DP_CMD_FAST_LINK_TRAINING_NO                        (0x00000000U)
#define LW0073_CTRL_DP_CMD_FAST_LINK_TRAINING_YES                       (0x00000001U)
#define LW0073_CTRL_DP_CMD_NO_LINK_TRAINING                         6:6
#define LW0073_CTRL_DP_CMD_NO_LINK_TRAINING_NO                          (0x00000000U)
#define LW0073_CTRL_DP_CMD_NO_LINK_TRAINING_YES                         (0x00000001U)
#define LW0073_CTRL_DP_CMD_SET_ENHANCED_FRAMING                     7:7
#define LW0073_CTRL_DP_CMD_SET_ENHANCED_FRAMING_FALSE                   (0x00000000U)
#define LW0073_CTRL_DP_CMD_SET_ENHANCED_FRAMING_TRUE                    (0x00000001U)
#define LW0073_CTRL_DP_CMD_USE_DOWNSPREAD_SETTING                   8:8
#define LW0073_CTRL_DP_CMD_USE_DOWNSPREAD_SETTING_DEFAULT               (0x00000000U)
#define LW0073_CTRL_DP_CMD_USE_DOWNSPREAD_SETTING_FORCE                 (0x00000001U)
#define LW0073_CTRL_DP_CMD_SKIP_HW_PROGRAMMING                      9:9
#define LW0073_CTRL_DP_CMD_SKIP_HW_PROGRAMMING_NO                       (0x00000000U)
#define LW0073_CTRL_DP_CMD_SKIP_HW_PROGRAMMING_YES                      (0x00000001U)
#define LW0073_CTRL_DP_CMD_POST_LT_ADJ_REQ_GRANTED                10:10
#define LW0073_CTRL_DP_CMD_POST_LT_ADJ_REQ_GRANTED_NO                   (0x00000000U)
#define LW0073_CTRL_DP_CMD_POST_LT_ADJ_REQ_GRANTED_YES                  (0x00000001U)
#define LW0073_CTRL_DP_CMD_FAKE_LINK_TRAINING                     12:11
#define LW0073_CTRL_DP_CMD_FAKE_LINK_TRAINING_NO                        (0x00000000U)
#define LW0073_CTRL_DP_CMD_FAKE_LINK_TRAINING_DONOT_TOGGLE_TRANSMISSION (0x00000001U)
#define LW0073_CTRL_DP_CMD_FAKE_LINK_TRAINING_TOGGLE_TRANSMISSION_ON    (0x00000002U)
#define LW0073_CTRL_DP_CMD_TRAIN_PHY_REPEATER                     13:13
#define LW0073_CTRL_DP_CMD_TRAIN_PHY_REPEATER_NO                        (0x00000000U)
#define LW0073_CTRL_DP_CMD_TRAIN_PHY_REPEATER_YES                       (0x00000001U)
#define LW0073_CTRL_DP_CMD_FALLBACK_CONFIG                        14:14
#define LW0073_CTRL_DP_CMD_FALLBACK_CONFIG_FALSE                        (0x00000000U)
#define LW0073_CTRL_DP_CMD_FALLBACK_CONFIG_TRUE                         (0x00000001U)
#define LW0073_CTRL_DP_CMD_ENABLE_FEC                             15:15
#define LW0073_CTRL_DP_CMD_ENABLE_FEC_FALSE                             (0x00000000U)
#define LW0073_CTRL_DP_CMD_ENABLE_FEC_TRUE                              (0x00000001U)

#define LW0073_CTRL_DP_CMD_BANDWIDTH_TEST                         29:29
#define LW0073_CTRL_DP_CMD_BANDWIDTH_TEST_NO                            (0x00000000U)
#define LW0073_CTRL_DP_CMD_BANDWIDTH_TEST_YES                           (0x00000001U)
#define LW0073_CTRL_DP_CMD_LINK_CONFIG_CHECK_DISABLE              30:30
#define LW0073_CTRL_DP_CMD_LINK_CONFIG_CHECK_DISABLE_FALSE              (0x00000000U)
#define LW0073_CTRL_DP_CMD_LINK_CONFIG_CHECK_DISABLE_TRUE               (0x00000001U)
#define LW0073_CTRL_DP_CMD_DISABLE_LINK_CONFIG                    31:31
#define LW0073_CTRL_DP_CMD_DISABLE_LINK_CONFIG_FALSE                    (0x00000000U)
#define LW0073_CTRL_DP_CMD_DISABLE_LINK_CONFIG_TRUE                     (0x00000001U)

#define LW0073_CTRL_DP_DATA_SET_LANE_COUNT                          4:0
#define LW0073_CTRL_DP_DATA_SET_LANE_COUNT_0                            (0x00000000U)
#define LW0073_CTRL_DP_DATA_SET_LANE_COUNT_1                            (0x00000001U)
#define LW0073_CTRL_DP_DATA_SET_LANE_COUNT_2                            (0x00000002U)
#define LW0073_CTRL_DP_DATA_SET_LANE_COUNT_4                            (0x00000004U)
#define LW0073_CTRL_DP_DATA_SET_LANE_COUNT_8                            (0x00000008U)
#define LW0073_CTRL_DP_DATA_SET_LINK_BW                            15:8
#define LW0073_CTRL_DP_DATA_SET_LINK_BW_1_62GBPS                        (0x00000006U)
#define LW0073_CTRL_DP_DATA_SET_LINK_BW_2_16GBPS                        (0x00000008U)
#define LW0073_CTRL_DP_DATA_SET_LINK_BW_2_43GBPS                        (0x00000009U)
#define LW0073_CTRL_DP_DATA_SET_LINK_BW_2_70GBPS                        (0x0000000AU)
#define LW0073_CTRL_DP_DATA_SET_LINK_BW_3_24GBPS                        (0x0000000LW)
#define LW0073_CTRL_DP_DATA_SET_LINK_BW_4_32GBPS                        (0x00000010U)
#define LW0073_CTRL_DP_DATA_SET_LINK_BW_5_40GBPS                        (0x00000014U)
#define LW0073_CTRL_DP_DATA_SET_LINK_BW_8_10GBPS                        (0x0000001EU)
#define LW0073_CTRL_DP_DATA_SET_ENHANCED_FRAMING                  18:18
#define LW0073_CTRL_DP_DATA_SET_ENHANCED_FRAMING_NO                     (0x00000000U)
#define LW0073_CTRL_DP_DATA_SET_ENHANCED_FRAMING_YES                    (0x00000001U)
#define LW0073_CTRL_DP_DATA_TARGET                                22:19
#define LW0073_CTRL_DP_DATA_TARGET_SINK                                 (0x00000000U)
#define LW0073_CTRL_DP_DATA_TARGET_PHY_REPEATER_0                       (0x00000001U)
#define LW0073_CTRL_DP_DATA_TARGET_PHY_REPEATER_1                       (0x00000002U)
#define LW0073_CTRL_DP_DATA_TARGET_PHY_REPEATER_2                       (0x00000003U)
#define LW0073_CTRL_DP_DATA_TARGET_PHY_REPEATER_3                       (0x00000004U)
#define LW0073_CTRL_DP_DATA_TARGET_PHY_REPEATER_4                       (0x00000005U)
#define LW0073_CTRL_DP_DATA_TARGET_PHY_REPEATER_5                       (0x00000006U)
#define LW0073_CTRL_DP_DATA_TARGET_PHY_REPEATER_6                       (0x00000007U)
#define LW0073_CTRL_DP_DATA_TARGET_PHY_REPEATER_7                       (0x00000008U)

#define LW0073_CTRL_DP_ERR_SET_LANE_COUNT                           0:0
#define LW0073_CTRL_DP_ERR_SET_LANE_COUNT_NOERR                         (0x00000000U)
#define LW0073_CTRL_DP_ERR_SET_LANE_COUNT_ERR                           (0x00000001U)
#define LW0073_CTRL_DP_ERR_SET_LINK_BW                              1:1
#define LW0073_CTRL_DP_ERR_SET_LINK_BW_NOERR                            (0x00000000U)
#define LW0073_CTRL_DP_ERR_SET_LINK_BW_ERR                              (0x00000001U)
#define LW0073_CTRL_DP_ERR_DISABLE_DOWNSPREAD                       2:2
#define LW0073_CTRL_DP_ERR_DISABLE_DOWNSPREAD_NOERR                     (0x00000000U)
#define LW0073_CTRL_DP_ERR_DISABLE_DOWNSPREAD_ERR                       (0x00000001U)
#define LW0073_CTRL_DP_ERR_UNUSED                                   3:3
#define LW0073_CTRL_DP_ERR_CLOCK_RECOVERY                           4:4
#define LW0073_CTRL_DP_ERR_CLOCK_RECOVERY_NOERR                         (0x00000000U)
#define LW0073_CTRL_DP_ERR_CLOCK_RECOVERY_ERR                           (0x00000001U)
#define LW0073_CTRL_DP_ERR_CHANNEL_EQUALIZATION                     5:5
#define LW0073_CTRL_DP_ERR_CHANNEL_EQUALIZATION_NOERR                   (0x00000000U)
#define LW0073_CTRL_DP_ERR_CHANNEL_EQUALIZATION_ERR                     (0x00000001U)
#define LW0073_CTRL_DP_ERR_TRAIN_PHY_REPEATER                       6:6
#define LW0073_CTRL_DP_ERR_TRAIN_PHY_REPEATER_NOERR                     (0x00000000U)
#define LW0073_CTRL_DP_ERR_TRAIN_PHY_REPEATER_ERR                       (0x00000001U)
#define LW0073_CTRL_DP_ERR_ENABLE_FEC                               7:7
#define LW0073_CTRL_DP_ERR_ENABLE_FEC_NOERR                             (0x00000000U)
#define LW0073_CTRL_DP_ERR_ENABLE_FEC_ERR                               (0x00000001U)
#define LW0073_CTRL_DP_ERR_CR_DONE_LANE                            11:8
#define LW0073_CTRL_DP_ERR_CR_DONE_LANE_0_LANE                          (0x00000000U)
#define LW0073_CTRL_DP_ERR_CR_DONE_LANE_1_LANE                          (0x00000001U)
#define LW0073_CTRL_DP_ERR_CR_DONE_LANE_2_LANE                          (0x00000002U)
#define LW0073_CTRL_DP_ERR_CR_DONE_LANE_4_LANE                          (0x00000004U)
#define LW0073_CTRL_DP_ERR_CR_DONE_LANE_8_LANE                          (0x00000008U)
#define LW0073_CTRL_DP_ERR_EQ_DONE_LANE                           15:12
#define LW0073_CTRL_DP_ERR_EQ_DONE_LANE_0_LANE                          (0x00000000U)
#define LW0073_CTRL_DP_ERR_EQ_DONE_LANE_1_LANE                          (0x00000001U)
#define LW0073_CTRL_DP_ERR_EQ_DONE_LANE_2_LANE                          (0x00000002U)
#define LW0073_CTRL_DP_ERR_EQ_DONE_LANE_4_LANE                          (0x00000004U)
#define LW0073_CTRL_DP_ERR_EQ_DONE_LANE_8_LANE                          (0x00000008U)
#define LW0073_CTRL_DP_ERR_ILWALID_PARAMETER                      30:30
#define LW0073_CTRL_DP_ERR_ILWALID_PARAMETER_NOERR                      (0x00000000U)
#define LW0073_CTRL_DP_ERR_ILWALID_PARAMETER_ERR                        (0x00000001U)
#define LW0073_CTRL_DP_ERR_LINK_TRAINING                          31:31
#define LW0073_CTRL_DP_ERR_LINK_TRAINING_NOERR                          (0x00000000U)
#define LW0073_CTRL_DP_ERR_LINK_TRAINING_ERR                            (0x00000001U)

/*
 * LW0073_CTRL_DP_LANE_DATA_PARAMS
 *
 * This structure provides lane characteristics.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the display for which the dfp
 *     caps should be returned.  The display ID must a dfp display.
 *     If more than one displayId bit is set or the displayId is not a dfp,
 *     this call will return LW_ERR_ILWALID_ARGUMENT.
 *   numLanes
 *      Indicates number of lanes for which the data is valid
 *   data
 *     This parameter is an input to this command.
 *     Here are the current defined fields:
 *       LW0073_CTRL_DP_LANE_DATA_PREEMPHASIS
 *         This field specifies the preemphasis level set in the lane.
 *         The valid values for this field are:
 *           LW0073_CTRL_DP_LANE_DATA_PREEMPHASIS_NONE
 *             No-preemphais for this lane.
 *           LW0073_CTRL_DP_LANE_DATA_PREEMPHASIS_LEVEL1
 *             Preemphasis set to 3.5 dB.
 *           LW0073_CTRL_DP_LANE_DATA_PREEMPHASIS_LEVEL2
 *             Preemphasis set to 6.0 dB.
 *           LW0073_CTRL_DP_LANE_DATA_PREEMPHASIS_LEVEL3
 *             Preemphasis set to 9.5 dB.
 *       LW0073_CTRL_DP_LANE_DATA_DRIVELWRRENT
 *         This field specifies the drive current set in the lane.
 *         The valid values for this field are:
 *           LW0073_CTRL_DP_LANE_DATA_DRIVELWRRENT_LEVEL0
 *             Drive current level is set to 8 mA
 *           LW0073_CTRL_DP_LANE_DATA_DRIVELWRRENT_LEVEL1
 *             Drive current level is set to 12 mA
 *           LW0073_CTRL_DP_LANE_DATA_DRIVELWRRENT_LEVEL2
 *             Drive current level is set to 16 mA
 *           LW0073_CTRL_DP_LANE_DATA_DRIVELWRRENT_LEVEL3
 *             Drive current level is set to 24 mA
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_MAX_LANES                                           8U

typedef struct LW0073_CTRL_DP_LANE_DATA_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 numLanes;
    LwU32 data[LW0073_CTRL_MAX_LANES];
} LW0073_CTRL_DP_LANE_DATA_PARAMS;

#define LW0073_CTRL_DP_LANE_DATA_PREEMPHASIS                   1:0
#define LW0073_CTRL_DP_LANE_DATA_PREEMPHASIS_NONE    (0x00000000U)
#define LW0073_CTRL_DP_LANE_DATA_PREEMPHASIS_LEVEL1  (0x00000001U)
#define LW0073_CTRL_DP_LANE_DATA_PREEMPHASIS_LEVEL2  (0x00000002U)
#define LW0073_CTRL_DP_LANE_DATA_PREEMPHASIS_LEVEL3  (0x00000003U)
#define LW0073_CTRL_DP_LANE_DATA_DRIVELWRRENT                  3:2
#define LW0073_CTRL_DP_LANE_DATA_DRIVELWRRENT_LEVEL0 (0x00000000U)
#define LW0073_CTRL_DP_LANE_DATA_DRIVELWRRENT_LEVEL1 (0x00000001U)
#define LW0073_CTRL_DP_LANE_DATA_DRIVELWRRENT_LEVEL2 (0x00000002U)
#define LW0073_CTRL_DP_LANE_DATA_DRIVELWRRENT_LEVEL3 (0x00000003U)

/*
 * LW0073_CTRL_CMD_GET_DP_LANE_DATA
 *
 * This command is used to get the current pre-emphasis and drive current
 * level values for the specified number of lanes.
 *
 * The command takes a LW0073_CTRL_DP_LANE_DATA_PARAMS structure as the
 * argument with the appropriate subDeviceInstance and displayId filled.
 * The arguments of this structure and the format of  preemphasis and drive-
 * current levels are described above.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *
 * NOTE: This control call is only for testing purposes and
 *       should not be used in normal DP operations. Preemphais
 *       and drive current level will be set during Link training
 *       in normal DP operations
 *
 */

#define LW0073_CTRL_CMD_DP_GET_LANE_DATA             (0x731345U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | 0x45" */


/*
 * LW0073_CTRL_CMD_SET_DP_LANE_DATA
 *
 * This command is used to set the pre-emphasis and drive current
 * level values for the specified number of lanes.
 *
 * The command takes a LW0073_CTRL_DP_LANE_DATA_PARAMS structure as the
 * argument with the appropriate subDeviceInstance, displayId, number of
 * lanes, preemphasis and drive current values filled in.
 * The arguments of this structure and the format of  preemphasis and drive-
 * current levels are described above.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *
 * NOTE: This control call is only for testing purposes and
 *       should not be used in normal DP operations. Preemphais
 *       and drivelwrrent will be set during Link training in
 *       normal DP operations
 *
 */

#define LW0073_CTRL_CMD_DP_SET_LANE_DATA             (0x731346U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | 0x46" */

/*
 * LW0073_CTRL_DP_CSTM
 *
 * This structure specifies the 80 bit DP CSTM Test Pattern data
 * The fields of this structure are to be specified as follows:
 *      lower   takes bits 31:0
 *      middle  takes bits 63:32
 *      upper   takes bits 79:64
 *
 */
typedef struct LW0073_CTRL_DP_CSTM {
    LwU32 lower;
    LwU32 middle;
    LwU32 upper;
} LW0073_CTRL_DP_CSTM;

/*
 * LW0073_CTRL_DP_TESTPATTERN
 *
 * This structure specifies the possible test patterns available in
 * display port. The field testPattern can be one of the following
 * values.
 *          LW0073_CTRL_DP_SET_TESTPATTERN_DATA_NONE
 *              No test pattern on the main link
 *          LW0073_CTRL_DP_SET_TESTPATTERN_DATA_D10_2
 *              D10.2 pattern on the main link
 *          LW0073_CTRL_DP_SET_TESTPATTERN_DATA_SERMP
 *              SERMP pattern on main link
 *          LW0073_CTRL_DP_SET_TESTPATTERN_DATA_PRBS_7
 *              PRBS7 pattern on the main link
 *
 */

typedef struct LW0073_CTRL_DP_TESTPATTERN {
    LwU32 testPattern;
} LW0073_CTRL_DP_TESTPATTERN;

#define LW0073_CTRL_DP_TESTPATTERN_DATA                              2:0
#define LW0073_CTRL_DP_TESTPATTERN_DATA_NONE           (0x00000000U)
#define LW0073_CTRL_DP_TESTPATTERN_DATA_D10_2          (0x00000001U)
#define LW0073_CTRL_DP_TESTPATTERN_DATA_SERMP          (0x00000002U)
#define LW0073_CTRL_DP_TESTPATTERN_DATA_PRBS_7         (0x00000003U)
#define LW0073_CTRL_DP_TESTPATTERN_DATA_CSTM           (0x00000004U)
#define LW0073_CTRL_DP_TESTPATTERN_DATA_HBR2COMPLIANCE (0x00000005U)
#define LW0073_CTRL_DP_TESTPATTERN_DATA_CP2520PAT3     (0x00000006U)

/*
 * LW0073_CTRL_CMD_DP_SET_TESTPATTERN
 *
 * This command forces the main link to output the selected test patterns
 * supported in DP specs.
 *
 * The command takes a LW0073_CTRL_DP_SET_TESTPATTERN_PARAMS structure as the
 * argument with the appropriate subDeviceInstance, displayId and test pattern
 * to be set as inputs.
 * The arguments of this structure and the format of  test patterns are
 * described above.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the display for which the dfp
 *     caps should be returned.  The display ID must a dfp display.
 *     If more than one displayId bit is set or the displayId is not a dfp,
 *     this call will return LW_ERR_ILWALID_ARGUMENT.
 *   testPattern
 *     This parameter is of type LW0073_CTRL_DP_TESTPATTERN and specifies
 *     the testpattern to set on displayport. The format of this structure
 *     is described above.
 *   laneMask
 *     This parameter specifies the bit mask of DP lanes on which test
 *     pattern is to be applied.
 *   lower
 *     This parameter specifies the lower 64 bits of the CSTM test pattern
 *   upper
 *     This parameter specifies the upper 16 bits of the CSTM test pattern
 *   bIsHBR2
 *     This Boolean parameter is set to TRUE if HBR2 compliance test is
 *     being performed.
  *   bSkipLaneDataOverride
 *      skip override of pre-emp and drive current
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *
 * NOTE: This control call is only for testing purposes and
 *       should not be used in normal DP operations. Preemphais
 *       and drivelwrrent will be set during Link training in
 *       normal DP operations
 *
 */

#define LW0073_CTRL_DP_SET_TESTPATTERN_PARAMS_MESSAGE_ID (0x47U)

typedef struct LW0073_CTRL_DP_SET_TESTPATTERN_PARAMS {
    LwU32                      subDeviceInstance;
    LwU32                      displayId;
    LW0073_CTRL_DP_TESTPATTERN testPattern;
    LwU8                       laneMask;
    LW0073_CTRL_DP_CSTM        cstm;
    LwBool                     bIsHBR2;
    LwBool                     bSkipLaneDataOverride;
} LW0073_CTRL_DP_SET_TESTPATTERN_PARAMS;

#define LW0073_CTRL_CMD_DP_SET_TESTPATTERN (0x731347U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_DP_SET_TESTPATTERN_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DP_SET_TESTPATTERN_PARAMS_CSTM0    31:0
#define LW0073_CTRL_DP_SET_TESTPATTERN_PARAMS_CSTM1    63:32
#define LW0073_CTRL_DP_SET_TESTPATTERN_PARAMS_CSTM2    15:0

/*
 * LW0073_CTRL_CMD_GET_DP_TESTPATTERN
 *
 * This command returns the current test pattern set on the main link of
 * Display Port.
 *
 * The command takes a LW0073_CTRL_DP_GET_TESTPATTERN_PARAMS structure as the
 * argument with the appropriate subDeviceInstance, displayId as inputs and
 * returns the current test pattern in testPattern field of the structure.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the display for which the dfp
 *     caps should be returned.  The display ID must a dfp display.
 *     If more than one displayId bit is set or the displayId is not a dfp,
 *     this call will return LW_ERR_ILWALID_ARGUMENT.
 *   testPattern
 *     This parameter is of type LW0073_CTRL_DP_TESTPATTERN and specifies the
 *     testpattern set on displayport. The format of this structure is
 *     described above.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *
 * NOTE: This control call is only for testing purposes and
 *       should not be used in normal DP operations.
 *
 */

#define LW0073_CTRL_DP_GET_TESTPATTERN_PARAMS_MESSAGE_ID (0x48U)

typedef struct LW0073_CTRL_DP_GET_TESTPATTERN_PARAMS {
    LwU32                      subDeviceInstance;
    LwU32                      displayId;
    LW0073_CTRL_DP_TESTPATTERN testPattern;
} LW0073_CTRL_DP_GET_TESTPATTERN_PARAMS;


#define LW0073_CTRL_CMD_DP_GET_TESTPATTERN  (0x731348U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_DP_GET_TESTPATTERN_PARAMS_MESSAGE_ID" */

/*
 * LW0073_CTRL_DP_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_DATA
 *
 * This structure specifies the Pre-emphasis/Drive Current/Postlwrsor2/TxPu information
 * for a display port device. These are the the current values that RM is
 * using to map the levels for Pre-emphasis and Drive Current for Link
 * Training.
 *   preEmphasis
 *     This field specifies the preemphasis values.
 *   driveLwrrent
 *     This field specifies the driveLwrrent values.
 *   postlwrsor2
 *     This field specifies the postlwrsor2 values.
 *   TxPu
 *     This field specifies the pull-up current source drive values.
 */
#define LW0073_CTRL_MAX_DRIVELWRRENT_LEVELS 4U
#define LW0073_CTRL_MAX_PREEMPHASIS_LEVELS  4U
#define LW0073_CTRL_MAX_POSTLWRSOR2_LEVELS  4U

typedef struct LW0073_CTRL_DP_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_1 {
    LwU32 preEmphasis;
    LwU32 driveLwrrent;
    LwU32 postLwrsor2;
    LwU32 TxPu;
} LW0073_CTRL_DP_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_1;

typedef LW0073_CTRL_DP_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_1 LW0073_CTRL_DP_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_SLICE1[LW0073_CTRL_MAX_PREEMPHASIS_LEVELS];

typedef LW0073_CTRL_DP_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_SLICE1 LW0073_CTRL_DP_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_SLICE2[LW0073_CTRL_MAX_DRIVELWRRENT_LEVELS];

typedef LW0073_CTRL_DP_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_SLICE2 LW0073_CTRL_DP_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_DATA[LW0073_CTRL_MAX_POSTLWRSOR2_LEVELS];


/*
 * LW0073_CTRL_DP_SET_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_DATA
 *
 * This command is used to override the Pre-emphasis/Drive Current/PostLwrsor2/TxPu
 * data in the RM.
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the digital display for which the
 *     data should be returned.  The display ID must a digital display.
 *     If more than one displayId bit is set or the displayId is not a DP,
 *     this call will return LW_ERR_ILWALID_ARGUMENT.
 *   dpData
 *     This parameter is of type LW0073_CTRL_DP_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_DATA
 *     and specifies the Pre-emphasis/Drive Current/Postlwrsor2/TxPu information
 *     for a display port device.
 * The command takes a LW0073_CTRL_DP_SET_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_DATA_PARAMS
 * structure as the argument with the appropriate subDeviceInstance, displayId,
 * and dpData.  The fields of this structure are described above.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *
 */
#define LW0073_CTRL_DP_SET_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_DATA_PARAMS_MESSAGE_ID (0x51U)

typedef struct LW0073_CTRL_DP_SET_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_DATA_PARAMS {
    LwU32                                                    subDeviceInstance;
    LwU32                                                    displayId;
    LW0073_CTRL_DP_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_DATA dpData;
} LW0073_CTRL_DP_SET_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_DATA_PARAMS;

#define LW0073_CTRL_CMD_DP_SET_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_DATA (0x731351U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_DP_SET_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_DATA_PARAMS_MESSAGE_ID" */

/*
 * LW0073_CTRL_DP_GET_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_DATA
 *
 * This command is used to get the Pre-emphasis/Drive Current/PostLwrsor2/TxPu data.
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the digital display for which the
 *     data should be returned.  The display ID must a digital display.
 *     If more than one displayId bit is set or the displayId is not a DP,
 *     this call will return LW_ERR_ILWALID_ARGUMENT.
 * The command takes a LW0073_CTRL_DP_GET_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_DATA_PARAMS
 * structure as the argument with the appropriate subDeviceInstance, displayId,
 * and dpData.  The fields of this structure are described above.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *
 */
#define LW0073_CTRL_DP_GET_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_DATA_PARAMS_MESSAGE_ID (0x52U)

typedef struct LW0073_CTRL_DP_GET_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_DATA_PARAMS {
    LwU32                                                    subDeviceInstance;
    LwU32                                                    displayId;
    LW0073_CTRL_DP_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_DATA dpData;
} LW0073_CTRL_DP_GET_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_DATA_PARAMS;

#define LW0073_CTRL_CMD_DP_GET_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_DATA (0x731352U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_DP_GET_PREEMPHASIS_DRIVELWRRENT_POSTLWRSOR2_DATA_PARAMS_MESSAGE_ID" */

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW0073_CTRL_CMD_DP_EXELWTE_PRE_LINK_TRAINING
 *
 * This command is used to run pre-LinkTraining Scripts.
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the digital display for which the
 *     data should be returned.  The display ID must a digital display.
 *     If more than one displayId bit is set or the displayId is not a DP,
 *     this call will return LW_ERR_ILWALID_ARGUMENT.
 *   cmd
 *     This parameter is an input to this command.
 *     Here are the current defined fields:
 *       LW0073_CTRL_DP_EXELWTE_PRE_LINK_TRAINING_CMD_IED
 *         Set to specify if pre-LinkTraining IED script should be run.
 *           LW0073_CTRL_DP_EXELWTE_PRE_LINK_TRAINING_CMD_IED_FALSE
 *             No request to run the pre-LinkTraining IED script.
 *           LW0073_CTRL_DP_EXELWTE_PRE_LINK_TRAINING_CMD_IED_TRUE
 *             Set this value to indicate request to run the
 *             pre-LinkTraining IED script.
 *       LW0073_CTRL_DP_EXELWTE_BEFORE_LINK_SPEED_CMD_IED
 *         Set to specify if BeforeLinkSpeed IED script should be run.
 *           LW0073_CTRL_DP_EXELWTE_BEFORE_LINK_SPEED_CMD_IED_FALSE
 *             No request to run the BeforeLinkSpeed IED script.
 *           LW0073_CTRL_DP_EXELWTE_BEFORE_LINK_SPEED_CMD_IED_TRUE
 *             Set this value to indicate request to run the
 *             BeforeLinkSpeed IED script.
 *   lwrLinkBw
 *     This parameter is used to pass in the link bandwidth required to run the
 *     BeforeLinkSpeedIED script. Refer enum DP_LINK_BANDWIDTH for valid values.
 *     If invalid value is passed, RM will default to use the current value from
 *     LW_PDISP_FE_CMGR_CLK_SOR register.
 *
 * The command takes a LW0073_CTRL_DP_EXELWTE_PRE_LINK_TRAINING_PARAMS
 * structure as the argument with the appropriate subDeviceInstance, displayId,
 * cmd and lwrLinkBw. The fields of this structure are described above.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *
 */
#define LW0073_CTRL_CMD_DP_EXELWTE_PRE_LINK_TRAINING                     (0x731354U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_DP_EXELWTE_PRE_LINK_TRAINING_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DP_EXELWTE_PRE_LINK_TRAINING_PARAMS_MESSAGE_ID (0x54U)

typedef struct LW0073_CTRL_DP_EXELWTE_PRE_LINK_TRAINING_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 cmd;
    LwU32 lwrLinkBw;
} LW0073_CTRL_DP_EXELWTE_PRE_LINK_TRAINING_PARAMS;

#define LW0073_CTRL_DP_EXELWTE_PRE_LINK_TRAINING_CMD_IED                 0:0
#define LW0073_CTRL_DP_EXELWTE_PRE_LINK_TRAINING_CMD_IED_FALSE (0x00000000U)
#define LW0073_CTRL_DP_EXELWTE_PRE_LINK_TRAINING_CMD_IED_TRUE  (0x00000001U)
#define LW0073_CTRL_DP_EXELWTE_BEFORE_LINK_SPEED_CMD_IED                 1:1
#define LW0073_CTRL_DP_EXELWTE_BEFORE_LINK_SPEED_CMD_IED_FALSE (0x00000000U)
#define LW0073_CTRL_DP_EXELWTE_BEFORE_LINK_SPEED_CMD_IED_TRUE  (0x00000001U)

/*
 * LW0073_CTRL_CMD_DP_EXELWTE_POST_LINK_TRAINING
 *
 * This command is used to run post-LinkTraining Scripts.
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the digital display for which the
 *     data should be returned.  The display ID must a digital display.
 *     If more than one displayId bit is set or the displayId is not a DP,
 *     this call will return LW_ERR_ILWALID_ARGUMENT.
 *   cmd
 *     This parameter is an input to this command.
 *     Here are the current defined fields:
 *       LW0073_CTRL_DP_EXELWTE_POST_LINK_TRAINING_CMD_IED
 *         Set to specify if post-LinkTraining IED script should be run.
 *           LW0073_CTRL_DP_EXELWTE_POST_LINK_TRAINING_CMD_IED_FALSE
 *             No request to run the post-LinkTraining IED script.
 *           LW0073_CTRL_DP_EXELWTE_POST_LINK_TRAINING_CMD_IED_TRUE
 *             Set this value to indicate request to run the
 *             post-LinkTraining IED script.
 *
 * The command takes a LW0073_CTRL_DP_EXELWTE_POST_LINK_TRAINING_PARAMS
 * structure as the argument with the appropriate subDeviceInstance, displayId,
 * and cmd.  The fields of this structure are described above.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *
 */
#define LW0073_CTRL_CMD_DP_EXELWTE_POST_LINK_TRAINING          (0x731355U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_DP_EXELWTE_POST_LINK_TRAINING_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DP_EXELWTE_POST_LINK_TRAINING_PARAMS_MESSAGE_ID (0x55U)

typedef struct LW0073_CTRL_DP_EXELWTE_POST_LINK_TRAINING_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 cmd;
} LW0073_CTRL_DP_EXELWTE_POST_LINK_TRAINING_PARAMS;

#define LW0073_CTRL_DP_EXELWTE_POST_LINK_TRAINING_CMD_IED                 0:0
#define LW0073_CTRL_DP_EXELWTE_POST_LINK_TRAINING_CMD_IED_FALSE (0x00000000U)
#define LW0073_CTRL_DP_EXELWTE_POST_LINK_TRAINING_CMD_IED_TRUE  (0x00000001U)

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW0073_CTRL_CMD_DP_MAIN_LINK_CTRL
 *
 * This command is used to set various Main Link configurations for
 * the specified displayId such as powering up/down Main Link.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the DP display which owns
 *     the Main Link to be adjusted.  The display ID must a DP display
 *     as determined with the LW0073_CTRL_CMD_SPECIFIC_GET_TYPE command.
 *     If more than one displayId bit is set or the displayId is not a DP,
 *     this call will return LW_ERR_ILWALID_ARGUMENT.
 *   ctrl
 *     Here are the current defined fields:
 *       LW0073_CTRL_DP_MAIN_LINK_CTRL_POWER_STATE_POWERDOWN
 *         This value will power down Main Link.
 *       LW0073_CTRL_DP_MAIN_LINK_CTRL_POWER_STATE_POWERUP
 *         This value will power up Main Link.
 *
*/
#define LW0073_CTRL_CMD_DP_MAIN_LINK_CTRL                       (0x731356U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_DP_MAIN_LINK_CTRL_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DP_MAIN_LINK_CTRL_PARAMS_MESSAGE_ID (0x56U)

typedef struct LW0073_CTRL_DP_MAIN_LINK_CTRL_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 ctrl;
} LW0073_CTRL_DP_MAIN_LINK_CTRL_PARAMS;

#define  LW0073_CTRL_DP_MAIN_LINK_CTRL_POWER_STATE                          0:0
#define LW0073_CTRL_DP_MAIN_LINK_CTRL_POWER_STATE_POWERDOWN (0x00000000U)
#define LW0073_CTRL_DP_MAIN_LINK_CTRL_POWER_STATE_POWERUP   (0x00000001U)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

/*
 * LW0073_CTRL_CMD_DP_SET_AUXCH_ATTRIBUTES
 *
 * This command is used to set various DP Aux Channel Transactions settings
 * for the specified displayId such max amount of deferred retries.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the DP display which owns
 *     the Main Link to be adjusted.  The display ID must a DP display
 *     as determined with the LW0073_CTRL_CMD_SPECIFIC_GET_TYPE command.
 *     If more than one displayId bit is set or the displayId is not a DP,
 *     this call will return LW_ERR_ILWALID_ARGUMENT.
 *   cmd
 *     This parameter is an input to this command.
 *     Here are the current defined fields:
 *  LW0073_CTRL_DP_SET_AUXCH_ATTRIBUTES_CMD_DEFER_MAXRETRIES
 *         Set to specify the maximum amount of defer retries to be used during
 *     Aux Channel Transactions.
 *           LW0073_CTRL_DP_SET_AUXCH_ATTRIBUTES_CMD_DEFER_MAXRETRIES_FALSE
 *             No request to set amount of defer retries.
 *           LW0073_CTRL_DP_SET_AUXCH_ATTRIBUTES_CMD_DEFER_MAXRETRIES_TRUE
 *             Set this value to indicate amount of defer retries change.
 *   data
 *     This parameter is an input to this command.
 *     Here are the current defined fields:
 *  LW0073_CTRL_DP_SET_AUXCH_ATTRIBUTES_DATA_DEFER_MAXRETRIES
 *     This field specifies the desired maximum amount of defer retries. A client
 *     may choose a valid value between 7 to 63.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *
 */
#define LW0073_CTRL_CMD_DP_SET_AUXCH_ATTRIBUTES             (0x731357U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_DP_SET_AUXCH_ATTRIBUTES_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DP_SET_AUXCH_ATTRIBUTES_PARAMS_MESSAGE_ID (0x57U)

typedef struct LW0073_CTRL_DP_SET_AUXCH_ATTRIBUTES_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 cmd;
    LwU32 data;
} LW0073_CTRL_DP_SET_AUXCH_ATTRIBUTES_PARAMS;

#define LW0073_CTRL_DP_SET_AUXCH_ATTRIBUTES_CMD_DEFER_MAXRETRIES               0:0
#define LW0073_CTRL_DP_SET_AUXCH_ATTRIBUTES_CMD_DEFER_MAXRETRIES_FALSE (0x00000000U)
#define LW0073_CTRL_DP_SET_AUXCH_ATTRIBUTES_CMD_DEFER_MAXRETRIES_TRUE  (0x00000001U)

#define LW0073_CTRL_DP_SET_AUXCH_ATTRIBUTES_DATA_DEFER_MAXRETRIES          5:0

/*
 * LW0073_CTRL_CMD_DP_GET_AUDIO_MUTESTREAM
 *
 * This command returns the current audio mute state on the main link of Display Port
 *
 * The command takes a LW0073_CTRL_DP_GET_AUDIO_MUTESTREAM_PARAMS structure as the
 * argument with the appropriate subDeviceInstance, displayId as inputs and returns the
 * current mute status in mute field of the structure.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the display for which the audio stream
 *     state should be returned.  The display ID must a DP display.
 *     If the display ID is invalid or if it is not a DP display,
 *     this call will return LW_ERR_ILWALID_ARGUMENT.
 *   mute
 *     This parameter will return one of the following values:
 *       LW0073_CTRL_DP_AUDIO_MUTESTREAM_MUTE_DISABLE
 *         Audio mute is lwrrently disabled.
 *       LW0073_CTRL_DP_AUDIO_MUTESTREAM_MUTE_ENABLE
 *         Audio mute is lwrrently enabled.
 *       LW0073_CTRL_DP_AUDIO_MUTESTREAM_MUTE_AUTO
 *         Audio mute is automatically controlled by hardware.
 *       LW0073_CTRL_DP_AUDIO_MUTESTREAM_MUTE_UNKNOWN
 *         Audio mute is lwrrently in an unknown state.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *
 *
 */
#define LW0073_CTRL_CMD_DP_GET_AUDIO_MUTESTREAM                        (0x731358U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_DP_GET_AUDIO_MUTESTREAM_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DP_GET_AUDIO_MUTESTREAM_PARAMS_MESSAGE_ID (0x58U)

typedef struct LW0073_CTRL_DP_GET_AUDIO_MUTESTREAM_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 mute;
} LW0073_CTRL_DP_GET_AUDIO_MUTESTREAM_PARAMS;

#define LW0073_CTRL_DP_AUDIO_MUTESTREAM_MUTE_DISABLE (0x00000000U)
#define LW0073_CTRL_DP_AUDIO_MUTESTREAM_MUTE_ENABLE  (0x00000001U)
#define LW0073_CTRL_DP_AUDIO_MUTESTREAM_MUTE_AUTO    (0x00000002U)
#define LW0073_CTRL_DP_AUDIO_MUTESTREAM_MUTE_UNKNOWN (0x00000003U)

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW0073_CTRL_CMD_DP_SET_AUDIO_MUTESTREAM
 *
 * This command sets the current audio mute state on the main link of Display Port
 *
 * The command takes a LW0073_CTRL_DP_SET_AUDIO_MUTESTREAM_PARAMS structure as the
 * argument with the appropriate subDeviceInstance, displayId as inputs and whether to enable
 * or disable mute in the parameter - mute.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the display for which the audio stream
 *     state should be returned.  The display ID must a DP display.
 *     If the display ID is invalid or if it is not a DP display,
 *     this call will return LW_ERR_ILWALID_ARGUMENT.
 *   mute
 *     This parameter is an input to this command.
 *     Here are the current defined values:
 *       LW0073_CTRL_DP_AUDIO_MUTESTREAM_MUTE_DISABLE
 *         Audio mute will be disabled.
 *       LW0073_CTRL_DP_AUDIO_MUTESTREAM_MUTE_ENABLE
 *         Audio mute will be enabled.
 *       LW0073_CTRL_DP_AUDIO_MUTESTREAM_MUTE_AUTO
 *         Audio mute will be automatically controlled by hardware.
 *
 *      Note:  Any other value for mute in LW0073_CTRL_DP_SET_AUDIO_MUTESTREAM_PARAMS is not allowed and
 *              the API will return an error.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *
 *
 */
#define LW0073_CTRL_CMD_DP_SET_AUDIO_MUTESTREAM      (0x731359U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_DP_SET_AUDIO_MUTESTREAM_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DP_SET_AUDIO_MUTESTREAM_PARAMS_MESSAGE_ID (0x59U)

typedef struct LW0073_CTRL_DP_SET_AUDIO_MUTESTREAM_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 mute;
} LW0073_CTRL_DP_SET_AUDIO_MUTESTREAM_PARAMS;

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

/*
 * LW0073_CTRL_CMD_DP_ASSR_CTRL
 *
 * This command is used to control and query DisplayPort ASSR
 * settings for the specified displayId.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the DP display which owns
 *     the Main Link to be adjusted.  The display ID must a DP display
 *     as determined with the LW0073_CTRL_CMD_SPECIFIC_GET_TYPE command.
 *     If more than one displayId bit is set or the displayId is not a DP,
 *     this call will return LW_ERR_ILWALID_ARGUMENT.
 *   cmd
 *     This input parameter specifies the command to execute.  Legal
 *     values for this parameter include:
 *       LW0073_CTRL_DP_ASSR_CMD_QUERY_STATE
 *         This field can be used to query ASSR state. When used the ASSR
 *         state value is returned in the data parameter.
 *       LW0073_CTRL_DP_ASSR_CMD_DISABLE
 *         This field can be used to control the ASSR disable state.
 *       LW0073_CTRL_DP_ASSR_CMD_FORCE_STATE
 *         This field can be used to control ASSR State without looking at
 *         whether the display supports it. Used in conjunction with
 *         Fake link training. Note that this updates the state on the
 *         source side only. The sink is assumed to be configured for ASSR
 *         by the client (DD).
 *   data
 *     This parameter specifies the data associated with the cmd
 *     parameter.
 *       LW0073_CTRL_DP_ASSR_DATA_STATE_ENABLED
 *         This field indicates the state of ASSR when queried using cmd
 *         parameter. When used to control the State, it indicates whether
 *         ASSR should be enabled or disabled.
 *           LW0073_CTRL_DP_ASSR_DATA_STATE_ENABLED_NO
 *             When queried this flag indicates that ASSR is not enabled on the sink.
 *             When used as the data for CMD_FORCE_STATE, it requests ASSR to
 *             to be disabled on the source side.
 *           LW0073_CTRL_DP_ASSR_DATA_STATE_ENABLED_YES
 *             When queried this flag indicates that ASSR is not enabled on the sink.
 *             When used as the data for CMD_FORCE_STATE, it requests ASSR to
 *             to be enabled on the source side.
 *   err
 *     This output parameter specifies any errors associated with the cmd
 *     parameter.
 *       LW0073_CTRL_DP_ASSR_ERR_CAP
 *         This field indicates the error pertaining to ASSR capability of
 *         the sink device.
 *           LW0073_CTRL_DP_ASSR_ERR_CAP_NOERR
 *             This flag indicates there is no error.
 *           LW0073_CTRL_DP_ASSR_ERR_CAP_ERR
 *             This flag indicates that the sink is not ASSR capable.
 *
 *  Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *
 */
#define LW0073_CTRL_CMD_DP_ASSR_CTRL (0x73135aU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_DP_ASSR_CTRL_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DP_ASSR_CTRL_PARAMS_MESSAGE_ID (0x5AU)

typedef struct LW0073_CTRL_DP_ASSR_CTRL_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 cmd;
    LwU32 data;
    LwU32 err;
} LW0073_CTRL_DP_ASSR_CTRL_PARAMS;

#define  LW0073_CTRL_DP_ASSR_CMD                                31:0
#define LW0073_CTRL_DP_ASSR_CMD_QUERY_STATE             (0x00000001U)
#define LW0073_CTRL_DP_ASSR_CMD_DISABLE                 (0x00000002U)
#define LW0073_CTRL_DP_ASSR_CMD_FORCE_STATE             (0x00000003U)
#define LW0073_CTRL_DP_ASSR_CMD_ENABLE                  (0x00000004U)
#define  LW0073_CTRL_DP_ASSR_DATA_STATE_ENABLED                  0:0
#define LW0073_CTRL_DP_ASSR_DATA_STATE_ENABLED_NO       (0x00000000U)
#define LW0073_CTRL_DP_ASSR_DATA_STATE_ENABLED_YES      (0x00000001U)
#define  LW0073_CTRL_DP_ASSR_ERR_CAP                             0:0
#define LW0073_CTRL_DP_ASSR_ERR_CAP_NOERR               (0x00000000U)
#define LW0073_CTRL_DP_ASSR_ERR_CAP_ERR                 (0x00000001U)
/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW0073_CTRL_CMD_DP_TOPOLOGY_ALLOCATE_DISPLAYID
 *
 * This command is used to assign a displayId from the free pool
 * to a specific AUX Address in a DP 1.2 topology.  The topology
 * is uniquely identified by the DisplayId of the DP connector.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This is the DisplayId of the DP connector to which the topology
 *     is rooted.
 *   preferredDisplayId
 *      Client can sent a preferredDisplayID which RM can use during allocation
 *      if available. If this Id is a part of allDisplayMask in RM then we return
 *      a free available Id to the client. However, if this is set to
 *      LW0073_CTRL_CMD_DP_ILWALID_PREFERRED_DISPLAY_ID then we return allDisplayMask value.
 *   useBFM
 *      Set to true if DP-BFM is used during emulation/RTL Sim.
 *
 *   [out] displayIdAssigned
 *     This is the out field that will receive the new displayId.  If the
 *     function fails this is guaranteed to be 0.
 *   [out] allDisplayMask
 *      This is allDisplayMask RM variable which is returned only when
 *      preferredDisplayId is set to LW0073_CTRL_CMD_DP_ILWALID_PREFERRED_DISPLAY_ID
 *
 *
 *  Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *
 */
#define LW0073_CTRL_CMD_DP_TOPOLOGY_ALLOCATE_DISPLAYID  (0x73135bU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_TOPOLOGY_ALLOCATE_DISPLAYID_PARAMS_MESSAGE_ID" */

/*
 *  There cannot be more than 128 devices in a topology (also by DP 1.2 specification)
 *  NOTE: Temporarily lowered to pass XAPI RM tests. Should be reevaluated!
 */
#define LW0073_CTRL_CMD_DP_MAX_TOPOLOGY_NODES           120U
#define LW0073_CTRL_CMD_DP_ILWALID_PREFERRED_DISPLAY_ID 0xffffffffU

#define LW0073_CTRL_CMD_DP_TOPOLOGY_ALLOCATE_DISPLAYID_PARAMS_MESSAGE_ID (0x5BU)

typedef struct LW0073_CTRL_CMD_DP_TOPOLOGY_ALLOCATE_DISPLAYID_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  displayId;
    LwU32  preferredDisplayId;

    LwBool force;
    LwBool useBFM;

    LwU32  displayIdAssigned;
    LwU32  allDisplayMask;
} LW0073_CTRL_CMD_DP_TOPOLOGY_ALLOCATE_DISPLAYID_PARAMS;

/*
 * LW0073_CTRL_CMD_DP_TOPOLOGY_FREE_DISPLAYID
 *
 * This command is used to return a multistream displayid to the unused pool.
 * You must not call this function while either the ARM or ASSEMBLY state cache
 * refers to this display-id.  The head must not be attached.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This is the displayId to free.
 *
 *  Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *
 *
 */
#define LW0073_CTRL_CMD_DP_TOPOLOGY_FREE_DISPLAYID (0x73135lw) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_TOPOLOGY_FREE_DISPLAYID_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_TOPOLOGY_FREE_DISPLAYID_PARAMS_MESSAGE_ID (0x5LW)

typedef struct LW0073_CTRL_CMD_DP_TOPOLOGY_FREE_DISPLAYID_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
} LW0073_CTRL_CMD_DP_TOPOLOGY_FREE_DISPLAYID_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW0073_CTRL_CMD_DP_TOPOLOGY_QUERY
 *
 * This command is used to query the multistream topology rooted at a connector.
 * If the connector does not support DP 1.2
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This is the DisplayId of the DP connector to which the topology
 *     is rooted.
 *   numNodes
 *     This out parameter returns the number of nodes returned in the
 *     addresses structure.
 *   nodes
 *     An array of structures, each representing one node on a DP 1.2 topology.
 *       address
 *         The aux address of the given node. The lowest nibble specifies the number
 *         of hops present in the remaining address. Each hop is represented as a nibble
 *
 *         CAVEAT: We consider the complete aux address to be the address + downstream
 *                 port.  The specification considers addresses to only point to branch
 *                 devices.
 *       id
 *         Unique identifier for the given node:
 *           branchGuid
 *             Valid for non-sinks, this is a unique identifier for branch devices.
 *           displayId
 *             Valid for sinks only, the displayId if assigned. NOTE: On boot the
 *             VBIOS-driven display automatically is assigned by RM.
 *       flags
 *         Contains three useful pieces of information about a node:
 *
 *  Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *
 *
 */
#define LW0073_CTRL_CMD_DP_TOPOLOGY_QUERY                               (0x73135dU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_TOPOLOGY_QUERY_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DP_TOPOLOGY_QUERY_FLAGS_VIDEO_SINK                            0:0
#define LW0073_CTRL_DP_TOPOLOGY_QUERY_FLAGS_VIDEO_SINK_NO               (0x00000000U)
#define LW0073_CTRL_DP_TOPOLOGY_QUERY_FLAGS_VIDEO_SINK_YES              (0x00000001U)
#define LW0073_CTRL_DP_TOPOLOGY_QUERY_FLAGS_AUDIO_SINK                            1:1
#define LW0073_CTRL_DP_TOPOLOGY_QUERY_FLAGS_AUDIO_SINK_NO               (0x00000000U)
#define LW0073_CTRL_DP_TOPOLOGY_QUERY_FLAGS_AUDIO_SINK_YES              (0x00000001U)
#define LW0073_CTRL_DP_TOPOLOGY_QUERY_FLAGS_CONNECTOR_TYPE                        3:2
#define LW0073_CTRL_DP_TOPOLOGY_QUERY_FLAGS_CONNECTOR_TYPE_DP           (0x00000000U)
#define LW0073_CTRL_DP_TOPOLOGY_QUERY_FLAGS_CONNECTOR_TYPE_HDMI         (0x00000001U)
#define LW0073_CTRL_DP_TOPOLOGY_QUERY_FLAGS_CONNECTOR_TYPE_DVI          (0x00000002U)
#define LW0073_CTRL_DP_TOPOLOGY_QUERY_FLAGS_CONNECTOR_TYPE_VGA          (0x00000003U)
#define LW0073_CTRL_DP_TOPOLOGY_QUERY_FLAGS_PRESENT                               4:4
#define LW0073_CTRL_DP_TOPOLOGY_QUERY_FLAGS_PRESENT_NO                  (0x00000000U)
#define LW0073_CTRL_DP_TOPOLOGY_QUERY_FLAGS_PRESENT_YES                 (0x00000001U)
#define LW0073_CTRL_DP_TOPOLOGY_QUERY_FLAGS_LOOP                                  5:5
#define LW0073_CTRL_DP_TOPOLOGY_QUERY_FLAGS_LOOP_NO                     (0x00000000U)
#define LW0073_CTRL_DP_TOPOLOGY_QUERY_FLAGS_LOOP_YES                    (0x00000001U)
#define LW0073_CTRL_DP_TOPOLOGY_QUERY_FLAGS_REDUNDANT                             6:6
#define LW0073_CTRL_DP_TOPOLOGY_QUERY_FLAGS_REDUNDANT_NO                (0x00000000U)
#define LW0073_CTRL_DP_TOPOLOGY_QUERY_FLAGS_REDUNDANT_YES               (0x00000001U)

#define LW0073_CTRL_DP_TOPOLOGY_QUERY_ADDRESS_NUM_HOPS                            3:0
#define LW0073_CTRL_DP_TOPOLOGY_QUERY_ADDRESS_NUM_HOPS_MAX_WITH_HDCP    (0x00000007U)
#define LW0073_CTRL_DP_TOPOLOGY_QUERY_ADDRESS_NUM_HOPS_MAX_WITHOUT_HDCP (0x0000000fU)

typedef struct LW0073_CTRL_CMD_DP_TOPOLOGY_NODE_DATA {
    LW_DECLARE_ALIGNED(LwU64 address, 8);
    LwU32 branchGuid[4];
    LwU32 displayId;
    LwU8  flags;
} LW0073_CTRL_CMD_DP_TOPOLOGY_NODE_DATA;

#define LW0073_CTRL_CMD_DP_TOPOLOGY_QUERY_PARAMS_MESSAGE_ID (0x5DU)

typedef struct LW0073_CTRL_CMD_DP_TOPOLOGY_QUERY_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;

    LwU8  numNodes;

    LW_DECLARE_ALIGNED(LW0073_CTRL_CMD_DP_TOPOLOGY_NODE_DATA nodes[LW0073_CTRL_CMD_DP_MAX_TOPOLOGY_NODES], 8);
} LW0073_CTRL_CMD_DP_TOPOLOGY_QUERY_PARAMS;
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW0073_CTRL_CMD_DP_GET_LINK_CONFIG
 *
 * This command is used to query DisplayPort link config
 * settings on the transmitter side.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the DP display which owns
 *     the Main Link to be queried.
 *     If more than one displayId bit is set or the displayId is not a DP,
 *     this call will return LW_ERR_ILWALID_ARGUMENT.
 *   laneCount
 *     Number of lanes the DP transmitter hardware is set up to drive.
 *   linkBW
 *     The BW of each lane that the DP transmitter hardware is set up to drive.
 *     The values returned will be according to the DP specifications.
 *
 */
#define LW0073_CTRL_CMD_DP_GET_LINK_CONFIG (0x731360U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_DP_GET_LINK_CONFIG_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DP_GET_LINK_CONFIG_PARAMS_MESSAGE_ID (0x60U)

typedef struct LW0073_CTRL_DP_GET_LINK_CONFIG_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 laneCount;
    LwU32 linkBW;
} LW0073_CTRL_DP_GET_LINK_CONFIG_PARAMS;

#define LW0073_CTRL_CMD_DP_GET_LINK_CONFIG_LANE_COUNT                          3:0
#define LW0073_CTRL_CMD_DP_GET_LINK_CONFIG_LANE_COUNT_0     (0x00000000U)
#define LW0073_CTRL_CMD_DP_GET_LINK_CONFIG_LANE_COUNT_1     (0x00000001U)
#define LW0073_CTRL_CMD_DP_GET_LINK_CONFIG_LANE_COUNT_2     (0x00000002U)
#define LW0073_CTRL_CMD_DP_GET_LINK_CONFIG_LANE_COUNT_4     (0x00000004U)
#define LW0073_CTRL_CMD_DP_GET_LINK_CONFIG_LINK_BW                             3:0
#define LW0073_CTRL_CMD_DP_GET_LINK_CONFIG_LINK_BW_1_62GBPS (0x00000006U)
#define LW0073_CTRL_CMD_DP_GET_LINK_CONFIG_LINK_BW_2_70GBPS (0x0000000aU)

/*
 * LW0073_CTRL_CMD_DP_GET_EDP_DATA
 *
 * This command is used to query Embedded DisplayPort information.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     This parameter specifies the ID of the eDP display which owns
 *     the Main Link to be queried.
 *     If more than one displayId bit is set or the displayId is not a eDP,
 *     this call will return LW_ERR_ILWALID_ARGUMENT.
 *   data
 *     This output parameter specifies the data associated with the eDP display.
 *     It is only valid if this function returns LW_OK.
 *       LW0073_CTRL_CMD_DP_GET_EDP_DATA_PANEL_POWER
 *         This field indicates the state of the eDP panel power.
 *           LW0073_CTRL_CMD_DP_GET_EDP_DATA_PANEL_POWER_OFF
 *             This eDP panel is powered off.
 *           LW0073_CTRL_CMD_DP_GET_EDP_DATA_PANEL_POWER_ON
 *             This eDP panel is powered on.
 *       LW0073_CTRL_CMD_DP_GET_EDP_DATA_DPCD_POWER_OFF
 *         This field tells the client if DPCD power off command
 *         should be used for the current eDP panel.
 *           LW0073_CTRL_DP_GET_EDP_DATA_DPCD_POWER_OFF_ENABLE
 *             This eDP panel can use DPCD to power off the panel.
 *           LW0073_CTRL_DP_GET_EDP_DATA_DPCD_POWER_OFF_DISABLE
 *             This eDP panel cannot use DPCD to power off the panel.
 *       LW0073_CTRL_DP_GET_EDP_DATA_DPCD_SET_POWER
 *         This field tells the client current eDP panel DPCD SET_POWER (0x600) status
 *            LW0073_CTRL_DP_GET_EDP_DATA_DPCD_SET_POWER_D0
 *              This eDP panel is current up and in full power mode.
 *            LW0073_CTRL_DP_GET_EDP_DATA_DPCD_SET_POWER_D3
 *              This eDP panel is current standby.
 */
#define LW0073_CTRL_CMD_DP_GET_EDP_DATA                     (0x731361U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_DP_GET_EDP_DATA_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DP_GET_EDP_DATA_PARAMS_MESSAGE_ID (0x61U)

typedef struct LW0073_CTRL_DP_GET_EDP_DATA_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 data;
} LW0073_CTRL_DP_GET_EDP_DATA_PARAMS;

#define LW0073_CTRL_DP_GET_EDP_DATA_PANEL_POWER                                0:0
#define LW0073_CTRL_DP_GET_EDP_DATA_PANEL_POWER_OFF        (0x00000000U)
#define LW0073_CTRL_DP_GET_EDP_DATA_PANEL_POWER_ON         (0x00000001U)
#define LW0073_CTRL_DP_GET_EDP_DATA_DPCD_POWER_OFF                             1:1
#define LW0073_CTRL_DP_GET_EDP_DATA_DPCD_POWER_OFF_ENABLE  (0x00000000U)
#define LW0073_CTRL_DP_GET_EDP_DATA_DPCD_POWER_OFF_DISABLE (0x00000001U)
#define LW0073_CTRL_DP_GET_EDP_DATA_DPCD_POWER_STATE                           2:2
#define LW0073_CTRL_DP_GET_EDP_DATA_DPCD_POWER_STATE_D0    (0x00000000U)
#define LW0073_CTRL_DP_GET_EDP_DATA_DPCD_POWER_STATE_D3    (0x00000001U)
/*
 * LW0073_CTRL_CMD_DP_CONFIG_STREAM
 *
 * This command sets various multi/single stream related params for
 * for a given head.
 *
 *   subDeviceInstance
 *          This parameter specifies the subdevice instance within the
 *          LW04_DISPLAY_COMMON parent device to which the operation should be
 *          directed. This parameter must specify a value between zero and the
 *          total number of subdevices within the parent device.  This parameter
 *          should be set to zero for default behavior.
 *      Head
 *          Specifies the head index for the stream.
 *      sorIndex
 *          Specifies the SOR index for the stream.
 *      dpLink
 *          Specifies the DP link: either 0 or 1 (A , B)
 *      bEnableOverride
 *          Specifies whether we're manually configuring this stream.
 *          If not set, none of the remaining settings have any effect.
 *      bMST
 *          Specifies whether in Multistream or Singlestream mode.
 *      MST/SST
 *          Structures for passing in either Multistream or Singlestream params
 *      slotStart
 *          Specifies the start value of the timeslot
 *      slotEnd
 *          Specifies the end value of the timeslot
 *      PBN
 *          Specifies the PBN for the timeslot.
 *      minHBlank
 *          Specifies the min HBlank
 *      milwBlank
 *          Specifies the min VBlank
 *      sendACT   -- deprecated. A new control call has been added.
 *          Specifies whether ACT has to be sent or not.
 *      tuSize
 *          Specifies TU size value
 *      watermark
 *          Specifies stream watermark.
 *      linkClkFreqHz -- moving to MvidWarParams. Use that instead.
 *          Specifies the link freq in Hz. Note that this is the byte clock.
 *          eg: = (5.4 Ghz / 10)
 *      actualPclkHz; -- moving to MvidWarParams. Use that instead.
 *          Specifies the actual pclk freq in Hz.
 *      mvidWarEnabled
 *          Specifies whether MVID WAR is enabled.
 *      MvidWarParams
 *          Is valid if mvidWarEnabled is true.
 *      bEnableTwoHeadOneOr
 *          Whether two head one OR is enabled. If this is set then RM will
 *          replicate SF settings of Master head on Slave head. Head index
 *          passed should be of Master Head.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC: when this command has already been called
 *
 */
#define LW0073_CTRL_CMD_DP_CONFIG_STREAM                   (0x731362U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_CONFIG_STREAM_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_CONFIG_STREAM_PARAMS_MESSAGE_ID (0x62U)

typedef struct LW0073_CTRL_CMD_DP_CONFIG_STREAM_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  head;
    LwU32  sorIndex;
    LwU32  dpLink;

    LwBool bEnableOverride;
    LwBool bMST;
    LwU32  singleHeadMultistreamMode;
    LwU32  hBlankSym;
    LwU32  vBlankSym;
    LwU32  colorFormat;
    LwBool bEnableTwoHeadOneOr;

    struct {
        LwU32  slotStart;
        LwU32  slotEnd;
        LwU32  PBN;
        LwU32  Timeslice;
        LwBool sendACT;          // deprecated -Use LW0073_CTRL_CMD_DP_SEND_ACT
        LwU32  singleHeadMSTPipeline;
        LwBool bEnableAudioOverRightPanel;
    } MST;

    struct {
        LwBool bEnhancedFraming;
        LwU32  tuSize;
        LwU32  waterMark;
        LwU32  actualPclkHz;     // deprecated  -Use MvidWarParams
        LwU32  linkClkFreqHz;    // deprecated  -Use MvidWarParams
        LwBool bEnableAudioOverRightPanel;
        struct {
            LwU32  activeCnt;
            LwU32  activeFrac;
            LwU32  activePolarity;
            LwBool mvidWarEnabled;
            struct {
                LwU32 actualPclkHz;
                LwU32 linkClkFreqHz;
            } MvidWarParams;
        } Legacy;
    } SST;
} LW0073_CTRL_CMD_DP_CONFIG_STREAM_PARAMS;

/*
 * LW0073_CTRL_CMD_DP_SET_RATE_GOV
 *
 * This command enables rate governing for a MST.
 *
 *      subDeviceInstance
 *          This parameter specifies the subdevice instance within the
 *          LW04_DISPLAY_COMMON parent device to which the operation should be
 *          directed. This parameter must specify a value between zero and the
 *          total number of subdevices within the parent device.  This parameter
 *          should be set to zero for default behavior.
 *      Head
 *          Specifies the head index for the stream.
 *      sorIndex
 *          Specifies the SOR index for the stream.
 *      flags
 *          Specifies Rate Governing, trigger type and wait on trigger and operation type.
 *
 *     _FLAGS_OPERATION: whether this control call should program or check for status of previous operation.
 *
 *     _FLAGS_STATUS: Out only. Caller should check the status for _FLAGS_OPERATION_CHECK_STATUS through
 *                    this bit.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC: when this command has already been called
 *
 */
#define LW0073_CTRL_CMD_DP_SET_RATE_GOV (0x731363U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_SET_RATE_GOV_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_SET_RATE_GOV_PARAMS_MESSAGE_ID (0x63U)

typedef struct LW0073_CTRL_CMD_DP_SET_RATE_GOV_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 head;
    LwU32 sorIndex;
    LwU32 flags;
} LW0073_CTRL_CMD_DP_SET_RATE_GOV_PARAMS;

#define LW0073_CTRL_CMD_DP_SET_RATE_GOV_FLAGS_ENABLE_RG                0:0
#define LW0073_CTRL_CMD_DP_SET_RATE_GOV_FLAGS_ENABLE_RG_OFF          (0x00000000U)
#define LW0073_CTRL_CMD_DP_SET_RATE_GOV_FLAGS_ENABLE_RG_ON           (0x00000001U)
#define LW0073_CTRL_CMD_DP_SET_RATE_GOV_FLAGS_TRIGGER_MODE             1:1
#define LW0073_CTRL_CMD_DP_SET_RATE_GOV_FLAGS_TRIGGER_MODE_LOADV     (0x00000000U)
#define LW0073_CTRL_CMD_DP_SET_RATE_GOV_FLAGS_TRIGGER_MODE_IMMEDIATE (0x00000001U)
#define LW0073_CTRL_CMD_DP_SET_RATE_GOV_FLAGS_WAIT_TRIGGER             2:2
#define LW0073_CTRL_CMD_DP_SET_RATE_GOV_FLAGS_WAIT_TRIGGER_OFF       (0x00000000U)
#define LW0073_CTRL_CMD_DP_SET_RATE_GOV_FLAGS_WAIT_TRIGGER_ON        (0x00000001U)
#define LW0073_CTRL_CMD_DP_SET_RATE_GOV_FLAGS_OPERATION                3:3
#define LW0073_CTRL_CMD_DP_SET_RATE_GOV_FLAGS_OPERATION_PROGRAM      (0x00000000U)
#define LW0073_CTRL_CMD_DP_SET_RATE_GOV_FLAGS_OPERATION_CHECK_STATUS (0x00000001U)
#define LW0073_CTRL_CMD_DP_SET_RATE_GOV_FLAGS_STATUS                   31:31
#define LW0073_CTRL_CMD_DP_SET_RATE_GOV_FLAGS_STATUS_FAIL            (0x00000000U)
#define LW0073_CTRL_CMD_DP_SET_RATE_GOV_FLAGS_STATUS_PASS            (0x00000001U)

/*
 * LW0073_CTRL_CMD_DP_SET_MANUAL_DISPLAYPORT
 *
 *  This call is used by the displayport library.  Once
 *  all of the platforms have ported, this call will be
 *  deprecated and made the default behavior.
 *
 *   Disables automatic watermark programming
 *   Disables automatic DP IRQ handling (CP IRQ)
 *   Disables automatic retry on defers
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
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *
 */
#define LW0073_CTRL_CMD_DP_SET_MANUAL_DISPLAYPORT                    (0x731365U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_SET_MANUAL_DISPLAYPORT_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_SET_MANUAL_DISPLAYPORT_PARAMS_MESSAGE_ID (0x65U)

typedef struct LW0073_CTRL_CMD_DP_SET_MANUAL_DISPLAYPORT_PARAMS {
    LwU32 subDeviceInstance;
} LW0073_CTRL_CMD_DP_SET_MANUAL_DISPLAYPORT_PARAMS;

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

/*
 * LW0073_CTRL_CMD_DP_SET_ECF
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   sorIndex
 *     This parameter specifies the Index of sor for which ecf
 *     should be updated.
 *   ecf
 *      This parameter has the ECF bit mask.
 *
 *  Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *
 */
#define LW0073_CTRL_CMD_DP_SET_ECF (0x731366U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_SET_ECF_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_SET_ECF_PARAMS_MESSAGE_ID (0x66U)

typedef struct LW0073_CTRL_CMD_DP_SET_ECF_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  sorIndex;
    LW_DECLARE_ALIGNED(LwU64 ecf, 8);
    LwBool bForceClearEcf;
    LwBool bAddStreamBack;
} LW0073_CTRL_CMD_DP_SET_ECF_PARAMS;
/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW0073_CTRL_CMD_DP_SEND_ACT
 *
 * This command sends ACT.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *
 *   displayId
 *      Specifies the root port displayId for which the trigger has to be done.
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_GENERIC: when this command has already been called
 *
 */
#define LW0073_CTRL_CMD_DP_SEND_ACT (0x731367U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_SEND_ACT_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_SEND_ACT_PARAMS_MESSAGE_ID (0x67U)

typedef struct LW0073_CTRL_CMD_DP_SEND_ACT_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
} LW0073_CTRL_CMD_DP_SEND_ACT_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW0073_CTRL_CMD_DP_SET_VERSION
 *
 * This command sets the value of the DisplayPort version number
 * used by Audio TimeStamp Header and Audio InfoFrame Packet Header.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   Head
 *     Specifies the head index for the stream.
 *   version
 *     Specifies the version number. Valid values for this parameter include:
 *         LW0073_CTRL_DP_SET_VERSION_11
 *         LW0073_CTRL_DP_SET_VERSION_12
 *         LW0073_CTRL_DP_SET_VERSION_13
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_NOT_SUPPORTED
 *
 */
#define LW0073_CTRL_CMD_DP_SET_VERSION (0x731368U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_SET_VERSION_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_SET_VERSION_PARAMS_MESSAGE_ID (0x68U)

typedef struct LW0073_CTRL_CMD_DP_SET_VERSION_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 head;
    LwU32 version;
} LW0073_CTRL_CMD_DP_SET_VERSION_PARAMS;

#define LW0073_CTRL_DP_SET_VERSION_11 (0x00000000U)
#define LW0073_CTRL_DP_SET_VERSION_12 (0x00000001U)
#define LW0073_CTRL_DP_SET_VERSION_13 (0x00000002U)

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW0073_CTRL_CMD_DP_GET_CAPS
 *
 * This command returns the following info
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   sorIndex
 *     Specifies the SOR index.
 *   bIsDp12Supported
 *     Returns LW_TRUE if DP1.2 is supported by the GPU else LW_FALSE
 *   bIsMultistreamSupported
 *     Returns LW_TRUE if MST is supported by the GPU else LW_FALSE
 *   bIsSCEnabled
 *     Returns LW_TRUE if Stream cloning is supported by the GPU else LW_FALSE
 *   maxLinkRate
 *     Returns Maximum allowed orclk for DP mode of SOR
 *     1 signifies 5.40(HBR2), 2 signifies 2.70(HBR), 3 signifies 1.62(RBR)
 *   bHasIncreasedWatermarkLimits
 *     Returns LW_TRUE if the GPU uses higher watermark limits, else LW_FALSE
 *   bIsPC2Disabled
 *     Returns LW_TRUE if VBIOS flag to disable PostLwrsor2 is set, else LW_FALSE
 *   bFECSupported
 *     Returns LW_TRUE if GPU supports FEC, else LW_FALSE
 *   bIsTrainPhyRepeater
 *     Returns LW_TRUE if LTTPR Link Training feature is set
 *   bOverrideLinkBw
 *     Returns LW_TRUE if DFP limits defined in DCB have to be honored, else LW_FALSE
 *
 *   DSC caps -
 *      bDscSupported
 *          If GPU supports DSC or not
 *
 *      encoderColorFormatMask
 *          Mask of all color formats for which DSC
 *          encoding is supported by GPU
 *
 *      lineBufferSizeKB
 *          Size of line buffer.
 *
 *      rateBufferSizeKB
 *          Size of rate buffer per slice.
 *
 *      bitsPerPixelPrecision
 *          Bits per pixel precision for DSC e.g. 1/16, 1/8, 1/4, 1/2, 1bpp
 *
 *      maxNumHztSlices
 *          Maximum number of horizontal slices supported by DSC encoder
 *
 *      lineBufferBitDepth
 *          Bit depth used by the GPU to store the reconstructed pixels within
 *          the line buffer
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_NOT_SUPPORTED
 *
 */
#define LW0073_CTRL_CMD_DP_GET_CAPS   (0x731369U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_GET_CAPS_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_GET_CAPS_PARAMS_MESSAGE_ID (0x69U)

typedef struct LW0073_CTRL_CMD_DP_GET_CAPS_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  sorIndex;
    LwU32  maxLinkRate;
    LwBool bIsDp12Supported;
    LwBool bIsDp14Supported;
    LwBool bIsMultistreamSupported;
    LwBool bIsSCEnabled;
    LwBool bHasIncreasedWatermarkLimits;
    LwBool bIsPC2Disabled;
    LwBool isSingleHeadMSTSupported;
    LwBool bFECSupported;
    LwBool bIsTrainPhyRepeater;
    LwBool bOverrideLinkBw;

    struct {
        LwBool bDscSupported;
        LwU32  encoderColorFormatMask;
        LwU32  lineBufferSizeKB;
        LwU32  rateBufferSizeKB;
        LwU32  bitsPerPixelPrecision;
        LwU32  maxNumHztSlices;
        LwU32  lineBufferBitDepth;
    } DSC;
} LW0073_CTRL_CMD_DP_GET_CAPS_PARAMS;

#define LW0073_CTRL_CMD_DP_GET_CAPS_MAX_LINK_RATE                           2:0
#define LW0073_CTRL_CMD_DP_GET_CAPS_MAX_LINK_RATE_NONE                          (0x00000000U)
#define LW0073_CTRL_CMD_DP_GET_CAPS_MAX_LINK_RATE_1_62                          (0x00000001U)
#define LW0073_CTRL_CMD_DP_GET_CAPS_MAX_LINK_RATE_2_70                          (0x00000002U)
#define LW0073_CTRL_CMD_DP_GET_CAPS_MAX_LINK_RATE_5_40                          (0x00000003U)
#define LW0073_CTRL_CMD_DP_GET_CAPS_MAX_LINK_RATE_8_10                          (0x00000004U)

#define LW0073_CTRL_CMD_DP_GET_CAPS_DSC_ENCODER_COLOR_FORMAT_RGB                (0x00000001U)
#define LW0073_CTRL_CMD_DP_GET_CAPS_DSC_ENCODER_COLOR_FORMAT_Y_CB_CR_444        (0x00000002U)
#define LW0073_CTRL_CMD_DP_GET_CAPS_DSC_ENCODER_COLOR_FORMAT_Y_CB_CR_NATIVE_422 (0x00000004U)
#define LW0073_CTRL_CMD_DP_GET_CAPS_DSC_ENCODER_COLOR_FORMAT_Y_CB_CR_NATIVE_420 (0x00000008U)

#define LW0073_CTRL_CMD_DP_GET_CAPS_DSC_BITS_PER_PIXEL_PRECISION_1_16           (0x00000001U)
#define LW0073_CTRL_CMD_DP_GET_CAPS_DSC_BITS_PER_PIXEL_PRECISION_1_8            (0x00000002U)
#define LW0073_CTRL_CMD_DP_GET_CAPS_DSC_BITS_PER_PIXEL_PRECISION_1_4            (0x00000003U)
#define LW0073_CTRL_CMD_DP_GET_CAPS_DSC_BITS_PER_PIXEL_PRECISION_1_2            (0x00000004U)
#define LW0073_CTRL_CMD_DP_GET_CAPS_DSC_BITS_PER_PIXEL_PRECISION_1              (0x00000005U)

/*
 * LW0073_CTRL_CMD_DP_SET_MSA_PROPERTIES
 *
 * This command returns the following info
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     should be for DP only
 *   bEnableMSA
 *     To enable or disable MSA
 *   bStereoPhaseIlwerse
 *     To enable or disable Stereo Phase Ilwerse value
 *   bCacheMsaOverrideForNextModeset
 *     Cache the values and don't apply them until next modeset
 *   featureMask
 *     Enable/Disable mask of individual MSA property
 *   featureValues
 *     MSA property value to write
 *   pFeatureDebugValues
 *     It will actual MSA property value being written on HW.
 *     If its NULL then no error but return nothing
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_NOT_SUPPORTED
 *      LW_ERR_TIMEOUT
 *
 */
#define LW0073_CTRL_CMD_DP_SET_MSA_PROPERTIES                                   (0x73136aU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_SET_MSA_PROPERTIES_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_MSA_PROPERTIES_SYNC_POLARITY_LOW                     (0U)
#define LW0073_CTRL_CMD_DP_MSA_PROPERTIES_SYNC_POLARITY_HIGH                    (1U)

typedef struct LW0073_CTRL_DP_MSA_PROPERTIES_MASK {
    LwU8   miscMask[2];
    LwBool bRasterTotalHorizontal;
    LwBool bRasterTotalVertical;
    LwBool bActiveStartHorizontal;
    LwBool bActiveStartVertical;
    LwBool bSurfaceTotalHorizontal;
    LwBool bSurfaceTotalVertical;
    LwBool bSyncWidthHorizontal;
    LwBool bSyncPolarityHorizontal;
    LwBool bSyncHeightVertical;
    LwBool bSyncPolarityVertical;
    LwBool bReservedEnable[3];
} LW0073_CTRL_DP_MSA_PROPERTIES_MASK;

typedef struct LW0073_CTRL_DP_MSA_PROPERTIES_VALUES {
    LwU8  misc[2];
    LwU16 rasterTotalHorizontal;
    LwU16 rasterTotalVertical;
    LwU16 activeStartHorizontal;
    LwU16 activeStartVertical;
    LwU16 surfaceTotalHorizontal;
    LwU16 surfaceTotalVertical;
    LwU16 syncWidthHorizontal;
    LwU16 syncPolarityHorizontal;
    LwU16 syncHeightVertical;
    LwU16 syncPolarityVertical;
    LwU8  reserved[3];
} LW0073_CTRL_DP_MSA_PROPERTIES_VALUES;

#define LW0073_CTRL_CMD_DP_SET_MSA_PROPERTIES_PARAMS_MESSAGE_ID (0x6AU)

typedef struct LW0073_CTRL_CMD_DP_SET_MSA_PROPERTIES_PARAMS {
    LwU32                                subDeviceInstance;
    LwU32                                displayId;
    LwBool                               bEnableMSA;
    LwBool                               bStereoPhaseIlwerse;
    LwBool                               bCacheMsaOverrideForNextModeset;
    LW0073_CTRL_DP_MSA_PROPERTIES_MASK   featureMask;
    LW0073_CTRL_DP_MSA_PROPERTIES_VALUES featureValues;
    LW_DECLARE_ALIGNED(struct LW0073_CTRL_DP_MSA_PROPERTIES_VALUES *pFeatureDebugValues, 8);
} LW0073_CTRL_CMD_DP_SET_MSA_PROPERTIES_PARAMS;

/*
 * LW0073_CTRL_CMD_DP_GENERATE_FAKE_INTERRUPT
 *
 * This command can be used to ilwoke a fake interrupt for the operation of DP1.2 branch device
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   interruptType
 *     This parameter specifies the type of fake interrupt to be ilwoked. Possible values are:
 *     0 => IRQ
 *     1 => HPDPlug
 *     2 => HPDUnPlug
 *   displayId
 *     should be for DP only
 *
 */

#define LW0073_CTRL_CMD_DP_GENERATE_FAKE_INTERRUPT (0x73136bU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_GENERATE_FAKE_INTERRUPT_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_GENERATE_FAKE_INTERRUPT_PARAMS_MESSAGE_ID (0x6BU)

typedef struct LW0073_CTRL_CMD_DP_GENERATE_FAKE_INTERRUPT_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 interruptType;
} LW0073_CTRL_CMD_DP_GENERATE_FAKE_INTERRUPT_PARAMS;

#define LW0073_CTRL_CMD_DP_GENERATE_FAKE_INTERRUPT_IRQ    (0x00000000U)
#define LW0073_CTRL_CMD_DP_GENERATE_FAKE_INTERRUPT_PLUG   (0x00000001U)
#define LW0073_CTRL_CMD_DP_GENERATE_FAKE_INTERRUPT_UNPLUG (0x00000002U)

/*
 * LW0073_CTRL_CMD_DP_CONFIG_RAD_SCRATCH_REG
 *
 * This command sets the MS displayId lit up by driver for further use of VBIOS
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     should be for DP only
 *   activeDevAddr
 *     Active MS panel address
 *   sorIndex
 *     SOR Index
 *   dpLink
 *     DP Sub Link Index
 *   hopCount
 *     Maximum hopcounts in MS address
 *   dpMsDevAddrState
 *     DP Multistream Device Address State. The values can be
 *
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_TIMEOUT
 *
 */
#define LW0073_CTRL_CMD_DP_CONFIG_RAD_SCRATCH_REG         (0x73136lw) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_CONFIG_RAD_SCRATCH_REG_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_CONFIG_RAD_SCRATCH_REG_PARAMS_MESSAGE_ID (0x6LW)

typedef struct LW0073_CTRL_CMD_DP_CONFIG_RAD_SCRATCH_REG_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 activeDevAddr;
    LwU32 sorIndex;
    LwU32 dpLink;
    LwU32 hopCount;
    LwU32 dpMsDevAddrState;
} LW0073_CTRL_CMD_DP_CONFIG_RAD_SCRATCH_REG_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW0073_CTRL_CMD_DP_SET_DISPLAYPORT_IRQ_LOG
 *
 *  This call is used by the displayport library.
 *  Its main function is to enable the logging of AUX interrupts for DTI testing.
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
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *
 */
#define LW0073_CTRL_CMD_DP_SET_DISPLAYPORT_IRQ_LOG (0x73136dU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_SET_DISPLAYPORT_IRQ_LOG_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_SET_DISPLAYPORT_IRQ_LOG_PARAMS_MESSAGE_ID (0x6DU)

typedef struct LW0073_CTRL_CMD_DP_SET_DISPLAYPORT_IRQ_LOG_PARAMS {
    LwU32  subDeviceInstance;
    LwBool bEnable;
} LW0073_CTRL_CMD_DP_SET_DISPLAYPORT_IRQ_LOG_PARAMS;
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
* LW0073_CTRL_CMD_DP_SET_TRIGGER_SELECT
*
* This command configures a new bit, LW_PDISP_SF_DP_LINKCTL_TRIGGER_SELECT
* to indicate which pipeline will handle the
* time slots allocation in single head MST mode
*
*      subDeviceInstance
*          This parameter specifies the subdevice instance within the
*          LW04_DISPLAY_COMMON parent device to which the operation should be
*          directed. This parameter must specify a value between zero and the
*          total number of subdevices within the parent device.  This parameter
*          should be set to zero for default behavior
*      Head
*          Specifies the head index for the stream
*      sorIndex
*          Specifies the SOR index for the stream
*      streamIndex
*          Stream Identifier
*
*
* Possible status values returned are:
*      LW_OK
*      LW_ERR_ILWALID_ARGUMENT
*      LW_ERR_GENERIC: when this command has already been called
*
*/
#define LW0073_CTRL_CMD_DP_SET_TRIGGER_SELECT (0x73136fU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_SET_TRIGGER_SELECT_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_SET_TRIGGER_SELECT_PARAMS_MESSAGE_ID (0x6FU)

typedef struct LW0073_CTRL_CMD_DP_SET_TRIGGER_SELECT_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 head;
    LwU32 sorIndex;
    LwU32 singleHeadMSTPipeline;
} LW0073_CTRL_CMD_DP_SET_TRIGGER_SELECT_PARAMS;

/*
* LW0073_CTRL_CMD_DP_CONFIG_SINGLE_HEAD_MULTI_STREAM
*
*  This call is used by the displayport library.& clients of RM
*  Its main function is to configure single Head Multi stream mode
 * this call configures internal RM datastructures to support required mode.
*
*   subDeviceInstance
*     This parameter specifies the subdevice instance within the
*     LW04_DISPLAY_COMMON parent device to which the operation should be
*     directed. This parameter must specify a value between zero and the
*     total number of subdevices within the parent device.  This parameter
*     should be set to zero for default behavior.
*
*   displayIDs
*     This parameter specifies array of DP displayIds to be configured which are driven out from a single head.
*
*   numStreams
*     This parameter specifies number of streams driven from a single head
*     ex: for 2SST & 2MST its value is 2.
*
*   mode
*     This parameter specifies single head multi stream mode to be configured.
*
*   bSetConfigure
*     This parameter configures single head multistream mode
*     if TRUE it sets SST or MST based on 'mode' parameter and updates internal driver data structures with the given information.
*     if FALSE clears the configuration of single head multi stream mode.
*
*   vbiosPrimaryDispIdIndex
*    This parameter specifies vbios master displayID index in displayIDs input array.
*
*  Possible status values returned are:
*   LW_OK
*   LW_ERR_ILWALID_ARGUMENT
*   LW_ERR_NOT_SUPPORTED
*
*/
#define LW0073_CTRL_CMD_DP_CONFIG_SINGLE_HEAD_MULTI_STREAM (0x73136eU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_CONFIG_SINGLE_HEAD_MULTI_STREAM_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_SINGLE_HEAD_MAX_STREAMS         (0x00000002U)
#define LW0073_CTRL_CMD_DP_CONFIG_SINGLE_HEAD_MULTI_STREAM_PARAMS_MESSAGE_ID (0x6EU)

typedef struct LW0073_CTRL_CMD_DP_CONFIG_SINGLE_HEAD_MULTI_STREAM_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  displayIDs[LW0073_CTRL_CMD_DP_SINGLE_HEAD_MAX_STREAMS];
    LwU32  numStreams;
    LwU32  mode;
    LwBool bSetConfig;
    LwU8   vbiosPrimaryDispIdIndex;
} LW0073_CTRL_CMD_DP_CONFIG_SINGLE_HEAD_MULTI_STREAM_PARAMS;

#define LW0073_CTRL_CMD_DP_SINGLE_HEAD_MULTI_STREAM_NONE     (0x00000000U)
#define LW0073_CTRL_CMD_DP_SINGLE_HEAD_MULTI_STREAM_MODE_SST (0x00000001U)
#define LW0073_CTRL_CMD_DP_SINGLE_HEAD_MULTI_STREAM_MODE_MST (0x00000002U)

/*
* LW0073_CTRL_CMD_DP_SET_TRIGGER_ALL
*
* This command configures a new bit, LW_PDISP_SF_DP_LINKCTL_TRIGGER_ALL
* to indicate which if all the pipelines to take affect on ACT (sorFlushUpdates)
* in single head MST mode
*
*      subDeviceInstance
*          This parameter specifies the subdevice instance within the
*          LW04_DISPLAY_COMMON parent device to which the operation should be
*          directed. This parameter must specify a value between zero and the
*          total number of subdevices within the parent device.  This parameter
*          should be set to zero for default behavior
*      Head
*          Specifies the head index for the stream
*      sorIndex
*          Specifies the SOR index for the stream
*      streamIndex
*          Stream Identifier
*
*
* Possible status values returned are:
*      LW_OK
*      LW_ERR_ILWALID_ARGUMENT
*      LW_ERR_GENERIC: when this command has already been called
*
*/
#define LW0073_CTRL_CMD_DP_SET_TRIGGER_ALL                   (0x731370U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_SET_TRIGGER_ALL_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_SET_TRIGGER_ALL_PARAMS_MESSAGE_ID (0x70U)

typedef struct LW0073_CTRL_CMD_DP_SET_TRIGGER_ALL_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  head;
    LwBool enable;
} LW0073_CTRL_CMD_DP_SET_TRIGGER_ALL_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/* LW0073_CTRL_CMD_SPECIFIC_RETRIEVE_DP_RING_BUFFER
 *
 * These commands retrieves buffer from RM for
 * DP Library to dump logs
 *
 *
 * Possible status values returned include:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */

#define LW0073_CTRL_CMD_DP_RETRIEVE_DP_RING_BUFFER (0x731371U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_RETRIEVE_DP_RING_BUFFER_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_RETRIEVE_DP_RING_BUFFER_PARAMS_MESSAGE_ID (0x71U)

typedef struct LW0073_CTRL_CMD_DP_RETRIEVE_DP_RING_BUFFER_PARAMS {
    LW_DECLARE_ALIGNED(LwU8 *pDpRingBuffer, 8);
    LwU8  ringBufferType;
    LwU32 numRecords;
} LW0073_CTRL_CMD_DP_RETRIEVE_DP_RING_BUFFER_PARAMS;

/*
 * LW0073_CTRL_CMD_DP_QUERY_LT_STATS
 *
 * This command is used for querying link training related statistics
 *
 * subDeviceInstance
 *    client will give a subdevice to get right pGpu/pDisp for it
 * displayId
 *    DisplayId of the display for which the client needs the statistics
 * flag
 *    flag indicates request from client for fast/full/No LT
 * cmd
 *    cmd indicates request is for reset/query
 * dpLTSuccess
 *    Number of Full Link Trainings succeeded
 * dpLTFailure
 *    Number of Full Link Trainings failed
 * minTimeUs
 *    Min time required to complete LT in micro sec
 * maxTimeUs
 *    Max time required to complete LT in micro sec
 * avgTimeUs
 *    Avg time required to complete LT in micro sec
 *
 * Possible status values returned include:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW0073_CTRL_CMD_DP_LT_STATS (0x731372U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_LT_STATS_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_LT_STATS_PARAMS_MESSAGE_ID (0x72U)

typedef struct LW0073_CTRL_CMD_DP_LT_STATS_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 flag;
    LwU32 cmd;
    LwU32 dpLTSuccess;
    LwU32 dpLTFailure;
    LW_DECLARE_ALIGNED(LwU64 minTimeUs, 8);
    LW_DECLARE_ALIGNED(LwU64 maxTimeUs, 8);
    LW_DECLARE_ALIGNED(LwU64 avgTimeUs, 8);
} LW0073_CTRL_CMD_DP_LT_STATS_PARAMS;


#define FULL_LT_STATS                            0U
#define FAST_LT_STATS                            1U
#define NO_LT_STATS                              2U

#define RESET_LT_STATS                           0U
#define GET_LT_STATS                             1U

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
* LW0073_CTRL_CMD_DP_GET_AUXLOGGER_BUFFER_DATA
*
* This command collects the DP AUX log from the RM aux buffer and
* sends it to the application.
*
*      dpAuxBufferReadSize
*          Specifies the number of logs to be read from the
*          AUX buffer in RM
*      dpNumMessagesRead
*          Specifies the number of logs read from the AUX buffer
*      dpAuxBuffer
*          The local buffer to copy the specified number of logs
*          from RM to user application
*
*
* Possible status values returned are:
*      LW_OK
*      LW_ERR_ILWALID_ARGUMENT
*      LW_ERR_GENERIC: when this command has already been called
*
*
*DPAUXPACKET - This structure holds the log information
* auxPacket - carries the hex dump of the message transaction
* auxEvents - Contains the information as in what request and reply type where
* auxRequestTimeStamp - Request timestamp
* auxMessageReqSize - Request Message size
* auxMessageReplySize - Reply message size(how much information was actually send by receiver)
* auxOutPort - DP port number
* auxPortAddress - Address to which data was requested to be read or written
* auxReplyTimeStamp - Reply timestamp
* auxCount - Serial number to keep track of transactions
*/

/*Maximum dp messages size is 16 as per the protocol*/
#define DP_MAX_MSG_SIZE                          16U
#define MAX_LOGS_PER_POLL                        50U

/* Various kinds of DP Aux transactions */
#define LW_DP_AUXLOGGER_REQUEST_TYPE                     3:0
#define LW_DP_AUXLOGGER_REQUEST_TYPE_NULL        0x00000000U
#define LW_DP_AUXLOGGER_REQUEST_TYPE_I2CWR       0x00000001U
#define LW_DP_AUXLOGGER_REQUEST_TYPE_I2CREQWSTAT 0x00000002U
#define LW_DP_AUXLOGGER_REQUEST_TYPE_MOTWR       0x00000003U
#define LW_DP_AUXLOGGER_REQUEST_TYPE_MOTREQWSTAT 0x00000004U
#define LW_DP_AUXLOGGER_REQUEST_TYPE_AUXWR       0x00000005U
#define LW_DP_AUXLOGGER_REQUEST_TYPE_I2CRD       0x00000006U
#define LW_DP_AUXLOGGER_REQUEST_TYPE_MOTRD       0x00000007U
#define LW_DP_AUXLOGGER_REQUEST_TYPE_AUXRD       0x00000008U
#define LW_DP_AUXLOGGER_REQUEST_TYPE_UNKNOWN     0x00000009U

#define LW_DP_AUXLOGGER_REPLY_TYPE                       7:4
#define LW_DP_AUXLOGGER_REPLY_TYPE_NULL          0x00000000U
#define LW_DP_AUXLOGGER_REPLY_TYPE_SB_ACK        0x00000001U
#define LW_DP_AUXLOGGER_REPLY_TYPE_RETRY         0x00000002U
#define LW_DP_AUXLOGGER_REPLY_TYPE_TIMEOUT       0x00000003U
#define LW_DP_AUXLOGGER_REPLY_TYPE_DEFER         0x00000004U
#define LW_DP_AUXLOGGER_REPLY_TYPE_DEFER_TO      0x00000005U
#define LW_DP_AUXLOGGER_REPLY_TYPE_ACK           0x00000006U
#define LW_DP_AUXLOGGER_REPLY_TYPE_ERROR         0x00000007U
#define LW_DP_AUXLOGGER_REPLY_TYPE_UNKNOWN       0x00000008U

#define LW_DP_AUXLOGGER_EVENT_TYPE                       9:8
#define LW_DP_AUXLOGGER_EVENT_TYPE_AUX           0x00000000U
#define LW_DP_AUXLOGGER_EVENT_TYPE_HOT_PLUG      0x00000001U
#define LW_DP_AUXLOGGER_EVENT_TYPE_HOT_UNPLUG    0x00000002U
#define LW_DP_AUXLOGGER_EVENT_TYPE_IRQ           0x00000003U

#define LW_DP_AUXLOGGER_AUXCTL_CMD                       15:12
#define LW_DP_AUXLOGGER_AUXCTL_CMD_INIT          0x00000000U
#define LW_DP_AUXLOGGER_AUXCTL_CMD_I2CWR         0x00000000U
#define LW_DP_AUXLOGGER_AUXCTL_CMD_I2CRD         0x00000001U
#define LW_DP_AUXLOGGER_AUXCTL_CMD_I2CREQWSTAT   0x00000002U
#define LW_DP_AUXLOGGER_AUXCTL_CMD_MOTWR         0x00000004U
#define LW_DP_AUXLOGGER_AUXCTL_CMD_MOTRD         0x00000005U
#define LW_DP_AUXLOGGER_AUXCTL_CMD_MOTREQWSTAT   0x00000006U
#define LW_DP_AUXLOGGER_AUXCTL_CMD_AUXWR         0x00000008U
#define LW_DP_AUXLOGGER_AUXCTL_CMD_AUXRD         0x00000009U


typedef struct DPAUXPACKET {
    LwU32 auxEvents;
    LwU32 auxRequestTimeStamp;
    LwU32 auxMessageReqSize;
    LwU32 auxMessageReplySize;
    LwU32 auxOutPort;
    LwU32 auxPortAddress;
    LwU32 auxReplyTimeStamp;
    LwU32 auxCount;
    LwU8  auxPacket[DP_MAX_MSG_SIZE];
} DPAUXPACKET;
typedef struct DPAUXPACKET *PDPAUXPACKET;

#define LW0073_CTRL_CMD_DP_GET_AUXLOGGER_BUFFER_DATA (0x731373U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_GET_AUXLOGGER_BUFFER_DATA_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_GET_AUXLOGGER_BUFFER_DATA_PARAMS_MESSAGE_ID (0x73U)

typedef struct LW0073_CTRL_CMD_DP_GET_AUXLOGGER_BUFFER_DATA_PARAMS {
    //In
    LwU32       subDeviceInstance;
    LwU32       dpAuxBufferReadSize;

    //Out
    LwU32       dpNumMessagesRead;
    DPAUXPACKET dpAuxBuffer[MAX_LOGS_PER_POLL];
} LW0073_CTRL_CMD_DP_GET_AUXLOGGER_BUFFER_DATA_PARAMS;

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

// size 5 is sufficient for Channel Equalization. To be changed when moving to dynamic allocation
// Else the struct would have to be changed
#define MAX_PREEMP_SWING_TRIES 20

/*
* DPPreEmpSwingStats
*
*   This structure stores all the preemphasis current and the voltage swings
*   used during link training.
*
* dpPreEmpAndSwing0_1
*   Statistics for lanes 0 and 1
* dpPreEmpAndSwing2_3
*   Statistics for lanes 2 and 3
* dpNumPreEmpAndSwingTries
*   Number of tries with same or different preemphasis current and voltage
*   swings before successful link training
*/
typedef struct DPPreEmpSwingStats {
    LwU8 dpPreEmpAndSwing0_1[MAX_PREEMP_SWING_TRIES];
    LwU8 dpPreEmpAndSwing2_3[MAX_PREEMP_SWING_TRIES];
    LwU8 dpNumPreEmpAndSwingTries;
} DPPreEmpSwingStats;


/*
* DPLinkTrainingStats
*
*   This structure stores the statistics for the Clock Recovery phase
*   and for the Channel Equalization phase
*
* CRStats
*   Statistics for Clock Recovery phase
* CEStats
*   Statistics for Channel Equalization phase
*/

typedef struct DPLinkTrainingStats {
    DPPreEmpSwingStats CRStats;
    DPPreEmpSwingStats CEStats;
} DPLinkTrainingStats;


/*
* LW0073_CTRL_CMD_DP_LINK_STATISTICS
*
* This command collects the preemphasis and voltage swing statistics stored
* during the full link training phase
*
* The command takes a LW0073_CTRL_CMD_DP_LINK_STATISTICS_PARAMS structure as the
* argument with the appropriate subDeviceInstance, displayId as inputs.
* The arguments of this structure are described below.
*
* subDeviceInstance
*    This parameter specifies the subdevice instance within the
*    LW04_DISPLAY_COMMON parent device to which the operation should be
*    directed. This parameter must specify a value between zero and the
*    total number of subdevices within the parent device.  This parameter
*    should be set to zero for default behavior.
* displayId
*    This parameter specifies the ID of the display for which the dfp
*    caps should be returned.  The display ID must a dfp display.
*    If more than one displayId bit is set or the displayId is not a dfp,
*    this call will return LW_ERR_ILWALID_ARGUMENT.
* dpLTSuccess
*    Number of Full Link Trainings succeeded
* dpLTFailure
*    Number of Full Link Trainings failed
* minTimeUs
*    Min time required to complete LT in micro sec
* maxTimeUs
*    Max time required to complete LT in micro sec
* avgTimeUs
*    Avg time required to complete LT in micro sec
* fallbackConfig
*    Statistics corresponding to the last fallback config in case the
*    requested config fails
* reqConfig
*    Statistics corresponding to the requested link config
*
* Possible status values returned are:
*   LW_OK
*   LW_ERR_ILWALID_ARGUMENT
*
* NOTE: This control call is mainly for testing purposes and
*       should not be used in normal DP operations. Lwrrently
*       it stores the statistics for only full link training
*/
#define LW0073_CTRL_CMD_DP_LINK_STATISTICS (0x731375U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_LINK_STATISTICS_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_LINK_STATISTICS_PARAMS_MESSAGE_ID (0x75U)

typedef struct LW0073_CTRL_CMD_DP_LINK_STATISTICS_PARAMS {
    LwU32               subDeviceInstance;
    LwU32               displayId;
    LwU32               dpLTSuccess;
    LwU32               dpLTFailure;
    LW_DECLARE_ALIGNED(LwU64 dpLTAvgTimeUs, 8);
    LW_DECLARE_ALIGNED(LwU64 dpLTMinTimeUs, 8);
    LW_DECLARE_ALIGNED(LwU64 dpLTMaxTimeUs, 8);
    DPLinkTrainingStats fallbackConfig;
    DPLinkTrainingStats reqConfig;
} LW0073_CTRL_CMD_DP_LINK_STATISTICS_PARAMS;


/*
* LW0073_CTRL_CMD_DP_GET_SF_FLUSH_STATUS
*
* This command is used for querying the Sf flush status
* It also checks if both Sor and Sf are in the same flush mode.
*
* The command takes a LW0073_CTRL_CMD_DP_GET_SF_FLUSH_STATUS_PARAMS structure as
* the argument with the appropriate subDeviceInstance and heads as inputs.
* The arguments of this structure are described below.
*
* subDeviceInstance
*   This parameter specifies the subdevice instance within the
*   LW04_DISPLAY_COMMON parent device to which the operation should be
*   directed. This parameter must specify a value between zero and the
*   total number of subdevices within the parent device.  This parameter
*   should be set to zero for default behavior.
*
* headIndex
*   Specifies the head index for the stream
*
* bResult
*   Boolean value 1 indicating Enabled, 0 indicating Disabled
*
* Possible status values returned include:
*   LW_OK
*   LW_ERR_ILWALID_ARGUMENT
*/
#define LW0073_CTRL_CMD_DP_GET_SF_FLUSH_STATUS (0x731376U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_GET_SF_FLUSH_STATUS_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_GET_SF_FLUSH_STATUS_PARAMS_MESSAGE_ID (0x76U)

typedef struct LW0073_CTRL_CMD_DP_GET_SF_FLUSH_STATUS_PARAMS {
    // In
    LwU32  subDeviceInstance;
    LwU32  headIndex;

    // Out
    LwBool bResult;
} LW0073_CTRL_CMD_DP_GET_SF_FLUSH_STATUS_PARAMS;
/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)




/* LW0073_CTRL_CMD_DP_CONFIG_INDEXED_LINK_RATES
 *
 * This setup link rate table for target display to enable indexed link rate
 * and export valid link rates back to client. Client may pass empty table to
 * reset previous setting.
 *
 * subDeviceInstance
 *    client will give a subdevice to get right pGpu/pDisp for it
 * displayId
 *    DisplayId of the display for which the client targets
 * linkRateTbl
 *    Link rates in 200KHz as native granularity from eDP 1.4
 * linkBwTbl
 *    Link rates in 270MHz and valid for client to apply to
 * linkBwCount
 *    Total valid link rates
 *
 * Possible status values returned include:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0073_CTRL_CMD_DP_CONFIG_INDEXED_LINK_RATES (0x731377U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_CONFIG_INDEXED_LINK_RATES_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DP_MAX_INDEXED_LINK_RATES        8U

#define LW0073_CTRL_CMD_DP_CONFIG_INDEXED_LINK_RATES_PARAMS_MESSAGE_ID (0x77U)

typedef struct LW0073_CTRL_CMD_DP_CONFIG_INDEXED_LINK_RATES_PARAMS {
    // In
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU16 linkRateTbl[LW0073_CTRL_DP_MAX_INDEXED_LINK_RATES];

    // Out
    LwU8  linkBwTbl[LW0073_CTRL_DP_MAX_INDEXED_LINK_RATES];
    LwU8  linkBwCount;
} LW0073_CTRL_CMD_DP_CONFIG_INDEXED_LINK_RATES_PARAMS;


/*
 * LW0073_CTRL_CMD_DP_SET_STEREO_MSA_PROPERTIES
 *
 * This command is used to not depend on supervisor interrupts for setting the
 * stereo msa params. We will not cache the values and can toggle stereo using
 * this ctrl call on demand. Note that this control call will only change stereo
 * settings and will leave other settings as is.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId
 *     should be for DP only
 *   bEnableMSA
 *     To enable or disable MSA
 *   bStereoPhaseIlwerse
 *     To enable or disable Stereo Phase Ilwerse value
 *   featureMask
 *     Enable/Disable mask of individual MSA property.
 *   featureValues
 *     MSA property value to write
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_NOT_SUPPORTED
 *      LW_ERR_TIMEOUT
 *
 */
#define LW0073_CTRL_CMD_DP_SET_STEREO_MSA_PROPERTIES (0x731378U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_SET_STEREO_MSA_PROPERTIES_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_SET_STEREO_MSA_PROPERTIES_PARAMS_MESSAGE_ID (0x78U)

typedef struct LW0073_CTRL_CMD_DP_SET_STEREO_MSA_PROPERTIES_PARAMS {
    LwU32                                subDeviceInstance;
    LwU32                                displayId;
    LwBool                               bEnableMSA;
    LwBool                               bStereoPhaseIlwerse;
    LW0073_CTRL_DP_MSA_PROPERTIES_MASK   featureMask;
    LW0073_CTRL_DP_MSA_PROPERTIES_VALUES featureValues;
} LW0073_CTRL_CMD_DP_SET_STEREO_MSA_PROPERTIES_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW0073_CTRL_CMD_DP_SET_FORCE_IDLEPATTERN
 *
 * This command is used to enable/disable force idle pattern on specific DP OD.
 * This usually to blank the display for a short period of time, to avoid noise
 * and corruption.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *
 *   displayId
 *     Can only be 1 and must be DP.
 *
 *   bEnableForceIdlePattern
 *     To enable or disable FORCE_IDLEPATTERN
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_NOT_SUPPORTED
 *
 */
#define LW0073_CTRL_CMD_DP_SET_FORCE_IDLEPATTERN (0x731379U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_SET_FORCE_IDLEPATTERN_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_SET_FORCE_IDLEPATTERN_PARAMS_MESSAGE_ID (0x79U)

typedef struct LW0073_CTRL_CMD_DP_SET_FORCE_IDLEPATTERN_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  displayId;
    LwBool bEnableForceIdlePattern;
} LW0073_CTRL_CMD_DP_SET_FORCE_IDLEPATTERN_PARAMS;
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW0073_CTRL_CMD_DP_CONFIGURE_FEC
 *
 * This command is used to enable/disable FEC on DP Mainlink.
 * FEC is a prerequisite to DSC. This should be called only
 * after LT completes (including PostLT LQA) while enabling.
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *
 *   displayId
 *     Can only be 1 and must be DP.
 *
 *   bEnableFec
 *     To enable or disable FEC
 *
 * Possible status values returned are:
 *      LW_OK
 *      LW_ERR_ILWALID_ARGUMENT
 *      LW_ERR_NOT_SUPPORTED
 *
 */
#define LW0073_CTRL_CMD_DP_CONFIGURE_FEC (0x73137aU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_CONFIGURE_FEC_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_CONFIGURE_FEC_PARAMS_MESSAGE_ID (0x7AU)

typedef struct LW0073_CTRL_CMD_DP_CONFIGURE_FEC_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  displayId;
    LwBool bEnableFec;
} LW0073_CTRL_CMD_DP_CONFIGURE_FEC_PARAMS;

/*
 * LW0073_CTRL_CMD_DP_CONFIG_MACRO_PAD
 *
 *   subDeviceInstance
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior
 *   cmd
 *     This parameter is an input to this command.
 *     Here are the current defined fields:
 *       LW0073_CTRL_CMD_DP_CONFIG_MACRO_PAD_CMD_POWER
 *         Set to specify what operation to run.
 *           LW0073_CTRL_CMD_DP_CONFIG_MACRO_PAD_CMD_POWER_UP
 *             Request to power up pad.
 *           LW0073_CTRL_CMD_DP_CONFIG_MACRO_PAD_CMD_POWER_DOWN
 *             Request to power down the pad.
 *   linkBw
 *     This parameter is used to pass in the link bandwidth required to run the
 *     power up sequence. Refer enum DP_LINK_BANDWIDTH for valid values.
 *   laneCount
 *     This parameter is used to pass the lanecount.
 *   sorIndex
 *     This parameter is used to pass the SOR index.
 *   sublinkIndex
 *     This parameter is used to pass the sublink index. Please refer
 *     enum DFPLINKINDEX for valid values
 *   priPadLinkIndex
 *     This parameter is used to pass the padlink index for primary link.
 *     Please refer enum DFPPADLINK for valid index values for Link A~F.
 *   secPadLinkIndex
 *     This parameter is used to pass the padlink index for secondary link.
 *     For Single SST pass in LW0073_CTRL_CMD_DP_CONFIG_MACRO_PAD_PADLINK_INDEX_ILWALID
 *   bEnableSpread
 *     This parameter is boolean value used to indicate if spread is to be enabled or disabled.
 */

#define LW0073_CTRL_CMD_DP_CONFIG_MACRO_PAD (0x73137bU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_CONFIG_MACRO_PAD_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_CONFIG_MACRO_PAD_PARAMS_MESSAGE_ID (0x7BU)

typedef struct LW0073_CTRL_CMD_DP_CONFIG_MACRO_PAD_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  cmd;
    LwU32  linkBw;
    LwU32  laneCount;
    LwU32  sorIndex;
    LwU32  sublinkIndex;          // sublink A/B
    LwU32  priPadLinkIndex;       // padlink A/B/C/D/E/F
    LwU32  secPadLinkIndex;       // padlink A/B/C/D/E/F for secondary link in DualSST case.
    LwBool bEnableSpread;
} LW0073_CTRL_CMD_DP_CONFIG_MACRO_PAD_PARAMS;

#define LW0073_CTRL_CMD_DP_CONFIG_MACRO_PAD_CMD_POWER                        0:0
#define LW0073_CTRL_CMD_DP_CONFIG_MACRO_PAD_CMD_POWER_UP          (0x00000000U)
#define LW0073_CTRL_CMD_DP_CONFIG_MACRO_PAD_CMD_POWER_DOWN        (0x00000001U)

#define LW0073_CTRL_CMD_DP_CONFIG_MACRO_PAD_PADLINK_INDEX_ILWALID (0x000000FFU)

/*
 * LW0073_CTRL_CMD_DP_AUXCH_CTRL
 *
 * This command can be used to perform the I2C Bulk transfer over
 * DP Aux channel. This is the display port specific implementation
 * for sending bulk data over the DpAux channel, by splitting up the
 * data into pieces and retrying for pieces that aren't ACK'd.
 *
 *   subDeviceInstance [IN]
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId [IN]
 *     This parameter specifies the ID of the display for which the dfp
 *     caps should be returned.  The display ID must a dfp display.
 *     If more than one displayId bit is set or the displayId is not a dfp,
 *     this call will return LW_ERR_ILWALID_ARGUMENT.
 *   addr [IN]
 *     This parameter is an input to this command.  The addr parameter follows
 *     Section 2.4 in DisplayPort spec and the client should refer to the valid
 *     address in DisplayPort spec.  Only the first 20 bits are valid.
 *   bWrite [IN]
 *     This parameter specifies whether the command is a I2C write (LW_TRUE) or
 *     a I2C read (LW_FALSE).
 *   data [IN/OUT]
 *     In the case of a read transaction, this parameter returns the data from
 *     transaction request.  In the case of a write transaction, the client
 *     should write to this buffer for the data to send.
 *   size [IN/OUT]
 *     Specifies how many data bytes to read/write depending on the
 *     transaction type.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0073_CTRL_CMD_DP_AUXCH_I2C_TRANSFER_CTRL                (0x73137lw) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_DP_AUXCH_I2C_TRANSFER_CTRL_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DP_AUXCH_I2C_TRANSFER_MAX_DATA_SIZE           256U

#define LW0073_CTRL_DP_AUXCH_I2C_TRANSFER_CTRL_PARAMS_MESSAGE_ID (0x7LW)

typedef struct LW0073_CTRL_DP_AUXCH_I2C_TRANSFER_CTRL_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  displayId;
    LwU32  addr;
    LwBool bWrite;
    LwU8   data[LW0073_CTRL_DP_AUXCH_I2C_TRANSFER_MAX_DATA_SIZE];
    LwU32  size;
} LW0073_CTRL_DP_AUXCH_I2C_TRANSFER_CTRL_PARAMS;

/*
 * LW0073_CTRL_CMD_DP_ENABLE_VRR
 *
 * The command is used to enable VRR.
 *
 *   subDeviceInstance [IN]
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior
 *   displayId [IN]
 *     This parameter is an input to this command, specifies the ID of the display
 *     for client targeted to.
 *     The display ID must a DP display.
 *     If more than one displayId bit is set or the displayId is not a DP,
 *     this call will return LW_ERR_ILWALID_ARGUMENT.
 *   cmd [IN]
 *     This parameter is an input to this command.
 *
 *      _STAGE: specifies the stage id to execute in the VRR enablement sequence.
 *        _MONITOR_ENABLE_BEGIN:      Send command to the monitor to start monitor
 *                                    enablement procedure.
 *        _MONITOR_ENABLE_CHALLENGE:  Send challenge to the monitor
 *        _MONITOR_ENABLE_CHECK:      Read digest from the monitor, and verify
 *                                    if the result is valid.
 *        _DRIVER_ENABLE_BEGIN:       Send command to the monitor to start driver
 *                                    enablement procedure.
 *        _DRIVER_ENABLE_CHALLENGE:   Read challenge from the monitor and write back
 *                                    corresponding digest.
 *        _DRIVER_ENABLE_CHECK:       Check if monitor enablement worked.
 *        _RESET_MONITOR:             Set the FW state m/c to a known state.
 *        _INIT_PUBLIC_INFO:          Send command to the monitor to prepare public info.
 *        _GET_PUBLIC_INFO:           Read public info from the monitor.
 *        _STATUS_CHECK:              Check if monitor is ready for next command.
 *   result [OUT]
 *     This is an output parameter to reflect the result of the operation.
 */
#define LW0073_CTRL_CMD_DP_ENABLE_VRR (0x73137dU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_ENABLE_VRR_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_ENABLE_VRR_PARAMS_MESSAGE_ID (0x7DU)

typedef struct LW0073_CTRL_CMD_DP_ENABLE_VRR_PARAMS {
    LwU32 subDeviceInstance;
    LwU32 displayId;
    LwU32 cmd;
    LwU32 result;
} LW0073_CTRL_CMD_DP_ENABLE_VRR_PARAMS;

#define LW0073_CTRL_DP_CMD_ENABLE_VRR_CMD_STAGE                                   3:0
#define LW0073_CTRL_DP_CMD_ENABLE_VRR_CMD_STAGE_MONITOR_ENABLE_BEGIN     (0x00000000U)
#define LW0073_CTRL_DP_CMD_ENABLE_VRR_CMD_STAGE_MONITOR_ENABLE_CHALLENGE (0x00000001U)
#define LW0073_CTRL_DP_CMD_ENABLE_VRR_CMD_STAGE_MONITOR_ENABLE_CHECK     (0x00000002U)
#define LW0073_CTRL_DP_CMD_ENABLE_VRR_CMD_STAGE_DRIVER_ENABLE_BEGIN      (0x00000003U)
#define LW0073_CTRL_DP_CMD_ENABLE_VRR_CMD_STAGE_DRIVER_ENABLE_CHALLENGE  (0x00000004U)
#define LW0073_CTRL_DP_CMD_ENABLE_VRR_CMD_STAGE_DRIVER_ENABLE_CHECK      (0x00000005U)
#define LW0073_CTRL_DP_CMD_ENABLE_VRR_CMD_STAGE_RESET_MONITOR            (0x00000006U)
#define LW0073_CTRL_DP_CMD_ENABLE_VRR_CMD_STAGE_INIT_PUBLIC_INFO         (0x00000007U)
#define LW0073_CTRL_DP_CMD_ENABLE_VRR_CMD_STAGE_GET_PUBLIC_INFO          (0x00000008U)
#define LW0073_CTRL_DP_CMD_ENABLE_VRR_CMD_STAGE_STATUS_CHECK             (0x00000009U)

#define LW0073_CTRL_DP_CMD_ENABLE_VRR_STATUS_OK                          (0x00000000U)
#define LW0073_CTRL_DP_CMD_ENABLE_VRR_STATUS_PENDING                     (0x80000001U)
#define LW0073_CTRL_DP_CMD_ENABLE_VRR_STATUS_READ_ERROR                  (0x80000002U)
#define LW0073_CTRL_DP_CMD_ENABLE_VRR_STATUS_WRITE_ERROR                 (0x80000003U)
#define LW0073_CTRL_DP_CMD_ENABLE_VRR_STATUS_DEVICE_ERROR                (0x80000004U)

/*
 * LW0073_CTRL_CMD_DP_GET_GENERIC_INFOFRAME
 *
 * This command is used to capture the display output packets for DP protocol.
 * Common supported packets are Dynamic Range and mastering infoframe SDP for HDR,
 * VSC SDP for colorimetry and pixel encoding info.
 *
 *   displayID (in)
 *     This parameter specifies the displayID for the display resource to configure.
 *   subDeviceInstance (in)
 *     This parameter specifies the subdevice instance within the LW04_DISPLAY_COMMON
 *     parent device to which the operation should be directed.
 *   infoframeIndex (in)
 *     HW provides support to program 2 generic infoframes per frame for DP.
 *     This parameter indicates which infoframe packet is to be captured.
 *     Possible flags are as follows:
 *       LW0073_CTRL_DP_GET_GENERIC_INFOFRAME_CAPTURE_MODE
 *         This flag indicates the INFOFRAME that needs to be read.
 *         Set to _INFOFRAME0 if RM should read GENERIC_INFOFRAME
 *         Set to _INFOFRAME1 if RM should read GENERIC_INFOFRAME1
 *   packet (out)
 *     pPacket points to the memory for reading the infoframe packet.
 *   bTransmitControl (out)
 *     This gives the transmit mode of infoframes.
 *       If set, means infoframe will be sent as soon as possible and then on
 *       every frame during vblank.
 *       If cleared, means the infoframe will be sent once as soon as possible.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0073_CTRL_CMD_DP_GET_GENERIC_INFOFRAME                         (0x73137eU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_DP_GET_GENERIC_INFOFRAME_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DP_GENERIC_INFOFRAME_MAX_PACKET_SIZE                 36U

#define LW0073_CTRL_DP_GET_GENERIC_INFOFRAME_PARAMS_MESSAGE_ID (0x7EU)

typedef struct LW0073_CTRL_DP_GET_GENERIC_INFOFRAME_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  displayId;
    LwU32  infoframeIndex;
    LwU8   packet[LW0073_CTRL_DP_GENERIC_INFOFRAME_MAX_PACKET_SIZE];
    LwBool bTransmitControl;
} LW0073_CTRL_DP_GET_GENERIC_INFOFRAME_PARAMS;


#define LW0073_CTRL_DP_GET_GENERIC_INFOFRAME_CAPTURE_MODE                       0:0
#define LW0073_CTRL_DP_GET_GENERIC_INFOFRAME_CAPTURE_MODE_INFOFRAME0 (0x0000000U)
#define LW0073_CTRL_DP_GET_GENERIC_INFOFRAME_CAPTURE_MODE_INFOFRAME1 (0x0000001U)


/*
 * LW0073_CTRL_CMD_DP_GET_MSA_ATTRIBUTES
 *
 * This command is used to capture the various data attributes sent in the MSA for DP protocol.
 * Refer table 2-94 'MSA Data Fields' in DP1.4a spec document for MSA data field description.
 *
 *   displayID (in)
 *     This parameter specifies the displayID for the display resource to configure.
 *   subDeviceInstance (in)
 *     This parameter specifies the subdevice instance within the LW04_DISPLAY_COMMON
 *     parent device to which the operation should be directed.
 *   mvid, lwid (out)
 *     Video timestamp used by DP sink for regenerating pixel clock.
 *   misc0, misc1 (out)
 *     Miscellaneous MSA attributes.
 *   hTotal, vTotal (out)
 *     Htotal measured in pixel count and vtotal measured in line count.
 *   hActiveStart, vActiveStart (out)
 *     Active start measured from start of leading edge of the sync pulse.
 *   hActiveWidth, vActiveWidth (out)
 *     Active video width and height.
 *   hSyncWidth, vSyncWidth (out)
 *     Width of sync pulse.
 *   hSyncPolarity, vSyncPolarity (out)
 *     Polarity of sync pulse.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0073_CTRL_CMD_DP_GET_MSA_ATTRIBUTES                        (0x73137fU) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_DP_GET_MSA_ATTRIBUTES_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_DP_MSA_MAX_DATA_SIZE                             7U

#define LW0073_CTRL_DP_GET_MSA_ATTRIBUTES_PARAMS_MESSAGE_ID (0x7FU)

typedef struct LW0073_CTRL_DP_GET_MSA_ATTRIBUTES_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  displayId;
    LwU32  mvid;
    LwU32  lwid;
    LwU8   misc0;
    LwU8   misc1;
    LwU16  hTotal;
    LwU16  vTotal;
    LwU16  hActiveStart;
    LwU16  vActiveStart;
    LwU16  hActiveWidth;
    LwU16  vActiveWidth;
    LwU16  hSyncWidth;
    LwU16  vSyncWidth;
    LwBool hSyncPolarity;
    LwBool vSyncPolarity;
} LW0073_CTRL_DP_GET_MSA_ATTRIBUTES_PARAMS;

#define LW0073_CTRL_DP_MSA_ATTRIBUTES_MVID                              23:0
#define LW0073_CTRL_DP_MSA_ATTRIBUTES_LWID                              23:0
#define LW0073_CTRL_DP_MSA_ATTRIBUTES_MISC0                              7:0
#define LW0073_CTRL_DP_MSA_ATTRIBUTES_MISC1                             15:8
#define LW0073_CTRL_DP_MSA_ATTRIBUTES_HTOTAL                            15:0
#define LW0073_CTRL_DP_MSA_ATTRIBUTES_VTOTAL                           31:16
#define LW0073_CTRL_DP_MSA_ATTRIBUTES_HACTIVE_START                     15:0
#define LW0073_CTRL_DP_MSA_ATTRIBUTES_VACTIVE_START                    31:16
#define LW0073_CTRL_DP_MSA_ATTRIBUTES_HACTIVE_WIDTH                     15:0
#define LW0073_CTRL_DP_MSA_ATTRIBUTES_VACTIVE_WIDTH                    31:16
#define LW0073_CTRL_DP_MSA_ATTRIBUTES_HSYNC_WIDTH                       14:0
#define LW0073_CTRL_DP_MSA_ATTRIBUTES_HSYNC_POLARITY                   15:15
#define LW0073_CTRL_DP_MSA_ATTRIBUTES_VSYNC_WIDTH                      30:16
#define LW0073_CTRL_DP_MSA_ATTRIBUTES_VSYNC_POLARITY                   31:31

/*
 * LW0073_CTRL_CMD_DP_AUXCH_OD_CTRL
 *
 * This command is used to query OD capability and status as well as
 * control OD functionality of eDP LCD panels.
 *
 *   subDeviceInstance [in]
 *     This parameter specifies the subdevice instance within the
 *     LW04_DISPLAY_COMMON parent device to which the operation should be
 *     directed. This parameter must specify a value between zero and the
 *     total number of subdevices within the parent device.  This parameter
 *     should be set to zero for default behavior.
 *   displayId [in]
 *     This parameter specifies the ID of the DP display which owns
 *     the Main Link to be adjusted.  The display ID must a DP display
 *     as determined with the LW0073_CTRL_CMD_SPECIFIC_GET_TYPE command.
 *     If more than one displayId bit is set or the displayId is not a DP,
 *     this call will return LW_ERR_ILWALID_ARGUMENT.
 *   cmd [in]
 *     This parameter is an input to this command.  The cmd parameter tells
 *     whether we have to get the value of a specific field or set the
 *     value in case of a writeable field.
 *   control [in]
 *     This parameter is input by the user. It is used by the user to decide the control
 *     value to be written to change the Sink OD mode. The command to write is
 *     the LW0073_CTRL_CMD_DP_AUXCH_OD_CTL_SET command.
 *   bOdCapable [out]
 *     This parameter reflects the OD capability of the Sink which can be
 *     fetched by using the LW0073_CTRL_CMD_DP_AUXCH_OD_CAPABLE_QUERY command.
 *   bOdControlCapable [out]
 *     This parameter reflects the OD control capability of the Sink which can be
 *     fetched by using the LW0073_CTRL_CMD_DP_AUXCH_OD_CTL_CAPABLE_QUERY command.
 *   bOdStatus [out]
 *     This parameter reflects the Sink OD status which can be
 *     fetched by using the LW0073_CTRL_CMD_DP_AUXCH_OD_STATUS_QUERY command.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0073_CTRL_CMD_DP_AUXCH_OD_CTRL (0x731380U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_DP_INTERFACE_ID << 8) | LW0073_CTRL_CMD_DP_AUXCH_OD_CTRL_PARAMS_MESSAGE_ID" */

#define LW0073_CTRL_CMD_DP_AUXCH_OD_CTRL_PARAMS_MESSAGE_ID (0x80U)

typedef struct LW0073_CTRL_CMD_DP_AUXCH_OD_CTRL_PARAMS {
    LwU32  subDeviceInstance;
    LwU32  displayId;
    LwU8   control;
    LwU8   cmd;
    LwBool bOdCapable;
    LwBool bOdControlCapable;
    LwBool bOdStatus;
} LW0073_CTRL_CMD_DP_AUXCH_OD_CTRL_PARAMS;

/* _ctrl0073dp_h_ */

/* valid commands */
#define LW0073_CTRL_CMD_DP_AUXCHQUERY_OD_CAPABLE       0x00000000
#define LW0073_CTRL_CMD_DP_AUXCHQUERY_OD_CTL_CAPABLE   0x00000001
#define LW0073_CTRL_CMD_DP_AUXCHQUERY_OD_STATUS        0x00000002
#define LW0073_CTRL_CMD_DP_AUXCH_OD_CTL_SET            0x00000003

/* valid state values */
#define LW0073_CTRL_CMD_DP_AUXCH_OD_CTL_SET_AUTONOMOUS 0x00000000
#define LW0073_CTRL_CMD_DP_AUXCH_OD_CTL_SET_DISABLE_OD 0x00000002
#define LW0073_CTRL_CMD_DP_AUXCH_OD_CTL_SET_ENABLE_OD  0x00000003
