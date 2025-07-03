/*
 * SPDX-FileCopyrightText: Copyright (c) 2010-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0073/ctrl0073svp.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#include "ctrl/ctrl0073/ctrl0073base.h"

/* LW04_DISPLAY_COMMON 3DVP/SVP-specific control commands and parameters */

/*
 * LW0073_CTRL_SVP_ACCESS
 *
 * This enum defines R/W access modes to 3DVP HW entities.
 * These access modes are:
 *
 *  LW0073_CTRL_SVP_ACC_NONE
 *    No access
 *  LW0073_CTRL_SVP_ACC_READ
 *    Read-only access
 *  LW0073_CTRL_SVP_ACC_READWRITE
 *    Read+write access
 *  LW0073_CTRL_SVP_ACC_COUNT 
 *    Total number of access modes - not a valid access mode by itself
 */
typedef enum LW0073_CTRL_SVP_ACCESS {
    LW0073_CTRL_SVP_ACC_NONE = 0,
    LW0073_CTRL_SVP_ACC_READ = 1,
    LW0073_CTRL_SVP_ACC_READWRITE = 2,
    LW0073_CTRL_SVP_ACC_COUNT = 3,
} LW0073_CTRL_SVP_ACCESS;

/*
 * LW0073_CTRL_SVP_RF_ADDRESS
 *
 * This struct holds a 3DVP RF address which consists of five
 * opaque bytes (kind of a MAC address for RF communication).
 */
typedef struct LW0073_CTRL_SVP_RF_ADDRESS {
    LwU8 a0;
    LwU8 a1;
    LwU8 a2;
    LwU8 a3;
    LwU8 a4;
} LW0073_CTRL_SVP_RF_ADDRESS;

/*
 * LW0073_CTRL_SVP_SERIAL_NUMBER
 *
 * This struct holds a LWPU-specific serial number
 */
#define LW0073_CTRL_SVP_SERIAL_NUMBER_LENGTH 13U
typedef LwU8 LW0073_CTRL_SVP_SERIAL_NUMBER[LW0073_CTRL_SVP_SERIAL_NUMBER_LENGTH];

/*
 * LW0073_CTRL_SVP_TRANSCEIVER_INFO
 *
 * This struct holds imutable information of a 3DVP transceiver:
 *
 *   hwFeatures
 *     Opaque bitfield that encodes available HW features
 *   fwRevA
 *     Firmware revision of chip A (name obfuscated)
 *   fwRevB
 *     Firmware revision of chip B (name obfuscated)
 *   fwRevC
 *     Firmware revision of chip C (name obfuscated)
 *   rfAddress
 *     RF address used by this transceiver for RF communication
 *   channelCount
 *     Number of communication channels available on this transceiver
 *   serialNumber
 *     Serial number of the transceiver
 *   isEmbedded
 *     Indicates that this is an embedded emitter
 */
typedef struct LW0073_CTRL_SVP_TRANSCEIVER_INFO {
    LwU32                         hwFeatures;
    LwU32                         fwRevA;
    LwU32                         fwRevB;
    LwU32                         fwRevC;
    LW0073_CTRL_SVP_RF_ADDRESS    rfAddress;
    LwU32                         channelCount;
    LW0073_CTRL_SVP_SERIAL_NUMBER serialNumber;
    LwBool                        isEmbedded;
} LW0073_CTRL_SVP_TRANSCEIVER_INFO;

/*
 * LW0073_CTRL_SVP_CHANNEL_INFO
 *
 * This struct holds all info about a RF channel:
 *
 *   frequency
 *     Frequency [kHz] of this channel
 *   quality
 *     Quality [percent] of this channel
 */
typedef struct LW0073_CTRL_SVP_CHANNEL_INFO {
    LwU32 frequency;
    LwU32 quality;
} LW0073_CTRL_SVP_CHANNEL_INFO;

/*
 * LW0073_CTRL_SVP_TRANSCEIVER_MODE
 *
 * This enum defines power modes that may be used for RF communication:
 *
 *   LW0073_CTRL_SVP_TM_ILWALID
 *     Invalid mode
 *   LW0073_CTRL_SVP_TM_LOW_RANGE
 *     Low range power mode - fully bidirectional
 *   LW0073_CTRL_SVP_TM_MEDIUM_RANGE
 *     Medium range power mode - fully bidirectional
 *   LW0073_CTRL_SVP_TM_HIGH_RANGE
 *     Long range power mode - bidirectional up to a given range and
 *     unidirectional beyond it
 *   LW0073_CTRL_SVP_TM_COUNT
 *     Total number of transceiver power modes - not a valid mode by itself
 */
typedef enum LW0073_CTRL_SVP_TRANSCEIVER_MODE {
    LW0073_CTRL_SVP_TM_ILWALID = 0,
    LW0073_CTRL_SVP_TM_LOW_RANGE = 1,
    LW0073_CTRL_SVP_TM_MEDIUM_RANGE = 2,
    LW0073_CTRL_SVP_TM_HIGH_RANGE = 3,
    LW0073_CTRL_SVP_TM_COUNT = 4,
} LW0073_CTRL_SVP_TRANSCEIVER_MODE;

/*
 * LW0073_CTRL_SVP_TRANSCEIVER_STATE
 *
 * This struct holds the volatile state of a 3DVP transceiver:
 *
 *   button
 *     Button states [bitfield: button0, ..., button31]
 *   wheel
 *     Wheel position [relative clicks]
 *
 */
typedef struct LW0073_CTRL_SVP_TRANSCEIVER_STATE {
    LwU32 button;
    LwS32 wheel;
} LW0073_CTRL_SVP_TRANSCEIVER_STATE;

/*
 * LW0073_CTRL_SVP_PAIRING_MODE
 *
 * This enum defines special timeout values that may be used to
 * start/stop beacon-based pairing of RF glasses:
 *
 *   LW0073_CTRL_SVP_PM_STOP_PAIRING
 *     Stops any pairing
 *   LW0073_CTRL_SVP_PM_START_PAIRING_BEACON
 *     Starts pairing glasses in beacon mode
 */
typedef enum LW0073_CTRL_SVP_PAIRING_MODE {
    LW0073_CTRL_SVP_PM_STOP_PAIRING = 0,
    LW0073_CTRL_SVP_PM_START_PAIRING_BEACON = 2147483647,
} LW0073_CTRL_SVP_PAIRING_MODE;

/*
 * LW0073_CTRL_SVP_GLASSES_INFO
 *
 * This struct holds imutable information on a pair of 3DVP glasses:
 *
 *   hwFeatures
 *     Opaque bitfield of HW features
 *   fwRevA
 *     Firmware revision of chip A (name obfuscated)
 *   rfAddress
 *     RF address used by this pair of glasses for communication
 *   repairCount
 *     Number of times these glasses have been re-paired
 */
typedef struct LW0073_CTRL_SVP_GLASSES_INFO {
    LwU32                         hwFeatures;
    LwU32                         fwRevA;
    LW0073_CTRL_SVP_RF_ADDRESS    rfAddress;
    LW0073_CTRL_SVP_SERIAL_NUMBER serialNumber;
    LwU32                         repairCount;
} LW0073_CTRL_SVP_GLASSES_INFO;

/*
 * LW0073_CTRL_SVP_NAME
 *
 * Defines a zero-terminated Unicode string of up to 64 chars that may be used
 * to label 3DVP HW entities.
 */
#define LW0073_CTRL_SVP_NAME_SIZE 64U
typedef LwU16 LW0073_CTRL_SVP_NAME[LW0073_CTRL_SVP_NAME_SIZE];

/*
 * LW0073_CTRL_SVP_GLASSES_STATE
 *
 * This struct holds all volatile glasses data:
 *
 *   missedCycles
 *     Number of state sync cycles that glasses did not answer in a row
 *   battery
 *     Battery level [percent]
 *   batteryVoltage
 *     Battery level [mV?]
 *   batteryCharging
 *     Battery charge state [bool]
 *   compass
 *     Compass state [TBD]
 *   accel
 *     Accelera-o-meter state [TBD]
 */
typedef struct LW0073_CTRL_SVP_GLASSES_STATE {
    LwU32  missedCycles;
    LwU32  battery;
    LwU32  batteryVoltage;
    LwBool batteryCharging;
    LwU32  compass;
    LwU32  accel;
} LW0073_CTRL_SVP_GLASSES_STATE;

/*
 * LW0073_CTRL_SVP_EVENT_TYPE
 *
 * This enum defines event types that might be retrieved using
 * LW0073_CTRL_SVP_GET_EVENT_DATA_OPCODE:
 *
 *   LW0073_CTRL_SVP_ET_ILWALID
 *     No event
 *   LW0073_CTRL_SVP_ET_CONTEXT_DESTROYED
 *     Context has been destroyed
 *   LW0073_CTRL_SVP_ET_TRANSCEIVER_ENUM_DIRTY
 *     Transceiver enumeration is dirty (PnP event)
 *   LW0073_CTRL_SVP_ET_TRANSCEIVER_STALLED
 *     Transceiver stalled due to an internal error
 *   LW0073_CTRL_SVP_ET_AIRPLANE_MODE_TOGGLED
 *     Airplane mode was toggled for given transceiver
 *   LW0073_CTRL_SVP_ET_SIGNAL_QUALITY_CHANGED
 *     Signal quality changed for given transceiver
 *   LW0073_CTRL_SVP_ET_PAIRING_GLASSES_STARTED
 *     Transceiver started pairing glasses
 *   LW0073_CTRL_SVP_ET_PAIRING_GLASSES_COMPLETE
 *     Transceiver completed pairing glasses
 *   LW0073_CTRL_SVP_ET_DISCOVERING_GLASSES_STARTED
 *     Transceiver started discovering glasses
 *   LW0073_CTRL_SVP_ET_DISCOVERING_GLASSES_COMPLETE
 *     Transceiver completed discovering glasses
 *   LW0073_CTRL_SVP_ET_GLASSES_ENUM_DIRTY
 *     Glasses enumeration is dirty (PnP-like event)
 *   LW0073_CTRL_SVP_ET_GLASSES_NAME_CHANGED
 *     Glasses name changed
 *   LW0073_CTRL_SVP_ET_GLASSES_STATE_CHANGED
 *     Glasses state changed
 *   LW0073_CTRL_SVP_ET_COUNT
 *     Number of event types (not an event type itself)
 */
typedef enum LW0073_CTRL_SVP_EVENT_TYPE {
    LW0073_CTRL_SVP_ET_ILWALID = 0,
    LW0073_CTRL_SVP_ET_CONTEXT_DESTROYED = 1,
    LW0073_CTRL_SVP_ET_TRANSCEIVER_ENUM_DIRTY = 2,
    LW0073_CTRL_SVP_ET_TRANSCEIVER_STALLED = 3,
    LW0073_CTRL_SVP_ET_AIRPLANE_MODE_TOGGLED = 4,
    LW0073_CTRL_SVP_ET_SIGNAL_QUALITY_CHANGED = 5,
    LW0073_CTRL_SVP_ET_PAIRING_GLASSES_STARTED = 6,
    LW0073_CTRL_SVP_ET_PAIRING_GLASSES_COMPLETE = 7,
    LW0073_CTRL_SVP_ET_DISCOVERING_GLASSES_STARTED = 8,
    LW0073_CTRL_SVP_ET_DISCOVERING_GLASSES_COMPLETE = 9,
    LW0073_CTRL_SVP_ET_GLASSES_ENUM_DIRTY = 10,
    LW0073_CTRL_SVP_ET_GLASSES_NAME_CHANGED = 11,
    LW0073_CTRL_SVP_ET_GLASSES_STATE_CHANGED = 12,
    LW0073_CTRL_SVP_ET_COUNT = 13,
} LW0073_CTRL_SVP_EVENT_TYPE;

/*
 * LW0073_CTRL_SVP_CHANNELS
 *
 * This struct defines a sequence of four channels.
 */
typedef struct LW0073_CTRL_SVP_CHANNELS {
    LwU8 c0;
    LwU8 c1;
    LwU8 c2;
    LwU8 c3;
} LW0073_CTRL_SVP_CHANNELS;

/*
 * LW0073_CTRL_SVP_OPCODE
 *
 * This enum defines opcodes that identify 3DVP operations which can
 * be triggered by LW0073_CTRL_CMD_SVP. The parameters for each
 * operation are passed by operands, which are defined below the
 * LW0073_CTRL_SVP_OPCODE enum.
 */
typedef enum LW0073_CTRL_SVP_OPCODE {
/* context registry */
    LW0073_CTRL_SVP_REGISTER_CONTEXT_OPCODE = 0,
    LW0073_CTRL_SVP_UNREGISTER_CONTEXT_OPCODE = 1,
/* transceiver enumeration, access and utility fns */
    LW0073_CTRL_SVP_ENUM_TRANSCEIVER_OPCODE = 2,
    LW0073_CTRL_SVP_OPEN_TRANSCEIVER_OPCODE = 3,
    LW0073_CTRL_SVP_OPEN_TRANSCEIVER_PRIVILEGED_OPCODE = 4,
    LW0073_CTRL_SVP_CLOSE_TRANSCEIVER_OPCODE = 5,
    LW0073_CTRL_SVP_GET_TRANSCEIVER_ACCESS_OPCODE = 6,
    LW0073_CTRL_SVP_IDENTIFY_TRANSCEIVER_OPCODE_REMOVED = 7,
    LW0073_CTRL_SVP_RESET_TRANSCEIVER_OPCODE = 8,
/* transceiver info retrieval */
    LW0073_CTRL_SVP_GET_TRANSCEIVER_INFO_OPCODE = 9,
    LW0073_CTRL_SVP_GET_CHANNEL_INFO_OPCODE = 10,
/* transceiver configuration */
    LW0073_CTRL_SVP_GET_TRANSCEIVER_TIMING_SOURCE_OPCODE_REMOVED = 11,
    LW0073_CTRL_SVP_SET_TRANSCEIVER_TIMING_SOURCE_OPCODE_REMOVED = 12,
    LW0073_CTRL_SVP_GET_TRANSCEIVER_CHANNEL_OPCODE = 13,
    LW0073_CTRL_SVP_SET_TRANSCEIVER_CHANNEL_OPCODE = 14,
    LW0073_CTRL_SVP_GET_TRANSCEIVER_MODE_OPCODE = 15,
    LW0073_CTRL_SVP_SET_TRANSCEIVER_MODE_OPCODE = 16,
    LW0073_CTRL_SVP_GET_TRANSCEIVER_DATA_RATE_OPCODE_REMOVED = 17,
    LW0073_CTRL_SVP_SET_TRANSCEIVER_DATA_RATE_OPCODE_REMOVED = 18,
/* transceiver state retrieval */
    LW0073_CTRL_SVP_GET_TRANSCEIVER_STATE_OPCODE = 19,
/* glasses pairing, enumeration and utility fns */
    LW0073_CTRL_SVP_PAIR_GLASSES_OPCODE = 20,
    LW0073_CTRL_SVP_UNPAIR_GLASSES_OPCODE = 21,
    LW0073_CTRL_SVP_DISCOVER_GLASSES_OPCODE = 22,
    LW0073_CTRL_SVP_ENUM_GLASSES_OPCODE = 23,
    LW0073_CTRL_SVP_IDENTIFY_GLASSES_OPCODE = 24,
    LW0073_CTRL_SVP_RESET_GLASSES_OPCODE = 25,
/* glasses info retrieval */
    LW0073_CTRL_SVP_GET_GLASSES_INFO_OPCODE = 26,
/* glasses configuration */
    LW0073_CTRL_SVP_GET_GLASSES_SYNC_CYCLE_OPCODE = 27,
    LW0073_CTRL_SVP_SET_GLASSES_SYNC_CYCLE_OPCODE = 28,
    LW0073_CTRL_SVP_GET_GLASSES_NAME_OPCODE = 29,
    LW0073_CTRL_SVP_SET_GLASSES_NAME_OPCODE = 30,
/* glasses state retrieval */
    LW0073_CTRL_SVP_GET_GLASSES_STATE_OPCODE = 31,
/* events */
    LW0073_CTRL_SVP_REGISTER_EVENTS_OPCODE = 32,
    LW0073_CTRL_SVP_UNREGISTER_EVENTS_OPCODE = 33,
    LW0073_CTRL_SVP_GET_EVENT_DATA_OPCODE = 34,
/* timing override */
    LW0073_CTRL_SVP_SET_TIMING_OVERRIDE_OPCODE = 35,
/* transceiver configuration version 3 extensions */
    LW0073_CTRL_SVP_GET_TRANSCEIVER_SIGNAL_QUALITY_OPCODE = 36,
    LW0073_CTRL_SVP_GET_TRANSCEIVER_CHANNELS_OPCODE = 37,
    LW0073_CTRL_SVP_SET_TRANSCEIVER_CHANNELS_OPCODE = 38,
/* airplane mode version 4 extensions */
    LW0073_CTRL_SVP_GET_AIRPLANE_MODE_ENABLED_OPCODE = 39,
} LW0073_CTRL_SVP_OPCODE;

/*
 * LW0073_CTRL_SVP_REGISTER_CONTEXT_OPERANDS
 *
 * LW0073_CTRL_SVP_REGISTER_CONTEXT_OPCODE tells RM to register a
 * specific process for access to the 3D VISION PRO feature.
 * LW0073_CTRL_SVP_REGISTER_CONTEXT_OPERANDS defines the parameters
 * for this operation:
 *
 *   process
 *     This input parameter specifies an OS process for which a
 *     3DVP context should be registered.
 *
 *   context
 *     This output parameter holds a RM-specific handle for the
 *     context that was registered for given process.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_OUT_OF_MEMORY
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_REGISTER_CONTEXT_OPERANDS {
    LwU32 process;
    LwU32 context;
} LW0073_CTRL_SVP_REGISTER_CONTEXT_OPERANDS;

/*
 * LW0073_CTRL_SVP_UNREGISTER_CONTEXT_OPERANDS
 *
 * LW0073_CTRL_SVP_UNREGISTER_CONTEXT_OPCODE tells RM to unregister
 * a context from 3DVP access.
 * LW0073_CTRL_SVP_UNREGISTER_CONTEXT_OPERANDS defines the parameters
 * for this operation:
 *
 *   context
 *     This input parameter specifies the context that is to be
 *     unregistereed from 3DVP access.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_UNREGISTER_CONTEXT_OPERANDS {
    LwU32 context;
} LW0073_CTRL_SVP_UNREGISTER_CONTEXT_OPERANDS;

/*
 * LW0073_CTRL_SVP_ENUM_TRANSCEIVER_OPERANDS
 *
 * LW0073_CTRL_SVP_ENUM_TRANSCEIVER_OPCODE causes RM to enumerate
 * 3DVP transceivers.
 * LW0073_CTRL_SVP_ENUM_TRANSCEIVER_OPERANDS defines the parameters
 * for this operation:
 *
 *   context
 *     This parameter specifies whether the enumeration is to be
 *     restarted (context == 0) or to be continued (context == last
 *     enumeration result), and is used to return the handle of the
 *     next transceiver in the enumeration or 0 if there are no more
 *     transceivers to enumerate.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_ENUM_TRANSCEIVER_OPERANDS {
    LwU32 context;
    LwU16 transceiver;
} LW0073_CTRL_SVP_ENUM_TRANSCEIVER_OPERANDS;

/*
 * LW0073_CTRL_SVP_OPEN_TRANSCEIVER_OPERANDS
 *
 * LW0073_CTRL_SVP_OPEN_TRANSCEIVER_OPCODE causes RM to open a
 * transceiver for a given context with a specific access mode.
 * LW0073_CTRL_SVP_OPEN_TRANSCEIVER_PRIVILEGED_OPCODE implements
 * the same functionality with the enhancement that callers that
 * use the privileged opcode will rule out callers that used the
 * non-privileged version with conflicting access rights.
 * LW0073_CTRL_SVP_OPEN_TRANSCEIVER_OPERANDS defines the parameters
 * for these operations:
 *
 *   context
 *     Specifies the context for which to open the transceiver.
 *
 *   transceiver
 *     Holds the handle of the transceiver that is to be opened.
 *   
 *   access
 *     Specifies the access mode in which to open given transceiver.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ILWALID_ARGUMENT
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_OPEN_TRANSCEIVER_OPERANDS {
    LwU32                  context;
    LwU16                  transceiver;
    LW0073_CTRL_SVP_ACCESS access;
} LW0073_CTRL_SVP_OPEN_TRANSCEIVER_OPERANDS;

/*
 * LW0073_CTRL_SVP_CLOSE_TRANSCEIVER_OPERANDS
 *
 * LW0073_CTRL_SVP_CLOSE_TRANSCEIVER_OPCODE closes a specific transceiver
 * for a given context.
 * LW0073_CTRL_SVP_CLOSE_TRANSCEIVER_OPERANDS defines the parameters for
 * this operation:
 *
 *   context
 *     Defines the context for which to close the transceiver.
 *
 *   transceiver
 *     Specifies the transceiver that is to be closed.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_CLOSE_TRANSCEIVER_OPERANDS {
    LwU32 context;
    LwU16 transceiver;
} LW0073_CTRL_SVP_CLOSE_TRANSCEIVER_OPERANDS;

/*
 * LW0073_CTRL_SVP_GET_TRANSCEIVER_ACCESS_OPERANDS
 *
 * LW0073_CTRL_SVP_GET_TRANSCEIVER_ACCESS_OPCODE queries the access
 * rights by which a given context opened a specific transceiver.
 * LW0073_CTRL_SVP_GET_TRANSCEIVER_ACCESS_OPERANDS defines the parameters
 * for this operation:
 *
 *   context
 *     Defines the context for which to get the access rights for given
 *     transceiver.
 *
 *   transceiver
 *     Defines the transceiver for which to get the access rights.
 *   
 *   access
 *     Returns the access rights of given context to given transceiver.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_GET_TRANSCEIVER_ACCESS_OPERANDS {
    LwU32                  context;
    LwU16                  transceiver;
    LW0073_CTRL_SVP_ACCESS access;
} LW0073_CTRL_SVP_GET_TRANSCEIVER_ACCESS_OPERANDS;

/*
 * LW0073_CTRL_SVP_RESET_TRANSCEIVER_OPERANDS
 *
 * LW0073_CTRL_SVP_RESET_TRANSCEIVER_OPCODE causes a specific transceiver
 * to be reset to factory settings.
 * LW0073_CTRL_SVP_RESET_TRANSCEIVER_OPERANDS defines the parameters
 * for this operation:
 *
 *   context
 *     Specifies a context with write-access to given transceiver.
 *
 *   transceiver
 *     Defines the transceiver that is to be reset to factory settings.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_RESET_TRANSCEIVER_OPERANDS {
    LwU32 context;
    LwU16 transceiver;
} LW0073_CTRL_SVP_RESET_TRANSCEIVER_OPERANDS;

/*
 * LW0073_CTRL_SVP_GET_TRANSCEIVER_INFO_OPERANDS
 *
 * LW0073_CTRL_SVP_GET_TRANSCEIVER_INFO_OPCODE queries information
 * about a specific transceiver.
 * LW0073_CTRL_SVP_GET_TRANSCEIVER_INFO_OPERANDS defines the parameters
 * for this operation:
 *
 *   context
 *     Specifies a context with read-access to given transceiver.
 *
 *   transceiver
 *     Defines the transceiver of which to get information.
 *
 *   transceiverInfo
 *     Returns information about given transceiver.
 *   
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_GET_TRANSCEIVER_INFO_OPERANDS {
    LwU32                            context;
    LwU16                            transceiver;
    LW0073_CTRL_SVP_TRANSCEIVER_INFO info;
} LW0073_CTRL_SVP_GET_TRANSCEIVER_INFO_OPERANDS;

/*
 * LW0073_CTRL_SVP_GET_CHANNEL_INFO_OPERANDS
 *
 * LW0073_CTRL_SVP_GET_CHANNEL_INFO_OPCODE queries information about
 * a specific channel.
 * LW0073_CTRL_SVP_GET_CHANNEL_INFO_OPERANDS defines the parameters
 * for this operation:
 *
 *   context
 *     Specifies a context with read-access to given transceiver.
 *
 *   transceiver
 *     Defines the transceiver for which to get the channel information.
 *
 *   channel
 *     Defines the channel of which to get the information.
 *
 *   channelInfo
 *     Returns information about given channel.
 *   
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ILWALID_CHANNEL
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_GET_CHANNEL_INFO_OPERANDS {
    LwU32                        context;
    LwU16                        transceiver;
    LwU32                        channelIndex;
    LW0073_CTRL_SVP_CHANNEL_INFO info;
} LW0073_CTRL_SVP_GET_CHANNEL_INFO_OPERANDS;


/*
 * LW0073_CTRL_SVP_GET_TRANSCEIVER_CHANNEL_OPERANDS
 *
 * LW0073_CTRL_SVP_GET_TRANSCEIVER_CHANNEL_OPCODE queries the communication
 * channel that is used by a specific transceiver.
 * LW0073_CTRL_SVP_GET_TRANSCEIVER_CHANNEL_OPERANDS defines the parameters for
 * this operation:
 *
 *   context
 *     Specifies a context with read-access to given transceiver.
 *
 *   transceiver
 *     Defines the transceiver of which to get the communication
 *     channel.
 *
 *   channelIndex
 *     Returns the communication channel that is lwrrently used by
 *     given transceiver.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ILWALID_CHANNEL
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_GET_TRANSCEIVER_CHANNEL_OPERANDS {
    LwU32 context;
    LwU16 transceiver;
    LwU32 channelIndex;
} LW0073_CTRL_SVP_GET_TRANSCEIVER_CHANNEL_OPERANDS;

/*
 * LW0073_CTRL_SVP_SET_TRANSCEIVER_CHANNEL_OPERANDS
 *
 * LW0073_CTRL_SVP_SET_TRANSCEIVER_CHANNEL_OPCODE defines the communication
 * channel that is to be used by a specific transceiver.
 * LW0073_CTRL_SVP_SET_TRANSCEIVER_CHANNEL_OPERANDS defines the parameters
 * for this operation:
 *
 *   context
 *     Specifies a context with write-access to given transceiver.
 *
 *   transceiver
 *     Defines the transceiver for which to set the communication
 *     channel.
 *
 *   channel
 *     Defines the communication channel that is to be used by given
 *     transceiver.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ILWALID_CHANNEL
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_SET_TRANSCEIVER_CHANNEL_OPERANDS {
    LwU32 context;
    LwU16 transceiver;
    LwU32 channelIndex;
} LW0073_CTRL_SVP_SET_TRANSCEIVER_CHANNEL_OPERANDS;

/*
 * LW0073_CTRL_SVP_GET_TRANSCEIVER_MODE_OPERANDS
 *
 * LW0073_CTRL_SVP_GET_TRANSCEIVER_MODE_OPCODE queries the power mode that
 * is used by a specific transceiver.
 * LW0073_CTRL_SVP_GET_TRANSCEIVER_MODE_OPERANDS defines the parameters for
 * this operation:
 *
 *   context
 *     Specifies a context with read-access to given transceiver.
 *
 *   transceiver
 *     Defines the transceiver of which to get the power mode.
 *
 *   mode
 *     Returns the power mode that is lwrrently used by the given transceiver.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_GET_TRANSCEIVER_MODE_OPERANDS {
    LwU32                            context;
    LwU16                            transceiver;
    LW0073_CTRL_SVP_TRANSCEIVER_MODE mode;
} LW0073_CTRL_SVP_GET_TRANSCEIVER_MODE_OPERANDS;

/*
 * LW0073_CTRL_SVP_SET_TRANSCEIVER_MODE_OPERANDS
 *
 * LW0073_CTRL_SVP_SET_TRANSCEIVER_MODE_OPCODE defines the power mode
 * that is to be used by a specific transceiver.
 * LW0073_CTRL_SVP_SET_TRANSCEIVER_MODE_OPERANDS defines the parameters
 * for this operation:
 *
 *   context
 *     Specifies a context with write-access to given transceiver.
 *
 *   transceiver
 *     Defines the transceiver for which to set the current mode.
 *
 *   mode
 *     Defines the power mode that is to be used by given transceiver.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ILWALID_ARGUMENT
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_SET_TRANSCEIVER_MODE_OPERANDS {
    LwU32                            context;
    LwU16                            transceiver;
    LW0073_CTRL_SVP_TRANSCEIVER_MODE mode;
} LW0073_CTRL_SVP_SET_TRANSCEIVER_MODE_OPERANDS;

/*
 * LW0073_CTRL_SVP_GET_TRANSCEIVER_STATE_OPERANDS
 *
 * LW0073_CTRL_SVP_GET_TRANSCEIVER_STATE_OPCODE queries the current state
 * of a specific transceiver.
 * LW0073_CTRL_SVP_GET_TRANSCEIVER_STATE_OPERANDS defines the parameters
 * for this operation:
 *
 *   context
 *     Specifies a context with read-access to given transceiver.
 *
 *   transceiver
 *     Defines the transceiver of which to get the current state.
 *
 *   state
 *     Returns the current state of given transceiver.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_GET_TRANSCEIVER_STATE_OPERANDS {
    LwU32                             context;
    LwU16                             transceiver;
    LW0073_CTRL_SVP_TRANSCEIVER_STATE state;
} LW0073_CTRL_SVP_GET_TRANSCEIVER_STATE_OPERANDS;

/*
 * LW0073_CTRL_SVP_PAIR_GLASSES_OPERANDS
 *
 * LW0073_CTRL_SVP_PAIR_GLASSES_OPCODE causes a specific transceiver to
 * try to pair glasses for a given amount of time.
 * LW0073_CTRL_SVP_PAIR_GLASSES_OPERANDS defines the parameters for this
 * operation:
 *
 *   context
 *     Specifies a context with write-access to given transceiver.
 *
 *   transceiver
 *     Defines the transceiver on which to pair glasses.
 *
 *   timeOut
 *     Defines the pairing timeout in milliseconds or holds special values
 *     to control beacon-based pairing (see LW0073_CTRL_SVP_PAIRING_MODE).
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ILWALID_ARGUMENT
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_PAIR_GLASSES_OPERANDS {
    LwU32 context;
    LwU16 transceiver;
    LwU32 timeOut;
} LW0073_CTRL_SVP_PAIR_GLASSES_OPERANDS;

/*
 * LW0073_CTRL_SVP_UNPAIR_GLASSES_OPERANDS
 *
 * LW0073_CTRL_SVP_UNPAIR_GLASSES_OPCODE causes glasses to be unpaired
 * from given transceiver.
 * LW0073_CTRL_SVP_UNPAIR_GLASSES_OPERANDS defines the parameters for this
 * operation:
 *
 *   context
 *     Specifies a context with write-access to given transceiver.
 *
 *   transceiver
 *     Defines the transceiver from which to unpair given glasses.
 *
 *   glasses
 *     Defines the pair of glasses that is to be unpaired from given
 *     transceiver. Set this parameter to null to unpair all glasses
 *     from given transceiver.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ILWALID_GLASSES
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_UNPAIR_GLASSES_OPERANDS {
    LwU32 context;
    LwU16 transceiver;
    LwU16 glasses;
} LW0073_CTRL_SVP_UNPAIR_GLASSES_OPERANDS;

/*
 * LW0073_CTRL_SVP_DISCOVER_GLASSES_OPERANDS
 *
 * LW0073_CTRL_SVP_DISCOVER_GLASSES_OPCODE causes a transceiver to discover
 * glasses that have been paired anonymously by pairing beacon mode.
 * LW0073_CTRL_SVP_DISCOVER_GLASSES_OPERANDS defines the parameters for this
 * operation:
 *
 *   context
 *     Specifies a context with write-access to given transceiver.
 *
 *   transceiver
 *     Defines the transceiver at which to discover glasses.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_DISCOVER_GLASSES_OPERANDS {
    LwU32 context;
    LwU16 transceiver;
} LW0073_CTRL_SVP_DISCOVER_GLASSES_OPERANDS;

/*
 * LW0073_CTRL_SVP_ENUM_GLASSES_OPERANDS
 *
 * LW0073_CTRL_SVP_ENUM_GLASSES_OPCODE enumerates glasses that are paired
 * to a specific transceiver.
 * LW0073_CTRL_SVP_ENUM_GLASSES_OPERANDS defines the parameters for this
 * operation:
 *
 *   context
 *     Specifies a context with read-access to given transceiver.
 *
 *   transceiver
 *     Defines the transceiver on which to enumerate glasses.
 *
 *   glasses
 *     Defines the pair of glasses on which to continue the enumeration,
 *     and returns the enumeration result. Set this parameter to null to
 *     start a new enumeration.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ILWALID_GLASSES
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_ENUM_GLASSES_OPERANDS {
    LwU32 context;
    LwU16 transceiver;
    LwU16 glasses;
} LW0073_CTRL_SVP_ENUM_GLASSES_OPERANDS;

/*
 * LW0073_CTRL_SVP_IDENTIFY_GLASSES_OPERANDS
 *
 * LW0073_CTRL_SVP_IDENTIFY_GLASSES_OPCODE causes a specific pair of glasses
 * to identify themselves by flashing LEDs.
 * LW0073_CTRL_SVP_IDENTIFY_GLASSES_OPERANDS defines the parameters for this
 * operation:
 *
 *   context
 *     Specifies a context with write-access to the transceiver on
 *     which given glasses have been enumerated.
 *
 *   glasses
 *     Defines the pair of glasses to identify.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ILWALID_GLASSES
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_IDENTIFY_GLASSES_OPERANDS {
    LwU32 context;
    LwU16 transceiver;
    LwU16 glasses;
} LW0073_CTRL_SVP_IDENTIFY_GLASSES_OPERANDS;

/*
 * LW0073_CTRL_SVP_RESET_GLASSES_OPERANDS
 *
 * LW0073_CTRL_SVP_RESET_GLASSES_OPCODE causes a specific pair of glasses
 * to be reset to factory settings.
 * LW0073_CTRL_SVP_RESET_GLASSES_OPERANDS defines the parameters of this
 * operation:
 *
 *   context
 *     Specifies a context with write-access to the transceiver on
 *     which given glasses have been enumerated.
 *
 *   glasses
 *     Defines the pair of glasses that is to be reset to factory settings.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ILWALID_GLASSES
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_RESET_GLASSES_OPERANDS {
    LwU32 context;
    LwU16 transceiver;
    LwU16 glasses;
} LW0073_CTRL_SVP_RESET_GLASSES_OPERANDS;

/*
 * LW0073_CTRL_SVP_GET_GLASSES_INFO_OPERANDS
 *
 * LW0073_CTRL_SVP_GET_GLASSES_INFO_OPCODE queries information about a
 * specific pair of glasses.
 * LW0073_CTRL_SVP_GET_GLASSES_INFO_OPERANDS defines the parameters for
 * this operation:
 *
 *   context
 *     Specifies a context with read-access to the transceiver on
 *     which given glasses have been enumerated.
 *
 *   glasses
 *     Defines the pair of glasses of which to get information.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ILWALID_GLASSES
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_GET_GLASSES_INFO_OPERANDS {
    LwU32                        context;
    LwU16                        transceiver;
    LwU16                        glasses;
    LW0073_CTRL_SVP_GLASSES_INFO info;
} LW0073_CTRL_SVP_GET_GLASSES_INFO_OPERANDS;

/*
 * LW0073_CTRL_SVP_GET_GLASSES_SYNC_CYCLE_OPERANDS
 *
 * LW0073_CTRL_SVP_GET_GLASSES_SYNC_CYCLE_OPCODE queries the sync cycle that
 * is used to update the state of a specific pair of glasses.
 * LW0073_CTRL_SVP_GET_GLASSES_SYNC_CYCLE_OPERANDS defines the parameters
 * of this operation:
 *
 *   context
 *     Specifies a context with read-access to the transceiver on
 *     which given glasses have been enumerated.
 *
 *   glasses
 *     Defines the pair of glasses of which to get the current sync cycle.
 *
 *   syncCycle
 *     Returns the sync cycle in milliseconds that is lwrrently used to
 *     update given glasses.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ILWALID_GLASSES
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_GET_GLASSES_SYNC_CYCLE_OPERANDS {
    LwU32 context;
    LwU16 transceiver;
    LwU16 glasses;
    LwU32 syncCycle;
} LW0073_CTRL_SVP_GET_GLASSES_SYNC_CYCLE_OPERANDS;

/*
 * LW0073_CTRL_SVP_SET_GLASSES_SYNC_CYCLE_OPERANDS
 *
 * LW0073_CTRL_SVP_SET_GLASSES_SYNC_CYCLE_OPCODE defines the sync cycle that
 * is to be used to update the state of a specific pair of glasses.
 * LW0073_CTRL_SVP_SET_GLASSES_SYNC_CYCLE_OPERANDS defines the parameters of
 * this operation:
 *
 *   context
 *     Specifies a context with write-access to the transceiver on
 *     which given glasses have been enumerated.
 *
 *   glasses
 *     Defines the pair of glasses for which to set the current sync cycle.
 *
 *   syncCycle
 *     Defines the sync cycle in milliseconds that is to be used to update
 *     given glasses.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ILWALID_GLASSES
 *   LW0073_CTRL_SVP_ILWALID_ARGUMENT
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_SET_GLASSES_SYNC_CYCLE_OPERANDS {
    LwU32 context;
    LwU16 transceiver;
    LwU16 glasses;
    LwU32 syncCycle;
} LW0073_CTRL_SVP_SET_GLASSES_SYNC_CYCLE_OPERANDS;

/*
 * LW0073_CTRL_SVP_GET_GLASSES_NAME_OPERANDS
 *
 * LW0073_CTRL_SVP_GET_GLASSES_NAME_OPCODE queries the name of a specific
 * pair of glasses.
 * LW0073_CTRL_SVP_GET_GLASSES_NAME_OPERANDS defines the parameters of
 * this operation:
 *
 *   context
 *     Specifies a context with read-access to the transceiver on
 *     which given glasses have been enumerated.
 *
 *   glasses
 *     Defines the pair of glasses of which to get the name.
 *
 *   name
 *     Returns the name that is lwrrently assigned to given glasses.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ILWALID_GLASSES
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_GET_GLASSES_NAME_OPERANDS {
    LwU32                context;
    LwU16                transceiver;
    LwU16                glasses;
    LW0073_CTRL_SVP_NAME name;
} LW0073_CTRL_SVP_GET_GLASSES_NAME_OPERANDS;

/*
 * LW0073_CTRL_SVP_SET_GLASSES_NAME_OPERANDS
 *
 * LW0073_CTRL_SVP_SET_GLASSES_NAME_OPCODE defines the name that is to
 * be used by a specific pair of glasses.
 * LW0073_CTRL_SVP_SET_GLASSES_NAME_OPERANDS defines the parameters of
 * this operation:
 *
 *   context
 *     Specifies a context with write-access to the transceiver on
 *     which given glasses have been enumerated.
 *
 *   glasses
 *     Defines the pair of glasses for which to set the name.
 *
 *   name
 *     Defines the name that is to be assigned to given glasses.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ILWALID_GLASSES
 *   LW0073_CTRL_SVP_ILWALID_ARGUMENT
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_SET_GLASSES_NAME_OPERANDS {
    LwU32                context;
    LwU16                transceiver;
    LwU16                glasses;
    LW0073_CTRL_SVP_NAME name;
} LW0073_CTRL_SVP_SET_GLASSES_NAME_OPERANDS;

/*
 * LW0073_CTRL_SVP_GET_GLASSES_STATE_OPERANDS
 *
 * LW0073_CTRL_SVP_GET_GLASSES_STATE_OPCODE queries the current state
 * of a specific pair of glasses.
 * LW0073_CTRL_SVP_GET_GLASSES_STATE_OPERANDS defines the parameters
 * of this operation:
 *
 *   context
 *     Specifies a context with read-access to the transceiver on
 *     which given glasses have been enumerated.
 *
 *   glasses
 *     Defines the pair of glasses of which to get the current state.
 *
 *   state
 *     Returns the current state of given glasses.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ILWALID_GLASSES
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_GET_GLASSES_STATE_OPERANDS {
    LwU32                         context;
    LwU16                         transceiver;
    LwU16                         glasses;
    LW0073_CTRL_SVP_GLASSES_STATE state;
} LW0073_CTRL_SVP_GET_GLASSES_STATE_OPERANDS;

/*
 * LW0073_CTRL_SVP_REGISTER_EVENTS_OPERANDS
 *
 * LW0073_CTRL_SVP_REGISTER_EVENTS_OPCODE registers a given context for
 * retrieving events from RM and from the USB driver. This is done by
 * handing over a user-mode semaphore that is to be referenced by RM
 * and the USB driver, and which gets released by those whenever an
 * event oclwrs. A user-mode function that waits on the release of this
 * semaphore then should use LW0073_CTRL_SVP_GET_EVENT_DATA to retrieve
 * data that describes the nature of this event. Using this two-step
 * approach enables the USB driver to directly signal events bypassing
 * RM and should overcome the complexity of registering events per
 * SVP entity.
 * LW0073_CTRL_SVP_REGISTER_EVENTS_OPERANDS defines the parameters
 * of this operation:
 *
 *   context
 *     Specifies a context for which to enable event signalling.
 *
 *   semaphore
 *     Semaphore identifier.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_REGISTER_EVENTS_OPERANDS {
    LwU32 context;
    LW_DECLARE_ALIGNED(LwU64 semaphore, 8);
} LW0073_CTRL_SVP_REGISTER_EVENTS_OPERANDS;

/*
 * LW0073_CTRL_SVP_UNREGISTER_EVENTS_OPERANDS
 *
 * LW0073_CTRL_SVP_UNREGISTER_EVENTS_OPCODE unregisters a given context
 * from retrieving events by RM and the USB driver. This operation should
 * effectively release all RM and USB driver references to the semaphore
 * used to signal events.
 * LW0073_CTRL_SVP_UNREGISTER_EVENTS_OPERANDS defines the parameters
 * of this operation:
 *
 *   context
 *     Specifies a context for which to disable event signalling.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_UNREGISTER_EVENTS_OPERANDS {
    LwU32 context;
} LW0073_CTRL_SVP_UNREGISTER_EVENTS_OPERANDS;

/*
 * LW0073_CTRL_SVP_GET_EVENT_DATA_OPERANDS
 *
 * LW0073_CTRL_SVP_GET_EVENT_DATA_OPCODE retrieves data that describes
 * a event signalled by RM or the USB driver.
 * LW0073_CTRL_SVP_GET_EVENT_DATA_OPERANDS defines the parameters
 * of this operation:
 *
 *   context
 *     Specifies a context for which to retrieve event data.
 *
 *   eventType
 *     Returns the type of an event recently signalled by RM or the USB driver.
 *   transceiver
 *     Returns the transceiver which caused the event (if any).
 *   glasses
 *     Returns the glasses which caused the event (if any).
 *   timeStamp
 *     Returns event time stamp [ms] with no specific base (just provides
 *     a value to measure event intervals).
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_GET_EVENT_DATA_OPERANDS {
    LwU32 context;
    LwU32 eventType;
    LwU16 transceiver;
    LwU16 glasses;
    LwU32 timeStamp;
} LW0073_CTRL_SVP_GET_EVENT_DATA_OPERANDS;

/*
 * LW0073_CTRL_SVP_SET_TIMING_OVERRIDE_OPERANDS
 *
 * LW0073_CTRL_SVP_SET_TIMING_OVERRIDE_OPCODE allows to override
 * timings passed to the transceiver during activation.
 * LW0073_CTRL_SVP_SET_TIMING_OVERRIDE_OPERANDS defines the
 * parameters of this operation:
 *
 *   context
 *     Specifies a context - optional and may be zero (note: all
 *     non-zero values need to identify a valid context).
 *
 *   transceiver
 *     Specifies the transceiver to set the override for - use zero for all.
 *
 *   refreshRate
 *     Specifies the refresh rate [mHz] that is to be used by the transceiver.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_SET_TIMING_OVERRIDE_OPERANDS {
    LwU32 context;
    LwU16 transceiver;
    LwU32 refreshRate;
} LW0073_CTRL_SVP_SET_TIMING_OVERRIDE_OPERANDS;

/*
 * LW0073_CTRL_SVP_GET_TRANSCEIVER_SIGNAL_QUALITY_OPERANDS
 *
 * LW0073_CTRL_SVP_GET_TRANSCEIVER_SIGNAL_QUALITY_OPCODE queries the current
 * transceiver x glasses signal quality.
 * LW0073_CTRL_SVP_GET_TRANSCEIVER_SIGNAL_QUALITY_OPERANDS defines the
 * parameters for this operation:
 *
 *   context
 *     Specifies a context with read-access to given transceiver.
 *
 *   transceiver
 *     Defines the transceiver of which to get the signal quality.
 *
 *   quality
 *     Returns the signal quality in percent.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_GET_TRANSCEIVER_SIGNAL_QUALITY_OPERANDS {
    LwU32 context;
    LwU16 transceiver;
    LwU8  quality;
} LW0073_CTRL_SVP_GET_TRANSCEIVER_SIGNAL_QUALITY_OPERANDS;


/*
 * LW0073_CTRL_SVP_GET_TRANSCEIVER_CHANNELS_OPERANDS
 *
 * LW0073_CTRL_SVP_GET_TRANSCEIVER_CHANNELS_OPCODE queries the communication
 * channel sequence that is used by a specific transceiver.
 * LW0073_CTRL_SVP_GET_TRANSCEIVER_CHANNEL_OPERANDS defines the parameters for
 * this operation:
 *
 *   context
 *     Specifies a context with read-access to given transceiver.
 *
 *   transceiver
 *     Defines the transceiver of which to get the communication
 *     channel sequence.
 *
 *   channels
 *     Returns the communication channel sequence that is lwrrently used by
 *     given transceiver.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_GET_TRANSCEIVER_CHANNELS_OPERANDS {
    LwU32                    context;
    LwU16                    transceiver;
    LW0073_CTRL_SVP_CHANNELS channels;
} LW0073_CTRL_SVP_GET_TRANSCEIVER_CHANNELS_OPERANDS;

/*
 * LW0073_CTRL_SVP_SET_TRANSCEIVER_CHANNELS_OPERANDS
 *
 * LW0073_CTRL_SVP_SET_TRANSCEIVER_CHANNELS_OPCODE defines the communication
 * channel sequence that is to be used by a specific transceiver.
 * LW0073_CTRL_SVP_SET_TRANSCEIVER_CHANNELS_OPERANDS defines the parameters
 * for this operation:
 *
 *   context
 *     Specifies a context with write-access to given transceiver.
 *
 *   transceiver
 *     Defines the transceiver for which to set the communication
 *     channel sequence.
 *
 *   channels
 *     Defines the communication channel sequence that is to be used by given
 *     transceiver.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ILWALID_CHANNEL
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_SET_TRANSCEIVER_CHANNELS_OPERANDS {
    LwU32                    context;
    LwU16                    transceiver;
    LW0073_CTRL_SVP_CHANNELS channels;
} LW0073_CTRL_SVP_SET_TRANSCEIVER_CHANNELS_OPERANDS;

/*
 * LW0073_CTRL_SVP_GET_AIRPLANE_MODE_ENABLED_OPERANDS
 *
 * LW0073_CTRL_SVP_GET_AIRPLANE_MODE_ENABLED_OPCODE retrieves whether
 * the transceiver is in airplane mode (= RF I/O disabled).
 * LW0073_CTRL_SVP_GET_AIRPLANE_MODE_ENABLED_OPERANDS defines the parameters
 * for this operation:
 *
 *   context
 *     Specifies a context with read-access to given transceiver.
 *
 *   transceiver
 *     Defines the transceiver for which to query the airplane mode state.
 *
 *   enabled
 *     Returns whether the transceiver airplane mode is enabled.
 *
 * Possible result values returned in LW0073_CTRL_SVP_PARAMS.result are:
 *   LW0073_CTRL_SVP_OK
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *   LW0073_CTRL_SVP_ILWALID_CHANNEL
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *   LW0073_CTRL_SVP_ERROR
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_GENERIC
 *   TBD
 */
typedef struct LW0073_CTRL_SVP_GET_AIRPLANE_MODE_ENABLED_OPERANDS {
    LwU32 context;
    LwU16 transceiver;
    LwU32 enabled;
} LW0073_CTRL_SVP_GET_AIRPLANE_MODE_ENABLED_OPERANDS;

/*
 * LW0073_CTRL_SVP_OPERANDS
 *
 * This union encapsulates the operands of any operation listed above.
 */


/*
 * LW0073_CTRL_SVP_RESULT
 *
 * This enum defines 3DVP-related ctrl cmd return values:
 *
 *   LW0073_CTRL_SVP_OK
 *     Operation succeeded
 *   LW0073_CTRL_SVP_ERROR
 *     Operation failed due to an undolwmented error
 *   LW0073_CTRL_SVP_OUT_OF_MEMORY
 *     Operation failed due to a lack of memory
 *   LW0073_CTRL_SVP_NOT_SUPPORTED
 *     Operation is not supported
 *   LW0073_CTRL_SVP_ILWALID_OPCODE
 *     Invalid or unknown operation opcode
 *   LW0073_CTRL_SVP_ILWALID_OPERAND
 *     Invalid or unknown operation operand
 *   LW0073_CTRL_SVP_ILWALID_CONTEXT
 *     Operand specifies an invalid context
 *   LW0073_CTRL_SVP_ILWALID_TRANSCEIVER
 *     Operand specifies an invalid transceiver
 *   LW0073_CTRL_SVP_ILWALID_CHANNEL
 *     Operand specifies an invalid channel
 *   LW0073_CTRL_SVP_ILWALID_GLASSES
 *     Operand specifies an invalid pair of glasses
 *   LW0073_CTRL_SVP_ILWALID_ARGUMENT
 *     Operand specifies misc invalid arguments
 *   LW0073_CTRL_SVP_END_ENUMERATION
 *     Operation couldn't enumerate any mode entities
 *   LW0073_CTRL_SVP_ACCESS_DENIED
 *     Operation failed due to insufficient access rights
 *   LW0073_CTRL_SVP_DEVICE_BUSY
 *     Operation failed because device is busy
 */
typedef enum LW0073_CTRL_SVP_RESULT {
    LW0073_CTRL_SVP_OK = 0,
    LW0073_CTRL_SVP_ERROR = 1,
    LW0073_CTRL_SVP_OUT_OF_MEMORY = 2,
    LW0073_CTRL_SVP_NOT_SUPPORTED = 3,
    LW0073_CTRL_SVP_INCOMPATIBLE_PARAMS = 4,
    LW0073_CTRL_SVP_INCOMPATIBLE_USB_DRIVER = 5,
    LW0073_CTRL_SVP_ILWALID_OPCODE = 6,
    LW0073_CTRL_SVP_ILWALID_OPERAND = 7,
    LW0073_CTRL_SVP_ILWALID_CONTEXT = 8,
    LW0073_CTRL_SVP_ILWALID_TRANSCEIVER = 9,
    LW0073_CTRL_SVP_ILWALID_CHANNEL = 10,
    LW0073_CTRL_SVP_ILWALID_GLASSES = 11,
    LW0073_CTRL_SVP_ILWALID_ARGUMENT = 12,
    LW0073_CTRL_SVP_END_ENUMERATION = 13,
    LW0073_CTRL_SVP_ACCESS_DENIED = 14,
    LW0073_CTRL_SVP_DEVICE_BUSY = 15,
} LW0073_CTRL_SVP_RESULT;

/*
 * LW0073_CTRL_CMD_SVP
 *
 * The one and only SVP ctrl cmd.
 */
#define LW0073_CTRL_CMD_SVP (0x731401U) /* finn: Evaluated from "(FINN_LW04_DISPLAY_COMMON_SVP_INTERFACE_ID << 8) | LW0073_CTRL_SVP_PARAMS_MESSAGE_ID" */

/*
 * LW0073_CTRL_SVP_PARAMS
 *
 * This struct defines the params for the SVP ctrl cmd. Specifically,
 * LW0073_CTRL_SVP_PARAMS is a container class that defines the
 * actual SVP function and its parameters by an opcode and operands.
 * It serves as input to the stereo_dongle's svp function which may
 * forward it to the USB driver's SVP interface function.
 *
 *   version
 *     Version of the LW0073_CTRL_SVP_PARAMS struct. This version is
 *     defined by LW0073_CTRL_SVP_PARAMS_VERSION during compile-time
 *     and is checked by the 3DVP USB driver during run-time to detect
 *     whether it's binary-compatible to the display driver and can
 *     execute the operations defined by the LW0073_CTRL_SVP_PARAMS
 *     struct. The USB driver will just report INCOMPATIBLE_USB_DRIVER
 *     if it detects that it's not compatible with the display driver.
 *     RM itself won't operate opcodes/operands of PARAMS with a
 *     version that is different to LW0073_CTRL_SVP_VERSION but will
 *     just return INCOMPATIBLE_PARAMS, so that all RM clients using
 *     this CTRL should come from the same CL (as LwAPI does, which
 *     is actually the only supposed LW0073_CTRL_SVP_PARAMS client).
 *   result
 *     Returns the operation result - needs to be located in front
 *     of opcode/operands to have the same offset across all/future
 *     LW0073_CTRL_SVP_PARAMS versions (operands union might grow
 *     in size).
 *   opcode
 *     Defines the 3DVP operation to execute
 *   operands
 *     Defines the parameters for given operation
 */
#define LW0073_CTRL_SVP_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0073_CTRL_SVP_PARAMS {
    LwU32                  version;
    LW0073_CTRL_SVP_RESULT result;
    LW0073_CTRL_SVP_OPCODE opcode;
    union {
        LW0073_CTRL_SVP_REGISTER_CONTEXT_OPERANDS               registerContext;

        LW0073_CTRL_SVP_UNREGISTER_CONTEXT_OPERANDS             unregisterContext;

        LW0073_CTRL_SVP_ENUM_TRANSCEIVER_OPERANDS               enumTransceiver;

        LW0073_CTRL_SVP_OPEN_TRANSCEIVER_OPERANDS               openTransceiver;

        LW0073_CTRL_SVP_OPEN_TRANSCEIVER_OPERANDS               openTransceiverPrivileged;

        LW0073_CTRL_SVP_CLOSE_TRANSCEIVER_OPERANDS              closeTransceiver;

        LW0073_CTRL_SVP_GET_TRANSCEIVER_ACCESS_OPERANDS         getTransceiverAccess;

        LW0073_CTRL_SVP_RESET_TRANSCEIVER_OPERANDS              resetTransceiver;

        LW0073_CTRL_SVP_GET_TRANSCEIVER_INFO_OPERANDS           getTransceiverInfo;

        LW0073_CTRL_SVP_GET_CHANNEL_INFO_OPERANDS               getChannelInfo;

        LW0073_CTRL_SVP_GET_TRANSCEIVER_CHANNEL_OPERANDS        getTransceiverChannel;

        LW0073_CTRL_SVP_SET_TRANSCEIVER_CHANNEL_OPERANDS        setTransceiverChannel;

        LW0073_CTRL_SVP_GET_TRANSCEIVER_MODE_OPERANDS           getTransceiverMode;

        LW0073_CTRL_SVP_SET_TRANSCEIVER_MODE_OPERANDS           setTransceiverMode;

        LW0073_CTRL_SVP_GET_TRANSCEIVER_STATE_OPERANDS          getTransceiverState;

        LW0073_CTRL_SVP_PAIR_GLASSES_OPERANDS                   pairGlasses;

        LW0073_CTRL_SVP_UNPAIR_GLASSES_OPERANDS                 unpairGlasses;

        LW0073_CTRL_SVP_DISCOVER_GLASSES_OPERANDS               discoverGlasses;

        LW0073_CTRL_SVP_ENUM_GLASSES_OPERANDS                   enumGlasses;

        LW0073_CTRL_SVP_IDENTIFY_GLASSES_OPERANDS               identifyGlasses;

        LW0073_CTRL_SVP_RESET_GLASSES_OPERANDS                  resetGlasses;

        LW0073_CTRL_SVP_GET_GLASSES_INFO_OPERANDS               getGlassesInfo;

        LW0073_CTRL_SVP_GET_GLASSES_SYNC_CYCLE_OPERANDS         getGlassesSyncCycle;

        LW0073_CTRL_SVP_SET_GLASSES_SYNC_CYCLE_OPERANDS         setGlassesSyncCycle;

        LW0073_CTRL_SVP_GET_GLASSES_NAME_OPERANDS               getGlassesName;

        LW0073_CTRL_SVP_SET_GLASSES_NAME_OPERANDS               setGlassesName;

        LW0073_CTRL_SVP_GET_GLASSES_STATE_OPERANDS              getGlassesState;

        LW_DECLARE_ALIGNED(LW0073_CTRL_SVP_REGISTER_EVENTS_OPERANDS registerEvents, 8);

        LW0073_CTRL_SVP_UNREGISTER_EVENTS_OPERANDS              unregisterEvents;

        LW0073_CTRL_SVP_GET_EVENT_DATA_OPERANDS                 getEventData;

        LW0073_CTRL_SVP_SET_TIMING_OVERRIDE_OPERANDS            setTimingOverride;

        LW0073_CTRL_SVP_GET_TRANSCEIVER_SIGNAL_QUALITY_OPERANDS getTransceiverSignalQuality;

        LW0073_CTRL_SVP_GET_TRANSCEIVER_CHANNELS_OPERANDS       getTransceiverChannels;

        LW0073_CTRL_SVP_SET_TRANSCEIVER_CHANNELS_OPERANDS       setTransceiverChannels;

        LW0073_CTRL_SVP_GET_AIRPLANE_MODE_ENABLED_OPERANDS      getAirplaneModeEnabled;
    } operands;
} LW0073_CTRL_SVP_PARAMS;

/*
 * LW0073_CTRL_SVP_PARAMS_VERSION
 *
 * To be incremented for each LW0073_CTRL_SVP_PARAMS change.
 *
 * Version history:
 *
 * 0x00000001 - initial version
 * 0x00000002 - added LW0073_CTRL_SVP_SET_TIMING_OVERRIDE_*
 * 0x00000003 - removed (marked opcodes as _REMOVED and deleted operands)
 *                LW0073_CTRL_SVP_IDENTIFY_TRANSCEIVER_OPCODE
 *                LW0073_CTRL_SVP_GET_TRANSCEIVER_TIMING_SOURCE_OPCODE
 *                LW0073_CTRL_SVP_SET_TRANSCEIVER_TIMING_SOURCE_OPCODE
 *                LW0073_CTRL_SVP_GET_TRANSCEIVER_DATA_RATE_OPCODE
 *                LW0073_CTRL_SVP_SET_TRANSCEIVER_DATA_RATE_OPCODE
 *            - added
 *                LW0073_CTRL_SVP_GET_TRANSCEIVER_SIGNAL_QUALITY_OPCODE
 *                LW0073_CTRL_SVP_GET_TRANSCEIVER_CHANNELS_OPCODE
 *                LW0073_CTRL_SVP_SET_TRANSCEIVER_CHANNELS_OPCODE
 * 0x00000004 - added
 *                LW0073_CTRL_SVP_TRANSCEIVER_INFO.serialNumber
 *                LW0073_CTRL_SVP_TRANSCEIVER_INFO.isEmbedded
 *                LW0073_CTRL_SVP_GLASSES_INFO.serialNumber
 *                LW0073_CTRL_SVP_GLASSES_INFO.repairCount
 *                LW0073_CTRL_SVP_GET_AIRPLANE_MODE_ENABLED
 *                LW0073_CTRL_SVP_ET_CONTEXT_DESTROYED
 *                LW0073_CTRL_SVP_ET_AIRPLANE_MODE_TOGGLED
 *                LW0073_CTRL_SVP_ET_SIGNAL_QUALITY_CHANGED
 *                LW0073_CTRL_SVP_ET_PAIRING_GLASSES_STARTED
 *                LW0073_CTRL_SVP_ET_PAIRING_GLASSES_COMPLETE
 *                LW0073_CTRL_SVP_ET_DISCOVERING_GLASSES_STARTED
 *                LW0073_CTRL_SVP_ET_DISCOVERING_GLASSES_COMPLETE
 *                LW0073_CTRL_SVP_ET_GLASSES_NAME_CHANGED
 *                LW0073_CTRL_SVP_ET_GLASSES_STATE_CHANGED
 *                LW0073_CTRL_SVP_GET_EVENT_DATA_OPERANDS.glasses
 *                LW0073_CTRL_SVP_GET_EVENT_DATA_OPERANDS.timeStamp
 *
 */

#define LW0073_CTRL_SVP_PARAMS_VERSION 0x00000004U

/* _ctrl0073svp_h_ */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

