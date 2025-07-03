/*
 * SPDX-FileCopyrightText: Copyright (c) 2005-2015 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080i2c.finn
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

/* LW20_SUBDEVICE_XX i2c-related control commands and parameters */

/*
 * LW2080_CTRL_I2C_VERSION
 *   
 *   LW2080_CTRL_I2C_VERSION_0_0: 
 *       This return state specifies that support is only available 
 *       for single subAddr reads.
 *
 */
#define LW2080_CTRL_I2C_VERSION_0                         0x00

/* maximum number of i2c entries support */
#define LW2080_CTRL_I2C_MAX_ENTRIES                       256
#define LW2080_CTRL_I2C_MAX_REG_LEN                       8
#define LW2080_CTRL_I2C_MAX_ADDR_ENTRIES                  20

/*
 * LW2080_CTRL_I2C_FLAGS
 *
 *   LW2080_CTRL_I2C_FLAGS_NONSTD_SI1930UC: 
 *       This option specified that non-compliant i2c for SI1930UC is required
 *
 *   LW2080_CTRL_I2C_FLAGS_PRIVILEGE
 *       This option specified that the i2c access is privileged
 *
 *   LW2080_CTRL_I2C_FLAGS_PX3540
 *       This option specified that the i2c device -PX3540/3544- is accessed
 */
#define LW2080_CTRL_I2C_FLAGS_NONSTD_SI1930UC             (0x00000001)
#define LW2080_CTRL_I2C_FLAGS_PRIVILEGE                   (0x00000002)
#define LW2080_CTRL_I2C_FLAGS_DATA_ENCRYPTED              (0x00000004)
#define LW2080_CTRL_I2C_FLAGS_PX3540                      (0x00000010)
#define LW2080_CTRL_I2C_FLAGS_ADDR_AUTO_INC_NOT_SUPPORTED (0x00000008)

/*
 * LW2080_CTRL_CMD_I2C_READ_BUFFER
 *
 * This command allocates video memory for a particular subset of microcode.
 *
 *   version
 *       This field is returned to the client and indicates the current
 *       supported I2C controls available.  
 *
 *   port
 *       This field must be specified by the client to indicate which port/bus
 *       in which i2c access is desired.
 *
 *   flags
 *       This field is specified by the client to request additional options
 *       as provided by LW2080_CTRL_I2C_FLAGS.
 *
 *   inputCount
 *       This field specifies the total # of elements contained in inputBuffer
 *
 *   inputBuffer
 *       This should contain the chipaddr as the first element, followed by
 *       the each subAddress in which to access the first element of data
 *       Eg. <chipAddr> <subAddr1> ... <subAddrX> 
 *       In general, client will only have 2 elements <chipAddr> <subAddr>
 *
 *   outputCount
 *       This field specifies how many registers from the start register index.
 *       The maximum values allow are LW2080_CTRL_I2C_MAX_ENTRIES.
 *
 *   outputBuffer
 *       This buffer is returned to the client with the data read from
 *       the start register index.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_STATE_IN_USE
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_NOT_SUPPORTED
 *
 */
#define LW2080_CTRL_I2C_READ_BUFFER_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_I2C_READ_BUFFER_PARAMS {
    LwU32 version;
    LwU32 port;
    LwU32 flags;
    LwU32 inputCount;
    // C form: LwU8    inputBuffer[LW2080_CTRL_I2C_MAX_ENTRIES];
    LwU8  inputBuffer[LW2080_CTRL_I2C_MAX_ENTRIES];
    LwU32 outputCount;
    // C form: LwU8    outputBuffer[LW2080_CTRL_I2C_MAX_ENTRIES];
    LwU8  outputBuffer[LW2080_CTRL_I2C_MAX_ENTRIES];
} LW2080_CTRL_I2C_READ_BUFFER_PARAMS;

#define LW2080_CTRL_CMD_I2C_READ_BUFFER (0x20800601) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_I2C_INTERFACE_ID << 8) | LW2080_CTRL_I2C_READ_BUFFER_PARAMS_MESSAGE_ID" */

/*
 * LW2080_CTRL_CMD_I2C_WRITE_BUFFER
 *
 * This command allocates video memory for a particular subset of microcode.
 *
 *   version
 *       This field is returned to the client and indicates the current
 *       supported I2C controls available.  
 *
 *   port
 *       This field must be specified by the client to indicate which port/bus
 *       in which i2c access is desired.
 *
 *   flags
 *       This field is specified by the client to request additional options.
 *       LW2080_CTRL_I2C_FLAGS_NONSTD_SI1930UC: 
 *       - Specifies that non-compliant i2c access for SI1930UC is required
 *
 *   inputCount
 *       This field specifies the total # of elements contained in inputBuffer
 *
 *   inputBuffer
 *       This should contain the chipaddr as the first element, followed by
 *       the each subAddress in which to access the first element of data,
 *       and finally the data to be programmed.
 *       Eg. <chipAddr> <subAddr1> ... <subAddrX> <data1> ... <dataX>
 *       In general, client will have 2 elements + data to be programmed.
 *       <chipAddr> <subAddr> <data1> ... <dataX>
 *
 *   encrClientID
 *      This field is specified by client, which is used to uniquely access
 *      the client's encryption context
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_STATE_IN_USE
 *   LW_ERR_ILWALID_STATE
 *
 */

#define LW2080_CTRL_I2C_WRITE_BUFFER_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW2080_CTRL_I2C_WRITE_BUFFER_PARAMS {
    LwU32 version;
    LwU32 port;
    LwU32 flags;
    LwU32 inputCount;
    // C form: LwU8    inputBuffer[LW2080_CTRL_I2C_MAX_ENTRIES];
    LwU8  inputBuffer[LW2080_CTRL_I2C_MAX_ENTRIES];
    LwU32 encrClientID;
} LW2080_CTRL_I2C_WRITE_BUFFER_PARAMS;

#define LW2080_CTRL_CMD_I2C_WRITE_BUFFER (0x20800602) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_I2C_INTERFACE_ID << 8) | LW2080_CTRL_I2C_WRITE_BUFFER_PARAMS_MESSAGE_ID" */


/*
 * LW2080_CTRL_CMD_I2C_READ_REG
 *
 * This command allocates video memory for a particular subset of microcode.
 *
 *   version
 *       This field is returned to the client and indicates the current
 *       supported I2C controls available.  
 *
 *   port
 *       This field must be specified by the client to indicate which port/bus
 *       in which i2c access is desired.
 *
 *   flags
 *       This field is specified by the client to request additional options.
 *       LW2080_CTRL_I2C_FLAGS_NONSTD_SI1930UC: 
 *       - Specifies that non-compliant i2c access for SI1930UC is required
 *   addr
 *       This field is specified by the client to target address.
 *   reg
 *       This field is specified by the client to target register address.
 *
 *   bufsize
 *       This field specifies the total bytes # of register size
 *
 *   buffer
 *       when used for read, it used as buffer that store returned register content
 *       when used for write, It include data that will be written.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_STATE_IN_USE
 *   LW_ERR_ILWALID_STATE
 *
 */
typedef struct LW2080_CTRL_I2C_RW_REG_PARAMS {
    LwU32 version;
    LwU32 port;
    LwU32 flags;
    LwU32 addr;
    LwU8  reg;
    LwU8  bufsize;
    // C form: LwU8    buffer[LW2080_CTRL_I2C_MAX_ENTRIES - 1];
    LwU8  buffer[(LW2080_CTRL_I2C_MAX_ENTRIES - 1)];
} LW2080_CTRL_I2C_RW_REG_PARAMS;

// provide LW2080_CTRL_I2C_READ_REG_PARAMS as the historical name
typedef LW2080_CTRL_I2C_RW_REG_PARAMS LW2080_CTRL_I2C_READ_REG_PARAMS;
#define LW2080_CTRL_CMD_I2C_READ_REG  (0x20800603) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_I2C_INTERFACE_ID << 8) | 0x3" */

#define LW2080_CTRL_CMD_I2C_WRITE_REG (0x20800604) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_I2C_INTERFACE_ID << 8) | 0x4" */

/*
 * LW006F_CTRL_CMD_SYSTEM_I2C_ACCESS
 *
 * This command allows Clients to read and write data using the I2C ports
 * 
 *   token [IN]
 *      This used in i2cAcquirePort
 *
 *   cmd [IN]
 *      The I2CAccess command
 *     
 *   port [IN]
 *       The port ID of the concerned display
 *     
 *   flags [IN]
 *      The I2CAccess Flags such ack,start,stop
 *   
 *   data  [OUT/IN]
 *      Data that needs to be pass or read out
 *
 *   dataBuffSize [IN]
 *       Size of the data buffer.
 *   
 *   speed [IN]
 *      Speed of transaction.
 *
 *   status [OUT]
 *      The I2CAccess Status returned
 *
 *   encrClientID [IN]
 *      This field is specified by client, which is used to uniquely access
 *      the client's encryption context
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW2080_CTRL_CMD_I2C_ACCESS    (0x20800610) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_I2C_INTERFACE_ID << 8) | LW2080_CTRL_I2C_ACCESS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_I2C_ACCESS_PARAMS_MESSAGE_ID (0x10U)

typedef struct LW2080_CTRL_I2C_ACCESS_PARAMS {
    LwU32 token;
    LwU32 cmd;
    LwU32 port;
    LwU32 flags;
    LW_DECLARE_ALIGNED(LwP64 data, 8);
    LwU32 status;
    LwU32 dataBuffSize;
    LwU32 speed;
    LwU32 encrClientID;
} LW2080_CTRL_I2C_ACCESS_PARAMS;

// commands
#define LW2080_CTRL_I2C_ACCESS_CMD_ACQUIRE                   0x1
#define LW2080_CTRL_I2C_ACCESS_CMD_RELEASE                   0x2
#define LW2080_CTRL_I2C_ACCESS_CMD_WRITE_BYTE                0x3
#define LW2080_CTRL_I2C_ACCESS_CMD_READ_BYTE                 0x4
#define LW2080_CTRL_I2C_ACCESS_CMD_NULL                      0x5
#define LW2080_CTRL_I2C_ACCESS_CMD_RESET                     0x6
#define LW2080_CTRL_I2C_ACCESS_CMD_TEST_PORT                 0x11
#define LW2080_CTRL_I2C_ACCESS_CMD_SET_FAST_MODE             0x12
#define LW2080_CTRL_I2C_ACCESS_CMD_SET_NORMAL_MODE           0x13
#define LW2080_CTRL_I2C_ACCESS_CMD_WRITE_BUFFER              0x14
#define LW2080_CTRL_I2C_ACCESS_CMD_READ_BUFFER               0x15
#define LW2080_CTRL_I2C_ACCESS_CMD_START                     0x17
#define LW2080_CTRL_I2C_ACCESS_CMD_STOP                      0x18
#define LW2080_CTRL_I2C_ACCESS_CMD_SET_SLOW_MODE             0x20

// flags
#define LW2080_CTRL_I2C_ACCESS_FLAG_START                    0x1
#define LW2080_CTRL_I2C_ACCESS_FLAG_STOP                     0x2
#define LW2080_CTRL_I2C_ACCESS_FLAG_ACK                      0x4
#define LW2080_CTRL_I2C_ACCESS_FLAG_RAB                      0x8
#define LW2080_CTRL_I2C_ACCESS_FLAG_ADDR_10BITS              0x10
#define LW2080_CTRL_I2C_ACCESS_FLAG_PRIVILEGE                0x20
#define LW2080_CTRL_I2C_ACCESS_FLAG_DATA_ENCRYPTED           0x40
#define LW2080_CTRL_I2C_ACCESS_FLAG_RESTART                  0x80
#define LW2080_CTRL_I2C_ACCESS_FLAG_SLOW_MODE_33_33PCT       0x100
#define LW2080_CTRL_I2C_ACCESS_FLAG_SLOW_MODE_33PCT          0x200
#define LW2080_CTRL_I2C_ACCESS_FLAG_SLOW_MODE_10PCT          0x400
#define LW2080_CTRL_I2C_ACCESS_FLAG_SLOW_MODE_3_33PCT        0x800
#define LW2080_CTRL_I2C_ACCESS_FLAG_SLOW_MODE_3PCT           0x1000

// port
#define LW2080_CTRL_I2C_ACCESS_PORT_DYNAMIC                  0x0
#define LW2080_CTRL_I2C_ACCESS_PORT_PRIMARY                  0x1
#define LW2080_CTRL_I2C_ACCESS_PORT_SECONDARY                0x2
#define LW2080_CTRL_I2C_ACCESS_PORT_TERTIARY                 0x3
#define LW2080_CTRL_I2C_ACCESS_PORT_QUARTIARY                0x4

// Alternate numeric port designators
#define LW2080_CTRL_I2C_ACCESS_PORT_1                        0x1
#define LW2080_CTRL_I2C_ACCESS_PORT_2                        0x2
#define LW2080_CTRL_I2C_ACCESS_PORT_3                        0x3
#define LW2080_CTRL_I2C_ACCESS_PORT_4                        0x4
#define LW2080_CTRL_I2C_ACCESS_PORT_5                        0x5
#define LW2080_CTRL_I2C_ACCESS_PORT_6                        0x6
#define LW2080_CTRL_I2C_ACCESS_PORT_7                        0x7
#define LW2080_CTRL_I2C_ACCESS_PORT_8                        0x8
#define LW2080_CTRL_I2C_ACCESS_PORT_9                        0x9
#define LW2080_CTRL_I2C_ACCESS_PORT_10                       0x10

// Total ports count
#define LW2080_CTRL_I2C_ACCESS_NUM_PORTS                     LW2080_CTRL_I2C_ACCESS_PORT_10

// status
#define LW2080_CTRL_I2C_ACCESS_STATUS_SUCCESS                0x0
#define LW2080_CTRL_I2C_ACCESS_STATUS_ERROR                  0x1
#define LW2080_CTRL_I2C_ACCESS_STATUS_PROTOCOL_ERROR         0x2
#define LW2080_CTRL_I2C_ACCESS_STATUS_DEVICE_BUSY            0x3
#define LW2080_CTRL_I2C_ACCESS_STATUS_NACK_AFTER_SEND        0x4
#define LW2080_CTRL_I2C_ACCESS_STATUS_DP2TMDS_DONGLE_MISSING 0x5

#define LW2080_CTRL_CMD_I2C_ENABLE_MONITOR_3D_MODE           (0x20800620) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_I2C_INTERFACE_ID << 8) | LW2080_CTRL_I2C_ENABLE_MONITOR_3D_MODE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_I2C_ENABLE_MONITOR_3D_MODE_PARAMS_MESSAGE_ID (0x20U)

typedef struct LW2080_CTRL_I2C_ENABLE_MONITOR_3D_MODE_PARAMS {
    LwU32 head;
    LwU32 authType;
    LwU32 status;
} LW2080_CTRL_I2C_ENABLE_MONITOR_3D_MODE_PARAMS;

/* _ctrl2080i2c_h_ */


#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

