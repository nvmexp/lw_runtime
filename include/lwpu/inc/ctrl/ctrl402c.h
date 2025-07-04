/*
 * SPDX-FileCopyrightText: Copyright (c) 2010-2020 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl402c.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
/* LW40_I2C control commands and parameters */
#define LW402C_CTRL_CMD(cat,idx) \
                                LWXXXX_CTRL_CMD(0x402C, LW402C_CTRL_##cat, idx)

/* I2C command categories (6 bits) */
#define LW402C_CTRL_RESERVED                              (0x00)
#define LW402C_CTRL_I2C                                   (0x01)


/* This field specifies the maximum regular port identifier allowed. */
#define LW402C_CTRL_NUM_I2C_PORTS                         16
/* This temporary field specifies the dynamic port identifier. */
#define LW402C_CTRL_DYNAMIC_PORT                          LW_U8_MAX

/*
 * LW402C_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *     LW_OK
 */
#define LW402C_CTRL_CMD_NULL                              (0x402c0000) /* finn: Evaluated from "(FINN_LW40_I2C_RESERVED_INTERFACE_ID << 8) | 0x0" */



/*
 * LW402C_CTRL_I2C_GET_PORT_INFO_IMPLEMENTED
 *   The port exists on this hardware.
 * LW402C_CTRL_I2C_GET_PORT_INFO_DCB_DECLARED
 *   The port has an entry in the DCB.
 * LW402C_CTRL_I2C_GET_PORT_INFO_DDC_CHANNEL
 *   The port is used to read EDIDs via DDC.
 * LW402C_CTRL_I2C_GET_PORT_INFO_CRTC_MAPPED
 *   The port is accessible via the CRTC register space.
 * LW402C_CTRL_I2C_GET_PORT_INFO_VALID
 *   The port is validated using I2C device.
 */
#define LW402C_CTRL_I2C_GET_PORT_INFO_IMPLEMENTED                          0:0
#define LW402C_CTRL_I2C_GET_PORT_INFO_IMPLEMENTED_NO      0x00
#define LW402C_CTRL_I2C_GET_PORT_INFO_IMPLEMENTED_YES     0x01
#define LW402C_CTRL_I2C_GET_PORT_INFO_DCB_DECLARED                         1:1
#define LW402C_CTRL_I2C_GET_PORT_INFO_DCB_DECLARED_NO     0x00
#define LW402C_CTRL_I2C_GET_PORT_INFO_DCB_DECLARED_YES    0x01
#define LW402C_CTRL_I2C_GET_PORT_INFO_DDC_CHANNEL                          2:2
#define LW402C_CTRL_I2C_GET_PORT_INFO_DDC_CHANNEL_ABSENT  0x00
#define LW402C_CTRL_I2C_GET_PORT_INFO_DDC_CHANNEL_PRESENT 0x01
#define LW402C_CTRL_I2C_GET_PORT_INFO_CRTC_MAPPED                          3:3
#define LW402C_CTRL_I2C_GET_PORT_INFO_CRTC_MAPPED_NO      0x00
#define LW402C_CTRL_I2C_GET_PORT_INFO_CRTC_MAPPED_YES     0x01
#define LW402C_CTRL_I2C_GET_PORT_INFO_VALID                                4:4
#define LW402C_CTRL_I2C_GET_PORT_INFO_VALID_NO            0x00
#define LW402C_CTRL_I2C_GET_PORT_INFO_VALID_YES           0x01
#define LW402C_CTRL_I2C_GET_PORT_INFO_ALL                                  4:0
#define LW402C_CTRL_I2C_GET_PORT_INFO_ALL_DEFAULT         0x00

/*
 * LW402C_CTRL_CMD_I2C_GET_PORT_INFO
 *
 * Returns information for the first eight I2C ports.
 *
 *   info
 *     This parameter is an output from the command and is ignored as an
 *     input.  Each element contains the flags described previously named
 *     LW402C_CTRL_I2C_GET_PORT_INFO*.  Note that the index into the info
 *     array is one less than the port identifier that would be returned from
 *     LW0073_CTRL_CMD_SPECIFIC_GET_I2C_PORTID; the port numbers here are
 *     0-indexed as opposed to 1-indexed.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */




#define LW402C_CTRL_CMD_I2C_GET_PORT_INFO (0x402c0101) /* finn: Evaluated from "(FINN_LW40_I2C_I2C_INTERFACE_ID << 8) | LW402C_CTRL_I2C_GET_PORT_INFO_PARAMS_MESSAGE_ID" */



#define LW402C_CTRL_I2C_GET_PORT_INFO_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW402C_CTRL_I2C_GET_PORT_INFO_PARAMS {
    LwU8 info[LW402C_CTRL_NUM_I2C_PORTS];
} LW402C_CTRL_I2C_GET_PORT_INFO_PARAMS;
#define LW402C_CTRL_I2C_INDEX_LENGTH_MAX                                        4
#define LW402C_CTRL_I2C_MESSAGE_LENGTH_MAX                                      4096

//! Minimum and maximum valid read/write message length for block process protocol.
#define LW402C_CTRL_I2C_BLOCK_PROCESS_PROTOCOL_MIN                              3
#define LW402C_CTRL_I2C_BLOCK_PROCESS_PROTOCOL_MAX                              32

/*
 * LW402C_CTRL_I2C_FLAGS_ADDRESS_MODE
 *   A client uses this field to indicate the I2C addressing mode to be
 *   used.
 *   Possible values are:
 *     LW402C_CTRL_I2C_FLAGS_ADDRESS_MODE_7BIT
 *       The default, this value specifies the master to operate in the
 *       basic 7-bit addressing mode, which is available on all
 *       implementations.
 *     LW402C_CTRL_I2C_FLAGS_ADDRESS_MODE_10BIT
 *       This I2C mode allows for 10 bits of addressing space and is
 *       reverse compatible with 7-bit addressing.
 */
#define LW402C_CTRL_I2C_FLAGS_ADDRESS_MODE                                 0:0
#define LW402C_CTRL_I2C_FLAGS_ADDRESS_MODE_7BIT                                 (0x00000000)
#define LW402C_CTRL_I2C_FLAGS_ADDRESS_MODE_10BIT                                (0x00000001)
#define LW402C_CTRL_I2C_FLAGS_ADDRESS_MODE_DEFAULT                              LW402C_CTRL_I2C_FLAGS_ADDRESS_MODE_7BIT
/*
 * LW402C_CTRL_I2C_FLAGS_SPEED_MODE
 *   A client uses this field to indicate the target speed at which the
 *   I2C master should attempt to drive the bus.  The master may throttle
 *   its own speed for various reasons, and devices may slow the bus
 *   using clock-streching.  Neither of these possibilities are
 *   considered failures.
 */
#define LW402C_CTRL_I2C_FLAGS_SPEED_MODE                                   4:1
#define LW402C_CTRL_I2C_FLAGS_SPEED_MODE_100KHZ                                 (0x00000000)
#define LW402C_CTRL_I2C_FLAGS_SPEED_MODE_200KHZ                                 (0x00000001)
#define LW402C_CTRL_I2C_FLAGS_SPEED_MODE_400KHZ                                 (0x00000002)
#define LW402C_CTRL_I2C_FLAGS_SPEED_MODE_33KHZ                                  (0x00000003)
#define LW402C_CTRL_I2C_FLAGS_SPEED_MODE_10KHZ                                  (0x00000004)
#define LW402C_CTRL_I2C_FLAGS_SPEED_MODE_3KHZ                                   (0x00000005)
#define LW402C_CTRL_I2C_FLAGS_SPEED_MODE_DEFAULT                                (0x00000006)
#define LW402C_CTRL_I2C_FLAGS_SPEED_MODE_300KHZ                                 (0x00000007)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW402C_CTRL_I2C_FLAGS_PRIVILEGE
 *   A client uses this field to specify the privilege level at which to
 *   attempt the transaction.  Certain devices may be blocked by the driver if
 *   privilege is not sufficiently high.  While the direct caller may choose
 *   any flag, this feature is typically used for pass-through calls, such as
 *   those available in LWAPI.
 *   Possible values are:
 *     LW402C_CTRL_I2C_FLAGS_PRIVILEGE_PUBLIC
 *       The default, this value lets the driver block traffic to sensitive
 *       devices.
 *     LW402C_CTRL_I2C_FLAGS_PRIVILEGE_NDA
 *       The NDA privilege level overrides the blacklist.
 */
#define LW402C_CTRL_I2C_FLAGS_PRIVILEGE                                    6:5
#define LW402C_CTRL_I2C_FLAGS_PRIVILEGE_PUBLIC                                  (0x00000000)
#define LW402C_CTRL_I2C_FLAGS_PRIVILEGE_NDA                                     (0x00000001)
#define LW402C_CTRL_I2C_FLAGS_PRIVILEGE_DEFAULT                                 LW402C_CTRL_I2C_FLAGS_PRIVILEGE_PUBLIC
/*
 * LW402C_CTRL_I2C_FLAGS_WAR
 *   A client uses this field to specify a workaround code path for a specific
 *   device.  Most devices follow standard I2C, but a few do not.
 *   Possible values are:
 *     LW402C_CTRL_I2C_FLAGS_WAR_NONE
 *       The default, this value indicates to use standard code paths.
 *     LW402C_CTRL_I2C_FLAGS_WAR_SI1930
 *       This value specifies the device is a SI1930 microcontroller.
 *     LW402C_CTRL_I2C_FLAGS_WAR_PX3540
 *       This value specifies the device is a PX3540 or PX3544.
 *     LW402C_CTRL_I2C_FLAGS_WAR_NO_AUTO_INC
 *       This value specifies that the device does not support auto-increment.
 *       Most devices allow you to write multiple bytes after specifying a
 *       register address, and the subsequent bytes will go to incremented
 *       addresses.  Without auto-increment, we write a buffer of data as a
 *       sequence of address-register-value triplets separated by starts.
 */
#define LW402C_CTRL_I2C_FLAGS_WAR                                         9:7
#define LW402C_CTRL_I2C_FLAGS_WAR_NONE                                          (0x00000000)
#define LW402C_CTRL_I2C_FLAGS_WAR_SI1930                                        (0x00000001)
#define LW402C_CTRL_I2C_FLAGS_WAR_PX3540                                        (0x00000002)
#define LW402C_CTRL_I2C_FLAGS_WAR_NO_AUTO_INC                                   (0x00000003)
#define LW402C_CTRL_I2C_FLAGS_WAR_FORCE_MOT                                     (0x00000004)
#define LW402C_CTRL_I2C_FLAGS_WAR_DEFAULT                                       LW402C_CTRL_I2C_FLAGS_WAR_NONE

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


/*
 * LW402C_CTRL_I2C_FLAGS_TRANSACTION_MODE
 *   A client uses this field to specify a transaction mode.
 *   Possible values are:
 *     LW402C_CTRL_I2C_FLAGS_TRANSACTION_MODE_NORMAL
 *       The default, this value indicates to use the normal I2C transaction 
 *       mode which will involve read/write operations depending on client's
 *       needs. 
 *     LW402C_CTRL_I2C_FLAGS_TRANSACTION_MODE_PING
 *       This value specifies that the device only needs to be pinged. No need
 *       of performing a complete read/write transaction. This will send a
 *       single byte to the device to be pinged. On receiving an ACK, we will
 *       get a confirmation on the device's availability.
 */
#define LW402C_CTRL_I2C_FLAGS_TRANSACTION_MODE                          11:10
#define LW402C_CTRL_I2C_FLAGS_TRANSACTION_MODE_NORMAL                           (0x00000000)
#define LW402C_CTRL_I2C_FLAGS_TRANSACTION_MODE_PING                             (0x00000001)
#define LW402C_CTRL_I2C_FLAGS_TRANSACTION_MODE_DEFAULT                          LW402C_CTRL_I2C_FLAGS_TRANSACTION_MODE_NORMAL
/*!
 * LW402C_CTRL_I2C_FLAGS_RESERVED
 *   A client must leave this field as 0, as it is reserved for future use.
 */
#define LW402C_CTRL_I2C_FLAGS_RESERVED                                  31:12

/*!
 * The following defines specify WAR flags that can be specified during
 * I2C Quick Read or Write command (Refer LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_QUICK_RW).
 *
 * _NONE
 *    No workaround is needed.
 *
 * _TEST_PORT
 *     Use this flag to have the client sent a request to test a port instead 
 *     of performing any transaction on it. Transaction type has to be 
 *     LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_QUICK_RW.
 */
#define LW402C_CTRL_I2C_SMBUS_QUICK_RW_WAR_FLAGS                                          0:0
#define LW402C_CTRL_I2C_SMBUS_QUICK_RW_WAR_FLAGS_NONE                           0x00000000
#define LW402C_CTRL_I2C_SMBUS_QUICK_RW_WAR_FLAGS_TEST_PORT                      0x00000001

/*!
 * The following defines specify WAR flags that can be specified during
 * I2C Register Read or Write buffer command 
 * (Refer LW402C_CTRL_I2C_TRANSACTION_TYPE_I2C_BUFFER_RW)
 *
 * _NONE
 *    No workaround is needed.
 *
 * _SI1930
 *     SI1930 microcontroller register read or write requested by a client. 
 *     Transaction type has to be LW402C_CTRL_I2C_TRANSACTION_TYPE_I2C_BUFFER_RW.
 *
 * _PX3540
 *     Register read from PX3540 or PX3544 device. Transaction type has to be 
 *     LW402C_CTRL_I2C_TRANSACTION_TYPE_I2C_BUFFER_RW and bWrite must be TRUE to 
 *     indicate READ operation 
 */
#define LW402C_CTRL_I2C_BUFFER_RW_WAR_FLAGS                                               1:0
#define LW402C_CTRL_I2C_BUFFER_RW_WAR_FLAGS_NONE                                0x00000000
#define LW402C_CTRL_I2C_BUFFER_RW_WAR_FLAGS_SI1930                              0x00000001
#define LW402C_CTRL_I2C_BUFFER_RW_WAR_FLAGS_PX3540                              0x00000002

/*!
 * The following defines specify WAR flags that can be specified during
 * I2C buffer Read or Write to Multibyte Register 
 * (Refer LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_MULTIBYTE_REGISTER_BLOCK_RW)
 *
 * _NONE
 *    No workaround is needed.
 *
 * _NO_AUTO_INC
 *   This value specifies that the device does not support auto-increment.
 *   Most devices allow you to write multiple bytes after specifying a
 *   register address, and the subsequent bytes will go to incremented
 *   addresses. Without auto-increment, we write a buffer of data as a
 *   sequence of address-register-value triplets separated by starts.
 */
#define LW402C_CTRL_I2C_SMBUS_MULTIBYTE_REGISTER_BLOCK_RW_WAR_FLAGS                       0:0
#define LW402C_CTRL_I2C_SMBUS_MULTIBYTE_REGISTER_BLOCK_RW_WAR_FLAGS_NONE        0x00000000
#define LW402C_CTRL_I2C_SMBUS_MULTIBYTE_REGISTER_BLOCK_RW_WAR_FLAGS_NO_AUTO_INC 0x00000001

/*
 * LW402C_CTRL_CMD_I2C_SYNC
 *
 * Perform a basic I2C transaction synchronously.
 *
 *   portId
 *     This field must be specified by the client to indicate the logical
 *     port/bus for which the transaction is requested. The port identifier
 *     is one less than the value returned by
 *     LW0073_CTRL_CMD_SPECIFIC_GET_I2C_PORTID unless that value was 0 (the
 *     'dynamic' port).  For the 'dynamic' port, this should be 0xFF.  Note
 *     that future versions of the API may obsolete use of the 'dynamic' port;
 *     please contact the RM if you begin using this portion of the API so we
 *     can help you migrate when the time comes.
 *
 *   bIsWrite
 *     This field must be specified by the client to indicate whether the
 *     command is a write (TRUE) or a read (FALSE).
 *
 *   flags
 *     This parameter specifies optional flags used to control certain modal
 *     features such as target speed and addressing mode.  The lwrrently
 *     defined fields are described previously; see LW402C_I2C_FLAGS_*.
 *
 *   address
 *     The address of the I2C slave.  The address should be shifted left by
 *     one.  For example, the I2C address 0x50, often used for reading EDIDs,
 *     would be stored here as 0xA0.  This matches the position within the
 *     byte sent by the master, as the last bit is reserved to specify the
 *     read or write direction.
 *
 *   indexLength
 *     This required parameter specifies how many bytes to write as part of the
 *     first index.  If zero is specified, then no index will be sent.
 *
 *   index
 *     This parameter, required of the client if index is one or more,
 *     specifies the index to be written.  The buffer should be arranged such
 *     that index[0] will be the first byte sent.
 *
 *   messageLength
 *     This parameter, required of the client, specifies the number of bytes to
 *     read or write from the slave after the index is written.
 *
 *   pMessage
 *     This parameter, required of the client, specifies the data to be written
 *     to the slave.  The buffer should be arranged such that pMessage[0] will
 *     be the first byte read or written.  If the transaction is a read, then
 *     it will follow the combined format described in the I2C specification.
 *     If the transaction is a write, the message will immediately follow the
 *     index without a restart.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_STATE_IN_USE
 *   LW_ERR_GENERIC, if the I2C transaction fails.
 */
#define LW402C_CTRL_I2C_INDEXED_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW402C_CTRL_I2C_INDEXED_PARAMS {
    LwU8  portId;
    LwU8  bIsWrite;
    LwU16 address;
    LwU32 flags;

    LwU32 indexLength;
    LwU8  index[LW402C_CTRL_I2C_INDEX_LENGTH_MAX];

    LwU32 messageLength;
    LW_DECLARE_ALIGNED(LwP64 pMessage, 8);
} LW402C_CTRL_I2C_INDEXED_PARAMS;

#define LW402C_CTRL_CMD_I2C_INDEXED (0x402c0102) /* finn: Evaluated from "(FINN_LW40_I2C_I2C_INTERFACE_ID << 8) | LW402C_CTRL_I2C_INDEXED_PARAMS_MESSAGE_ID" */

/*
 * LW402C_CTRL_CMD_I2C_GET_PORT_SPEED
 *
 * Returns information for the I2C ports.
 *
 *   portSpeed
 *     This parameter is an output from the command and is ignored as an
 *     input.  Each element contains the current I2C speed of the port.
 *     Note that the index into the info array is one less than the 
 *     port identifier that would be returned from
 *     LW0073_CTRL_CMD_SPECIFIC_GET_I2C_PORTID; the port numbers here are
 *     0-indexed as opposed to 1-indexed.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW402C_CTRL_I2C_GET_PORT_SPEED_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW402C_CTRL_I2C_GET_PORT_SPEED_PARAMS {
    LwU32 portSpeed[LW402C_CTRL_NUM_I2C_PORTS];
} LW402C_CTRL_I2C_GET_PORT_SPEED_PARAMS;

#define LW402C_CTRL_CMD_I2C_GET_PORT_SPEED (0x402c0103) /* finn: Evaluated from "(FINN_LW40_I2C_I2C_INTERFACE_ID << 8) | LW402C_CTRL_I2C_GET_PORT_SPEED_PARAMS_MESSAGE_ID" */

/*
 * LW402C_CTRL_I2C_DEVICE_INFO
 *
 * This structure describes the basic I2C Device information.
 *
 *   type
 *     This field return the type of device LW_DCB4X_I2C_DEVICE_TYPE_<xyz>
 *   i2cAddress
 *     This field contains the 7 bit/10 bit address of the I2C device.
 *   i2cLogicalPort
 *     This field contains the Logical port of the I2C device.
 */
typedef struct LW402C_CTRL_I2C_DEVICE_INFO {
    LwU8  type;
    LwU16 i2cAddress;
    LwU8  i2cLogicalPort;
    LwU8  i2cDevIdx;
} LW402C_CTRL_I2C_DEVICE_INFO;

/* Maximum number of I2C devices in DCB */
#define LW402C_CTRL_I2C_MAX_DEVICES 32

/*
 * LW402C_CTRL_CMD_I2C_TABLE_GET_DEV_INFO
 *
 * RM Control to get I2C device info from the DCB I2C Devices Table.
 *
 *   i2cDevCount
 *     The value of this parameter will give the number of
 *     I2C devices found in DCB.
 * 
 *   i2cDevInfo 
 *     For each device found in DCB the control call will write the info 
 *     in this parameter.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */
#define LW402C_CTRL_I2C_TABLE_GET_DEV_INFO_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW402C_CTRL_I2C_TABLE_GET_DEV_INFO_PARAMS {
    LwU8                        i2cDevCount;
    LW402C_CTRL_I2C_DEVICE_INFO i2cDevInfo[LW402C_CTRL_I2C_MAX_DEVICES];
} LW402C_CTRL_I2C_TABLE_GET_DEV_INFO_PARAMS;

#define LW402C_CTRL_CMD_I2C_TABLE_GET_DEV_INFO (0x402c0104) /* finn: Evaluated from "(FINN_LW40_I2C_I2C_INTERFACE_ID << 8) | LW402C_CTRL_I2C_TABLE_GET_DEV_INFO_PARAMS_MESSAGE_ID" */

/*!
 * The IDs of each type of I2C command available.
 */
typedef enum LW402C_CTRL_I2C_TRANSACTION_TYPE {
   /*!
    * This transaction type is used to perform the Quick SMBus Read/write command 
    * on a slave device. No data is sent or received, just used to verify the 
    * presence of the device.
    * Refer SMBus spec 2.0 (section 5.5.1 Quick Command)
    * SMBus Quick Write : S Addr|Wr [A] P
    * SMBus Quick Read  : S Addr|Rd [A] P
    */
    LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_QUICK_RW = 0,
   /*!
    * This transaction type is used to perform the I2C byte read/write from/to 
    * a slave device. As per the spec last byte should be NA (Not Acknolwedged) 
    * by slave.
    * Refer I2CBus spec 3.0 (section 9 Fig 11 and Fig 12) or Refer SMBus spec 
    * 2.0 (section 5.5.2 Send Byte and 5.5.3 Receive Byte).
    * I2C Byte Write : S Addr|Wr [A] Data [NA] P
    * I2C Byte Read  : S Addr|Rd [A] Data NA P
    */
    LW402C_CTRL_I2C_TRANSACTION_TYPE_I2C_BYTE_RW = 1,
   /*!
    * This transaction type is used to perform the I2C block (buffer) 
    * read/write from/to a slave device. As per the spec last byte should be NA
    * (Not Acknolwedged) by slave.
    * Refer I2CBus spec 3.0 (section 9 Fig 11 and Fig 12)
    * I2C Byte Write : S Addr|Wr [A] Data1 [A]...Data(N-1) [A] DataN [NA] P
    * I2C Byte Read  : S Addr|Rd [A] Data1 A...Data(N-1) A DataN NA P
    *
    * Distinction between I2C_BLOCK and SMBUS_BLOCK protocol:
    * In I2C Block write it is the slave device (and in I2C Block read it's 
    * the master device) that determines the number of bytes to transfer by 
    * asserting the NAK at last bit before stop. This differs from the SMBus 
    * block mode write command in which the master determines the block 
    * write transfer size. In I2c Block read there is no limit to maximum size 
    * of data that could be transferred whereas in SMBus block it is restricted
    * to 255 bytes (0xFF).
    */
    LW402C_CTRL_I2C_TRANSACTION_TYPE_I2C_BLOCK_RW = 2,
   /*!
    * This transaction type is used to perform the I2C Buffer read/write 
    * from/to a register of a slave device. It does not send bytecount as 
    * part of data buffer.
    * Not a part of SMBus spec.
    * I2C Buffer Write : S Addr|Wr [A] cmd [A] Data1 [A]...DataN[A] P
    * I2C Buffer Read  : S Addr|Wr [A] cmd [A] Sr Addr|Rd [A] Data1 A...
    *                    DataN-1 A DataN A P
    *
    * Distinction between SMBUS_BLOCK and I2C_BUFFER protocol:
    * In SMBUS_BLOCK Read/write the first byte of data buffer contains the 
    * count size (The number of bytes to be transferred) and it is restricted 
    * to 255 bytes whereas in I2C_BUFFER, count size is not sent during the 
    * transfer and there is no restriction in terms of size.
    *
    * Distinction between I2C_BLOCK and I2C_BUFFER protocol:
    * I2C_BUFFER takes the register address as argument whereas I2C_BLOCK does
    * not have any register or command provision.
    */
    LW402C_CTRL_I2C_TRANSACTION_TYPE_I2C_BUFFER_RW = 3,
   /*!
    * This transaction type is used to perform the I2C byte read/write from/to
    * a slave device
    * Refer SMBus spec 2.0 (section 5.5.4 Write Byte and 5.5.5 Read Byte)
    * SMBus Byte Write : S Addr|Wr [A] cmd [A] Data [A] P
    * SMBus Byte Read  : S Addr|Wr [A] cmd [A] Sr Addr|Rd [A] Data A P
    */
    LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_BYTE_RW = 4,
   /*!
    * This transaction type is used to perform the SMBus byte read/write 
    * from/to a register of a slave device
    * Refer SMBus spec 2.0 (section 5.5.4 Write Word and 5.5.5 Read Word)
    * SMBus Word Write : S Addr|Wr [A] cmd [A] DataLow [A] DataHigh [A] P 
    * SMBus Word Read  : S Addr|Wr [A] cmd [A] Sr Addr|Rd [A] DataLow A 
    *                    DataHigh A P
    */
    LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_WORD_RW = 5,
   /*!
    * This transaction type is used to perform the SMBus Block read/write 
    * from/to a register of a slave device
    * Refer SMBus spec 2.0 (section 5.5.7 Block Write/Read)
    * SMBus Block Write : S Addr|Wr [A] cmd [A] ByteCount [A] Data1 [A]...
    *                     DataN-1 [A] DataN[A] P
    * SMBus Block Read  : S Addr|Wr [A] cmd [A] Sr Addr|Rd [A] ByteCount A 
    *                     Data1 A...DataN-1 A DataN A P
    *
    * Distinction between I2C_BLOCK and SMBUS_BLOCK protocol:
    * In I2C Block write it is the slave device (and in I2C Block read it's 
    * the master device) that determines the number of bytes to transfer by 
    * asserting the NAK at last bit before stop. This differs from the SMBus 
    * block mode write/Read command in which the master determines the block 
    * write transfer size. In I2c Block read/Write there is no limit to maximum
    * size of data that could be transferred whereas in SMBus block it is 
    * restricted to 255 bytes (0xFF).
    */
    LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_BLOCK_RW = 6,
   /*!
    * This transaction type is used to perform the SMBus process call. It sends
    * data and waits for the slave to return a value dependent on that data. 
    * The protocol is simply a SMBus write Word followed by a SMBus Read Word 
    * without the Read-Word command field and the Write-Word STOP bit. 
    * Note that there is no STOP condition before the repeated START condition,
    * and that a NACK signifies the end of the read transfer. 
    *
    * Refer SMBus spec 2.0 (section 5.5.6 Process Call)
    * SMBus Process Call : S Addr|Wr [A] cmd [A] DataLow [A] DataHigh [A] 
    *                      Sr Addr|Rd [A] DataLow [A] DataHigh [NA] P
    */
    LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_PROCESS_CALL = 7,
   /*!
    * This transaction type is used to perform the SMBus Block Write Block Read
    * process call.
    * The block write-block read process call is a two-part message. The call 
    * begins with a slave address and a write condition. After the command code
    * the host issues a write byte count (M) that describes how many more bytes
    * will be written in the first part of the message.
    * If a master has 6 bytes to send, the byte count field will have the value
    * 6 (0000 0110b), followed by the 6 bytes of data. The write byte count (M)
    * cannot be zero. 
    * The second part of the message is a block of read data beginning with a 
    * repeated start condition followed by the slave address and a Read bit.
    * The next byte is the read byte count (N), which may differ from the write
    * byte count (M). The read byte count (N) cannot be zero. The combined data
    * payload must not exceed 32 bytes. 
    * The byte length restrictions of this process call are summarized as 
    * follows:
    * M >= 1 byte 
    * N >= 1 byte 
    * M + N <= 32 bytes 
    * Note that there is no STOP condition before the repeated START condition, 
    * and that a NACK signifies the end of the read transfer.
    *
    * Refer SMBus spec 2.0 (section 5.5.8 Block Write Block Read Process Call)
    * SMBus Process Call : S Addr|Wr [A] cmd [A] ByteCount=M [A] Data1 [A]...
    *                      DataN-1 [A] DataM[A] Sr Addr|Rd [A] ByteCount=N [A] 
    *                      Data1 [A]...DataN [NA] P
    */
    LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_BLOCK_PROCESS_CALL = 8,
   /*!
    * This transaction type is used to perform SMBus buffer read/write 
    * from/to multiple registers of a slave device known as Auto Increment.
    * It is not a part of any standard I2C/SMBus spec but a feature of many
    * SMBus devices like EEPROM.
    * It is also used for reading a block of bytes from a designated register 
    * that is specified through the two Comm bytes.of a slave device or writing
    * a block of bytes from a designated register of a slave device (Note : The
    * command byte in this case could be 0, 2 or 4 Bytes)
    * SMBus Multi-Byte Register Block Write : S Addr|Wr [A] cmd1 A cmd 2 [A]...
    *                cmdN [A] data1 [A] Data2 [A].....DataN [A] P
    * SMBus Multi-Byte Register Block Read : S Addr|Rd [A] cmd1 A cmd 2 [A]...
    *                cmdN [A] data1 [A] Sr Addr [A] Data1 A Data2 A...DataN A P
    *
    * This transaction type could be also used for those devices which supports 
    * AUTO_INC. Even though it is frequently related to I2C/SMBus, automatic 
    * incrementation is not part of any I2C standard but rather a common
    * feature found in many I2C devices. What it means is that the device
    * maintains in internal pointer which is  automatically incremented upon 
    * data read or write activities and which can be manually set to a fixed 
    * value. This comes in handy when storing larger amounts of data for 
    * instance in an ordinary I2C RAM or EEPROM.
    * SMBus AUTO_INC Write : S Addr|Wr [A] cmd1 A Data1 [A] Data2 [A]...
    *                        DataN [A] P
    * SMBus AUTO_INC Read : S Addr|Rd [A] cmd1 A data1 [A] Sr Addr [A] Data1 A
    *                       Data2 A...DataN A P
    * If the device does not support AUTO_INC set warFlags of 
    * LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_MULTIBYTE_REGISTER_BLOCK_RW to 
    * LW402C_CTRL_I2C_SMBUS_MULTIBYTE_REGISTER_BLOCK_RW_WAR_FLAGS_NO_AUTO_INC.
    */
    LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_MULTIBYTE_REGISTER_BLOCK_RW = 9,
   /*!
    * This transaction type is used to perform the EDID read via DDC.
    */
    LW402C_CTRL_I2C_TRANSACTION_TYPE_READ_EDID_DDC = 10,
} LW402C_CTRL_I2C_TRANSACTION_TYPE;

/*!
 * LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_QUICK_RW
 *
 * Specifies the structure of data filled by the client for I2C transaction 
 * when the.transaction type is LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_QUICK_RW.
 *
 */
typedef struct LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_QUICK_RW {
   /*!
    * This field must be specified by the client to indicate whether the
    * command is a write (TRUE) or a read (FALSE).
    */
    LwBool bWrite;
   /*!
    * Transaction specific flags
    * (see LW402C_CTRL_I2C_SMBUS_QUICK_RW_WAR_FLAGS_*).
    */
    LwU32  warFlags;
} LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_QUICK_RW;

/*!
 * LW402C_CTRL_I2C_TRANSACTION_DATA_I2C_BYTE_RW
 *
 * Specifies the structure of data filled by the client for I2C transaction 
 * when the.transaction type is LW402C_CTRL_I2C_TRANSACTION_TYPE_I2C_BYTE_RW.
 *
 */
typedef struct LW402C_CTRL_I2C_TRANSACTION_DATA_I2C_BYTE_RW {
   /*!
    * This field must be specified by the client to indicate whether the
    * command is a write (TRUE) or a read (FALSE).
    */
    LwBool bWrite;
   /*!
    * The main message data.
    */
    LwU8   message;
} LW402C_CTRL_I2C_TRANSACTION_DATA_I2C_BYTE_RW;

/*!
 * LW402C_CTRL_I2C_TRANSACTION_DATA_I2C_BLOCK_RW
 *
 * Specifies the structure of data filled by the client for I2C transaction 
 * when the transaction type is LW402C_CTRL_I2C_TRANSACTION_TYPE_I2C_BLOCK_RW.
 *
 */
typedef struct LW402C_CTRL_I2C_TRANSACTION_DATA_I2C_BLOCK_RW {
   /*!
    * This field must be specified by the client to indicate whether the
    * command is a write (TRUE) or a read (FALSE).
    */
    LwBool bWrite;
   /*!
    * This parameter specifies the number of bytes to read or
    * write from the slave after the register address is written.
    */
    LwU32  messageLength;
   /*!
    * The main message data. 
    */
    LW_DECLARE_ALIGNED(LwP64 pMessage, 8);
} LW402C_CTRL_I2C_TRANSACTION_DATA_I2C_BLOCK_RW;

/*!
 * LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_BYTE_RW
 *
 * Specifies the structure of data filled by the client for I2C transaction 
 * when the.transaction type is LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_BYTE_RW.
 *
 */
typedef struct LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_BYTE_RW {
   /*!
    * This field must be specified by the client to indicate whether the
    * command is a write (TRUE) or a read (FALSE).
    */
    LwBool bWrite;
   /*!
    * The address of the register.
    */
    LwU8   registerAddress;
   /*!
    * The main message data.
    */
    LwU8   message;
} LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_BYTE_RW;

/*!
 * LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_WORD_RW
 *
 * Specifies the structure of data filled by the client for I2C transaction 
 * when the.transaction type is LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_WORD_RW.
 *
 */
typedef struct LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_WORD_RW {
   /*!
    * This field must be specified by the client to indicate whether the
    * command is a write (TRUE) or a read (FALSE).
    */
    LwBool bWrite;
   /*!
    * The address of the register.
    */
    LwU8   registerAddress;
   /*!
    * The main message data.
    */
    LwU16  message;
} LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_WORD_RW;

/*!
 * LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_BUFFER_RW
 *
 * Specifies the structure of data filled by the client for I2C transaction 
 * when the.transaction type is 
 * LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_BUFFER_RW.
 */
typedef struct LW402C_CTRL_I2C_TRANSACTION_DATA_I2C_BUFFER_RW {
   /*!
    * This field must be specified by the client to indicate whether the
    * command is a write (TRUE) or a read (FALSE).
    */
    LwBool bWrite;
   /*!
    * Transaction specific flags to be set (see 
    * LW_LW402C_CTRL_I2C_BUFFER_RW_WAR_FLAGS_*)
    */
    LwU32  warFlags;
   /*!
    * This parameter specifies how many bytes to write as part of the
    * register address. If zero is specified, then no index will be sent.
    */
    LwU8   registerAddress;
   /*!
    * This parameter specifies the number of bytes to read or 
    * write from the slave after the register address is written.
    */
    LwU32  messageLength;
   /*!
    * The main message data. 
    * C form: LwU8 message[LW402C_CTRL_I2C_MESSAGE_LENGTH_MAX]
    */
    LW_DECLARE_ALIGNED(LwP64 pMessage, 8);
} LW402C_CTRL_I2C_TRANSACTION_DATA_I2C_BUFFER_RW;

/*!
 * LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_BLOCK_RW
 *
 * Specifies the structure of data filled by the client for I2C transaction 
 * when the.transaction type is 
 * LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_BLOCK_RW.
 */
typedef struct LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_BLOCK_RW {
   /*!
    * This field must be specified by the client to indicate whether the
    * command is a write (TRUE) or a read (FALSE).
    */
    LwBool bWrite;
   /*!
    * This parameter specifies how many bytes to write as part of the
    * register address. If zero is specified, then no index will be sent.
    */
    LwU8   registerAddress;
   /*!
    * This parameter specifies the number of bytes to read or 
    * write from the slave after the register address is written.
    */
    LwU32  messageLength;
   /*!
    * The main message data.
    * C form: LwU8 message[LW402C_CTRL_I2C_MESSAGE_LENGTH_MAX]
    */
    LW_DECLARE_ALIGNED(LwP64 pMessage, 8);
} LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_BLOCK_RW;

/*!
 * LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_PROCESS_CALL
 *
 * Specifies the structure of data filled by the client for I2C transaction 
 * when the.transaction type is
 * LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_PROCESS_CALL.
 */
typedef struct LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_PROCESS_CALL {
   /*!
    * This parameter specifies how many bytes to write as part of the
    * register address. If zero is specified, then no index will be sent.
    */
    LwU8  registerAddress;
   /*!
    * The message data to be written to the slave.
    */
    LwU16 writeMessage;
   /*!
    * The message data to be read from the slave.
    */
    LwU16 readMessage;
} LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_PROCESS_CALL;

/*!
 * LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_BLOCK_PROCESS_CALL
 *
 * Specifies the structure of data filled by the client for I2C transaction 
 * when the.transaction type is 
 * LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_BLOCK_PROCESS_CALL.
 *
 */
typedef struct LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_BLOCK_PROCESS_CALL {
   /*!
    * This parameter specifies how many bytes to write as part of the
    * register address. If zero is specified, then no index will be sent.
    */
    LwU8  registerAddress;
   /*!
    * This parameter specifies the number of bytes to write the the slave 
    * after the writeByteCount is sent to the slave.
    */
    LwU32 writeMessageLength;
   /*!
    * The message buffer to be written to the slave.
    * C form: LwU8 writeMessage[LW402C_CTRL_I2C_BLOCK_PROCESS_PROTOCOL_MAX]
    */
    LwU8  writeMessage[LW402C_CTRL_I2C_BLOCK_PROCESS_PROTOCOL_MAX];
   /*!
    * This parameter specifies the number of bytes to read from the slave 
    * after the readByteCount is sent to the slave.
    */
    LwU32 readMessageLength;
   /*!
    * The message buffer to be read from the slave.
    * C form: LwU8 readMessage[LW402C_CTRL_I2C_BLOCK_PROCESS_PROTOCOL_MAX]
    */
    LwU8  readMessage[LW402C_CTRL_I2C_BLOCK_PROCESS_PROTOCOL_MAX];
} LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_BLOCK_PROCESS_CALL;

/*!
 * LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_MULTIBYTE_REGISTER_BLOCK_RW
 *
 * Specifies the structure of data filled by the client for I2C transaction 
 * when the.transaction type is 
 * LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_MULTIBYTE_REGISTER_BLOCK_RW.
 */
typedef struct LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_MULTIBYTE_REGISTER_BLOCK_RW {
   /*!
    * This field must be specified by the client to indicate whether the
    * command is a write (TRUE) or a read (FALSE).
    */
    LwBool bWrite;
   /*!
    * Transaction specific flags (see 
    * LW402C_CTRL_I2C_SMBUS_MULTIBYTE_REGISTER_BLOCK_RW_WAR_FLAGS_*)
    */
    LwU32  warFlags;
   /*!
    * This parameter specifies how many bytes to write as part of the
    * register address. If zero is specified, then no index will be sent.
    */
    LwU32  indexLength;
   /*!
    * Optional indexing data; aka register address.
    * C form: LwU8 index[LW402C_CTRL_I2C_INDEX_LENGTH_MAX]
    */
    LwU8   index[LW402C_CTRL_I2C_INDEX_LENGTH_MAX];
   /*!
    * This parameter specifies the number of bytes to read or 
    * write from the slave after the register address is written.
    */
    LwU32  messageLength;
   /*!
    * The main message data. 
    * C form: LwU8 message[LW402C_CTRL_I2C_MESSAGE_LENGTH_MAX]
    */
    LW_DECLARE_ALIGNED(LwP64 pMessage, 8);
} LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_MULTIBYTE_REGISTER_BLOCK_RW;

/*!
 * LW402C_CTRL_I2C_TRANSACTION_DATA_READ_EDID_DDC
 *
 * Specifies the structure of data filled by the client for I2C transaction 
 * when the.transaction type is LW402C_CTRL_I2C_TRANSACTION_TYPE_READ_EDID_DDC.
 *
 */
typedef struct LW402C_CTRL_I2C_TRANSACTION_DATA_READ_EDID_DDC {
    /*!
     * The segment number of the EDID block which is to be read.
     */
    LwU8  segmentNumber;
    /*!
     * The address of the register.
     */
    LwU8  registerAddress;
   /*!
    * This parameter specifies the number of bytes to read or
    * write from the slave after the register address is written.
    */
    LwU32 messageLength;
   /*!
    * The main message data. 
    * C form: LwU8 message[LW402C_CTRL_I2C_MESSAGE_LENGTH_MAX]
    */
    LW_DECLARE_ALIGNED(LwP64 pMessage, 8);
} LW402C_CTRL_I2C_TRANSACTION_DATA_READ_EDID_DDC;

/*!
 * LW402C_CTRL_I2C_TRANSACTION_DATA
 *
 * This union encapsulates the transaction data corresponding to the 
 * transaction type enlisted above.
 */
typedef union LW402C_CTRL_I2C_TRANSACTION_DATA {
       /*!
        * This transaction data is to be filled when transaction type is 
        * LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_QUICK_RW.
        */
    LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_QUICK_RW           smbusQuickData;
       /*!
        * This transaction data is to be filled when transaction type is 
        * LW402C_CTRL_I2C_TRANSACTION_TYPE_I2C_BYTE_RW.
        */
    LW402C_CTRL_I2C_TRANSACTION_DATA_I2C_BYTE_RW              i2cByteData;
       /*!
        * This transaction data is to be filled when transaction type is 
        * LW402C_CTRL_I2C_TRANSACTION_TYPE_I2C_BLOCK_RW.
        */
    LW_DECLARE_ALIGNED(LW402C_CTRL_I2C_TRANSACTION_DATA_I2C_BLOCK_RW i2cBlockData, 8);
       /*!
        * This transaction data is to be filled when transaction type is 
        * LW402C_CTRL_I2C_TRANSACTION_TYPE_I2C_BUFFER_RW.
        */
    LW_DECLARE_ALIGNED(LW402C_CTRL_I2C_TRANSACTION_DATA_I2C_BUFFER_RW i2cBufferData, 8);
       /*!
        * This transaction data is to be filled when transaction type is 
        * LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_BYTE_RW.
        */
    LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_BYTE_RW            smbusByteData;
       /*!
        * This transaction data is to be filled when transaction type is 
        * LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_WORD_RW.
        */
    LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_WORD_RW            smbusWordData;
       /*!
        * This transaction data is to be filled when transaction type is 
        * LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_BLOCK_RW.
        */
    LW_DECLARE_ALIGNED(LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_BLOCK_RW smbusBlockData, 8);
       /*!
        * This transaction data is to be filled when transaction type is 
        * LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_PROCESS_CALL.
        */
    LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_PROCESS_CALL       smbusProcessData;
       /*!
        * This transaction data is to be filled when transaction type is 
        * LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_BLOCK_PROCESS_CALL.
        */
    LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_BLOCK_PROCESS_CALL smbusBlockProcessData;
       /*!
        * This transaction data is to be filled when transaction type is 
        * LW402C_CTRL_I2C_TRANSACTION_TYPE_SMBUS_MULTIBYTE_REGISTER_BLOCK_RW.
        */
    LW_DECLARE_ALIGNED(LW402C_CTRL_I2C_TRANSACTION_DATA_SMBUS_MULTIBYTE_REGISTER_BLOCK_RW smbusMultibyteRegisterData, 8);
       /*!
        * This transaction data is to be filled when transaction type is 
        * LW402C_CTRL_I2C_TRANSACTION_TYPE_READ_EDID_DDC.
        */
    LW_DECLARE_ALIGNED(LW402C_CTRL_I2C_TRANSACTION_DATA_READ_EDID_DDC edidData, 8);
} LW402C_CTRL_I2C_TRANSACTION_DATA;


/*!
 * LW402C_CTRL_I2C_TRANSACTION_PARAMS
 *
 * The params data structure for LW402C_CTRL_CMD_I2C_TRANSACTION.
 */
#define LW402C_CTRL_I2C_TRANSACTION_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW402C_CTRL_I2C_TRANSACTION_PARAMS {
   /*!
    * The logical port ID.
    */
    LwU8                             portId;
   /*!
    * This parameter specifies optional flags used to control certain modal
    * features such as target speed and addressing mode. The lwrrently
    * defined fields are described previously; see LW402C_CTRL_I2C_FLAGS_* 
    */
    LwU32                            flags;
   /*!
    * The address of the I2C slave.
    */
    LwU16                            deviceAddress;
   /*!
    * The transaction type.
    */
    LW402C_CTRL_I2C_TRANSACTION_TYPE transType;
   /*!
    * The transaction data corresponding transaction type.
    */
    LW_DECLARE_ALIGNED(LW402C_CTRL_I2C_TRANSACTION_DATA transData, 8);
} LW402C_CTRL_I2C_TRANSACTION_PARAMS;

#define LW402C_CTRL_CMD_I2C_TRANSACTION (0x402c0105) /* finn: Evaluated from "(FINN_LW40_I2C_I2C_INTERFACE_ID << 8) | LW402C_CTRL_I2C_TRANSACTION_PARAMS_MESSAGE_ID" */


/* _ctrl402c_h_ */

