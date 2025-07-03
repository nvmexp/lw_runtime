/*
 * SPDX-FileCopyrightText: Copyright (c) 2014-2016 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080spi.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#include "ctrl/ctrl2080/ctrl2080base.h"

/* LW20_SUBDEVICE_XX spi-related control commands and parameters */

/*!
 * Macros for SPI_DEVICE types.
 */
#define LW2080_CTRL_SPI_DEVICE_TYPE_DISABLED 0x00
#define LW2080_CTRL_SPI_DEVICE_TYPE_ROM      0x01

/*!
 * Structure of static information specific to the SPI ROM device.
 */
typedef struct LW2080_CTRL_SPI_DEVICE_INFO_DATA_ROM {
    /*!
     * This checks that SPI ROM is been initialized succefully or not.
     */
    LwBool bInitialized;
} LW2080_CTRL_SPI_DEVICE_INFO_DATA_ROM;

/*!
 * Union of SPI_DEVICE type-specific data.
 */


/*!
 * Structure of static information describing a SPI_DEVICE on board.
 */
typedef struct LW2080_CTRL_SPI_DEVICE_INFO {
    /*!
     * @ref LW2080_CTRL_SPI_DEVICE_TYPE_<xyz>
     */
    LwU8 type;

    /*!
     * Type-specific information.
     */
    union {
        LW2080_CTRL_SPI_DEVICE_INFO_DATA_ROM rom;
    } data;
} LW2080_CTRL_SPI_DEVICE_INFO;

/*!
 * LW2080_CTRL_CMD_SPI_DEVICES_GET_INFO
 *
 * This command returns the static state describing the topology of SPI_DEVICEs
 * on the board.  This state primarily of the number of devices, their type and
 * initialization status.
 *
 * See @ref LW2080_CTRL_SPI_DEVICES_GET_INFO_PARAMS for documentation on the
 * parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_SPI_DEVICES_GET_INFO (0x20802b01) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_SPI_INTERFACE_ID << 8) | LW2080_CTRL_SPI_DEVICES_GET_INFO_PARAMS_MESSAGE_ID" */

/*!
 * Maximum number of SPI_DEVICES supported
 */
#define LW2080_CTRL_SPI_DEVICES_MAX_DEVICES  (1)

#define LW2080_CTRL_SPI_DEVICES_GET_INFO_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_SPI_DEVICES_GET_INFO_PARAMS {
    /*!
     * [out] - Returns the mask of valid entries in the Spi devices Table.
     * The table may contain disabled entries, but in order for indexes to work
     * correctly, we need to reserve those entries.  The mask helps in this
     * regard.
     */
    LwU32                       devMask;

    /*!
     * [out] An array (of fixed size LW2080_CTRL_SPI_DEVICES_MAX_DEVICES)
     * describing the individual SPI_DEVICES.  Has valid indexes corresponding to
     * bits set in @ref devMask.
     */
    LW2080_CTRL_SPI_DEVICE_INFO devices[LW2080_CTRL_SPI_DEVICES_MAX_DEVICES];
} LW2080_CTRL_SPI_DEVICES_GET_INFO_PARAMS;

/*
 * LW2080_CTRL_SPI_FLAGS
 *
 */
#define LW2080_CTRL_SPI_FLAGS_NONE (0x00000000)

/*!
 * Structure which defines parameters require to read buffer from SPI ROM
 * devices.
 */
typedef struct LW2080_CTRL_SPI_DEVICE_READ_BUFFER_IN_DATA_ROM {
    /*!
     * [in] Relative start address from where client wants to read
     */
    LwU32 startAddress;

    /*!
     * [in] No. of bytes Client want to read from startAddress
     */
    LwU32 sizeInBytes;
} LW2080_CTRL_SPI_DEVICE_READ_BUFFER_IN_DATA_ROM;

/*!
 * Union of all SPI DEVICEs specific input buffer for read.
 */


/*!
 * Structure which defines output of read buffer from SPI ROM devices.
 */
typedef struct LW2080_CTRL_SPI_DEVICE_READ_BUFFER_OUT_DATA_ROM {
    /*!
     * [out] Buffer which will contain the read data
     */
    LW_DECLARE_ALIGNED(LwU8 *pBuffer, 8);

    /*!
     * [out] No. of bytes devices read successfully.
     */
    LwU32 sizeInBytes;
} LW2080_CTRL_SPI_DEVICE_READ_BUFFER_OUT_DATA_ROM;

/*!
 * Union of all SPI DEVICEs specific input buffer for read.
 */


/*
 * LW2080_CTRL_CMD_SPI_DEVICE_READ_BUFFER
 *
 * This command reads buffer from SPI device.
 *
 * See @ref LW2080_CTRL_SPI_DEVICE_READ_BUFFER_PARAMS for documentation on the
 * parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_STATE_IN_USE
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_NOT_SUPPORTED
 *   <TODO> add specific command for write/read failure
 */
#define LW2080_CTRL_CMD_SPI_DEVICE_READ_BUFFER (0x20802b02) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_SPI_INTERFACE_ID << 8) | LW2080_CTRL_SPI_DEVICE_READ_BUFFER_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_SPI_DEVICE_READ_BUFFER_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW2080_CTRL_SPI_DEVICE_READ_BUFFER_PARAMS {
    /*!
     * [in] Version no. of SPI controls available.
     */
    LwU32 version;

    /*!
     * [in] Device index of SPI device. This field must be specified by the 
     * client to indicate which SPI device access is required.
     */
    LwU8  devIndex;

    /*!
     * [in] This field is specified by the client to request additional 
     * options as provided by @ref LW2080_CTRL_SPI_FLAGS_<XYZ>.
     */
    LwU32 flags;

    /*!
     * [in] Device type to select in/out data. 
     * @ref LW2080_CTRL_SPI_DEVICE_TYPE_<xyz>
     */
    LwU8  type;

    /*!
     * [in] Type-specific IN buffer information.
     */
    union {
        LW2080_CTRL_SPI_DEVICE_READ_BUFFER_IN_DATA_ROM rom;
    } in;

    /*!
     * [out] Type-specific OUT buffer information.
     */
    union {
        LW_DECLARE_ALIGNED(LW2080_CTRL_SPI_DEVICE_READ_BUFFER_OUT_DATA_ROM rom, 8);
    } out;
} LW2080_CTRL_SPI_DEVICE_READ_BUFFER_PARAMS;


/*!
 * Structure which defines parameters require to write buffer from SPI ROM
 * devices.
 */
typedef struct LW2080_CTRL_SPI_DEVICE_WRITE_BUFFER_IN_DATA_ROM {
    /*!
     * [in] Relative start address from where client wants to write
     */
    LwU32 startAddress;

    /*!
     * [in] No. of bytes Client want to write from startAddress
     */
    LwU32 sizeInBytes;

    /*!
     * [in] Buffer which will contain the write data
     */
    LW_DECLARE_ALIGNED(LwU8 *pBuffer, 8);
} LW2080_CTRL_SPI_DEVICE_WRITE_BUFFER_IN_DATA_ROM;

/*!
 * Union of all SPI DEVICEs specific input buffer for write.
 */


/*!
 * Structure which defines output of write buffer from SPI ROM devices.
 */
typedef struct LW2080_CTRL_SPI_DEVICE_WRITE_BUFFER_OUT_DATA_ROM {
    /*!
     * [out] No. of bytes written successfully.
     */
    LwU32 sizeInBytes;
} LW2080_CTRL_SPI_DEVICE_WRITE_BUFFER_OUT_DATA_ROM;

/*!
 * Union of all SPI DEVICEs specific output buffer for write.
 */


/*
 * LW2080_CTRL_CMD_SPI_DEVICE_WRITE_BUFFER
 *
 * This command write buffer from SPI device.
 *
 * See @ref LW2080_CTRL_SPI_DEVICE_WRITE_BUFFER_PARAMS for documentation on the
 * parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_STATE_IN_USE
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_NOT_SUPPORTED
 *   <TODO> add specific command for write/read failure
 */
#define LW2080_CTRL_CMD_SPI_DEVICE_WRITE_BUFFER (0x20802b03) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_SPI_INTERFACE_ID << 8) | LW2080_CTRL_SPI_DEVICE_WRITE_BUFFER_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_SPI_DEVICE_WRITE_BUFFER_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW2080_CTRL_SPI_DEVICE_WRITE_BUFFER_PARAMS {
    /*!
     * [in] Version no. of SPI controls available.
     */
    LwU32 version;

    /*!
     * [in] Device index of SPI device. This field must be specified by the 
     * client to indicate which SPI device access is required.
     */
    LwU8  devIndex;

    /*!
     * [in] This field is specified by the client to request additional 
     * options as provided by @ref LW2080_CTRL_SPI_FLAGS_<XYZ>.
     */
    LwU32 flags;

    /*!
     * [in] Device type to select in/out data. 
     * @ref LW2080_CTRL_SPI_DEVICE_TYPE_<xyz>
     */
    LwU8  type;

    /*!
     * [in] Type-specific IN buffer information.
     */
    union {
        LW_DECLARE_ALIGNED(LW2080_CTRL_SPI_DEVICE_WRITE_BUFFER_IN_DATA_ROM rom, 8);
    } in;

    /*!
     * [out] Type-specific OUT buffer information.
     */
    union {
        LW2080_CTRL_SPI_DEVICE_WRITE_BUFFER_OUT_DATA_ROM rom;
    } out;
} LW2080_CTRL_SPI_DEVICE_WRITE_BUFFER_PARAMS;

/*!
 * Structure which defines parameters require to Erase buffer from SPI ROM
 * devices.
 */
typedef struct LW2080_CTRL_SPI_DEVICE_ERASE_BUFFER_IN_DATA_ROM {
    /*!
     * [in] Relative start address from where client wants to erase
     */
    LwU32 startAddress;

    /*!
     * [in] No. of bytes Client want to erase from startAddress
     */
    LwU32 sizeInBytes;
} LW2080_CTRL_SPI_DEVICE_ERASE_BUFFER_IN_DATA_ROM;

/*!
 * Union of all SPI DEVICEs specific input buffer for Erase.
 */


/*!
 * Structure which defines output of erase buffer from SPI ROM devices.
 */
typedef struct LW2080_CTRL_SPI_DEVICE_ERASE_BUFFER_OUT_DATA_ROM {
    /*!
     * [out] No. of bytes erased successfully.
     */
    LwU32 sizeInBytes;
} LW2080_CTRL_SPI_DEVICE_ERASE_BUFFER_OUT_DATA_ROM;

/*!
 * Union of all SPI DEVICEs specific output buffer for erase.
 */


/*
 * LW2080_CTRL_CMD_SPI_DEVICE_ERASE_BUFFER
 *
 * This command erase buffer from SPI device.
 *
 * See @ref LW2080_CTRL_SPI_DEVICE_ERASE_BUFFER_PARAMS for documentation on the
 * parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_STATE_IN_USE
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_CMD_SPI_DEVICE_ERASE_BUFFER (0x20802b04) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_SPI_INTERFACE_ID << 8) | LW2080_CTRL_SPI_DEVICE_ERASE_BUFFER_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_SPI_DEVICE_ERASE_BUFFER_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW2080_CTRL_SPI_DEVICE_ERASE_BUFFER_PARAMS {
    /*!
     * [in] Version no. of SPI controls available.
     */
    LwU32 version;

    /*!
     * [in] Device index of SPI device. This field must be specified by the 
     * client to indicate which SPI device access is required.
     */
    LwU8  devIndex;

    /*!
     * [in] This field is specified by the client to request additional 
     * options as provided by @ref LW2080_CTRL_SPI_FLAGS_<XYZ>.
     */
    LwU32 flags;

    /*!
     * [in] Device type to select in/out data. 
     * @ref LW2080_CTRL_SPI_DEVICE_TYPE_<xyz>
     */
    LwU8  type;

    /*!
     * [in] Type-specific IN buffer information.
     */
    union {
        LW2080_CTRL_SPI_DEVICE_ERASE_BUFFER_IN_DATA_ROM rom;
    } in;

    /*!
     * [out] Type-specific OUT buffer information.
     */
    union {
        LW2080_CTRL_SPI_DEVICE_ERASE_BUFFER_OUT_DATA_ROM rom;
    } out;
} LW2080_CTRL_SPI_DEVICE_ERASE_BUFFER_PARAMS;

/* _ctrl2080spi_h_ */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

