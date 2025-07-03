/*
 * SPDX-FileCopyrightText: Copyright (c) 2007-2015 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl2080/ctrl2080gpio.finn
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

/*
 * LW2080_CTRL_GPIO_ACCESS_PORT_MIN
 * LW2080_CTRL_GPIO_ACCESS_PORT_MAX
 * LW2080_CTRL_GPIO_ACCESS_PORT_COUNT_MAX
 *
 * These values represent the maximum number of pins and the legal minimum and maximum GPIO
 * pin numbers for use with the LW2080_CTRL_CMD_GPIO_READ and LW2080_CTRL_CMD_GPIO_WRITE
 * commands.
 * The actual maximum pin number supported is GPU-dependent but is guaranteed to be no more
 * than LW2080_CTRL_GPIO_ACCESS_PORT_MAX.
 */
#define LW2080_CTRL_GPIO_ACCESS_PORT_MIN            (0x00000000)
#define LW2080_CTRL_GPIO_ACCESS_PORT_MAX            (0x0000001F)
#define LW2080_CTRL_GPIO_ACCESS_PORT_COUNT_MAX      (0x20) /* finn: Evaluated from "(LW2080_CTRL_GPIO_ACCESS_PORT_MAX - LW2080_CTRL_GPIO_ACCESS_PORT_MIN + 1)" */


/*
 * LW2080_CTRL_GPIO_DIRECTION
 *
 * These two values represent the possible GPIO direction configurations.
 * LW2080_CTRL_GPIO_DIRECTION_INPUT indicates that the associated GPIO pin
 * is configured as an input.  LW2080_CTRL_GPIO_DIRECTION_OUTPUT indicates
 * that the associated GPIO pin is configured as an output.
 */
#define LW2080_CTRL_GPIO_DIRECTION_INPUT            (0x00000000)
#define LW2080_CTRL_GPIO_DIRECTION_OUTPUT           (0x00000001)

/*
 * LW2080_CTRL_CMD_GPIO_LWSTOMER_ASYNCRW_QUERY
 *
 * This command will return the number of available gpio customer asyncrw instance.
 *
 * gpioPinCount
 *   This parameter specifies the number of GPIO pins associated with the LWSTOMER_ASYNCRW
 *   function.
 *
 * gpioPins
 *   This parameter holds the list of GPIO pin numbers associated with LWSTOMER_ASYNCRW
 *   function. These GPIO pin numbers can be used with the READ and WRITE commands to
 *   initiate operations to specific GPIO pins.
 *
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *
 */

#define LW2080_CTRL_CMD_GPIO_LWSTOMER_ASYNCRW_QUERY (0x20802304) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPIO_INTERFACE_ID << 8) | LW2080_CTRL_GPIO_LWSTOMER_ASYNCRW_QUERY_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPIO_LWSTOMER_ASYNCRW_QUERY_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW2080_CTRL_GPIO_LWSTOMER_ASYNCRW_QUERY_PARAMS {
    LwU32 gpioPinCount;
    LwU32 gpioPins[LW2080_CTRL_GPIO_ACCESS_PORT_COUNT_MAX];
} LW2080_CTRL_GPIO_LWSTOMER_ASYNCRW_QUERY_PARAMS;

/*
 * LW2080_CTRL_CMD_GPIO_LWSTOMER_ASYNCRW_READ
 *
 * This command can be used to read GPIO data to the specified GPIO pin number.
 * Each write request accepts a single 32 bit unsigned integer value.
 *
 * gpioPin
 *   This parameter specifies the GPIO pin number to which the write data is to be
 *   sent.  Legal values for this parameter must be fetched by
 *   LW2080_CTRL_CMD_GPIO_LWSTOMER_ASYNCRW_QUERY command.
 *
 * gpioData
 *   This parameter returns the read data from the specified GPIO pin.
 *
 * reserved00
 *   This parameter is reserved for future use.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_NOT_SUPPORTED
 *
 */

#define LW2080_CTRL_CMD_GPIO_LWSTOMER_ASYNCRW_READ (0x20802305) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPIO_INTERFACE_ID << 8) | LW2080_CTRL_GPIO_LWSTOMER_ASYNCRW_READ_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPIO_LWSTOMER_ASYNCRW_READ_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW2080_CTRL_GPIO_LWSTOMER_ASYNCRW_READ_PARAMS {
    LwU32 gpioPin;
    LwU32 gpioDirection;
    LwU32 gpioData;
    LwU32 reserved00[2];
} LW2080_CTRL_GPIO_LWSTOMER_ASYNCRW_READ_PARAMS;

/*
 * LW2080_CTRL_CMD_GPIO_LWSTOMER_ASYNCRW_WRITE
 *
 * This command can be used to write GPIO data to the specified GPIO pin number.
 * Each write request accepts a single 32 bit unsigned integer value.
 *
 * gpioPin
 *   This parameter specifies the GPIO pin number to which the write data is to be
 *   sent.  Legal values for this parameter must be fetched by
 *   LW2080_CTRL_CMD_GPIO_LWSTOMER_ASYNCRW_QUERY command.
 *
 * gpioData
 *   This parameter returns the read data from the specified GPIO pin.
 *
 * reserved00
 *   This parameter is reserved for future use.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_NOT_SUPPORTED
 *
 */

#define LW2080_CTRL_CMD_GPIO_LWSTOMER_ASYNCRW_WRITE (0x20802306) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPIO_INTERFACE_ID << 8) | LW2080_CTRL_GPIO_LWSTOMER_ASYNCRW_WRITE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPIO_LWSTOMER_ASYNCRW_WRITE_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW2080_CTRL_GPIO_LWSTOMER_ASYNCRW_WRITE_PARAMS {
    LwU32 gpioPin;
    LwU32 gpioDirection;
    LwU32 gpioData;
    LwU32 reserved00[2];
} LW2080_CTRL_GPIO_LWSTOMER_ASYNCRW_WRITE_PARAMS;


/*
 * LW2080_CTRL_CMD_GPIO_GET_CAPABILITIES
 *
 * This command can be used to get the cpapbilities of gpio hw.
 *
 * numGpioPins
 *   This parameter specifies the number of GPIO pins numbers
 * numDcbEntries
 *   This parameter specifies the num of DCB entries.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_NOT_SUPPORTED
 *
 */
#define LW2080_CTRL_CMD_GPIO_GET_CAPABILITIES (0x20802307) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPIO_INTERFACE_ID << 8) | LW2080_CTRL_GPIO_GET_CAPABILITIES_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPIO_GET_CAPABILITIES_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW2080_CTRL_GPIO_GET_CAPABILITIES_PARAMS {
    LwU32 numGpioPins;
    LwU32 numDcbEntries;
} LW2080_CTRL_GPIO_GET_CAPABILITIES_PARAMS;

/*
 * LW2080_CTRL_CMD_GPIO_READ_DCB_ENTRIES
 *
 * This command can be used to read the sw cache of gpio DCB entries.
 *
 * pinMaskIn
 *   This parameter specifies the GPIO pin numbers(corresponding to the bits set)
 *   for which we are requesting the DCB entries.
 * pinMaskOut
 *   This parameter specifies the pin numbers for which we could read the DCB entries.
 * dcbEntries
 *   This parameter is the pointer to the buffer of entries being passed in.
 * size
 *   This parameter specifies the number of entries that the buffer can hold.
 * numRead
 *   This parameter specifies the number of entries that the buffer holds post fetch.
 * Uasge note: The size of buffer should be large enough to hold all the requested enties.
 *             The buffer can be filled in any order wrt to the pin number.
 *             Please refer to the gpioPin param in the LW2080_CTRL_GPIO_DCB_ENTRY struct to find out
 *             which pin number the entry corresponds to.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_NOT_SUPPORTED
 *
 */
#define LW2080_CTRL_CMD_GPIO_READ_DCB_ENTRIES (0x20802308) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPIO_INTERFACE_ID << 8) | LW2080_CTRL_GPIO_READ_DCB_ENTRIES_PARAMS_MESSAGE_ID" */

typedef struct LW2080_CTRL_GPIO_DCB_ENTRY {
    LwU8  GpioFunction;
    LwU8  OffData;
    LwU8  OffEnable;
    LwU8  OnData;
    LwU8  OnEnable;
    LwU8  PWM;
    LwU8  Mode;              //only for pre-gf11x
    LwU8  OutputHwEnum;      //only for gf11x+
    LwU8  InputHwEnum;       //only for gf11x+
    LwU8  Init;
    LwU32 GpioPin;
} LW2080_CTRL_GPIO_DCB_ENTRY;

#define LW2080_CTRL_GPIO_READ_DCB_ENTRIES_PARAMS_MESSAGE_ID (0x8U)

typedef struct LW2080_CTRL_GPIO_READ_DCB_ENTRIES_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 pinMaskIn, 8);
    LW_DECLARE_ALIGNED(LwU64 pinMaskOut, 8);
    LW_DECLARE_ALIGNED(LwP64 dcbEntries, 8);
    LwU32 size;
    LwU32 numRead;
} LW2080_CTRL_GPIO_READ_DCB_ENTRIES_PARAMS;

#define LW2080_CTRL_GPIO_DCB_ENTRY_GPIO_NOT_VALID              0x000000ff

#define LW2080_CTRL_GPIO_DCB_ENTRY_MODE_NORMAL                 0
#define LW2080_CTRL_GPIO_DCB_ENTRY_MODE_ALT                    1
#define LW2080_CTRL_GPIO_DCB_ENTRY_MODE_SEQ                    2

#define LW2080_CTRL_GPIO_DCB_ENTRY_PWM_NO                      0
#define LW2080_CTRL_GPIO_DCB_ENTRY_PWM_YES                     1

#define LW2080_CTRL_GPIO_DCB_ENTRY_OFF_DATA_LOW                0
#define LW2080_CTRL_GPIO_DCB_ENTRY_OFF_DATA_HIGH               1

#define LW2080_CTRL_GPIO_DCB_ENTRY_OFF_ENABLE_NO               0
#define LW2080_CTRL_GPIO_DCB_ENTRY_OFF_ENABLE_YES              1

#define LW2080_CTRL_GPIO_DCB_ENTRY_ON_DATA_LOW                 0
#define LW2080_CTRL_GPIO_DCB_ENTRY_ON_DATA_HIGH                1

#define LW2080_CTRL_GPIO_DCB_ENTRY_ON_ENABLE_NO                0
#define LW2080_CTRL_GPIO_DCB_ENTRY_ON_ENABLE_YES               1

#define LW2080_CTRL_GPIO_DCB_ENTRY_INIT_NO                     0
#define LW2080_CTRL_GPIO_DCB_ENTRY_INIT_YES                    1

// One of the gpio virtual functions can be used for debug purposes.
#define LW2080_CTRL_GPIO_DCB_ENTRY_DEBUG_FUNCTION              0x000000DE

#define LW2080_CTRL_GPIO_DCB_ENTRY_OUT_ENUM_NORMAL             0x00000000
#define LW2080_CTRL_GPIO_DCB_ENTRY_OUT_ENUM_FAN_ALERT          0x00000059
#define LW2080_CTRL_GPIO_DCB_ENTRY_OUT_ENUM_PWM_OUTPUT         0x0000005C

#define LW2080_CTRL_GPIO_DCB_ENTRY_IN_ENUM_UNASSIGNED          0x0
#define LW2080_CTRL_GPIO_DCB_ENTRY_IN_ENUM_AUX_HPD(i)               ((i)+1)
#define LW2080_CTRL_GPIO_DCB_ENTRY_IN_ENUM_AUX_HPD__SIZE_1     4
#define LW2080_CTRL_GPIO_DCB_ENTRY_IN_ENUM_RASTER_SYNC(i)           ((i)+9)
#define LW2080_CTRL_GPIO_DCB_ENTRY_IN_ENUM_RASTER_SYNC__SIZE_1 4
#define LW2080_CTRL_GPIO_DCB_ENTRY_IN_ENUM_TACH                24

/*
 * LW2080_CTRL_CMD_GPIO_WRITE_DCB_ENTRY
 *
 * This command can be used to write a DCB entry for a given pin in sw cache.
 * This command can potentially override a given DCB entry in sw cache. Use carefully!
 *
 * dcbEntry
 *   This parameter specifies the DCB entry param.
 *
 * commitToHw
 *   This parameter specifies whether the overridden entry needs to be committed to hw.(pin reprogramming)
 *
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_NOT_SUPPORTED
 *
 */
#define LW2080_CTRL_CMD_GPIO_WRITE_DCB_ENTRY                   (0x20802309) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPIO_INTERFACE_ID << 8) | LW2080_CTRL_GPIO_WRITE_DCB_ENTRY_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPIO_WRITE_DCB_ENTRY_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW2080_CTRL_GPIO_WRITE_DCB_ENTRY_PARAMS {
    LW2080_CTRL_GPIO_DCB_ENTRY dcbEntry;
    LwBool                     commitToHw;
} LW2080_CTRL_GPIO_WRITE_DCB_ENTRY_PARAMS;



/*
 * LW2080_CTRL_CMD_GPIO_TYPE
 *
 * These defines allow for internal drivers to access specific GPIO functions
 * and control those GPIOs via reading their input, or writing their output.
 * Current Values are:
 *   LW2080_CTRL_CMD_GPIO_TYPE_CAMERA_FRONT_SELECT
 *     This type is used to select the input data stream to our chip from
 *     either a front-facing camera or a back-facing camera.
 *
 */
#define LW2080_CTRL_CMD_GPIO_TYPE_CAMERA_FRONT_SELECT 0x00000001


/*
 * LW2080_CTRL_CMD_GPIO_INTERNAL_QUERY
 *
 * This command will return the number of available gpios that can be used for
 * Lwpu drivers only.  Do not expose this API to end users!
 *
 * gpioCount
 *   This parameter specifies the number of accessible internal GPIO pins
 *
 * gpioTypes
 *   This parameter holds the list of LW2080_CTRL_CMD_GPIO_INTERNAL_TYPE that
 *   can be accessed through these internal functions. These GPIO pin types
 *   can be used with the READ and WRITE commands to initiate operations
 *   to specific GPIO pins.
 *
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *
 */

#define LW2080_CTRL_CMD_GPIO_INTERNAL_QUERY           (0x2080230a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPIO_INTERFACE_ID << 8) | LW2080_CTRL_GPIO_INTERNAL_QUERY_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPIO_INTERNAL_QUERY_PARAMS_MESSAGE_ID (0xAU)

typedef struct LW2080_CTRL_GPIO_INTERNAL_QUERY_PARAMS {
    LwU32 gpioCount;
    LwU32 gpioTypes[LW2080_CTRL_GPIO_ACCESS_PORT_COUNT_MAX];
} LW2080_CTRL_GPIO_INTERNAL_QUERY_PARAMS;

/*
 * LW2080_CTRL_CMD_GPIO_INTERNAL_WRITE
 *
 * This command can be used to output GPIO data to a specific GPIO type.
 * Do not expose this API to end users!
 *
 * gpioType
 *   This parameter specifies the LW2080_CTRL_CMD_GPIO_INTERNAL_TYPE to write.
 *
 * gpioOutput
 *   This parameter tells our driver to output a 1 or 0.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_NOT_SUPPORTED
 *
 */

#define LW2080_CTRL_CMD_GPIO_INTERNAL_WRITE (0x2080230b) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPIO_INTERFACE_ID << 8) | LW2080_CTRL_GPIO_INTERNAL_WRITE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPIO_INTERNAL_WRITE_PARAMS_MESSAGE_ID (0xBU)

typedef struct LW2080_CTRL_GPIO_INTERNAL_WRITE_PARAMS {
    LwU32  gpioType;
    LwBool gpioOutput;
} LW2080_CTRL_GPIO_INTERNAL_WRITE_PARAMS;

/*
 * LW2080_CTRL_CMD_GPIO_PROCESS_HOTPLUG
 *
 * This command is used to ask RM/GPIO to process a pending hotplug. This call
 * will only process hotplug and skip the other pending GPIO interrupts
 *
 * intrStatus
 *      This will return true if there are any other pending interrupts
 *
 */

#define LW2080_CTRL_CMD_GPIO_PROCESS_HOTPLUG (0x2080230c) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GPIO_INTERFACE_ID << 8) | LW2080_CTRL_GPIO_PROCESS_HOTPLUG_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GPIO_PROCESS_HOTPLUG_PARAMS_MESSAGE_ID (0xLW)

typedef struct LW2080_CTRL_GPIO_PROCESS_HOTPLUG_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 intrStatus, 8);
} LW2080_CTRL_GPIO_PROCESS_HOTPLUG_PARAMS;

/* _ctrl2080gpio_h_ */

#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

