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
// Source file: ctrl/ctrl2080/ctrl2080thermal.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



#include "lwfixedtypes.h"
#include "ctrl/ctrl2080/ctrl2080base.h"
#include "ctrl/ctrl2080/ctrl2080boardobj.h"
#include "ctrl/ctrl2080/ctrl2080pmgr.h"
#include "ctrl/ctrl2080/ctrl2080pmumon.h"

/* LW20_SUBDEVICE_XX thermal control commands and parameters */

/*
 * Thermal System rmcontrol api versioning
 */
#define THERMAL_SYSTEM_API_VER                    1U
#define THERMAL_SYSTEM_API_REV                    0U

/*
 * Cooler rmcontrol api versioning
 */
#define THERMAL_COOLER_API_VER                    1U
#define THERMAL_COOLER_API_REV                    0U



/*
 *
 * LW2080_CTRL_THERMAL
 *
 * Thermal system access/control functionality.
 *
 */



/*
 * LW2080_CTRL_THERMAL_TABLE
 *
 * Thermal table functionality
 */


/*
 * LW2080_CTRL_THERMAL_TABLE_VERSION
 *
 * The thermal table versions listed here are those that are supported by
 * the current driver.
 *
 * Versions returned or set by the control functions are the actual version
 * stored in the thermal table.
 *
 * Versions not listed in the following list can be used, but the driver may
 * not implement the various features implied by the version number.
 */
#define LW2080_CTRL_THERMAL_TABLE_VERSION_2_0     0x20U
#define LW2080_CTRL_THERMAL_TABLE_VERSION_2_1     0x21U
#define LW2080_CTRL_THERMAL_TABLE_VERSION_2_2     0x22U
#define LW2080_CTRL_THERMAL_TABLE_VERSION_2_3     0x23U


/*
 * LW2080_CTRL_THERMAL_TABLE_MAX_ENTRIES
 *
 * The maximum number of thermal tables entries supported by this interface.
 *
 * Sync this up with one in lwapi.spec! (LWAPI_GPU_THERMAL_TABLE_MAX_ENTRIES)
 */
#define LW2080_CTRL_THERMAL_TABLE_MAX_ENTRIES     256U


/*
 * LW2080_CTRL_CMD_THERMAL_GET_THERMAL_TABLE
 *
 *   flags
 *       This field is specified by the client to request additional options
 *       as provided by LW2080_CTRL_THERMAL_GET_THERMAL_TABLE_FLAGS.
 *   version
 *       This field is returned to the client and indicates the current
 *       thermal table version.
 *   entrycount
 *       This field specifies the total # of elements contained in entries
 *       field.
 *   entries
 *       This field contains the actual thermal table entries.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_THERMAL_GET_THERMAL_TABLE (0x20800501U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | LW2080_CTRL_THERMAL_GET_THERMAL_TABLE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_THERMAL_GET_THERMAL_TABLE_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_THERMAL_GET_THERMAL_TABLE_PARAMS {
    LwU32 flags;
    LwU32 version;
    LwU32 entrycount;
    LwU32 entries[LW2080_CTRL_THERMAL_TABLE_MAX_ENTRIES];
} LW2080_CTRL_THERMAL_GET_THERMAL_TABLE_PARAMS;


/*
 * LW2080_CTRL_THERMAL_GET_THERMAL_TABLE_FLAGS
 *
 * The following flags select the thermal table to target:
 *
 *   LW2080_CTRL_THERMAL_GET_THERMAL_TABLE_FLAGS_VBIOS
 *       Returns the thermal table stored/derived from the vbios or firmware
 *
 *   LW2080_CTRL_THERMAL_GET_THERMAL_TABLE_FLAGS_LWRRENT
 *       Returns the thermal table lwrrently in use
 *
 *   LW2080_CTRL_THERMAL_GET_THERMAL_TABLE_FLAGS_REGISTRY
 *       Returns the thermal table stored in the registry
 */
#define LW2080_CTRL_THERMAL_GET_THERMAL_TABLE_FLAGS_VBIOS            (0x00000001U)
#define LW2080_CTRL_THERMAL_GET_THERMAL_TABLE_FLAGS_LWRRENT          (0x00000002U)
#define LW2080_CTRL_THERMAL_GET_THERMAL_TABLE_FLAGS_REGISTRY         (0x00000004U)

/*
 * LW2080_CTRL_THERMAL_SYSTEM api
 *
 *  The thermal system interfaces provide access to the thermal support
 *  in the LWPU driver.
 *
 *  The main sections of the thermal system api consist of the following:
 *
 *   LW2080_CTRL_THERMAL_SYSTEM constants
 *    A set of constants used to interact with the thermal system api.
 *    Typical constants include providers, targets, etc.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM rmcontrol calls
 *    A set of rmcontrol calls used to interact with the
 *    thermal system through the driver.
 *
 */


/*
 * LW2080_CTRL_THERMAL_SYSTEM constants
 *
 */

/*
 * LW2080_CTRL_THERMAL_SYSTEM_TARGET
 *
 *  Targets (ie the things the thermal system can observe). Target mask
 *  have to be in sync with corresponding element of LWAPI_THERMAL_TARGET
 *  enum, until there is a translation layer between these two.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_TARGET_NONE
 *       There is no target.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_TARGET_GPU
 *       The GPU is the target.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_TARGET_MEMORY
 *       The memory is the target.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_TARGET_POWER_SUPPLY
 *       The power supply is the target.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_TARGET_BOARD
 *       The board (PCB) is the target.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_TARGET_VTSLAVE<1|2|3|4>
 *       VT1165 slave is the target
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_TARGET_UNKNOWN
 *       The target is unknown.
 */
#define LW2080_CTRL_THERMAL_SYSTEM_TARGET_NONE                       (0x00000000U)
#define LW2080_CTRL_THERMAL_SYSTEM_TARGET_GPU                        (0x00000001U)
#define LW2080_CTRL_THERMAL_SYSTEM_TARGET_MEMORY                     (0x00000002U)
#define LW2080_CTRL_THERMAL_SYSTEM_TARGET_POWER_SUPPLY               (0x00000004U)
#define LW2080_CTRL_THERMAL_SYSTEM_TARGET_BOARD                      (0x00000008U)
#define LW2080_CTRL_THERMAL_SYSTEM_TARGET_VTSLAVE1                   (0x00000010U)
#define LW2080_CTRL_THERMAL_SYSTEM_TARGET_VTSLAVE2                   (0x00000020U)
#define LW2080_CTRL_THERMAL_SYSTEM_TARGET_VTSLAVE3                   (0x00000040U)
#define LW2080_CTRL_THERMAL_SYSTEM_TARGET_VTSLAVE4                   (0x00000080U)
#define LW2080_CTRL_THERMAL_SYSTEM_TARGET_UNKNOWN                    (0xFFFFFFFFU)

/*
 * LW2080_CTRL_THERMAL_SYSTEM_SENSOR_IMPLEMENTATION
 *
 *  A sensor's physical (mechanical) implementation.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_SENSOR_IMPLEMENTATION_NONE
 *       This sensor has no physical implementation.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_SENSOR_IMPLEMENTATION_DIODE
 *       This sensor is a diode (band gap).
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_SENSOR_IMPLEMENTATION_UNKNOWN
 *       This sensor has an unknown implementation.
 */
#define LW2080_CTRL_THERMAL_SYSTEM_SENSOR_IMPLEMENTATION_NONE        (0x00000000U)
#define LW2080_CTRL_THERMAL_SYSTEM_SENSOR_IMPLEMENTATION_DIODE       (0x00000001U)
#define LW2080_CTRL_THERMAL_SYSTEM_SENSOR_IMPLEMENTATION_UNKNOWN     (0xFFFFFFFFU)

/*
 * LW2080_CTRL_THERMAL_SYSTEM_SENSOR_LOCATION
 *
 *  Sensor location relative to a target. In otherwords,
 *  is this sensor internal (inside) or external to (outside) the target.
 *  This distinction can be important in the determining the trustworthiness
 *  of the sensor readings.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_SENSOR_LOCATION_NONE
 *       This sensor has no location.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_SENSOR_LOCATION_INTERNAL
 *       This sensor is inside the target.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_SENSOR_LOCATION_EXTERNAL
 *       This sensor is outside the target.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_SENSOR_LOCATION_UNKNOWN
 *       This sensor's location is unknown.
 */
#define LW2080_CTRL_THERMAL_SYSTEM_SENSOR_LOCATION_NONE              (0x00000000U)
#define LW2080_CTRL_THERMAL_SYSTEM_SENSOR_LOCATION_INTERNAL          (0x00000001U)
#define LW2080_CTRL_THERMAL_SYSTEM_SENSOR_LOCATION_EXTERNAL          (0x00000002U)
#define LW2080_CTRL_THERMAL_SYSTEM_SENSOR_LOCATION_UNKNOWN           (0xFFFFFFFFU)

/*
 * LW2080_CTRL_THERMAL_SYSTEM_PROVIDER
 *
 *  Types of providers (agents that provide sensors, controls, etc.)
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_NONE
 *       There is no provider.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_GPU_INTERNAL
 *       This provider is internal to the GPU.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_ADM1032
 *       This provider is an ADM1023 device.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_ADT7461
 *       This provider is an ADT7461 device.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_MAX6649
 *       This provider is a MAX6649 device.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_MAX1617
 *       This provider is an MAX1617 device.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_LM99
 *       This provider is an LM99 device.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_LM89
 *       This provider is an LM89 device.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_LM64
 *       This provider is an LM64 device.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_G781
 *       This provider is a G781 device.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_ADT7473
 *       This provider is an ADT7473 device.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_SBMAX6649
 *       This provider is an SBMAX6649 device.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_VBIOSEVT
 *       This provider is a VBIOSEVT device.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_OS
 *       This provider is the OS.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_UNKNOWN
 *       This provider is unknown.
 *   LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_LWSYSON_E551
 *       This provider is an LWPU E551 MXM SLI interposer system controller.
 */
#define LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_NONE                     (0x00000000U)
#define LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_GPU_INTERNAL             (0x00000001U)
#define LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_ADM1032                  (0x00000002U)
#define LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_ADT7461                  (0x00000003U)
#define LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_MAX6649                  (0x00000004U)
#define LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_MAX1617                  (0x00000005U)
#define LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_LM99                     (0x00000006U)
#define LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_LM89                     (0x00000007U)
#define LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_LM64                     (0x00000008U)
#define LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_G781                     (0x00000009U)
#define LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_ADT7473                  (0x0000000AU)
#define LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_SBMAX6649                (0x0000000BU)
#define LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_VBIOSEVT                 (0x0000000LW)    // Deprecated
#define LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_OS                       (0x0000000DU)    // Deprecated
#define LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_MAX6649R                 (0x00000010U)
#define LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_ADT7473S                 (0x00000011U)
#define LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_UNKNOWN                  (0xFFFFFFFFU)


/*
 * LW2080_CTRL_THERMAL_SYSTEM api
 *
 *  The thermal system interfaces provide access to the thermal support
 *  in the LWPU driver.
 *
 *  Lwrrently, the interface exports the following rmcontrol calls:
 *
 *  LW2080_CTRL_CMD_THERMAL_SYSTEM_GET_VERSION_INFO
 *   Gets the version of the thermal system interface in use by the driver.
 *
 *  LW2080_CTRL_CMD_THERMAL_SYSTEM_GET_INSTRUCTION_SIZE
 *   Gets the thermal system instruction size in use by the driver.
 *
 *  LW2080_CTRL_CMD_THERMAL_SYSTEM_EXELWTE
 *   Exelwtes a set of thermal system instructions provided by the client.
 */

#define LW2080_CTRL_CMD_THERMAL_GET_LIMITS_LIMITMASK_HWOVERTEMP      (0x00000001U)

/*
 * LW2080_CTRL_CMD_THERMAL_SYSTEM_EXELWTE
 *
 * This command will execute a list of thermal system instructions:
 *
 *   clientAPIVersion
 *       This field must be set by the client to THERMAL_SYSTEM_API_VER,
 *       which allows the driver to determine api compatibility.
 *
 *   clientAPIRevision
 *       This field must be set by the client to THERMAL_SYSTEM_API_REV,
 *       which allows the driver to determine api compatibility.
 *
 *   clientInstructionSizeOf
 *       This field must be set by the client to
 *       sizeof(LW2080_CTRL_THERMAL_SYSTEM_INSTRUCTION), which allows the
 *       driver to determine api compatibility.
 *
 *   exelwteFlags
 *       This field is set by the client to control instruction exelwtion.
 *        LW2080_CTRL_THERMAL_SYSTEM_EXELWTE_FLAGS_DEFAULT
 *         Execute instructions normally. The first instruction
 *         failure will cause exelwtion to stop.
 *        LW2080_CTRL_THERMAL_SYSTEM_EXELWTE_FLAGS_IGNORE_FAIL
 *         Execute all instructions, ignoring individual instruction failures.
 *
 *   successfulInstructions
 *       This field is set by the driver and is the number of instructions
 *       that returned LW_OK on exelwtion.  If this field
 *       matches instructionListSize, all instructions exelwted successfully.
 *
 *   instructionListSize
 *       This field is set by the client to the number of instructions in
 *       instruction list.
 *
 *   instructionList
 *       This field is set by the client to point to an array of thermal system
 *       instructions (LW2080_CTRL_THERMAL_SYSTEM_INSTRUCTION) to execute.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */

/*
 * exelwteFlags values
 */
#define LW2080_CTRL_THERMAL_SYSTEM_EXELWTE_FLAGS_DEFAULT             (0x00000000U)
#define LW2080_CTRL_THERMAL_SYSTEM_EXELWTE_FLAGS_IGNORE_FAIL         (0x00000001U)
#define LW2080_CTRL_THERMAL_SYSTEM_EXELWTE_FLAGS_INTERNAL                  1:1
#define LW2080_CTRL_THERMAL_SYSTEM_EXELWTE_FLAGS_INTERNAL_FALSE      0x00000000U
#define LW2080_CTRL_THERMAL_SYSTEM_EXELWTE_FLAGS_INTERNAL_TRUE       0x00000001U



/*
 * LW2080_CTRL_THERMAL_SYSTEM_GET_INFO instructions...
 *
 */

/*
 * LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGETS_AVAILABLE instruction
 *
 * Get the number of available targets.
 *
 *   availableTargets
 *       Returns the number of available targets.  Targets are
 *       identified by an index, starting with 0.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGETS_AVAILABLE_OPCODE (0x00000100U)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGETS_AVAILABLE_OPERANDS {
    LwU32 availableTargets;
} LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGETS_AVAILABLE_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGET_TYPE instruction
 *
 * Get a target's type.
 *
 *   targetIndex
 *       Set by the client to the desired target index.
 *
 *   type
 *       Returns a target's type.
 *       Possible values returned are:
 *          LW2080_CTRL_THERMAL_SYSTEM_TARGET_NONE
 *          LW2080_CTRL_THERMAL_SYSTEM_TARGET_GPU
 *          LW2080_CTRL_THERMAL_SYSTEM_TARGET_MEMORY
 *          LW2080_CTRL_THERMAL_SYSTEM_TARGET_POWER_SUPPLY
 *          LW2080_CTRL_THERMAL_SYSTEM_TARGET_BOARD
 *          LW2080_CTRL_THERMAL_SYSTEM_TARGET_UNKNOWN
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGET_TYPE_OPCODE (0x00000101U)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGET_TYPE_OPERANDS {
    LwU32 targetIndex;
    LwU32 type;
} LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGET_TYPE_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGET_SENSORS instruction
 *
 * Get a target's group of related sensors.
 *
 *   targetIndex
 *       Set by the client to the desired target index.
 *
 *   sensors
 *       Returns a target's group of related sensors, in mask form.
 *       A bit position represents a sensor index.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGET_SENSORS_OPCODE (0x00000110U)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGET_SENSORS_OPERANDS {
    LwU32 targetIndex;
    LwU32 sensors;
} LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGET_SENSORS_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGET_PROVIDERS instruction
 *
 * Get a target's group of related providers.
 *
 *   targetIndex
 *       Set by the client to the desired target index.
 *
 *   providers
 *       Returns a target's group of related providers, in mask form.
 *       A bit position represents a provider index.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGET_PROVIDERS_OPCODE (0x00000120U)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGET_PROVIDERS_OPERANDS {
    LwU32 targetIndex;
    LwU32 providers;
} LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGET_PROVIDERS_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDERS_AVAILABLE instruction
 *
 * Get the number of available providers.
 *
 *   availableProviders
 *       Returns the number of available providers.  Providers are
 *       identified by an index, starting with 0.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDERS_AVAILABLE_OPCODE (0x00000300U)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDERS_AVAILABLE_OPERANDS {
    LwU32 availableProviders;
} LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDERS_AVAILABLE_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDER_TYPE instruction
 *
 * Get a providers's type.
 *
 *   providerIndex
 *       Set by the client to the desired provider index.
 *
 *   type
 *       Returns a provider's type.
 *       Possible values returned are:
 *          LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_NONE
 *          LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_GPU_INTERNAL
 *          LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_ADM1032
 *          LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_MAX6649
 *          LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_MAX1617
 *          LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_LM99
 *          LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_LM89
 *          LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_LM64
 *          LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_ADT7473
 *          LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_SBMAX6649
 *          LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_VBIOSEVT
 *          LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_OS
 *          LW2080_CTRL_THERMAL_SYSTEM_PROVIDER_UNKNOWN
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDER_TYPE_OPCODE (0x00000301U)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDER_TYPE_OPERANDS {
    LwU32 providerIndex;
    LwU32 type;
} LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDER_TYPE_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDER_TARGETS instruction
 *
 * Get a provider's related targets.
 *
 *   providerIndex
 *       Set by the client to the desired provider index.
 *
 *   targets
 *       Returns a provider's related targets, in mask form.
 *       A bit position represents a target index.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDER_TARGETS_OPCODE (0x00001310U)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDER_TARGETS_OPERANDS {
    LwU32 providerIndex;
    LwU32 targets;
} LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDER_TARGETS_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDER_SENSORS instruction
 *
 * Get a provider's related sensors.
 *
 *   providerIndex
 *       Set by the client to the desired provider index.
 *
 *   targets
 *       Returns a provider's related sensors, in mask form.
 *       A bit position represents a sensor index.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDER_SENSORS_OPCODE (0x00001320U)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDER_SENSORS_OPERANDS {
    LwU32 providerIndex;
    LwU32 sensors;
} LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDER_SENSORS_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSORS_AVAILABLE instruction
 *
 * Get the number of available sensors.
 *
 *   availableSensors
 *       Returns the number of available sensors.  Sensors are
 *       identified by an index, starting with 0.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSORS_AVAILABLE_OPCODE (0x00000500U)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSORS_AVAILABLE_OPERANDS {
    LwU32 availableSensors;
} LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSORS_AVAILABLE_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_PROVIDER instruction
 *
 * Get a sensor's provider index.
 *
 *   sensorIndex
 *       Set by the client to the desired sensor index.
 *
 *   providerIndex
 *       Returns a sensor's provider index.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_PROVIDER_OPCODE (0x00000510U)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_PROVIDER_OPERANDS {
    LwU32 sensorIndex;
    LwU32 providerIndex;
} LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_PROVIDER_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_TARGET instruction
 *
 * Get a sensor's target index.
 *
 *   sensorIndex
 *       Set by the client to the desired sensor index.
 *
 *   targetIndex
 *       Returns a sensor's target index.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_TARGET_OPCODE (0x00000520U)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_TARGET_OPERANDS {
    LwU32 sensorIndex;
    LwU32 targetIndex;
} LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_TARGET_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_IMPLEMENTATION instruction
 *
 * Get a sensor's implementation.
 *
 *   sensorIndex
 *       Set by the client to the desired sensor index.
 *
 *   physical
 *       Returns a sensor's implementation.
 *       Possible values returned are:
 *          LW2080_CTRL_THERMAL_SYSTEM_SENSOR_IMPLEMENTATION_NONE
 *          LW2080_CTRL_THERMAL_SYSTEM_SENSOR_IMPLEMENTATION_DIODE
 *          LW2080_CTRL_THERMAL_SYSTEM_SENSOR_IMPLEMENTATION_UNKNOWN
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_IMPLEMENTATION_OPCODE (0x00000530U)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_IMPLEMENTATION_OPERANDS {
    LwU32 sensorIndex;
    LwU32 implementation;
} LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_IMPLEMENTATION_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_READING_RANGE instruction
 *
 * Get a sensor's readings range (ie min, max).
 *
 *   sensorIndex
 *       Set by the client to the desired sensor index.
 *
 *   minimum
 *       Returns a sensor's range minimum.
 *
 *   maximum
 *       Returns a sensor's range maximum.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_READING_RANGE_OPCODE (0x00000540U)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_READING_RANGE_OPERANDS {
    LwU32 sensorIndex;
    LwS32 minimum;
    LwS32 maximum;
} LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_READING_RANGE_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_PUBLIC instruction
 *
 * Get a sensor's public exposure.  If a sensor is public, it's reading
 * should be made available to a user.
 *
 *   sensorIndex
 *       Set by the client to the desired sensor index.
 *
 *   isPublic
 *       Set by the driver to one of the following:
 *        0: Sensor reading should not be made public.
 *        1: Sensor reading should be made public.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_PUBLIC_OPCODE (0x00000550U)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_PUBLIC_OPERANDS {
    LwU32 sensorIndex;
    LwU32 isPublic;
} LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_PUBLIC_OPERANDS;


/*!
 * LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR instruction
 *
 * Returns descriptive information about a SENSOR.  Right now, only the hotspot
 * offset.
 *
 * Note, eventually this opcode will include all the information from each of
 * the above GET_INFO_SENSOR_*_OPCODE opcodes.  Those opcodes will be deprecated
 * and replaced by this one.
 *
 *   sensorIndex [in]
 *       Set by the client to the desired sensor index.
 *
 *   hotspotOffset [out]
 *       The hotspot offset of the desired sensor.  Though the type is a signed
 *       32 bit integer, this is actually a signed fixed-point 24.8 value.
 *
 *       The hotspot offset is description of the expected max temperature delta
 *       between the diode reading and the actual hottest spot on the TARGET.
 *       For instance when measuring the GPU die, the internal diode is in the
 *       middle in the chip, but the hottest spots are in the TPCs/GPC clusters
 *       in the corners of the die - those clusters may be 6-9C hotter than the
 *       diode reading.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_OPCODE (0x00000560U)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_OPERANDS {
    LwU32 sensorIndex;
    LwS32 hotspotOffset;
} LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_OPERANDS;

/*
 * LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_DYNAMIC_SLOWDOWN_OPCODE
 *
 * bSupported
 *    Returns whether the RM supports DYNAMIC_SLOWDOWN functionality on this
 *    GPU.
 *
 * If LW2080_CTRL_THERMAL_GET_INFO_DYNAMIC_SLOWDOWN_AVAILABLE is supported,
 * clients can take advantage of dynamic slowdown for GPU slowdown.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_DYNAMIC_SLOWDOWN_OPCODE (0x00000570U)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_DYNAMIC_SLOWDOWN_OPERANDS {
    LwBool bSupported;
} LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_DYNAMIC_SLOWDOWN_OPERANDS;

/*!
 * Structure for defining a slowdown amount ratio of a numerator and denominator.
 * RM will try to match this ratio to the closest possible supported HW
 * factor whose slowdown is greater than or equal to this ratio.
 *
 * num
 *     Numerator of the ratio.
 *
 * denom
 *     Denominator of the ratio.
 */
typedef struct LW2080_CTRL_THERMAL_SLOWDOWN_AMOUNT {
    LwU32 num;
    LwU32 denom;
} LW2080_CTRL_THERMAL_SLOWDOWN_AMOUNT;

/*!
 * Supported DYNAMIC_SLOWDOWN_MODEs.
 *
 * LW2080_CTRL_THERMAL_SYSTEM_DYNAMIC_SLOWDOWN_MODE_DISABLED:
 *     Disabled.  Now slowdown is lwrrently enabled.
 *
 * LW2080_CTRL_THERMAL_SYSTEM_DYNAMIC_SLOWDOWN_MODE_SINGLE_THRESHOLD:
 *     Will assert the specified slowdown amount until changed or disabled.
 */
typedef enum LW2080_CTRL_THERMAL_SYSTEM_DYNAMIC_SLOWDOWN_MODE {
    LW2080_CTRL_THERMAL_SYSTEM_DYNAMIC_SLOWDOWN_MODE_DISABLED = 0,
    LW2080_CTRL_THERMAL_SYSTEM_DYNAMIC_SLOWDOWN_MODE_SINGLE_THRESHOLD = 1,
    // Add new entries here
    LW2080_CTRL_THERMAL_SYSTEM_DYNAMIC_SLOWDOWN_MODE_END = 2, // Should always be last entry
} LW2080_CTRL_THERMAL_SYSTEM_DYNAMIC_SLOWDOWN_MODE;

/*!
 * Arguments related to _SINGLE_THRESHOLD DYNAMIC_SLOWDOWN
 *
 * amount
 *     LW2080_CTRL_THERMAL_SLOWDOWN_AMOUNT structure specifying the desired
 *     slowdown amount.
 */
typedef struct LW2080_CTRL_THERMAL_SYSTEM_DYNAMIC_SLOWDOWN_ARGS_SINGLE_THRESHOLD {
    LW2080_CTRL_THERMAL_SLOWDOWN_AMOUNT amount;
} LW2080_CTRL_THERMAL_SYSTEM_DYNAMIC_SLOWDOWN_ARGS_SINGLE_THRESHOLD;


/*!
 * Union of mode-specific arguments.
 */


/*!
 * Structure used for both accessing and mutating DYNAMIC_SLOWDOWN functionality.
 *
 * mode
 *     accessor - The current mode of DYNAMIC_SLOWDOWN.
 *     mutator - The mode to set for DYNAMIC_SLOWDOWN.
 *
 * args
 *     accessor - The current mode-specific arguments of DYNAMIC_SLOWDOWN.
 *     mutator - The mode-specific arguments to set for DYNAMIC_SLOWDOWN.
 */
typedef struct LW2080_CTRL_THERMAL_SYSTEM_STATUS_DYNAMIC_SLOWDOWN_OPERANDS {
    LW2080_CTRL_THERMAL_SYSTEM_DYNAMIC_SLOWDOWN_MODE mode;

    union {
        LW2080_CTRL_THERMAL_SYSTEM_DYNAMIC_SLOWDOWN_ARGS_SINGLE_THRESHOLD singleThreshold;
    } args;
} LW2080_CTRL_THERMAL_SYSTEM_STATUS_DYNAMIC_SLOWDOWN_OPERANDS;

/*!
 * LW2080_CTRL_THERMAL_SYSTEM_GET_STATUS_DYNAMIC_SLOWDOWN_OPCODE
 *
 * Retrieves the current settings of DYNAMIC_SLOWDOWN.
 *
 * See documentation of
 * LW2080_CTRL_THERMAL_SYSTEM_STATUS_DYNAMIC_SLOWDOWN_OPERANDS.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_THERMAL_SYSTEM_GET_STATUS_DYNAMIC_SLOWDOWN_OPCODE (0x00000571U)

/*!
 * LW2080_CTRL_THERMAL_SYSTEM_SET_STATUS_DYNAMIC_SLOWDOWN_OPCODE
 *
 * Sets the status of DYNAMIC_SLOWDOWN.
 *
 * See documentation of
 * LW2080_CTRL_THERMAL_SYSTEM_STATUS_DYNAMIC_SLOWDOWN_OPERANDS.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_THERMAL_SYSTEM_SET_STATUS_DYNAMIC_SLOWDOWN_OPCODE (0x00000572U)

/*
 * LW2080_CTRL_THERMAL_SYSTEM_GET_STATUS_SENSOR_READING instruction
 *
 * Get a sensor's current reading.
 *
 *   sensorIndex
 *       Set by the client to the desired sensor index.
 *
 *   value
 *       Returns a sensor's current reading.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_SYSTEM_GET_STATUS_SENSOR_READING_OPCODE   (0x00001500U)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_GET_STATUS_SENSOR_READING_OPERANDS {
    LwU32 sensorIndex;
    LwS32 value;
} LW2080_CTRL_THERMAL_SYSTEM_GET_STATUS_SENSOR_READING_OPERANDS;


/*
 * ThermalSlowdownDisabled related instructions....
 *
 */

/*
 * LW2080_CTRL_THERMAL_SLOWDOWN_MODE
 *
 * Possible values are:
 *      LW2080_CTRL_THERMAL_SLOWDOWN_ENABLED
 *      LW2080_CTRL_THERMAL_SLOWDOWN_DISABLED_ALL
 *
 * Expand if needed to allow disabling only certain forms of thermal slowdown.
 * Sync this up with one in lwapi.spec!
 */

typedef enum LW2080_CTRL_THERMAL_SLOWDOWN_STATE {
    LW2080_CTRL_THERMAL_SLOWDOWN_ENABLED = 0,
    LW2080_CTRL_THERMAL_SLOWDOWN_DISABLED_ALL = 65535,
} LW2080_CTRL_THERMAL_SLOWDOWN_STATE;

/*
 * LW2080_CTRL_THERMAL_SYSTEM_GET_SLOWDOWN_STATE_OPCODE
 *
 * Get a GPU's current ThermalSlowdownDisabled state.
 *
 *   slowdownState
 *       Set by the driver to one of the following:
 *        LW2080_CTRL_THERMAL_SLOWDOWN_ENABLED
 *        LW2080_CTRL_THERMAL_SLOWDOWN_DISABLED_ALL
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW2080_CTRL_THERMAL_SYSTEM_GET_SLOWDOWN_STATE_OPCODE (0x00007001U)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_GET_SLOWDOWN_STATE_OPERANDS {
    LW2080_CTRL_THERMAL_SLOWDOWN_STATE slowdownState;
    LwU32                              reserved;
} LW2080_CTRL_THERMAL_SYSTEM_GET_SLOWDOWN_STATE_OPERANDS;

/*
 * LW2080_CTRL_THERMAL_SYSTEM_SET_SLOWDOWN_STATE_OPCODE
 *
 * Set a GPU's ThermalSlowdownDisabled state.
 *
 *   slowdownState
 *       Set by the client to the desired slowdown state:
 *        LW2080_CTRL_THERMAL_SLOWDOWN_ENABLED
 *        LW2080_CTRL_THERMAL_SLOWDOWN_DISABLED_ALL
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW2080_CTRL_THERMAL_SYSTEM_SET_SLOWDOWN_STATE_OPCODE (0x00007002U)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_SET_SLOWDOWN_STATE_OPERANDS {
    LW2080_CTRL_THERMAL_SLOWDOWN_STATE slowdownState;
    LwU32                              reserved;
} LW2080_CTRL_THERMAL_SYSTEM_SET_SLOWDOWN_STATE_OPERANDS;

/*
 * LW2080_CTRL_THERMAL_SYSTEM_GET_DIAG_ZONE_OPCODE
 *
 * Possible values are:
 *          LW2080_CTRL_THERMAL_DIAG_ZONE_ILWALID
 *          LW2080_CTRL_THERMAL_DIAG_ZONE_NORMAL
 *          LW2080_CTRL_THERMAL_DIAG_ZONE_WARNING
 *          LW2080_CTRL_THERMAL_DIAG_ZONE_CRITICAL
 */
typedef enum LW2080_CTRL_THERMAL_DIAG_ZONE_TYPE {
    LW2080_CTRL_THERMAL_DIAG_ZONE_ILWALID = 0,
    LW2080_CTRL_THERMAL_DIAG_ZONE_NORMAL = 1,
    LW2080_CTRL_THERMAL_DIAG_ZONE_WARNING = 2,
    LW2080_CTRL_THERMAL_DIAG_ZONE_CRITICAL = 4,
} LW2080_CTRL_THERMAL_DIAG_ZONE_TYPE;

// Update this if any additional levels are added
#define LW2080_CTRL_THERMAL_VALID_DIAG_ZONES            3U

// Maximum zones limits
#define LW2080_CTRL_THERMAL_DIAG_MAX_ZONE_LIMITS        (0x2U) /* finn: Evaluated from "(LW2080_CTRL_THERMAL_VALID_DIAG_ZONES - 1)" */

/*
 * LW2080_CTRL_THERMAL_SYSTEM_GET_DIAG_ZONE_OPCODE
 *
 * Get a GPU's current Thermal Zone State.
 *
 *   lwrrentZone
 *       Set by the driver to one of the following:
 *          LW2080_CTRL_THERMAL_DIAG_ZONE_ILWALID
 *          LW2080_CTRL_THERMAL_DIAG_ZONE_NORMAL
 *          LW2080_CTRL_THERMAL_DIAG_ZONE_WARNING
 *          LW2080_CTRL_THERMAL_DIAG_ZONE_CRITICAL
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
  */
#define LW2080_CTRL_THERMAL_SYSTEM_GET_DIAG_ZONE_OPCODE (0x00007003U)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_GET_DIAG_ZONE_OPERANDS {
    LW2080_CTRL_THERMAL_DIAG_ZONE_TYPE lwrrentZone;
} LW2080_CTRL_THERMAL_SYSTEM_GET_DIAG_ZONE_OPERANDS;

/*
 * LW2080_CTRL_THERMAL_DIAG_LIMIT_INFO
 *
 * limit      - Temperature limit in Celsius
 * hysteresis - Hysteresis for this limit
 * zone       - Zone type. Possible values are:
 *              LW2080_CTRL_THERMAL_DIAG_ZONE_NORMAL
 *              LW2080_CTRL_THERMAL_DIAG_ZONE_WARNING
 *              LW2080_CTRL_THERMAL_DIAG_ZONE_CRITICAL
 *              Values for _NORMAL zone is not returned using this structure and
 *              it's assumed that temperature below _WARNING is considered as _NORMAL.
 */
typedef struct LW2080_CTRL_THERMAL_DIAG_LIMIT_INFO {
    LwS32                              limit;
    LwU32                              hysteresis;
    LW2080_CTRL_THERMAL_DIAG_ZONE_TYPE zone;
} LW2080_CTRL_THERMAL_DIAG_LIMIT_INFO;

/*
 * LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_DIAG
 *
 * Get a GPU's zone limits information.
 *
 *   zoneLimits
 *       Set by the driver
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
  */
#define LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_DIAG_OPCODE (0x00007004U)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_DIAG_OPERANDS {
    LW2080_CTRL_THERMAL_DIAG_LIMIT_INFO zoneLimits[LW2080_CTRL_THERMAL_DIAG_MAX_ZONE_LIMITS];
} LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_DIAG_OPERANDS;

/*
 * TEMPSIM related instructions....
 *
 */

/*
 * LW2080_CTRL_THERMAL_SYSTEM_TEMPSIM_CONTROL
 *
 *  Possible TEMPSIM Control values.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_TEMPSIM_CONTROL_ENABLE_NO
 *       TEMPSIM is disabled (GET_STATUS) or disable TEMPSIM (SET_CONTROL).
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_TEMPSIM_CONTROL_ENABLE_YES
 *       TEMPSIM is enabled (GET_STATUS) or enable TEMPSIM (SET_CONTROL).
 *
 * Sync this up with one in lwapi.spec!
 */
#define LW2080_CTRL_THERMAL_SYSTEM_TEMPSIM_CONTROL_ENABLE_NO        (0x00000000U)
#define LW2080_CTRL_THERMAL_SYSTEM_TEMPSIM_CONTROL_ENABLE_YES       (0x00000001U)

/*
 * LW2080_CTRL_THERMAL_SYSTEM_TEMPSIM_SUPPORT
 *
 *  Possible TEMPSIM Support values.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_TEMPSIM_SUPPORT_NO
 *       Sensor does not support temperature simulation.
 *
 *   LW2080_CTRL_THERMAL_SYSTEM_TEMPSIM_SUPPORT_YES
 *       Sensor does support temperature simulation.
 */
#define LW2080_CTRL_THERMAL_SYSTEM_TEMPSIM_SUPPORT_NO               (0x00000000U)
#define LW2080_CTRL_THERMAL_SYSTEM_TEMPSIM_SUPPORT_YES              (0x00000001U)

/*
 * LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SUPPORTS_TEMPSIM instruction
 *
 * Get a sensor's support for temperature simulation (TEMPSIM).
 *
 *   sensorIndex
 *       Set by the client to the desired sensor index.
 *
 *   support
 *       Set by the driver to one of the following:
 *        LW2080_CTRL_THERMAL_SYSTEM_TEMPSIM_SUPPORT_NO
 *        LW2080_CTRL_THERMAL_SYSTEM_TEMPSIM_SUPPORT_YES
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SUPPORTS_TEMPSIM_OPCODE (0x00000555U)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SUPPORTS_TEMPSIM_OPERANDS {
    LwU32 sensorIndex;
    LwU32 support;
} LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SUPPORTS_TEMPSIM_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_SYSTEM_GET_TEMPSIM_CONTROL instruction
 *
 * Get a sensor's current TEMPSIM control value.
 *
 *   sensorIndex
 *       Set by the client to the desired sensor index.
 *
 *   control
 *       Set by the driver to one of the following:
 *        LW2080_CTRL_THERMAL_SYSTEM_TEMPSIM_CONTROL_ENABLE_NO
 *        LW2080_CTRL_THERMAL_SYSTEM_TEMPSIM_CONTROL_ENABLE_YES
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_SYSTEM_GET_TEMPSIM_CONTROL_OPCODE (0x00000556U)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_GET_TEMPSIM_CONTROL_OPERANDS {
    LwU32 sensorIndex;
    LwU32 control;
} LW2080_CTRL_THERMAL_SYSTEM_GET_TEMPSIM_CONTROL_OPERANDS;

/*
 * LW2080_CTRL_THERMAL_SYSTEM_SET_TEMPSIM_CONTROL_AND_TEMPERATURE instruction
 *
 * Set a sensor's current TempSim control and temperature
 *
 * This instruction is more secure to use, because it handles all corner cases
 * that can occur while calling:
 * LW2080_CTRL_THERMAL_SYSTEM_SET_TEMPSIM_CONTROL &
 * LW2080_CTRL_THERMAL_SYSTEM_SET_TEMPSIM_TEMPERATURE.
 *
 *   sensorIndex
 *       Set by the client to the desired sensor index.
 *
 *   control
 *       Set by the client to the desired control value.
 *       Possible values are the following:
 *        LW2080_CTRL_THERMAL_SYSTEM_TEMPSIM_CONTROL_ENABLE_NO
 *        LW2080_CTRL_THERMAL_SYSTEM_TEMPSIM_CONTROL_ENABLE_YES
 *
 *   temperature
 *       Set by the client to the desired simulation temperature
 *       if(control==LW2080_CTRL_THERMAL_SYSTEM_TEMPSIM_CONTROL_ENABLE_YES).
 *       Possible values are values at or in between values returned by:
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_READING_RANGE
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_SYSTEM_SET_TEMPSIM_CONTROL_AND_TEMPERATURE_OPCODE (0x00000559U)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_SET_TEMPSIM_CONTROL_AND_TEMPERATURE_OPERANDS {
    LwU32 sensorIndex;
    LwU32 control;
    LwS32 temperature;
} LW2080_CTRL_THERMAL_SYSTEM_SET_TEMPSIM_CONTROL_AND_TEMPERATURE_OPERANDS;

/*
 * LW2080_CTRL_THERMAL_SYSTEM_GET_INT_SENSOR_RAW_DATA_AND_COEFS instruction
 *
 * Retrieves internal thermal sensor's RAW reading and equation parameters.
 *
 * Current temperature can be callwlated using equation T = A * RAW + B, and
 * result will be temperature in Celsius as fixed point integer with 16
 * decimal bits.
 *
 *   sensorIndex
 *       Set by the client to the desired internal sensor index.
 *
 *   rawReading
 *       Set by the driver to the RAW reading of desired int. thermal sensor.
 *
 *   slope
 *       Set by the driver to lwrrently used "A" value (see equation above).
 *       Value is fixed point integer with 16 decimal bits.
 *
 *   offset
 *       Set by the driver to lwrrently used "B" value (see equation above).
 *       Value is fixed point integer with 16 decimal bits.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_SYSTEM_GET_INT_SENSOR_RAW_DATA_AND_COEFS_OPCODE (0x0000055AU)
typedef struct LW2080_CTRL_THERMAL_SYSTEM_GET_INT_SENSOR_RAW_DATA_AND_COEFS_OPERANDS {
    LwU32 sensorIndex;
    LwU32 rawReading;
    LwS32 slope;
    LwS32 offset;
} LW2080_CTRL_THERMAL_SYSTEM_GET_INT_SENSOR_RAW_DATA_AND_COEFS_OPERANDS;



/*
 * Thermal System instruction operand
 */
typedef union LW2080_CTRL_THERMAL_SYSTEM_INSTRUCTION_OPERANDS {

    /*
     *  GetInfo instruction operands...
     */
    LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGETS_AVAILABLE_OPERANDS          getInfoTargetsAvailable;

    LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGET_TYPE_OPERANDS                getInfoTargetType;

    LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGET_SENSORS_OPERANDS             getInfoTargetSensors;

    LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGET_PROVIDERS_OPERANDS           getInfoTargetProviders;

    LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDERS_AVAILABLE_OPERANDS        getInfoProvidersAvailable;

    LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDER_TYPE_OPERANDS              getInfoProviderType;

    LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDER_TARGETS_OPERANDS           getInfoProviderTargets;

    LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDER_SENSORS_OPERANDS           getInfoProviderSensors;

    LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSORS_AVAILABLE_OPERANDS          getInfoSensorsAvailable;

    LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_PROVIDER_OPERANDS            getInfoSensorProvider;

    LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_TARGET_OPERANDS              getInfoSensorTarget;

    LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_IMPLEMENTATION_OPERANDS      getInfoSensorImplementation;

    LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_READING_RANGE_OPERANDS       getInfoSensorReadingRange;

    LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_PUBLIC_OPERANDS              getInfoSensorPublic;

    LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_OPERANDS                     getInfoSensor;

    LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_DYNAMIC_SLOWDOWN_OPERANDS           getInfoDynamicSlowdown;

    LW2080_CTRL_THERMAL_SYSTEM_STATUS_DYNAMIC_SLOWDOWN_OPERANDS             getStatusDynamicSlowdown;

    LW2080_CTRL_THERMAL_SYSTEM_STATUS_DYNAMIC_SLOWDOWN_OPERANDS             setStatusDynamicSlowdown;

    LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SUPPORTS_TEMPSIM_OPERANDS           getInfoSupportsTempSim;

    LW2080_CTRL_THERMAL_SYSTEM_GET_TEMPSIM_CONTROL_OPERANDS                 getTempSimControl;

    LW2080_CTRL_THERMAL_SYSTEM_SET_TEMPSIM_CONTROL_AND_TEMPERATURE_OPERANDS setTempSimControlAndTemperature;

    LW2080_CTRL_THERMAL_SYSTEM_GET_INT_SENSOR_RAW_DATA_AND_COEFS_OPERANDS   getIntSensorRawDataAndCoefs;

        /*
         *  GetStatus instruction opcodes...
         */

    LW2080_CTRL_THERMAL_SYSTEM_GET_STATUS_SENSOR_READING_OPERANDS           getStatusSensorReading;

        /*
         * Thermal slowdown control instruction operands...
         */

    LW2080_CTRL_THERMAL_SYSTEM_GET_SLOWDOWN_STATE_OPERANDS                  getThermalSlowdownState;

    LW2080_CTRL_THERMAL_SYSTEM_SET_SLOWDOWN_STATE_OPERANDS                  setThermalSlowdownState;

        /*
         * Diag related instruction operands...
        */
    LW2080_CTRL_THERMAL_SYSTEM_GET_DIAG_ZONE_OPERANDS                       getThermalZone;
    LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_DIAG_OPERANDS                       getInfoDiag;

        /*
         *  Minimum operand size. This should be larger than the largest operand,
         *  for growth, but please don't change, as it will break the api.
         *  Client instruction _must_ match driver instruction size.
         *
         * XAPIGEN: mark "hidden" since not needed for XAPI.
         */
    LwU32                                                                   space[8];
} LW2080_CTRL_THERMAL_SYSTEM_INSTRUCTION_OPERANDS;



/*
 * LW2080_CTRL_THERMAL_SYSTEM_INSTRUCTION
 *
 * All thermal system instructions have the following layout:
 *
 *   result
 *       This field is set by the driver, and is the result of the
 *       instruction's exelwtion. This value is only valid if the
 *       exelwted field is not 0 upon return.
 *       Possible status values returned are:
 *        LW_OK
 *        LW_ERR_ILWALID_ARGUMENT
 *        LW_ERR_ILWALID_PARAM_STRUCT
 *
 *   exelwted
 *       This field is set by the driver, and
 *       indicates if the instruction was exelwted.
 *       Possible status values returned are:
 *        0: Not exelwted
 *        1: Exelwted
 *
 *   opcode
 *       This field is set by the client to the desired instruction opcode.
 *       Possible values are:
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGETS_AVAILABLE_OPCODE
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGET_TYPE_OPCODE
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGET_SENSORS_OPCODE
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGET_PROVIDERS_OPCODE
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDERS_AVAILABLE_OPCODE
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDER_TYPE_OPCODE
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDER_TARGETS_OPCODE
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDER_SENSORS_OPCODE
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSORS_AVAILABLE_OPCODE
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_PROVIDER_OPCODE
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_TARGET_OPCODE
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_IMPLEMENTATION_OPCODE
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_READING_RANGE_OPCODE
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_PUBLIC_OPCODE
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_OPCODE
 *        LW2080_CTRL_THERMAL_SYSTEM_SET_STATUS_DYNAMIC_SLOWDOWN_OPCODE
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_DYNAMIC_SLOWDOWN_OPCODE
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_STATUS_DYNAMIC_SLOWDOWN_OPCODE
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_STATUS_SENSOR_READING_OPCODE
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_SLOWDOWN_STATE_OPCODE
 *        LW2080_CTRL_THERMAL_SYSTEM_SET_SLOWDOWN_STATE_OPCODE
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_DIAG_ZONE_OPCODE
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_DIAG_OPCODE
 *
 *   operands
 *       This field is actually a union of all of the available operands.
 *       The interpretation of this field is opcode context dependent.
 *       Possible values are:
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGETS_AVAILABLE_OPERANDS
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGET_TYPE_OPERANDS
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGET_SENSORS_OPERANDS
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_TARGET_PROVIDERS_OPERANDS
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDERS_AVAILABLE_OPERANDS
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDER_TYPE_OPERANDS
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDER_TARGETS_OPERANDS
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_PROVIDER_SENSORS_OPERANDS
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSORS_AVAILABLE_OPERANDS
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_PROVIDER_OPERANDS
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_TARGET_OPERANDS
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_IMPLEMENTATION_OPERANDS
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_READING_RANGE_OPERANDS
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_PUBLIC_OPERANDS
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_SENSOR_OPERANDS
 *        LW2080_CTRL_THERMAL_SYSTEM_SET_STATUS_DYNAMIC_SLOWDOWN_OPERANDS
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_DYNAMIC_SLOWDOWN_OPERANDS
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_STATUS_DYNAMIC_SLOWDOWN_OPERANDS
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_STATUS_SENSOR_READING_OPERANDS
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_SLOWDOWN_STATE_OPERANDS
 *        LW2080_CTRL_THERMAL_SYSTEM_SET_SLOWDOWN_STATE_OPERANDS
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_DIAG_ZONE_OPERANDS
 *        LW2080_CTRL_THERMAL_SYSTEM_GET_INFO_DIAG_OPERANDS
 *
 */
typedef struct LW2080_CTRL_THERMAL_SYSTEM_INSTRUCTION {
    LwU32                                           result;
    LwU32                                           exelwted;
    LwU32                                           opcode;
    // C Form: LW2080_CTRL_THERMAL_SYSTEM_INSTRUCTION_OPERANDS operands;
    LW2080_CTRL_THERMAL_SYSTEM_INSTRUCTION_OPERANDS operands;
} LW2080_CTRL_THERMAL_SYSTEM_INSTRUCTION;


#define LW2080_CTRL_CMD_THERMAL_SYSTEM_EXELWTE (0x20800512U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | 0x12" */

typedef struct LW2080_CTRL_THERMAL_SYSTEM_EXELWTE_PARAMS {
    LwU32 clientAPIVersion;
    LwU32 clientAPIRevision;
    LwU32 clientInstructionSizeOf;
    LwU32 exelwteFlags;
    LwU32 successfulInstructions;
    LwU32 instructionListSize;
    LW_DECLARE_ALIGNED(LwP64 instructionList, 8);
} LW2080_CTRL_THERMAL_SYSTEM_EXELWTE_PARAMS;

// Same as LW2080_CTRL_CMD_THERMAL_SYSTEM_EXELWTE but without embedded pointer
#define LW2080_CTRL_CMD_THERMAL_SYSTEM_EXELWTE_V2        (0x20800513U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | LW2080_CTRL_THERMAL_SYSTEM_EXELWTE_V2_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_THERMAL_SYSTEM_INSTRUCTION_MAX_COUNT 0x20U
#define LW2080_CTRL_THERMAL_SYSTEM_EXELWTE_V2_PARAMS_MESSAGE_ID (0x13U)

typedef struct LW2080_CTRL_THERMAL_SYSTEM_EXELWTE_V2_PARAMS {
    LwU32                                  clientAPIVersion;
    LwU32                                  clientAPIRevision;
    LwU32                                  clientInstructionSizeOf;
    LwU32                                  exelwteFlags;
    LwU32                                  successfulInstructions;
    LwU32                                  instructionListSize;
    LW2080_CTRL_THERMAL_SYSTEM_INSTRUCTION instructionList[LW2080_CTRL_THERMAL_SYSTEM_INSTRUCTION_MAX_COUNT];
} LW2080_CTRL_THERMAL_SYSTEM_EXELWTE_V2_PARAMS;



/*
 * LW2080_CTRL_THERMAL_COOLER api
 *
 *  The thermal cooler interfaces provide access to the cooling support
 *  in the LWPU driver.
 *
 *  The main sections of the cooler api consist of the following:
 *
 *   LW2080_CTRL_THERMAL_COOLER constants
 *    A set of constants used to interact with the cooler api.
 *    Typical constants include modes, cooling level, ranges, etc.
 *
 *   LW2080_CTRL_THERMAL_COOLER rmcontrol calls
 *    A set of rmcontrol calls used to interact with the
 *    cooling system through the driver.
 *
 */


/*
 * LW2080_CTRL_THERMAL_COOLER constants
 *
 */

/*
 * LW2080_CTRL_THERMAL_COOLER_LEVEL
 *
 *  A cooler level is a percent, from 0 to 100.
 *
 *   LW2080_CTRL_THERMAL_COOLER_LEVEL_OFF
 *       This cooler is off.
 *
 *   LW2080_CTRL_THERMAL_COOLER_LEVEL_FULL
 *       This cooler is at its highest level of operation.
 *
 */
#define LW2080_CTRL_THERMAL_COOLER_LEVEL_OFF                        (0U)
#define LW2080_CTRL_THERMAL_COOLER_LEVEL_FULL                       (100U)

/*
 * LW2080_CTRL_THERMAL_COOLER_TARGET
 *
 *  Cooler targets (ie the things a cooler cools). A cooler
 *  may cool more than one target.
 *
 *   LW2080_CTRL_THERMAL_COOLER_TARGET_NONE
 *       This cooler cools nothing.
 *
 *   LW2080_CTRL_THERMAL_COOLER_TARGET_GPU
 *       This cooler can cool the GPU.
 *
 *   LW2080_CTRL_THERMAL_COOLER_TARGET_MEMORY
 *       This cooler can cool the memory.
 *
 *   LW2080_CTRL_THERMAL_COOLER_TARGET_POWER_SUPPLY
 *       This cooler can cool the power supply.
 *
 *   LW2080_CTRL_THERMAL_COOLER_TARGET_GPU_RELATED
 *       This cooler cools all of the components related to its target gpu.
 */
#define LW2080_CTRL_THERMAL_COOLER_TARGET_NONE                      (0x00000000U)
#define LW2080_CTRL_THERMAL_COOLER_TARGET_GPU                       (0x00000001U)
#define LW2080_CTRL_THERMAL_COOLER_TARGET_MEMORY                    (0x00000002U)
#define LW2080_CTRL_THERMAL_COOLER_TARGET_POWER_SUPPLY              (0x00000004U)
#define LW2080_CTRL_THERMAL_COOLER_TARGET_GPU_RELATED               (0x7U) /* finn: Evaluated from "LW2080_CTRL_THERMAL_COOLER_TARGET_GPU | LW2080_CTRL_THERMAL_COOLER_TARGET_MEMORY | LW2080_CTRL_THERMAL_COOLER_TARGET_POWER_SUPPLY" */

/*
 * LW2080_CTRL_THERMAL_COOLER_PHYSICAL
 *
 *  Physical (mechanical) aspects, or pieces that make up a cooler.
 *
 *   LW2080_CTRL_THERMAL_COOLER_PHYSICAL_NONE
 *       This cooler has no physical implementation.
 *
 *   LW2080_CTRL_THERMAL_COOLER_PHYSICAL_FAN
 *       This cooler includes a fan.
 */
#define LW2080_CTRL_THERMAL_COOLER_PHYSICAL_NONE                    (0x00000000U)
#define LW2080_CTRL_THERMAL_COOLER_PHYSICAL_FAN                     (0x00000001U)

/*
 * LW2080_CTRL_THERMAL_COOLER_CONTROLLER
 *
 *  Aspects of the controller (device) that sends signals to the cooler.
 *
 *   LW2080_CTRL_THERMAL_COOLER_CONTROLLER_NONE
 *       This cooler has no controller.
 *
 *   LW2080_CTRL_THERMAL_COOLER_CONTROLLER_INTERNAL
 *       This cooler is controlled by a GPU internal controller.
 *
 *   LW2080_CTRL_THERMAL_COOLER_CONTROLLER_ADI7473
 *       This cooler is controlled by an ADI7473 fan controller.
 */
#define LW2080_CTRL_THERMAL_COOLER_CONTROLLER_NONE                  (0x00000000U)
#define LW2080_CTRL_THERMAL_COOLER_CONTROLLER_INTERNAL              (0x00000001U)
#define LW2080_CTRL_THERMAL_COOLER_CONTROLLER_ADI7473               (0x00000002U)

/*
 * LW2080_CTRL_THERMAL_COOLER_SIGNAL
 *
 *  A cooler is controlled by some electrical signal, typically a
 *  switch (ON/OFF) or a knob (PWM).
 *
 *   LW2080_CTRL_THERMAL_COOLER_SIGNAL_NONE
 *       This cooler has no control signal.
 *
 *   LW2080_CTRL_THERMAL_COOLER_SIGNAL_TOGGLE
 *       This cooler can only be toggled either ON or OFF (eg a switch).
 *       A level of 0 is OFF, and greater than 0 is ON.
 *
 *   LW2080_CTRL_THERMAL_COOLER_SIGNAL_VARIABLE
 *       This cooler's level can be adjusted from some minimum
 *       to some maximum (eg a knob).
 */
#define LW2080_CTRL_THERMAL_COOLER_SIGNAL_NONE                      (0x00000000U)
#define LW2080_CTRL_THERMAL_COOLER_SIGNAL_TOGGLE                    (0x00000001U)
#define LW2080_CTRL_THERMAL_COOLER_SIGNAL_VARIABLE                  (0x00000002U)

/*
 * LW2080_CTRL_THERMAL_COOLER_POLICY
 *
 *  A cooler typically has some over-seeing authority, or its control policy.
 *
 *  MUST keep these macros in sync with LWAPI LW_COOLER_POLICY!!!!!  We should
 *  really decouple these definitions with a translation layer.  Otherwise, we
 *  expose too much stuff directly to user mode.
 *
 *   LW2080_CTRL_THERMAL_COOLER_POLICY_NONE
 *       This cooler has no policies.
 *
 *   LW2080_CTRL_THERMAL_COOLER_POLICY_MANUAL
 *       Manual adjustment of cooler level.
 *
 *   LW2080_CTRL_THERMAL_COOLER_POLICY_PERF
 *       Perf table-controlled adjustment of cooler level.
 *
 *   LW2080_CTRL_THERMAL_COOLER_POLICY_TEMPERATURE_DISCRETE
 *       Discrete thermal limit adjustment of cooler level.
 *
 *   LW2080_CTRL_THERMAL_COOLER_POLICY_TEMPERATURE_CONTINUOUS
 *       Continuous thermal adjustment of cooler level by some HW unit -
 *       ADT7473, PMU, etc.  We want to keep all of our HW control logic behind
 *       this macro so we don't expose to much about the internals of our
 *       thermal design.
 *
 *   LW2080_CTRL_THERMAL_COOLER_POLICY_TEMPERATURE_CONTINUOUS_SW
 *       Continuous thermal adjustment of cooler level by a software agent.
 *
 *   LW2080_CTRL_THERMAL_COOLER_POLICY_DEFAULT
 *       Let RM to choose default cooler policy, that can be any one of above
 *       policies.  This really should be last (bit 31), as it is a special
 *       case.
 */
#define LW2080_CTRL_THERMAL_COOLER_POLICY_NONE                      (0x00000000U)
#define LW2080_CTRL_THERMAL_COOLER_POLICY_MANUAL                    (0x00000001U)
#define LW2080_CTRL_THERMAL_COOLER_POLICY_PERF                      (0x00000002U)
#define LW2080_CTRL_THERMAL_COOLER_POLICY_TEMPERATURE_DISCRETE      (0x00000004U)
#define LW2080_CTRL_THERMAL_COOLER_POLICY_TEMPERATURE_CONTINUOUS    (0x00000008U)
#define LW2080_CTRL_THERMAL_COOLER_POLICY_TEMPERATURE_CONTINUOUS_SW (0x00000010U)
#define LW2080_CTRL_THERMAL_COOLER_POLICY_DEFAULT                   (0x00000020U)


/*
 * LW2080_CTRL_THERMAL_COOLER api
 *
 *  The thermal cooler interfaces provide access to the cooling support
 *  in the LWPU driver.
 *
 *  Lwrrently, the interface exports the following rmcontrol calls:
 *
 *  LW2080_CTRL_CMD_THERMAL_COOLER_GET_VERSION_INFO
 *   Gets the version of the cooler interface in use by the driver.
 *
 *  LW2080_CTRL_CMD_THERMAL_COOLER_GET_INSTRUCTION_SIZE
 *   Gets the instruction size in use by the driver.
 *
 *  LW2080_CTRL_CMD_THERMAL_COOLER_EXELWTE
 *   Exelwtes a set of cooler instructions provided by the client.
 */

/*
 * LW2080_CTRL_CMD_THERMAL_COOLER_EXELWTE
 *
 * This command will execute a list of cooler instructions:
 *
 *   clientAPIVersion
 *       This field must be set by the client to THERMAL_COOLER_API_VER,
 *       which allows the driver to determine api compatibility.
 *
 *   clientAPIRevision
 *       This field must be set by the client to THERMAL_COOLER_API_REV,
 *       which allows the driver to determine api compatibility.
 *
 *   clientInstructionSizeOf
 *       This field must be set by the client to
 *       sizeof(LW2080_CTRL_THERMAL_COOLER_INSTRUCTION), which allows the
 *       driver to determine api compatibility.
 *
 *   exelwteFlags
 *       This field is set by the client to control instruction exelwtion.
 *        LW2080_CTRL_THERMAL_COOLER_EXELWTE_FLAGS_DEFAULT
 *         Execute instructions normally. The first instruction
 *         failure will cause exelwtion to stop.
 *        LW2080_CTRL_THERMAL_COOLER_EXELWTE_FLAGS_IGNORE_FAIL
 *         Execute all instructions, ignoring individual instruction failures.
 *
 *   successfulInstructions
 *       This field is set by the driver and is the number of instructions
 *       that returned LW_OK on exelwtion.  If this field
 *       matches instructionListSize, all instructions exelwted successfully.
 *
 *   instructionListSize
 *       This field is set by the client to the number of instructions in
 *       instruction list.
 *
 *   instructionList
 *       This field is set by the client to point to an array of cooler
 *       instructions (LW2080_CTRL_THERMAL_COOLER_INSTRUCTION) to execute.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW2080_CTRL_CMD_THERMAL_COOLER_EXELWTE                      (0x20800522U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | LW2080_CTRL_THERMAL_COOLER_EXELWTE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_THERMAL_COOLER_EXELWTE_PARAMS_MESSAGE_ID (0x22U)

typedef struct LW2080_CTRL_THERMAL_COOLER_EXELWTE_PARAMS {
    LwU32 clientAPIVersion;
    LwU32 clientAPIRevision;
    LwU32 clientInstructionSizeOf;
    LwU32 exelwteFlags;
    LwU32 successfulInstructions;
    LwU32 instructionListSize;
    LW_DECLARE_ALIGNED(LwP64 instructionList, 8);
} LW2080_CTRL_THERMAL_COOLER_EXELWTE_PARAMS;

/*
 * exelwteFlags values
 */
#define LW2080_CTRL_THERMAL_COOLER_EXELWTE_FLAGS_DEFAULT     (0x00000000U)
#define LW2080_CTRL_THERMAL_COOLER_EXELWTE_FLAGS_IGNORE_FAIL (0x00000001U)


/*
 * LW2080_CTRL_THERMAL_COOLER instructions
 *
 *   The cooler instruction is central to the majority
 *   of cooler system interactions.
 *
 *   The cooler index is central to the cooler instruction.
 *   There will be some number of available coolers, and
 *   the values in the range of 0 to this number minus 1
 *   are cooler indexes.
 *
 *   A client can get the number of available coolers using the
 *   LW2080_CTRL_THERMAL_COOLER_GET_INFO_AVAILABLE instruction.
 *
 *   A client can group any number of instructions on any or all
 *   cooler indexes and execute them in a single call to the api, using
 *   LW2080_CTRL_CMD_THERMAL_COOLER_EXELWTE.  Grouping as many
 *   instructions as possible is more efficient as it lessens the
 *   number of possible client-driver transitions required.
 */

/*
 * LW2080_CTRL_THERMAL_COOLER_GET_INFO instructions...
 *
 */

/*
 * LW2080_CTRL_THERMAL_COOLER_GET_INFO_AVAILABLE instruction
 *
 * Get the number of available coolers.
 *
 *   availableCoolers
 *       Returns the number of available coolers.  Coolers are
 *       identified by an index, starting with 0.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW2080_CTRL_THERMAL_COOLER_GET_INFO_AVAILABLE_OPCODE (0x00001000U)
typedef struct LW2080_CTRL_THERMAL_COOLER_GET_INFO_AVAILABLE_OPERANDS {
    LwU32 availableCoolers;
} LW2080_CTRL_THERMAL_COOLER_GET_INFO_AVAILABLE_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_COOLER_GET_INFO_TARGETS instruction
 *
 * Get a cooler's targets.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   targets
 *       Returns a cooler's cooling targets.
 *       Possible values returned are:
 *          LW2080_CTRL_THERMAL_COOLER_TARGET_NONE
 *          LW2080_CTRL_THERMAL_COOLER_TARGET_GPU
 *          LW2080_CTRL_THERMAL_COOLER_TARGET_MEMORY
 *          LW2080_CTRL_THERMAL_COOLER_TARGET_POWER_SUPPLY
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_GET_INFO_TARGETS_OPCODE (0x00001010U)
typedef struct LW2080_CTRL_THERMAL_COOLER_GET_INFO_TARGETS_OPERANDS {
    LwU32 coolerIndex;
    LwU32 targets;
} LW2080_CTRL_THERMAL_COOLER_GET_INFO_TARGETS_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_COOLER_GET_INFO_PHYSICAL instruction
 *
 * Get a cooler's physical characteristics.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   physical
 *       Returns a cooler's physical characteristics.
 *       Possible values returned are:
 *          LW2080_CTRL_THERMAL_COOLER_PHYSICAL_NONE
 *          LW2080_CTRL_THERMAL_COOLER_PHYSICAL_FAN
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_GET_INFO_PHYSICAL_OPCODE (0x00001020U)
typedef struct LW2080_CTRL_THERMAL_COOLER_GET_INFO_PHYSICAL_OPERANDS {
    LwU32 coolerIndex;
    LwU32 physical;
} LW2080_CTRL_THERMAL_COOLER_GET_INFO_PHYSICAL_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_COOLER_GET_INFO_CONTROLLER instruction
 *
 * Get a cooler's controller implementation.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   controller
 *       Returns a cooler's controller implementation.
 *       Possible values returned are:
 *          LW2080_CTRL_THERMAL_COOLER_CONTROLLER_NONE
 *          LW2080_CTRL_THERMAL_COOLER_CONTROLLER_INTERNAL
 *          LW2080_CTRL_THERMAL_COOLER_CONTROLLER_ADI7473
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_GET_INFO_CONTROLLER_OPCODE (0x00001030U)
typedef struct LW2080_CTRL_THERMAL_COOLER_GET_INFO_CONTROLLER_OPERANDS {
    LwU32 coolerIndex;
    LwU32 controller;
} LW2080_CTRL_THERMAL_COOLER_GET_INFO_CONTROLLER_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_COOLER_GET_INFO_SIGNAL instruction
 *
 * Get a cooler's control signal characteristics.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   signal
 *       Returns a cooler's control signal characteristics.
 *       Possible values returned are:
 *          LW2080_CTRL_THERMAL_COOLER_SIGNAL_NONE
 *          LW2080_CTRL_THERMAL_COOLER_SIGNAL_TOGGLE
 *          LW2080_CTRL_THERMAL_COOLER_SIGNAL_VARIABLE
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_GET_INFO_SIGNAL_OPCODE (0x00001040U)
typedef struct LW2080_CTRL_THERMAL_COOLER_GET_INFO_SIGNAL_OPERANDS {
    LwU32 coolerIndex;
    LwU32 signal;
} LW2080_CTRL_THERMAL_COOLER_GET_INFO_SIGNAL_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_COOLER_GET_INFO_POLICIES instruction
 *
 * Get a cooler's supported control policies.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   policies
 *       Returns a cooler's supported control policies.
 *       Possible values returned are one or more of:
 *          LW2080_CTRL_THERMAL_COOLER_POLICY_NONE
 *          LW2080_CTRL_THERMAL_COOLER_POLICY_MANUAL
 *          LW2080_CTRL_THERMAL_COOLER_POLICY_PERF
 *          LW2080_CTRL_THERMAL_COOLER_POLICY_TEMPERATURE_DISCRETE
 *          LW2080_CTRL_THERMAL_COOLER_POLICY_TEMPERATURE_CONTINUOUS
 *          LW2080_CTRL_THERMAL_COOLER_POLICY_TEMPERATURE_CONTINUOUS_SW
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_GET_INFO_POLICIES_OPCODE (0x00001050U)
typedef struct LW2080_CTRL_THERMAL_COOLER_GET_INFO_POLICIES_OPERANDS {
    LwU32 coolerIndex;
    LwU32 policies;
} LW2080_CTRL_THERMAL_COOLER_GET_INFO_POLICIES_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_COOLER_GET_INFO_POLICY_DEFAULT instruction
 *
 * Get a cooler's default control policy.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   policy
 *       Returns a cooler's default control policy.
 *       Possible values returned are:
 *          LW2080_CTRL_THERMAL_COOLER_POLICY_NONE
 *          LW2080_CTRL_THERMAL_COOLER_POLICY_MANUAL
 *          LW2080_CTRL_THERMAL_COOLER_POLICY_PERF
 *          LW2080_CTRL_THERMAL_COOLER_POLICY_TEMPERATURE_DISCRETE
 *          LW2080_CTRL_THERMAL_COOLER_POLICY_TEMPERATURE_CONTINUOUS
 *          LW2080_CTRL_THERMAL_COOLER_POLICY_TEMPERATURE_CONTINUOUS_SW
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_GET_INFO_POLICY_DEFAULT_OPCODE (0x00001060U)
typedef struct LW2080_CTRL_THERMAL_COOLER_GET_INFO_POLICY_DEFAULT_OPERANDS {
    LwU32 coolerIndex;
    LwU32 policy;
} LW2080_CTRL_THERMAL_COOLER_GET_INFO_POLICY_DEFAULT_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_COOLER_GET_INFO_LEVEL_MINIMUM_DEFAULT instruction
 *
 * Get a cooler's default minimum operating level.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   policy
 *       Returns a cooler's default minimum operating level.
 *       Possible values returned are the following, or values in between:
 *        LW2080_CTRL_THERMAL_COOLER_LEVEL_OFF
 *        LW2080_CTRL_THERMAL_COOLER_LEVEL_FULL
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_GET_INFO_LEVEL_MINIMUM_DEFAULT_OPCODE (0x00001070U)
typedef struct LW2080_CTRL_THERMAL_COOLER_GET_INFO_LEVEL_MINIMUM_DEFAULT_OPERANDS {
    LwU32 coolerIndex;
    LwU32 level;
} LW2080_CTRL_THERMAL_COOLER_GET_INFO_LEVEL_MINIMUM_DEFAULT_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_COOLER_GET_INFO_LEVEL_MAXIMUM_DEFAULT instruction
 *
 * Get a cooler's default maximum operating level.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   policy
 *       Returns a cooler's default maximum operating level.
 *       Possible values returned are the following, or values in between:
 *        LW2080_CTRL_THERMAL_COOLER_LEVEL_OFF
 *        LW2080_CTRL_THERMAL_COOLER_LEVEL_FULL
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_GET_INFO_LEVEL_MAXIMUM_DEFAULT_OPCODE (0x00001080U)
typedef struct LW2080_CTRL_THERMAL_COOLER_GET_INFO_LEVEL_MAXIMUM_DEFAULT_OPERANDS {
    LwU32 coolerIndex;
    LwU32 level;
} LW2080_CTRL_THERMAL_COOLER_GET_INFO_LEVEL_MAXIMUM_DEFAULT_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_COOLER_GET_INFO_PERF_LEVEL_LEVEL_DEFAULT instruction
 *
 * Get a cooler's performance level default operating level.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   perfLevel
 *       Set by the client to the desired perf level to query.
 *
 *   level
 *       Returns a cooler's performance level default operating level.
 *       Possible values returned are the following, or values in between:
 *        LW2080_CTRL_THERMAL_COOLER_LEVEL_OFF
 *        LW2080_CTRL_THERMAL_COOLER_LEVEL_FULL
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_GET_INFO_PERF_LEVEL_LEVEL_DEFAULT_OPCODE (0x00001090U)
typedef struct LW2080_CTRL_THERMAL_COOLER_GET_INFO_PERF_LEVEL_LEVEL_DEFAULT_OPERANDS {
    LwU32 coolerIndex;
    LwU32 perfLevel;
    LwU32 level;
} LW2080_CTRL_THERMAL_COOLER_GET_INFO_PERF_LEVEL_LEVEL_DEFAULT_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_COOLER_GET_INFO_ACTIVITY_SENSE instruction
 *
 * Get a cooler's activity sensing support information.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   supported
 *       Set by the driver to one of the following:
 *        0: Cooler activity sensing is not supported.
 *        1: Cooler activity sensing is supported.
 *
 *   recommendedTimeout
 *       Set by the driver to indicate the time (in milliseconds) a client
 *       should wait from the issue of a reset instruction until the
 *       issue of a sense instruction.  If the cooler is not active
 *       after the timeout period, the cooler may be in a failure state.
 *       Possible values returned are the following:
 *        0: There is no known timeout period.
 *       !0: A value in milliseconds
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_GET_INFO_ACTIVITY_SENSE_OPCODE (0x000010A0U)
typedef struct LW2080_CTRL_THERMAL_COOLER_GET_INFO_ACTIVITY_SENSE_OPERANDS {
    LwU32 coolerIndex;
    LwU32 supported;
    LwU32 recommendedTimeout;
} LW2080_CTRL_THERMAL_COOLER_GET_INFO_ACTIVITY_SENSE_OPERANDS;


/*!
 * LW2080_CTRL_THERMAL_COOLER_GET_INFO_TACHOMETER_OPCODE instruction
 *
 * Returns the cooler's tachometer information.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   bSupported
 *       Returns whether this cooler supports the tachometer function.  If
 *       unsupported, reads via
 *       LW2080_CTRL_THERMAL_COOLER_GET_STATUS_TACHOMETER_OPCODE will fail:
 *       LW_FALSE/0 - Tachometer is unsupported
 *       LW_TRUE/1  - Tachometer is supported
 *
 *   bExpectedRPMSupported
 *       Returns whether this cooler supports callwlating the expected RPMs
 *       based on interpolation with the min and max data below.  If this bit is
 *       not LW_FALSE, clients cannot trust the data below.
 *       LW_FALSE/0 - Expected RPMS not supported
 *       LW_TRUE/1 - Expected RPMS supported
 *
 *   lowEndpointExpectedErrorPct
 *       Returns the expected error pct (as integer - scaled by 100) for this
 *       board/cooler at the min speed pct endpoint below.
 *
 *   highEndpointExpectedErrorPct
 *       Returns the expected error pct (as integer - scaled by 100) for this
 *       board/cooler at the max speed pct endpoint below.
 *
 *   interpolationExpectedErrorPct
 *       Returns the expected error pct (as integer - scaled by 100) for this
 *       board/cooler at all points interpolated between the corresponding
 *       min/max pct/RPM endpoints below.

 *   maxSpeedRPM
 *       Returns the RPMs of the cooler corresponding to maxSpeedPct.
 *
 *   maxSpeedPct
 *       Returns the fan speed percentage corresponding to maxSpeedRPM, meaning
 *       that at this fan speed setting the fan should spin at maxSpeedRPM RPMs.
 *       Note: This is a descriptive value of the fan cooler, not a policy
 *       setting for fan control.
 *
 *   minSpeedRPM
 *       Returns the RPMs of the cooler corresponding to minSpeedPct.
 *
 *   minSpeedPct
 *       Returns the fan speed percentage corresponding to minSpeedRPM, meaning
 *       that at this fan speed setting the fan should spin at minSpeedRPM RPMs.
 *       This is also the minimal fan speed percentage at which the fan will
 *       spin, below this setting the fan should flat line.
 *       Note: This is a descriptive value of the fan cooler, not a policy
 *       setting for fan control.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_GET_INFO_TACHOMETER_OPCODE (0x000010B0U)
typedef struct LW2080_CTRL_THERMAL_COOLER_GET_INFO_TACHOMETER_OPERANDS {
    LwU32  coolerIndex;
    LwBool bSupported;
    LwBool bExpectedRPMSupported;
    LwU32  lowEndpointExpectedErrorPct;
    LwU32  highEndpointExpectedErrorPct;
    LwU32  interpolationExpectedErrorPct;
    LwU32  maxSpeedRPM;
    LwU32  maxSpeedPct;
    LwU32  minSpeedRPM;
    LwU32  minSpeedPct;
} LW2080_CTRL_THERMAL_COOLER_GET_INFO_TACHOMETER_OPERANDS;

/*
 * LW2080_CTRL_THERMAL_COOLER_GET_STATUS instructions...
 *
 */

/*
 * LW2080_CTRL_THERMAL_COOLER_GET_STATUS_POLICY instruction
 *
 * Get a cooler's current control policy.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   policy
 *       Returns a cooler's current control policy.
 *       Possible values returned are:
 *          LW2080_CTRL_THERMAL_COOLER_POLICY_NONE
 *          LW2080_CTRL_THERMAL_COOLER_POLICY_MANUAL
 *          LW2080_CTRL_THERMAL_COOLER_POLICY_PERF
 *          LW2080_CTRL_THERMAL_COOLER_POLICY_TEMPERATURE_DISCRETE
 *          LW2080_CTRL_THERMAL_COOLER_POLICY_TEMPERATURE_CONTINUOUS
 *          LW2080_CTRL_THERMAL_COOLER_POLICY_TEMPERATURE_CONTINUOUS_SW
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_GET_STATUS_POLICY_OPCODE (0x00002010U)
typedef struct LW2080_CTRL_THERMAL_COOLER_GET_STATUS_POLICY_OPERANDS {
    LwU32 coolerIndex;
    LwU32 policy;
} LW2080_CTRL_THERMAL_COOLER_GET_STATUS_POLICY_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_COOLER_GET_STATUS_LEVEL instruction
 *
 * Get a cooler's current operating level.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   level
 *       Returns a cooler's current operating level.
 *       Possible values returned are the following, or values in between:
 *        LW2080_CTRL_THERMAL_COOLER_LEVEL_OFF
 *        LW2080_CTRL_THERMAL_COOLER_LEVEL_FULL
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_GET_STATUS_LEVEL_OPCODE (0x00002020U)
typedef struct LW2080_CTRL_THERMAL_COOLER_GET_STATUS_LEVEL_OPERANDS {
    LwU32 coolerIndex;
    LwU32 level;
} LW2080_CTRL_THERMAL_COOLER_GET_STATUS_LEVEL_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_COOLER_GET_STATUS_LEVEL_MINIMUM instruction
 *
 * Get a cooler's current minimum operating level.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   level
 *       Returns a cooler's current minimum operating level.
 *       Possible values returned are the following, or values in between:
 *        LW2080_CTRL_THERMAL_COOLER_LEVEL_OFF
 *        LW2080_CTRL_THERMAL_COOLER_LEVEL_FULL
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_GET_STATUS_LEVEL_MINIMUM_OPCODE (0x00002030U)
typedef struct LW2080_CTRL_THERMAL_COOLER_GET_STATUS_LEVEL_MINIMUM_OPERANDS {
    LwU32 coolerIndex;
    LwU32 level;
} LW2080_CTRL_THERMAL_COOLER_GET_STATUS_LEVEL_MINIMUM_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_COOLER_GET_STATUS_LEVEL_MAXIMUM instruction
 *
 * Get a cooler's current maximum operating level.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   level
 *       Returns a cooler's current maximum operating level.
 *       Possible values returned are the following, or values in between:
 *        LW2080_CTRL_THERMAL_COOLER_LEVEL_OFF
 *        LW2080_CTRL_THERMAL_COOLER_LEVEL_FULL
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_GET_STATUS_LEVEL_MAXIMUM_OPCODE (0x00002040U)
typedef struct LW2080_CTRL_THERMAL_COOLER_GET_STATUS_LEVEL_MAXIMUM_OPERANDS {
    LwU32 coolerIndex;
    LwU32 level;
} LW2080_CTRL_THERMAL_COOLER_GET_STATUS_LEVEL_MAXIMUM_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_COOLER_GET_STATUS_PERF_LEVEL_LEVEL instruction
 *
 * Get a cooler's performance level current operating level.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   perfLevel
 *       Set by the client to the desired perf level to query.
 *
 *   level
 *       Returns a perf level's cooler's operating level.
 *       Possible values returned are the following, or values in between:
 *        LW2080_CTRL_THERMAL_COOLER_LEVEL_OFF
 *        LW2080_CTRL_THERMAL_COOLER_LEVEL_FULL
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_GET_STATUS_PERF_LEVEL_LEVEL_OPCODE (0x00002050U)
typedef struct LW2080_CTRL_THERMAL_COOLER_GET_STATUS_PERF_LEVEL_LEVEL_OPERANDS {
    LwU32 coolerIndex;
    LwU32 perfLevel;
    LwU32 level;
} LW2080_CTRL_THERMAL_COOLER_GET_STATUS_PERF_LEVEL_LEVEL_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_COOLER_GET_STATUS_ACTIVITY_SENSE instruction
 *
 * Get a cooler's current activity sense state.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   active
 *       Set by the driver to one of the following:
 *        0: Cooler is not active.
 *        1: Cooler is active.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_GET_STATUS_ACTIVITY_SENSE_OPCODE (0x00002070U)
typedef struct LW2080_CTRL_THERMAL_COOLER_GET_STATUS_ACTIVITY_SENSE_OPERANDS {
    LwU32 coolerIndex;
    LwU32 active;
} LW2080_CTRL_THERMAL_COOLER_GET_STATUS_ACTIVITY_SENSE_OPERANDS;


/*!
 * LW2080_CTRL_THERMAL_COOLER_GET_STATUS_TACHOMETER_OPCODE instruction
 *
 * Returns the cooler's current tachometer reading.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   speedRPM
 *       Returns the current RPM of this cooler as read from the tachometer.  If
 *       tachometer is not supported for this cooler, value will be 0.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_GET_STATUS_TACHOMETER_OPCODE (0x00002080U)
typedef struct LW2080_CTRL_THERMAL_COOLER_GET_STATUS_TACHOMETER_OPERANDS {
    LwU32 coolerIndex;
    LwU32 speedRPM;
} LW2080_CTRL_THERMAL_COOLER_GET_STATUS_TACHOMETER_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_COOLER_GET_DIAG_ZONE_OPCODE
 *
 * Possible values are:
 *          LW2080_CTRL_COOLER_DIAG_ZONE_ILWALID
 *          LW2080_CTRL_COOLER_DIAG_ZONE_NORMAL
 *          LW2080_CTRL_COOLER_DIAG_ZONE_WARNING
 *          LW2080_CTRL_COOLER_DIAG_ZONE_CRITICAL
 */
typedef enum LW2080_CTRL_COOLER_DIAG_ZONE_TYPE {
    LW2080_CTRL_COOLER_DIAG_ZONE_ILWALID = 0,
    LW2080_CTRL_COOLER_DIAG_ZONE_NORMAL = 1,
    LW2080_CTRL_COOLER_DIAG_ZONE_WARNING = 2,
    LW2080_CTRL_COOLER_DIAG_ZONE_CRITICAL = 4,
} LW2080_CTRL_COOLER_DIAG_ZONE_TYPE;

// Update this if any additional levels are added
#define LW2080_CTRL_COOLER_VALID_DIAG_ZONES             3U

/*!
 * LW2080_CTRL_THERMAL_COOLER_GET_DIAG_ZONE_OPCODE instruction
 *
 * Returns the cooler's current zone.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   lwrrentZone
 *       Returns the current zone of this cooler as read from the tachometer.
 *        LW2080_CTRL_COOLER_DIAG_ZONE_NORMAL,
 *        LW2080_CTRL_COOLER_DIAG_ZONE_WARNING,
 *        LW2080_CTRL_COOLER_DIAG_ZONE_CRITICAL,
 *        LW2080_CTRL_COOLER_DIAG_ZONE_ILWALID,
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_THERMAL_COOLER_GET_DIAG_ZONE_OPCODE (0x00002090U)
typedef struct LW2080_CTRL_THERMAL_COOLER_GET_DIAG_ZONE_OPERANDS {
    LwU32                             coolerIndex;
    LW2080_CTRL_COOLER_DIAG_ZONE_TYPE lwrrentZone;
} LW2080_CTRL_THERMAL_COOLER_GET_DIAG_ZONE_OPERANDS;

/*
 * LW2080_CTRL_COOLER_DIAG_LIMIT_INFO
 *
 * variance   - This is the difference in RPM wrt to Expected RPM
 * zone       - Zone type. Possible values are:
 *              LW2080_CTRL_COOLER_DIAG_ZONE_WARNING,
 *              LW2080_CTRL_COOLER_DIAG_ZONE_CRITICAL,
 *              LW2080_CTRL_COOLER_DIAG_ZONE_ILWALID,
 *
 */
typedef struct LW2080_CTRL_COOLER_DIAG_LIMIT_INFO {
    LwU32                             variance;
    LW2080_CTRL_COOLER_DIAG_ZONE_TYPE zone;
} LW2080_CTRL_COOLER_DIAG_LIMIT_INFO;

/*
 * LW2080_CTRL_THERMAL_COOLER_GET_INFO_DIAG
 *
 * Get a GPU's zone limits information.
 *
 *   zoneLimits
 *       Set by the driver
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
  */
#define LW2080_CTRL_THERMAL_COOLER_GET_INFO_DIAG_OPCODE (0x000020A0U)
typedef struct LW2080_CTRL_THERMAL_COOLER_GET_INFO_DIAG_OPERANDS {
    LwU32                              coolerIndex;
    LW2080_CTRL_COOLER_DIAG_LIMIT_INFO zoneLimits[LW2080_CTRL_COOLER_VALID_DIAG_ZONES];
} LW2080_CTRL_THERMAL_COOLER_GET_INFO_DIAG_OPERANDS;

/*
 * LW2080_CTRL_THERMAL_COOLER_SET_CONTROL instructions...
 *
 */

/*
 * LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_POLICY instruction
 *
 * Set a cooler's current policy.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   policy
 *       Set by the client to the desired control policy.
 *       Possible values are:
 *          LW2080_CTRL_THERMAL_COOLER_POLICY_MANUAL
 *          LW2080_CTRL_THERMAL_COOLER_POLICY_PERF
 *          LW2080_CTRL_THERMAL_COOLER_POLICY_TEMPERATURE_DISCRETE
 *          LW2080_CTRL_THERMAL_COOLER_POLICY_TEMPERATURE_CONTINUOUS
 *          LW2080_CTRL_THERMAL_COOLER_POLICY_TEMPERATURE_CONTINUOUS_SW
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_POLICY_OPCODE (0x00003010U)
typedef struct LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_POLICY_OPERANDS {
    LwU32 coolerIndex;
    LwU32 policy;
} LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_POLICY_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_LEVEL instruction
 *
 * Set a cooler's current operating level.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   level
 *       Set by the client to the desired cooler level.
 *       Possible values are the following, or values in between:
 *        LW2080_CTRL_THERMAL_COOLER_LEVEL_OFF
 *        LW2080_CTRL_THERMAL_COOLER_LEVEL_FULL
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_LEVEL_OPCODE (0x00003020U)
typedef struct LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_LEVEL_OPERANDS {
    LwU32 coolerIndex;
    LwU32 level;
} LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_LEVEL_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_LEVEL_MINIMUM instruction
 *
 * Set a cooler's minimum operating level.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   level
 *       Set by the client to the desired minimum cooler level.
 *       Possible values are the following, or values in between:
 *       The instruction will fail if the value is greater than the
 *       current maximum operating level.
 *        LW2080_CTRL_THERMAL_COOLER_LEVEL_OFF
 *        LW2080_CTRL_THERMAL_COOLER_LEVEL_FULL
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_LEVEL_MINIMUM_OPCODE (0x00003030U)
typedef struct LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_LEVEL_MINIMUM_OPERANDS {
    LwU32 coolerIndex;
    LwU32 level;
} LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_LEVEL_MINIMUM_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_LEVEL_MAXIMUM instruction
 *
 * Set a cooler's maximum operating level.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   level
 *       Set by the client to the desired maximum cooler level.
 *       The instruction will fail if the value is less than the
 *       current minimum operating level.
 *       Possible values are the following, or values in between:
 *        LW2080_CTRL_THERMAL_COOLER_LEVEL_OFF
 *        LW2080_CTRL_THERMAL_COOLER_LEVEL_FULL
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_LEVEL_MAXIMUM_OPCODE (0x00003040U)
typedef struct LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_LEVEL_MAXIMUM_OPERANDS {
    LwU32 coolerIndex;
    LwU32 level;
} LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_LEVEL_MAXIMUM_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_PERF_LEVEL_LEVEL instruction
 *
 * Set a cooler's performance level current operating level.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   perfLevel
 *       Set by the client to the desired performance level.
 *
 *   level
 *       Set by the client to the desired cooler level.
 *       Possible values are the following, or values in between:
 *        LW2080_CTRL_THERMAL_COOLER_LEVEL_OFF
 *        LW2080_CTRL_THERMAL_COOLER_LEVEL_FULL
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_PERF_LEVEL_LEVEL_OPCODE (0x00003050U)
typedef struct LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_PERF_LEVEL_LEVEL_OPERANDS {
    LwU32 coolerIndex;
    LwU32 perfLevel;
    LwU32 level;
} LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_PERF_LEVEL_LEVEL_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_ACTIVITY_SENSE instruction
 *
 * Reset a cooler's current activity sense state.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   reset
 *       Set by the client to reset the activity sense state.
 *       Possible values are the following:
 *        0: Ignore
 *        1: Reset the activity sense state
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_ACTIVITY_SENSE_OPCODE (0x00003070U)
typedef struct LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_ACTIVITY_SENSE_OPERANDS {
    LwU32 coolerIndex;
    LwU32 reset;
} LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_ACTIVITY_SENSE_OPERANDS;


/*
 * LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_DEFAULTS instruction
 *
 * Restore a cooler setting(s) to default.
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   defaultsType
 *       Set by the client to select which defaults.
 *       Possible values are the following:
 *        LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_DEFAULTS_NONE
 *        LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_DEFAULTS_ALL
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_DEFAULTS_OPCODE (0x00003080U)
typedef struct LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_DEFAULTS_OPERANDS {
    LwU32 coolerIndex;
    LwU32 defaultsType;
} LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_DEFAULTS_OPERANDS;

#define LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_DEFAULTS_NONE (0x00000000U)
#define LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_DEFAULTS_ALL  (0xFFFFFFFFU)

/*
 * RPMSIM related instructions....
 *
 */

/*
 * LW2080_CTRL_THERMAL_COOLER_GET_RPMSIM_CONTROL instruction
 *
 * Get the coolers's current RPMSIM control value.
 *
 *   sensorIndex
 *       Set by the client to the desired sensor index.
 *
 *   bControl
 *       Set by the driver to current control
 *       i.e. LW_TRUE/LW_FALSE
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_THERMAL_COOLER_GET_RPMSIM_CONTROL_OPCODE (0x00003090U)
typedef struct LW2080_CTRL_THERMAL_COOLER_GET_RPMSIM_CONTROL_OPERANDS {
    LwU32  coolerIndex;
    LwBool bControl;
} LW2080_CTRL_THERMAL_COOLER_GET_RPMSIM_CONTROL_OPERANDS;

/*
 * LW2080_CTRL_THERMAL_COOLER_SET_RPMSIM_CONTROL_AND_VALUE instruction
 *
 * Set the cooler's current RPMSim control and value
 *
 *   coolerIndex
 *       Set by the client to the desired cooler index.
 *
 *   bControl
 *       Set by the client to the desired control
 *       i.e. LW_TRUE/LW_FALSE
 *
 *   value
 *       Set by the client to the desired simulation value
 *       if (control == LW_TRUE).
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_THERMAL_COOLER_SET_RPMSIM_CONTROL_AND_VALUE_OPCODE (0x000030A0U)
typedef struct LW2080_CTRL_THERMAL_COOLER_SET_RPMSIM_CONTROL_AND_VALUE_OPERANDS {
    LwU32  coolerIndex;
    LwBool bControl;
    LwU32  value;
} LW2080_CTRL_THERMAL_COOLER_SET_RPMSIM_CONTROL_AND_VALUE_OPERANDS;

/*!
 * Macros for Thermal Device Classes
 */
#define LW2080_CTRL_THERMAL_THERM_DEVICE_CLASS_ILWALID                          0x00U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_CLASS_GPU                              0x01U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_CLASS_GPU_GPC_TSOSC                    0x02U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_CLASS_GPU_SCI                          0x03U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_CLASS_GPU_GPC_COMBINED                 0x04U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_CLASS_GDDR6_X_COMBINED                 0x05U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_CLASS_I2CS_GT21X                       0x20U // Not supported on Kepler and deprecated from Maxwell
#define LW2080_CTRL_THERMAL_THERM_DEVICE_CLASS_I2CS_GF11X                       0x21U // Not supported on Kepler and deprecated from Maxwell
#define LW2080_CTRL_THERMAL_THERM_DEVICE_CLASS_I2C_ADT7473_1                    0x40U // Not supported on Kepler and deprecated from Maxwell
#define LW2080_CTRL_THERMAL_THERM_DEVICE_CLASS_I2C_ADM1032                      0x41U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_CLASS_I2C_VT1165                       0x42U // Not supported on Kepler and deprecated from Maxwell
#define LW2080_CTRL_THERMAL_THERM_DEVICE_CLASS_I2C_MAX6649                      0x43U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_CLASS_I2C_TMP411                       0x44U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_CLASS_I2C_ADT7461                      0x45U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_CLASS_I2C_TMP451                       0x46U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_CLASS_RM                               0x60U // Not supported on Kepler and deprecated from Maxwell
#define LW2080_CTRL_THERMAL_THERM_DEVICE_CLASS_HBM2_SITE                        0x70U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_CLASS_HBM2_COMBINED                    0x71U

/*!
 * Macros for Thermal Channel Classes
 */
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_CLASS_ILWALID                         0x00U
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_CLASS_DEVICE                          0x01U

/*!
 * Macros for Thermal Channel Types
 */
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_TYPE_GPU_AVG                          0x00U
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_TYPE_GPU_MAX                          0x01U
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_TYPE_BOARD                            0x02U
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_TYPE_MEMORY                           0x03U
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_TYPE_PWR_SUPPLY                       0x04U
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_TYPE_MAX_COUNT                        0x05U

/*!
 * Special value corresponding to an invalid Channel Index.
 */
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_INDEX_ILWALID                         LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Macros for Channel's Relative Location
 */
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_REL_LOC_INT                           0x00U
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_REL_LOC_EXT                           0x01U

/*!
 * Macros for Channel's target GPU
 */
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_TGT_GPU_0                             0x00U
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_TGT_GPU_1                             0x01U
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_TGT_GPU_2                             0x02U
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_TGT_GPU_3                             0x03U

/*!
 * Macros for Channel's flags.
 *
 * @note Must be kept in sync with @ref
 *  LW_VBIOS_THERM_CHANNEL_1X_ENTRY_FLAGS_CHANNEL_<xyz>
 */
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_FLAGS_CHANNEL_DRIVE_HW                0:0
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_FLAGS_CHANNEL_DRIVE_HW_NO             0x0U
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_FLAGS_CHANNEL_DRIVE_HW_YES            0x1U
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_FLAGS_CHANNEL_RSVD                    5:1
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_FLAGS_CHANNEL_VISIBLE_PUBLIC_IB       6:6
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_FLAGS_CHANNEL_VISIBLE_PUBLIC_IB_NO    0x0U
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_FLAGS_CHANNEL_VISIBLE_PUBLIC_IB_YES   0x1U
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_FLAGS_CHANNEL_VISIBLE_PUBLIC_OOB      7:7
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_FLAGS_CHANNEL_VISIBLE_PUBLIC_OOB_NO   0x0U
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_FLAGS_CHANNEL_VISIBLE_PUBLIC_OOB_YES  0x1U

/*!
 * Macros for Thermal Monitor Classes.
 */
#define LW2080_CTRL_THERMAL_THERM_MONITOR_CLASS_ILWALID                         0x00U
#define LW2080_CTRL_THERMAL_THERM_MONITOR_CLASS_VOLTAGE_REGULATOR               0x01U
#define LW2080_CTRL_THERMAL_THERM_MONITOR_CLASS_BLOCK_ACTIVITY                  0x02U
#define LW2080_CTRL_THERMAL_THERM_MONITOR_CLASS_EDPP_VMIN                       0x03U
#define LW2080_CTRL_THERMAL_THERM_MONITOR_CLASS_EDPP_FONLY                      0x04U
#define LW2080_CTRL_THERMAL_THERM_MONITOR_CLASS_ADC_IPC                         0x05U
#define LW2080_CTRL_THERMAL_THERM_MONITOR_CLASS_ADC_VID                         0x06U

/*!
 * Macros for Thermal Monitor Types.
 */
#define LW2080_CTRL_THERMAL_THERM_MONITOR_TYPE_ILWALID                          0x00U
#define LW2080_CTRL_THERMAL_THERM_MONITOR_TYPE_EDGE                             0x01U
#define LW2080_CTRL_THERMAL_THERM_MONITOR_TYPE_LEVEL                            0x02U

/*!
 * Special macro for invalid Thermal Monitor index.
 */
#define LW2080_CTRL_THERMAL_THERM_MONITOR_IDX_ILWALID                           LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Providers supported by GPU THERM_DEVICE:
 *
 * TSENSE               - GPU TSENSE internal thermal sensor temperature
 * TSENSE_OFFSET        - Offsetted GPU TSENSE internal thermal sensor temperature
 * CONST                - Constant temperature
 * MAX                  - Maximum of one or more providers
 * GPU_MAX              - GPU max temperature
 * GPU_AVG              - GPU average temperaure
 * GPU_OFFSET_MAX       - Maximum of (GPU + HS)
 * GPU_OFFSET_AVG       - Average of (GPU + HS)
 * DYNAMIC_HOTSPOT      - Hybrid of TSENSE and TSOSC
 * NUM_PROVS            - Maximum number of supported providers
 */
#define LW2080_CTRL_THERMAL_THERM_DEVICE_GPU_PROV_TSENSE                        0x00U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_GPU_PROV_TSENSE_OFFSET                 0x01U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_GPU_PROV_CONST                         0x02U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_GPU_PROV_MAX                           0x03U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_GPU_PROV_GPU_MAX                       0x04U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_GPU_PROV_GPU_AVG                       0x05U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_GPU_PROV_GPU_OFFSET_MAX                0x06U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_GPU_PROV_GPU_OFFSET_AVG                0x07U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_GPU_PROV_DYNAMIC_HOTSPOT               0x08U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_GPU_PROV_DYNAMIC_HOTSPOT_OFFSET        0x09U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_GPU_PROV__NUM_PROVS                    (0xaU) /* finn: Evaluated from "(LW2080_CTRL_THERMAL_THERM_DEVICE_GPU_PROV_DYNAMIC_HOTSPOT_OFFSET + 1)" */

/*!
 * Providers supported by GPU_GPC_TSOSC THERM_DEVICE:
 *
 * TSOSC        - GPU_GPC_TSOSC internal thermal sensor temperature
 * TSOSC_OFFSET - Offsetted GPU_GPC_TSOSC internal thermal sensor temperature
 * NUM_PROVS    - Maximum number of supported providers
 */
#define LW2080_CTRL_THERMAL_THERM_DEVICE_GPU_GPC_TSOSC_PROV_TSOSC               0x00U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_GPU_GPC_TSOSC_PROV_TSOSC_OFFSET        0x01U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_GPU_GPC_TSOSC_PROV__NUM_PROVS          (0x2U) /* finn: Evaluated from "(LW2080_CTRL_THERMAL_THERM_DEVICE_GPU_GPC_TSOSC_PROV_TSOSC_OFFSET + 1)" */

/*!
 * Providers supported by GPU_SCI device:
 *
 * MINI_TSENSE  - GPU_SCI internal thermal sensor temperature
 * NUM_PROVS    - Maximum number of supported providers
 */
#define LW2080_CTRL_THERMAL_THERM_DEVICE_GPU_SCI_PROV_MINI_TSENSE               0x00U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_GPU_SCI_PROV__NUM_PROVS                (0x1U) /* finn: Evaluated from "(LW2080_CTRL_THERMAL_THERM_DEVICE_GPU_SCI_PROV_MINI_TSENSE + 1)" */

/*!
 * Providers supported by TMP411/ADM1032/MAX6649 I2C THERM_DEVICE:
 *
 * LOW_PRECISION_EXT - Low precision external thermal sensor temperature
 * LOW_PRECISION_INT - Low precision internal thermal sensor temperature
 * NUM_PROVS         - Maximum number of supported providers
 *
 * @note ADM1032/MAX6649 is capable of providing high precision external
 * thermal sensor temperature but it is lwrrently not supported.
 *
 * @note TMP411 is capable of providing high precision external as well as
 * internal thermal sensor temperature but it is lwrrently not supported.
 * Lwrrently used functionality is compatible with ADM1032/MAX6649 and hence
 * RM/PMU can safely share providers for all these THERM_DEVICEs.
 */
#define LW2080_CTRL_THERMAL_THERM_DEVICE_I2C_TMP411_PROV_LOW_PRECISION_EXT      0x00U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_I2C_TMP411_PROV_LOW_PRECISION_INT      0x01U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_I2C_TMP411_PROV_NUM_PROVS              (0x2U) /* finn: Evaluated from "(LW2080_CTRL_THERMAL_THERM_DEVICE_I2C_TMP411_PROV_LOW_PRECISION_INT + 1)" */

/*!
 * Providers supported by HBM2_SITE THERM_DEVICE:
 *
 * DEFAULT    - HBM2_SITE internal thermal sensor temperature
 * NUM_PROVS  - Maximum number of supported providers
 */
#define LW2080_CTRL_THERMAL_THERM_DEVICE_HBM2_SITE_PROV_DEFAULT                 0x00U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_HBM2_SITE_PROV__NUM_PROVS              (0x1U) /* finn: Evaluated from "(LW2080_CTRL_THERMAL_THERM_DEVICE_HBM2_SITE_PROV_DEFAULT + 1)" */

/*!
 * Providers supported by HBM2_COMBINED THERM_DEVICE:
 *
 * MAX        - HBM2 MAX temperature among all sites.
 * NUM_PROVS  - Maximum number of supported providers.
 */
#define LW2080_CTRL_THERMAL_THERM_DEVICE_HBM2_COMBINED_PROV_MAX                 0x00U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_HBM2_COMBINED_PROV__NUM_PROVS          (0x1U) /* finn: Evaluated from "(LW2080_CTRL_THERMAL_THERM_DEVICE_HBM2_COMBINED_PROV_MAX + 1)" */

/*!
 * Providers supported by GPU_GPC_COMBINED THERM_DEVICE:
 *
 * GPC_AVG_UNMUNGED - GPC internal thermal sensor AVG GPC's Unmunged temperature.
 * GPC_AVG_MUNGED   - GPC internal thermal sensor AVG GPC's Munged temperature
 * NUM_PROVS    - Maximum number of supported providers
 */
#define LW2080_CTRL_THERMAL_THERM_DEVICE_GPU_GPC_COMBINED_PROV_GPC_AVG_UNMUNGED 0x00U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_GPU_GPC_COMBINED_PROV_GPC_AVG_MUNGED   0x01U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_GPU_GPC_COMBINED_PROV__NUM_PROVS       (0x2U) /* finn: Evaluated from "(LW2080_CTRL_THERMAL_THERM_DEVICE_GPU_GPC_COMBINED_PROV_GPC_AVG_MUNGED + 1)" */

/*!
 * Providers supported by GDDR6_X_COMBINED THERM_DEVICE:
 *
 * MAX       - GDDR6/GDDR6X MAX temperature among all partitions
 * NUM_PROVS - Maximum number of supported providers
 */
#define LW2080_CTRL_THERMAL_THERM_DEVICE_GDDR6_X_COMBINED_PROV_MAX              0x00U
#define LW2080_CTRL_THERMAL_THERM_DEVICE_GDDR6_X_COMBINED_PROV__NUM_PROVS       (0x1U) /* finn: Evaluated from "(LW2080_CTRL_THERMAL_THERM_DEVICE_GDDR6_X_COMBINED_PROV_MAX + 1)" */

/*
 * Special value corresponding to invalid number of THERM_DEVICE providers.
 * This is not specific to any THERM_DEVICE.
 */
#define LW2080_CTRL_THERMAL_THERM_DEVICE_PROV__NUM_PROV_ILWALID                 0XFFU

/*
 * PMU FAN Control related instructions....
 *
 */

#define LW2080_CTRL_PMU_FAN_PRIVATE_DATA_HEADER_REVISION_MAJOR_REVISION            7:4
#define LW2080_CTRL_PMU_FAN_PRIVATE_DATA_HEADER_REVISION_MAJOR_REVISION_1X      0x00000001U
#define LW2080_CTRL_PMU_FAN_PRIVATE_DATA_HEADER_REVISION_MINOR_REVISION            3:0
#define LW2080_CTRL_PMU_FAN_PRIVATE_DATA_HEADER_REVISION_MINOR_REVISION_0       0x00000000U

typedef struct LW2080_CTRL_PMU_FAN_PRIVATE_DATA_HEADER {
    LwU8 revision;  //The private data revision
    LwU8 size;      //The private data size
} LW2080_CTRL_PMU_FAN_PRIVATE_DATA_HEADER;

#define LW2080_CTRL_PMU_FAN_PRIVATE_DATA_1X_CONTROL_FIELD_ENABLE                    0:0
#define LW2080_CTRL_PMU_FAN_PRIVATE_DATA_1X_CONTROL_FIELD_ENABLE_DISABLED  (0x00000000U)
#define LW2080_CTRL_PMU_FAN_PRIVATE_DATA_1X_CONTROL_FIELD_ENABLE_ENABLED   (0x00000001U)
#define LW2080_CTRL_PMU_FAN_PRIVATE_DATA_1X_CONTROL_FIELD_MODE                      2:1
#define LW2080_CTRL_PMU_FAN_PRIVATE_DATA_1X_CONTROL_FIELD_MODE_AUTOMATIC   (0x00000000U)
#define LW2080_CTRL_PMU_FAN_PRIVATE_DATA_1X_CONTROL_FIELD_MODE_MANUAL      (0x00000001U)
#define LW2080_CTRL_PMU_FAN_PRIVATE_DATA_1X_CONTROL_FIELD_GEMINI                    4:3
#define LW2080_CTRL_PMU_FAN_PRIVATE_DATA_1X_CONTROL_FIELD_GEMINI_DISABLED  (0x00000000U)
#define LW2080_CTRL_PMU_FAN_PRIVATE_DATA_1X_CONTROL_FIELD_GEMINI_MASTER    (0x00000001U)
#define LW2080_CTRL_PMU_FAN_PRIVATE_DATA_1X_CONTROL_FIELD_GEMINI_SLAVE     (0x00000002U)
#define LW2080_CTRL_PMU_FAN_PRIVATE_DATA_1X_CONTROL_FIELD_GEMINI_SHARING   (0x00000003U)
#define LW2080_CTRL_PMU_FAN_PRIVATE_DATA_1X_CONTROL_FIELD_ALGORITHM                7:5
#define LW2080_CTRL_PMU_FAN_PRIVATE_DATA_1X_CONTROL_FIELD_ALGORITHM_LINEAR (0x00000000U)
#define LW2080_CTRL_PMU_FAN_PRIVATE_DATA_1X_CONTROL_FIELD_ALGORITHM_2_SEG  (0x00000001U)

typedef struct LW2080_CTRL_PMU_FAN_PRIVATE_DATA_1X {
    //Configuration fields
    LwSFXP8_8  m;              //Fixed point M - F8.8 colwerts temperature/100 C to PWM fraction
    LwSFXP8_8  b;              //Fixed point B - F8.8 typically m=1...3 and b=0..-0.5
    LwSFXP8_8  accelGain;      //Fan acceleration gain - F8.8 typically +2...-0.9
    LwSFXP8_8  midpointGain;   //Gain to use in history - F8.8
    LwU8       historyCount;   //Number of second samples to keep - unsigned units
    LwU8       lookback;       //Number of samples for lookback smoothing for acceleration - maximum of 10
    LwU16      controlField;   //Bit field for various PMU control bits - see macros above
    LwSFXP8_8  slopeLimit0;    //Minimum slope of EF line segment, typically 1 - F8.8
    LwSFXP8_8  slope1;         //Fixed slope of FG line segment, typically 2 - F8.8
    LwSFXP11_5 gravity;        //Gravity to add to Fx at every iteration
} LW2080_CTRL_PMU_FAN_PRIVATE_DATA_1X;

typedef struct LW2080_CTRL_PMU_FAN_PRIVATE_DATA {
    LW2080_CTRL_PMU_FAN_PRIVATE_DATA_HEADER header;
    LW2080_CTRL_PMU_FAN_PRIVATE_DATA_1X     pvtData1X; //Private data revision 1.X
} LW2080_CTRL_PMU_FAN_PRIVATE_DATA;

/*!
 *  Available PWM sources:
 *
 *  RM_PMU_PMGR_PWM_SOURCE_ILWALID
 *      Default return value in all error cases (when PWM source expected).
 *  RM_PMU_PMGR_PWM_SOURCE_PMGR_FAN
 *      PWM source is PMGR "FAN" that allows OVERTEMP override.
 *  RM_PMU_PMGR_PWM_SOURCE_PMGR_PWM
 *      PWM source is PMGR "PWM".
 *  RM_PMU_PMGR_PWM_SOURCE_THERM_PWM
 *      PWM source belongs to OBJTHERM (GF11x+) but here we are providing common
 *      interface to access it (SmartFan).
 */
typedef enum LW2080_CTRL_PMU_PMGR_PWM_SOURCE {
    LW2080_CTRL_PMU_PMGR_PWM_SOURCE_ILWALID = 0,
    LW2080_CTRL_PMU_PMGR_PWM_SOURCE_PMGR_FAN = 1,
    LW2080_CTRL_PMU_PMGR_PWM_SOURCE_PMGR_PWM = 2,
    LW2080_CTRL_PMU_PMGR_PWM_SOURCE_THERM_PWM = 3,
} LW2080_CTRL_PMU_PMGR_PWM_SOURCE;

#define LW2080_CTRL_PMU_FAN_PWM_SOURCE                     2:0
#define LW2080_CTRL_PMU_FAN_PWM_ILWERT                     3:3
#define LW2080_CTRL_PMU_FAN_PWM_ILWERT_DISABLED 0x00000000U
#define LW2080_CTRL_PMU_FAN_PWM_ILWERT_ENABLED  0x00000001U

#define LW2080_CTRL_PMU_FAN_RAMP_SLOPE_NOT_USED 0U

typedef struct LW2080_CTRL_PMU_FAN_FAN_DESCRIPTION {
    LwU8       pwmPctMin;        //Minimum permitted fan percent 0..100, typically 0..40%
    LwU8       pwmPctMax;        //Maximum permitted fan percent 0..100, typically TDPMAX+10% 90..100
    LwU8       pwmPctManual;     //Raw duty cycle for PWM, used in manual mode, typically less than 16 bits
    LwU8       pwmSource;        //The pwm source which is driving the GPIO - used to determine which registers to write.
    LwSFXP4_12 pwmScaleSlope;    //The slope/m for scaling eletrical pwm % -> electrical pwm duty cycle - FXP4.12
    LwSFXP4_12 pwmScaleOffset;   //The offset/b for scaling electrical pwm % -> electrical pwm duty cycle - FXP4.12
    LwU32      pwmRawPeriod;     //Raw period for PWM, typically less than 16 bits
    LwU16      pwmRampUpSlope;   //Ramp up slope in ms / %    - This is an *optional* parameter
    LwU16      pwmRampDownSlope; //Ramp down slope in ms / %  - This is an *optional* parameter
} LW2080_CTRL_PMU_FAN_FAN_DESCRIPTION;

typedef struct LW2080_CTRL_PMU_FAN_CONTROL_BLOCK {
    LW2080_CTRL_PMU_FAN_FAN_DESCRIPTION fanDesc;
    LW2080_CTRL_PMU_FAN_PRIVATE_DATA    pvtData;
} LW2080_CTRL_PMU_FAN_CONTROL_BLOCK;

/*
 * LW2080_CTRL_THERMAL_COOLER_GET_PMU_FAN_CONTROL_BLOCK instruction
 *
 * Get current PMU fan control block.
 *
 *   sensorIndex
 *       Set by the client to the desired sensor index.
 *
 *   pmuFanControlBlock
 *       Set by the driver to current PMU fan control block.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_THERMAL_COOLER_GET_PMU_FAN_CONTROL_BLOCK_OPCODE (0x000030B0U)
typedef struct LW2080_CTRL_THERMAL_COOLER_GET_PMU_FAN_CONTROL_BLOCK_OPERANDS {
    LwU32                             coolerIndex;
    LW2080_CTRL_PMU_FAN_CONTROL_BLOCK pmuFanControlBlock;
} LW2080_CTRL_THERMAL_COOLER_GET_PMU_FAN_CONTROL_BLOCK_OPERANDS;

/*
 * LW2080_CTRL_THERMAL_COOLER_SET_PMU_FAN_CONTROL_BLOCK instruction
 *
 * Set current PMU fan control block with values supplied by the client.
 *
 *   sensorIndex
 *       Set by the client to the desired sensor index.
 *
 *   pmuFanControlBlock
 *       Set by the client to desired PMU fan control block.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW2080_CTRL_THERMAL_COOLER_SET_PMU_FAN_CONTROL_BLOCK_OPCODE (0x000030C0U)
typedef struct LW2080_CTRL_THERMAL_COOLER_SET_PMU_FAN_CONTROL_BLOCK_OPERANDS {
    LwU32                             coolerIndex;
    LW2080_CTRL_PMU_FAN_CONTROL_BLOCK pmuFanControlBlock;
} LW2080_CTRL_THERMAL_COOLER_SET_PMU_FAN_CONTROL_BLOCK_OPERANDS;

/*
 * Cooler instruction operand
 */
typedef union LW2080_CTRL_THERMAL_COOLER_INSTRUCTION_OPERANDS {

       /*
        *  GetInfo instruction operands...
        */
    LW2080_CTRL_THERMAL_COOLER_GET_INFO_AVAILABLE_OPERANDS                getInfoAvailable;
    LW2080_CTRL_THERMAL_COOLER_GET_INFO_TARGETS_OPERANDS                  getInfoTargets;
    LW2080_CTRL_THERMAL_COOLER_GET_INFO_PHYSICAL_OPERANDS                 getInfoPhysical;
    LW2080_CTRL_THERMAL_COOLER_GET_INFO_CONTROLLER_OPERANDS               getInfoController;
    LW2080_CTRL_THERMAL_COOLER_GET_INFO_SIGNAL_OPERANDS                   getInfoSignal;
    LW2080_CTRL_THERMAL_COOLER_GET_INFO_POLICIES_OPERANDS                 getInfoPolicies;
    LW2080_CTRL_THERMAL_COOLER_GET_INFO_POLICY_DEFAULT_OPERANDS           getInfoPolicyDefault;
    LW2080_CTRL_THERMAL_COOLER_GET_INFO_LEVEL_MINIMUM_DEFAULT_OPERANDS    getInfoLevelMinimumDefault;
    LW2080_CTRL_THERMAL_COOLER_GET_INFO_LEVEL_MAXIMUM_DEFAULT_OPERANDS    getInfoLevelMaximumDefault;
    LW2080_CTRL_THERMAL_COOLER_GET_INFO_PERF_LEVEL_LEVEL_DEFAULT_OPERANDS getInfoPerfLevelLevelDefault;
    LW2080_CTRL_THERMAL_COOLER_GET_INFO_ACTIVITY_SENSE_OPERANDS           getInfoActivitySense;
    LW2080_CTRL_THERMAL_COOLER_GET_INFO_TACHOMETER_OPERANDS               getInfoTachometer;

        /*
         *  GetStatus instruction opcodes...
         */
    LW2080_CTRL_THERMAL_COOLER_GET_STATUS_POLICY_OPERANDS                 getStatusPolicy;
    LW2080_CTRL_THERMAL_COOLER_GET_STATUS_LEVEL_OPERANDS                  getStatusLevel;
    LW2080_CTRL_THERMAL_COOLER_GET_STATUS_LEVEL_MINIMUM_OPERANDS          getStatusLevelMinimum;
    LW2080_CTRL_THERMAL_COOLER_GET_STATUS_LEVEL_MAXIMUM_OPERANDS          getStatusLevelMaximum;
    LW2080_CTRL_THERMAL_COOLER_GET_STATUS_PERF_LEVEL_LEVEL_OPERANDS       getStatusPerfLevelLevel;
    LW2080_CTRL_THERMAL_COOLER_GET_STATUS_ACTIVITY_SENSE_OPERANDS         getStatusActivitySense;
    LW2080_CTRL_THERMAL_COOLER_GET_STATUS_TACHOMETER_OPERANDS             getStatusTachometer;

        /*
         * Diag related instruction operands...
        */
    LW2080_CTRL_THERMAL_COOLER_GET_DIAG_ZONE_OPERANDS                     getLwrrentZone;
    LW2080_CTRL_THERMAL_COOLER_GET_INFO_DIAG_OPERANDS                     getInfoDiag;

        /*
         *  SetControl instruction opcodes...
         */
    LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_POLICY_OPERANDS                setControlPolicy;
    LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_LEVEL_OPERANDS                 setControlLevel;
    LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_LEVEL_MINIMUM_OPERANDS         setControlLevelMinimum;
    LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_LEVEL_MAXIMUM_OPERANDS         setControlLevelMaximum;
    LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_PERF_LEVEL_LEVEL_OPERANDS      setControlPerfLevelLevel;
    LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_ACTIVITY_SENSE_OPERANDS        setControlActivitySense;
    LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_DEFAULTS_OPERANDS              setControlDefaults;

        /*
         *  RPMSim instruction opcodes...
         */
    LW2080_CTRL_THERMAL_COOLER_GET_RPMSIM_CONTROL_OPERANDS                getRPMSimControl;
    LW2080_CTRL_THERMAL_COOLER_SET_RPMSIM_CONTROL_AND_VALUE_OPERANDS      setRPMSimControlAndValue;

        /*
         *  PMU fan instruction opcodes...
         */
    LW2080_CTRL_THERMAL_COOLER_GET_PMU_FAN_CONTROL_BLOCK_OPERANDS         getPmuFanControlBlock;
    LW2080_CTRL_THERMAL_COOLER_SET_PMU_FAN_CONTROL_BLOCK_OPERANDS         setPmuFanControlBlock;

        /*
         *  Minimum operand size. This should be larger than the largest operand,
         *  for growth, but please don't change, as it will break the api.
         *  Client intstruction _must_ match driver instruction size.
         *
         * XAPIGEN: mark "hidden" since not needed for XAPI.
         */
    LwU32                                                                 space[8];
} LW2080_CTRL_THERMAL_COOLER_INSTRUCTION_OPERANDS;



/*
 * LW2080_CTRL_THERMAL_COOLER_INSTRUCTION
 *
 * All cooler instructions have the following layout:
 *
 *   result
 *       This field is set by the driver, and is the result of the
 *       instruction's exelwtion. This value is only valid if the
 *       exelwted field is not 0 upon return.
 *       Possible status values returned are:
 *        LW_OK
 *        LW_ERR_ILWALID_ARGUMENT
 *        LW_ERR_ILWALID_PARAM_STRUCT
 *
 *   exelwted
 *       This field is set by the driver, and
 *       indicates if the instruction was exelwted.
 *       Possible status values returned are:
 *        0: Not exelwted
 *        1: Exelwted
 *
 *   opcode
 *       This field is set by the client to the desired instruction opcode.
 *       Possible values are:
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_AVAILABLE_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_TARGETS_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_PHYSICAL_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_CONTROLLER_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_SIGNAL_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_POLICIES_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_POLICY_DEFAULT_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_LEVEL_MINIMUM_DEFAULT_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_LEVEL_MAXIMUM_DEFAULT_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_PERF_LEVEL_LEVEL_DEFAULT_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_ACTIVITY_SENSE_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_TACHOMETER_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_GET_STATUS_POLICY_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_GET_STATUS_LEVEL_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_GET_STATUS_LEVEL_MINIMUM_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_GET_STATUS_LEVEL_MAXIMUM_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_GET_STATUS_PERF_LEVEL_LEVEL_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_GET_STATUS_ACTIVITY_SENSE_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_GET_STATUS_TACHOMETER_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_GET_DIAG_ZONE_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_DIAG_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_POLICY_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_LEVEL_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_LEVEL_MINIMUM_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_LEVEL_MAXIMUM_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_PERF_LEVEL_LEVEL_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_ACTIVITY_SENSE_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_DEFAULTS_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_GET_RPMSIM_CONTROL_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_SET_RPMSIM_CONTROL_AND_VALUE_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_GET_PMU_FAN_CONTROL_BLOCK_OPCODE
 *        LW2080_CTRL_THERMAL_COOLER_SET_PMU_FAN_CONTROL_BLOCK_OPCODE
 *
 *   operands
 *       This field is actually a union of all of the available operands.
 *       The interpretation of this field is opcode context dependent.
 *       Possible values are:
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_AVAILABLE_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_TARGETS_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_PHYSICAL_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_CONTROLLER_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_SIGNAL_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_POLICIES_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_POLICY_DEFAULT_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_LEVEL_MINIMUM_DEFAULT_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_LEVEL_MAXIMUM_DEFAULT_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_PERF_LEVEL_LEVEL_DEFAULT_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_ACTIVITY_SENSE_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_TACHOMETER_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_GET_STATUS_POLICY_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_GET_STATUS_LEVEL_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_GET_STATUS_LEVEL_MINIMUM_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_GET_STATUS_LEVEL_MAXIMUM_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_GET_STATUS_PERF_LEVEL_LEVEL_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_GET_STATUS_ACTIVITY_SENSE_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_GET_STATUS_TACHOMETER_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_GET_DIAG_ZONE_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_GET_INFO_DIAG_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_POLICY_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_LEVEL_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_LEVEL_MINIMUM_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_LEVEL_MAXIMUM_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_PERF_LEVEL_LEVEL_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_ACTIVITY_SENSE_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_SET_CONTROL_DEFAULTS_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_GET_RPMSIM_CONTROL_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_SET_RPMSIM_CONTROL_AND_VALUE_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_GET_PMU_FAN_CONTROL_BLOCK_OPERANDS
 *        LW2080_CTRL_THERMAL_COOLER_SET_PMU_FAN_CONTROL_BLOCK_OPERANDS
 *
 */
typedef struct LW2080_CTRL_THERMAL_COOLER_INSTRUCTION {
    LwU32                                           result;
    LwU32                                           exelwted;
    LwU32                                           opcode;
    // C Form: LW2080_CTRL_THERMAL_COOLER_INSTRUCTION_OPERANDS operands;
    LW2080_CTRL_THERMAL_COOLER_INSTRUCTION_OPERANDS operands;
} LW2080_CTRL_THERMAL_COOLER_INSTRUCTION;


/*
 * LW2080_CTRL_THERMAL_GET_THERMAL_SETUP
 *
 *   Possible Thermal setup state.
 *     LW2080_CTRL_THERMAL_SETUP_CORRECT
 *       Thermal setup is correct.
 *
 *     LW2080_CTRL_THERMAL_SETUP_WRONG
 *       Thermal setup is wrong.
 */
#define LW2080_CTRL_THERMAL_SETUP_CORRECT (0x00000000U)
#define LW2080_CTRL_THERMAL_SETUP_WRONG   (0x00000001U)

/*
 * LW2080_CTRL_THERMAL_GET_THERMAL_SETUP instruction
 *
 * Get the thermal setup from RM by reading thermal fuse and
 * presence if internal sensor.
 *
 *   setupState
 *     Set by the driver to one of the following:
 *       LW2080_CTRL_THERMAL_SETUP_CORRECT
 *       LW2080_CTRL_THERMAL_SETUP_WRONG
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW2080_CTRL_CMD_GET_THERMAL_SETUP (0x20800524U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | LW2080_CTRL_THERMAL_GET_THERMAL_SETUP_INFO_MESSAGE_ID" */

#define LW2080_CTRL_THERMAL_GET_THERMAL_SETUP_INFO_MESSAGE_ID (0x24U)

typedef struct LW2080_CTRL_THERMAL_GET_THERMAL_SETUP_INFO {
    LwU32 setupState;
} LW2080_CTRL_THERMAL_GET_THERMAL_SETUP_INFO;

/*
 * LW2080_CTRL_CMD_THERMAL_GENERIC_TEST
 * Thermal Tests.
 */

/*
 ****** Important Notice ******
 * Please ensure that the test name identifiers below, match exactly with the test name strings in rmt_therm.h file
 * These identifiers are used in lw2080CtrlCmdThermalGenericTest() function, in file thrmctrl.c
 */
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_VERIFY_SMBPBI            0x00000000U
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS              0x00000001U
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_THERMAL_SLOWDOWN         0x00000002U
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BA_SLOWDOWN              0x00000003U
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DYNAMIC_HOTSPOT          0x00000004U
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_TEMP_OVERRIDE            0x00000005U
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_THERMAL_MONITORS         0x00000006U
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DEDICATED_OVERT          0x00000007U
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DEDICATED_OVERT_NEGATIVE 0x00000008U
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_PEAK_SLOWDOWN            0x00000009U
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_HW_ADC_SLOWDOWN          0x0000000aU
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT          0x0000000bU
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BCAST_ACCESS             0x0000000lw

/*
 * LW2080_CTRL_CMD_THERMAL_GENERIC_TEST
 *
 *   Possible Thermal Generic Test Result.
 *
 * _SUCCESS:        Test completed successfully
 * _NOT_IMPLEMENTED:Test is not implemented in RM/PMU
 * _NOT_SUPPORTED:  Test is not supported on the GPU
 * _INSUFFICIENT_PRIVILEDGE:
 *                  SCRATCH register does not have TEST_MODE_ENABLE set
 * _UNSPECIFIED_PMU_ERROR:
 *                  Test ran into an unspecified PMU error
 * _ERROR_GENERIC:  Otherwise
 *
 */
#define LW2080_CTRL_THERMAL_GENERIC_TEST_SUCCESS                     0x00000000U
#define LW2080_CTRL_THERMAL_GENERIC_TEST_NOT_IMPLEMENTED             0x00000001U
#define LW2080_CTRL_THERMAL_GENERIC_TEST_NOT_SUPPORTED               0x00000002U
#define LW2080_CTRL_THERMAL_GENERIC_TEST_INSUFFICIENT_PRIVILEDGE     0x00000003U
#define LW2080_CTRL_THERMAL_GENERIC_TEST_UNSPECIFIED_PMU_ERROR       0x00000004U
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ERROR_GENERIC               0xFFFFFFFFU

/*!
 * Macro to build a unique error code for each test ID.
 */
#define LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(_testid, _status) (((_testid) << 16) | (_status))

/*!
 * LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS
 *
 * Possible reasons for LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS to fail
 *
 * _SUCCESS                      : Test *_ID_INT_SENSORS is a success
 * _TSENSE_VALUE_FAILURE         : Test *_ID_INT_SENSORS failed because value of _TSENSE is invalid
 * _TSENSE_OFFSET_VALUE_FAILURE  : Test *_ID_INT_SENSORS failed because we read an incorrect value of OFFSET_TSENSE
 * _MAX_VALUE_FAILURE            : Test *_ID_INT_SENSORS failed because _SENSOR_MAX reflects an incorrect value
 * _CONSTANT_VALUE_FAILURE       : Test *_ID_INT_SENSORS failed because _SENSOR_CONSTANT reflects an incorrect value
 * _TSOSC_RAW_TEMP_FAILURE       : Test *_ID_INT_SENSORS failed because raw logic for TSOSC did not match the actual TSOSC temperature
 * _TSOSC_RAW_TEMP_HS_FAILURE    : Test *_ID_INT_SENSORS failed because raw logic for TSOSC did not match the actual TSOSC + HS temperature
 * _RAW_CODE_MIN_FAILURE         : Test *_ID_INT_SENSORS failed because raw code min was incorrect
 * _RAW_CODE_MAX_FAILURE         : Test *_ID_INT_SENSORS failed because raw code max was incorrect
 * _TEMP_FAKING_FAILURE          : Test *_ID_INT_SENSORS failed because TSOSC temperature could not be faked
 * _TSOSC_MAX_FAILURE            : Test *_ID_INT_SENSORS failed because we read an incorrect value for TSOSC_MAX
 * _TSOSC_OFFSET_MAX_FAILURE     : Test *_ID_INT_SENSORS failed because we read an incorrect value for TSOSC_OFFSET_MAX
 * _TSOSC_AVG_FAILURE            : Test *_ID_INT_SENSORS failed because we read an incorrect value for TSOSC_AVG
 * _TSOSC_OFFSET_AVG_FAILURE     : Test *_ID_INT_SENSORS failed because we read an incorrect value for TSOSC_OFFSET_AVG
 */
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS_STATUS_SUCCESS                           LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS, 0x0)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS_STATUS_TSENSE_VALUE_FAILURE              LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS, 0x1)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS_STATUS_TSENSE_OFFSET_VALUE_FAILURE       LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS, 0x2)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS_STATUS_MAX_VALUE_FAILURE                 LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS, 0x3)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS_STATUS_CONSTANT_VALUE_FAILURE            LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS, 0x4)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS_STATUS_TSOSC_RAW_TEMP_FAILURE            LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS, 0x5)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS_STATUS_TSOSC_RAW_TEMP_HS_FAILURE         LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS, 0x6)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS_STATUS_TSOSC_RAW_CODE_MIN_FAILURE        LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS, 0x7)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS_STATUS_TSOSC_RAW_CODE_MAX_FAILURE        LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS, 0x8)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS_STATUS_TSOSC_TEMP_FAKING_FAILURE         LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS, 0x9)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS_STATUS_TSOSC_MAX_FAILURE                 LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS, 0xA)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS_STATUS_TSOSC_OFFSET_MAX_FAILURE          LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS, 0xB)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS_STATUS_TSOSC_AVG_FAILURE                 LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS, 0xC)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS_STATUS_TSOSC_OFFSET_AVG_FAILURE          LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS, 0xD)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS_STATUS_SLOPEA_NULL_FAILURE               LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS, 0xE)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS_STATUS_SYS_TSENSE_RAW_TEMP_FAILURE       LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS, 0xF)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS_STATUS_GPC_TSENSE_RAW_TEMP_FAILURE       LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS, 0x10)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS_STATUS_LWL_TSENSE_RAW_TEMP_FAILURE       LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_INT_SENSORS, 0x11)

/*!
 * LW2080_CTRL_THERMAL_GENERIC_TEST_ID_THERMAL_SLOWDOWN
 *
 * Possible reasons for LW2080_CTRL_THERMAL_GENERIC_TEST_ID_THERMAL_SLOWDOWN to fail
 *
 * _SUCCESS                      : Test *_ID_THERMAL_SLOWDOWN is a success
 * _SLOWDOWN_EVT_TRIGGER_FAILURE : Test *_ID_THERMAL_SLOWDOWN failed because event could not be triggered
 * _SLOWDOWN_FAILURE             : Test *_ID_THERMAL_SLOWDOWN failed because clock could not slowdown
 * _SLOWDOWN_RESTORE_FAILURE     : Test *_ID_THERMAL_SLOWDOWN failed because we could not restore slowdown settings after unfaking event
 */
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_THERMAL_SLOWDOWN_STATUS_SUCCESS                  LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_THERMAL_SLOWDOWN, 0x0)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_THERMAL_SLOWDOWN_STATUS_EVT_TRIGGER_FAILURE      LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_THERMAL_SLOWDOWN, 0x1)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_THERMAL_SLOWDOWN_STATUS_SLOWDOWN_FAILURE         LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_THERMAL_SLOWDOWN, 0x2)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_THERMAL_SLOWDOWN_STATUS_SLOWDOWN_RESTORE_FAILURE LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_THERMAL_SLOWDOWN, 0x3)

/*!
 * LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BA_SLOWDOWN
 *
 * _SUCCESS                         : Test *_ID_BA_SLOWDOWN is a success
 * _EVT_NOT_CLEARED_FAILURE         : Test *_ID_BA_SLOWDOWN failed because previous events could not be cleared
 * _EVT_TRIGGER_FAILURE             : Test *_ID_BA_SLOWDOWN failed because event could not be triggered
 * _HIGH_THRESHOLD_SLOWDOWN_FAILURE : Test *_ID_BA_SLOWDOWN failed because slowdown could not be triggered after violating BA high threshold
 * _WINDOW_SLOWDOWN_FAILURE         : Test *_ID_BA_SLOWDOWN failed because slowdown was restored within BA window
 * _SLOWDOWN_RESTORE_FAILURE        : Test *_ID_BA_SLOWDOWN failed because slowdown could not be restored
 */
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BA_SLOWDOWN_STATUS_SUCCESS                         LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BA_SLOWDOWN, 0x0)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BA_SLOWDOWN_STATUS_PREV_EVT_NOT_CLEARED_FAILURE    LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BA_SLOWDOWN, 0x1)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BA_SLOWDOWN_STATUS_EVT_TRIGGER_FAILURE             LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BA_SLOWDOWN, 0x2)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BA_SLOWDOWN_STATUS_HIGH_THRESHOLD_SLOWDOWN_FAILURE LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BA_SLOWDOWN, 0x3)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BA_SLOWDOWN_STATUS_WINDOW_SLOWDOWN_FAILURE         LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BA_SLOWDOWN, 0x4)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BA_SLOWDOWN_STATUS_SLOWDOWN_RESTORE_FAILURE        LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BA_SLOWDOWN, 0x5)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BA_SLOWDOWN_STATUS_EVT_NOT_CLEARED_FAILURE         LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BA_SLOWDOWN, 0x6)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BA_SLOWDOWN_STATUS_FACTOR_A_SET_FAILURE            LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BA_SLOWDOWN, 0x7)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BA_SLOWDOWN_STATUS_LEAKAGE_C_SET_FAILURE           LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BA_SLOWDOWN, 0x8)

/*!
 * LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DYNAMIC_HOTSPOT
 *
 * _SUCCESS                    : Test *ID_DYNAMIC_HOTSPOT is a success
 * _TSOSC_MAX_FAKING_FAILURE   : Test *ID_DYNAMIC_HOTSPOT failed because test could not fake TSOSC_MAX
 * _TSENSE_FAKING_FAILURE      : Test *ID_DYNAMIC_HOTSPOT failed because test could not fake TSENSE
 * _DYNAMIC_HOTSPOT_FAILURE    : Test *ID_DYNAMIC_HOTSPOT failed because temp for DYNAMIC_HOTSPOT is incorrect
 */
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DYNAMIC_HOTSPOT_STATUS_SUCCESS                          LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DYNAMIC_HOTSPOT, 0x0)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DYNAMIC_HOTSPOT_STATUS_TSOSC_MAX_FAKING_FAILURE         LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DYNAMIC_HOTSPOT, 0x1)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DYNAMIC_HOTSPOT_STATUS_TSENSE_FAKING_FAILURE            LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DYNAMIC_HOTSPOT, 0x2)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DYNAMIC_HOTSPOT_STATUS_DYNAMIC_HOTSPOT_FAILURE          LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DYNAMIC_HOTSPOT, 0x3)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DYNAMIC_HOTSPOT_STATUS_DYNAMIC_HOTSPOT_OFFSET_FAILURE   LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DYNAMIC_HOTSPOT, 0x4)

/*!
 * LW2080_CTRL_THERMAL_GENERIC_TEST_ID_TSOSC_OVERRIDE
 *
 * _SUCCESS                          : Test *ID_TSOSC_OVERRIDE is a success
 * _GLOBAL_OVERRIDE_PRIORITY_FAILURE : Test *ID_TSOSC_OVERRIDE failed because global override took priority over local override
 * _TSOSC_RAW_TEMP_FAILURE           : Test *ID_TSOSC_OVERRIDE failed because test could not fake raw temperature
 * _RAW_OVERRIDE_PRIORITY_FAILURE    : Test *ID_TSOSC_OVERRIDE failed because raw temperature took priority over local override
 * _RAW_CODE_FAILURE                 : Test *ID_TSOSC_OVERRIDE failed because of failure to obtain raw code
 * _GPC_AVG_TEMP_FAILURE             : Test *ID_TSOSC_OVERRIDE failed because of GPC Average temperature mismatch after temperature override
 * _GPU_MAX_TEMP_FAILURE             : Test *ID_TSOSC_OVERRIDE failed because of GPU Maximum temperature mismatch after temperature override
 * _GPC_AVG_RAWCODE_FAILURE          : Test *ID_TSOSC_OVERRIDE failed because of GPC Average temperature mismatch after Rawcode override
 * _GPC_MAX_RAWCODE_FAILURE          : Test *ID_TSOSC_OVERRIDE failed because of GPU Maximum temperature mismatch after Rawcode override
 */
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_TSOSC_OVERRIDE_STATUS_SUCCESS                           LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_TEMP_OVERRIDE, 0x0)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_TSOSC_OVERRIDE_STATUS_GLOBAL_OVERRIDE_PRIORITY_FAILURE  LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_TEMP_OVERRIDE, 0x1)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_TSOSC_OVERRIDE_STATUS_TSOSC_RAW_TEMP_FAILURE            LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_TEMP_OVERRIDE, 0x2)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_TSOSC_OVERRIDE_STATUS_RAW_OVERRIDE_PRIORITY_FAILURE     LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_TEMP_OVERRIDE, 0x3)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_TSOSC_OVERRIDE_STATUS_RAW_CODE_FAILURE                  LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_TEMP_OVERRIDE, 0x4)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_TSOSC_OVERRIDE_STATUS_SLOPEA_NULL_FAILURE               LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_TEMP_OVERRIDE, 0x5)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_TSOSC_OVERRIDE_STATUS_GPC_AVG_TEMP_FAILURE              LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_TEMP_OVERRIDE, 0x6)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_TSOSC_OVERRIDE_STATUS_GPU_MAX_TEMP_FAILURE              LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_TEMP_OVERRIDE, 0x7)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_TSOSC_OVERRIDE_STATUS_GPC_AVG_RAWCODE_FAILURE           LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_TEMP_OVERRIDE, 0x8)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_TSOSC_OVERRIDE_STATUS_GPU_MAX_RAWCODE_FAILURE           LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_TEMP_OVERRIDE, 0x9)

/*!
 * LW2080_CTRL_THERMAL_GENERIC_TEST_ID_THERMAL_MONITORS
 *
 * _SUCCESS           : Test *ID_THERMAL_MONITORS is a success
 * _INCREMENT_FAILURE : Test *ID_THERMAL_MONITORS failed because the monitor did not increment
 */
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_THERMAL_MONITORS_STATUS_SUCCESS                         LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_THERMAL_MONITORS, 0x0)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_THERMAL_MONITORS_STATUS_INCREMENT_FAILURE               LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_THERMAL_MONITORS, 0x1)

#define LW2080_CTRL_THERMAL_TEST_STATUS(testid, status) (LW2080_CTRL_THERMAL_GENERIC_TEST_ID_##testid##_STATUS_##status)

/*!
 * LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DEDICATED_OVERT
 *
 * _SUCCESS                         : Test *ID_DEDICATED_OVERT is a success
 * _SHUTDOWN_FAILURE                : Test *ID_DEDICATED_OVERT failed because violation of overt threshold did not make GPU go off the bus
 * _TSOSC_RAW_TEMP_FAILURE          : Test *ID_DEDICATED_OVERT failed because temp callwlated from raw code does not match HW temp
 * _TSOSC_RAW_TEMP_HS_FAILURE       : Test *ID_DEDICATED_OVERT failed because temp callwlated from raw code does not match HW temp with hotspot
 */
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DEDICATED_OVERT_STATUS_SUCCESS                            LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DEDICATED_OVERT, 0x0)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DEDICATED_OVERT_STATUS_SHUTDOWN_FAILURE                   LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DEDICATED_OVERT, 0x1)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DEDICATED_OVERT_STATUS_TSENSE_RAW_MISMATCH_FAILURE        LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DEDICATED_OVERT, 0x2)

/*!
 * LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DEDICATED_OVERT_NEGATIVE
 *
 * _SUCCESS                         : Test *ID_DEDICATED_OVERT_NEGATIVE is a success
 * _SHUTDOWN_FAILURE_TSENSE         : Test *ID_DEDICATED_OVERT_NEGATIVE failed because TSENSE SW override triggered a violation
 * _SHUTDOWN_FAILURE_TSOSC_LOCAL    : Test *ID_DEDICATED_OVERT_NEGATIVE failed because TSOSC local SW override triggered a violation
 * _SHUTDOWN_FAILURE_TSOSC_GLOBAL   : Test *ID_DEDICATED_OVERT_NEGATIVE failed because TSOSC glocal SW override triggered a violation
 */
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DEDICATED_OVERT_NEGATIVE_STATUS_SUCCESS                         LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DEDICATED_OVERT_NEGATIVE, 0x0)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DEDICATED_OVERT_NEGATIVE_STATUS_SHUTDOWN_FAILURE_TSENSE         LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DEDICATED_OVERT_NEGATIVE, 0x1)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DEDICATED_OVERT_NEGATIVE_STATUS_SHUTDOWN_FAILURE_TSOSC_LOCAL    LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DEDICATED_OVERT_NEGATIVE, 0x2)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DEDICATED_OVERT_NEGATIVE_STATUS_SHUTDOWN_FAILURE_TSOSC_GLOBAL   LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_DEDICATED_OVERT_NEGATIVE, 0x3)

/*!
 * LW2080_CTRL_THERMAL_GENERIC_TEST_ID_PEAK_SLOWDOWN
 *
 * _SUCCESS                         : Test *_ID_PEAK_SLOWDOWN is a success
 * _INTR_NOT_CLEARED_FAILURE        : Test *_ID_PEAK_SLOWDOWN failed because previous interrupts could not be cleared
 * _EVT_LATCH_FAILURE               : Test *_ID_PEAK_SLOWDOWN failed because the peak slowdown event was not latched
 * _EVT_TRIGGER_FAILURE             : Test *_ID_PEAK_SLOWDOWN failed because the peak event was latched but not triggered
 * _CLK_SLOWDOWN_FAILURE            : Test *_ID_PEAK_SLOWDOWN failed because the peak event triggered but clock slowdown did not happen
 * _EVT_CLEAR_FAILURE               : Test *_ID_PEAK_SLOWDOWN failed because the peak event could not be cleared
 * _CLK_SLOWDOWN_CLEAR_FAILURE      : Test *_ID_PEAK_SLOWDOWN failed because the peak event was cleared but clock slowdown did not stop
 */
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_PEAK_SLOWDOWN_STATUS_SUCCESS                         LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_PEAK_SLOWDOWN, 0x0)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_PEAK_SLOWDOWN_STATUS_PREV_EVT_NOT_CLEARED_FAILURE    LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_PEAK_SLOWDOWN, 0x1)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_PEAK_SLOWDOWN_STATUS_EVT_LATCH_FAILURE               LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_PEAK_SLOWDOWN, 0x2)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_PEAK_SLOWDOWN_STATUS_EVT_TRIGGER_FAILURE             LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_PEAK_SLOWDOWN, 0x3)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_PEAK_SLOWDOWN_STATUS_CLK_SLOWDOWN_FAILURE            LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_PEAK_SLOWDOWN, 0x4)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_PEAK_SLOWDOWN_STATUS_EVT_CLEAR_FAILURE               LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_PEAK_SLOWDOWN, 0x5)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_PEAK_SLOWDOWN_STATUS_CLK_SLOWDOWN_CLEAR_FAILURE      LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_PEAK_SLOWDOWN, 0x6)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_PEAK_SLOWDOWN_STATUS_FACTOR_A_SET_FAILURE            LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_PEAK_SLOWDOWN, 0x7)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_PEAK_SLOWDOWN_STATUS_LEAKAGE_C_SET_FAILURE           LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_PEAK_SLOWDOWN, 0x8)

/*!
 * LW2080_CTRL_THERMAL_GENERIC_TEST_ID_HW_ADC_SLOWDOWN
 *
 * _SUCCESS                         : Test *ID_HW_ADC_SLOWDOWN is a success
 * _INTR_NOT_CLEARED_FAILURE        : Test *ID_HW_ADC_SLOWDOWN failed because a previous interrupt could not be cleared
 * _EVT_TRIGGER_FAILURE             : Test *ID_HW_ADC_SLOWDOWN failed because the HW ADC event was not triggered
 * _CLK_SLOWDOWN_FAILURE            : Test *ID_HW_ADC_SLOWDOWN failed because clock slowdown was not triggered
 * _EVT_CLEAR                       : Test *ID_HW_ADC_SLOWDOWN failed because the HW ADC event could not be cleared
 * _CLK_SLOWDOWN_CLEAR_FAILURE      : Test *ID_HW_ADC_SLOWDOWN failed because clock slowdown could not be cleared
 */
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_HW_ADC_SLOWDOWN_STATUS_SUCCESS                     LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_HW_ADC_SLOWDOWN, 0x0)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_HW_ADC_SLOWDOWN_STATUS_INTR_NOT_CLEARED_FAILURE    LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_HW_ADC_SLOWDOWN, 0x1)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_HW_ADC_SLOWDOWN_STATUS_EVT_TRIGGER_FAILURE         LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_HW_ADC_SLOWDOWN, 0x2)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_HW_ADC_SLOWDOWN_STATUS_CLK_SLOWDOWN_FAILURE        LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_HW_ADC_SLOWDOWN, 0x3)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_HW_ADC_SLOWDOWN_STATUS_EVT_CLEAR_FAILURE           LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_HW_ADC_SLOWDOWN, 0x4)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_HW_ADC_SLOWDOWN_STATUS_CLK_SLOWDOWN_CLEAR_FAILURE  LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_HW_ADC_SLOWDOWN, 0x5)

/*!
 * LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT
 *
 * _SUCCESS                         : Test *ID_GLOBAL_SNAPSHOT is a success
 */
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT_STATUS_SUCCESS                              LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT, 0x0)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT_STATUS_HW_PIPELINE_TEMP_OVERRIDE_FAILURE    LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT, 0x1)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT_STATUS_UNDER_TEMP_EVENT_TRIGGER_FAILURE     LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT, 0x2)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT_STATUS_OVER_TEMP_EVENT_TRIGGER_FAILURE      LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT, 0x3)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT_STATUS_EVENT_TRIGGER_ROLLBACK_FAILURE       LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT, 0x4)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT_STATUS_SAVE_NON_SNAPSHOT_FAILURE            LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT, 0x5)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT_STATUS_TEMP_OVERRIDE_FAILURE                LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT, 0x6)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT_STATUS_OVERRIDE_SNAPSHOT_FAILURE            LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT, 0x7)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT_STATUS_LWL_TSENSE_SNAPSHOT_MISMATCH         LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT, 0x8)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT_STATUS_LWL_OFFSET_TSENSE_SNAPSHOT_MISMATCH  LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT, 0x9)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT_STATUS_GPC_TSENSE_SNAPSHOT_MISMATCH         LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT, 0xA)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT_STATUS_GPC_OFFSET_TSENSE_SNAPSHOT_MISMATCH  LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT, 0xB)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT_STATUS_SYS_TSENSE_SNAPSHOT_MISMATCH         LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT, 0xC)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT_STATUS_SYS_OFFSET_TSENSE_SNAPSHOT_MISMATCH  LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_GLOBAL_SNAPSHOT, 0xD)


/*!
 * LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BCAST_ACCESS
 *
 * _SUCCESS           : Test *_ID_BCAST_ACCESS_ is a success.
 * _INCREMENT_FAILURE : Test *_ID_BCAST_ACCESS_ failed.
 */
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BCAST_ACCESS_STATUS_SUCCESS                         LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BCAST_ACCESS, 0x0)
#define LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BCAST_ACCESS_STATUS_FAILURE                         LW2080_CTRL_THERMAL_GENERIC_TEST_BUILD_STATUS(LW2080_CTRL_THERMAL_GENERIC_TEST_ID_BCAST_ACCESS, 0x1)


/*
 * LW2080_CTRL_CMD_THERMAL_GENERIC_TEST
 *
 * This command runs one of a set of thermal halified tests.
 *
 *   index
 *     This field specifies the index number of the test
 *   outStatus
       To return status SUCCESS, NOT_IMPLEMENTED, NOT_SUPPORTED or ERROR_GENERIC
 *   outData
 *     This field stores test-specific error codes
 */
#define LW2080_CTRL_CMD_THERMAL_GENERIC_TEST                         (0x20800528U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | LW2080_CTRL_THERMAL_GENERIC_TEST_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_THERMAL_GENERIC_TEST_PARAMS_MESSAGE_ID (0x28U)

typedef struct LW2080_CTRL_THERMAL_GENERIC_TEST_PARAMS {
    LwU32 index;
    LwU32 outStatus;
    LwU32 outData;
} LW2080_CTRL_THERMAL_GENERIC_TEST_PARAMS;

typedef struct THERM_TEST_SMBPBI_CMD_IN {
    LwU8 opcode;
    LwU8 arg1;
    LwU8 arg2;
} THERM_TEST_SMBPBI_CMD_IN;
typedef struct THERM_TEST_SMBPBI_CMD_OUT {
    LwU32 cmd;
    LwU32 dataIn;
    LwU32 dataOut;
} THERM_TEST_SMBPBI_CMD_OUT;
typedef struct THERM_TEST_SMBPBI_CMD {
    THERM_TEST_SMBPBI_CMD_IN  cmdIn;
    THERM_TEST_SMBPBI_CMD_OUT cmdOut;
} THERM_TEST_SMBPBI_CMD;
typedef struct THERM_TEST_SMBPBI_CMD *PTHERM_TEST_SMBPBI_CMD;

/*!
 * Maximum number of Thermal Channels that can be used by Thermal Policies.
 */
#define LW2080_CTRL_THERMAL_POLICY_MAX_CHANNELS LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS

/*!
 * Structure containing dynamic data related to limitCountdown metrics
 * (Bug 3276847) at a policyGrp level.
 */
typedef struct LW2080_CTRL_THERMAL_POLICYS_GLOBAL_DIAGNOSTICS_INFO_LIMIT_COUNTDOWN {
    /*!
     * Boolean to indicate whether the global minimum distance from thermal
     * capping (in degC) will be computed.
     */
    LwBool bEnable;
} LW2080_CTRL_THERMAL_POLICYS_GLOBAL_DIAGNOSTICS_INFO_LIMIT_COUNTDOWN;

/*!
 * Structure containing dynamic data related to limiting temperature metrics
 * (Bug 3287873) at a policyGrp level.
 */
typedef struct LW2080_CTRL_THERMAL_POLICYS_GLOBAL_DIAGNOSTICS_INFO_CAPPED {
    /*!
     * Boolean to indicate whether the limiting temperature metrics causing
     * thermal capping will be reported.
     */
    LwBool bEnable;
} LW2080_CTRL_THERMAL_POLICYS_GLOBAL_DIAGNOSTICS_INFO_CAPPED;

/*!
 * Structure containing static data related to Thermal Policy diagnostic
 * features at a policyGrp level.
 */
typedef struct LW2080_CTRL_THERMAL_POLICYS_GLOBAL_DIAGNOSTICS_INFO {
    /*!
     * Structure containing static data related to limitCountdown metrics
     * (Bug 3276847) at a policyGrp level.
     */
    LW2080_CTRL_THERMAL_POLICYS_GLOBAL_DIAGNOSTICS_INFO_LIMIT_COUNTDOWN limitCountdown;

    /*!
     * Structure containing static data related to limiting temperature metrics
     * (Bug 3287873) at a policyGrp level.
     */
    LW2080_CTRL_THERMAL_POLICYS_GLOBAL_DIAGNOSTICS_INFO_CAPPED          capped;
} LW2080_CTRL_THERMAL_POLICYS_GLOBAL_DIAGNOSTICS_INFO;

/*!
 * Structure containing dynamic data related to limitCountdown metrics
 * (Bug 3276847) at a per-channel level.
 */
typedef struct LW2080_CTRL_THERMAL_POLICYS_CHANNEL_DIAGNOSTICS_STATUS_LIMIT_COUNTDOWN {
    /*!
     * Minimum distance from current temperature to policy limit across
     * all thermal policies using a particular thermal channel.
     * (in degC).
     */
    LwTemp limitCountdown;
} LW2080_CTRL_THERMAL_POLICYS_CHANNEL_DIAGNOSTICS_STATUS_LIMIT_COUNTDOWN;

/*!
 * Structure containing dynamic data related to limiting temperature metrics
 * (Bug 3287873) at a per-channel level.
 */
typedef struct LW2080_CTRL_THERMAL_POLICYS_CHANNEL_DIAGNOSTICS_STATUS_CAPPED {
    /*!
     * Thermal policy mask corresponding to a particular thermal channel.
     * Policies using this channel that are lwrrently capping will have the
     * corresponding bit set in the mask. Policies using this channel that are
     * lwrrently not capping will have the corresponding bit unset in the mask.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32 thermalCappingPolicyMask;
} LW2080_CTRL_THERMAL_POLICYS_CHANNEL_DIAGNOSTICS_STATUS_CAPPED;

/*!
 * Structure containing dynamic data related to Thermal Policy diagnostic
 * features at a per-channel level.
 */
typedef struct LW2080_CTRL_THERMAL_POLICYS_CHANNEL_DIAGNOSTICS_STATUS {
    /*!
     * Structure containing dynamic data related to limitCountdown metrics
     * (Bug 3276847) at a per-channel level.
     */
    LW2080_CTRL_THERMAL_POLICYS_CHANNEL_DIAGNOSTICS_STATUS_LIMIT_COUNTDOWN limitCountdown;

    /*!
     * Structure containing dynamic data related to limiting temperature metrics
     * (Bug 3287873) at a per-channel level.
     */
    LW2080_CTRL_THERMAL_POLICYS_CHANNEL_DIAGNOSTICS_STATUS_CAPPED          capped;
} LW2080_CTRL_THERMAL_POLICYS_CHANNEL_DIAGNOSTICS_STATUS;

/*!
 * Structure containing dynamic data related to limitCountdown metrics
 * (Bug 3276847) at a policyGrp level.
 */
typedef struct LW2080_CTRL_THERMAL_POLICYS_GLOBAL_DIAGNOSTICS_STATUS_LIMIT_COUNTDOWN {
    /*!
     * Global minimum distance from current temperature to policy limit across
     * all thermal policies (in degC).
     */
    LwTemp limitCountdown;
} LW2080_CTRL_THERMAL_POLICYS_GLOBAL_DIAGNOSTICS_STATUS_LIMIT_COUNTDOWN;

/*!
 * Structure containing dynamic data related to Thermal Policy diagnostic
 * features at a policyGrp level.
 */
typedef struct LW2080_CTRL_THERMAL_POLICYS_GLOBAL_DIAGNOSTICS_STATUS {
    /*!
     * Structure containing dynamic data related to limitCountdown metrics
     * (Bug 3276847) at a policyGrp level.
     */
    LW2080_CTRL_THERMAL_POLICYS_GLOBAL_DIAGNOSTICS_STATUS_LIMIT_COUNTDOWN limitCountdown;
} LW2080_CTRL_THERMAL_POLICYS_GLOBAL_DIAGNOSTICS_STATUS;

/*!
 * Structure containing static data related to limitCountdown metrics
 * (Bug 3276847) at a per-policy level.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_DIAGNOSTICS_INFO_LIMIT_COUNTDOWN {
    /*!
     * Boolean to indicate whether the distance in degC from the current
     * temperature to the policy limit will be computed and stored for the
     * current policy.
     */
    LwBool bEnable;
} LW2080_CTRL_THERMAL_POLICY_DIAGNOSTICS_INFO_LIMIT_COUNTDOWN;

/*!
 * Structure containing static data related to limiting temperature metrics
 * (Bug 3287873) at a per-policy level.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_DIAGNOSTICS_INFO_CAPPED {
    /*!
     * Boolean to indicate whether the current policy will be considered for
     * reporting the limiting temperature metrics causing thermal capping.
     */
    LwBool bEnable;
} LW2080_CTRL_THERMAL_POLICY_DIAGNOSTICS_INFO_CAPPED;

/*!
 * Structure containing dynamic data related to limitCountdown metrics
 * (Bug 3276847) at a per-policy level.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_DIAGNOSTICS_STATUS_LIMIT_COUNTDOWN {
    /*!
     * Distance from current temperature to policy limit for the current policy
     * (in degC).
     */
    LwTemp limitCountdown;
} LW2080_CTRL_THERMAL_POLICY_DIAGNOSTICS_STATUS_LIMIT_COUNTDOWN;

/*!
 * Structure containing dynamic data related to limiting temperature metrics
 * (Bug 3287873) at a per-policy level.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_DIAGNOSTICS_STATUS_CAPPED {
    /*!
     * Boolean to indicate whether the current policy is capping.
     */
    LwBool bIsCapped;
} LW2080_CTRL_THERMAL_POLICY_DIAGNOSTICS_STATUS_CAPPED;

/*!
 * Maximum number of thermal policies allowed. Value must be greater than
 * or equal to the maximum number of thermal policies found in the VBIOS.
 */
#define LW2080_CTRL_THERMAL_POLICY_MAX_POLICIES     16U

/*!
 * Special value corresponding to an invalid Thermal Policy index.  This value
 * means the policy is not specified.
 */
#define LW2080_CTRL_THERMAL_POLICY_INDEX_ILWALID    LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Macros encoding types/classes of THERMAL_POLICY entries.
 *
 * Implementation THERMAL_POLICY classes are indexed starting from 0x0.
 * Virtual THERMAL_POLICY classes are indexed starting from 0xFF.
 */
#define LW2080_CTRL_THERMAL_POLICY_TYPE_DTC_ILWALID 0x00000000U
#define LW2080_CTRL_THERMAL_POLICY_TYPE_DTC_VPSTATE 0x00000001U
#define LW2080_CTRL_THERMAL_POLICY_TYPE_DTC_VF      0x00000002U
#define LW2080_CTRL_THERMAL_POLICY_TYPE_DTC_VOLT    0x00000003U
#define LW2080_CTRL_THERMAL_POLICY_TYPE_DTC_PWR     0x00000004U
#define LW2080_CTRL_THERMAL_POLICY_TYPE_DTC         0x000000FDU
#define LW2080_CTRL_THERMAL_POLICY_TYPE_DOMGRP      0x000000FEU
#define LW2080_CTRL_THERMAL_POLICY_TYPE_UNKNOWN     0x000000FFU

/*!
 * This structure represents the static data found in the VPSTATE domain group
 * controller.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_INFO_DATA_DTC_VPSTATE {
    /*!
     * Number of virtual P-states.
     */
    LwU32 vpstateNum;

    /*!
     * Virtual P-states that contains the rated TDP clocks.
     */
    LwU32 vpstateTdp;
} LW2080_CTRL_THERMAL_POLICY_INFO_DATA_DTC_VPSTATE;

/*!
 * This structure represents the static data found in the VF domain group
 * controller.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_INFO_DATA_DTC_VF {
    /*!
     * Maximum frequency used by the controller for the GPC2CLK limit.
     */
    LwU32 limitFreqMaxKHz;

    /*!
     * Minimum frequency used by the controller for the GPC2CLK limit.
     */
    LwU32 limitFreqMinKHz;

    /*!
     * Rated TDP frequency.
     */
    LwU32 ratedTdpFreqKHz;
} LW2080_CTRL_THERMAL_POLICY_INFO_DATA_DTC_VF;

/*!
 * This structure represents the static data found in the VOLT domain group
 * controller.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_INFO_DATA_DTC_VOLT {
    /*!
     * Rated TDP frequency.
     */
    LwU32 freqTdpKHz;

    /*!
     * Rated TDP P-state index.
     */
    LwU32 pstateIdxTdp;

    /*!
     * Maximum voltage value to limit. Limiting to this voltage will cause the
     * GPU to be perf. limited the least.
     */
    LwU32 voltageMaxuV;

    /*!
     * Minimum voltage value to limit. Limiting to this voltage will cause the
     * GPU to be perf. limited the most.
     */
    LwU32 voltageMinuV;

    /*!
     * The size of the voltage limit step.
     */
    LwU32 voltageStepuV;
} LW2080_CTRL_THERMAL_POLICY_INFO_DATA_DTC_VOLT;

/*!
 * This structure represents the static data found in the PWR domain group
 * controller.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_INFO_DATA_DTC_PWR {
    /*!
     * The size of the power limit step.
     */
    LwU32  powerStepmW;

    /*!
     * Specifies whether the policy is capable of limiting its output to boost
     * clocks only.
     */
    LwBool bBaseClockFloorAvailable;
} LW2080_CTRL_THERMAL_POLICY_INFO_DATA_DTC_PWR;

/*!
 * Structure containing static data related to Thermal Policy diagnostic
 * features at a per-policy level.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_DIAGNOSTICS_INFO {
    /*!
     * Structure containing static data related to limitCountdown metrics
     * (Bug 3276847) at a per-policy level.
     */
    LW2080_CTRL_THERMAL_POLICY_DIAGNOSTICS_INFO_LIMIT_COUNTDOWN limitCountdown;

    /*!
     * Structure containing static data related to limiting temperature metrics
     * (Bug 3287873) at a per-policy level.
     */
    LW2080_CTRL_THERMAL_POLICY_DIAGNOSTICS_INFO_CAPPED          capped;
} LW2080_CTRL_THERMAL_POLICY_DIAGNOSTICS_INFO;

/*!
 * Union of type-specific static state data.
 */


/*!
 * Static information for a particular thermal policy.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_INFO {
    /*!
     * @ref LW2080_CTRL_THERMAL_POLICY_TYPE_<xyz>.
     */
    LwU8                                        type;

    /*!
     * Index into the Thermal Channel Table.
     */
    LwU8                                        chIdx;

    /*!
     * Minimum allowed limit value. Signed number of 1/256 degrees Celsius.
     */
    LwS32                                       limitMin;

    /*!
     * Rated/default limit value. Signed number of 1/256 degrees Celsius.
     */
    LwS32                                       limitRated;

    /*!
     * Maximum allowed limit value. Signed number of 1/256 degrees Celsius.
     */
    LwS32                                       limitMax;

    /*!
     * VBIOS info concerning the policy's pff.
     */
    LW2080_CTRL_PMGR_PFF_INFO                   pff;

    /*!
     * Structure containing static data related to Thermal Policy diagnostic
     * features at a per-policy level.
     */
    LW2080_CTRL_THERMAL_POLICY_DIAGNOSTICS_INFO diagnostics;

    /*!
     * Type-specific information.
     */
    union {
        LW2080_CTRL_THERMAL_POLICY_INFO_DATA_DTC_VPSTATE dtcVpstate;
        LW2080_CTRL_THERMAL_POLICY_INFO_DATA_DTC_VF      dtcVf;
        LW2080_CTRL_THERMAL_POLICY_INFO_DATA_DTC_VOLT    dtcVolt;
        LW2080_CTRL_THERMAL_POLICY_INFO_DATA_DTC_PWR     dtcPwr;
    } data;
} LW2080_CTRL_THERMAL_POLICY_INFO;

/*!
 * Structure containing static data related to Thermal Policy diagnostic
 * features at a policyGrp and a per-channel level.
 */
typedef struct LW2080_CTRL_THERMAL_POLICYS_DIAGNOSTICS_INFO {
    /*!
     * Mask of THERM_CHANNEL entries being used by at least one THERM_POLICY
     * entry.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32                    activeChannelMask;

    /*!
     * Structure containing static data related to Thermal Policy diagnostic
     * features at a policyGrp level.
     */
    LW2080_CTRL_THERMAL_POLICYS_GLOBAL_DIAGNOSTICS_INFO global;
} LW2080_CTRL_THERMAL_POLICYS_DIAGNOSTICS_INFO;

/*!
 * LW2080_CTRL_CMD_THERMAL_POLICY_GET_INFO
 *
 * This command returns the THERM_POLICY static information as specified by
 * the Thermal Policy Table.
 *
 * https://wiki.lwpu.com/engwiki/index.php/Resman/Thermal_Control/Thermal_Policy_Table_1.0_Specification
 *
 * See LW2080_CTRL_THERMAL_POLICY_INFO_PARAMS for documentation on the
 * parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_THERMAL_POLICY_GET_INFO (0x2080052aU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | LW2080_CTRL_THERMAL_POLICY_INFO_PARAMS_MESSAGE_ID" */

/*!
 * Structure representing the static state information associated with the
 * GPU's THERM_POLICY thermal policy functionality.
 */
#define LW2080_CTRL_THERMAL_POLICY_INFO_PARAMS_MESSAGE_ID (0x2AU)

typedef struct LW2080_CTRL_THERMAL_POLICY_INFO_PARAMS {
    /*!
     * [out] - Mask of THERM_POLICY entries specified on this GPU.
     */
    LwU32                                        policyMask;

    /*!
     * [out] - Thermal Policy Table index for the Thermal Policy controlling
     * the GPS temperature controller.
     */
    LwU8                                         gpsPolicyIdx;

    /*!
     * [out] - Thermal Policy Table index for the Thermal Policy controlling
     * acoustics.
     */
    LwU8                                         acousticPolicyIdx;

    /*!
     * [out] - Thermal Policy Table index for the Thermal Policy controlling
     * the memory temperature.
     */
    LwU8                                         memPolicyIdx;

    /*!
     * [out] - Thermal Policy Table index for the Thermal Policy controlling
     * GPU SW slowdown.
     */
    LwU8                                         gpuSwSlowdownPolicyIdx;

    /*!
     * Structure containing static data related to Thermal Policy diagnostic
     * features at a policyGrp and a per-channel level.
     */
    LW2080_CTRL_THERMAL_POLICYS_DIAGNOSTICS_INFO diagnostics;

    /*!
     * [out] - Array of THERM_POLICY entries.  Has valid indexes corresponding
     * to the bits in @ref policyMask.
     */
    LW2080_CTRL_THERMAL_POLICY_INFO              policies[LW2080_CTRL_THERMAL_POLICY_MAX_POLICIES];
} LW2080_CTRL_THERMAL_POLICY_INFO_PARAMS;

/*!
 * Macros encoding types of temperature controller limits.
 */
#define LW2080_CTRL_THERMAL_POLICY_PERF_LIMIT_IDX_PSTATE  0x00U
#define LW2080_CTRL_THERMAL_POLICY_PERF_LIMIT_IDX_GPC2CLK 0x01U
#define LW2080_CTRL_THERMAL_POLICY_PERF_LIMIT_IDX_LWVDD   0x02U
#define LW2080_CTRL_THERMAL_POLICY_PERF_LIMIT_NUM_LIMITS  0x03U

/*!
 * Macro encoding that perf limit is not in place for the limit type.
 */
#define LW2080_CTRL_THERMAL_POLICY_PERF_LIMIT_VAL_NONE    (LW_U32_MAX)

/*!
 * This structure represents the state of thermal policy perf. limits.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_PERF_LIMITS {
    LwU32 limit[LW2080_CTRL_THERMAL_POLICY_PERF_LIMIT_NUM_LIMITS];
} LW2080_CTRL_THERMAL_POLICY_PERF_LIMITS;

/*!
 * This structure represents the current dynamic status of a particular
 * domain group controller.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_STATUS_DATA_DOMGRP {
    /*!
     * Limits imposed by the controller.
     */
    LW2080_CTRL_THERMAL_POLICY_PERF_LIMITS limits;
} LW2080_CTRL_THERMAL_POLICY_STATUS_DATA_DOMGRP;

/*!
 * This structure represents the current dynamic status of a particular DTC
 * controller.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_STATUS_DATA_DTC {
    /*!
     * Keeps track of the number of samples taken for the current threshold
     * range. If the number of samples exceeds the sample threshold, the
     * algorithm will alter its behavior.
     */
    LwU8 sampleCount;
} LW2080_CTRL_THERMAL_POLICY_STATUS_DATA_DTC;

/*!
 * This structure represents the current dynamic status of a particular VPSTATE
 * domain group controller.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_STATUS_DATA_DTC_VPSTATE {
    /*!
     * Domain Group dynamic data. Must always be first!
     */
    LW2080_CTRL_THERMAL_POLICY_STATUS_DATA_DOMGRP super;

    /*!
     * DTC controller dynamic data.
     */
    LW2080_CTRL_THERMAL_POLICY_STATUS_DATA_DTC    dtc;

    /*!
     * Current virtual P-state used as the controller's limit.
     */
    LwU32                                         vpstateLimitLwrr;
} LW2080_CTRL_THERMAL_POLICY_STATUS_DATA_DTC_VPSTATE;

/*!
 * This structure represents the current dynamic status of a particular VF
 * domain group controller.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_STATUS_DATA_DTC_VF {
    /*!
     * Domain Group dynamic data. Must always be first!
     */
    LW2080_CTRL_THERMAL_POLICY_STATUS_DATA_DOMGRP super;

    /*!
     * DTC controller dynamic data.
     */
    LW2080_CTRL_THERMAL_POLICY_STATUS_DATA_DTC    dtc;
} LW2080_CTRL_THERMAL_POLICY_STATUS_DATA_DTC_VF;

/*!
 * This structure represents the current dynamic status of a particular VOLT
 * domain group controller.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_STATUS_DATA_DTC_VOLT {
    /*!
     * Domain Group dynamic data. Must always be first!
     */
    LW2080_CTRL_THERMAL_POLICY_STATUS_DATA_DOMGRP super;

    /*!
     * DTC controller dynamic data.
     */
    LW2080_CTRL_THERMAL_POLICY_STATUS_DATA_DTC    dtc;
} LW2080_CTRL_THERMAL_POLICY_STATUS_DATA_DTC_VOLT;

/*!
 * This structure represents the current dynamic status of a particular PWR
 * domain group controller.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_STATUS_DATA_DTC_PWR {
    /*!
     * DTC controller dynamic data.
     */
    LW2080_CTRL_THERMAL_POLICY_STATUS_DATA_DTC dtc;

    /*!
     * Power limit imposed by the controller.
     */
    LwU32                                      limitLwrrmW;

    /*!
     * Current power policy used to set the power limit.
     */
    LwU8                                       powerPolicyIdx;
} LW2080_CTRL_THERMAL_POLICY_STATUS_DATA_DTC_PWR;

/*!
 * Structure containing dynamic data related to Thermal Policy diagnostic
 * features at a per-policy level.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_DIAGNOSTICS_STATUS {
    /*!
     * Structure containing dynamic data related to limitCountdown metrics
     * (Bug 3276847) at a per-policy level.
     */
    LW2080_CTRL_THERMAL_POLICY_DIAGNOSTICS_STATUS_LIMIT_COUNTDOWN limitCountdown;

    /*!
     * Structure containing dynamic data related to limiting temperature metrics
     * (Bug 3287873) at a per-policy level.
     */
    LW2080_CTRL_THERMAL_POLICY_DIAGNOSTICS_STATUS_CAPPED          capped;
} LW2080_CTRL_THERMAL_POLICY_DIAGNOSTICS_STATUS;

/*!
 * Union of type-specific dynamic state data.
 */


/*!
 * Structure representing the dynamic state associated with a THERM_POLICY
 * entry.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_STATUS {
    /*!
     * @ref LW2080_CTRL_THERMAL_POLICY_TYPE_<xyz>.
     */
    LwU8                                          type;

    /*!
     * Current value retrieved from the monitored THERM_CHANNEL. Signed
     * number of 1/256 degrees Celsius.
     */
    LwS32                                         valueLwrr;

    /*!
     * Dynamic state associated with this policy's pff.
     */
    LW2080_CTRL_PMGR_PFF_STATUS                   pff;

    /*!
     * Structure to encapsulate functionality for Thermal Policy Diagnostic
     * features at a per-policy level.
     */
    LW2080_CTRL_THERMAL_POLICY_DIAGNOSTICS_STATUS diagnostics;

    /*!
     * Type-specific information.
     */
    union {
        LW2080_CTRL_THERMAL_POLICY_STATUS_DATA_DOMGRP      domGrp;
        LW2080_CTRL_THERMAL_POLICY_STATUS_DATA_DTC_VPSTATE dtcVpstate;
        LW2080_CTRL_THERMAL_POLICY_STATUS_DATA_DTC_VF      dtcVf;
        LW2080_CTRL_THERMAL_POLICY_STATUS_DATA_DTC_VOLT    dtcVolt;
        LW2080_CTRL_THERMAL_POLICY_STATUS_DATA_DTC_PWR     dtcPwr;
    } data;
} LW2080_CTRL_THERMAL_POLICY_STATUS;

/*!
 * Structure containing dynamic data related to Thermal Policy diagnostic
 * features at a policyGrpand a per-channel level.
 */
typedef struct LW2080_CTRL_THERMAL_POLICYS_DIAGNOSTICS_STATUS {
    /*!
     * Number of unique thermal channel entries being used by thermal policy
     * entries.
     */
    LwU8                                                   numChannels;

    /*!
     * Structure containing dynamic data related to Thermal Policy diagnostic
     * features at a per-channel level.
     */
    LW2080_CTRL_THERMAL_POLICYS_CHANNEL_DIAGNOSTICS_STATUS channels[LW2080_CTRL_THERMAL_POLICY_MAX_CHANNELS];

    /*!
     * Structure containing dynamic data related to Thermal Policy diagnostic
     * features at a policyGrp level.
     */
    LW2080_CTRL_THERMAL_POLICYS_GLOBAL_DIAGNOSTICS_STATUS  global;
} LW2080_CTRL_THERMAL_POLICYS_DIAGNOSTICS_STATUS;

/*!
 * LW2080_CTRL_CMD_THERMAL_POLICY_GET_STATUS
 *
 * This command returns the dynamic status of a set of client-specified
 * THERM_POLICY entries in the Thermal Policy Table.
 *
 * https://wiki.lwpu.com/engwiki/index.php/Resman/Thermal_Control/Thermal_Policy_Table_1.0_Specification
 *
 * See LW2080_CTRL_THERMAL_POLICY_STATUS_PARAMS for documentation on the
 * parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_THERMAL_POLICY_GET_STATUS (0x2080052bU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | LW2080_CTRL_THERMAL_POLICY_STATUS_PARAMS_MESSAGE_ID" */

/*!
 * Structure representing the dynamic status information associated with a set
 * of THERM_POLICYs within the GPU's THERM_POLICY thermal policy functionality.
 */
#define LW2080_CTRL_THERMAL_POLICY_STATUS_PARAMS_MESSAGE_ID (0x2BU)

typedef struct LW2080_CTRL_THERMAL_POLICY_STATUS_PARAMS {
    /*!
     * [in] - Mask of THERM_POLICY entries requested by the client.
     */
    LwU32                                          policyMask;

    /*!
     * Structure to encapsulate functionality for Thermal Policy Diagnostic
     * features at a policyGrp and a per-channel level.
     */
    LW2080_CTRL_THERMAL_POLICYS_DIAGNOSTICS_STATUS diagnostics;

    /*!
     * [out] - Array of THERM_POLICY entries.  Has valid indixes corresponding
     *         to the bits set in @ref policyMask.
     */
    LW2080_CTRL_THERMAL_POLICY_STATUS              policies[LW2080_CTRL_THERMAL_POLICY_MAX_POLICIES];
} LW2080_CTRL_THERMAL_POLICY_STATUS_PARAMS;

typedef struct LW2080_CTRL_THERMAL_POLICY_CONTROL_DATA_DTC {
    /*!
     * The number of levels the controller will step when the temperature lies
     * in the aggressive range.
     */
    LwU8  aggressiveStep;

    /*!
     * The number of levels the controller will step when the temperature lies
     * in the release range.
     */
    LwU8  releaseStep;

    /*!
     * The number of contiguous samples the controller must have in the hold
     * range before increasing the perf. by one controller level. A value of
     * 0xFF specifies the controller will not increase perf. while in the hold
     * range.
     */
    LwU8  holdSampleThreshold;

    /*!
     * The number of contiguous samples the controller must have in the
     * aggressive, moderate, or release range before holding the perf. level,
     * allowing any temperature lag to catch up.
     */
    LwU8  stepSampleThreshold;

    /*!
     * Critical temperature threshold. Value is stored as signed 1/256 degrees
     * Celsius.
     */
    LwS32 thresholdCritical;

    /*!
     * Aggressive temperature threshold. Value is stored as signed 1/256
     * degrees Celsius.
     */
    LwS32 thresholdAggressive;

    /*!
     * Moderate temperature threshold. Value is stored as signed 1/256 degrees
     * Celsius.
     */
    LwS32 thresholdModerate;

    /*!
     * Release temperature threshold. Value is stored as signed 1/256 degrees
     * Celsius.
     */
    LwS32 thresholdRelease;

    /*!
     * Disengage temperature threshold. Value is stored as signed 1/256 degrees
     * Celsius.
     */
    LwS32 thresholdDisengage;
} LW2080_CTRL_THERMAL_POLICY_CONTROL_DATA_DTC;

/*!
 * Structure representing the current control parameters of a particular
 * domain group controller.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_CONTROL_DATA_DOMGRP {
    /*!
     * A boolean flag to indicate the controller shall not impose a perf.
     * limit that would cause the clocks to fall below the Rated TDP VPstate.
     */
    LwBool bRatedTdpVpstateFloor;
} LW2080_CTRL_THERMAL_POLICY_CONTROL_DATA_DOMGRP;

/*!
 * Structure representing the current control parameters of a particular
 * VPSTATE domain group controller.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_CONTROL_DATA_DTC_VPSTATE {
    /*!
     * Domain group controller control parameters. Must always be first!
     */
    LW2080_CTRL_THERMAL_POLICY_CONTROL_DATA_DOMGRP super;

    /*!
     * DTC control algorithm control parameters.
     */
    LW2080_CTRL_THERMAL_POLICY_CONTROL_DATA_DTC    dtc;
} LW2080_CTRL_THERMAL_POLICY_CONTROL_DATA_DTC_VPSTATE;

/*!
 * Structure representing the current control parameters of a particular VF
 * domain group controller.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_CONTROL_DATA_DTC_VF {
    /*!
     * Domain group controller control parameters. Must always be first!
     */
    LW2080_CTRL_THERMAL_POLICY_CONTROL_DATA_DOMGRP super;

    /*!
     * DTC control algorithm control parameters.
     */
    LW2080_CTRL_THERMAL_POLICY_CONTROL_DATA_DTC    dtc;
} LW2080_CTRL_THERMAL_POLICY_CONTROL_DATA_DTC_VF;

/*!
 * Structure representing the current control parameters of a particular VOLT
 * domain group controller.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_CONTROL_DATA_DTC_VOLT {
    /*!
     * Domain group controller control parameters. Must always be first!
     */
    LW2080_CTRL_THERMAL_POLICY_CONTROL_DATA_DOMGRP super;

    /*!
     * DTC control algorithm control parameters.
     */
    LW2080_CTRL_THERMAL_POLICY_CONTROL_DATA_DTC    dtc;
} LW2080_CTRL_THERMAL_POLICY_CONTROL_DATA_DTC_VOLT;

/*!
 * Structure representing the current control parameters of a particular PWR
 * domain group controller.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_CONTROL_DATA_DTC_PWR {
    /*!
     * DTC control algorithm control parameters.
     */
    LW2080_CTRL_THERMAL_POLICY_CONTROL_DATA_DTC dtc;

    /*!
     * A boolean flag to indicate the controller shall use a power policy that
     * keeps limits above the base clocks only.
     */
    LwBool                                      bBaseClockFloor;
} LW2080_CTRL_THERMAL_POLICY_CONTROL_DATA_DTC_PWR;

/*!
 * Union of type-specific control parameters.
 */


/*!
 * Structure representing the control/policy parameters of a THERM_POLICY
 * entry.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_CONTROL {
    /*!
     * @ref LW2080_CTRL_THERMAL_POLICY_TYPE_<xyz>.
     */
    LwU8                            type;

    /*!
     * Current limit value to enforce.  Must always be within range of
     * [limitMin, limitMax]. Signed number of 1/256 degrees Celsius.
     */
    LwS32                           limitLwrr;

    /*!
     * Specifies whether the thermal policy is allowed to drop below rated TDP even
     * when the VBIOS flag is set.
     */
    LwBool                          bAllowBelowRatedTdp;

    /*!
     * Specifies the polling period (ms) of the policy. A value of 0 specifies
     * the policy does not actively poll.
     */
    LwU32                           pollingPeriodms;

    /*!
     * Control data concerning the policy's pff.
     */
    LW2080_CTRL_PMGR_PFF_RM_CONTROL pff;

    /*!
     * Type-specific information.
     */
    union {
        LW2080_CTRL_THERMAL_POLICY_CONTROL_DATA_DOMGRP      domGrp;
        LW2080_CTRL_THERMAL_POLICY_CONTROL_DATA_DTC_VPSTATE dtcVpstate;
        LW2080_CTRL_THERMAL_POLICY_CONTROL_DATA_DTC_VF      dtcVf;
        LW2080_CTRL_THERMAL_POLICY_CONTROL_DATA_DTC_VOLT    dtcVolt;
        LW2080_CTRL_THERMAL_POLICY_CONTROL_DATA_DTC_PWR     dtcPwr;
    } data;
} LW2080_CTRL_THERMAL_POLICY_CONTROL;

/*!
 * LW2080_CTRL_CMD_THERMAL_POLICY_GET_CONTROL
 *
 * This command returns the control/policy parameters for a client-specified
 * set of THERM_POLICY entries in the Thermal Policy Table.
 *
 * https://wiki.lwpu.com/engwiki/index.php/Resman/Thermal_Control/Thermal_Policy_Table_1.0_Specification
 *
 * See LW2080_CTRL_THERMAL_POLICY_CONTROL_PARAMS for documentation on the
 * parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_THERMAL_POLICY_GET_CONTROL (0x2080052lw) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | 0x2C" */

/*!
 * LW2080_CTRL_CMD_THERMAL_POLICY_SET_CONTROL
 *
 * This command accepts client-specified control/policy parameters for a set
 * of THERM_POLICY entries in the Thermal Policy Table, and applies these new
 * parameters to the set of THERM_POLICY entries.
 *
 * https://wiki.lwpu.com/engwiki/index.php/Resman/Thermal_Control/Thermal_Policy_Table_1.0_Specification
 *
 * See LW2080_CTRL_THERMAL_POLICY_CONTROL_PARAMS for documentation on the
 * parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_THERMAL_POLICY_SET_CONTROL (0x2080052dU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | 0x2D" */

/*!
 * Structure representing the control/policy parameters associated with a set
 * of THERM_POLICY entries in the GPU's THERM_POLICY thermal policy
 * functionality.
 */
typedef struct LW2080_CTRL_THERMAL_POLICY_CONTROL_PARAMS {
    /*!
     * [in] - Mask of THERM_POLICY entries requested by the client.
     */
    LwU32                              policyMask;

    /*!
     * [in,out] - Array of THERM_POLICY entries.  Has valid indexes
     *            corresponding to the bits set in @ref policyMask.
     */
    LW2080_CTRL_THERMAL_POLICY_CONTROL policies[LW2080_CTRL_THERMAL_POLICY_MAX_POLICIES];
} LW2080_CTRL_THERMAL_POLICY_CONTROL_PARAMS;

/*!
 * Denotes the control action to be taken on the fan.
 */
#define LW2080_CTRL_FAN_CTRL_ACTION_ILWALID              0x00U
#define LW2080_CTRL_FAN_CTRL_ACTION_SPEED_CTRL           0x01U
#define LW2080_CTRL_FAN_CTRL_ACTION_STOP                 0x02U
#define LW2080_CTRL_FAN_CTRL_ACTION_RESTART              0x03U

/*!
 * Macros for Fan Cooler Types.
 *
 * Virtual FAN_COOLER types are indexed starting from 0xFF.
 */
#define LW2080_CTRL_FAN_COOLER_TYPE_ILWALID              0x00U
#define LW2080_CTRL_FAN_COOLER_TYPE_ACTIVE_PWM           0x01U
#define LW2080_CTRL_FAN_COOLER_TYPE_ACTIVE_PWM_TACH_CORR 0x02U
#define LW2080_CTRL_FAN_COOLER_TYPE_ACTIVE               0xFEU
#define LW2080_CTRL_FAN_COOLER_TYPE_UNKNOWN              0xFFU

/*!
 * Special value corresponding to an invalid Fan Cooler Index.
 */
#define LW2080_CTRL_FAN_COOLER_INDEX_ILWALID             LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Maximum number of FAN_COOLERS which can be supported in the RM or PMU.
 */
#define LW2080_CTRL_FAN_COOLER_MAX_COOLERS               16U

/*!
 * Macros for Fan Cooler Control Units.
 */
#define LW2080_CTRL_FAN_COOLER_ACTIVE_CONTROL_UNIT_NONE  0x00U
#define LW2080_CTRL_FAN_COOLER_ACTIVE_CONTROL_UNIT_PWM   0x01U
#define LW2080_CTRL_FAN_COOLER_ACTIVE_CONTROL_UNIT_RPM   0x02U

/* ---------- FAN_COOLER's GET_INFO RMCTRL defines and structures ---------- */

/*!
 * Structure representing the static information associated with a
 * ACTIVE FAN_COOLER.
 */
typedef struct LW2080_CTRL_FAN_COOLER_INFO_DATA_ACTIVE {
    /*!
     * Tachometer GPIO Function.
     */
    LwU32  tachFuncFan;

    /*!
     * Tachometer Rate (PPR).
     */
    LwU8   tachRate;

    /*!
     * Tachometer is present and supported.
     */
    LwBool bTachSupported;

    /*!
     * Control unit used (by policy) to drive this cooler as
     * LW2080_CTRL_FAN_COOLER_ACTIVE_CONTROL_UNIT_<xyz>.
     */
    LwU8   controlUnit;

    /*!
     * Tachometer GPIO pin.
     */
    LwU8   tachPin;
} LW2080_CTRL_FAN_COOLER_INFO_DATA_ACTIVE;

/*!
 * Structure representing the static information associated with a
 * ACTIVE_PWM FAN_COOLER.
 */
typedef struct LW2080_CTRL_FAN_COOLER_INFO_DATA_ACTIVE_PWM {
    /*!
     * LW2080_CTRL_FAN_COOLER_INFO_DATA_ACTIVE super class.
     * This should always be the first member!
     */
    LW2080_CTRL_FAN_COOLER_INFO_DATA_ACTIVE active;

    /*!
     * Fan Speed Scaling.
     */
    LwSFXP4_12                              scaling;

    /*!
     * Fan Speed Offset.
     */
    LwSFXP4_12                              offset;

    /*!
     * Fan GPIO Function.
     */
    LwU32                                   gpioFuncFan;

    /*!
     * PWM frequency.
     */
    LwU32                                   freq;

    /*!
     * RM_PMU_THERM_EVENT_<xyz> bitmask triggering MAX Fan speed (if engaged).
     */
    LwU32                                   maxFanEvtMask;

    /*!
     * MIN time [ms] to spend at MAX Fan speed to prevent trashing at slowdown.
     */
    LwU16                                   maxFanMinTimems;
} LW2080_CTRL_FAN_COOLER_INFO_DATA_ACTIVE_PWM;

/*!
 * Structure representing the static information associated with a
 * ACTIVE_PWM_TACH_CORR FAN_COOLER.
 */
typedef struct LW2080_CTRL_FAN_COOLER_INFO_DATA_ACTIVE_PWM_TACH_CORR {
    /*!
     * LW2080_CTRL_FAN_COOLER_INFO_DATA_ACTIVE_PWM super class.
     * This should always be the first member!
     */
    LW2080_CTRL_FAN_COOLER_INFO_DATA_ACTIVE_PWM activePwm;
} LW2080_CTRL_FAN_COOLER_INFO_DATA_ACTIVE_PWM_TACH_CORR;

/*!
 * Union of type-specific static information data.
 */
typedef union LW2080_CTRL_FAN_COOLER_INFO_DATA {
    LW2080_CTRL_FAN_COOLER_INFO_DATA_ACTIVE               active;
    LW2080_CTRL_FAN_COOLER_INFO_DATA_ACTIVE_PWM           activePwm;
    LW2080_CTRL_FAN_COOLER_INFO_DATA_ACTIVE_PWM_TACH_CORR activePwmTachCorr;
} LW2080_CTRL_FAN_COOLER_INFO_DATA;


/*!
 * Structure representing the static information associated with a FAN_COOLER.
 */
typedef struct LW2080_CTRL_FAN_COOLER_INFO {
    /*!
     * @ref LW2080_CTRL_FAN_COOLER_TYPE_<xyz>.
     */
    LwU8                             type;

    /*!
     * Type-specific information.
     */
    LW2080_CTRL_FAN_COOLER_INFO_DATA data;
} LW2080_CTRL_FAN_COOLER_INFO;

/*!
 * LW2080_CTRL_CMD_FAN_COOLER_GET_INFO
 *
 * This command returns the FAN_COOLER static information as specified by the
 * Fan Cooler Table.
 *
 * https://wiki.lwpu.com/engwiki/index.php/Resman/Thermal_Control/Fan_Tables(s)_1.0_Specification#Fan_Cooler_Table
 *
 * See LW2080_CTRL_FAN_COOLER_INFO_PARAMS for documentation on the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FAN_COOLER_GET_INFO (0x2080052eU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | LW2080_CTRL_FAN_COOLER_INFO_PARAMS_MESSAGE_ID" */

/*!
 * Structure representing the static state information associated with the GPU's
 * FAN_COOLER fan cooler functionality.
 */
#define LW2080_CTRL_FAN_COOLER_INFO_PARAMS_MESSAGE_ID (0x2EU)

typedef struct LW2080_CTRL_FAN_COOLER_INFO_PARAMS {
    /*!
     * [out] - Mask of FAN_COOLER entries specified on this GPU.
     */
    LwU32                       coolerMask;

    /*!
     * [out] - Fan Cooler Table index for Fan Cooler controlling the first GPU fan.
     *
     * @note LW2080_CTRL_FAN_COOLER_INDEX_ILWALID indicates that
     * this cooler is not present/specified on this GPU.
     */
    LwU8                        gpuCoolerIdx0;

    /*!
     * [out] - Fan Cooler Table index for Fan Cooler controlling the second GPU fan.
     *
     * @note LW2080_CTRL_FAN_COOLER_INDEX_ILWALID indicates that
     * this cooler is not present/specified on this GPU.
     */
    LwU8                        gpuCoolerIdx1;

    /*!
     * [out] - Array of FAN_COOLER entries. Has valid indexes corresponding to
     * the bits set in @ref coolerMask.
     */
    LW2080_CTRL_FAN_COOLER_INFO coolers[LW2080_CTRL_FAN_COOLER_MAX_COOLERS];
} LW2080_CTRL_FAN_COOLER_INFO_PARAMS;

/* --------- FAN_COOLER's GET_STATUS RMCTRL defines and structures --------- */

#define FAN30_LEVEL_MIN_PCT (30U)
#define FAN30_LEVEL_MAX_PCT (100U)

#define LW2080_CTRL_FAN_30_UFXP1616_LEVEL_MIN   LW_UNSIGNED_ROUNDED_DIV((FAN30_LEVEL_MIN_PCT << 16), FAN30_LEVEL_MAX_PCT)
#define LW2080_CTRL_FAN_30_UFXP1616_LEVEL_MAX   LW_TYPES_U32_TO_UFXP_X_Y(16, 16, 1)

/*!
 * Structure representing the dynamic state associated with a
 * ACTIVE FAN_COOLER.
 */
typedef struct LW2080_CTRL_FAN_COOLER_STATUS_DATA_ACTIVE {
    /*!
     * Current RPM.
     */
    LwU32       rpmLwrr;

    /*!
     * Minimum fan level, as UFXP percentage (computed from driving RPM or PWM
     * value normalized to max).
     */
    LwUFXP16_16 levelMin;

    /*!
     * Maximum fan level, as UFXP percentage (computed from driving RPM or PWM
     * value normalized to max, always 100%).
     */
    LwUFXP16_16 levelMax;

    /*!
     * Current (dynamic) fan level, as UFXP percentage (computed from driving
     * RPM or PWM value normalized to max).
     */
    LwUFXP16_16 levelLwrrent;

    /*!
     * Target (lwrrently requested, static) fan level, as UFXP percentage
     * (computed from driving RPM or PWM value normalized to max).
     */
    LwUFXP16_16 levelTarget;
} LW2080_CTRL_FAN_COOLER_STATUS_DATA_ACTIVE;

/*!
 * Structure representing the dynamic state associated with a
 * ACTIVE_PWM FAN_COOLER.
 */
typedef struct LW2080_CTRL_FAN_COOLER_STATUS_DATA_ACTIVE_PWM {
    /*!
     * LW2080_CTRL_FAN_COOLER_STATUS_DATA_ACTIVE super class.
     * This should always be the first member!
     */
    LW2080_CTRL_FAN_COOLER_STATUS_DATA_ACTIVE active;

    /*!
     * Current Electrical Fan Speed (PWM rate in percents).
     */
    LwUFXP16_16                               pwmLwrr;

    /*!
     * PWM requested to ACTIVE_PWM FAN_COOLER.
     */
    LwUFXP16_16                               pwmRequested;

    /*!
     * Set while PWM Fan is in MAX-ed state.
     */
    LwBool                                    bMaxFanActive;
} LW2080_CTRL_FAN_COOLER_STATUS_DATA_ACTIVE_PWM;

/*!
 * Structure representing the dynamic state associated with a
 * ACTIVE_PWM_TACH_CORR FAN_COOLER.
 */
typedef struct LW2080_CTRL_FAN_COOLER_STATUS_DATA_ACTIVE_PWM_TACH_CORR {
    /*!
     * LW2080_CTRL_FAN_COOLER_STATUS_DATA_ACTIVE_PWM super class.
     * This should always be the first member!
     */
    LW2080_CTRL_FAN_COOLER_STATUS_DATA_ACTIVE_PWM activePwm;

    /*!
     * Last RPM reading used for tach correction.
     */
    LwU32                                         rpmLast;

    /*!
     * Target RPM requested by tach correction.
     */
    LwU32                                         rpmTarget;

    /*!
     * Actual measured PWM used for applying floor limit offset on Gemini fan
     * sharing designs.
     */
    LwUFXP16_16                                   pwmActual;
} LW2080_CTRL_FAN_COOLER_STATUS_DATA_ACTIVE_PWM_TACH_CORR;

/*!
 * Union of type-specific dynamic data.
 */
typedef union LW2080_CTRL_FAN_COOLER_STATUS_DATA {
    LW2080_CTRL_FAN_COOLER_STATUS_DATA_ACTIVE               active;
    LW2080_CTRL_FAN_COOLER_STATUS_DATA_ACTIVE_PWM           activePwm;
    LW2080_CTRL_FAN_COOLER_STATUS_DATA_ACTIVE_PWM_TACH_CORR activePwmTachCorr;
} LW2080_CTRL_FAN_COOLER_STATUS_DATA;


/*!
 * Structure representing the dynamic state associated with a FAN_COOLER.
 */
typedef struct LW2080_CTRL_FAN_COOLER_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ               super;

    /*!
     * @ref LW2080_CTRL_FAN_COOLER_TYPE_<xyz>.
     */
    LwU8                               type;

    /*!
     * Type-specific information.
     */
    LW2080_CTRL_FAN_COOLER_STATUS_DATA data;
} LW2080_CTRL_FAN_COOLER_STATUS;

/*!
 * LW2080_CTRL_CMD_FAN_COOLER_GET_STATUS
 *
 * This command returns the FAN_COOLER dynamic state as specified by the
 * Fan Cooler Table.
 *
 * https://wiki.lwpu.com/engwiki/index.php/Resman/Thermal_Control/Fan_Tables(s)_1.0_Specification#Fan_Cooler_Table
 *
 * See LW2080_CTRL_FAN_COOLER_STATUS_PARAMS for documentation on the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FAN_COOLER_GET_STATUS (0x2080052fU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | LW2080_CTRL_FAN_COOLER_STATUS_PARAMS_MESSAGE_ID" */

/*!
 * Structure representing the dynamic state associated with the GPU's
 * FAN_COOLER fan cooler functionality.
 */
#define LW2080_CTRL_FAN_COOLER_STATUS_PARAMS_MESSAGE_ID (0x2FU)

typedef struct LW2080_CTRL_FAN_COOLER_STATUS_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32   super;

    /*!
     * [out] - Array of FAN_COOLER entries. Has valid indexes corresponding to
     * the bits set in @ref objMask.
     */
    LW2080_CTRL_FAN_COOLER_STATUS coolers[LW2080_CTRL_FAN_COOLER_MAX_COOLERS];
} LW2080_CTRL_FAN_COOLER_STATUS_PARAMS;
typedef struct LW2080_CTRL_FAN_COOLER_STATUS_PARAMS *PLW2080_CTRL_FAN_COOLER_STATUS_PARAMS;

/* ------- FAN_COOLER's GET/SET_CONTROL RMCTRL defines and structures ------ */

/*!
 * Structure representing the control parameters associated with a
 * ACTIVE FAN_COOLER.
 */
typedef struct LW2080_CTRL_FAN_COOLER_CONTROL_DATA_ACTIVE {
    /*!
     * Minimum RPM.
     */
    LwU32       rpmMin;

    /*!
     * Acoustic Maximum RPM.
     */
    LwU32       rpmMax;

    /*!
     * Reflects the state of fan level (normalized %) override.
     * Level override has highest priority since it was selected that
     * the parent classes has precedence over children.
     */
    LwBool      bLevelSimActive;

    /*!
     * Override value for fan level (normalized %).
     * Applicable only when @ref bLevelSimActive == LW_TRUE.
     */
    LwUFXP16_16 levelSim;
} LW2080_CTRL_FAN_COOLER_CONTROL_DATA_ACTIVE;

/*!
 * Structure representing the control parameters associated with a
 * ACTIVE_PWM FAN_COOLER.
 */
typedef struct LW2080_CTRL_FAN_COOLER_CONTROL_DATA_ACTIVE_PWM {
    /*!
     * LW2080_CTRL_FAN_COOLER_CONTROL_DATA_ACTIVE super class.
     * This should always be the first member!
     */
    LW2080_CTRL_FAN_COOLER_CONTROL_DATA_ACTIVE active;

    /*!
     * Electrical Minimum Fan Speed (PWM rate in percents).
     */
    LwUFXP16_16                                pwmMin;

    /*!
     * Electrical Maximum Fan Speed (PWM rate in percents).
     */
    LwUFXP16_16                                pwmMax;

    /*!
     * Reflects the state of Electrical Fan Speed (PWM) override.
     * PWM override has lower priority than parent's class level override since
     * it was selected that the parent classes has precedence over children.
     */
    LwBool                                     bPwmSimActive;

    /*!
     * Override value for Electrical Fan Speed (PWM rate in percents).
     * Applicable only when @ref bPwmSimActive == LW_TRUE.
     */
    LwUFXP16_16                                pwmSim;

    /*!
     * MAX Fan speed settings (PWM rate in percents).
     */
    LwUFXP16_16                                maxFanPwm;
} LW2080_CTRL_FAN_COOLER_CONTROL_DATA_ACTIVE_PWM;

/*!
 * Structure representing the control parameters associated with a
 * ACTIVE_PWM_TACH_CORR FAN_COOLER.
 */
typedef struct LW2080_CTRL_FAN_COOLER_CONTROL_DATA_ACTIVE_PWM_TACH_CORR {
    /*!
     * LW2080_CTRL_FAN_COOLER_CONTROL_DATA_ACTIVE_PWM super class.
     * This should always be the first member!
     */
    LW2080_CTRL_FAN_COOLER_CONTROL_DATA_ACTIVE_PWM activePwm;

    /*!
     * Tachometer Feedback Proportional Gain.
     */
    LwSFXP16_16                                    propGain;

    /*!
     * An absolute offset (in PWM percents) from actual/measured PWM speed, used
     * on systems exploiting fan-sharing, preventing tachometer driven policies
     * to reduce PWM speed to @ref pwmMin (see bug 1398842).
     */
    LwUFXP16_16                                    pwmFloorLimitOffset;

    /*!
     * Reflects the state of RPM override.
     * RPM override has lower priority than both level and PWM overrides since
     * it was selected that the parent classes has precedence over children.
     */
    LwBool                                         bRpmSimActive;

    /*!
     * Override value for RPM settings.
     * Applicable only when @ref bRpmSimActive == LW_TRUE.
     */
    LwU32                                          rpmSim;
} LW2080_CTRL_FAN_COOLER_CONTROL_DATA_ACTIVE_PWM_TACH_CORR;

/*!
 * Union of type-specific control parameters.
 */
typedef union LW2080_CTRL_FAN_COOLER_CONTROL_DATA {
    LW2080_CTRL_FAN_COOLER_CONTROL_DATA_ACTIVE               active;
    LW2080_CTRL_FAN_COOLER_CONTROL_DATA_ACTIVE_PWM           activePwm;
    LW2080_CTRL_FAN_COOLER_CONTROL_DATA_ACTIVE_PWM_TACH_CORR activePwmTachCorr;
} LW2080_CTRL_FAN_COOLER_CONTROL_DATA;


/*!
 * Structure representing the control parameters associated with a FAN_COOLER.
 */
typedef struct LW2080_CTRL_FAN_COOLER_CONTROL {
    /*!
     * @ref LW2080_CTRL_FAN_COOLER_TYPE_<xyz>.
     */
    LwU8                                type;

    /*!
     * Type-specific information.
     */
    LW2080_CTRL_FAN_COOLER_CONTROL_DATA data;
} LW2080_CTRL_FAN_COOLER_CONTROL;

/*!
 * LW2080_CTRL_CMD_FAN_COOLER_GET_CONTROL
 *
 * This command returns the FAN_COOLER control parameters as specified by the
 * Fan Cooler Table.
 *
 * https://wiki.lwpu.com/engwiki/index.php/Resman/Thermal_Control/Fan_Tables(s)_1.0_Specification#Fan_Cooler_Table
 *
 * See LW2080_CTRL_FAN_COOLER_CONTROL_PARAMS for documentation on the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FAN_COOLER_GET_CONTROL (0x20800530U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | 0x30" */

/*!
 * LW2080_CTRL_CMD_FAN_COOLER_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set of
 * FAN_COOLER entries in the Fan Cooler Table, and applies these new
 * parameters to the set of FAN_COOLER entries.
 *
 * https://wiki.lwpu.com/engwiki/index.php/Resman/Thermal_Control/Fan_Tables(s)_1.0_Specification#Fan_Cooler_Table
 *
 * See LW2080_CTRL_FAN_COOLER_CONTROL_PARAMS for documentation on the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FAN_COOLER_SET_CONTROL (0x20800531U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | 0x31" */

/*!
 * Structure representing the control parameters associated with the GPU's
 * FAN_COOLER fan cooler functionality.
 */
typedef struct LW2080_CTRL_FAN_COOLER_CONTROL_PARAMS {
    /*!
     * [in] - Mask of FAN_COOLER entries requested by the client.
     */
    LwU32                          coolerMask;

    /*!
     * [in] Flag specifying the set of values to retrieve:
     * - VBIOS default (LW_TRUE)
     * - lwrrently active (LW_FALSE)
     */
    LwBool                         bDefault;

    /*!
     * [out] - Array of FAN_COOLER entries. Has valid indexes corresponding to
     * the bits set in @ref coolerMask.
     */
    LW2080_CTRL_FAN_COOLER_CONTROL coolers[LW2080_CTRL_FAN_COOLER_MAX_COOLERS];
} LW2080_CTRL_FAN_COOLER_CONTROL_PARAMS;

/*!
 * Macros for Fan Policy Types.
 */
#define LW2080_CTRL_FAN_POLICY_TYPE_ILWALID                                   0x00U
#define LW2080_CTRL_FAN_POLICY_TYPE_IIR_TJ_FIXED_DUAL_SLOPE_PWM_V20           0x01U
#define LW2080_CTRL_FAN_POLICY_TYPE_IIR_TJ_FIXED_DUAL_SLOPE_PWM_V30           0x02U

/*!
 * Macros for version of underlying Fan Policy Table in the driver.
 */
#define LW2080_CTRL_FAN_POLICIES_VERSION_ILWALID                              0x00U
#define LW2080_CTRL_FAN_POLICIES_VERSION_20                                   0x01U
#define LW2080_CTRL_FAN_POLICIES_VERSION_30                                   0x02U

/*!
 * Macros for Fan Policy Interface Types.
 */
#define LW2080_CTRL_FAN_FAN_POLICY_INTERFACE_TYPE_IIR_TJ_FIXED_DUAL_SLOPE_PWM 0x01U

/*!
 * Special value corresponding to an invalid Fan Policy Index.
 */
#define LW2080_CTRL_FAN_POLICY_INDEX_ILWALID                                  LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Maximum number of FAN_POLICIES which can be supported in the RM or PMU.
 */
#define LW2080_CTRL_FAN_POLICY_MAX_POLICIES                                   16U

/*!
 * Number of Fan Lwrve Points for IIR_TJ_FIXED_DUAL_SLOPE_PWM policy.
 */
#define LW2080_CTRL_FAN_POLICY_IIR_TJ_FIXED_DUAL_SLOPE_PWM_FAN_LWRVE_PTS      3U


/*!
 * @brief Maximum size of the cirlwlar buffer of samples to keep a history.
 */
#define LW2080_CTRL_FAN_PMUMON_FAN_COOLER_SAMPLE_COUNT                        (50U)

/*!
 * @brief A single sample of the thermal channels at a particular point in time.
 */
typedef struct LW2080_CTRL_FAN_PMUMON_FAN_COOLER_SAMPLE {
    /*!
     * @brief Ptimer timestamp of when this data was collected.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_PMUMON_SAMPLE super, 8);

    /*!
     * @deprecated. To be deleted when lwapi usage of this field is removed.
     */
    LwS32                                rpm[LW2080_CTRL_FAN_COOLER_MAX_COOLERS];

    /*!
     * @brief Output parameter: the current fan cooler statuses.
     */
    LW2080_CTRL_FAN_COOLER_STATUS_PARAMS data;
} LW2080_CTRL_FAN_PMUMON_FAN_COOLER_SAMPLE;

/*!
 * @brief Input/output parameters for
 * @ref LW2080_CTRL_CMD_FAN_PMUMON_FAN_COOLER_SAMPLES.
 */
#define LW2080_CTRL_FAN_PMUMON_FAN_COOLER_GET_SAMPLES_PARAMS_MESSAGE_ID (0x37U)

typedef struct LW2080_CTRL_FAN_PMUMON_FAN_COOLER_GET_SAMPLES_PARAMS {
    /*!
     * @brief Metadata for the samples.
     */
    LW2080_CTRL_PMUMON_GET_SAMPLES_SUPER super;

    /*!
     * @brief A collection of data samples for thermal data.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_FAN_PMUMON_FAN_COOLER_SAMPLE samples[LW2080_CTRL_FAN_PMUMON_FAN_COOLER_SAMPLE_COUNT], 8);
} LW2080_CTRL_FAN_PMUMON_FAN_COOLER_GET_SAMPLES_PARAMS;
typedef struct LW2080_CTRL_FAN_PMUMON_FAN_COOLER_GET_SAMPLES_PARAMS *PLW2080_CTRL_FAN_PMUMON_FAN_COOLER_GET_SAMPLES_PARAMS;

#define LW2080_CTRL_CMD_FAN_PMUMON_FAN_COOLER_GET_SAMPLES (0x20800537U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | LW2080_CTRL_FAN_PMUMON_FAN_COOLER_GET_SAMPLES_PARAMS_MESSAGE_ID" */

/* ---------- FAN_POLICY's GET_INFO RMCTRL defines and structures ---------- */

/*!
 * Structure representing the static information associated with a
 * IIR_TJ_FIXED_DUAL_SLOPE_PWM FAN_POLICY.
 */
typedef struct LW2080_CTRL_FAN_POLICY_INFO_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM {
    /*!
     * BOARDOBJ_INTERFACE super class. This should always be the first member!
     */
    LW2080_CTRL_BOARDOBJ_INTERFACE super;

    /*!
     * Policy Control Selection.
     */
    LwBool                         bUsePwm;

    /*!
     * Specifies whether Fan Stop sub-policy is supported.
     */
    LwBool                         bFanStopSupported;

    /*!
     * Specifies whether Fan Stop sub-policy is enabled by default.
     */
    LwBool                         bFanStopEnableDefault;

    /*!
     * Fan Start Min Hold Time - (ms). Minimum time to spend at Electrical Fan
     * Speed Min to prevent fan overshoot after being re-started.
     */
    LwU16                          fanStartMinHoldTimems;

    /*!
     * Power Topology Index in the Power Topology Table. If the value is 0xFF,
     * only GPU Tj needs to be considered for Fan Stop sub-policy.
     */
    LwU8                           fanStopPowerTopologyIdx;
} LW2080_CTRL_FAN_POLICY_INFO_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM;

/*!
 * Structure representing the static information associated with a
 * V20 FAN_POLICY.
 */
typedef struct LW2080_CTRL_FAN_POLICY_INFO_V20 {
    /*!
     * Index into the Fan Cooler Table.
     */
    LwU8  coolerIdx;

    /*!
     * Sampling period.
     */
    LwU16 samplingPeriodms;
} LW2080_CTRL_FAN_POLICY_INFO_V20;

/*!
 * Structure representing the static information associated with a
 * IIR_TJ_FIXED_DUAL_SLOPE_PWM_V20 FAN_POLICY.
 */
typedef struct LW2080_CTRL_FAN_POLICY_INFO_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM_V20 {
    /*!
     * Super class.
     */
    LW2080_CTRL_FAN_POLICY_INFO_V20                              fanPolicyV20;

    /*!
     * IIR_TJ_FIXED_DUAL_SLOPE_PWM interface class.
     */
    LW2080_CTRL_FAN_POLICY_INFO_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM iirTFDSP;
} LW2080_CTRL_FAN_POLICY_INFO_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM_V20;

/*!
 * Structure representing the static information associated with a
 * IIR_TJ_FIXED_DUAL_SLOPE_PWM_V30 FAN_POLICY.
 */
typedef struct LW2080_CTRL_FAN_POLICY_INFO_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM_V30 {
    /*!
     * Specifies whether Fan Lwrve Point 2 - Tj (C) needs to be overridden.
     */
    LwBool                                                       bFanLwrvePt2TjOverride;

    /*!
     * Specifies whether Fan Lwrves can be adjusted.
     */
    LwBool                                                       bFanLwrveAdjSupported;

    /*!
     * Offset to be applied for Fan Lwrve Point 2 Tj (C).
     */
    LwTemp                                                       fanLwrveTjPt2Offset;

    /*!
     * IIR_TJ_FIXED_DUAL_SLOPE_PWM interface class.
     */
    LW2080_CTRL_FAN_POLICY_INFO_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM iirTFDSP;
} LW2080_CTRL_FAN_POLICY_INFO_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM_V30;

/*!
 * Union of type-specific static information data.
 */
typedef union LW2080_CTRL_FAN_POLICY_INFO_DATA {
    LW2080_CTRL_FAN_POLICY_INFO_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM_V20 iirTFDSPV20;
    LW2080_CTRL_FAN_POLICY_INFO_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM_V30 iirTFDSPV30;
} LW2080_CTRL_FAN_POLICY_INFO_DATA;


/*!
 * Structure representing the static information associated with a FAN_POLICY.
 */
typedef struct LW2080_CTRL_FAN_POLICY_INFO {
    /*!
     * @ref LW2080_CTRL_FAN_POLICY_TYPE_<xyz>.
     */
    LwU8                             type;

    /*!
     * Index into the Thermal Channel Table.
     */
    LwU8                             thermChIdx;

    /*!
     * Type-specific information.
     */
    LW2080_CTRL_FAN_POLICY_INFO_DATA data;
} LW2080_CTRL_FAN_POLICY_INFO;

/*!
 * LW2080_CTRL_CMD_FAN_POLICY_GET_INFO
 *
 * This command returns the FAN_POLICY static information as specified by the
 * Fan Policy Table.
 *
 * https://wiki.lwpu.com/engwiki/index.php/Resman/Thermal_Control/Fan_Tables(s)_1.0_Specification#Fan_Policy_Table
 *
 * See LW2080_CTRL_FAN_POLICY_INFO_PARAMS for documentation on the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FAN_POLICY_GET_INFO (0x20800532U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | LW2080_CTRL_FAN_POLICY_INFO_PARAMS_MESSAGE_ID" */

typedef struct LW2080_CTRL_FAN_POLICIES_V20 {
    /*!
     * Fan Policy Table index for Fan Policy controlling the first GPU fan.
     *
     * @note LW2080_CTRL_FAN_POLICY_INDEX_ILWALID indicates that
     * this policy is not present/specified on this GPU.
     */
    LwU8 gpuPolicyIdx0;

    /*!
     * Fan Policy Table index for Fan Policy controlling the second GPU fan.
     *
     * @note LW2080_CTRL_FAN_POLICY_INDEX_ILWALID indicates that
     * this policy is not present/specified on this GPU.
     */
    LwU8 gpuPolicyIdx1;
} LW2080_CTRL_FAN_POLICIES_V20;

/*!
 * Union of type-specific static information data.
 */


/*!
 * Structure representing the static state information associated with the GPU's
 * FAN_POLICY fan policy functionality.
 */
#define LW2080_CTRL_FAN_POLICY_INFO_PARAMS_MESSAGE_ID (0x32U)

typedef struct LW2080_CTRL_FAN_POLICY_INFO_PARAMS {
    /*!
     * [out] - Mask of FAN_POLICY entries specified on this GPU.
     */
    LwU32 policyMask;

    /*!
     * [out] - Version of underlying Fan Policy Table in the driver @ref
     * LW2080_CTRL_FAN_POLICIES_VERSION_<xyz>.
     */
    LwU8  version;

    /*!
     * Type-specific information.
     */
    union {
        LW2080_CTRL_FAN_POLICIES_V20 v20;
    } data;

    /*!
     * [out] - Array of FAN_POLICY entries. Has valid indexes corresponding to
     * the bits set in @ref policyMask.
     */
    LW2080_CTRL_FAN_POLICY_INFO policies[LW2080_CTRL_FAN_POLICY_MAX_POLICIES];
} LW2080_CTRL_FAN_POLICY_INFO_PARAMS;

/* --------- FAN_POLICY's GET_STATUS RMCTRL defines and structures --------- */

/*!
 * Structure representing the dynamic state associated with a
 * IIR_TJ_FIXED_DUAL_SLOPE_PWM FAN_POLICY.
 */
typedef struct LW2080_CTRL_FAN_POLICY_STATUS_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM {
    /*!
     * BOARDOBJ_INTERFACE super class. This should always be the first member!
     */
    LW2080_CTRL_BOARDOBJ_INTERFACE super;

    /*!
     * Current Short Term Average Temperature (C).
     */
    LwSFXP10_22                    tjAvgShortTerm;

    /*!
     * Current Long Term Average Temperature (C).
     */
    LwSFXP10_22                    tjAvgLongTerm;

    /*!
     * Target PWM (in percents), used when @ref INFO::bUsePwm is set.
     */
    LwUFXP16_16                    targetPwm;

    /*!
     * Target RPM, used when @ref INFO::bUsePwm is not set.
     */
    LwU32                          targetRpm;

    /*!
     * Current GPU Tj.
     */
    LwTemp                         tjLwrrent;

    /*!
     * Filtered PWR_CHANNEL value, used when Fan Stop sub-policy is enabled.
     */
    LwU32                          filteredPwrmW;

    /*!
     * Specifies whether Fan Stop sub-policy is active.
     */
    LwBool                         bFanStopActive;
} LW2080_CTRL_FAN_POLICY_STATUS_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM;

/*!
 * Structure representing the dynamic state associated with a
 * IIR_TJ_FIXED_DUAL_SLOPE_PWM_V20 FAN_POLICY.
 */
typedef struct LW2080_CTRL_FAN_POLICY_STATUS_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM_V20 {
    /*!
     * IIR_TJ_FIXED_DUAL_SLOPE_PWM interface class.
     */
    LW2080_CTRL_FAN_POLICY_STATUS_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM iirTFDSP;
} LW2080_CTRL_FAN_POLICY_STATUS_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM_V20;

/*!
 * Structure representing the dynamic state associated with a
 * IIR_TJ_FIXED_DUAL_SLOPE_PWM_V30 FAN_POLICY.
 */
typedef struct LW2080_CTRL_FAN_POLICY_STATUS_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM_V30 {
    /*!
     * IIR_TJ_FIXED_DUAL_SLOPE_PWM interface class.
     */
    LW2080_CTRL_FAN_POLICY_STATUS_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM iirTFDSP;
} LW2080_CTRL_FAN_POLICY_STATUS_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM_V30;

/*!
 * Union of type-specific dynamic data.
 */
typedef union LW2080_CTRL_FAN_POLICY_STATUS_DATA {
    LW2080_CTRL_FAN_POLICY_STATUS_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM_V20 iirTFDSPV20;
    LW2080_CTRL_FAN_POLICY_STATUS_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM_V30 iirTFDSPV30;
} LW2080_CTRL_FAN_POLICY_STATUS_DATA;


/*!
 * Structure representing the dynamic state associated with a FAN_POLICY.
 */
typedef struct LW2080_CTRL_FAN_POLICY_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ               super;

    /*!
     * @ref LW2080_CTRL_FAN_POLICY_TYPE_<xyz>.
     */
    LwU8                               type;

    /*!
     * Denotes the control action to be taken on the fan described via
     * @ref LW2080_CTRL_FAN_CTRL_ACTION_<xyz>.
     */
    LwU8                               fanCtrlAction;

    /*!
     * Type-specific information.
     */
    LW2080_CTRL_FAN_POLICY_STATUS_DATA data;
} LW2080_CTRL_FAN_POLICY_STATUS;

/*!
 * LW2080_CTRL_CMD_FAN_POLICY_GET_STATUS
 *
 * This command returns the FAN_POLICY dynamic state as specified by the
 * Fan Policy Table.
 *
 * https://wiki.lwpu.com/engwiki/index.php/Resman/Thermal_Control/Fan_Tables(s)_1.0_Specification#Fan_Policy_Table
 *
 * See LW2080_CTRL_FAN_POLICY_STATUS_PARAMS for documentation on the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FAN_POLICY_GET_STATUS (0x20800533U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | LW2080_CTRL_FAN_POLICY_STATUS_PARAMS_MESSAGE_ID" */

/*!
 * Structure representing the dynamic state associated with the GPU's
 * FAN_POLICY fan policy functionality.
 */
#define LW2080_CTRL_FAN_POLICY_STATUS_PARAMS_MESSAGE_ID (0x33U)

typedef struct LW2080_CTRL_FAN_POLICY_STATUS_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32   super;

    /*!
     * [out] - Array of FAN_POLICY entries. Has valid indexes corresponding to
     * the bits set in @ref objMask.
     */
    LW2080_CTRL_FAN_POLICY_STATUS policies[LW2080_CTRL_FAN_POLICY_MAX_POLICIES];
} LW2080_CTRL_FAN_POLICY_STATUS_PARAMS;
typedef struct LW2080_CTRL_FAN_POLICY_STATUS_PARAMS *PLW2080_CTRL_FAN_POLICY_STATUS_PARAMS;

/* ------- FAN_POLICY's GET/SET_CONTROL RMCTRL defines and structures ------ */

/*!
 * Structure representing single fan operating point {Tj, PWM, RPM}.
 */

typedef struct LW2080_CTRL_FAN_POLICY_CONTROL_DATA_FAN_OPERATING_POINT_ITFDSP {
    /*!
     * Tj Fan Lwrve Point (C).
     */
    LwSFXP24_8  tj;

    /*!
     * PWM Fan Lwrve Point.
     */
    LwUFXP16_16 pwm;

    /*!
     * RPM Fan Lwrve Point.
     */
    LwU32       rpm;
} LW2080_CTRL_FAN_POLICY_CONTROL_DATA_FAN_OPERATING_POINT_ITFDSP;

/*!
 * Structure representing the control parameters associated with a
 * IIR_TJ_FIXED_DUAL_SLOPE_PWM FAN_POLICY.
 */
typedef struct LW2080_CTRL_FAN_POLICY_CONTROL_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM {
    /*!
     * BOARDOBJ_INTERFACE super class. This should always be the first member!
     */
    LW2080_CTRL_BOARDOBJ_INTERFACE                                 super;

    /*!
     * Minimum IIR Gain.
     */
    LwSFXP16_16                                                    iirGainMin;

    /*!
     * Maximum IIR Gain.
     */
    LwSFXP16_16                                                    iirGainMax;

    /*!
     * Short Term IIR Gain.
     */
    LwSFXP16_16                                                    iirGainShortTerm;

    /*!
     * IIR Filter Power.
     */
    LwU8                                                           iirFilterPower;

    /*!
     * IIR Long Term Sampling Ratio.
     */
    LwU8                                                           iirLongTermSamplingRatio;

    /*!
     * IIR Filter Lower Width (C).
     */
    LwSFXP24_8                                                     iirFilterWidthUpper;

    /*!
     * IIR Filter Upper Width (C).
     */
    LwSFXP24_8                                                     iirFilterWidthLower;

    LW2080_CTRL_FAN_POLICY_CONTROL_DATA_FAN_OPERATING_POINT_ITFDSP fanLwrvePts[LW2080_CTRL_FAN_POLICY_IIR_TJ_FIXED_DUAL_SLOPE_PWM_FAN_LWRVE_PTS];

    /*!
     * Specifies whether Fan Stop sub-policy needs to be enabled.
     */
    LwBool                                                         bFanStopEnable;

    /*!
     * Fan Stop Lower Temperature Limit - Tj (C). Fan will be stopped when GPU
     * Tj falls below this temperature limit.
     */
    LwTemp                                                         fanStopTempLimitLower;

    /*!
     * Fan Start Upper Temperature Limit - Tj (C). Fan will be re-started when
     * GPU Tj is at or above this temperature limit.
     */
    LwTemp                                                         fanStartTempLimitUpper;

    /*!
     * Fan Stop Lower Power Limit - (mW). Fan will be stopped when both power
     * (specified by @ref fanStopPowerTopologyIdx) and GPU Tj (specified by
     * @ref fanStopTempLimitLower) fall below their respective limits.
     */
    LwU32                                                          fanStopPowerLimitLowermW;

    /*!
     * Fan Start Upper Power Limit - (mW). Fan will be re-started when either
     * power (specified by @ref fanStopPowerTopologyIdx) or GPU Tj (specified
     * by @ref fanStartTempLimitUpper) is at or above their respective limits.
     */
    LwU32                                                          fanStartPowerLimitUppermW;

    /*!
     * Fan Stop IIR Power Gain. The gain value is used for filtering power
     * (specified by @ref fanStopPowerTopologyIdx) before being compared
     * against @ref fanStopPowerLimitLower and @ref fanStartPowerLimitUpper.
     */
    LwSFXP20_12                                                    fanStopIIRGainPower;
} LW2080_CTRL_FAN_POLICY_CONTROL_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM;

/*!
 * Structure representing the control parameters associated with a
 * IIR_TJ_FIXED_DUAL_SLOPE_PWM_V20 FAN_POLICY.
 */
typedef struct LW2080_CTRL_FAN_POLICY_CONTROL_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM_V20 {
    /*!
     * IIR_TJ_FIXED_DUAL_SLOPE_PWM interface class.
     */
    LW2080_CTRL_FAN_POLICY_CONTROL_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM iirTFDSP;
} LW2080_CTRL_FAN_POLICY_CONTROL_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM_V20;

/*!
 * Structure representing the control parameters associated with a
 * IIR_TJ_FIXED_DUAL_SLOPE_PWM_V30 FAN_POLICY.
 */
typedef struct LW2080_CTRL_FAN_POLICY_CONTROL_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM_V30 {
    /*!
     * IIR_TJ_FIXED_DUAL_SLOPE_PWM interface class.
     */
    LW2080_CTRL_FAN_POLICY_CONTROL_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM iirTFDSP;
} LW2080_CTRL_FAN_POLICY_CONTROL_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM_V30;

/*!
 * Union of type-specific control parameters.
 */
typedef union LW2080_CTRL_FAN_POLICY_CONTROL_DATA {
    LW2080_CTRL_FAN_POLICY_CONTROL_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM_V20 iirTFDSPV20;
    LW2080_CTRL_FAN_POLICY_CONTROL_DATA_IIR_TJ_FIXED_DUAL_SLOPE_PWM_V30 iirTFDSPV30;
} LW2080_CTRL_FAN_POLICY_CONTROL_DATA;


/*!
 * Structure representing the control parameters associated with a FAN_POLICY.
 */
typedef struct LW2080_CTRL_FAN_POLICY_CONTROL {
    /*!
     * @ref LW2080_CTRL_FAN_POLICY_TYPE_<xyz>.
     */
    LwU8                                type;

    /*!
     * Type-specific information.
     */
    LW2080_CTRL_FAN_POLICY_CONTROL_DATA data;
} LW2080_CTRL_FAN_POLICY_CONTROL;

/*!
 * LW2080_CTRL_CMD_FAN_POLICY_GET_CONTROL
 *
 * This command returns the FAN_POLICY control parameters as specified by the
 * Fan Policy Table.
 *
 * https://wiki.lwpu.com/engwiki/index.php/Resman/Thermal_Control/Fan_Tables(s)_1.0_Specification#Fan_Policy_Table
 *
 * See LW2080_CTRL_FAN_POLICY_CONTROL_PARAMS for documentation on the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FAN_POLICY_GET_CONTROL (0x20800534U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | 0x34" */

/*!
 * LW2080_CTRL_CMD_FAN_POLICY_SET_CONTROL
 *
 * This command accepts client-specified control parameters for a set of
 * FAN_POLICY entries in the Fan Policy Table, and applies these new
 * parameters to the set of FAN_POLICY entries.
 *
 * https://wiki.lwpu.com/engwiki/index.php/Resman/Thermal_Control/Fan_Tables(s)_1.0_Specification#Fan_Policy_Table
 *
 * See LW2080_CTRL_FAN_POLICY_CONTROL_PARAMS for documentation on the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FAN_POLICY_SET_CONTROL (0x20800535U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | 0x35" */

/*!
 * Structure representing the control parameters associated with the GPU's
 * FAN_POLICY fan policy functionality.
 */
typedef struct LW2080_CTRL_FAN_POLICY_CONTROL_PARAMS {
    /*!
     * [in] - Mask of FAN_POLICY entries requested by the client.
     */
    LwU32                          policyMask;

    /*!
     * [in] Flag specifying the set of values to retrieve:
     * - VBIOS default (LW_TRUE)
     * - lwrrently active (LW_FALSE)
     */
    LwBool                         bDefault;

    /*!
     * [out] - Array of FAN_POLICY entries. Has valid indexes corresponding to
     * the bits set in @ref policyMask.
     */
    LW2080_CTRL_FAN_POLICY_CONTROL policies[LW2080_CTRL_FAN_POLICY_MAX_POLICIES];
} LW2080_CTRL_FAN_POLICY_CONTROL_PARAMS;

/*!
 * Maximum number of THERM_DEVICEs which can be supported in the RM or PMU.
 */
#define LW2080_CTRL_THERMAL_THERM_DEVICE_MAX_COUNT LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS

/*!
 * Structure of static information specific to I2C devices.
 */
typedef struct LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_I2C {
    /*!
     * Specifies the I2C Device Index in the DCB I2C Devices Table.
     */
    LwU8 i2cDevIdx;
} LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_I2C;

/*!
 * Structure of static information specific to I2C ADM1032 device.
 */
typedef struct LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_I2C_ADM1032 {
    /*!
     * I2C Device information.
     */
    LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_I2C i2c;
} LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_I2C_ADM1032;

/*!
 * Structure of static information specific to I2C MAX6649 device.
 */
typedef struct LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_I2C_MAX6649 {
    /*!
     * I2C Device information.
     */
    LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_I2C i2c;
} LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_I2C_MAX6649;

/*!
 * Structure of static information specific to I2C TMP411 device.
 */
typedef struct LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_I2C_TMP411 {
    /*!
     * I2C Device information.
     */
    LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_I2C i2c;
} LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_I2C_TMP411;

/*!
 * Structure of static information specific to I2C ADT7461 device.
 */
typedef struct LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_I2C_ADT7461 {
    /*!
     * I2C Device information.
     */
    LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_I2C i2c;
} LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_I2C_ADT7461;

/*!
 * Structure of static information specific to I2C TMP451 device.
 */
typedef struct LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_I2C_TMP451 {
    /*!
     * I2C Device information.
     */
    LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_I2C i2c;
} LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_I2C_TMP451;

/*!
 * Structure of static information specific to GPU_GPC_TSOSC device.
 */
typedef struct LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_GPU_GPC_TSOSC {
    /*!
     * Index that selects the corresponding GPC for this GPU_GPC_TSOSC THERM_DEVICE.
     */
    LwU8 gpcTsoscIdx;
} LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_GPU_GPC_TSOSC;

/*!
 * Structure of static information specific to HBM2_SITE device.
 */
typedef struct LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_HBM2_SITE {
    /*!
     * Index that selects the corresponding HBM2 site for this HBM2_SITE THERM_DEVICE.
     */
    LwU8 siteIdx;
} LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_HBM2_SITE;

/*!
 * Union of THERMAL_DEVICE_CLASS.
 */


/*!
 * Structure representing the static information associated with a THERM_DEVICE.
 */
typedef struct LW2080_CTRL_THERMAL_DEVICE_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;

     /*!
     * @ref LW2080_CTRL_THERMAL_THERM_DEVICE_CLASS_<xyz>.
     */
    LwU8                 type;

    /*!
     * Class-specific information.
     */
    union {
        LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_I2C_ADM1032   adm1032;
        LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_I2C_MAX6649   max6649;
        LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_I2C_TMP411    tmp411;
        LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_I2C_ADT7461   adt7461;
        LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_I2C_TMP451    tmp451;
        LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_GPU_GPC_TSOSC gpuGpcTsosc;
        LW2080_CTRL_THERMAL_DEVICE_INFO_DATA_HBM2_SITE     hbm2Site;
    } data;
} LW2080_CTRL_THERMAL_DEVICE_INFO;

/*!
 * Structure representing the static state information associated with the
 * GPU's THERM_DEVICE thermal device functionality.
 */
#define LW2080_CTRL_THERMAL_DEVICE_INFO_PARAMS_MESSAGE_ID (0x36U)

typedef struct LW2080_CTRL_THERMAL_DEVICE_INFO_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32     super;

    /*!
     * [out] - Array of THERMAL_DEVICE entries. Has valid indexes corresponding to
     * the bits set in @ref objMask.
     */
    LW2080_CTRL_THERMAL_DEVICE_INFO device[LW2080_CTRL_THERMAL_THERM_DEVICE_MAX_COUNT];
} LW2080_CTRL_THERMAL_DEVICE_INFO_PARAMS;

/*!
 * LW2080_CTRL_CMD_THERMAL_DEVICE_GET_INFO
 *
 * This command returns the THERM_DEVICE static information as specified by
 * the Thermal device Table.
 *
 * https://wiki.lwpu.com/engwiki/index.php?title=Resman/Thermal_Control/Thermal_Sensor_Table(s)_1.0_Specification
 *
 * See LW2080_CTRL_THERMAL_DEVICE_INFO_PARAMS for documentation on the
 * parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_THERMAL_DEVICE_GET_INFO (0x20800536U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | LW2080_CTRL_THERMAL_DEVICE_INFO_PARAMS_MESSAGE_ID" */

/*!
 * Structure representing the attributes specific to THERM_CHANNEL_DEVICE.
 */
typedef struct LW2080_CTRL_THERMAL_CHANNEL_INFO_DATA_DEVICE {
    /*!
     * Index into the Thermal Device Table for the THERM_DEVICE from which this
     * THERM_CHANNEL should query temperature value.
     */
    LwU8 thermDevIdx;

    /*!
     * Provider index to query temperature value.
     */
    LwU8 thermDevProvIdx;
} LW2080_CTRL_THERMAL_CHANNEL_INFO_DATA_DEVICE;

/*!
 * Union of THERMAL_CHANNEL class-specific data.
 */


/*!
 * Maximum number of THERM_CHANNELs which can be supported in the RM or PMU.
 */
#define LW2080_CTRL_THERMAL_THERM_CHANNEL_MAX_CHANNELS LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS

/*!
 * Structure representing the static information associated with a THERM_CHANNEL.
 */
typedef struct LW2080_CTRL_THERMAL_CHANNEL_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;

    /*!
     * @ref LW2080_CTRL_THERMAL_THERM_CHANNEL_CLASS_<xyz>.
     */
    LwU8                 type;

    /*!
     * @ref LW2080_CTRL_THERMAL_THERM_CHANNEL_TYPE_<xyz>.
     */
    LwU8                 chType;

    /*!
     * @ref LW2080_CTRL_THERMAL_THERM_CHANNEL_REL_LOC_<xyz>.
     */
    LwU8                 relLoc;

    /*!
     * @ref LW2080_CTRL_THERMAL_THERM_CHANNEL_TGT_<xyz>.
     */
    LwU8                 tgtGPU;

    /*!
     * Minimum temperature for a channel.
     */
    LwTemp               minTemp;

    /*!
     * Maximum temperature for a channel.
     */
    LwTemp               maxTemp;

    /*!
     * Temperature scaling. This value is in SFXP24_8 format.
     */
    LwS32                scaling;

    /*!
     * SW Temperature offset.
     */
    LwTemp               offsetSw;

    /*!
     * HW Temperature offset.
     */
    LwTemp               offsetHw;

    /*!
     * Temp sim capability of the channel.
     */
    LwBool               bIsTempSimSupported;

    /*!
     * Temp sim capability of the channel.
     */
    LwBool               bIsTempSimEnabled;

    /*!
     * @ref LW2080_CTRL_THERMAL_THERM_CHANNEL_FLAGS_CHANNEL_<xyz>.
     */
    LwU8                 flags;

    /*!
     * Type-specific information.
     */
    union {
        LW2080_CTRL_THERMAL_CHANNEL_INFO_DATA_DEVICE device;
    } data;
} LW2080_CTRL_THERMAL_CHANNEL_INFO;

/*!
 * Structure representing the static state information associated with the
 * GPU's THERM_CHANNEL thermal channel functionality.
 */
#define LW2080_CTRL_THERMAL_CHANNEL_INFO_PARAMS_MESSAGE_ID (0x3AU)

typedef struct LW2080_CTRL_THERMAL_CHANNEL_INFO_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32      super;

    /*!
     * Primary Channel Index for each
     * LW2080_CTRL_THERMAL_THERM_CHANNEL_TYPE_<xyz>
     */
    LwU8                             priChIdx[LW2080_CTRL_THERMAL_THERM_CHANNEL_TYPE_MAX_COUNT];

    /*!
     * [out] - Array of THERMAL_CHANNEL entries.
     */
    LW2080_CTRL_THERMAL_CHANNEL_INFO channel[LW2080_CTRL_THERMAL_THERM_CHANNEL_MAX_CHANNELS];
} LW2080_CTRL_THERMAL_CHANNEL_INFO_PARAMS;
typedef struct LW2080_CTRL_THERMAL_CHANNEL_INFO_PARAMS *PLW2080_CTRL_THERMAL_CHANNEL_INFO_PARAMS;

/*!
 * LW2080_CTRL_CMD_THERMAL_CHANNEL_GET_INFO
 *
 * This command returns the THERM_CHANNEL static information as specified by
 * the Thermal channel Table.
 *
 * https://wiki.lwpu.com/engwiki/index.php?title=Resman/Thermal_Control/Thermal_Sensor_Table(s)_1.0_Specification
 *
 * See LW2080_CTRL_THERMAL_CHANNEL_INFO_PARAMS for documentation on the
 * parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_THERMAL_CHANNEL_GET_INFO (0x2080053aU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | LW2080_CTRL_THERMAL_CHANNEL_INFO_PARAMS_MESSAGE_ID" */

/*!
 * Structure representing the dynamic state associated with
 * a THERM_CHANNEL.
 */
typedef struct LW2080_CTRL_THERMAL_CHANNEL_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;

    /*!
     * Current temperature of the channel.
     */
    LwTemp               lwrrentTemp;
} LW2080_CTRL_THERMAL_CHANNEL_STATUS;

/*!
 * Structure representing the dynamic state associated with the
 * GPU's THERM_CHANNEL thermal channel functionality.
 */
#define LW2080_CTRL_THERMAL_CHANNEL_STATUS_PARAMS_MESSAGE_ID (0x3BU)

typedef struct LW2080_CTRL_THERMAL_CHANNEL_STATUS_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32        super;

    /*!
     * [out] - Array of THERMAL_CHANNEL entries. Has valid indexes corresponding to
     * the bits set in @ref objMask.
     */
    LW2080_CTRL_THERMAL_CHANNEL_STATUS channel[LW2080_CTRL_THERMAL_THERM_CHANNEL_MAX_CHANNELS];
} LW2080_CTRL_THERMAL_CHANNEL_STATUS_PARAMS;
typedef struct LW2080_CTRL_THERMAL_CHANNEL_STATUS_PARAMS *PLW2080_CTRL_THERMAL_CHANNEL_STATUS_PARAMS;

/*!
 * LW2080_CTRL_CMD_THERMAL_CHANNEL_GET_STATUS
 *
 * This command returns the THERM_CHANNEL dynamic state as specified by
 * the Thermal channel Table.
 *
 * https://wiki.lwpu.com/engwiki/index.php?title=Resman/Thermal_Control/Thermal_Sensor_Table(s)_1.0_Specification
 *
 * See LW2080_CTRL_THERMAL_CHANNEL_STATUS_PARAMS for documentation on the
 * parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_THERMAL_CHANNEL_GET_STATUS (0x2080053bU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | LW2080_CTRL_THERMAL_CHANNEL_STATUS_PARAMS_MESSAGE_ID" */

/*!
 * Structure representing the control parameters associated with
 * a THERM_CHANNEL.
 */
typedef struct LW2080_CTRL_THERMAL_CHANNEL_CONTROL {
    /*!
     * LW2080_CTRL_BOARDOBJ super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;

    /*!
     * Reflects the state of current temperature override.
     */
    LwBool               bTempSimEnable;

    /*!
     * Override the current temperature.
     * Applicable only when @ref bTempSimEnable == LW_TRUE.
     */
    LwTemp               targetTemp;
} LW2080_CTRL_THERMAL_CHANNEL_CONTROL;

/*!
 * Structure representing the control parameters associated with the GPU's
 * THERM_CHANNEL thermal channel functionality.
 */
typedef struct LW2080_CTRL_THERMAL_CHANNEL_CONTROL_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP_E32 super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32         super;

    /*!
     * [out] - Array of THERMAL_CHANNEL entries. Has valid indexes corresponding to
     * the bits set in @ref objMask.
     */
    LW2080_CTRL_THERMAL_CHANNEL_CONTROL channel[LW2080_CTRL_THERMAL_THERM_CHANNEL_MAX_CHANNELS];
} LW2080_CTRL_THERMAL_CHANNEL_CONTROL_PARAMS;

/*!
 * LW2080_CTRL_CMD_THERMAL_CHANNEL_GET_CONTROL
 *
 * This command returns the THERM_CHANNEL control parameters as specified by the
 * THERM_CHANNEL entries in the Thermal channel Table.
 *
 * https://wiki.lwpu.com/engwiki/index.php?title=Resman/Thermal_Control/Thermal_Sensor_Table(s)_1.0_Specification
 *
 * See LW2080_CTRL_THERMAL_CHANNEL_CONTROL_PARAMS for documentation on the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_THERMAL_CHANNEL_GET_CONTROL         (0x2080053lw) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | 0x3C" */

/*!
 * LW2080_CTRL_CMD_THERMAL_CHANNEL_SET_CONTROL
 *
 * This command accepts client-specified control parameters and applies these new
 * parameters to the set of THERM_CHANNEL entries.
 *
 * https://wiki.lwpu.com/engwiki/index.php?title=Resman/Thermal_Control/Thermal_Sensor_Table(s)_1.0_Specification
 *
 * See LW2080_CTRL_THERMAL_CHANNEL_CONTROL_PARAMS for documentation on the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_THERMAL_CHANNEL_SET_CONTROL         (0x2080053dU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | 0x3D" */


/*!
 * @brief Maximum size of the cirlwlar buffer of samples to keep a history.
 */
#define LW2080_CTRL_THERM_PMUMON_THERM_CHANNEL_SAMPLE_COUNT (50U)

/*!
 * @brief A single sample of the thermal channels at a particular point in time.
 */
typedef struct LW2080_CTRL_THERM_PMUMON_THERM_CHANNEL_SAMPLE {
    /*!
     * @brief Ptimer timestamp of when this data was collected.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_PMUMON_SAMPLE super, 8);

    /*!
     * @brief Output parameter: the current channel temperatures.
     */
    LwTemp temperature[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_THERM_PMUMON_THERM_CHANNEL_SAMPLE;

/*!
 * @brief Input/output parameters for
 * @ref LW2080_CTRL_CMD_THERMAL_PMUMON_THERM_CHANNEL_SAMPLES.
 */
#define LW2080_CTRL_THERMAL_PMUMON_THERM_CHANNEL_GET_SAMPLES_PARAMS_MESSAGE_ID (0x39U)

typedef struct LW2080_CTRL_THERMAL_PMUMON_THERM_CHANNEL_GET_SAMPLES_PARAMS {
    /*!
     * @brief Metadata for the samples.
     */
    LW2080_CTRL_PMUMON_GET_SAMPLES_SUPER super;

    /*!
     * @brief A collection of data samples for thermal data.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_THERM_PMUMON_THERM_CHANNEL_SAMPLE samples[LW2080_CTRL_THERM_PMUMON_THERM_CHANNEL_SAMPLE_COUNT], 8);
} LW2080_CTRL_THERMAL_PMUMON_THERM_CHANNEL_GET_SAMPLES_PARAMS;
typedef struct LW2080_CTRL_THERMAL_PMUMON_THERM_CHANNEL_GET_SAMPLES_PARAMS *PLW2080_CTRL_THERMAL_PMUMON_THERM_CHANNEL_GET_SAMPLES_PARAMS;

#define LW2080_CTRL_CMD_THERMAL_PMUMON_THERM_CHANNEL_SAMPLES (0x20800539U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | LW2080_CTRL_THERMAL_PMUMON_THERM_CHANNEL_GET_SAMPLES_PARAMS_MESSAGE_ID" */

/* ------------------ FAN_ARBITER's defines and structures ----------------- */

/*!
 * Macros for Fan Arbiter Types.
 */
#define LW2080_CTRL_FAN_ARBITER_TYPE_ILWALID                 0x00U
#define LW2080_CTRL_FAN_ARBITER_TYPE_V10                     0x01U

/*!
 * Macros for Fan Arbiter Computation Modes.
 */
#define LW2080_CTRL_FAN_ARBITER_MODE_ILWALID                 0x00U
#define LW2080_CTRL_FAN_ARBITER_MODE_MAX                     0x01U

/*!
 * Special value corresponding to an invalid Fan Arbiter Index.
 */
#define LW2080_CTRL_FAN_ARBITER_INDEX_ILWALID                LW2080_CTRL_BOARDOBJ_IDX_ILWALID_8BIT

/*!
 * Maximum number of FAN_ARBITERS which can be supported in the RM or PMU.
 */
#define LW2080_CTRL_FAN_ARBITER_MAX_ARBITERS                 16U

/* -------------- End of FAN_ARBITER's defines and structures -------------- */

/*!
 * Maximum number of FAN_TEST type supported.
 */
#define LW2080_CTRL_FAN_TEST_MAX_TEST                        32U

/*!
 * Macros for Fan Test Types
 */
#define LW2080_CTRL_FAN_TEST_ILWALID                         0x00U
#define LW2080_CTRL_FAN_TEST_COOLER_SANITY                   0x01U

/*!
 * Structure of static information specific to COOLER_SANITY test type.
 */
typedef struct LW2080_CTRL_FAN_TEST_INFO_DATA_COOLER_SANITY {
    /*!
     * Fan Cooler table index.
     */
    LwU8  coolerTableIdx;

    /*!
     * Error tolerance in percentage.
     */
    LwU8  measurementTolerancePct;

    /*!
     * Colwergence Time in millisecond.
     */
    LwU16 colwergenceTimems;
} LW2080_CTRL_FAN_TEST_INFO_DATA_COOLER_SANITY;

/*!
 * Union of FAN_TEST type.
 */


/*!
 * Structure representing the static information associated with a FAN_TEST.
 */
typedef struct LW2080_CTRL_FAN_TEST_INFO {
    /*!
     * @ref LW2080_CTRL_FAN_TEST_<xyz>.
     */
    LwU8 type;

    /*!
     * type-specific information.
     */
    union {
        LW2080_CTRL_FAN_TEST_INFO_DATA_COOLER_SANITY coolerSanity;
    } data;
} LW2080_CTRL_FAN_TEST_INFO;

/*!
 * LW2080_CTRL_CMD_FAN_TEST_GET_INFO
 *
 * This command returns the FAN_TEST static information as specified by
 * the Fan Test Table.
 *
 * https://wiki.lwpu.com/engwiki/index.php/Resman/Thermal_Control/Fan_Tables(s)_1.0_Specification#Fan_Test_Table
 *
 * See LW2080_CTRL_FAN_TEST_INFO_PARAMS for documentation on the
 * parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_FAN_TEST_GET_INFO (0x2080053eU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | LW2080_CTRL_FAN_TEST_INFO_PARAMS_MESSAGE_ID" */

/*!
 * Structure representing the static state information associated with the
 * FAN_TEST.
 */
#define LW2080_CTRL_FAN_TEST_INFO_PARAMS_MESSAGE_ID (0x3EU)

typedef struct LW2080_CTRL_FAN_TEST_INFO_PARAMS {
    /*!
     * [out] - Mask of FAN_TEST entries.
     */
    LwU32                     testMask;

    /*!
     * [out] - Array of FAN_TEST entries. Has valid indexes corresponding to
     * the bits set in @ref testMask.
     */
    LW2080_CTRL_FAN_TEST_INFO test[LW2080_CTRL_FAN_TEST_MAX_TEST];
} LW2080_CTRL_FAN_TEST_INFO_PARAMS;

/*!
 * LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_<xyz>
 *
 * This defines describe all HW failsafe events supported by the
 * LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_SETTINGS_GET &
 * LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_SETTINGS_SET commands.
 */
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_EXT_OVERT       0x00U
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_EXT_ALERT       0x01U
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_EXT_POWER       0x02U
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_OVERT           0x03U
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_ALERT_0H        0x04U
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_ALERT_1H        0x05U
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_ALERT_2H        0x06U
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_ALERT_3H        0x07U
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_ALERT_4H        0x08U
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_ALERT_NEG1H     0x09U

// Only supported Pascal and onwards.
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_THERMAL_0       0x0AU
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_THERMAL_1       0x0BU
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_THERMAL_2       0x0LW
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_THERMAL_3       0x0DU
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_THERMAL_4       0x0EU
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_THERMAL_5       0x0FU
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_THERMAL_6       0x10U
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_THERMAL_7       0x11U
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_THERMAL_8       0x12U
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_THERMAL_9       0x13U
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_THERMAL_10      0x14U
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_THERMAL_11      0x15U
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_DEDICATED_OVERT 0x16U

// Only supported Volta and onwards.
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_SCI_FS_OVERT    0x17U

// EXT_ALERT_0 and EXT_ALERT_1 only supported Turing and onwards.
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_EXT_ALERT_0     0x18U
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_EXT_ALERT_1     0x19U
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_VOLTAGE_HW_ADC  0x1AU
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_EDPP_VMIN       0x1BU
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_EDPP_FONLY      0x1LW
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_BA_BA_W2_T1H    0x1DU

#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID__COUNT          0x1EU

/*!
 * Value used to specify that HWFS event does not use temperature threshold.
 */
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_TEMP_ILWALID       0x7FFFFFFFU

/*!
 * Mask value used to initialize a thermal event mask.
 */
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_MASK_NONE          0x00000000U

/*!
 * LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_SETTINGS_GET
 *
 * This command returns the settings (slowdown, temp. threshold) of the
 * requested HW failsafe event.
 *
 * See LW2080_CTRL_THERMAL_HWFS_EVENT_SETTINGS_PARAMS for the documentation of
 * the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_SETTINGS_GET       (0x20800540U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | 0x40" */

/*!
 * LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_SETTINGS_SET
 *
 * This command applies provided settings (slowdown, temp. threshold) to the
 * requested HW failsafe event.
 *
 * Note: This command applies all settings atomically. To alter single parameter
 *      at a time a LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_SETTINGS_GET() call must
 *      be ilwoked first to obtain current settings (before changing them).
 *
 * See LW2080_CTRL_THERMAL_HWFS_EVENT_SETTINGS_PARAMS for the documentation of
 * the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_SETTINGS_SET       (0x20800541U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | 0x41" */

/*!
 * Structure representing the settings of the HW failsafe event.
 */
typedef struct LW2080_CTRL_THERMAL_HWFS_EVENT_SETTINGS_PARAMS {
    /*!
     * [in] - HWFS event's ID as LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_<xyz>
     */
    LwU8                                eventId;

    /*!
     * [in/out] - HWFS event's temperatrue threshold (if applicable) in 1/256[C]
     */
    LwTemp                              temperature;

    /*!
     * [in] - Bool to include hotspot offset in HWFS event's reported
     *        temperature threshold.
     */
    LwBool                              bIncludeHotspotOffset;

    /*!
     * [in/out] - HWFS event's slowdown specified as a fraction (num/denom)
     */
    LW2080_CTRL_THERMAL_SLOWDOWN_AMOUNT slowdown;

    /*!
     * [out] - sensorId corresponding to the HWFS event.
     */
    LwU8                                sensorId;
} LW2080_CTRL_THERMAL_HWFS_EVENT_SETTINGS_PARAMS;

/*!
 * LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_STATUS_GET
 *
 * This command gets HW failsafe events status. Status includes all run-time
 * data related to events. For parameter dolwmentations see
 * @ref LW2080_CTRL_THERMAL_HWFS_EVENT_STATUS_GET_PARAMS.
 *
 * Clients can determine the HW failsafe events that have caused slowdown.
 */
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_STATUS_GET (0x20800542U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | LW2080_CTRL_THERMAL_HWFS_EVENT_STATUS_GET_PARAMS_MESSAGE_ID" */

/*!
 * Structure representing one HW failsafe THERMAL event status.
 */
typedef struct LW2080_CTRL_THERMAL_HWFS_EVENT_STATUS {
    /*!
     * [out] - Total number of times event was asserted.
     */
    LW_DECLARE_ALIGNED(LwU64 violCount, 8);

    /*!
     * [out] - Total duration for which the event was asserted in ns.
     */
    LW_DECLARE_ALIGNED(LwU64 violTimens, 8);
} LW2080_CTRL_THERMAL_HWFS_EVENT_STATUS;

/*!
 * Structure containing HW failsafe thermal events status
 */
#define LW2080_CTRL_THERMAL_HWFS_EVENT_STATUS_GET_PARAMS_MESSAGE_ID (0x42U)

typedef struct LW2080_CTRL_THERMAL_HWFS_EVENT_STATUS_GET_PARAMS {
    /*!
     * [out] - Mask of LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID_<xyz> returned.
     */
    LwU32 eventMask;
    /*!
     * [out] - Each event's status.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_THERMAL_HWFS_EVENT_STATUS events[LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_ID__COUNT], 8);
} LW2080_CTRL_THERMAL_HWFS_EVENT_STATUS_GET_PARAMS;

/* ---------- THERM_MONITOR's GET_INFO RMCTRL defines and structures ---------- */

/*!
 * Structure representing the static information associated with a THERM_MONITOR
 */
typedef struct LW2080_CTRL_THERM_THERM_MONITOR_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;

    /*!
     * Index into physical instance of Thermal Interrupt Monitor
     */
    LwU8                 phyInstIdx;
} LW2080_CTRL_THERM_THERM_MONITOR_INFO;
typedef struct LW2080_CTRL_THERM_THERM_MONITOR_INFO *PLW2080_CTRL_THERM_THERM_MONITOR_INFO;

/*!
 * Structure representing the static state information associated with the GPU's
 * THERM_MONITOR functionality.
 */
#define LW2080_CTRL_THERM_THERM_MONITORS_INFO_PARAMS_MESSAGE_ID (0x43U)

typedef struct LW2080_CTRL_THERM_THERM_MONITORS_INFO_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32          super;

    /*!
     * Utils clock frequency in Khz. This is used in PMU for callwlating
     * monitor high(low) time.
     */
    LwU32                                utilsClkFreqKhz;

    /*!
     * [out] - Array of THERM_MONITOR entries. Has valid indexes corresponding to
     * the bits set in @ref monMask.
     */
    LW2080_CTRL_THERM_THERM_MONITOR_INFO monitors[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS];
} LW2080_CTRL_THERM_THERM_MONITORS_INFO_PARAMS;
typedef struct LW2080_CTRL_THERM_THERM_MONITORS_INFO_PARAMS *PLW2080_CTRL_THERM_THERM_MONITORS_INFO_PARAMS;

/*!
 * LW2080_CTRL_CMD_THERM_THERM_MONITORS_GET_INFO
 *
 * This command returns the THERM_MONITOR static information as specified by the
 * Thermal Monitor Table.
 *
 * See LW2080_CTRL_THERM_THERM_MONITORS_INFO_PARAMS for documentation on the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_THERMAL_MONITORS_GET_INFO (0x20800543U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | LW2080_CTRL_THERM_THERM_MONITORS_INFO_PARAMS_MESSAGE_ID" */

/* --------- THERM_MONITOR's GET_STATUS RMCTRL defines and structures --------- */

/*!
 * Structure representing the dynamic state associated with a THERM_MONITOR.
 */
typedef struct LW2080_CTRL_THERM_THERM_MONITOR_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class.  Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJ super;

    /*!
     * SW abstraction of 64 bit counter as we have 32 bit counters lwrrently
     * which can overflow.
     */
    LW_DECLARE_ALIGNED(LwU64 counter, 8);

    /*!
     * Engaged time in ns, overflows normally when it reaches LwU64_MAX
     */
    LW_DECLARE_ALIGNED(LwU64 engagedTimens, 8);
} LW2080_CTRL_THERM_THERM_MONITOR_STATUS;
typedef struct LW2080_CTRL_THERM_THERM_MONITOR_STATUS *PLW2080_CTRL_THERM_THERM_MONITOR_STATUS;

/*!
 * Structure representing the dynamic state associated with the GPU's
 * THERM_MONITOR functionality.
 */
#define LW2080_CTRL_THERM_THERM_MONITORS_STATUS_PARAMS_MESSAGE_ID (0x44U)

typedef struct LW2080_CTRL_THERM_THERM_MONITORS_STATUS_PARAMS {
    /*!
     * LW2080_CTRL_BOARDOBJGRP super class. Must always be first object in
     * structure.
     */
    LW2080_CTRL_BOARDOBJGRP_E32 super;

    /*!
     * [out] - Array of THERM_MONITOR entries.
     */
    LW_DECLARE_ALIGNED(LW2080_CTRL_THERM_THERM_MONITOR_STATUS monitors[LW2080_CTRL_BOARDOBJGRP_E32_MAX_OBJECTS], 8);
} LW2080_CTRL_THERM_THERM_MONITORS_STATUS_PARAMS;
typedef struct LW2080_CTRL_THERM_THERM_MONITORS_STATUS_PARAMS *PLW2080_CTRL_THERM_THERM_MONITORS_STATUS_PARAMS;

/*!
 * LW2080_CTRL_CMD_THERM_THERM_MONITORS_GET_STATUS
 *
 * This command returns the THERM_MONITOR dynamic state information associated by the
 * THERM_MONITOR functionality
 *
 * See LW2080_CTRL_THERM_THERM_MONITORS_STATUS_PARAMS for documentation on the parameters.
 *
 * Possible status values returned are
 *   LWOS_STATUS_SUCCESS
 *   LWOS_STATUS_ERROR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_THERMAL_MONITORS_GET_STATUS (0x20800544U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | LW2080_CTRL_THERM_THERM_MONITORS_STATUS_PARAMS_MESSAGE_ID" */

/*!
 * Structure containing current thermal slowdown parameters
 * Slowdown[%] of an input clock domain = 100 * numerator / denominator.
 */
#define LW2080_CTRL_CMD_THERMAL_HWFS_SLOWDOWN_AMOUNT_GET_PARAMS_MESSAGE_ID (0x45U)

typedef struct LW2080_CTRL_CMD_THERMAL_HWFS_SLOWDOWN_AMOUNT_GET_PARAMS {
    /*!
     * [in] - Clock domain LW2080_CTRL_CLK_DOMAIN_*.
     */
    LwU32 clkDomain;
    /*!
     * [out] - Numerator of resulting slowdown fraction.
     */
    LwU32 numerator;
    /*!
     * [out] - Denominator of resulting slowdown fraction.
     */
    LwU32 denominator;
} LW2080_CTRL_CMD_THERMAL_HWFS_SLOWDOWN_AMOUNT_GET_PARAMS;
typedef struct LW2080_CTRL_CMD_THERMAL_HWFS_SLOWDOWN_AMOUNT_GET_PARAMS *PLW2080_CTRL_CMD_THERMAL_HWFS_SLOWDOWN_AMOUNT_GET_PARAMS;

/*!
 * LW2080_CTRL_CMD_THERMAL_HWFS_SLOWDOWN_AMOUNT_GET
 *
 * This command gets slowdown amount as a numerator and log2denominator
 * for the input clock domain. For parameter documentation see
 * @ref LW2080_CTRL_CMD_THERMAL_HWFS_SLOWDOWN_AMOUNT_GET_PARAMS.
 *
 */
#define LW2080_CTRL_CMD_THERMAL_HWFS_SLOWDOWN_AMOUNT_GET          (0x20800545U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | LW2080_CTRL_CMD_THERMAL_HWFS_SLOWDOWN_AMOUNT_GET_PARAMS_MESSAGE_ID" */

/*!
 * LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_REPORTING_SETTINGS_GET
 *
 * This command returns the settings (slowdown, temp. threshold) of the
 * requested HW failsafe event for external reporting purposes.
 * This RMCTRL returns identical information as
 * @ref LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_SETTINGS_GET for non-TSOSC sensors.
 * For TSOSC, hotspot offset is dynamic so an alternative mechanism where
 * worst case Tmax_gpu to Tavg_gpu delta is used for reporting purpose.
 *
 * See LW2080_CTRL_THERMAL_HWFS_EVENT_SETTINGS_PARAMS for the documentation of
 * the parameters.
 *
 * Possible status values returned are
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_ILWALID_STATE
 */
#define LW2080_CTRL_CMD_THERMAL_HWFS_EVENT_REPORTING_SETTINGS_GET (0x20800546U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | 0x46" */

/* ---------- FAN_ARBITER's GET_INFO RMCTRL defines and structures ---------- */

/*!
 * Structure representing the static information associated with a FAN_ARBITER.
 */
typedef struct LW2080_CTRL_FAN_ARBITER_INFO {
    /*!
     * LW2080_CTRL_BOARDOBJ super class. This should always be the first member!
     */
    LW2080_CTRL_BOARDOBJ             super;

    /*!
     * Computation mode.
     */
    LwU8                             mode;

    /*!
     * Index into the Fan Cooler Table.
     */
    LwU8                             coolerIdx;

    /*!
     * Sampling period.
     */
    LwU16                            samplingPeriodms;

    /*!
     * Mask of FAN_POLICIES that RM/PMU will use.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32 fanPoliciesMask;

    /*!
     * Mask of FAN_POLICIES that VBIOS will use.
     */
    LW2080_CTRL_BOARDOBJGRP_MASK_E32 vbiosFanPoliciesMask;
} LW2080_CTRL_FAN_ARBITER_INFO;

/*!
 * Structure representing the static state information associated with the GPU's
 * FAN_ARBITER functionality.
 */
#define LW2080_CTRL_FAN_ARBITER_INFO_PARAMS_MESSAGE_ID (0x47U)

typedef struct LW2080_CTRL_FAN_ARBITER_INFO_PARAMS {
    /*!
     * [out] - BOARDOBJGRP super class. This should always be the first member!
     */
    LW2080_CTRL_BOARDOBJGRP_E32  super;

    /*!
     * [out] - Array of FAN_ARBITER entries. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW2080_CTRL_FAN_ARBITER_INFO arbiters[LW2080_CTRL_FAN_ARBITER_MAX_ARBITERS];
} LW2080_CTRL_FAN_ARBITER_INFO_PARAMS;

/*!
 * LW2080_CTRL_CMD_FAN_ARBITER_GET_INFO
 *
 * This command returns the FAN_ARBITER static information as specified by the
 * Fan Arbiter Table.
 *
 * https://confluence.lwpu.com/display/BS/Fan+Arbiter+Table
 *
 * See @ref LW2080_CTRL_FAN_ARBITER_INFO_PARAMS for documentation on the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetInfo.
 */
#define LW2080_CTRL_CMD_FAN_ARBITER_GET_INFO (0x20800547U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | LW2080_CTRL_FAN_ARBITER_INFO_PARAMS_MESSAGE_ID" */

/* ------- End of FAN_ARBITER's GET_INFO RMCTRL defines and structures ------ */

/* -------- FAN_ARBITER's GET_STATUS RMCTRL defines and structures ---------- */

/*!
 * Structure representing the dynamic state associated with a FAN_ARBITER.
 */
typedef struct LW2080_CTRL_FAN_ARBITER_STATUS {
    /*!
     * LW2080_CTRL_BOARDOBJ super class. This should always be the first member!
     */
    LW2080_CTRL_BOARDOBJ super;

    /*!
     * Denotes the control action to be taken on the fan described via
     * @ref LW2080_CTRL_FAN_CTRL_ACTION_<xyz>.
     */
    LwU8                 fanCtrlAction;

    /*!
     * Denotes the policy that is lwrrently driving the fan.
     */
    LwBoardObjIdx        drivingPolicyIdx;

    /*!
     * Target PWM (in percents), used when @ref fanPoliciesControlUnit is
     * LW2080_CTRL_FAN_COOLER_ACTIVE_CONTROL_UNIT_PWM.
     */
    LwUFXP16_16          targetPwm;

    /*!
     * Target RPM, used when @ref fanPoliciesControlUnit is
     * LW2080_CTRL_FAN_COOLER_ACTIVE_CONTROL_UNIT_RPM.
     */
    LwU32                targetRpm;
} LW2080_CTRL_FAN_ARBITER_STATUS;

/*!
 * Structure representing the dynamic state associated with the GPU's
 * FAN_ARBITER functionality.
 */
#define LW2080_CTRL_FAN_ARBITER_STATUS_PARAMS_MESSAGE_ID (0x48U)

typedef struct LW2080_CTRL_FAN_ARBITER_STATUS_PARAMS {
    /*!
     * [out] - BOARDOBJGRP super class. This should always be the first member!
     */
    LW2080_CTRL_BOARDOBJGRP_E32    super;

    /*!
     * [out] - Array of FAN_ARBITER entries. Has valid indexes corresponding to
     * the bits set in @ref super.objMask.
     */
    LW2080_CTRL_FAN_ARBITER_STATUS arbiters[LW2080_CTRL_FAN_ARBITER_MAX_ARBITERS];
} LW2080_CTRL_FAN_ARBITER_STATUS_PARAMS;

/*!
 * LW2080_CTRL_CMD_FAN_ARBITER_GET_STATUS
 *
 * This command returns the FAN_ARBITER dynamic state as specified by the
 * Fan Arbiter Table.
 *
 * https://confluence.lwpu.com/display/BS/Fan+Arbiter+Table
 *
 * See @ref LW2080_CTRL_FAN_ARBITER_STATUS_PARAMS for documentation
 * on the parameters.
 *
 * Return values are specified per @ref BoardObjGrpGetStatus.
 */
#define LW2080_CTRL_CMD_FAN_ARBITER_GET_STATUS (0x20800548U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_THERMAL_INTERFACE_ID << 8) | LW2080_CTRL_FAN_ARBITER_STATUS_PARAMS_MESSAGE_ID" */

/* ----- End of FAN_ARBITER's GET_STATUS RMCTRL defines and structures ------ */

/* _ctrl2080thermal_h_ */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

