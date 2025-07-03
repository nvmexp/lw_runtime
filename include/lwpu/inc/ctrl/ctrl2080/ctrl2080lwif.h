/* 
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2007-2015 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl2080/ctrl2080lwif.finn
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

/* LW20_SUBDEVICE_XX LWIF control commands and parameters */

/*
 * LW2080_CTRL_CMD_LWIF_EXELWTE_METHOD
 *
 * This command is used to execute LWIF methods on the associated subdevice.
 * LWIF methods are used to access the LWPU ACPI Display Extensions
 * provided by a compliant ACPI SBIOS.  LWIF method data includes a
 * function/subfunction pair indicating the method to ilwoke and a
 * corresponding set of input and output parameters.
 *
 *   function
 *     This parameter specifies the desired LWIF method function to execute.
 *     Possible valid values for this parameter include:
 *       LW2080_CTRL_LWIF_FUNC_VERSION
 *         This function can be used to establish LWIF compatibility between
 *         software and the SBIOS.
 *       LW2080_CTRL_LWIF_FUNC_LID_STATUS
 *         This function can be used to access current lid status information.
 *       LW2080_CTRL_LWIF_FUNC_DOCK_STATUS
 *         This function can be used to access current docking status
 *         information.
 *       LW2080_CTRL_LWIF_FUNC_THERMAL_MONITOR
 *         This function can be used to access thermal monitor features.
 *       LW2080_CTRL_LWIF_FUNC_BRIGHTNESS_CONTROL
 *         This function can be used to access platform-dependent brightness
 *         control features.
 *       LW2080_CTRL_LWIF_FUNC_POWERMIZER_LIMIT
 *         This function can be used to retrieve PowerMizer level information.
 *       LW2080_CTRL_LWIF_FUNC_DISPLAY_ATTRIBUTES
 *         This function can be used to access display attribute features.
 *       LW2080_CTRL_LWIF_FUNC_HDCP
 *         This function can be used to access HDCP features.
 *       LW2080_CTRL_LWIF_FUNC_PLATCFG
 *         This function can be used to access platform configuration features.
 *       LW2080_CTRL_LWIF_FUNC_TEST
 *         This function can be used to test the LWIF method interface.
 *   subFunction
 *     This parameter specifies the LWIF function-specific subfunction value.
 *     Possible valid values for this parameter include:
 *
 *       LW2080_CTRL_LWIF_SUBFUNC_VERSION_GET
 *         This subfunction can be used with the VERSION function to establish
 *         compatibility between software and the SBIOS.  The LWIF version
 *         number supported by the SBIOS is returned in the outData parameter.
 *
 *       LW2080_CTRL_LWIF_SUBFUNC_LID_STATUS_SUPPORTED
 *         This subfunction can be used with the LID_STATUS function to
 *         determine if lid status operations are supported.  An outStatus
 *         value of LW2080_CTRL_LWIF_STATUS_SUCCESS indicates that the
 *         LID_STATUS function is supported.  An outStatus value of
 *         LW2080_CTRL_LWIF_STATUS_UNSUPPORTED indicates that the LID_STATUS
 *         function is not supported.
 *       LW2080_CTRL_LWIF_SUBFUNC_LID_STATUS_GET
 *         This subfunction can be used with the LID_STATUS function.  See the
 *         description of LW2080_CTRL_LWIF_SUBFUNC_LID_STATUS_GET_OUTPARAMS
 *         for more details.
 *
 *       LW2080_CTRL_LWIF_SUBFUNC_DOCK_STATUS_SUPPORTED
 *         This subfunction can be used with the DOCK_STATUS function to
 *         determine if dock status operations are supported.  An outStatus
 *         value of LW2080_CTRL_LWIF_STATUS_SUCCESS indicates that the
 *         DOCK_STATUS function is supported.  An outStatus value of
 *         LW2080_CTRL_LWIF_STATUS_UNSUPPORTED indicates that the DOCK_STATUS
 *         function is not supported.
 *       LW2080_CTRL_LWIF_SUBFUNC_DOCK_STATUS_GET
 *         This subfunction can be used with the DOCK_STATUS function.  See the
 *         description of LW2080_CTRL_LWIF_SUBFUNC_DOCK_STATUS_GET_OUTPARAMS
 *         for more details.
 *
 *       LW2080_CTRL_LWIF_SUBFUNC_THERMAL_MONITOR_SUPPORTED
 *         This subfunction can be used with the THERMAL_MONITOR function to
 *         determine if thermal monitor operations are supported.  An outStatus
 *         value of LW2080_CTRL_LWIF_STATUS_SUCCESS indicates that the
 *         THERMAL_MONITOR function is supported.  An outStatus value of
 *         LW2080_CTRL_LWIF_STATUS_UNSUPPORTED indicates that the
 *         THERMAL_MONITOR function is not supported.
 *       LW2080_CTRL_LWIF_SUBFUNC_THERMAL_MONITOR_GET
 *         This subfunction can be used with the THERMAL_MONITOR function to
 *         get the thermal monitor state of the GPU.  See the description of
 *         LW2080_CTRL_LWIF_SUBFUNC_THERMAL_MONITOR_GET_OUTPARAMS for more
 *         details.
 *       LW2080_CTRL_LWIF_SUBFUNC_THERMAL_MONITOR_SET_TEMP
 *         This subfunction can be used with the THERMAL_MONITOR function
 *         to notify the SBIOS of the current GPU temperature.
 *         See the description of
 *         LW2080_CTRL_LWIF_SUBFUNC_THERMAL_MONITOR_SET_TEMP_INPARAMS
 *         for more details.
 *         The frequency at which this call should be sent to the SBIOS
 *         depends on the current thermal monitor temperature resolution.
 *       LW2080_CTRL_LWIF_SUBFUNC_THERMAL_MONITOR_SET
 *         This subfunction can be used with the THERMAL_MONITOR function
 *         to set the thermal monitor parameters in the SBIOS.  See the
 *         description of LW2080_CTRL_LWIF_SUBFUNC_THERMAL_MONITOR_SET_INPARAMS
 *         for more details.
 *
 *       LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_SUPPORTED
 *         This subfunction can be used with the BRIGHTNESS_CONTROL function to
 *         determine if brightness control operations are supported.  An
 *         outStatus value of LW2080_CTRL_LWIF_STATUS_SUCCESS indicates that
 *         the THERMAL_MONITOR function is supported.  An outStatus value of
 *         LW2080_CTRL_LWIF_STATUS_UNSUPPORTED indicates that the
 *         THERMAL_MONITOR function is not supported.
 *       LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_GET
 *         This subfunction can be used with the BRIGHTNESS_CONTROL function
 *         to query brightness control hardware initialization information.
 *         See the descriptions of
 *         LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTRL_GET_INPARAMS and
 *         LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTRL_GET_OUTPARAMS for more
 *         details.
 *       LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_SET_LEVEL
 *         This subfunction can be used with the BRIGHTNESS_CONTROL function to
 *         set the brightness level in the SBIOS.  See the description of
 *         LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTRL_SET_LEVEL_INPARAMS for
 *         more details.
 *       LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_GET_LEVELS
 *         This subfunction can be used with the BRIGHTNESS_CONTROL function to
 *         query the list of supported brightness control levels.
 *         See the description of
 *         LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTRL_GET_LEVEL_INPARAMS and
 *         LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTRL_GET_LEVELS_OUTPARAMS for
 *         more details.
 *       LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_SET_MONITOR
 *         This subfunction can be used with the BRIGHTNESS_CONTROL function to
 *         notify the SBIOS of a monitor state change.  See the description of
 *         LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTRL_SET_MONITOR_INPARAMS
 *         for more details.
 *
 *       LW2080_CTRL_LWIF_SUBFUNC_POWERMIZER_LIMIT_SUPPORTED
 *         This subfunction can be used with the POWERMIZER_LIMIT function to
 *         determine if PowerMizer limit operations are supported.  An
 *         outStatus value of LW2080_CTRL_LWIF_STATUS_SUCCESS indicates that
 *         the POWERMIZER_LIMIT function is supported.  An outStatus value of
 *         LW2080_CTRL_LWIF_STATUS_UNSUPPORTED indicates that the
 *         POWERMIZER_LIMIT function is not supported.
 *       LW2080_CTRL_LWIF_SUBFUNC_POWERMIZER_LIMIT_GET
 *         This subfunction can be used with the POWERMIZER_LIMIT function to
 *         get the current PowerMizer level limit information from the SBIOS.
 *         See the description of
 *         LW2080_LWIF_SUBFUNC_POWERMIZER_LIMIT_GET_OUTPARAMS for more details.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SUPPORTED
 *         This subfunction can be used with the DISPLAY_ATTRIBUTES function to
 *         determine if display attribute operations are supported.  An
 *         outStatus value of LW2080_CTRL_LWIF_STATUS_SUCCESS indicates that
 *         the DISPLAY_ATTRIBUTES function is supported.  An outStatus value of
 *         LW2080_CTRL_LWIF_STATUS_UNSUPPORTED indicates that the
 *         DISPLAY_ATTRIBUTES function is not supported.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET
 *         This subfunction can be used with the DISPLAY_ATTRIBUTES function to
 *         query the SBIOS for display device attribute preferences.
 *         See the description of
 *         LW2080_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_OUTPARAMS for more
 *         details.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET
 *         This subfunction can be used with the DISPLAY_ATTRIBUTES function to
 *         notify the SBIOS of display device attribute updates by software.
 *         See the description of
 *         LW2080_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_INPARAMS for more
 *         details.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_ALT
 *         This subfunction can be used with the DISPLAY_ATTRIBUTES function to
 *         notify the SBIOS of explicit display device attribute updates by
 *         software.  See the description of
 *         LW2080_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_INPARAMS for more
 *         details.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_HOTKEY
 *         This subfunction notifies SBIOS the status of all display devices. 
 *         Format is same as LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET 
 *         but it will be called only during hot key display switch and if
 *         SBIOS has implemented LWIF based hot key display switching. 
 *         
 *       LW2080_CTRL_LWIF_SUBFUNC_HDCP_SUPPORTED
 *         This subfunction can be used with the HDCP function to
 *         determine if HDCP operations are supported.  An outStatus
 *         value of LW2080_CTRL_LWIF_STATUS_SUCCESS indicates that the
 *         HDCP function is supported.  An outStatus value of
 *         LW2080_CTRL_LWIF_STATUS_UNSUPPORTED indicates that the HDCP
 *         function is not supported.
 *
 *       LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_SUPPORTED
 *         This subfunction can be used with the PLATCFG function to
 *         determine if platform configuration operations are supported.
 *         An outStatus value of LW2080_CTRL_LWIF_STATUS_SUCCESS indicates
 *         that the PLATCFG function is supported.  An outStatus value of
 *         LW2080_CTRL_LWIF_STATUS_UNSUPPORTED indicates that the PLATCFG
 *         function is not supported.
 *       LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_GET_MEM_BW
 *         This subfunction can be used with the PLATCFG function to
 *         get system memory bandwidth and latency information from the
 *         ACPI SBIOS.  See the description of
 *         LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_GET_MEM_BW_OUTPARAMS for more
 *         details.
 *       LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_GET_SVM_SETTINGS
 *         This subfunction can be used with the PLATCFG function to
 *         get system video memory settings information from the
 *         ACPI SBIOS.  See the description of
 *         LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_GET_SVM_SETTINGS_OUTPARAMS for more
 *         details.
 *
 *   inParams
 *     This parameter specifies the LWIF method-specific arguments.
 *   inParamSize
 *     This parameter specifies the size in bytes of the LWIF method-specific
 *     arguments contained in the inParams parameter.
 *   outStatus
 *     This parameter returns the LWIF method status.  Possible valid values
 *     for this parameter include:
 *       LW2080_CTRL_LWIF_STATUS_SUCCESS
 *         This value indicates the method succeeded.
 *       LW2080_CTRL_LWIF_STATUS_ERROR_UNSPECIFIED
 *         This value indicates the method failed for an unspecified reason.
 *       LW2080_CTRL_LWIF_STATUS_ERROR_UNSUPPORTED
 *         This value indicates the method is unsupported.
 *       LW2080_CTRL_LWIF_STATUS_ERROR_PLATFORM
 *         This value indicates the method failed with an operating
 *         system-specific error.  The outData parameter will contain the value
 *         of this operating system-specific error.  For example, an ACPI
 *         interface failure will be flagged with this condition and the
 *         corresponding failure status returned in outData.
 *   outData
 *         This parameter returns the LWIF method-specific results.
 *   outDataSize
 *         This parameter returns the size in bytes of the method-specific
 *         results returned in the outData parameter. 
 * 
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW2080_CTRL_CMD_LWIF_EXELWTE_METHOD (0x20802101) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_LWIF_INTERFACE_ID << 8) | LW2080_CTRL_LWIF_EXELWTE_METHOD_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_LWIF_EXELWTE_METHOD_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_LWIF_EXELWTE_METHOD_PARAMS {
    LwU32 function;
    LwU32 subFunction;
    LW_DECLARE_ALIGNED(LwP64 inParams, 8);
    LwU32 inParamSize;
    LwU32 outStatus;
    LW_DECLARE_ALIGNED(LwP64 outData, 8);
    LwU32 outDataSize;
} LW2080_CTRL_LWIF_EXELWTE_METHOD_PARAMS;

/* valid function parameter values */
#define LW2080_CTRL_LWIF_FUNC_VERSION                           (0x00000001)
#define LW2080_CTRL_LWIF_FUNC_LID_STATUS                        (0x00000003)
#define LW2080_CTRL_LWIF_FUNC_DOCKING_STATUS                    (0x00000005)
#define LW2080_CTRL_LWIF_FUNC_THERMAL_MONITOR                   (0x00000008)
#define LW2080_CTRL_LWIF_FUNC_BRIGHTNESS_CONTROL                (0x00000009)
#define LW2080_CTRL_LWIF_FUNC_POWERMIZER_LIMIT                  (0x0000000A)
#define LW2080_CTRL_LWIF_FUNC_DISPLAY_ATTRIBUTES                (0x0000000B)
#define LW2080_CTRL_LWIF_FUNC_HDCP                              (0x0000000C)
#define LW2080_CTRL_LWIF_FUNC_PLATCFG                           (0x0000000D)
#define LW2080_CTRL_LWIF_FUNC_TEST                              (0xAAAAEEEE)

/* LW2080_CTRL_LWIF_FUNC_VERSION subFunction parameter values */
#define LW2080_CTRL_LWIF_SUBFUNC_VERSION_GET                    (0x00000000)

/* LW2080_CTRL_LWIF_FUNC_LID_STATUS subFunction parameter values */
#define LW2080_CTRL_LWIF_SUBFUNC_LID_STATUS_SUPPORTED           (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_LID_STATUS_GET                 (0x00000001)

/* LW2080_CTRL_LWIF_FUNC_DOCKING_STATUS subFunction parameter values */
#define LW2080_CTRL_LWIF_SUBFUNC_DOCKING_STATUS_SUPPORTED       (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DOCKING_STATUS_GET             (0x00000001)

/* LW2080_CTRL_LWIF_FUNC_THERMAL_MONITOR subfunction parameter values */
#define LW2080_CTRL_LWIF_SUBFUNC_THERMAL_MONITOR_SUPPORTED      (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_THERMAL_MONITOR_GET            (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_THERMAL_MONITOR_SET_TEMP       (0x00000002)
#define LW2080_CTRL_LWIF_SUBFUNC_THERMAL_MONITOR_SET            (0x00000003)

/* LW2080_CTRL_LWIF_FUNC_BRIGHTNESS_CONTROL subfunction parameter values */
#define LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_SUPPORTED   (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_GET         (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_SET_LEVEL   (0x00000002)
#define LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_GET_LEVELS  (0x00000003)
#define LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_SET_MONITOR (0x00000004)

/* LW2080_CTRL_LWIF_FUNC_POWERMIZER_LIMIT_subfunction parameter values */
#define LW2080_CTRL_LWIF_SUBFUNC_POWERMIZER_LIMIT_SUPPORTED     (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_POWERMIZER_LIMIT_GET           (0x00000002)

/* LW2080_CTRL_LWIF_FUNC_DISPLAY_ATTRIBUTES subfunction parameter values */
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SUPPORTED   (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET         (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET         (0x00000002)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_ALT     (0x00000005)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_HOTKEY  (0x00000003)

/* LW2080_CTRL_LWIF_FUNC_HDCP subfunction parameter values */
#define LW2080_CTRL_LWIF_SUBFUNC_HDCP_SUPPORTED                 (0x00000000)

/* LW2080_CTRL_LWIF_FUNC_PLATCFG subfunction parameter values */
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_SUPPORTED              (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_GET_MEM_BW             (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP               (0x00000002)
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_GET_SVM_SETTINGS       (0x00000003)


/* valid outStatus parameter values */
#define LW2080_CTRL_LWIF_STATUS_SUCCESS                         (0x00000000)
#define LW2080_CTRL_LWIF_STATUS_ERROR_UNSPECIFIED               (0x80000001)
#define LW2080_CTRL_LWIF_STATUS_ERROR_UNSUPPORTED               (0x80000002)
#define LW2080_CTRL_LWIF_STATUS_ERROR_PLATFORM                  (0xC0000000)

/*
 * LW2080_CTRL_LWIF_SUBFUNC_VERSION_GET_INPARAMS
 *
 * This structure describes input parameters for the operation initiated
 * with the LW2080_CTRL_LWIF_SUBFUNC_VERSION_GET subfunction.
 *
 *   version
 *     This parameter specifies the LWIF version number supported by software.
 */
typedef struct LW2080_CTRL_LWIF_SUBFUNC_VERSION_GET_INPARAMS {
    LwU32 version;
} LW2080_CTRL_LWIF_SUBFUNC_VERSION_GET_INPARAMS;

/*
 * LW2080_CTRL_LWIF_SUBFUNC_VERSION_GET_OUTPARAMS
 *
 * This structure describes output parameters for the operation initiated
 * with the LW2080_CTRL_LWIF_SUBFUNC_VERSION_GET subfunction.
 *
 *  version
 *     This parameter returns the LWIF version number supported by the
 *     SBIOS.
 */
typedef struct LW2080_CTRL_LWIF_SUBFUNC_VERSION_GET_OUTPARAMS {
    LwU32 version;
} LW2080_CTRL_LWIF_SUBFUNC_VERSION_GET_OUTPARAMS;

/*
 * LW2080_CTRL_LWIF_SUBFUNC_LID_STATUS_GET_OUTPARAMS
 *
 * This structure describes output parameters for the operation initiated
 * with the LW2080_CTRL_LWIF_SUBFUNC_LID_STATUS_GET subfunction.
 *
 *  status
 *     This parameter returns the current lid status.  Legal values for
 *     this parameter include:
 *       LW2080_CTRL_LWIF_SUBFUNC_LID_STATUS_OPEN
 *         The lid is open.
 *       LW2080_CTRL_LWIF_SUBFUNC_LID_STATUS_CLOSED
 *         The lid is closed.
 */
typedef struct LW2080_CTRL_LWIF_SUBFUNC_LID_STATUS_GET_OUTPARAMS {
    LwU32 lidStatus;
} LW2080_CTRL_LWIF_SUBFUNC_LID_STATUS_GET_OUTPARAMS;

/* valid lidStatus status values */
#define LW2080_CTRL_LWIF_SUBFUNC_LID_STATUS_OPEN   (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_LID_STATUS_CLOSED (0x00000001)

/*
 * LW2080_CTRL_LWIF_SUBFUNC_DOCKING_STATUS_GET_OUTPARAMS
 *
 * This structure describes output parameters for the operation initiated
 * with the LW2080_CTRL_LWIF_SUBFUNC_DOCKING_STATUS_GET subfunction.
 *
 *  status
 *     This parameter returns the current lid status.  Legal values for
 *     this parameter include:
 *       LW2080_CTRL_LWIF_SUBFUNC_DOCKING_STATUS_UNDOCKED
 *         The system is undocked.
 *       LW2080_CTRL_LWIF_SUBFUNC_DOCKING_STATUS_DOCKED
 *         The system is docked.
 */
typedef struct LW2080_CTRL_LWIF_SUBFUNC_DOCKING_STATUS_GET_OUTPARAMS {
    LwU32 dockStatus;
} LW2080_CTRL_LWIF_SUBFUNC_DOCKING_STATUS_GET_OUTPARAMS;

/* valid dockStatus status values */
#define LW2080_CTRL_LWIF_SUBFUNC_DOCKING_STATUS_UNDOCKED (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DOCKING_STATUS_DOCKED   (0x00000001)

/*
 * LW2080_CTRL_LWIF_SUBFUNC_THERMAL_MONITOR_GET_OUTPARAMS
 *
 * This structure describes output parameters for the operation initiated
 * with the LW2080_CTRL_LWIF_SUBFUNC_THERMAL_MONITOR_GET subfunction.
 *
 *   tempThreshold
 *     This field contains the temperature threshold value in Celsius units
 *     in the form of a 16-bit signed integer.  Legal values for this field
 *     range from -100 to +300 degrees Celsius, inclusive.
 *   tempResolution
 *     This field contains the temperature resolution value in Celsius units
 *     in the form of a 16-bit signed integer.  Legal values for this field
 *     range from +5 to +300 degrees Celsius, inclusive.
 */
typedef struct LW2080_CTRL_LWIF_SUBFUNC_THERMAL_MONITOR_GET_OUTPARAMS {
    LwS16 tempThreshold;
    LwS16 tempResolution;
} LW2080_CTRL_LWIF_SUBFUNC_THERMAL_MONITOR_GET_OUTPARAMS;

/*
 * LW2080_CTRL_LWIF_SUBFUNC_THERMAL_MONITOR_SET_TEMP_INPARAMS
 *
 * This structure describes output parameters for the operation initiated
 * with the LW2080_CTRL_LWIF_SUBFUNC_THERMAL_MONITOR_SET_TEMP subfunction.
 *
 *   temperature
 *     This field specifies the current GPU temperature value in Celsius units
 *     in the form of a 32-bit signed integer.
 */
typedef struct LW2080_CTRL_LWIF_SUBFUNC_THERMAL_MONITOR_SET_TEMP_INPARAMS {
    LwS32 temperature;
} LW2080_CTRL_LWIF_SUBFUNC_THERMAL_MONITOR_SET_TEMP_INPARAMS;

/*
 * LW2080_CTRL_LWIF_SUBFUNC_THERMAL_MONITOR_SET_INPARAMS
 *
 * This structure describes input parameters for the operation initiated
 * with the LW2080_CTRL_LWIF_SUBFUNC_THERMAL_MONITOR_SET subfunction.
 *
 *   tempThreshold
 *     This field specifies the temperature threshold value in Celsius units
 *     in the form of a 16-bit signed integer.  Legal values for this field
 *     range from -100 to +300 degrees Celsius, inclusive.
 *   tempResolution
 *     This field specifies the temperature resolution value in Celsius units
 *     in the form of a 16-bit signed integer.  Legal values for this field
 *     range from +5 to +300 degrees Celsius, inclusive.
 */
typedef struct LW2080_CTRL_LWIF_SUBFUNC_THERMAL_MONITOR_SET_INPARAMS {
    LwS16 tempThreshold;
    LwS16 tempResolution;
} LW2080_CTRL_LWIF_SUBFUNC_THERMAL_MONITOR_SET_INPARAMS;

/*
 * LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_GET_INPARAMS
 *
 * This structure describes input parameters for the operation initiated
 * with the LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_GET subfunction.
 *
 *   panelIndex
 *     This field specifies the panel index for which brightness control
 *     data is desired.  The panel index organization uses the following
 *       The first byte of the panel index value (panelIndex[7:0]) contains
 *       byte 10 of the display EDID (byte 0 of the Product Code).
 *
 *       The second byte of the panel index value (panelIndex[15:8]) contains
 *       byte 11 of the display EDID (byte 1 of the Product Code).
 *
 *       The third byte of the panel index value (panelIndex[23:16]) contains
 *       byte 8 of the display EDID (byte 0 of the Manufacturer Code).
 *
 *       The fourth byte of the panel index value (panelIndex[31:24])
 *       contains byte 9 of the display EDID (byte 1 of the Manufacturer Code).
 */
typedef struct LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_GET_INPARAMS {
    LwU32 panelIndex;
} LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_GET_INPARAMS;

/*
 * LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_GET_OUTPARAMS
 *
 * This structure describes output parameters for the operation initiated
 * with the LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_GET subfunction.
 *
 *   reserved0
 *     This field is reserved.
 *   brightnessMinimum
 *     This field contains the panel brightness minimum value for PWM output
 *     signal zero.  Legal values for this field are in the range 0 to 255.
 *   brightnessMaximum
 *     This field contains the panel brightness minimum value for PWM output
 *     high signal.  Legal values for this field are in the range 0 to 255.
 *   flags
 *     This field contains brightness control flags.  Possible valid flags are:
 *       LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_FLAGS_LINMAP
 *         This flag can be used to enable linear mapping of brightness
 *         control levels.  When linear mapping is enabled, then the reported
 *         brightness levels are described by i * (100/N), where N is the
 *         number of elements in the brightness level list returned by
 *         LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_GET_LEVELS and i
 *         is the list position index.  If the number of levels in the list
 *         is not a multiple of 100 then all levels are shifted such that
 *         the highest level is 100.
 *   pwmFrequency
 *     This field contains the panel ilwerter PWM frequency in Hz units.
 *   pwmMinimumDutyCycle
 *     This field contains the minimum PWM duty cycle value in 0.1% units.
 *     For GPUs where VGA brightness control is required, this value must be
 *     0 (0.0%) as the VBIOS supports only PWM duty cycle range from 0 to 100%.
 *   pwmMaximumDutyCycle
 *     This field contains the maximum PWM duty cycle value in 0.1% units.
 *     For GPUs where VGA brightness control is required, this value must be
 *     1000 (100.0%) as the VBIOS supports only PWM duty cycle range from 0
 *     to 100%.
 */
typedef struct LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_GET_OUTPARAMS {
    LwU8  reserved0;
    LwU8  brightnessMinimum;
    LwU8  brightnessMaximum;
    LwU8  flags;
    LwU32 pwmFrequency;
    LwU16 pwmMinimumDutyCycle;
    LwU16 pwmMaximumDutyCycle;
} LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_GET_OUTPARAMS;

/* valid flags values */
#define LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_FLAGS_LINMAP            0:0
#define LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_FLAGS_LINMAP_DISABLE (0x00)
#define LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_FLAGS_LINMAP_ENABLE  (0x01)

/*
 * LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_SET_LEVEL_INPARAMS
 *
 * This structure describes input parameters for the operation initiated with
 * the LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_SET_LEVEL_IN subfunction.
 *
 *   level
 *     This field specifies the current brightness value.  Legal values for
 *     this parameter range from 0 (dimmest) to 100 (brightest) inclusive.
 */
typedef struct LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_SET_LEVEL_INPARAMS {
    LwU8  level;
    LwU8  reserved0;
    LwU16 reserved1;
} LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_SET_LEVEL_INPARAMS;

/*
 * LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_GET_LEVELS_INPARAMS
 *
 * This structure describes input parameters for the operation initiated
 * with the LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_GET_LEVELS subfunction.
 *
 *   panelIndex
 *     This field specifies the panel index for which brightness control
 *     data is desired.  The panel index organization uses the following
 *       The first byte of the panel index value (panelIndex[7:0]) contains
 *       byte 10 of the display EDID (byte 0 of the Product Code).
 *
 *       The second byte of the panel index value (panelIndex[15:8]) contains
 *       byte 11 of the display EDID (byte 1 of the Product Code).
 *
 *       The third byte of the panel index value (panelIndex[23:16]) contains
 *       byte 8 of the display EDID (byte 0 of the Manufacturer Code).
 *
 *       The fourth byte of the panel index value (panelIndex[31:24])
 *       contains byte 9 of the display EDID (byte 1 of the Manufacturer Code).
 */
typedef struct LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_GET_LEVELS_INPARAMS {
    LwU32 panelIndex;
} LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_GET_LEVELS_INPARAMS;

/*
 * LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_GET_LEVELS_OUTPARAMS
 *
 * This structure describes output parameters for the operation initiated
 * with the LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_GET_LEVELS subfunction.
 *
 *   acPowerLevelIndex
 *     This parameter returns the index into the brightness levels array used
 *     when the system is on AC power. Legal values for this parameter must be
 *     at least LW2080_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_AC_POWER_LEVEL_MIN and
 *     no more than LW2080_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_AC_POWER_LEVEL_MAX.
 *   dcPowerLevelIndex
 *     This parameter returns the index into the brightness levels array used
 *     when the system is on DC power. Legal values for this parameter must be
 *     at least LW2080_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_DC_POWER_LEVEL_MIN and
 *     no more than LW2080_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_DC_POWER_LEVEL_MAX.
 *     Brightness level when system is on DC power.
 *   levels
 *     This parameter returns the table of brightness levels in the form of
 *     an array of unsigned bytes.
 */
#define LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_MAX_LEVELS (0x100)
typedef struct LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_GET_LEVELS_OUTPARAMS {
    LwU8 acPowerLevel;
    LwU8 dcPowerLevel;
    LwU8 levels[LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_MAX_LEVELS];
} LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_GET_LEVELS_OUTPARAMS;

/* minimum/maximum brightness levels for AC/DC power */
#define LW2080_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_AC_POWER_LEVEL_MIN (0x00000000)
#define LW2080_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_AC_POWER_LEVEL_MAX (0x000000FF)
#define LW2080_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_DC_POWER_LEVEL_MIN (0x00000000)
#define LW2080_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_DC_POWER_LEVEL_MAX (0x000000FF)

/*
 * LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_SET_MONITOR_INPARAMS
 *
 * This structure describes input parameters for the operation initiated with
 * the LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_SET_MONITOR subfunction.
 */
typedef struct LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_SET_MONITOR_INPARAMS {
    LwU8  monitorState;
    LwU8  reserved0;
    LwU16 reserved1;
} LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_SET_MONITOR_INPARAMS;

/* valid monitorState values */
#define LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_SET_MONITOR_STATE_OFF (0x00)
#define LW2080_CTRL_LWIF_SUBFUNC_BRIGHTNESS_CONTROL_SET_MONITOR_STATE_ON  (0x01)

/*
 * LW2080_CTRL_LWIF_SUBFUNC_POWERMIZER_LIMIT_GET_OUTPARAMS
 *
 * This structure describes output parameters for the operation initiated with
 * the LW2080_CTRL_LWIF_SUBFUNC_POWERMIZER_LIMIT_GET subfunction.
 *
 *   type
 *     This parameter returns the current PowerMizer limit type from the
 *     SBIOS.  Legal values for this parameter include:
 *       LW2080_CTRL_LWIF_SUBFUNC_POWERMIZER_LIMIT_TYPE_MAXBATT
 *         The PowerMizer limit is for maximum battery savings.
 *       LW2080_CTRL_LWIF_SUBFUNC_POWERMIZER_LIMIT_TYPE_BALANCED
 *         The PowerMizer limit is for a balance between maximum power
 *         savings and maximum performance.
 *       LW2080_CTRL_LWIF_SUBFUNC_POWERMIZER_LIMIT_TYPE_MAXPERF
 *         The PowerMizer limit is for maximum performance.
 *   reason
 *     This parameter returns the current PowerMizer reason from the SBIOS.
 *     This value helps discern whether the limit being applied by the SBIOS
 *     is a "hard limit" (one that cannot be exceeded) or a soft limit (one
 *     that can be exceeded when sw deems it necessary).  Legal values for
 *     this parameter include:
 *       LW2080_CTRL_LWIF_SUBFUNC_POWERMIZER_LIMIT_REASON_NONE
 *         There is no reason specified.
 *       LW2080_CTRL_LWIF_SUBFUNC_POWERMIZER_LIMIT_REASON_TEMP_THRESH
 *         The temperature threshold has been crossed.  This is typically a
 *         hard limit.
 *       LW2080_CTRL_LWIF_SUBFUNC_POWERMIZER_LIMIT_REASON_CRIT_POWER_THRESH
 *         The critical power threshold has been crossed.  This is typically a
 *         hard limit.
 *       LW2080_CTRL_LWIF_SUBFUNC_POWERMIZER_LIMIT_REASON_SYS_POWER_THRESH
 *         The system power threshold has been crossed.  This is typically a
 *         soft limit.
 */
typedef struct LW2080_CTRL_LWIF_SUBFUNC_POWERMIZER_LIMIT_GET_OUTPARAMS {
    LwU8 type;
    LwU8 reserved0;
    LwU8 reserved1;
    LwU8 reason;
} LW2080_CTRL_LWIF_SUBFUNC_POWERMIZER_LIMIT_GET_OUTPARAMS;

/* valid limit values */
#define LW2080_LWIF_SUBFUNC_POWERMIZER_LIMIT_TYPE_MAXBATT             (0x00)
#define LW2080_LWIF_SUBFUNC_POWERMIZER_LIMIT_TYPE_BALANCED            (0x01)
#define LW2080_LWIF_SUBFUNC_POWERMIZER_LIMIT_TYPE_MAXPERF             (0x02)

/* valid reason values */
#define LW2080_LWIF_SUBFUNC_POWERMIZER_LIMIT_REASON_NONE              (0x00)
#define LW2080_LWIF_SUBFUNC_POWERMIZER_LIMIT_REASON_TEMP_THRESH       (0x01)
#define LW2080_LWIF_SUBFUNC_POWERMIZER_LIMIT_REASON_CRIT_POWER_THRESH (0x02)
#define LW2080_LWIF_SUBFUNC_POWERMIZER_LIMIT_REASON_SYS_POWER_THRESH  (0x03)

/*
 * LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_OUTPARAMS
 *
 * This structure describes output parameters for the operation initiated with
 * the LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET subfunction.
 *
 *   acpiDeviceId
 *     This field returns the display device ID.  The display device ID
 *     must match the ID specified by the GPU vendor.
 *   attrs
 *     This field returns the attributes for the associated display device.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_HOTPLUG
 *         This attribute indicates that the SBIOS is able to detect the
 *         device.  This is a platform-specific attribute.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_OVERRIDE_ENABLE
 *         This attribute returns the SBIOS device enable override.
 *         When set to _TRUE the SBIOS preference specified in the
 *         GET_ENABLE attribute should be used.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_ENABLE
 *         This attribute returns the device enable preference of the SBIOS.
 *         A value of _FALSE indicates that the display device should be
 *         disabled.  A value of _TRUE indicates that the display device should
 *         be enabled.  This field is only valid if the _OVERRIDE_ENABLE
 *         attribute is _TRUE.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_OVERRIDE_RES
 *         This attribute returns the SBIOS display device resolution override.
 *         When set to _TRUE the SBIOS preference specified in the GET_RES
 *         attribute should be used.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_RES
 *         This attribute returns the display device resolution preference of
 *         the SBIOS.  A value of _FALSE indicates that the highest supported
 *         display device resolution should be used.  A value of _TRUE is
 *         lwrrently reserved.  This field is only valid if the
 *         _OVERRIDE_RESOLUTION attribute is _TRUE.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_OVERRIDE_INFO
 *         This attribute returns the SBIOS display device information
 *         override.  When set to _TRUE the SBIOS display-specific preference
 *         specified in the GET_INFO attribute should be used.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_INFO
 *         This field returns the SBIOS display-specific information
 *         preference.  This field is only valid if the _OVERRIDE_INFO
 *         attribute is _TRUE.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_OVERRIDE_SCALE_FORMAT
 *         This attribute returns the SBIOS display scaling override for
 *         digital flat panel devices.  When set to _TRUE the SBIOS display
 *         scaling preference specified in the _SCALE_FORMAT attribute
 *         should be used.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_SCALE_FORMAT
 *         This attribute returns the SBIOS scaling preference.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_OVERRIDE_TV_FORMAT
 *         This attribute returns the SBIOS display override for TV format.
 *         When set to _TRUE the SBIOS TV format preference specified in
 *         the _TV_FORMAT attribute should be used.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_TV_FORMAT
 *         This attribute returns the SBIOS TV format preference.
 */
typedef struct LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_OUTPARAMS {
    LwU16 acpiDeviceId;
    LwU16 attrs;
} LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_OUTPARAMS;

/* attrs parameter fields and values */
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_HOTPLUG        0:0
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_HOTPLUG_FALSE               (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_HOTPLUG_TRUE                (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_OVERRIDE_ENABLE 1:1
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_OVERRIDE_ENABLE_FALSE       (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_OVERRIDE_ENABLE_TRUE        (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_ENABLE         2:2
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_ENABLE_FALSE                (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_ENABLE_TRUE                 (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_OVERRIDE_RES   3:3
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_OVERRIDE_RES_FALSE          (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_OVERRIDE_RES_TRUE               (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_RES            4:4
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_RES_FALSE                   (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_RES_TRUE                    (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_OVERRIDE_INFO  7:7
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_OVERRIDE_INFO_FALSE         (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_OVERRIDE_INFO_TRUE          (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_INFO           15:8
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_OVERRIDE_SCALE_FORMAT 7:7
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_OVERRIDE_SCALE_FORMAT_FALSE (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_OVERRIDE_SCALE_FORMAT_TRUE  (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_SCALE_FORMAT   9:8
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_SCALE_FORMAT_CENTERED       (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_SCALE_FORMAT_SCALED         (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_SCALE_FORMAT_ASPECT         (0x00000002)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_OVERRIDE_TV_FORMAT    7:7
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_OVERRIDE_TV_FORMAT_FALSE    (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_OVERRIDE_TV_FORMAT_TRUE     (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_GET_TV_FORMAT      12:8

/*
 * LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_INPARAMS
 *
 * This structure describes input parameters for the operation initiated with
 * the LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET subfunction.
 *
 *   acpiDeviceId
 *     This field returns the display device ID.  The display device ID
 *     must match the ID specified by the GPU vendor.
 *   attrs
 *     This field returns the attributes for the associated display device.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_UPDATE_ENABLE
 *         This attribute indicates that the device enable status should be
 *         updated.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_ENABLE
 *         This attribute contains the device enable status.  A value of
 *         _FALSE indicates the display device is disabled.  A value of _TRUE
 *         indicates the display device is enabled.  This field is only
 *         valid if _UPDATE_ENABLE is _TRUE.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_UPDATE_CONNECT
 *         This attribute indicates that the device connect status should be
 *         updated.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_CONNECT
 *         This attribute contains the device connect status.  A value of
 *         _FALSE indicates the display device is disconnected.  A value of
 *         _TRUE indicates the display device is connected.  This field is only
 *         valid if _UPDATE_CONNECT is _TRUE.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_UPDATE_INFO
 *         This attribute indicates that the device-specific attribute
 *         information should be updated.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_INFO
 *         This attribute contains the device-specific attribute information.
 *         This field is only valid if _UPDATE_INFO is _TRUE.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_UPDATE_SCALE_FORMAT
 *         This attribute notifies the SBIOS of the display scaling attribute
 *         for digital flat panel devices.  When set to _TRUE the display
 *         scaling setting is specified in the _SCALE_FORMAT attribute.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_SCALE_FORMAT
 *         This attribute sets the scaling format.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_UPDATE_TV_FORMAT
 *         This attribute notifies the SBIOS of the TV format.  When set to
 *         _TRUE the TV format is specified in the _TV_FORMAT attribute.
 *       LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_TV_FORMAT
 *         This attribute returns the SBIOS TV format preference.
 */
typedef struct LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_INPARAMS {
    LwU16 acpiDeviceId;
    LwU16 attrs;
} LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_INPARAMS;

/* attrs parameter fields and values */
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_UPDATE_ENABLE  1:1
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_UPDATE_ENABLE_FALSE       (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_UPDATE_ENABLE_TRUE        (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_ENABLE         2:2
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_ENABLE_FALSE              (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_ENABLE_TRUE               (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_UPDATE_CONNECT 3:3
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_UPDATE_CONNECT_FALSE      (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_UPDATE_CONNECT_TRUE       (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_CONNECT        4:4
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_CONNECT_FALSE             (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_CONNECT_TRUE              (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_UPDATE_INFO    7:7
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_UPDATE_INFO_FALSE         (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_UPDATE_INFO_TRUE          (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_INFO           15:8
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_UPDATE_SCALE_FORMAT 7:7
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_UPDATE_SCALE_FORMAT_FALSE (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_UPDATE_SCALE_FORMAT_TRUE  (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_SCALE_FORMAT   9:8
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_SCALE_FORMAT_CENTERED     (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_SCALE_FORMAT_SCALED       (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_SCALE_FORMAT_ASPECT       (0x00000002)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_UPDATE_TV_FORMAT 7:7
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_UPDATE_TV_FORMAT_FALSE    (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_UPDATE_TV_FORMAT_TRUE     (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_TV_FORMAT      12:8

typedef struct LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_OUTPARAMS {
    LwU16 acpiDeviceId;
    LwU16 attrs;
} LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_OUTPARAMS;

/* attrs parameter fields and values */
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_OUTPARAMS_BIOS_DETECT  0:0
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_OUTPARAMS_BIOS_DETECT_FALSE               (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_OUTPARAMS_BIOS_DETECT_TRUE                (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_OUTPARAMS_OVERRIDE  1:1
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_OUTPARAMS_OVERRIDE_FALSE                  (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_OUTPARAMS_OVERRIDE_TRUE                   (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_OUTPARAMS_OVERRIDE_SET  2:2
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_OUTPARAMS_OVERRIDE_SET_FALSE              (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_OUTPARAMS_OVERRIDE_SET_TRUE               (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_OUTPARAMS_OVERRIDE_RESOLUTION  3:3
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_OUTPARAMS_OVERRIDE_RESOLUTION_FALSE       (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_OUTPARAMS_OVERRIDE_RESOLUTION_TRUE        (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_OUTPARAMS_DEVICE_RESOLUTION  4:4
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_OUTPARAMS_DEVICE_RESOLUTION_USE_HIGHEST   (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_OUTPARAMS_DEVICE_RESOLUTION_RESERVED      (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_OUTPARAMS_BIOS_DEFINE_OVERRIDE_ATTR  7:7
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_OUTPARAMS_BIOS_DEFINE_OVERRIDE_ATTR_FALSE (0x00000000)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_OUTPARAMS_BIOS_DEFINE_OVERRIDE_ATTR_TRUE  (0x00000001)
#define LW2080_CTRL_LWIF_SUBFUNC_DISPLAY_ATTRIBUTES_SET_OUTPARAMS_BIOS_PREFERRED_ATTR  15:8

/*
 * LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_GET_MEM_BW_OUTPARAMS
 *
 * This structure describes output parameters for the operation initiated with
 * the LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_GET_MEM_BW subfunction.
 *
 *   memBandwidth0
 *     This parameter returns the first dword of bandwidth data.
 *       LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_BW0_TAVN
 *         This field returns the total latency.
 *       LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_BW0_BASL
 *         This field returns the base latency.
 *       LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_BW0_LBW
 *         This field returns the link bandwidth.
 *   memBandwidth1
 *       LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_BW1_ATBW
 *         This field returns the total available bandwidth.
 *       LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_BW1_CLT
 *         This field returns the cache line time.
 *       LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_BW1_PMP
 *         This field returns the page miss penalty.
 */
typedef struct LW2080_CTRL_LWIF_SUBFUNC_GET_MEM_BW_OUTPARAMS {
    LwU32 memBw0;
    LwU32 memBw1;
} LW2080_CTRL_LWIF_SUBFUNC_GET_MEM_BW_OUTPARAMS;

/* memBandwidth0 fields */
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_BW0_TAVN              10:0
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_BW0_BASL              20:11
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_BW0_LBW               31:21

/* memBandwidth1 fields */
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_BW1_ATBW              11:0
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_BW1_CLT               21:12
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_BW1_PMP               31:22


/*
 * LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_GET_SVM_SETTINGS_OUTPARAMS
 *
 * This structure describes output parameters for the operation initiated with
 * the LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_GET_SVM_SETTINGS subfunction.
 *
 *   tableInfo
 *     This parameter returns the general SBIOS table information.
 *   
 *   svmSettings
 *       LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_SVM_SIZE
 *         This field returns the SVM size in MBs.
 */
typedef struct LW2080_CTRL_LWIF_SUBFUNC_GET_SVM_SETTINGS_OUTPARAMS {
    LwU32 tableinfo;
    LwU16 svmSettings;
} LW2080_CTRL_LWIF_SUBFUNC_GET_SVM_SETTINGS_OUTPARAMS;

/* svmSize fields */
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_SVM_SIZE                   15:0

#define LW2080_CTRL_LWIF_SUBFUNC_GET_SVM_SETTINGS_OUTPARAMS_SIZE_V0 6

/* LW2080_CTRL_LWIF_FUNC_BRIGHTNESS_CONTROL subfunction parameter values */
#define LW2080_CTRL_LWIF_SUBFUNC_HDCP_GET_KEYS                      (0x00000001)



/* _ctrl2080lwif_h_ */

/* LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP subfunction fields */
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP_HEADER_MINOR_VERSION                  MW(3:0)
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP_HEADER_MAJOR_VERSION                  MW(7:4)
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP_HEADER_HEADER_SIZE                    MW(15:8)
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP_HEADER_NUMBER_REGION_ENTRIES          MW(19:16)
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP_HEADER_NUMBER_DISPLAY_ENTRIES         MW(23:20)
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP_HEADER_REGION_ENTRY_SIZE              MW(27:24)
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP_HEADER_DISPLAY_ENTRY_SIZE             MW(31:28)
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP_HEADER_PM_LATENCY                     MW(42:32)
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP_HEADER_SYS_MEM_CLOCK                  MW(56:43)

#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP_REGION_PROP_PARTITIONS                MW(2:0)
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP_REGION_PROP_MIN_DRAM_BANKS            MW(6:3)
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP_REGION_PROP_MAX_DRAM_BANKS            MW(10:7)
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP_REGION_PROP_MIN_DRAM_COLS             MW(14:11)
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP_REGION_PROP_MAX_DRAM_COLS             MW(18:15)
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP_REGION_PROP_SIZE_LOW32                MW(50:19)   
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP_REGION_PROP_SIZE_HIGH6                MW(56:51)   

#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP_DISP_PROP_BANDWIDTH                   MW(15:0)
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP_DISP_PROP_TAVN                        MW(26:16)
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP_DISP_PROP_BASL                        MW(36:27)
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP_DISP_PROP_LBW                         MW(47:37)
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP_DISP_PROP_ATBW                        MW(59:48)
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP_DISP_PROP_CLT                         MW(69:60)
#define LW2080_CTRL_LWIF_SUBFUNC_PLATCFG_MEM_PROP_DISP_PROP_PMP                         MW(79:70)
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

