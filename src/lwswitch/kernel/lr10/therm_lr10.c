/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "g_lwconfig.h"
#include "export_lwswitch.h"
#include "common_lwswitch.h"
#include "error_lwswitch.h"
#include "lr10/lr10.h"
#include "lr10/therm_lr10.h"
#include "lr10/fuse_lr10.h"
#include "soe/soeiftherm.h"
#include "rmflcncmdif_lwswitch.h"
#include "soe/soe_lwswitch.h"

#include "lwswitch/lr10/dev_therm.h"
#include "lwswitch/lr10/dev_fuse.h"
#include "lwswitch/lr10/dev_lwlsaw_ip.h"

//
// Thermal functions
//

//
// Initialize thermal offsets for External Tdiode.
//

LwlStatus
lwswitch_init_thermal_lr10
(
    lwswitch_device *device
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    PLWSWITCH_OBJI2C pI2c = device->pI2c;
    LwU32 i;
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

     // Mark everything invalid
    chip_device->tdiode.method = LWSWITCH_THERM_METHOD_UNKNOWN;

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    if (device->board_id == LWSWITCH_BOARD_ID_E4700_A02)
    {
        //
        // On E4700, LR10 external tdiode is attached to an TMP451 attached to WS_I2C_A.
        //
        // The tdiode characterization yields a set of coefficients:
        // temp = (sensors - offset)*A + B to aclwrately callwlate temperature.
        //
        // TODO : These coefficients needs to be updated with the to HW calibrated values.
        // Tracked in Bug 200507802.
        chip_device->tdiode.A = 1008;    //  1.0084
        chip_device->tdiode.B = -1634;    // -1.6341
        chip_device->tdiode.offset = lwswitch_fuse_opt_read_lr10(device, LWSWITCH_FUSE_OPT_TDIODE_LR10);
        chip_device->tdiode.method_i2c_info = NULL;

        for (i = 0; i < pI2c->device_list_size; i++)
        {
            if (pI2c->device_list[i].i2cDeviceType == LWSWITCH_I2C_DEVICE_TMP451)
            {
                chip_device->tdiode.method_i2c_info = &pI2c->device_list[i];
                chip_device->tdiode.method = LWSWITCH_THERM_METHOD_I2C;
                break;
            }
        }
    }
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    return LWL_SUCCESS;
}

static void
_lwswitch_read_max_tsense_temperature
(
    lwswitch_device *device,
    LWSWITCH_CTRL_GET_TEMPERATURE_PARAMS *info,
    LwU32 channel
)
{
    LwU32  offset;
    LwU32  temperature;

    temperature = lwswitch_reg_read_32(device, LW_THERM_TSENSE_MAXIMUM_TEMPERATURE);
    temperature = DRF_VAL(_THERM_TSENSE, _MAXIMUM_TEMPERATURE, _MAXIMUM_TEMPERATURE, temperature);

    if (channel == LWSWITCH_THERM_CHANNEL_LR10_TSENSE_MAX)
    {
        offset = lwswitch_reg_read_32(device, LW_THERM_TSENSE_U2_A_0_BJT_0_TEMPERATURE_MODIFICATIONS);
        offset = DRF_VAL(_THERM_TSENSE, _U2_A_0_BJT_0_TEMPERATURE_MODIFICATIONS, _TEMPERATURE_OFFSET, offset);

        // Temperature of the sensor reported equals callwlation of the max temperature reported
        // from the TSENSE HUB plus the temperature offset programmed by SW. This offset needs to
        // be substracted to get the actual temperature of the sensor.
        temperature -= offset;
    }

    info->temperature[channel] = LW_TSENSE_FXP_9_5_TO_24_8(temperature);
    info->status[channel] = LWL_SUCCESS;
}

static void
_lwswitch_read_external_tdiode_temperature
(
    lwswitch_device *device,
    LWSWITCH_CTRL_GET_TEMPERATURE_PARAMS *info,
    LwU32 channel
)
{
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LwlStatus retval;
    LwBool extended_mode;
    LwU32 integer;
    LwU32 fraction;
    LwU32 offset;
    LwTemp temperature;
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS i2cIndexed = { 0 };
    
    if (device->board_id != LWSWITCH_BOARD_ID_E4700_A02)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Unknown board Id\n",
            __FUNCTION__);

        info->temperature[channel] = 0;
        info->status[channel] = -LWL_NOT_FOUND;
        return;
    }

    i2cIndexed.bIsRead = LW_TRUE;
    i2cIndexed.port = chip_device->tdiode.method_i2c_info->i2cPortLogical;
    i2cIndexed.address = (LwU16) chip_device->tdiode.method_i2c_info->i2cAddress;
    i2cIndexed.messageLength = 1;
    i2cIndexed.flags =
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _START, _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _STOP, _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _RESTART, _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _ADDRESS_MODE, _7BIT) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, _ONE) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _FLAVOR, _HW) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, _400KHZ) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _BLOCK_PROTOCOL, _DISABLED) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _TRANSACTION_MODE, _NORMAL);

    // Read Configuration register (0x3) to get the mode of TMP451.
    // TMP451 supports a temperature range of 0C to 127C in default mode
    // In Extended mode it supports an Extended range of -40C to 125C.

    i2cIndexed.index[0] = 0x3;
    retval = lwswitch_ctrl_i2c_indexed(device, &i2cIndexed);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: I2C read to external tdiode failed\n",
            __FUNCTION__);

        info->temperature[channel] = 0;
        info->status[channel] = -LWL_IO_ERROR;
        return;
    }

    // The 2nd Bit in Configuration register gives the mode of the sensor.
    extended_mode = i2cIndexed.message[0] & LWBIT(2);
 
    // Read MSB bit[0x01] to get the integral part of temperature.
    // In Extended mode, the device returns an extra offset of 0x40 which should be removed.
    i2cIndexed.index[0] = 0x01;
    retval = lwswitch_ctrl_i2c_indexed(device, &i2cIndexed);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: I2C read to external tdiode failed\n",
            __FUNCTION__);

        info->temperature[channel] = 0;
        info->status[channel] = -LWL_IO_ERROR;
        return;
    }

    integer = i2cIndexed.message[0] - (extended_mode ? 0x40 : 0x0);

    // Read LSB bit[0x10] to get the fractional part of temperature.
    i2cIndexed.index[0] = 0x10;
    retval = lwswitch_ctrl_i2c_indexed(device, &i2cIndexed);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: I2C read to external tdiode failed\n",
            __FUNCTION__);

        info->temperature[channel] = 0;
        info->status[channel] = -LWL_IO_ERROR;
        return;
    }

    fraction = i2cIndexed.message[0];
    temperature = ((integer) << 8 | fraction);

    // TODO : Update the temperature with HW calibrated offset values.
    // Tracked in Bug 200507802.
    if (channel == LWSWITCH_THERM_CHANNEL_LR10_TDIODE)
    {
        offset = chip_device->tdiode.offset;
        temperature = ((integer - offset) << 8 | fraction);
    }

    info->temperature[channel] = temperature;
    info->status[channel] = LWL_SUCCESS;
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
}

LwlStatus
lwswitch_ctrl_therm_read_temperature_lr10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_GET_TEMPERATURE_PARAMS *info
)
{
    LwU32 channel;

    if (!info->channelMask)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: No channel given in the input.\n",
            __FUNCTION__);

        return -LWL_BAD_ARGS;
    }

    lwswitch_os_memset(info->temperature, 0x0, sizeof(info->temperature));

    channel = LWSWITCH_THERM_CHANNEL_LR10_TSENSE_MAX;
    if (info->channelMask & LWBIT(channel))
    {
        _lwswitch_read_max_tsense_temperature(device, info, channel);
        info->channelMask &= ~LWBIT(channel);
    }

    channel = LWSWITCH_THERM_CHANNEL_LR10_TSENSE_OFFSET_MAX;
    if (info->channelMask & LWBIT(channel))
    {
        _lwswitch_read_max_tsense_temperature(device, info, channel);        
        info->channelMask &= ~LWBIT(channel);
    }

    channel = LWSWITCH_THERM_CHANNEL_LR10_TDIODE;
    if (info->channelMask & LWBIT(channel))
    {
        _lwswitch_read_external_tdiode_temperature(device, info, channel);
        info->channelMask &= ~LWBIT(channel);
    }

    channel = LWSWITCH_THERM_CHANNEL_LR10_TDIODE_OFFSET;
    if (info->channelMask & LWBIT(channel))
    {
        _lwswitch_read_external_tdiode_temperature(device, info, channel);
        info->channelMask &= ~LWBIT(channel);
    }

    if (info->channelMask)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: ChannelMask %x absent on LR10.\n",
            __FUNCTION__, info->channelMask);

        return -LWL_BAD_ARGS;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_ctrl_therm_get_temperature_limit_lr10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_GET_TEMPERATURE_LIMIT_PARAMS *info
)
{
    LwU32 threshold;
    LwU32 temperature;

    threshold = lwswitch_reg_read_32(device, LW_THERM_TSENSE_THRESHOLD_TEMPERATURES);

    switch (info->thermalEventId)
    {
        case LWSWITCH_CTRL_THERMAL_EVENT_ID_WARN:
        {
            // Get Slowdown temperature
            temperature = DRF_VAL(_THERM_TSENSE, _THRESHOLD_TEMPERATURES,
                                  _WARNING_TEMPERATURE, threshold);
            break;
        }
        case LWSWITCH_CTRL_THERMAL_EVENT_ID_OVERT:
        {
            // Get Shutdown temperature
            temperature = DRF_VAL(_THERM_TSENSE, _THRESHOLD_TEMPERATURES,
                                  _OVERTEMP_TEMPERATURE, threshold);
            break;
        }
        default:
        {
            LWSWITCH_PRINT(device, ERROR, "Invalid Thermal Event Id: 0x%x\n", info->thermalEventId);
            return -LWL_BAD_ARGS;
        }
    }
    
    info->temperatureLimit = LW_TSENSE_FXP_9_5_TO_24_8(temperature);

    return LWL_SUCCESS;
}

// Background task to monitor thermal warn and adjust link mode
void
lwswitch_monitor_thermal_alert_lr10
(
    lwswitch_device *device
)
{
    return;
}

/*
 * @brief Callback function to recieve thermal messages from SOE.
 */
void
lwswitch_therm_soe_callback_lr10
(
    lwswitch_device *device,
    RM_FLCN_MSG *pGenMsg,
    void *pParams,
    LwU32 seqDesc,
    LW_STATUS status
)
{
    RM_SOE_THERM_MSG_SLOWDOWN_STATUS slowdown_status;
    RM_SOE_THERM_MSG_SHUTDOWN_STATUS shutdown_status;
    RM_FLCN_MSG_SOE *pMsg = (RM_FLCN_MSG_SOE *)pGenMsg;
    LwU32 temperature;
    LwU32 threshold;

    switch (pMsg->msg.soeTherm.msgType)
    {
        case RM_SOE_THERM_MSG_ID_SLOWDOWN_STATUS:
        {
            slowdown_status = pMsg->msg.soeTherm.slowdown;
            if (slowdown_status.bSlowdown)
            {
                if (slowdown_status.source.bTsense) // TSENSE_THERM_ALERT
                {
                    temperature = RM_SOE_LW_TEMP_TO_CELSIUS_TRUNCED(slowdown_status.maxTemperature);
                    threshold   = RM_SOE_LW_TEMP_TO_CELSIUS_TRUNCED(slowdown_status.warnThreshold);

                    LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_HOST_THERMAL_EVENT_START,
                        "LWSWITCH Temperature %dC | TSENSE WARN Threshold %dC\n",
                        temperature, threshold);

                    LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_HOST_THERMAL_EVENT_START,
                        "Thermal Slowdown Engaged | Temp higher than WARN Threshold\n");
                }

                if (slowdown_status.source.bPmgr) // PMGR_THERM_ALERT
                {
                    LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_HOST_THERMAL_EVENT_START,
                        "Thermal Slowdown Engaged | PMGR WARN Threshold reached\n");
                }
            }
            else // REVERT_SLOWDOWN
            {
                temperature = RM_SOE_LW_TEMP_TO_CELSIUS_TRUNCED(slowdown_status.maxTemperature);
                threshold   = RM_SOE_LW_TEMP_TO_CELSIUS_TRUNCED(slowdown_status.warnThreshold);

                LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_HOST_THERMAL_EVENT_END,
                    "LWSWITCH Temperature %dC | TSENSE WARN Threshold %dC\n",
                    temperature, threshold);

                LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_HOST_THERMAL_EVENT_END,
                    "Thermal slowdown Disengaged\n");
            }
            break;
        }

        case RM_SOE_THERM_MSG_ID_SHUTDOWN_STATUS:
        {
            shutdown_status = pMsg->msg.soeTherm.shutdown;
            if (shutdown_status.source.bTsense) // TSENSE_THERM_SHUTDOWN
            {
                temperature = RM_SOE_LW_TEMP_TO_CELSIUS_TRUNCED(shutdown_status.maxTemperature);
                threshold   = RM_SOE_LW_TEMP_TO_CELSIUS_TRUNCED(shutdown_status.overtThreshold);

                LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_HOST_THERMAL_SHUTDOWN,
                    "LWSWITCH Temperature %dC | OVERT Threshold %dC\n",
                    temperature, threshold);

                LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_HOST_THERMAL_SHUTDOWN,
                    "TSENSE OVERT Threshold reached. Shutting Down\n");
            }


            if (shutdown_status.source.bPmgr) // PMGR_THERM_SHUTDOWN
            {
                LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_HOST_THERMAL_EVENT_START,
                    "PMGR OVERT Threshold reached. Shutting Down\n");
            }
            break;
        }
        default:
        {
            LWSWITCH_PRINT(device, ERROR, "%s Unknown message Id\n", __FUNCTION__);
            LWSWITCH_ASSERT(0);
        }
    }
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
//
// lwswitch_therm_read_voltage
//
// Temperature and voltage are only available on SKUs which have thermal and
// voltage sensors.
//

LwlStatus
lwswitch_ctrl_therm_read_voltage_lr10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_GET_VOLTAGE_PARAMS *info
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_ctrl_force_thermal_slowdown_lr10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_SET_THERMAL_SLOWDOWN *p
)
{
    if (p->slowdown)
    {
        LWSWITCH_PRINT(device, INFO,
            "%s: Forcing thermal Slowdown for %dus.\n",
            __FUNCTION__, p->periodUs);
    }
    else
    {
        LWSWITCH_PRINT(device, INFO,
            "%s: Slowdown Reverted.\n", __FUNCTION__);
    }

    return soeForceThermalSlowdown_HAL(device,
        p->slowdown, p->periodUs);
}
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
