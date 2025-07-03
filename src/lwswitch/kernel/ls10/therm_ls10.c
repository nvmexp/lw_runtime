/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "export_lwswitch.h"
#include "g_lwconfig.h"
#include "common_lwswitch.h"
#include "ls10/ls10.h"
#include "ls10/therm_ls10.h"
#include "ls10/fuse_ls10.h"
#include "error_lwswitch.h"
#include "soe/soeiftherm.h"

#include "lwswitch/ls10/dev_therm.h"
#include "lwswitch/ls10/dev_fuse.h"

//
// Thermal functions
//

//
// Initialize thermal offsets for External Tdiode.
//

LwlStatus
lwswitch_init_thermal_ls10
(
    lwswitch_device *device
)
{
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    PLWSWITCH_OBJI2C pI2c = device->pI2c;
    LwU32 i;
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

     // Mark everything invalid
    chip_device->tdiode.method = LWSWITCH_THERM_METHOD_UNKNOWN;

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    if (device->board_id == LWSWITCH_BOARD_ID_E4840)
    {
        //
        // On E4840, LS10 external tdiode is attached to an TMP451 attached to WS_I2C_A.
        //
        // The tdiode characterization yields a set of coefficients:
        // temp = (sensors - offset)*A + B to aclwrately callwlate temperature.
        //
        // TODO : These coefficients needs to be updated with the to HW calibrated values.
        // Tracked in Bug 200507802.
        chip_device->tdiode.A = 1008;    //  1.0084
        chip_device->tdiode.B = -1634;    // -1.6341
        chip_device->tdiode.offset = lwswitch_fuse_opt_read_ls10(device, LW_FUSE_OPT_CP2_TDIODE_OFFSET);
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
_lwswitch_read_max_tsense_temperature_ls10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_GET_TEMPERATURE_PARAMS *info,
    LwU32 channel
)
{
    LwU32  offset;
    LwU32  temperature;

    temperature = LWSWITCH_REG_RD32(device, _THERM, _TSENSE_MAXIMUM_TEMPERATURE);
    temperature = DRF_VAL(_THERM_TSENSE, _MAXIMUM_TEMPERATURE, _MAXIMUM_TEMPERATURE, temperature);
    temperature = LW_TSENSE_FXP_9_5_TO_24_8(temperature);

    if (channel == LWSWITCH_THERM_CHANNEL_LR10_TSENSE_MAX)
    {
        offset = LWSWITCH_REG_RD32(device, _THERM, _TSENSE_U2_A_0_BJT_0_TEMPERATURE_MODIFICATIONS);
        offset = DRF_VAL(_THERM_TSENSE, _U2_A_0_BJT_0_TEMPERATURE_MODIFICATIONS, _TEMPERATURE_OFFSET, offset);

        // Temperature of the sensor reported equals callwlation of the max temperature reported
        // from the TSENSE HUB plus the temperature offset programmed by SW. This offset needs to
        // be substracted to get the actual temperature of the sensor.
        temperature -= LW_TSENSE_FXP_9_5_TO_24_8(offset);
    }

    info->temperature[channel] = temperature;
    info->status[channel] = LWL_SUCCESS;
}

static void
_lwswitch_read_external_tdiode_temperature_ls10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_GET_TEMPERATURE_PARAMS *info,
    LwU32 channel
)
{
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);
    LwlStatus retval;
    LwBool extended_mode;
    LwU32 integer;
    LwU32 fraction;
    LwU32 offset;
    LwTemp temperature;
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS i2cIndexed = { 0 };
    
    if (device->board_id != LWSWITCH_BOARD_ID_E4840)
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

//
// lwswitch_therm_read_temperature
//
// Temperature and voltage are only available on SKUs which have thermal and
// voltage sensors.
//

LwlStatus
lwswitch_ctrl_therm_read_temperature_ls10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_GET_TEMPERATURE_PARAMS *info
)
{
    LwU32 channel;
    LwU32 val;
    LwU32 offset;

    if (!info->channelMask)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: No channel given in the input.\n",
            __FUNCTION__);

        return -LWL_BAD_ARGS;
    }

    lwswitch_os_memset(info->temperature, 0x0, sizeof(info->temperature));

    channel = LWSWITCH_THERM_CHANNEL_LS10_TSENSE_MAX;
    if (info->channelMask & LWBIT(channel))
    {
        _lwswitch_read_max_tsense_temperature_ls10(device, info, channel);
        info->channelMask &= ~LWBIT(channel);
    }

    channel = LWSWITCH_THERM_CHANNEL_LS10_TSENSE_OFFSET_MAX;
    if (info->channelMask & LWBIT(channel))
    {
        _lwswitch_read_max_tsense_temperature_ls10(device, info, channel);        
        info->channelMask &= ~LWBIT(channel);
    }

    channel = LWSWITCH_THERM_CHANNEL_LS10_TDIODE;
    if (info->channelMask & LWBIT(channel))
    {
        _lwswitch_read_external_tdiode_temperature_ls10(device, info, channel);
        info->channelMask &= ~LWBIT(channel);
    }

    channel = LWSWITCH_THERM_CHANNEL_LS10_TDIODE_OFFSET;
    if (info->channelMask & LWBIT(channel))
    {
        _lwswitch_read_external_tdiode_temperature_ls10(device, info, channel);
        info->channelMask &= ~LWBIT(channel);
    }

    channel = LWSWITCH_THERM_CHANNEL_LS10_TSENSE_0;
    if (info->channelMask & LWBIT(channel))
    {
        offset = LWSWITCH_REG_RD32(device, _THERM, _TSENSE_U2_A_0_BJT_0_TEMPERATURE_MODIFICATIONS);
        offset = DRF_VAL(_THERM_TSENSE, _U2_A_0_BJT_0_TEMPERATURE_MODIFICATIONS, _TEMPERATURE_OFFSET, offset);
        offset = LW_TSENSE_FXP_9_5_TO_24_8(offset);

        val = LWSWITCH_REG_RD32(device, _THERM, _TSENSE_U2_A_0_BJT_0);
        val = DRF_VAL(_THERM, _TSENSE_U2_A_0_BJT_0, _TEMPERATURE, val);
        val = LW_TSENSE_FXP_9_5_TO_24_8(val);
        val -= offset;

        info->temperature[channel] = val;
        info->channelMask &= ~LWBIT(channel);
    }

    channel = LWSWITCH_THERM_CHANNEL_LS10_TSENSE_1;
    if (info->channelMask & LWBIT(channel))
    {
        offset = LWSWITCH_REG_RD32(device, _THERM, _TSENSE_U2_A_0_BJT_1_TEMPERATURE_MODIFICATIONS);
        offset = DRF_VAL(_THERM_TSENSE, _U2_A_0_BJT_1_TEMPERATURE_MODIFICATIONS, _TEMPERATURE_OFFSET, offset);
        offset = LW_TSENSE_FXP_9_5_TO_24_8(offset);

        val = LWSWITCH_REG_RD32(device, _THERM, _TSENSE_U2_A_0_BJT_1);
        val = DRF_VAL(_THERM, _TSENSE_U2_A_0_BJT_1, _TEMPERATURE, val);
        val = LW_TSENSE_FXP_9_5_TO_24_8(val);
        val -= offset;

        info->temperature[channel] = val;
        info->channelMask &= ~LWBIT(channel);
    }

    channel = LWSWITCH_THERM_CHANNEL_LS10_TSENSE_2;
    if (info->channelMask & LWBIT(channel))
    {
        offset = LWSWITCH_REG_RD32(device, _THERM, _TSENSE_U2_A_0_BJT_2_TEMPERATURE_MODIFICATIONS);
        offset = DRF_VAL(_THERM_TSENSE, _U2_A_0_BJT_2_TEMPERATURE_MODIFICATIONS, _TEMPERATURE_OFFSET, offset);
        offset = LW_TSENSE_FXP_9_5_TO_24_8(offset);

        val = LWSWITCH_REG_RD32(device, _THERM, _TSENSE_U2_A_0_BJT_2);
        val = DRF_VAL(_THERM, _TSENSE_U2_A_0_BJT_2, _TEMPERATURE, val);
        val = LW_TSENSE_FXP_9_5_TO_24_8(val);
        val -= offset;

        info->temperature[channel] = val;
        info->channelMask &= ~LWBIT(channel);
    }

    channel = LWSWITCH_THERM_CHANNEL_LS10_TSENSE_3;
    if (info->channelMask & LWBIT(channel))
    {
        offset = LWSWITCH_REG_RD32(device, _THERM, _TSENSE_U2_A_0_BJT_3_TEMPERATURE_MODIFICATIONS);
        offset = DRF_VAL(_THERM_TSENSE, _U2_A_0_BJT_3_TEMPERATURE_MODIFICATIONS, _TEMPERATURE_OFFSET, offset);
        offset = LW_TSENSE_FXP_9_5_TO_24_8(offset);

        val = LWSWITCH_REG_RD32(device, _THERM, _TSENSE_U2_A_0_BJT_3);
        val = DRF_VAL(_THERM, _TSENSE_U2_A_0_BJT_3, _TEMPERATURE, val);
        val = LW_TSENSE_FXP_9_5_TO_24_8(val);
        val -= offset;

        info->temperature[channel] = val;
        info->channelMask &= ~LWBIT(channel);
    }

    channel = LWSWITCH_THERM_CHANNEL_LS10_TSENSE_4;
    if (info->channelMask & LWBIT(channel))
    {
        offset = LWSWITCH_REG_RD32(device, _THERM, _TSENSE_U2_A_0_BJT_4_TEMPERATURE_MODIFICATIONS);
        offset = DRF_VAL(_THERM_TSENSE, _U2_A_0_BJT_4_TEMPERATURE_MODIFICATIONS, _TEMPERATURE_OFFSET, offset);
        offset = LW_TSENSE_FXP_9_5_TO_24_8(offset);

        val = LWSWITCH_REG_RD32(device, _THERM, _TSENSE_U2_A_0_BJT_4);
        val = DRF_VAL(_THERM, _TSENSE_U2_A_0_BJT_4, _TEMPERATURE, val);
        val = LW_TSENSE_FXP_9_5_TO_24_8(val);
        val -= offset;

        info->temperature[channel] = val;
        info->channelMask &= ~LWBIT(channel);
    }

    channel = LWSWITCH_THERM_CHANNEL_LS10_TSENSE_5;
    if (info->channelMask & LWBIT(channel))
    {
        offset = LWSWITCH_REG_RD32(device, _THERM, _TSENSE_U2_A_0_BJT_5_TEMPERATURE_MODIFICATIONS);
        offset = DRF_VAL(_THERM_TSENSE, _U2_A_0_BJT_5_TEMPERATURE_MODIFICATIONS, _TEMPERATURE_OFFSET, offset);
        offset = LW_TSENSE_FXP_9_5_TO_24_8(offset);

        val = LWSWITCH_REG_RD32(device, _THERM, _TSENSE_U2_A_0_BJT_5);
        val = DRF_VAL(_THERM, _TSENSE_U2_A_0_BJT_5, _TEMPERATURE, val);
        val = LW_TSENSE_FXP_9_5_TO_24_8(val);
        val -= offset;

        info->temperature[channel] = val;
        info->channelMask &= ~LWBIT(channel);
    }

    channel = LWSWITCH_THERM_CHANNEL_LS10_TSENSE_6;
    if (info->channelMask & LWBIT(channel))
    {
        offset = LWSWITCH_REG_RD32(device, _THERM, _TSENSE_U2_A_0_BJT_6_TEMPERATURE_MODIFICATIONS);
        offset = DRF_VAL(_THERM_TSENSE, _U2_A_0_BJT_6_TEMPERATURE_MODIFICATIONS, _TEMPERATURE_OFFSET, offset);
        offset = LW_TSENSE_FXP_9_5_TO_24_8(offset);

        val = LWSWITCH_REG_RD32(device, _THERM, _TSENSE_U2_A_0_BJT_6);
        val = DRF_VAL(_THERM, _TSENSE_U2_A_0_BJT_6, _TEMPERATURE, val);
        val = LW_TSENSE_FXP_9_5_TO_24_8(val);
        val -= offset;

        info->temperature[channel] = val;
        info->channelMask &= ~LWBIT(channel);
    }

    channel = LWSWITCH_THERM_CHANNEL_LS10_TSENSE_7;
    if (info->channelMask & LWBIT(channel))
    {
        offset = LWSWITCH_REG_RD32(device, _THERM, _TSENSE_U2_A_0_BJT_7_TEMPERATURE_MODIFICATIONS);
        offset = DRF_VAL(_THERM_TSENSE, _U2_A_0_BJT_7_TEMPERATURE_MODIFICATIONS, _TEMPERATURE_OFFSET, offset);
        offset = LW_TSENSE_FXP_9_5_TO_24_8(offset);

        val = LWSWITCH_REG_RD32(device, _THERM, _TSENSE_U2_A_0_BJT_7);
        val = DRF_VAL(_THERM, _TSENSE_U2_A_0_BJT_7, _TEMPERATURE, val);
        val = LW_TSENSE_FXP_9_5_TO_24_8(val);
        val -= offset;

        info->temperature[channel] = val;
        info->channelMask &= ~LWBIT(channel);
    }

    channel = LWSWITCH_THERM_CHANNEL_LS10_TSENSE_8;
    if (info->channelMask & LWBIT(channel))
    {
        offset = LWSWITCH_REG_RD32(device, _THERM, _TSENSE_U2_A_0_BJT_8_TEMPERATURE_MODIFICATIONS);
        offset = DRF_VAL(_THERM_TSENSE, _U2_A_0_BJT_8_TEMPERATURE_MODIFICATIONS, _TEMPERATURE_OFFSET, offset);
        offset = LW_TSENSE_FXP_9_5_TO_24_8(offset);

        val = LWSWITCH_REG_RD32(device, _THERM, _TSENSE_U2_A_0_BJT_8);
        val = DRF_VAL(_THERM, _TSENSE_U2_A_0_BJT_8, _TEMPERATURE, val);
        val = LW_TSENSE_FXP_9_5_TO_24_8(val);
        val -= offset;

        info->temperature[channel] = val;
        info->channelMask &= ~LWBIT(channel);
    }

    if (info->channelMask)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: ChannelMask %x absent on LS10.\n",
            __FUNCTION__, info->channelMask);

        return -LWL_BAD_ARGS;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_ctrl_therm_get_temperature_limit_ls10
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
lwswitch_monitor_thermal_alert_ls10
(
    lwswitch_device *device
)
{
    return;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
//
// lwswitch_therm_read_voltage
//
// Temperature and voltage are only available on SKUs which have thermal and
// voltage sensors.
//

LwlStatus
lwswitch_ctrl_therm_read_voltage_ls10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_GET_VOLTAGE_PARAMS *info
)
{
    LWSWITCH_PRINT(device, WARN, "%s: Function not implemented\n", __FUNCTION__);
    return LWL_SUCCESS;
}
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
