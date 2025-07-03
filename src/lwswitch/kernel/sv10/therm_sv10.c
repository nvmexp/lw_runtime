/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2020 by LWPU Corporation.  All rights reserved.  All
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
#include "sv10/sv10.h"
#include "sv10/therm_sv10.h"
#include "sv10/fuse_sv10.h"
#include "sv10/minion_sv10.h"

#include "lwswitch/svnp01/dev_pri_ringstation_sys.h"
#include "lwswitch/svnp01/dev_fuse.h"
#include "lwswitch/svnp01/dev_pmgr.h"
#include "lwswitch/svnp01/dev_lwl_ip.h"
#include "lwswitch/svnp01/dev_lwltlc_ip.h"
#include "lwswitch/svnp01/dev_minion_ip.h"
#include "lwswitch/svnp01/dev_nport_ip.h"

//
// Thermal functions
//

//
// Initialize the software state of the switch thermal info
//
// Temperature and voltage are only available on SKUs which have thermal and
// voltage sensors.
// On E3600, only the center Tdiode is attached to LWSwitch's I2C bus, via an
// ADT7461.  Voltage is readable via the INA3221 also attached to the I2C bus.
// Other SKUs may or may not have either of these sensors available.
//

LwlStatus
lwswitch_init_thermal_sv10
(
    lwswitch_device *device
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    PLWSWITCH_OBJI2C pI2c = device->pI2c;
    LwU32   i;

    // Mark everything invalid
    chip_device->thermal.tdiode_center.method = LWSWITCH_THERM_METHOD_UNKNOWN;
    chip_device->thermal.tdiode_east.method   = LWSWITCH_THERM_METHOD_UNKNOWN;
    chip_device->thermal.tdiode_west.method   = LWSWITCH_THERM_METHOD_UNKNOWN;

    chip_device->thermal.idx_i2c_dev_voltage = pI2c->device_list_size;

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    //
    // The tdiode characterization yields a set of coefficients:
    // temp = (sensors - offset)*A + B
    // to aclwrately callwlate temperature.
    //

    if ((device->board_id == LWSWITCH_BOARD_ID_E3600_A01) ||
        (device->board_id == LWSWITCH_BOARD_ID_E3600_A02))
    {
        if (pI2c->device_list_size > 0)
        {
            //
            // Tdiode-center
            // On E3600, SV10 center tdiode is attached to an ADT7461 attached to
            // WS_I2C_A.
            //
            chip_device->thermal.tdiode_center.A =  1008;    //  1.0084
            chip_device->thermal.tdiode_center.B = -1634;    // -1.6341
            chip_device->thermal.tdiode_center.offset = lwswitch_fuse_opt_read_sv10(device, LWSWITCH_FUSE_OPT_TDIODE_CENTER_SV10);

            for (i=0; i < pI2c->device_list_size; i++)
            {
                if ((pI2c->device_list[i].i2cDeviceType == LWSWITCH_I2C_DEVICE_ADT7461) ||
                    (pI2c->device_list[i].i2cDeviceType == LWSWITCH_I2C_DEVICE_ADT7473))
                {
                    chip_device->thermal.tdiode_center.method_i2c_info = &pI2c->device_list[i];
                    chip_device->thermal.tdiode_center.method = LWSWITCH_THERM_METHOD_I2C;
                    break;
                }
            }
        }

        //
        // Tdiode-east & west
        // On E3600, SV10 east & west tdiode is attached to a ADT7461s attached to
        // MLW's I2C bus.  These should be readable via the host SMBUS device as
        // subdevices 0x70, 0x71, or 0x72.
        // This access path is not lwrrently implemented in the LWSwitch kernel driver
        //

        //
        // Tdiode-east
        // Don't account for characterization data -- MLW should already handle it.
        //
        chip_device->thermal.tdiode_east.A = 1000;       //  1.000
        chip_device->thermal.tdiode_east.B = 1000;       //  1.000
        chip_device->thermal.tdiode_east.offset = 0;
        chip_device->thermal.tdiode_east.method = LWSWITCH_THERM_METHOD_MLW;
        chip_device->thermal.tdiode_east.method_i2c_info = NULL;

        //
        // Tdiode-west
        // Don't account for characterization data -- MLW should already handle it.
        //
        chip_device->thermal.tdiode_west.A = 1000;       //  1.000
        chip_device->thermal.tdiode_west.B = 1000;       //  1.000
        chip_device->thermal.tdiode_west.offset = 0;
        chip_device->thermal.tdiode_west.method = LWSWITCH_THERM_METHOD_MLW;
        chip_device->thermal.tdiode_west.method_i2c_info = NULL;

        // Find the current sensor (if present) in the I2C device list
        for (i=0; i < pI2c->device_list_size; i++)
        {
            if (pI2c->device_list[i].i2cDeviceType == LWSWITCH_I2C_DEVICE_INA3221)
            {
                chip_device->thermal.idx_i2c_dev_voltage = i;
                break;
            }
        }
    }
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    return LWL_SUCCESS;
}

//
// Read temperature
//
// Reading temperature is only available on SKUs that have sensors attached to
// LWSwitch I2C bus.
//

static LwS32
_lwswitch_therm_read_temp
(
    lwswitch_device *device,
    LwU32            position        // LWSWITCH_THERM_LOCATION_*
)
{
    LwlStatus retval = LWL_SUCCESS;
    LwS32   temperature = -1;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    switch (position)
    {
        case LWSWITCH_THERM_LOCATION_SV10_CENTER:         // Center
            if (chip_device->thermal.tdiode_center.method_i2c_info)
            {
                LWSWITCH_CTRL_I2C_INDEXED_PARAMS  i2cIndexed = {0};

                i2cIndexed.port = chip_device->thermal.tdiode_center.method_i2c_info->i2cPortLogical;
                i2cIndexed.bIsRead = LW_TRUE;
                i2cIndexed.address = (LwU16)chip_device->thermal.tdiode_center.method_i2c_info->i2cAddress;
                i2cIndexed.flags =
                    DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _START, _SEND) |
                    DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _RESTART, _SEND) |             // Write index/read data
                    DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _STOP, _SEND) |
                    DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _ADDRESS_MODE, _7BIT) |
                    DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, _ONE) |
                    DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _FLAVOR, _HW) |
                    DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, _400KHZ) |
                    DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _BLOCK_PROTOCOL, _DISABLED) |
                    DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _TRANSACTION_MODE, _NORMAL) |
                    0;
                i2cIndexed.index[0] = 0x01;        // ADT7461: Read [0x01] external temperature
                i2cIndexed.messageLength = 1;

                retval = lwswitch_ctrl_i2c_indexed(device, &i2cIndexed);
                if (retval == LWL_SUCCESS)
                {
                    temperature = ((i2cIndexed.message[ 0] - chip_device->thermal.tdiode_center.offset)*
                        chip_device->thermal.tdiode_center.A + chip_device->thermal.tdiode_center.B) / 1000;
                }
                else
                {
                    LWSWITCH_PRINT(device, ERROR,
                        "%s: Failed to read I2C thermal sensor\n",
                        __FUNCTION__);
                }
            }
            else
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: Unknown I2C thermal sensor\n",
                    __FUNCTION__);
            }
        break;

        default:
            LWSWITCH_PRINT(device, ERROR,
                "%s: Unsupported thermal sensor %d\n",
                __FUNCTION__,
                position);
    }

    return (temperature);
}

//
// lwswitch_therm_read_temperature
//
// Temperature and voltage are only available on SKUs which have thermal and
// voltage sensors.
//

LwlStatus
lwswitch_ctrl_therm_read_temperature_sv10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_GET_TEMPERATURE_PARAMS *info
)
{
    LwS32 temperature = -1;
    LwU32 channel;

    if (!info->channelMask)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: No channel given in the input.\n",
            __FUNCTION__);

        return -LWL_BAD_ARGS;
    }

    for (channel = 0; channel < LWSWITCH_NUM_CHANNELS_SV10; channel++)
    {
        if (info->channelMask & LWBIT(channel))
        {
            // Read the temperature
            temperature = _lwswitch_therm_read_temp(device, channel);

            if (temperature < 0)
            {
                info->temperature[channel] = 0;
                info->status[channel] = -LWL_IO_ERROR;
            }
            else
            {
                info->temperature[channel] = LW_TYPES_CELSIUS_TO_LW_TEMP(temperature);
                info->status[channel] = LWL_SUCCESS;
            }

            info->channelMask &= ~LWBIT(channel);
        }
    }

    if (info->channelMask)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: ChannelMask %x absent on SV10.\n",
            __FUNCTION__, info->channelMask);

        return -LWL_BAD_ARGS;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_ctrl_therm_get_temperature_limit_sv10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_GET_TEMPERATURE_LIMIT_PARAMS *info
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

//
// This function toggles link PM regardless of whether the feature has been
// disabled by regkey from general use.  Link PM is also used in the event
// a thermal warning is triggered.
//
static void
_lwswitch_force_pwr_mgmt
(
    lwswitch_device *device,
    LwU32            linkNumber,
    LwBool           bEnable
)
{
    LwlStatus       status  = LWL_SUCCESS;
    LwU32           val;

    if (bEnable)
    {
        status = lwswitch_minion_send_command_sv10(device, linkNumber,
                    LW_MINION_LWLINK_DL_CMD_COMMAND_ENABLEPM, 0);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: ENABLEPM CMD failed for link %d.\n",
                __FUNCTION__, linkNumber);
            return;
        }

        val = LWSWITCH_LINK_RD32_SV10(device, linkNumber, LWLTLC, _LWLTLC_TX, _PWRM_IC_SW_CTRL);
        val = FLD_SET_DRF(_LWLTLC_TX, _PWRM_IC_SW_CTRL, _SOFTWAREDESIRED, _LP, val);
        val = FLD_SET_DRF_NUM(_LWLTLC_TX, _PWRM_IC_SW_CTRL, _COUNTSTART, 0x1, val);
        LWSWITCH_LINK_WR32_SV10(device, linkNumber, LWLTLC, _LWLTLC_TX, _PWRM_IC_SW_CTRL, val);

        val = LWSWITCH_LINK_RD32_SV10(device, linkNumber, LWLTLC, _LWLTLC_RX, _PWRM_IC_SW_CTRL);
        val = FLD_SET_DRF(_LWLTLC_RX, _PWRM_IC_SW_CTRL, _SOFTWAREDESIRED, _LP, val);
        val = FLD_SET_DRF_NUM(_LWLTLC_RX, _PWRM_IC_SW_CTRL, _COUNTSTART, 0x1, val);
        LWSWITCH_LINK_WR32_SV10(device, linkNumber, LWLTLC, _LWLTLC_RX, _PWRM_IC_SW_CTRL, val);
    }
    else
    {
        status = lwswitch_minion_send_command_sv10(device, linkNumber,
                    LW_MINION_LWLINK_DL_CMD_COMMAND_DISABLEPM, 0);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: DISABLEPM CMD failed for link %d.\n",
                __FUNCTION__, linkNumber);
            return;
        }

        val = LWSWITCH_LINK_RD32_SV10(device, linkNumber, LWLTLC, _LWLTLC_TX, _PWRM_IC_SW_CTRL);
        val = FLD_SET_DRF(_LWLTLC_TX, _PWRM_IC_SW_CTRL, _SOFTWAREDESIRED, _FB, val);
        val = FLD_SET_DRF_NUM(_LWLTLC_TX, _PWRM_IC_SW_CTRL, _COUNTSTART, 0x0, val);
        LWSWITCH_LINK_WR32_SV10(device, linkNumber, LWLTLC, _LWLTLC_TX, _PWRM_IC_SW_CTRL, val);

        val = LWSWITCH_LINK_RD32_SV10(device, linkNumber, LWLTLC, _LWLTLC_RX, _PWRM_IC_SW_CTRL);
        val = FLD_SET_DRF(_LWLTLC_RX, _PWRM_IC_SW_CTRL, _SOFTWAREDESIRED, _FB, val);
        val = FLD_SET_DRF_NUM(_LWLTLC_RX, _PWRM_IC_SW_CTRL, _COUNTSTART, 0x0, val);
        LWSWITCH_LINK_WR32_SV10(device, linkNumber, LWLTLC, _LWLTLC_RX, _PWRM_IC_SW_CTRL, val);
    }
}

static void _lwswitch_check_link_pm_state
(
    lwswitch_device *device,
    LwU32 linkNumber
)
{
    LwU32 reg;

    reg = LWSWITCH_LINK_RD32_SV10(device, linkNumber, DLPL, _PLWL_SL0, _SLSM_STATUS_TX);

    if (!FLD_TEST_DRF(_PLWL_SL0, _SLSM_STATUS_TX, _PRIMARY_STATE, _EIGHTH, reg))
    {
        LWSWITCH_REPORT_NONFATAL_LINK(device, linkNumber, _HW_DLPL_TX_HS2LP_ERR,
                                      "Failed transition to LP mode (TX)");

        reg = LWSWITCH_LINK_RD32_SV10(device, linkNumber, LWLTLC, _LWLTLC_TX, _PWRM_IC_SW_CTRL);

        if (!FLD_TEST_DRF(_LWLTLC_TX, _PWRM_IC_SW_CTRL, _REMOTEDESIRED, _LP, reg))
        {
            LWSWITCH_PRINT(device, ERROR, "Incorrect remote-desired PM state on TX link %d.\n",
                       linkNumber);
        }
    }

    reg = LWSWITCH_LINK_RD32_SV10(device, linkNumber, DLPL, _PLWL_SL1, _SLSM_STATUS_RX);

    if (!FLD_TEST_DRF(_PLWL_SL1, _SLSM_STATUS_RX, _PRIMARY_STATE, _EIGHTH, reg))
    {
        LWSWITCH_REPORT_NONFATAL_LINK(device, linkNumber, _HW_DLPL_RX_HS2LP_ERR,
                                      "Failed transition to LP mode (RX)");

        reg = LWSWITCH_LINK_RD32_SV10(device, linkNumber, LWLTLC, _LWLTLC_RX, _PWRM_IC_SW_CTRL);

        if (!FLD_TEST_DRF(_LWLTLC_RX, _PWRM_IC_SW_CTRL, _REMOTEDESIRED, _LP, reg))
        {
            LWSWITCH_PRINT(device, ERROR, "Incorrect remote-desired PM state on RX link %d.\n",
                       linkNumber);
        }
    }
}

// Background task to monitor thermal warn and adjust link mode
void
lwswitch_monitor_thermal_alert_sv10
(
    lwswitch_device *device
)
{
    LwU32 data;
    LwU32 thermal_warn = 0;
    LwU64 time_nsec;
    LwBool target_low_power;
    LwU32 idx_link;
    LwBool is_trunk_link;
    LWSWITCH_THERMAL_ALERT_ENTRY_SV10    thermal;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    if (chip_device->thermal_alert.gpio_thermal_alert == NULL)
    {
        return;
    }

    time_nsec = lwswitch_os_get_platform_time();

    // Check THERMAL ALERT GPIO. If it changed, call each MINION
    data = LWSWITCH_REG_RD32(device, _PMGR, _GPIO_INPUT_CNTL
        (chip_device->thermal_alert.gpio_thermal_alert->hw_select));
    thermal_warn = DRF_VAL(_PMGR, _GPIO_INPUT_CNTL, _READ, data) ^
                   DRF_VAL(_PMGR, _GPIO_INPUT_CNTL, _ILW, data);

    target_low_power = !!thermal_warn;

    if (target_low_power != chip_device->thermal_alert.low_power_mode)
    {
        //
        // LWLINK2 WNF Bug  1902262 - Exiting 1/8th mode can lead to false CRC
        // errors. To reduce the chance that these will occur during thermal
        // testing, do not disengage throttling less than 1s after engagement.
        //
        if (!target_low_power &&
            chip_device->thermal_alert.time_last_change_nsec &&
            (time_nsec - chip_device->thermal_alert.time_last_change_nsec <
                LWSWITCH_INTERVAL_1SEC_IN_NS))
        {
            return;
        }

        //
        // Transition to target mode
        //
        for (idx_link=0; idx_link < LWSWITCH_NUM_LINKS_SV10; idx_link++)
        {
            if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, MINION, idx_link) ||
                !LWSWITCH_IS_LINK_ENG_VALID_SV10(device, LWLTLC, idx_link) ||
                !LWSWITCH_IS_LINK_ENG_VALID_SV10(device, DLPL, idx_link))
            {
                continue;
            }

            // force PM only on active access links
            data = LWSWITCH_LINK_RD32_SV10(device, idx_link, NPORT, _NPORT, _CTRL);
            is_trunk_link = !!DRF_VAL(_NPORT, _CTRL, _TRUNKLINKENB, data);
            data = LWSWITCH_LINK_RD32_SV10(device, idx_link, DLPL, _PLWL, _LINK_STATE);
            data = DRF_VAL(_PLWL, _LINK_STATE, _STATE, data);

            if (!is_trunk_link && (data == LW_PLWL_LINK_STATE_STATE_ACTIVE))
            {
                _lwswitch_force_pwr_mgmt(device, idx_link, target_low_power);

                if (target_low_power)
                {
                    _lwswitch_check_link_pm_state(device, idx_link);
                }
            }
        }

        chip_device->thermal_alert.low_power_mode = target_low_power;
        chip_device->thermal_alert.time_last_change_nsec = time_nsec;

        // Log the transition
        thermal.low_power_mode = target_low_power;
        thermal.time_elapsed_nsec = chip_device->thermal_alert.time_last_change_nsec ?
                                    time_nsec - chip_device->thermal_alert.time_last_change_nsec : 0;
        thermal.event_id = chip_device->thermal_alert.event_count;
        chip_device->thermal_alert.event_count++;

        if (target_low_power)
        {
            LWSWITCH_REPORT_CORRECTABLE_DEVICE_DATA(device, _HW_HOST_THERMAL_EVENT_START, &thermal,
                                                    "thermal alert: throttling enabled.");
        }
        else
        {
            LWSWITCH_REPORT_CORRECTABLE_DEVICE_DATA(device, _HW_HOST_THERMAL_EVENT_END, &thermal,
                                                    "thermal alert: throttling disabled.");
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
lwswitch_ctrl_therm_read_voltage_sv10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_GET_VOLTAGE_PARAMS *info
)
{
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS    i2cIndexed = {0};
    LwlStatus retval;
    PLWSWITCH_OBJI2C pI2c = device->pI2c;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    info->vdd_mv = 0;
    info->dvdd_mv = 0;
    info->hvdd_mv = 0;

    // Read the voltage
    if (chip_device->thermal.idx_i2c_dev_voltage < pI2c->device_list_size)
    {
        i2cIndexed.port = pI2c->device_list[chip_device->thermal.idx_i2c_dev_voltage].i2cPortLogical;
        i2cIndexed.bIsRead = LW_TRUE;
        i2cIndexed.address = (LwU16) pI2c->device_list[chip_device->thermal.idx_i2c_dev_voltage].i2cAddress;
        i2cIndexed.flags =
            DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _START, _SEND) |
            DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _RESTART, _SEND) |             // Write index/read data
            DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _STOP, _SEND) |
            DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _ADDRESS_MODE, _7BIT) |
            DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, _ONE) |
            DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _FLAVOR, _HW) |
            DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, _400KHZ) |
            DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _BLOCK_PROTOCOL, _DISABLED) |
            DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _TRANSACTION_MODE, _NORMAL) |
            0;
        i2cIndexed.messageLength = 2;

        i2cIndexed.index[0] = 0x02;        // INA3221: Read [0x02] Channel-1 Bus Voltage
        retval = lwswitch_ctrl_i2c_indexed(device, &i2cIndexed);
        if (retval == LWL_SUCCESS)
        {
            info->vdd_mv = (i2cIndexed.message[0] << 8) | i2cIndexed.message[1];
        }
        else
        {
            goto lwswitch_therm_read_voltage_exit;
        }

        i2cIndexed.index[0] = 0x04;        // INA3221: Read [0x04] Channel-2 Bus Voltage
        retval = lwswitch_ctrl_i2c_indexed(device, &i2cIndexed);
        if (retval == LWL_SUCCESS)
        {
            info->dvdd_mv = (i2cIndexed.message[0] << 8) | i2cIndexed.message[1];
        }
        else
        {
            goto lwswitch_therm_read_voltage_exit;
        }

        i2cIndexed.index[0] = 0x06;        // INA3221: Read [0x06] Channel-3 Bus Voltage
        retval = lwswitch_ctrl_i2c_indexed(device, &i2cIndexed);
        if (retval == LWL_SUCCESS)
        {
            info->hvdd_mv = (i2cIndexed.message[0] << 8) | i2cIndexed.message[1];
        }
        else
        {
            goto lwswitch_therm_read_voltage_exit;
        }
    }
    else
    {
        retval = -LWL_NOT_FOUND;
    }

lwswitch_therm_read_voltage_exit:
    return retval;
}

LwlStatus
lwswitch_ctrl_force_thermal_slowdown_sv10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_SET_THERMAL_SLOWDOWN *p
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
