/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "g_lwconfig.h"
#include "common_lwswitch.h"
#include "ls10/ls10.h"
#include "ls10/pmgr_ls10.h"
#include "error_lwswitch.h"
#include "pmgr_lwswitch.h"
#include "rom_lwswitch.h"
#include "export_lwswitch.h"
#include "lwswitch/ls10/dev_pmgr.h"

// Shared with LR10

void _lwswitch_i2c_set_port_pmgr(lwswitch_device *device, LwU32 port);

LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE lwswitch_i2c_device_allow_list_ls10[] =
{
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xA0, _CMIS4_MODULE, 
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xA2, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xA4, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xA6, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),

    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xC0, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xC2, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xC4, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xC6, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),

    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xA0, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xA2, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xA4, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xA6, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),

    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xC0, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xC2, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xC4, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xC6, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC))
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

};

const LwU32 lwswitch_i2c_device_allow_list_size_ls10 =
    LW_ARRAY_ELEMENTS(lwswitch_i2c_device_allow_list_ls10);

//
// E4840 bringup board
//
LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE lwswitch_i2c_device_list_E4840[] =
{
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CA, 0x90, _TMP451,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC))   // External TDIODE
};

const LwU32 lwswitch_i2c_device_list_E4840_size =
    LW_ARRAY_ELEMENTS(lwswitch_i2c_device_list_E4840);

//
// Pre-initialize the software & hardware state of the switch I2C & GPIO interface
//
void
lwswitch_init_pmgr_ls10
(
    lwswitch_device *device
)
{
    PLWSWITCH_OBJI2C pI2c;

    // Initialize I2C object
    lwswitch_i2c_init(device);

    pI2c = device->pI2c;

    //
    // Dynamically allocate the I2C device allowlist
    // once VBIOS table reads are implemented.
    //
    pI2c->i2c_allow_list = lwswitch_i2c_device_allow_list_ls10;
    pI2c->i2c_allow_list_size = lwswitch_i2c_device_allow_list_size_ls10;

    // Setup the 3 I2C ports
    _lwswitch_i2c_set_port_pmgr(device, LWSWITCH_I2C_PORT_I2CA);
    _lwswitch_i2c_set_port_pmgr(device, LWSWITCH_I2C_PORT_I2CB);
    _lwswitch_i2c_set_port_pmgr(device, LWSWITCH_I2C_PORT_I2CC);

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    //
    // Identify board type based on presence and absence of specific devices
    // on specific I2C busses
    //
    if ((IS_FMODEL(device) || IS_RTLSIM(device) || IS_EMULATION(device)))
    {
        device->board_id = LWSWITCH_BOARD_ID_UNKNOWN;
    }
    else
    {
        device->board_id = LWSWITCH_BOARD_ID_E4840;
    }
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

}

static const LWSWITCH_GPIO_INFO lwswitch_gpio_pin_Default[] =
{
    LWSWITCH_DESCRIBE_GPIO_PIN( 0, _INSTANCE_ID0,   0, IN),          // Instance ID bit 0
    LWSWITCH_DESCRIBE_GPIO_PIN( 1, _INSTANCE_ID1,   0, IN),          // Instance ID bit 1
    LWSWITCH_DESCRIBE_GPIO_PIN( 2, _INSTANCE_ID2,   0, IN),          // Instance ID bit 2
};

static const LwU32 lwswitch_gpio_pin_Default_size = LW_ARRAY_ELEMENTS(lwswitch_gpio_pin_Default);

//
// Initialize the software state of the switch I2C & GPIO interface
// Temporarily forcing default GPIO values.
//

// TODO: This function should be updated with the board values from DCB.

void
lwswitch_init_pmgr_devices_ls10
(
    lwswitch_device *device
)
{
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);
    PLWSWITCH_OBJI2C pI2c = device->pI2c;

    if (IS_FMODEL(device) || IS_EMULATION(device) || IS_RTLSIM(device))
    {
        // GPIOs not modelled on non-silicon
        chip_device->gpio_pin = NULL;
        chip_device->gpio_pin_size = 0;
    }
    else
    {
        chip_device->gpio_pin = lwswitch_gpio_pin_Default;
        chip_device->gpio_pin_size = lwswitch_gpio_pin_Default_size;
    }

    LWSWITCH_PRINT(device, SETUP,
        "%s: Assuming E4840 I2C configuration\n",
        __FUNCTION__);
    pI2c->device_list = lwswitch_i2c_device_list_E4840;
    pI2c->device_list_size = lwswitch_i2c_device_list_E4840_size;
}

LwlStatus
lwswitch_get_rom_info_ls10
(
    lwswitch_device *device,
    LWSWITCH_EEPROM_TYPE *eeprom
)
{
    LWSWITCH_PRINT(device, WARN, "%s: Function not implemented\n", __FUNCTION__);
    return LWL_SUCCESS;
}

/*!
 * RM Control command to determine the physical id of the device.
 */
LwU32
lwswitch_read_physical_id_ls10
(
    lwswitch_device *device
)
{
    ls10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LS10(device);
    LwU32 physical_id = 0;
    LwU32 data;
    LwU32 idx_gpio;
    LwU32 input_ilw;
    LwU32 function_offset;

    for (idx_gpio = 0; idx_gpio < chip_device->gpio_pin_size; idx_gpio++)
    {
        if ((chip_device->gpio_pin[idx_gpio].function >= LWSWITCH_GPIO_ENTRY_FUNCTION_INSTANCE_ID0) &&
            (chip_device->gpio_pin[idx_gpio].function <= LWSWITCH_GPIO_ENTRY_FUNCTION_INSTANCE_ID6))
        {
            if (chip_device->gpio_pin[idx_gpio].misc == LWSWITCH_GPIO_ENTRY_MISC_IO_ILW_IN)
            {
                input_ilw = LW_PMGR_GPIO_INPUT_CNTL_1_ILW_YES;
            }
            else
            {
                input_ilw = LW_PMGR_GPIO_INPUT_CNTL_1_ILW_NO;
            }

            LWSWITCH_REG_WR32(device, _PMGR, _GPIO_INPUT_CNTL_1,
                DRF_NUM(_PMGR, _GPIO_INPUT_CNTL_1, _PINNUM, chip_device->gpio_pin[idx_gpio].pin) |
                DRF_NUM(_PMGR, _GPIO_INPUT_CNTL_1, _ILW, input_ilw) |
                DRF_DEF(_PMGR, _GPIO_INPUT_CNTL_1, _BYPASS_FILTER, _NO));

            data = LWSWITCH_REG_RD32(device, _PMGR, _GPIO_INPUT_CNTL_1);
            function_offset = chip_device->gpio_pin[idx_gpio].function -
                          LWSWITCH_GPIO_ENTRY_FUNCTION_INSTANCE_ID0;
            physical_id |=
                (DRF_VAL(_PMGR, _GPIO_INPUT_CNTL_1, _READ, data) << function_offset);
        }
    }

    LWSWITCH_PRINT(device, SETUP, "%s Device position Id = 0x%x\n", __FUNCTION__, physical_id);

    return physical_id;
}

/*!
 * Return if I2C transactions are supported.
 *
 * @param[in] device        The LwSwitch Device.
 *
 */
LwBool
lwswitch_is_i2c_supported_ls10
(
    lwswitch_device *device
)
{
    return LW_TRUE;
}

