/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2021 by LWPU Corporation.  All rights reserved.  All
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
#include "rom_lwswitch.h"
#include "lr10/lr10.h"
#include "lr10/pmgr_lr10.h"
#include "lwswitch/lr10/dev_pmgr.h"

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
// Shared with LS10
#endif
/*
 * PMGR I2C support for LWSwitch is ported from GPU PMU I2C support
 * PMGR HW I2C support does not support the gamut of possible I2C transactions
 * thus some transactions may need to be performed directly using traditional
 * "bit-banging" of SCL & SDA.
 */

static LwlStatus _lwswitch_i2c_i2cPollHwUntilDoneOrTimeoutForStatus_lr10(lwswitch_device *device, LwU32 port);
void _lwswitch_i2c_set_port_pmgr(lwswitch_device *device, LwU32 port);

/*! The number of nanoseconds we will wait for slave clock stretching.
 *  Previously, this was set to 100us, but proved too
 *  short (see bug 630691) so was increased to 2ms.
 */
#define I2C_STRETCHED_LOW_TIMEOUT_NS_LR10 2000000

LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE lwswitch_i2c_device_allow_list_lr10[] =
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

    // Extra I2C devices for Wolf
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

    // Extra I2C devices for Wolf
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

const LwU32 lwswitch_i2c_device_allow_list_size_lr10 =
    LW_ARRAY_ELEMENTS(lwswitch_i2c_device_allow_list_lr10);

//
// PMGR functions
//

/*! 
 *  @brief Return I2c port info used in PMGR implementation.
 */
static LwU32
_lwswitch_i2c_get_port_info_lr10
(
    lwswitch_device *device,
    LwU32 port
)
{
    PLWSWITCH_OBJI2C pI2c = device->pI2c;

    if (port >= LWSWITCH_MAX_I2C_PORTS)
    {
        return 0;
    }
    else
    {
        return pI2c->PortInfo[port];
    }
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
//
// E4700 A02 bringup board
//
LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE lwswitch_i2c_device_list_E4700_A02[] =
{
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CA, 0x90, _TMP451,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC))   // External TDIODE
};

const LwU32 lwswitch_i2c_device_list_E4700_A02_size =
    LW_ARRAY_ELEMENTS(lwswitch_i2c_device_list_E4700_A02);
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

//
// Pre-initialize the software & hardware state of the switch I2C & GPIO interface
//
void
lwswitch_init_pmgr_lr10
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
    pI2c->i2c_allow_list = lwswitch_i2c_device_allow_list_lr10;
    pI2c->i2c_allow_list_size = lwswitch_i2c_device_allow_list_size_lr10;

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
        LwU32 idx_i2cdevice;
        LWSWITCH_CTRL_I2C_INDEXED_PARAMS i2c_params;

        // Initializing board Id to E4700_A02.
        device->board_id = LWSWITCH_BOARD_ID_E4700_A02;

        for (idx_i2cdevice = 0; idx_i2cdevice < lwswitch_i2c_device_list_E4700_A02_size; idx_i2cdevice++)
        {
            //
            // Test for E4700 presence by pinging the devices found on WS_I2C_A
            //
            if (lwswitch_i2c_device_list_E4700_A02[idx_i2cdevice].i2cPortLogical == LWSWITCH_I2C_PORT_I2CA)
            {
                i2c_params.port    = lwswitch_i2c_device_list_E4700_A02[idx_i2cdevice].i2cPortLogical;
                i2c_params.address = (LwU16) lwswitch_i2c_device_list_E4700_A02[idx_i2cdevice].i2cAddress;
                i2c_params.bIsRead = LW_TRUE;
                i2c_params.flags   =
                    DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _START, _SEND) |
                    DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _RESTART, _NONE) |
                    DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _STOP, _SEND) |
                    DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _ADDRESS_MODE, _7BIT) |
                    DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, _ZERO) |
                    DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _FLAVOR, _HW) |
                    DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, _400KHZ) |
                    DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _BLOCK_PROTOCOL, _DISABLED) |
                    DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _TRANSACTION_MODE, _PING);
                i2c_params.messageLength = 0;

                if (LWL_SUCCESS != lwswitch_ctrl_i2c_indexed(device, &i2c_params))
                {
                    // Failed to find the expected E4700 I2C_C devices
                    device->board_id = LWSWITCH_BOARD_ID_UNKNOWN;
                    LWSWITCH_PRINT(device, SETUP,
                        "%s: I2C failed to query for the board Id\n",
                        __FUNCTION__);
                    break;
                }
            }
        }
    }
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

}

static const LWSWITCH_GPIO_INFO lwswitch_gpio_pin_Default[] =
{
    LWSWITCH_DESCRIBE_GPIO_PIN( 0, _INSTANCE_ID0,   0, IN),          // Instance ID bit 0
    LWSWITCH_DESCRIBE_GPIO_PIN( 1, _INSTANCE_ID1,   0, IN),          // Instance ID bit 1
    LWSWITCH_DESCRIBE_GPIO_PIN( 2, _INSTANCE_ID2,   0, IN),          // Instance ID bit 2
    LWSWITCH_DESCRIBE_GPIO_PIN( 3, _INSTANCE_ID3,   0, IN),          // Instance ID bit 3
    LWSWITCH_DESCRIBE_GPIO_PIN( 4, _INSTANCE_ID4,   0, IN),          // Instance ID bit 4
    LWSWITCH_DESCRIBE_GPIO_PIN( 5, _INSTANCE_ID5,   0, IN),          // Instance ID bit 5
    LWSWITCH_DESCRIBE_GPIO_PIN( 6, _INSTANCE_ID6,   0, IN),          // Instance ID bit 6
};

static const LwU32 lwswitch_gpio_pin_Default_size = LW_ARRAY_ELEMENTS(lwswitch_gpio_pin_Default);

//
// Initialize the software state of the switch I2C & GPIO interface
// Temporarily forcing default GPIO values.
//

// TODO: This function should be updated with the board values from DCB.

void
lwswitch_init_pmgr_devices_lr10
(
    lwswitch_device *device
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    PLWSWITCH_OBJI2C pI2c = device->pI2c;

    chip_device->gpio_pin = lwswitch_gpio_pin_Default;
    chip_device->gpio_pin_size = lwswitch_gpio_pin_Default_size;

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    if (device->board_id == LWSWITCH_BOARD_ID_E4700_A02)
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s: Board E4700 identified via I2C\n",
            __FUNCTION__);
        pI2c->device_list = lwswitch_i2c_device_list_E4700_A02;
        pI2c->device_list_size = lwswitch_i2c_device_list_E4700_A02_size;

        return;
    }

    LWSWITCH_PRINT(device, SETUP,
        "%s: Unknown board ID (0x%x)\n",
        __FUNCTION__, device->board_id);
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    pI2c->device_list = NULL;
    pI2c->device_list_size = 0;
}

/*!
 * RM Control command to determine the physical id of the device.
 */
LwU32
lwswitch_read_physical_id_lr10
(
    lwswitch_device *device
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
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
 * Colwert a HW status to an LwlStatus.
 *
 * @param[in] hwStatus      The status from hardware.
 *
 * @return The pmuI2cError equivalent status.
 */
static LwlStatus
_lwswitch_i2c_i2cHwStatusToI2cStatus_lr10
(
    lwswitch_device *device,
    LwU32 hwStatus
)
{
    LwlStatus status = -LWL_ERR_GENERIC;

    if (hwStatus == LW_PMGR_I2C_CNTL_STATUS_OKAY)
    {
        status = LWL_SUCCESS;
    }
    else if (hwStatus == LW_PMGR_I2C_CNTL_STATUS_NO_ACK)
    {
        status = -LWL_NOT_FOUND;
    }
    else if (hwStatus == LW_PMGR_I2C_CNTL_STATUS_TIMEOUT)
    {
        status = -LWL_MORE_PROCESSING_REQUIRED;
    }

    return status;
}

/*!
 * Prepare next read command in sequence.
 *
 * @param[in,out]   pCmd            The command data.
 *
 * @param[in]       bytesRemaining  The number of bytes remaining.
 *
 * @return void
 */
static void
_lwswitch_i2c_i2cHwReadPrep_lr10
(
    lwswitch_device *device,
    PLWSWITCH_I2C_HW_CMD pCmd,
    LwU32       bytesRemaining
)
{
    LwU32 bytesToRead = 0;

    bytesToRead = LW_MIN(bytesRemaining,
                         LW_PMGR_I2C_CNTL_BURST_SIZE_MAXIMUM);
    pCmd->cntl = FLD_SET_DRF_NUM(_PMGR, _I2C_CNTL, _BURST_SIZE,
                                 bytesToRead, pCmd->cntl);

    if (bytesRemaining <= LW_PMGR_I2C_CNTL_BURST_SIZE_MAXIMUM)
    {
        pCmd->cntl = FLD_SET_DRF(_PMGR, _I2C_CNTL, _GEN_STOP, _YES,
                                 pCmd->cntl);
    }
}

// Reads the byte and initialize next command
static LwU32
_lwswitch_i2c_i2cHwReadNext_lr10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS *pParams,
    PLWSWITCH_I2C_HW_CMD          pCmd,
    LwBool               bIsReadNext
)
{
    LwlStatus status = -LWL_ERR_GENERIC;
    LwU32 cntl;
    LwS32 i = LW_S32_MAX;
    LwU8  messageSize;

    cntl = LWSWITCH_REG_RD32(device, _PMGR, _I2C_CNTL(pCmd->port));

    status = _lwswitch_i2c_i2cHwStatusToI2cStatus_lr10(device,
        DRF_VAL(_PMGR, _I2C_CNTL, _STATUS, cntl));
    LWSWITCH_CHECK_STATUS(device, status);
    if (status == LWL_SUCCESS)
    {
        pCmd->data = LWSWITCH_REG_RD32(device, _PMGR, _I2C_DATA(pCmd->port));

        // Migrate "data" from dword into respective bytes.
        i = LW_MIN(pCmd->bytesRemaining,
                   LW_PMGR_I2C_CNTL_BURST_SIZE_MAXIMUM) - 1;

        //
        // Block Protocol - read the size byte which should be the first byte in
        // the payload/message.
        //
        if (pCmd->bBlockProtocol)
        {
            messageSize = (pCmd->data >> (i * 8)) & 0xFF;
            if (messageSize != pParams->messageLength)
            {
                status = -LWL_BAD_ARGS;
                LWSWITCH_CHECK_STATUS(device, status);
            }

            pCmd->bBlockProtocol = LW_FALSE;
            i--;
            pCmd->bytesRemaining--;
        }


        for (; i >= 0; i--)
        {
            *pCmd->pMessage = (pCmd->data >> (i * 8)) & 0xFF;
            pCmd->pMessage++;
            pCmd->bytesRemaining--;
        }
    }

    LWSWITCH_CHECK_STATUS(device, status);

    if (status == LWL_SUCCESS)
    {
        if ((pCmd->bytesRemaining > 0) && bIsReadNext)
        {
            _lwswitch_i2c_i2cHwReadPrep_lr10(device, pCmd, pCmd->bytesRemaining);

            LWSWITCH_REG_WR32(device, _PMGR, _I2C_CNTL(pCmd->port), pCmd->cntl);
        }
        else // pCmd->bytesRemaining == 0
        {
            pCmd->status = LWL_SUCCESS;
        }
    }

    LWSWITCH_CHECK_STATUS(device, status);
    return status;
}

/*!
 * Read the SDA/SCL lines at bit position pos
 *
 * @param[in] pBus      The Bus whose lines we need to read
 *
 * @param[in] pos       The bit position of SCL/SDA lines to read
 *
 * @return 'LW_TRUE' if line is High, else 'LW_FALSE'.
 */
static LwBool
_lwswitch_i2c_i2cRead     // GM20X
(
    lwswitch_device *device,
    LWSWITCH_I2C_SW_BUS  *const pBus,
    LwU32       pos
)
{
    LwU32 reg32 = 0;

    reg32 = LWSWITCH_REG_RD32(device, _PMGR, _I2C_OVERRIDE(pBus->port));

    return (LwBool)LWSWITCH_GET_BIT(reg32, pos);
}

static LwlStatus
_lwswitch_i2c_i2cTranslateRegAddr_lr10
(
    lwswitch_device *device,
    LWSWITCH_PMGRREG_TYPE    regType,
    LwU32           port,
    LwU32           *regAddr
)
{
    LwlStatus status = (LWL_SUCCESS);

    //
    // Although the assert is using LW_PMGR_I2C_CNTL__SIZE_1, we can never have
    // a situation where the SIZE_1 is different for various registers being
    // returned by this function
    //
    LWSWITCH_ASSERT(port < LW_PMGR_I2C_CNTL__SIZE_1);

    switch (regType)
    {
        case pmgrReg_i2cAddr:
            *regAddr = LW_PMGR_I2C_ADDR(port);
            break;

        case pmgrReg_i2cCntl:
            *regAddr = LW_PMGR_I2C_CNTL(port);
            break;

        case pmgrReg_i2cTiming:
            *regAddr = LW_PMGR_I2C_TIMING(port);
            break;

        case pmgrReg_i2cOverride:
            *regAddr = LW_PMGR_I2C_OVERRIDE(port);
            break;

        case pmgrReg_i2cPoll:
            *regAddr = LW_PMGR_I2C_POLL(port);
            break;

        case pmgrReg_i2cData:
            *regAddr = LW_PMGR_I2C_DATA(port);
            break;
        default:
            LWSWITCH_PRINT(device, ERROR,
                "%s: Unsupported PMGR reg type 0x%x\n",
                __FUNCTION__,
                regType);
            status = (-LWL_BAD_ARGS);
            break;
    }

    return status;
}

//-----------------------------------------------------
// _lwswitch_i2c_i2cGetHwStatus
//
//-----------------------------------------------------
static void
_lwswitch_i2c_i2cGetHwStatus_lr10
(
    lwswitch_device *device,
    LwU32 port
)
{
#if defined(LWLINK_PRINT_ENABLED)
    LwU32       regAddr = 0;
    LwU32       regData = 0;
    LwlStatus   status  = (LWL_SUCCESS);

    status = _lwswitch_i2c_i2cTranslateRegAddr_lr10(device, pmgrReg_i2cOverride,
             port, &regAddr);
    LWSWITCH_ASSERT(status == (LWL_SUCCESS) && regAddr);
    regData = LWSWITCH_OFF_RD32(device, regAddr);
    LWSWITCH_PRINT(device, ERROR,
        "lw: LW_PMGR_I2C_OVERRIDE(%x):     0x%08x\n",
        regAddr, regData);

    status = _lwswitch_i2c_i2cTranslateRegAddr_lr10(device, pmgrReg_i2cCntl,
             port, &regAddr);
    LWSWITCH_ASSERT(status == (LWL_SUCCESS) && regAddr);
    regData = LWSWITCH_OFF_RD32(device, regAddr);
    LWSWITCH_PRINT(device, ERROR,
        "lw: LW_PMGR_I2C_CNTL(%x):         0x%08x\n",
        regAddr, regData);

    status = _lwswitch_i2c_i2cTranslateRegAddr_lr10(device, pmgrReg_i2cPoll,
             port, &regAddr);
    LWSWITCH_ASSERT(status == (LWL_SUCCESS) && regAddr);
    regData = LWSWITCH_OFF_RD32(device, regAddr);
    LWSWITCH_PRINT(device, ERROR,
        "lw: LW_PMGR_I2C_POLL(%x):         0x%08x\n",
        regAddr, regData);

    status = _lwswitch_i2c_i2cTranslateRegAddr_lr10(device, pmgrReg_i2cAddr,
             port, &regAddr);
    LWSWITCH_ASSERT(status == (LWL_SUCCESS) && regAddr);
    regData = LWSWITCH_OFF_RD32(device, regAddr);
    LWSWITCH_PRINT(device, ERROR,
        "lw: LW_PMGR_I2C_ADDR(%x):         0x%08x\n",
        regAddr, regData);

    status = _lwswitch_i2c_i2cTranslateRegAddr_lr10(device, pmgrReg_i2cData,
             port, &regAddr);
    LWSWITCH_ASSERT(status == (LWL_SUCCESS) && regAddr);
    regData = LWSWITCH_OFF_RD32(device, regAddr);
    LWSWITCH_PRINT(device, ERROR,
        "lw: LW_PMGR_I2C_DATA(%x):         0x%08x\n",
        regAddr, regData);

    status = _lwswitch_i2c_i2cTranslateRegAddr_lr10(device, pmgrReg_i2cTiming,
             port, &regAddr);
    LWSWITCH_ASSERT(status == (LWL_SUCCESS) && regAddr);
    regData = LWSWITCH_OFF_RD32(device, regAddr);
    LWSWITCH_PRINT(device, ERROR,
        "lw: LW_PMGR_I2C_TIMING(%x):       0x%08x\n",
        regAddr, regData);
#endif  //defined(LWLINK_PRINT_ENABLED)
}

/*! @brief Poll the HW CNTL register until the operation completes, the HW
 *         times out, or the SW backup times out.
 *
 *  The base worst-case timeout being used was callwlated based on the longest
 *  possible hardware command.  The base is then modified using the max clock
 *  stretch added with the normal cycle period.  At the time of this writing,
 *  the longest command was determined to be:
 *
 *  START ADDRESS|W REGISTER START ADDRESS|R/W DATA*4 STOP
 *    1      9         9       1        9        27     1   = 57 cycles
 */
static LwU32
_lwswitch_i2c_i2cPollHwUntilDoneOrTimeout_lr10
(
    lwswitch_device *device,
    LwU32   port
)
{
    LwU32       cntl = 0;
    LWSWITCH_TIMEOUT   timeout;
    LwU32       cntlReg  = 0;
    LwlStatus   status   = LWL_SUCCESS;
    LwU64       timeout_ns = I2C_HW_IDLE_TIMEOUT_NS;

    status = _lwswitch_i2c_i2cTranslateRegAddr_lr10(device,
                pmgrReg_i2cCntl, port, &cntlReg);
    LWSWITCH_ASSERT(status == LWL_SUCCESS && cntlReg);

    // Increase timeout on emulation
    if (IS_EMULATION(device))
    {
        timeout_ns *= 1000;
    }

    // Set up SW timeout, which is used as a back up in case HW timeout fails.
    lwswitch_timeout_create(timeout_ns, &timeout);

    // Poll HW until done or SW/HW timeout is indicated.
    cntl = LWSWITCH_OFF_RD32(device, cntlReg);

    // Tone down the MMIO debug spew across this polling loop
    while((!FLD_TEST_DRF(_PMGR, _I2C_CNTL, _CYCLE, _DONE, cntl)) &&
          (!FLD_TEST_DRF(_PMGR, _I2C_CNTL, _STATUS, _TIMEOUT, cntl)) &&
          (!lwswitch_timeout_check(&timeout)))
    {
        cntl = LWSWITCH_OFF_RD32(device, cntlReg);
    }

    // SW timeout is a back up - we shouldn't actually ever reach it.

    //
    // Report a timeout if we hit one.
    //
    if (FLD_TEST_DRF(_PMGR, _I2C_CNTL, _STATUS, _TIMEOUT, cntl) ||
        FLD_TEST_DRF(_PMGR, _I2C_CNTL, _STATUS, _BUS_BUSY, cntl) ||
        lwswitch_timeout_check(&timeout))
    {
        LWSWITCH_PRINT(device, ERROR,
            "Timeout waiting for CYCLE to be DONE on port: %d with CNTL: 0x%x\n",
             port, cntl);
        _lwswitch_i2c_i2cGetHwStatus_lr10(device, port);
    }

    return cntl;
}

static LwlStatus
_lwswitch_i2c_i2cPollHwUntilDoneOrTimeoutForStatus_lr10
(
    lwswitch_device *device,
    LwU32   port
)
{
    LwU32 cntl;
    LwlStatus status = -LWL_ERR_GENERIC;

    cntl = _lwswitch_i2c_i2cPollHwUntilDoneOrTimeout_lr10(device, port);
    switch (DRF_VAL(_PMGR, _I2C_CNTL, _STATUS, cntl))
    {
        case LW_PMGR_I2C_CNTL_STATUS_OKAY:
            status = LWL_SUCCESS;
            break;
        case LW_PMGR_I2C_CNTL_STATUS_NO_ACK:
            // Odd return value, but consistent with LW04.  TODO.
            status = -LWL_BAD_ARGS;
            break;
        case LW_PMGR_I2C_CNTL_STATUS_TIMEOUT:
        case LW_PMGR_I2C_CNTL_STATUS_BUS_BUSY:
            status = -LWL_ERR_GENERIC;
            break;
        default:
            LWSWITCH_PRINT(device, ERROR,
                "%s: Unknown I2C control status (0x%x)\n",
                __FUNCTION__, cntl);
            break;
    }
    return status;
}

/*!
 * Reset the HW engine and wait for it to idle.
 *
 * @param[in] port            The port identifier.
 *
 * @return LWL_SUCCESS
 *     upon success
 *
 * @return -LWL_ERR_ILWALID_STATE
 *     if the timeout is exceeded.
 */
static LwlStatus
_lwswitch_i2c_i2cHwReset_lr10
(
    lwswitch_device *device,
    LwU32 port
)
{
    LwU32 cntl = DRF_DEF(_PMGR, _I2C_CNTL, _CMD, _RESET);

    LWSWITCH_REG_WR32(device, _PMGR, _I2C_CNTL(port), cntl);

    return _lwswitch_i2c_i2cPollHwUntilDoneOrTimeoutForStatus_lr10(device, port);
}

/*
 * @LWSWITCH_OBJI2C Locking Mechanism
 *
 * The value I2CAcquired represents which component holds control of the
 * contained I2C lines, with '0' representing no component (unlocked).
 * All callers must check that no other component holds the lock, and hold
 * the lock themselves before performing operations on the bus.
 *
 * As not all callers correctly do this, the key "level 3" entry points into
 * LWSWITCH_OBJI2C which perform operations on the bus will automatically lock and
 * unlock LWSWITCH_OBJI2C.  The lock is only appropriate for full-transactions, so
 * only level 3 functions (and higher) should lock internally.  In the future,
 * access to lower level functions without holding the lock may be disabled.
 * Furthermore, external should not depend on the automatic locking
 * mechanisms, as they are considered temporary until there is enough time to
 * correct the offending components.
 *
 * Note: This methodology is not thread-safe.  It is expected to be protected
 * by the big RM lock, so contention is not lwrrently expected.  However,
 * locking is increasingly necessary as the RM bit-banging interface is
 * coming into contention with the HW I2C controller and the PMU, which can
 * perform offloaded operations for both the RM and VBIOS.  Further work will
 * be necessary for this locking to protect RM collisions if the RM lock is
 * dismantled.
 *
 * Note: This locking mechanism may not be called, even by the same owner,
 * while already being held.  Undefined behavior may result.
 *
 * @param[in] pI2c        LWSWITCH_OBJI2C pointer
 * @param[in] logicalPort The logical port which is to be acquired
 * @param[in] acquirer
 *     The ID of the component attempting to acquire the lock/mutex.  Special
 *     value @ref LWSWITCH_I2C_ACQUIRER_NONE is used to release the lock/mutex.
 * @param[in] pTimeout    RMTIMEOUT pointer
 *
 * @returns (LWL_SUCCESS)    if the component successfully acquired the lock.
 *          (-LWL_ERR_STATE_IN_USE) if the attempt to acquire lock failed.
 */
static LwlStatus
_lwswitch_i2c_i2cSetAcquired_lr10
(
    lwswitch_device *device,
    LwU32      logicalPort,
    LwU32      acquirer,
    LWSWITCH_TIMEOUT *pTimeout
)
{
    PLWSWITCH_OBJI2C pI2c = device->pI2c;
    LwlStatus status = LWL_SUCCESS;

    if (LWSWITCH_I2C_ACQUIRER_NONE != acquirer)
    {
        //
        // Ensure that not already acquired.  I2C mutex does not support
        // relwrsive acquires!
        //
        if (LWSWITCH_I2C_ACQUIRER_NONE != pI2c->I2CAcquired)
        {
            LWSWITCH_PRINT(device, MMIO,
                "%s: I2C mutex already acquired when attempting to acquire - I2CAcquired=%d.\n",
                __FUNCTION__, pI2c->I2CAcquired);
            status = (-LWL_ERR_STATE_IN_USE);
            goto _lwswitch_i2c_i2cSetAcquired_exit;
        }

        // Set the LWSWITCH_OBJI2C SW state
        pI2c->I2CAcquired = acquirer;
    }
    else
    {
        //
        // Ensure that I2C is acquired.  I2C does not support un-matched
        // releases.
        //
        if (LWSWITCH_I2C_ACQUIRER_NONE == pI2c->I2CAcquired)
        {
            LWSWITCH_PRINT(device, MMIO,
                "%s: I2C mutex not acquired when attempting to release - I2CAcquired=%d.\n",
                __FUNCTION__, pI2c->I2CAcquired);
            goto _lwswitch_i2c_i2cSetAcquired_exit;
        }

        // Set LWSWITCH_OBJI2C SW state
        pI2c->I2CAcquired = acquirer;
    }

_lwswitch_i2c_i2cSetAcquired_exit:
    return status;
}

/*!
 * Initialize the I2C lines
 *
 * @param[in] pBus      The Bus whose lines we need to initialize
 */
static void
_lwswitch_i2c_i2cInitSwi2c_lr10        // GM20X
(
    LWSWITCH_I2C_SW_BUS *const pBus
)
{
    pBus->sclOut = DRF_BASE(LW_PMGR_I2C_OVERRIDE_SCL);
    pBus->sdaOut = DRF_BASE(LW_PMGR_I2C_OVERRIDE_SDA);

    pBus->sclIn = DRF_BASE(LW_PMGR_I2C_OVERRIDE_SCLPAD_IN);
    pBus->sdaIn = DRF_BASE(LW_PMGR_I2C_OVERRIDE_SDAPAD_IN);

    pBus->regCache = 0;
    pBus->regCache = LWSWITCH_SET_BIT(pBus->regCache, pBus->sclOut);
    pBus->regCache = LWSWITCH_SET_BIT(pBus->regCache, pBus->sdaOut);
    pBus->regCache = LWSWITCH_SET_BIT(pBus->regCache, DRF_BASE(LW_PMGR_I2C_OVERRIDE_SIGNALS));
    pBus->lwrLine  = pBus->sclIn;
};

/*!
 * Drive the SCL/SDA pin defined by pos bit to High/Low as per bValue
 *
 * @param[in] pBus     The Bus whose lines we need to drive
 *
 * @param[in] pos      The bit position of SCL/SDA lines
 *
 * @param[in] bValue   Whether to drive lines High/Low
 */
static void
_lwswitch_i2c_i2cDrive_lr10        // GM20X
(
    lwswitch_device *device,
    LWSWITCH_I2C_SW_BUS *const pBus,
    LwU32   pos,
    LwBool  bValue
)
{
    pBus->regCache  = FLD_SET_DRF(_PMGR, _I2C_OVERRIDE, _SIGNALS, _ENABLE, pBus->regCache);
    pBus->regCache  = bValue ? LWSWITCH_SET_BIT(pBus->regCache, pos) :
                               LWSWITCH_CLEAR_BIT(pBus->regCache, pos);

    LWSWITCH_REG_WR32(device, _PMGR, _I2C_OVERRIDE(pBus->port), pBus->regCache);
}

/*!
 * Drive the SCL/SDA pin defined by pos bit to High/Low as per bValue
 *
 * @param[in] pBus      The Bus whose lines we need to drive
 *
 * @param[in] pos       The bit position of SCL/SDA lines
 *
 * @param[in] bValue    Whether to drive lines High/Low
 */
static void
_lwswitch_i2c_i2cDriveAndDelay_lr10
(
    lwswitch_device *device,
    LWSWITCH_I2C_SW_BUS *const pBus,
    LwU32             pos,
    LwBool            bValue,
    LwU32             delayNs
)
{
    _lwswitch_i2c_i2cDrive_lr10(device, pBus, pos, bValue);

    if (delayNs != 0)
    {
        LWSWITCH_I2C_DELAY(delayNs);
    }
}

/*!
 * @brief   Check the physical status of a line for a particular I2C bus.
 *
 * @param[in] pLine
 *          An integer representing an I2C line as determined previously from
 *          i2cGetScl(port) or i2cGetSda(port).  Note that pLine is really an
 *          integer, but since this is used as a callback it must meet the
 *          function type specification.
 *
 * @returns The state of the I2C line, with LW_FALSE representing low and
 *          LW_TRUE representing high.  The type is determined by the functional
 *          type specification of the callback for which this function is used.
 */

static LwBool
_lwswitch_i2c_i2cIsLineHigh_lr10
(
   lwswitch_device *device,
   LWSWITCH_I2C_SW_BUS *pBus
)
{
    LwBool status = LW_TRUE;

    if(pBus->lwrLine == pBus->sclIn)
    {
        status = LWSWITCH_I2C_READ(pBus, pBus->sclIn);
    }
    else if(pBus->lwrLine == pBus->sdaIn)
    {
        status = LWSWITCH_I2C_READ(pBus, pBus->sdaIn);
    }
    else
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Unknown I2C line(%d)\n",
            __FUNCTION__, pBus->lwrLine);
    }

    return status;
}

/*!
 * @brief   Send an initial start signal along the bus.
 *
 * @param[in] pBus
 *          The bus along which to send the start signal.
 *
 * @returns LWL_SUCCESS
 *
 * @section Preconditions
 *  - The bus must not be busy.
 *  - Both SCL and SDA must be high.
 *  - If the most recent bus activity was a stop signal, then the sum of tR and
 *    tBuf must have elapsed since that event.
 *
 * @section Postconditions
 *  - A start signal will have been sent along the bus.
 *  - SCL will be high and SDA will be low.
 *  - At least tF + tHdSta will have elapsed since the start signal was sent.
 */
static LwlStatus
_lwswitch_i2c_i2cSendStart_lr10
(
    lwswitch_device *device,
    LWSWITCH_I2C_SW_BUS *const pBus
)
{
    // A start signal is to pull down SDA while SCL is high.
    _lwswitch_i2c_i2cDriveAndDelay_lr10(device, pBus, pBus->sdaOut, LWSWITCH_LOW, pBus->tF + pBus->tHdSta);

    return LWL_SUCCESS;
}

/*!
 * @brief   Wait for SCL to be released by the slave.
 *
 * @param[in]   pBus    The bus information.
 *
 * @return  LW_TRUE     if waiting for SCL to get released timed out
 * @return  LW_FALSE    if SCL got released by the slave
 */
static LwBool
_lwswitch_i2c_i2cWaitSclReleasedTimedOut_lr10
(
    lwswitch_device *device,
    LWSWITCH_I2C_SW_BUS *pBus
)
{
    LWSWITCH_TIMEOUT   timeout;
    LwBool line_high = LW_FALSE;
    LwBool timed_out = LW_FALSE;
    LwU32 iter_count = 0;
    LwU64 timeout_ns = I2C_STRETCHED_LOW_TIMEOUT_NS_LR10;

    // Increase timeout on emulation
    if (IS_EMULATION(device))
    {
        timeout_ns *= 1000;
    }

    lwswitch_timeout_create(timeout_ns, &timeout);
    do
    {
        timed_out = lwswitch_timeout_check(&timeout);
        line_high = _lwswitch_i2c_i2cIsLineHigh_lr10(device, pBus);
        iter_count++;
    }
    while ((!line_high) && (!timed_out));

    if (!line_high)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Timeout waiting for SCL release\n",
            __FUNCTION__);
    }

    return (!line_high);
}

/*!
 * @brief   Send an restart signal along the bus.
 *
 * @param[in] pBus
 *          The bus along which to send the restart signal.
 *
 * @return  LWL_SUCCESS
 *          The operation completed successfully.
 *
 * @return  -LWL_IO_ERROR
 *          Clock stretching from the slave took too long and the operation
 *          aborted.  The bus is no longer in a valid state.
 *
 * @section Preconditions
 *  - SCL must be high.
 *  - The most recent bus activity must have been a byte + ack transfer.
 *  - The sum of tR and tHigh must have elapsed since the most recent event.
 *
 * @section Postconditions
 *
 * If LWL_SUCCESS were to be returned:
 *  - A restart signal will have been sent along the bus.
 *  - SCL will be high and SDA will be low.
 *  - At least tF + tHdSta elapsed since the start signal was sent.
 *
 * If -LWL_IO_ERROR were to be returned:
 *  - No postconditions; the bus will not be in a valid state.
 */
static LwlStatus
_lwswitch_i2c_i2cSendRestart_lr10
(
    lwswitch_device *device,
    LWSWITCH_I2C_SW_BUS *const pBus
)
{
    //
    // Prior to a restart, we need to prepare the bus.  After the next clock
    // cycle, we need both SCL and SDA to be high for tR + tSuSta.
    //

    _lwswitch_i2c_i2cDriveAndDelay_lr10(device, pBus, pBus->sclOut, LWSWITCH_LOW, pBus->tF);

    _lwswitch_i2c_i2cDriveAndDelay_lr10(device, pBus, pBus->sdaOut, LWSWITCH_HIGH, pBus->tR + pBus->tLow);

    _lwswitch_i2c_i2cDriveAndDelay_lr10(device, pBus, pBus->sclOut, LWSWITCH_HIGH, pBus->tR);

    // Wait for SCL to be released by the slave.
    if (_lwswitch_i2c_i2cWaitSclReleasedTimedOut_lr10(device, pBus))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: _lwswitch_i2c_i2cWaitSclReleasedTimedOut\n", __FUNCTION__);
        return -LWL_IO_ERROR;

    }
    LWSWITCH_I2C_DELAY(pBus->tSuSta);

    // A start signal is to pull down SDA while SCL is high.
    _lwswitch_i2c_i2cDriveAndDelay_lr10(device, pBus, pBus->sdaOut, LWSWITCH_LOW, pBus->tF + pBus->tHdSta);

    return LWL_SUCCESS;
}


/*!
 * @brief   Send a stop signal along the bus.
 *
 * @param[in] pBus
 *          The bus along which to send the stop signal.
 *
 * @return  LWL_SUCCESS
 *          The operation completed successfully.
 *
 * @return  -LWL_IO_ERROR
 *          Clock stretching from the slave took too long and the operation
 *          aborted.  The bus is no longer in a valid state.
 *
 * @section Preconditions
 *  - SCL must be high.
 *  - The most recent bus activity must have been a byte + ack transfer.
 *  - The sum of tR and tHigh must have elapsed since SCL was released (by both
 *    master and slave) for the latest ack bit.
 *
 * @section Postconditions
 *
 * If LWL_SUCCESS were to be returned, then:
 *  - A stop signal will have been sent along the bus.
 *  - Both SCL and SDA will be high.
 *  - At least tR + tBuf time will have elapsed.
 *
 * If -LWL_IO_ERROR were to be returned, then:
 *  - No postconditions; the bus will not be in a valid state.
 */
static LwlStatus
_lwswitch_i2c_i2cSendStop_lr10
(
    lwswitch_device *device,
    LWSWITCH_I2C_SW_BUS *const pBus
)
{
    LwlStatus  error = LWL_SUCCESS;

    //
    // Prior to a stop, we need to prepare the bus.  After the next clock
    // cycle, we need SCL high and SDA low for tR + tSuSto.
    //

    _lwswitch_i2c_i2cDriveAndDelay_lr10(device, pBus, pBus->sclOut, LWSWITCH_LOW, pBus->tF + pBus->tHdDat);

    // delay for SDA going low (tF) and additional tLow to guarantee
    // that the length for SCL being low isn't shorter than required
    // from the spec, but can subtract the time already spent with
    // the clock low (tHdDat).
    _lwswitch_i2c_i2cDriveAndDelay_lr10(device, pBus, pBus->sdaOut, LWSWITCH_LOW, pBus->tF + pBus->tLow - pBus->tHdDat);

    _lwswitch_i2c_i2cDriveAndDelay_lr10(device, pBus, pBus->sclOut, LWSWITCH_HIGH, pBus->tR);

    // Wait for SCL to be released by the slave.
    if (_lwswitch_i2c_i2cWaitSclReleasedTimedOut_lr10(device, pBus))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: _lwswitch_i2c_i2cWaitSclReleasedTimedOut\n", __FUNCTION__);
        // can't return early, bus must get back to a valid state
        error = -LWL_IO_ERROR;
    }
    LWSWITCH_I2C_DELAY(pBus->tSuSto);

    // A stop signal is to release SDA while SCL is high.
    _lwswitch_i2c_i2cDriveAndDelay_lr10(device, pBus, pBus->sdaOut, LWSWITCH_HIGH, pBus->tR + pBus->tBuf);

    return error;
}

/*!
 * @brief   Read/Write a byte from/to the bus.
 *
 * @param[in] pBus
 *          The bus along which to write the byte of data.
 *
 * @param[in] pData
 *      RD: The result of the read.  Written only if the full byte is read.
 *      WR: Pointer to the byte of data to transfer.  This is a pointer to have
 *          the same function type as _lwswitch_i2c_i2cReadByte.
 *
 * @param[in] bLast
 *      RD: Whether this is the last byte in a transfer.  The last byte should
 *          be followed by a nack instead of an ack.
 *      WR: Unused parameter to keep the same function type as _lwswitch_i2c_i2cReadByte.
 *
 * @param[in] bIsRead
 *          Defines the direction of the transfer.
 *
 * @return  LWL_SUCCESS
 *          The operation completed successfully.
 *
 * @return  -LWL_IO_ERROR
 *          Clock stretching from the slave took too long and the operation
 *          aborted.  The bus is no longer in a valid state.
 *
 * @return  -LWL_ERR_GENERIC
 *          The slave did not acknowledge the byte transfer.
 *
 * @section Preconditions
 *  - SCL must be high.
 *  - The most recent bus activity must have been either a start or a byte +
 *    ack transfer.
 *  - If the most recent bus activity was a start, then tF + tHdSta must have
 *    elapsed.
 *  - If the most recent bus activity was a byte + ack transfer, then tR +
 *    tHigh must have elapsed.
 *
 * @section Postconditions (RD)
 *
 * If LWL_SUCCESS were to be returned, then:
 *  - The byte will have been read and stored in pData.
 *  - If pLast was true, then a nack will have been sent; if false, an ack
 *    will have been sent.
 *  - SCL will be high and SDA will be low.
 *
 * If -LWL_IO_ERROR were to be returned, then:
 *  - The bus will not be in a valid state.
 *
 * @section Postconditions (WR)
 *
 * Regardless of return value:
 *  - The data contained in pData will remain unchanged.
 *
 * If LWL_SUCCESS were to be returned, then:
 *  - The byte will have been transferred and the ack made by the slave.
 *  - SCL will be high and SDA will be low..
 *
 * If -LWL_ERR_GENERIC were to be returned, then:
 *  - The byte will have been transferred and a nack will have been received
 *    from the slave.
 *  - Both SCL and SDA will be high.
 *
 * If -LWL_IO_ERROR were to be returned, then:
 *  - The bus will not be in a valid state.
 */
static LwlStatus
_lwswitch_i2c_i2cProcessByte_lr10
(
          lwswitch_device *device,
          LWSWITCH_I2C_SW_BUS *const pBus,
          LwU8       *const pData,
    const LwBool            bLast,
          LwBool            bIsRead
)
{
    LwU8 data;
    LwU8 i;

    data = bIsRead ? 0 : *pData;

    for (i = 0; i < LWSWITCH_BITS_PER_BYTE; i++)
    {
        // Read/Write and store each bit with the most significant coming first.
        if (bIsRead)
        {
            _lwswitch_i2c_i2cDriveAndDelay_lr10(device, pBus, pBus->sclOut, LWSWITCH_LOW,
                pBus->tF + pBus->tSuDat);

            // Don't mask the transmission.
            _lwswitch_i2c_i2cDriveAndDelay_lr10(device, pBus, pBus->sdaOut, LWSWITCH_HIGH,
                pBus->tR + pBus->tHdDat);
        }
        else
        {
            _lwswitch_i2c_i2cDriveAndDelay_lr10(device, pBus, pBus->sclOut, LWSWITCH_LOW,
                pBus->tF + pBus->tHdDat);

            _lwswitch_i2c_i2cDriveAndDelay_lr10(device, pBus, pBus->sdaOut,
                ((data << i) & 0x80) ? LWSWITCH_HIGH : LWSWITCH_LOW, pBus->tR + pBus->tSuDat);
        }

        // Note that here we do NOT have any delay for Write.
        _lwswitch_i2c_i2cDriveAndDelay_lr10(device, pBus, pBus->sclOut, LWSWITCH_HIGH, bIsRead ? pBus->tR : 0);

        // Wait for SCL to be released by the slave.
        if (_lwswitch_i2c_i2cWaitSclReleasedTimedOut_lr10(device, pBus))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: _lwswitch_i2c_i2cWaitSclReleasedTimedOut\n", __FUNCTION__);
            if (!bIsRead)
            {
                _lwswitch_i2c_i2cDriveAndDelay_lr10(device, pBus, pBus->sdaOut, LWSWITCH_HIGH,
                    pBus->tR + pBus->tSuDat);
            }

            return -LWL_IO_ERROR;
        }
        LWSWITCH_I2C_DELAY(pBus->tHigh);

        if (bIsRead)
        {
            // Read the data from the slave.
            data <<= 1;
            data |= LWSWITCH_I2C_READ(pBus, pBus->sdaIn);
        }
    }

    // Read the acknowledge bit from the slave, where low indicates an ack.
    _lwswitch_i2c_i2cDriveAndDelay_lr10(device, pBus, pBus->sclOut, LWSWITCH_LOW, pBus->tF + pBus->tHdDat);

    // Release SDA so as not to interfere with the slave's transmission.
    _lwswitch_i2c_i2cDriveAndDelay_lr10(device, pBus, pBus->sdaOut, bIsRead ? (bLast ? LWSWITCH_HIGH : LWSWITCH_LOW) : LWSWITCH_HIGH,
        pBus->tR + pBus->tSuDat);

    _lwswitch_i2c_i2cDriveAndDelay_lr10(device, pBus, pBus->sclOut, LWSWITCH_HIGH, pBus->tR);

    // Wait for SCL to be released by the slave.
    if (_lwswitch_i2c_i2cWaitSclReleasedTimedOut_lr10(device, pBus))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: _lwswitch_i2c_i2cWaitSclReleasedTimedOut\n", __FUNCTION__);
        if (bIsRead)
        {
            _lwswitch_i2c_i2cDriveAndDelay_lr10(device, pBus, pBus->sdaOut, LWSWITCH_HIGH,
                pBus->tR + pBus->tSuDat);
        }
        return -LWL_IO_ERROR;
    }
    LWSWITCH_I2C_DELAY(pBus->tHigh);

    if (bIsRead)
    {
        *pData = data;
        return LWL_SUCCESS;
    }
    else
    {
        LwlStatus retval;
        retval = LWSWITCH_I2C_READ(pBus, pBus->sdaIn) ? -LWL_ERR_GENERIC : LWL_SUCCESS;
        LWSWITCH_CHECK_STATUS(device, retval);
        return retval;
    }
}

static LwlStatus
_lwswitch_i2c_i2cWriteByte_lr10
(
          lwswitch_device *device,
          LWSWITCH_I2C_SW_BUS *const pBus,
          LwU8       *const pData,
    const LwBool            bIgnored
)
{
    return _lwswitch_i2c_i2cProcessByte_lr10(device, pBus, pData, bIgnored, LW_FALSE);
}

/*!
 * @brief   Read a byte from the I2C bus.
 *
 * @param[in] pBus
 *          The bus along which to write the byte of data.
 *
 * @param[out] pData
 *          The result of the read.  Written only if the full byte is read.
 *
 * @param[in] bLast
 *          Whether this is the last byte in a transfer.  The last byte should
 *          be followed by a nack instead of an ack.
 *
 * @return  LWL_SUCCESS
 *          The operation completed successfully.
 *
 * @return  -LWL_IO_ERROR
 *          Clock stretching from the slave took too long and the operation
 *          aborted.  The bus is no longer in a valid state.
 *
 * @section Preconditions
 *  - SCL must be high.
 *  - The most recent bus activity must have been either a start or a byte +
 *    ack transfer.
 *  - If the most recent bus activity was a start, then tF + tHdSta must have
 *    elapsed.
 *  - If the most recent bus activity was a byte + ack transfer, then tR +
 *    tHigh must have elapsed.
 *
 * @section Postconditions
 *
 * If LWL_SUCCESS were to be returned, then:
 *  - The byte will have been read and stored in pData.
 *  - If pLast was true, then a nack will have been sent; if false, an ack
 *    will have been sent.
 *  - SCL will be high and SDA will be low.
 *
 * If -LWL_IO_ERROR were to be returned, then:
 *  - The bus will not be in a valid state.
 */
static LwlStatus
_lwswitch_i2c_i2cReadByte_lr10
(
          lwswitch_device *device,
          LWSWITCH_I2C_SW_BUS *const pBus,
          LwU8       *const pData,
    const LwBool            bLast
)
{
    return _lwswitch_i2c_i2cProcessByte_lr10(device, pBus, pData, bLast, LW_TRUE);
}

/*!
 * Macro to set fast mode timing
 */
#define I2C_TIMING_FAST_LR10(timing, speed)  (((timing) * 400) / (speed))

/*!
 * @brief   Determine the worst-case delay needed between line transitions.
 *
 * @param[out] pBus Fill in the delay timings.
 * @param[in] speedMode
 *          The bitrate for communication.  See @ref rmdpucmdif.h "the command
 *          interface" for more details.
 * @param[in] port
 *          Port ID of the port.
 */
static void
_lwswitch_i2c_i2cBusAndSpeedInit_lr10
(
    lwswitch_device *device,
    LWSWITCH_I2C_SW_BUS     *pBus,
    const LwU8      speedMode,
    LwU32           port
)
{
    _lwswitch_i2c_i2cInitSwi2c_lr10(pBus);
    pBus->port = port;

    if (speedMode == LWSWITCH_CTRL_I2C_FLAGS_SPEED_MODE_100KHZ)
    {
        pBus->tF     = LWSWITCH_I2C_PROFILE_STANDARD_tF;
        pBus->tR     = LWSWITCH_I2C_PROFILE_STANDARD_tR;
        pBus->tSuDat = LWSWITCH_I2C_PROFILE_STANDARD_tSUDAT;
        pBus->tHdDat = LWSWITCH_I2C_PROFILE_STANDARD_tHDDAT;
        pBus->tHigh  = LWSWITCH_I2C_PROFILE_STANDARD_tHIGH;
        pBus->tSuSto = LWSWITCH_I2C_PROFILE_STANDARD_tSUSTO;
        pBus->tHdSta = LWSWITCH_I2C_PROFILE_STANDARD_tHDSTA;
        pBus->tSuSta = LWSWITCH_I2C_PROFILE_STANDARD_tSUSTA;
        pBus->tBuf   = LWSWITCH_I2C_PROFILE_STANDARD_tBUF;
        pBus->tLow   = LWSWITCH_I2C_PROFILE_STANDARD_tLOW;
    }
    else if (speedMode == LWSWITCH_CTRL_I2C_FLAGS_SPEED_MODE_200KHZ)
    {
        pBus->tF     = I2C_TIMING_FAST_LR10(LWSWITCH_I2C_PROFILE_FAST_tF, 200);
        pBus->tR     = I2C_TIMING_FAST_LR10(LWSWITCH_I2C_PROFILE_FAST_tR, 200);
        pBus->tSuDat = I2C_TIMING_FAST_LR10(LWSWITCH_I2C_PROFILE_FAST_tSUDAT, 200);
        pBus->tHdDat = I2C_TIMING_FAST_LR10(LWSWITCH_I2C_PROFILE_FAST_tHDDAT, 200);
        pBus->tHigh  = I2C_TIMING_FAST_LR10(LWSWITCH_I2C_PROFILE_FAST_tHIGH, 200);
        pBus->tSuSto = I2C_TIMING_FAST_LR10(LWSWITCH_I2C_PROFILE_FAST_tSUSTO, 200);
        pBus->tHdSta = I2C_TIMING_FAST_LR10(LWSWITCH_I2C_PROFILE_FAST_tHDSTA, 200);
        pBus->tSuSta = I2C_TIMING_FAST_LR10(LWSWITCH_I2C_PROFILE_FAST_tSUSTA, 200);
        pBus->tBuf   = I2C_TIMING_FAST_LR10(LWSWITCH_I2C_PROFILE_FAST_tBUF, 200);
        pBus->tLow   = I2C_TIMING_FAST_LR10(LWSWITCH_I2C_PROFILE_FAST_tLOW, 200);
    }
    else if (speedMode == LWSWITCH_CTRL_I2C_FLAGS_SPEED_MODE_300KHZ)
    {
        pBus->tF     = I2C_TIMING_FAST_LR10(LWSWITCH_I2C_PROFILE_FAST_tF, 300);
        pBus->tR     = I2C_TIMING_FAST_LR10(LWSWITCH_I2C_PROFILE_FAST_tR, 300);
        pBus->tSuDat = I2C_TIMING_FAST_LR10(LWSWITCH_I2C_PROFILE_FAST_tSUDAT, 300);
        pBus->tHdDat = I2C_TIMING_FAST_LR10(LWSWITCH_I2C_PROFILE_FAST_tHDDAT, 300);
        pBus->tHigh  = I2C_TIMING_FAST_LR10(LWSWITCH_I2C_PROFILE_FAST_tHIGH, 300);
        pBus->tSuSto = I2C_TIMING_FAST_LR10(LWSWITCH_I2C_PROFILE_FAST_tSUSTO, 300);
        pBus->tHdSta = I2C_TIMING_FAST_LR10(LWSWITCH_I2C_PROFILE_FAST_tHDSTA, 300);
        pBus->tSuSta = I2C_TIMING_FAST_LR10(LWSWITCH_I2C_PROFILE_FAST_tSUSTA, 300);
        pBus->tBuf   = I2C_TIMING_FAST_LR10(LWSWITCH_I2C_PROFILE_FAST_tBUF, 300);
        pBus->tLow   = I2C_TIMING_FAST_LR10(LWSWITCH_I2C_PROFILE_FAST_tLOW, 300);
    }
    else if (speedMode == LWSWITCH_CTRL_I2C_FLAGS_SPEED_MODE_400KHZ)
    {
        pBus->tF     = LWSWITCH_I2C_PROFILE_FAST_tF;
        pBus->tR     = LWSWITCH_I2C_PROFILE_FAST_tR;
        pBus->tSuDat = LWSWITCH_I2C_PROFILE_FAST_tSUDAT;
        pBus->tHdDat = LWSWITCH_I2C_PROFILE_FAST_tHDDAT;
        pBus->tHigh  = LWSWITCH_I2C_PROFILE_FAST_tHIGH;
        pBus->tSuSto = LWSWITCH_I2C_PROFILE_FAST_tSUSTO;
        pBus->tHdSta = LWSWITCH_I2C_PROFILE_FAST_tHDSTA;
        pBus->tSuSta = LWSWITCH_I2C_PROFILE_FAST_tSUSTA;
        pBus->tBuf   = LWSWITCH_I2C_PROFILE_FAST_tBUF;
        pBus->tLow   = LWSWITCH_I2C_PROFILE_FAST_tLOW;
    }
    else
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: unsupported I2C speed.  Defaulting to 100kHz\n",
            __FUNCTION__);

        pBus->tF     = LWSWITCH_I2C_PROFILE_STANDARD_tF;
        pBus->tR     = LWSWITCH_I2C_PROFILE_STANDARD_tR;
        pBus->tSuDat = LWSWITCH_I2C_PROFILE_STANDARD_tSUDAT;
        pBus->tHdDat = LWSWITCH_I2C_PROFILE_STANDARD_tHDDAT;
        pBus->tHigh  = LWSWITCH_I2C_PROFILE_STANDARD_tHIGH;
        pBus->tSuSto = LWSWITCH_I2C_PROFILE_STANDARD_tSUSTO;
        pBus->tHdSta = LWSWITCH_I2C_PROFILE_STANDARD_tHDSTA;
        pBus->tSuSta = LWSWITCH_I2C_PROFILE_STANDARD_tSUSTA;
        pBus->tBuf   = LWSWITCH_I2C_PROFILE_STANDARD_tBUF;
        pBus->tLow   = LWSWITCH_I2C_PROFILE_STANDARD_tLOW;
    }
}

/*!
 * Confirm that we are in the correct state to allow operation.
 *
 * @param[in]       device      LwSwitch Device.
 * @param[in]       port        port Id
 *
 * @return (LWL_SUCCESS), if the state is okay.
 * @return (-LWL_BAD_ARGS), if the arguments are invalid.
 * @return (-LWL_ERR_ILWALID_STATE), if the state does not allow the command.
 */
static LwlStatus
_lwswitch_i2c_indexed_check_state_lr10
(
    lwswitch_device *device,
    LwU32 port
)
{
    LwlStatus status;
    LWSWITCH_CTRL_I2C_GET_PORT_INFO_PARAMS port_info;

    status = lwswitch_ctrl_i2c_get_port_info(device, &port_info);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_CHECK_STATUS(device, status);
        return status;
    }

    if (!FLD_TEST_DRF(_I2C, _PORTINFO, _DEFINED, _PRESENT,
        port_info.info[port]))
    {
        LWSWITCH_PRINT(device, INFO,
            "%s: Invalid port access %d.\n",
            __FUNCTION__, port);
        return (-LWL_BAD_ARGS);
    }

    return (LWL_SUCCESS);
}

/*!
 * Check the validity of the RM Control command for I2C_INDEXED.
 *
 * @param[in]       pParams         The RM Control input to check.
 *
 * @return (LWL_SUCCESS),
 *      if the arguments are valid.
 *
 * @return (-LWL_BAD_ARGS),
 *      in most cases where the arguments are invalid.
 *
 * @return (-LWL_ERR_GENERIC),
 *      if a field reserved for future use is used.
 */
static LwlStatus
_lwswitch_ctrl_i2c_indexedCheckPreconditions_lr10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS *pParams
)
{
    LwlStatus status;

    if (pParams->port >= LWSWITCH_MAX_I2C_PORTS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Port %d out-of-range.\n", __FUNCTION__, pParams->port);
        return (-LWL_BAD_ARGS);
    }

    if (DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _RESERVED, pParams->flags) != 0)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Flags not recognized in 0x%x.\n", __FUNCTION__,
            pParams->flags);
        return (-LWL_ERR_GENERIC);
    }

    status = _lwswitch_i2c_indexed_check_state_lr10(device, pParams->port);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_CHECK_STATUS(device, status)
        return status;
    }
    if (pParams->address & LWSWITCH_I2C_READCYCLE)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Address %d invalid (r/w bit set).\n", __FUNCTION__,
            pParams->address);
        return (-LWL_BAD_ARGS);
    }
    if (FLD_TEST_DRF(SWITCH_CTRL, _I2C_FLAGS, _TRANSACTION_MODE, _PING, pParams->flags))
    {
        if ((!FLD_TEST_DRF(SWITCH_CTRL, _I2C_FLAGS, _START,        _SEND, pParams->flags)) ||
            (!FLD_TEST_DRF(SWITCH_CTRL, _I2C_FLAGS, _RESTART,      _NONE, pParams->flags)) ||
            (!FLD_TEST_DRF(SWITCH_CTRL, _I2C_FLAGS, _STOP,         _SEND, pParams->flags)) ||
            ( FLD_TEST_DRF(SWITCH_CTRL, _I2C_FLAGS, _ADDRESS_MODE, _NO_ADDRESS, pParams->flags)) ||
            (!FLD_TEST_DRF(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, _ZERO, pParams->flags)) ||
            (pParams->messageLength != 0))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Invalid PING arguments\n",
                __FUNCTION__);
            return (-LWL_BAD_ARGS);
        }
    }

    switch (DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _ADDRESS_MODE, pParams->flags))
    {
        case LWSWITCH_CTRL_I2C_FLAGS_ADDRESS_MODE_7BIT:
            if (!LWSWITCH_I2C_IS_7BIT_I2C_ADDRESS(pParams->address))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: Address 0x%x not in 7-bit address space.\n",
                    __FUNCTION__, pParams->address);
                return (-LWL_BAD_ARGS);
            }
            break;

        case LWSWITCH_CTRL_I2C_FLAGS_ADDRESS_MODE_10BIT:
            if (!LWSWITCH_I2C_IS_10BIT_I2C_ADDRESS(pParams->address))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: Address 0x%x not in 10-bit address space.\n",
                    __FUNCTION__, pParams->address);
                return (-LWL_BAD_ARGS);
            }
            break;

        case LWSWITCH_CTRL_I2C_FLAGS_ADDRESS_MODE_NO_ADDRESS:
            break;

        default:
            // Unreachable.
            LWSWITCH_PRINT(device, ERROR,
                "%s: Address mode %d not valid.\n", __FUNCTION__,
                DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _ADDRESS_MODE, pParams->flags));
            return (-LWL_BAD_ARGS);
    }
    switch (DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, pParams->flags))
    {
        case LWSWITCH_CTRL_I2C_FLAGS_SPEED_MODE_100KHZ:
        case LWSWITCH_CTRL_I2C_FLAGS_SPEED_MODE_200KHZ:
        case LWSWITCH_CTRL_I2C_FLAGS_SPEED_MODE_300KHZ:
        case LWSWITCH_CTRL_I2C_FLAGS_SPEED_MODE_400KHZ:
        case LWSWITCH_CTRL_I2C_FLAGS_SPEED_MODE_1000KHZ:
        case LWSWITCH_CTRL_I2C_FLAGS_SPEED_MODE_DEFAULT:
            break;

        default:
            LWSWITCH_PRINT(device, ERROR,
                "%s: Speed mode %d not valid.\n", __FUNCTION__,
                DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, pParams->flags));
            return (-LWL_ERR_GENERIC);
    }

    if (DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, pParams->flags) > LWSWITCH_CTRL_I2C_INDEX_LENGTH_MAX)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Index length %d exceeds max %d.\n", __FUNCTION__,
            DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, pParams->flags),
            LWSWITCH_CTRL_I2C_INDEX_LENGTH_MAX);
        return (-LWL_BAD_ARGS);
    }
    if (pParams->messageLength > LWSWITCH_CTRL_I2C_MESSAGE_LENGTH_MAX)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Message length %d exceeds max %d.\n", __FUNCTION__,
            pParams->messageLength, LWSWITCH_CTRL_I2C_MESSAGE_LENGTH_MAX);
        return (-LWL_BAD_ARGS);
    }
    return (LWL_SUCCESS);
}

/*!
 * Attempts to lock the I2C bus on LWSWITCH. This is a non-blocking 
 * call.
 *
 * @param[in]       port        The port which is to be acquired
 * @param[in]       acquirer    The identifier of the caller.
 *
 * @return (LWL_SUCCESS),
 *         if the lock was acquired.
 *
 * @return (-LWL_BAD_ARGS),
 *         if any of the pointer arguments are NULL.
 *
 * @return (-LWL_ERR_STATE_IN_USE),
 *         if the lock is already acquired by another client of pI2c.
 */
static LwlStatus
_lwswitch_i2c_LockObjI2c_lr10
(
    lwswitch_device *device,
    LwU32   port,
    LwU32 acquirer
)
{
    if ((device == NULL) || (device->pI2c == NULL))
    {
        return (-LWL_BAD_ARGS);
    }

    acquirer = acquirer ? acquirer : LWSWITCH_I2C_ACQUIRER_UNKNOWN;

    if ((LWL_SUCCESS) != _lwswitch_i2c_i2cSetAcquired_lr10(device, port, acquirer, NULL))
    {
        return (-LWL_ERR_STATE_IN_USE);
    }

    return (LWL_SUCCESS);
}

/*!
 * Unlocks the I2C bus on LWSWITCH.
 *
 * @param[in]       port    The port which is to be unlocked.
 *
 * @return void
 */
static void
_lwswitch_i2c_UnlockObjI2c_lr10
(
    lwswitch_device *device,
    LwU32   port
)
{
    _lwswitch_i2c_i2cSetAcquired_lr10(device, port, LWSWITCH_I2C_ACQUIRER_NONE, NULL);
    return;
}

/* ------------------------- Public Functions ------------------------------- */
/*!
 * @brief Is the bus associated with 'port' ready for a bus transaction?
 *
 * Check to see if SCL and SDA is lwrrently high (ie. not being driven low).
 * If either (or both) lines are low, consider the bus as "busy" and therefore
 * not ready to accept the next transaction.
 *
 * @param[in]  port  The physical port for the bus to consider
 *
 * @return LW_TRUE
 *     Both SCL and SDA are high. The master may commence the next transaction.
 *
 * @return LW_FALSE
 *     One (or both) of the I2C lines is being pulled/driven low.
 */
static LwBool
_lwswitch_i2c_i2cIsBusReady_lr10       // GM20X
(
    lwswitch_device *device,
    LwU32 port
)
{
    LwU32 reg32 = 0;

    reg32 = LWSWITCH_REG_RD32(device, _PMGR, _I2C_OVERRIDE(port));

    // Ensure that neither line is being driven low (by master or slave(s))
    return FLD_TEST_DRF(_PMGR, _I2C_OVERRIDE, _SCLPAD_IN, _ONE, reg32) &&
           FLD_TEST_DRF(_PMGR, _I2C_OVERRIDE, _SDAPAD_IN, _ONE, reg32);
}

/*!
 * Restore the previous operating mode.
 *
 * @param[in] port    The port identifier.
 *
 * @param[in] bWasBb    The previous operating mode.
 *                      (was it bitbanging i.e SwI2C ?)
 */
static void
_lwswitch_i2c_i2cRestoreMode_lr10      // GM20X
(
    lwswitch_device *device,
    LwU32 port,
    LwU32 bWasBb
)
{
    LwU32 override = DRF_DEF(_PMGR, _I2C_OVERRIDE, _PMU, _DISABLE) |
                     DRF_DEF(_PMGR, _I2C_OVERRIDE, _SCL, _OUT_ONE) |
                     DRF_DEF(_PMGR, _I2C_OVERRIDE, _SDA, _OUT_ONE);

    if (bWasBb)
    {
        override = FLD_SET_DRF(_PMGR, _I2C_OVERRIDE, _SIGNALS, _ENABLE,
                               override);
    }
    else // !bWasBb
    {
        override = FLD_SET_DRF(_PMGR, _I2C_OVERRIDE, _SIGNALS, _DISABLE,
                               override);
    }

    LWSWITCH_REG_WR32(device, _PMGR, _I2C_OVERRIDE(port), override);
}

/*!
 * Set the new operating mode, and return the previous mode.
 *
 * @param[in]  port  The port identifier.
 * @param[in]  bSwI2c
 *    The target mode (LW_TRUE = bit-bang, LW_FALSE = HWI2C).
 *
 * @param[out] pBWasSwI2c
 *    Optional (may be NULL) pointer written with the current I2C operating-
 *    mode when requested (LW_TRUE = SW bit-bang). Ignored when NULL.
 *
 * @return 'LWL_SUCCESS'
 *      if the operation completed successfully.
 *
 * @return 'i2cErrorBusy'
 *      if the operation timed out waiting for the HW controller.
 *
 * @return 'i2cErrorBusy'
 *      if the current mode cannot be exited safely.
 *
 * @note
 *     This function does not deal with HW polling (which is broken, see bug
 *     671708) or stereo vision.
 *
 * @note
 *     This function does not check the current bus state before setting the
 *     mode. In cases where it matters, the caller is responsible for ensuring
 *     the bus is in a good state before calling this function.
 */
static LwlStatus
_lwswitch_i2c_i2cSetMode_lr10      // GM20X
(
    lwswitch_device *device,
    LwU32   port,
    LwBool  bSwI2c,
    LwBool  *pBWasSwI2c
)
{
    LwU32 override;

    // if requested, return/save the current operating-mode
    if (pBWasSwI2c != NULL)
    {
        LwU32 i2c_override = LWSWITCH_REG_RD32(device, _PMGR, _I2C_OVERRIDE(port));

        *pBWasSwI2c = FLD_TEST_DRF(_PMGR, _I2C_OVERRIDE, _SIGNALS, _ENABLE,
            i2c_override);
    }

    //
    // Set the I2C override register for the port to allow the FLCN to master
    // the bus when SW bit-banging mode is requested. Make sure it is disabled
    // when HWI2C mode is requested.
    //
    override = DRF_DEF(_PMGR, _I2C_OVERRIDE, _PMU    , _DISABLE) |
               DRF_DEF(_PMGR, _I2C_OVERRIDE, _SDA    , _OUT_ONE) |
               DRF_DEF(_PMGR, _I2C_OVERRIDE, _SCL    , _OUT_ONE);

    if(bSwI2c)
    {
        override = FLD_SET_DRF(_PMGR, _I2C_OVERRIDE, _SIGNALS, _ENABLE , override);
    }
    else
    {
        override = FLD_SET_DRF(_PMGR, _I2C_OVERRIDE, _SIGNALS, _DISABLE, override);
    }
    LWSWITCH_REG_WR32(device, _PMGR, _I2C_OVERRIDE(port), override);

    //
    // When entering HWI2C-mode, we need to reset the controller.  Since we
    // don't necessarily do so every time we enter HWI2C-mode, reset it
    // regardless of the previous mode.
    //
    if (!bSwI2c)
    {
        return _lwswitch_i2c_i2cHwReset_lr10(device, port);
    }

    return LWL_SUCCESS;
}

/*!
 * @brief Recover the i2c bus from an unexpected HW-state.
 *
 * This function is to be called when the target I2C bus for a particular
 * transaction is not in an expected state. This is lwrrently limited to cases
 * where a slave device is perpetually pulling SDA low for some unknown reason.
 * It may however be extended in the future to cover various other unexpected
 * bus states. A "recovered" bus is defined by the state where both SCL and SDA
 * are released (pulled high).
 *
 * @param[in] pTa
 *     The transaction information. Used by this function for port and speed
 *     information.
 *
 * @return LWL_SUCCESS
 *     Both SCL and SDA are pulled high. There is no contention on the bus.
 *
 * @return i2cErrorBusIlwalid
 *     The bus remains in a bad/unusable state, even after the recovery
 *     attempt. Recovery failed.
 */
static LwlStatus
_lwswitch_i2c_i2cRecoverBusViaSw_lr10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS *pParams
)
{
    LWSWITCH_I2C_SW_BUS  bus;
    LwU8        junk;
    LwU32       flags  = pParams->flags;
    LwU32       port = pParams->port;   // DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _PORT, flags);

    _lwswitch_i2c_i2cBusAndSpeedInit_lr10(device,
        &bus, DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, flags), port);

    //
    // In all observed cases where recovery is necessary, performing a single-
    // byte read followed by sending a stop bit is sufficient to recover the
    // bus. It may be required to define a more elaborate recovery sequence in
    // the future where multiple recovery steps are performed and are perhaps
    // even based on the current "wedged" bus state.
    //
    _lwswitch_i2c_i2cReadByte_lr10(device, &bus, &junk, LW_TRUE);
    _lwswitch_i2c_i2cSendStop_lr10(device, &bus);

    return _lwswitch_i2c_i2cIsBusReady_lr10(device, port) ?
        LWL_SUCCESS : -LWL_ERR_ILWALID_STATE;
}

/*!
 * @brief   Send an address (if any) using the transaction's specified format.
 *
 * @param[in] pTa
 *          The transaction information, including the addressMode and address.
 *
 * @param[in] pBus
 *          The bus information.
 *
 * @param[in] bIsRead
 *          Whether to send a read or write bit.  Note that this is not
 *          necessarily the direction of the transaction, as certain preludes
 *          (for example, sending the index) may need to be in a different
 *          direction.
 *
 * @return  LWL_SUCCESS
 *          The address was sent successfully, or none was specified.
 *
 * @return  -LWL_ERR_GENERIC
 *          No device acknowledged the address.
 *
 * @return  -LWL_BAD_ARGS
 *          The addressMode within pTa was not a valid value so no address was
 *          sent.
 *
 * @return  -LWL_IO_ERROR
 *          Clock stretching from the slave took too long and the operation
 *          aborted.  The bus is no longer in a valid state.
 *
 * @section Preconditions
 *  - SCL must be high.
 *  - The most recent bus activity must have been a start or restart signal.
 *  - The sum tF + tHdSta must have elapsed since the last (re)start.
 *
 * @section Postconditions
 *
 * If LWL_SUCCESS were to be returned, then:
 *  - If addressMode were to indicate to send an address, then the address will
 *    have been sent and acknowledged correctly.
 *  - If addressMode were to indicate to do nothing, then no bus operation will
 *    have been performed.
 *  - Both SCL and SDA will be high.
 *
 * If -LWL_ERR_GENERIC were to be returned, then:
 *  - The address will have been sent correctly but no ack was received.
 *  - SCL will be high and SDA will be low.
 *
 * If -LWL_BAD_ARGS were to be returned, then:
 *  - The pTa->addressMode field will have been invalid.
 *  - No bus operation will have been performed.
 *
 * If dpuI2cErrorTimeout were to be returned, then:
 *  - The bus will not be in a valid state.
 */
static LwlStatus
_lwswitch_i2c_i2cSendAddress_lr10
(
    lwswitch_device *device,
            LWSWITCH_I2C_SW_BUS      *const  pBus,
            LWSWITCH_CTRL_I2C_INDEXED_PARAMS *pTa,
    const   LwBool                  bIsRead
)
{
    LwlStatus error   = LWL_SUCCESS;
    LwU32     address = pTa->address;
    LwU8      byte;

    switch (DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _ADDRESS_MODE, pTa->flags))
    {
        case LWSWITCH_CTRL_I2C_FLAGS_ADDRESS_MODE_NO_ADDRESS:
        {
            break;
        }

        case LWSWITCH_CTRL_I2C_FLAGS_ADDRESS_MODE_7BIT:
        {
            //
            // The address + R/W bit transfer for 7-bit addressing is simply
            // the address followed by the R/W bit.  Lwrrently the addresses
            // passed by the RM are already shifted by one to the left.
            //
            byte  = (LwU8)(address | bIsRead);
            error = _lwswitch_i2c_i2cWriteByte_lr10(device, pBus, &byte, LW_FALSE);
            break;
        }

        case LWSWITCH_CTRL_I2C_FLAGS_ADDRESS_MODE_10BIT:
        {
            //
            // The address + R/W bit transfer is a bit odd in 10-bit
            // addressing and the format is dependent on the R/W bit itself.
            // Due to backwards compatibility, the first byte is b11110xx0,
            // with 'xx' being the two most significant address bits.
            //
            // Technically the least significant bit indicates a write, but
            // must always be so for a ten-bit address.  If the R/W bit
            // actually is a write, then transmission simply continues after
            // the address.  If the R/W bit actually is a read, then a _second_
            // address transmission follows.  This second address is a resend
            // of the first byte of the 10-bit address with a read bit.
            //
            // If this is confusing (it is,) I highly recommend reading the
            // the specification.
            //

            //
            // First, transfer the address using the 10-bit format.
            //
            byte  = LWSWITCH_GET_ADDRESS_10BIT_FIRST(address);
            error = _lwswitch_i2c_i2cWriteByte_lr10(device, pBus, &byte, LW_FALSE);

            if (error == LWL_SUCCESS)
            {
                byte  = LWSWITCH_GET_ADDRESS_10BIT_SECOND(address);
                error = _lwswitch_i2c_i2cWriteByte_lr10(device, pBus, &byte, LW_FALSE);

                if (error == LWL_SUCCESS)
                {
                    //
                    // Now, if the transaction is a read, we send a restart and then
                    // the first byte with a read bit.
                    //
                    if (bIsRead)
                    {
                        error = _lwswitch_i2c_i2cSendRestart_lr10(device, pBus);

                        if (error == LWL_SUCCESS)
                        {
                            byte  = LWSWITCH_GET_ADDRESS_10BIT_FIRST(address) | 1;
                            error = _lwswitch_i2c_i2cWriteByte_lr10(device, pBus, &byte, LW_FALSE);
                        }
                    }
                }
            }
            break;
        }
        default:
        {
            error = -LWL_BAD_ARGS;
            break;
        }
    }

    return error;
}

/*!
 * @brief   Perform a generic I2C transaction using the SW bitbanging controls
 *          of the GPU.
 *
 * @param[in, out] pTa
 *          Transaction data; please see @ref ./inc/task4_i2c.h "task4_i2c.h".
 *
 * @return  LWL_SUCCESS
 *          The transaction completed successfully.
 *
 * @return  i2cErrorNackAddress
 *          No device acknowledged the address.
 *
 * @return  -LWL_ERR_GENERIC
 *          No device acknowledged one of the message bytes.
 *
 * @return  i2cErrorInput
 *          The transaciton pTa was detected to be invalid.
 *
 * @return  i2cErrorSclTimeout
 *          Clock stretching from the slave took too long and the transaction
 *          aborted.  The bus is no longer in a valid state.
 */
static LwlStatus
_lwswitch_i2c_i2cPerformTransactiolwiaSw_lr10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS *pTa
)
{
    LWSWITCH_I2C_SW_BUS bus;
    LwlStatus           error = LWL_SUCCESS;
    LwlStatus           error2;
    LwU8                junk;
    LwU16               i;
    LwU32               flags = pTa->flags;
    LwU32               indexLength = DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH,
                                            flags);
    LwU8                messageSize = (LwU8)pTa->messageLength;

    _lwswitch_i2c_i2cBusAndSpeedInit_lr10(device, &bus, DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, flags), pTa->port);

    //
    // If the command says we should begin with a start signal, send it.
    //
    if (FLD_TEST_DRF(SWITCH_CTRL, _I2C_FLAGS, _START, _SEND, flags))
    {
        error = _lwswitch_i2c_i2cSendStart_lr10(device, &bus);
        LWSWITCH_CHECK_STATUS(device, error);
    }

    //
    // Send the address, if any.  The direction depends on whether or not we're
    // sending an index.  If we are, then we need to write the index.  If not,
    // then use the direction of the transaction.
    //
    if (error == LWL_SUCCESS)
    {
        error = _lwswitch_i2c_i2cSendAddress_lr10(device, &bus, pTa, pTa->bIsRead &&
                DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, flags) == 0);
        LWSWITCH_CHECK_STATUS(device, error);
    }

    //
    // Send the indices, if any.
    //
    if ((error == LWL_SUCCESS) && indexLength)
    {
        // Send the first index, if any.
        for (i = 0; (error == LWL_SUCCESS) && (i < indexLength); i++)
        {
            error = _lwswitch_i2c_i2cWriteByte_lr10(device, &bus, &pTa->index[i], LW_FALSE);
            LWSWITCH_CHECK_STATUS(device, error);
        }

        // If a restart is needed between phrases, send it.
        if ((error == LWL_SUCCESS) &&
            FLD_TEST_DRF(SWITCH_CTRL, _I2C_FLAGS, _RESTART, _SEND, flags))
        {
            error = _lwswitch_i2c_i2cSendRestart_lr10(device, &bus);
            LWSWITCH_CHECK_STATUS(device, error);

            //
            // We know we are sending the message next, so send the message
            // direction.
            //
            if (error == LWL_SUCCESS)
            {
                error = _lwswitch_i2c_i2cSendAddress_lr10(device, &bus, pTa, pTa->bIsRead);
                LWSWITCH_CHECK_STATUS(device, error);
            }
        }
    }

    // If a block transaction read/write the size first!
    if ((error == LWL_SUCCESS) &&
         FLD_TEST_DRF(SWITCH_CTRL, _I2C_FLAGS, _BLOCK_PROTOCOL, _ENABLED, flags))
    {
        error = _lwswitch_i2c_i2cProcessByte_lr10(device, &bus, &messageSize, LW_FALSE, pTa->bIsRead);
        LWSWITCH_CHECK_STATUS(device, error);

        // If a read, ensure that device returned the expected size!
        if ((error == LWL_SUCCESS) && (pTa->bIsRead) &&
            (messageSize != pTa->messageLength))
        {
            error = -LWL_BAD_ARGS;
            LWSWITCH_CHECK_STATUS(device, error);
        }
    }

    //
    // Perform the main transaction.
    //
    for (i = 0; (error == LWL_SUCCESS) && (i < pTa->messageLength);
         i++)
    {
        // The last data byte requires special handling
        error = _lwswitch_i2c_i2cProcessByte_lr10(device, &bus, &pTa->message[i],
                                (i == (pTa->messageLength - 1)), pTa->bIsRead);
        LWSWITCH_CHECK_STATUS(device, error);
    }

    LWSWITCH_CHECK_STATUS(device, error);
    //
    // Bug 719104 : The target device may NACK the last byte written as
    // per I2C standard protocol. It should not be treated as an error.
    //
    if ((error == -LWL_ERR_GENERIC) && !pTa->bIsRead &&
        (pTa->messageLength != 0) && (i == pTa->messageLength))
    {
        error = LWL_SUCCESS;
    }

    //
    // Always send a stop, even if the command does not specify or we had an
    // earlier failure.  We must clean up after ourselves!
    //
    error2 = _lwswitch_i2c_i2cSendStop_lr10(device, &bus);
    LWSWITCH_CHECK_STATUS(device, error2);

    //
    // Bug 785366: Verify that the i2c bus is in a valid state. The data
    // line should be high.
    //

    bus.lwrLine = bus.sdaIn;
    if (!_lwswitch_i2c_i2cIsLineHigh_lr10(device, &bus))
    {
         bus.lwrLine = bus.sclIn;
        // another device may be pulling the line low.

        // clock the bus - by reading in a byte.
        (void)_lwswitch_i2c_i2cReadByte_lr10(device, &bus, &junk, LW_TRUE);

        // Attempt the Stop again.
        error2 = _lwswitch_i2c_i2cSendStop_lr10(device, &bus);
        LWSWITCH_CHECK_STATUS(device, error2);
    }

    LWSWITCH_CHECK_STATUS(device, error);
    LWSWITCH_CHECK_STATUS(device, error2);
    // Don't let a stop failure mask an earlier one.
    if (error == LWL_SUCCESS)
    {
        error = error2;
    }

    return error;
}

static LwlStatus
_lwswitch_i2c_i2cHwReadChunks_LongIndex_lr10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS *pParams,
    PLWSWITCH_I2C_HW_CMD          pCmd,
    LwBool               bSendStop
)
{
    LwU32   addr           = 0;
    LwU32   offset;
    LwBool  bSendAddress   = LW_FALSE;
    LwS32   index_size     = DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, pParams->flags);
    LwU32   burst_size;
    LwS32   i;

    //
    // The PMGR HW I2C state machine has a limited set of possible state
    // transitions.  Unfortunately, device sub-indexes of >1 byte can not be
    // handled as a streaming read using RAB.  So this function instead iterates, sending
    // the multi-byte index and reading a 4-byte burst.
    //

    pCmd->pMessage = pParams->message;

    switch (DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _ADDRESS_MODE, pParams->flags))
    {
        default:
        case LWSWITCH_CTRL_I2C_FLAGS_ADDRESS_MODE_NO_ADDRESS:
            bSendAddress = LW_FALSE;
            break;

        case LWSWITCH_CTRL_I2C_FLAGS_ADDRESS_MODE_7BIT:
            bSendAddress = LW_TRUE;
            addr = DRF_NUM(_PMGR, _I2C_ADDR, _DAB, pParams->address >> 1) |
                   DRF_DEF(_PMGR, _I2C_ADDR, _TEN_BIT, _DISABLE);
            break;

        case LWSWITCH_CTRL_I2C_FLAGS_ADDRESS_MODE_10BIT:
            bSendAddress = LW_TRUE;
            addr = DRF_NUM(_PMGR, _I2C_ADDR, _DAB, pParams->address >> 1) |
                   DRF_DEF(_PMGR, _I2C_ADDR, _TEN_BIT, _ENABLE);
            break;
    }

    pCmd->port = pParams->port;
    pCmd->data = 0;
    pCmd->bytesRemaining = pParams->messageLength;
    pCmd->status = LWL_SUCCESS;
    pCmd->pMessage = pParams->message;
    pCmd->bBlockProtocol = FLD_TEST_DRF(SWITCH_CTRL, _I2C_FLAGS, _BLOCK_PROTOCOL, _ENABLED, pParams->flags);

    if (pCmd->bBlockProtocol)
    {
        pCmd->bytesRemaining++;
    }

    //
    // Caveat emptor: This code assumes a big-endian addressing scheme on the
    // target device, where the high address bits are sent in the first byte
    // and the low address bits in the last byte.
    // This appears to be a fair assumption, based on reading some EEPROM specs,
    // which are the only devices this code is expected to work on.
    //

    // index[] is assumed big-endian.  Swizzle it for little-endian offset math.
    offset = 0;
    for (i = 0; i < index_size; i++)
    {
        offset |= pParams->index[i] << (8*(index_size-1 - i));
    }

    while ((pCmd->status == LWL_SUCCESS) && (pCmd->bytesRemaining > 0))
    {
        if (bSendAddress)
        {
            pCmd->bRead = LW_FALSE;

            //
            // I2C_WRITE_WAIT
            // .------------------------------------.
            // |S| dab[7:1]  |wr|A| data |A| data |A|
            // `------------------------------------'
            //
            pCmd->data = 0;
            for (i = 0; i < index_size; i++)
            {
                ((LwU8 *) &pCmd->data)[i] |= (offset >> (8*(index_size-1 - i))) & 0xFF;
            }

            pCmd->cntl =
                DRF_DEF(_PMGR, _I2C_CNTL, _GEN_RAB, _NO)         |
                DRF_NUM(_PMGR, _I2C_CNTL, _BURST_SIZE,
                    DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, pParams->flags))    |
                DRF_DEF(_PMGR, _I2C_CNTL, _CMD, _WRITE)          |
                DRF_DEF(_PMGR, _I2C_CNTL, _GEN_START, _YES)      |
                DRF_DEF(_PMGR, _I2C_CNTL, _GEN_STOP, _NO)        |
                DRF_DEF(_PMGR, _I2C_CNTL, _INTR_WHEN_DONE, _YES) |
                DRF_DEF(_PMGR, _I2C_CNTL, _GEN_NACK, _NO);

            LWSWITCH_REG_WR32(device, _PMGR, _I2C_ADDR(pCmd->port), addr);
            LWSWITCH_REG_WR32(device, _PMGR, _I2C_DATA(pCmd->port), pCmd->data);
            LWSWITCH_REG_WR32(device, _PMGR, _I2C_CNTL(pCmd->port), pCmd->cntl);
        }

        pCmd->status = _lwswitch_i2c_i2cPollHwUntilDoneOrTimeoutForStatus_lr10(device, pCmd->port);
        LWSWITCH_CHECK_STATUS(device, pCmd->status);
        if (pCmd->status == LWL_SUCCESS)
        {
            //
            // LWSWITCH_I2C_READ
            // .----------------------------------------.
            // |S| dab[7:1]  |rd|A| data |A| data |A*|P |
            // `----------------------------------------'
            //
            pCmd->bRead = LW_TRUE;
            pCmd->cntl = FLD_SET_DRF(_PMGR, _I2C_CNTL, _CMD, _READ, pCmd->cntl);

            burst_size = LW_MIN(LW_PMGR_I2C_CNTL_BURST_SIZE_MAXIMUM, pCmd->bytesRemaining);
            pCmd->cntl = FLD_SET_DRF_NUM(_PMGR, _I2C_CNTL, _BURST_SIZE, burst_size, pCmd->cntl);

            pCmd->cntl = FLD_SET_DRF(_PMGR, _I2C_CNTL, _GEN_STOP, _YES, pCmd->cntl);

            LWSWITCH_REG_WR32(device, _PMGR, _I2C_CNTL(pCmd->port), pCmd->cntl);

            pCmd->status = _lwswitch_i2c_i2cPollHwUntilDoneOrTimeoutForStatus_lr10(device, pCmd->port);
            LWSWITCH_CHECK_STATUS(device, pCmd->status);

            pCmd->data = LWSWITCH_REG_RD32(device, _PMGR, _I2C_DATA(pCmd->port));

            for (i = burst_size-1; i >= 0; i--)
            {
                if (pCmd->bBlockProtocol)
                {
                    if (((pCmd->data >> (i * 8)) & 0xFF) != pParams->messageLength)
                    {
                        pCmd->status = -LWL_BAD_ARGS;
                        LWSWITCH_CHECK_STATUS(device, pCmd->status);
                        break;
                    }

                    pCmd->bBlockProtocol = LW_FALSE;
                }
                else
                {
                    *pCmd->pMessage = (pCmd->data >> (i * 8)) & 0xFF;
                    pCmd->pMessage++;
                }
                pCmd->bytesRemaining--;
            }
            offset += burst_size;
        }
    }

    LWSWITCH_CHECK_STATUS(device, pCmd->status);
    return pCmd->status;
}

/*!
 * Write a byte in manual mode.
 *
 * @param[in] port            The port identifier.
 *
 * @param[in] data              The byte to write.
 *
 * @return LWL_STATUS
 */
static LwlStatus
_lwswitch_i2c_i2cWriteByteViaHw_lr10
(
    lwswitch_device *device,
    LwU32 port,
    LwU32 data
)
{
    LwU32 cntl = DRF_DEF(_PMGR, _I2C_CNTL, _CMD, _WRITE)          |
                 DRF_DEF(_PMGR, _I2C_CNTL, _INTR_WHEN_DONE, _YES) |
                 DRF_NUM(_PMGR, _I2C_CNTL, _BURST_SIZE, 1);

    LWSWITCH_REG_WR32(device, _PMGR, _I2C_DATA(port), data);
    LWSWITCH_REG_WR32(device, _PMGR, _I2C_CNTL(port), cntl);

    return _lwswitch_i2c_i2cPollHwUntilDoneOrTimeoutForStatus_lr10(device, port);
}

/*!
 * Send a start in manual mode.
 *
 * @param[in] port            The port identifier.
 *
 * @return LWL_STATUS
 */
static LwlStatus
_lwswitch_i2c_i2cSendStartViaHw_lr10
(
    lwswitch_device *device,
    LwU32 port
)
{
    LwU32 cntl = DRF_DEF(_PMGR, _I2C_CNTL, _GEN_START, _YES) |
                 DRF_DEF(_PMGR, _I2C_CNTL, _INTR_WHEN_DONE, _YES);

    LWSWITCH_REG_WR32(device, _PMGR, _I2C_CNTL(port), cntl);

    return _lwswitch_i2c_i2cPollHwUntilDoneOrTimeoutForStatus_lr10(device, port);
}

/*!
 * Send an address in manual mode.
 *
 * @param[in] pParams               The transaction description.
 *
 * @param[in] bSending          The read/write bit to use with the address.
 *
 * @return LWL_STATUS
 */
static LwU32
_lwswitch_i2c_i2cSendAddressViaHw_lr10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS *pParams,
    LwU32 bSending
)
{
    LwU32 flags  = pParams->flags;
    LwU32 port = pParams->port;
    LwU32 address = pParams->address;
    LwlStatus status;

    switch (DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _ADDRESS_MODE, flags))
    {
        default:
        case LWSWITCH_CTRL_I2C_FLAGS_ADDRESS_MODE_NO_ADDRESS:
            return LWL_SUCCESS;

        case LWSWITCH_CTRL_I2C_FLAGS_ADDRESS_MODE_7BIT:
            return _lwswitch_i2c_i2cWriteByteViaHw_lr10(device, port, (address | !bSending) & 0xFF);

        case LWSWITCH_CTRL_I2C_FLAGS_ADDRESS_MODE_10BIT:
            //
            // For 10-bits addressing, the 6 MSB bits on the 2 bytes are
            // b11110AAR; 10-address is 1111_0AAR_AAAA_AAAA
            //
            status = _lwswitch_i2c_i2cWriteByteViaHw_lr10(device, port, (((address >> 9) << 1) & 0xF7) |
                     0xF0 | !bSending);
            LWSWITCH_CHECK_STATUS(device, status);
            if (status == LWL_SUCCESS)
            {
                status = _lwswitch_i2c_i2cWriteByteViaHw_lr10(device, port, (address >> 1) & 0xFF);
                LWSWITCH_CHECK_STATUS(device, status);
            }
            return status;
    }
}

/*!
 * Send a stop in manual mode.
 *
 * @param[in] port            The port identifier.
 *
 * @return LWL_STATUS
 */
static LwlStatus
_lwswitch_i2c_i2cSendStopViaHw_lr10
(
    lwswitch_device *device,
    LwU32 port
)
{
    LwU32 cntl = DRF_DEF(_PMGR, _I2C_CNTL, _GEN_STOP, _YES) |
                 DRF_DEF(_PMGR, _I2C_CNTL, _INTR_WHEN_DONE, _YES);

    LWSWITCH_REG_WR32(device, _PMGR, _I2C_CNTL(port), cntl);

    return _lwswitch_i2c_i2cPollHwUntilDoneOrTimeoutForStatus_lr10(device, port);
}

static LwlStatus
_lwswitch_i2c_i2cSendByteCmd_lr10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS *pParams,
    LwBool               bSendStop,
    LwU32                offset
)
{
    LwU32 flags = pParams->flags;
    LwU32 port  = pParams->port;
    LwU32 status;

    if(DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, flags) != 1)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: indexLength %d unsupported\n",
            __FUNCTION__,
            DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, flags));
        return  -LWL_BAD_ARGS;
    }

    // Perform the initial operations we need before sending the message.
    status = _lwswitch_i2c_i2cSendStartViaHw_lr10(device, port);
    LWSWITCH_CHECK_STATUS(device, status);

    // send address
    if (status == LWL_SUCCESS)
    {
        status = _lwswitch_i2c_i2cSendAddressViaHw_lr10(device, pParams, LW_TRUE);
        LWSWITCH_CHECK_STATUS(device, status);
    }

    // send the index length
    if (status == LWL_SUCCESS)
    {
        status = _lwswitch_i2c_i2cWriteByteViaHw_lr10(device, port, offset);
        LWSWITCH_CHECK_STATUS(device, status);
    }

    // send the stop bit
    if ((status == LWL_SUCCESS) && bSendStop)
    {
        (void)_lwswitch_i2c_i2cSendStopViaHw_lr10(device, port);
    }

    LWSWITCH_CHECK_STATUS(device, status);
    return status;
}

/*!
 * Send a read with a 0/1-byte index.
 *
 * @param[in, out] pParams        The transaction description.
 *
 * @param[in, out] pCmd       The continuous command structure.
 *
 * @params[in] genStop        To issue Stop explicitly
 *
 * @params[in] bIndexPresent  To issue RAB or Not
 *
 * @params[in] bytesRemaining Bytes remaining to be read
 */
static void
_lwswitch_i2c_i2cHwInitCmdReadWithShortIndex_lr10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS *pParams,
    PLWSWITCH_I2C_HW_CMD          pCmd,
    LwBool               bIndexPresent,
    LwBool               genStop,
    LwU32                bytesRemaining
)
{
    LwU32 flags = pParams->flags;
    LwU32 cntl = 0;
    LwU32 data = 0;
    LwU32 addr = 0;
    LwU32 bytesToRead = LW_U32_MAX;
    LwU32 port = pParams->port;
    LwU32 address = pParams->address;
    LwU32 bSendAddress = LW_FALSE;

    //
    // Determine initial command.
    //
    // Note: GEN_RAB and BURST_SIZE done later; STATUS and CYCLE are outputs.
    cntl = DRF_DEF(_PMGR, _I2C_CNTL, _GEN_START, _YES)      |
           DRF_DEF(_PMGR, _I2C_CNTL, _GEN_STOP, _NO)        |
           DRF_DEF(_PMGR, _I2C_CNTL, _CMD, _READ)           |
           DRF_DEF(_PMGR, _I2C_CNTL, _INTR_WHEN_DONE, _YES) |
           DRF_DEF(_PMGR, _I2C_CNTL, _GEN_NACK, _NO);

    switch (DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _ADDRESS_MODE, flags))
    {
        default:
        case LWSWITCH_CTRL_I2C_FLAGS_ADDRESS_MODE_NO_ADDRESS:
            bSendAddress = LW_FALSE;
            break;

        case LWSWITCH_CTRL_I2C_FLAGS_ADDRESS_MODE_7BIT:
            bSendAddress = LW_TRUE;
            addr = DRF_NUM(_PMGR, _I2C_ADDR, _DAB, address >> 1) |
                   DRF_DEF(_PMGR, _I2C_ADDR, _TEN_BIT, _DISABLE);
            break;

        case LWSWITCH_CTRL_I2C_FLAGS_ADDRESS_MODE_10BIT:
            bSendAddress = LW_TRUE;
            addr = DRF_NUM(_PMGR, _I2C_ADDR, _DAB, address >> 1) |
                   DRF_DEF(_PMGR, _I2C_ADDR, _TEN_BIT, _ENABLE);
            break;
    }

    if (FLD_TEST_DRF(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, _ONE, flags) &&
        bIndexPresent)
    {
        cntl = FLD_SET_DRF(_PMGR, _I2C_CNTL, _GEN_RAB, _YES, cntl);
        addr = FLD_SET_DRF_NUM(_PMGR, _I2C_ADDR, _RAB,
                               pParams->index[0], addr);
    }
    else if (DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, flags) == 0)    // preconditions imply 'indexLength' == 0
    {
        cntl = FLD_SET_DRF(_PMGR, _I2C_CNTL, _GEN_RAB, _NO, cntl);
    }

    //
    // If block protocol, the size is the first byte, so we'll need to read
    // expected size + 1 bytes total.
    //
    if (FLD_TEST_DRF(SWITCH_CTRL, _I2C_FLAGS, _BLOCK_PROTOCOL, _ENABLED, flags))
    {
        bytesRemaining++;
    }

    if ((bytesRemaining <= LW_PMGR_I2C_CNTL_BURST_SIZE_MAXIMUM) || genStop)
    {
        cntl = FLD_SET_DRF(_PMGR, _I2C_CNTL, _GEN_STOP, _YES, cntl);
    }
    else // bytesRemaining > LW_PMGR_I2C_CNTL_BURST_SIZE_MAXIMUM
    {
        cntl = FLD_SET_DRF(_PMGR, _I2C_CNTL, _GEN_STOP, _NO, cntl);
    }

    // Value bytesRemaining does not decrease until the data is actually read.
    bytesToRead = LW_MIN(bytesRemaining, LW_PMGR_I2C_CNTL_BURST_SIZE_MAXIMUM);
    cntl = FLD_SET_DRF_NUM(_PMGR, _I2C_CNTL, _BURST_SIZE, bytesToRead, cntl);

    // Command submission.
    if (bSendAddress)
    {
        LWSWITCH_REG_WR32(device, _PMGR, _I2C_ADDR(port), addr);
    }

    LWSWITCH_REG_WR32(device, _PMGR, _I2C_CNTL(port), cntl);

    // Initialize next command.
    cntl = FLD_SET_DRF(_PMGR, _I2C_CNTL, _GEN_START, _NO, cntl);
    cntl = FLD_SET_DRF(_PMGR, _I2C_CNTL, _GEN_RAB, _NO, cntl);

    pCmd->port = port;
    pCmd->bRead = LW_TRUE;
    pCmd->cntl = cntl;
    pCmd->data = data;
    pCmd->bytesRemaining = bytesRemaining;
    pCmd->status = -LWL_ERR_ILWALID_STATE;
    if (FLD_TEST_DRF(SWITCH_CTRL, _I2C_FLAGS, _BLOCK_PROTOCOL, _ENABLED, flags))
    {
        pCmd->bBlockProtocol = LW_TRUE;
    }
    else
    {
        pCmd->bBlockProtocol = LW_FALSE;
    }
}

/*
 * This function does i2c reads in 4 bytes chunk only.
 * This can handle >4 bytes read but does so in 4 bytes transaction explicitly specifying
 * offset in each loop.
 *
 *@params[in] bSendStop        Issue stop after Cmd Byte
 */
static LwlStatus
_lwswitch_i2c_i2cHwReadChunks_lr10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS *pParams,
    PLWSWITCH_I2C_HW_CMD          pCmd,
    LwBool               bSendStop
)
{
    LwlStatus status       = LWL_SUCCESS;
    LwU32   bytesRemaining = pParams->messageLength;
    LwU32   offset         = 0;
    LwBool  start          = LW_TRUE;

    pCmd->pMessage = pParams->message;

    // Do transactions in 4 bytes chunk
    do {
        if (FLD_TEST_DRF(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, _ONE, pParams->flags))
        {
            if(start)
            {
                offset = pParams->index[0];
                start  = LW_FALSE;
            }
            else
            {
                offset += LW_PMGR_I2C_CNTL_BURST_SIZE_MAXIMUM;
            }
            status = _lwswitch_i2c_i2cSendByteCmd_lr10(device, pParams, bSendStop, offset);
            LWSWITCH_CHECK_STATUS(device, status);
        }

        if (status == LWL_SUCCESS)
        {
            _lwswitch_i2c_i2cHwInitCmdReadWithShortIndex_lr10(device, pParams, pCmd, LW_FALSE, LW_TRUE ,bytesRemaining);
        }

        if (status == LWL_SUCCESS)
        {
            status = _lwswitch_i2c_i2cPollHwUntilDoneOrTimeoutForStatus_lr10(device, pCmd->port);
            if (status == LWL_SUCCESS)
            {
                status = _lwswitch_i2c_i2cHwReadNext_lr10(device, pParams, pCmd, LW_FALSE);
                LWSWITCH_CHECK_STATUS(device, status);
            }
        }
        bytesRemaining = pCmd->bytesRemaining;
    } while ((bytesRemaining > 0) && (status == LWL_SUCCESS));

    return status;
}

static LwlStatus
_lwswitch_i2c_i2cHwRead_lr10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS *pParams,
    PLWSWITCH_I2C_HW_CMD          pCmd
)
{
    LwU32 status = LWL_SUCCESS;
    LwU32 flags  = pParams->flags;

    if (DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, flags) > 1)
    {
        status = _lwswitch_i2c_i2cHwReadChunks_LongIndex_lr10(device, pParams, pCmd, LW_FALSE);
    }
    else
    {
        status = _lwswitch_i2c_i2cHwReadChunks_lr10(device, pParams, pCmd, LW_FALSE);
    }
    if (status != LWL_SUCCESS)
    {
        (void)_lwswitch_i2c_i2cSendStopViaHw_lr10(device, pCmd->port);
    }

    return status;
}

/*!
 * Send a self-contained write command.
 *
 * @param[in,out] pParams           The transaction description.
 *
 * @return LWL_STATUS
 */
static void
_lwswitch_i2c_i2cHwShortWrite_lr10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS *pParams
)
{
    LwU32  flags  = pParams->flags;
    LwU32  port = pParams->port;
    LwU32  addr   = 0;
    LwU32  cntl   = 0;
    LwU32  data   = 0;
    LwU32  i      = 0;

    // Note: GEN_RAB done later; STATUS and CYCLE are outputs.
    cntl = DRF_DEF(_PMGR, _I2C_CNTL, _GEN_START, _YES)                |
           DRF_DEF(_PMGR, _I2C_CNTL, _GEN_STOP, _YES)                 |
           DRF_DEF(_PMGR, _I2C_CNTL, _CMD, _WRITE)                    |
           DRF_DEF(_PMGR, _I2C_CNTL, _INTR_WHEN_DONE, _YES)           |
           DRF_NUM(_PMGR, _I2C_CNTL, _BURST_SIZE, pParams->messageLength) |
           DRF_DEF(_PMGR, _I2C_CNTL, _GEN_NACK, _NO);

    switch (DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _ADDRESS_MODE, flags))
    {
        default:
        case LWSWITCH_CTRL_I2C_FLAGS_ADDRESS_MODE_NO_ADDRESS:
            break;

        case LWSWITCH_CTRL_I2C_FLAGS_ADDRESS_MODE_7BIT:
            addr = DRF_NUM(_PMGR, _I2C_ADDR, _DAB,
                           pParams->address >> 1) |
                   DRF_DEF(_PMGR, _I2C_ADDR, _TEN_BIT, _DISABLE);
            break;

        case LWSWITCH_CTRL_I2C_FLAGS_ADDRESS_MODE_10BIT:
            addr = DRF_NUM(_PMGR, _I2C_ADDR, _DAB,
                           pParams->address >> 1) |
                   DRF_DEF(_PMGR, _I2C_ADDR, _TEN_BIT, _ENABLE);
            break;
    }

    if (DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, flags) == 1)
    {
        cntl = FLD_SET_DRF(_PMGR, _I2C_CNTL, _GEN_RAB, _YES, cntl);
        addr = FLD_SET_DRF_NUM(_PMGR, _I2C_ADDR, _RAB, pParams->index[0],
                               addr);
    }
    else if (DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, flags) == 0)// preconditions imply 'indexLength' == 0
    {
        cntl = FLD_SET_DRF(_PMGR, _I2C_CNTL, _GEN_RAB, _NO, cntl);
    }

    for (i = 0; i < pParams->messageLength; i++)
    {
        data |= pParams->message[i] << (i * 8);
    }

    LWSWITCH_REG_WR32(device, _PMGR, _I2C_ADDR(port), addr);
    LWSWITCH_REG_WR32(device, _PMGR, _I2C_DATA(port), data);
    LWSWITCH_REG_WR32(device, _PMGR, _I2C_CNTL(port), cntl);
}

/*!
 * Start off a write.
 *
 * @param[in, out] pParams      The transaction description.
 *
 * @param[out]     pCmd     The continuous command structure.
 */
static LwU32
_lwswitch_i2c_i2cHwInitCmdWrite_lr10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS *pParams,
    PLWSWITCH_I2C_HW_CMD          pCmd
)
{
    LwU32 flags = pParams->flags;
    LwU32 port = pParams->port;
    LwU32 indexLength = DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, flags);
    LwlStatus status = LW_U32_MAX;
    LwU32 i = LW_U32_MAX;

    pCmd->port = port;
    pCmd->bRead = LW_FALSE;
    pCmd->pMessage = pParams->message;
    pCmd->status = -LWL_MORE_PROCESSING_REQUIRED;

    if ((indexLength <= 1) &&
        (pParams->messageLength < LW_PMGR_I2C_CNTL_BURST_SIZE_MAXIMUM))
    {
        _lwswitch_i2c_i2cHwShortWrite_lr10(device, pParams);
        pCmd->cntl = 0;
        pCmd->data = 0;
        pCmd->bytesRemaining = 0;

        return LWL_SUCCESS;
    }

    // Long write.

    // Perform the initial operations we need before sending the message.
    status = _lwswitch_i2c_i2cSendStartViaHw_lr10(device, port);

    if (status == LWL_SUCCESS)
    {
        status = _lwswitch_i2c_i2cSendAddressViaHw_lr10(device, pParams, LW_TRUE);
    }

    for (i = 0; (status == LWL_SUCCESS) && (i < indexLength); i++)
    {
        status = _lwswitch_i2c_i2cWriteByteViaHw_lr10(device, port, pParams->index[i]);
    }

    if ((status == LWL_SUCCESS) &&
        FLD_TEST_DRF(SWITCH_CTRL, _I2C_FLAGS, _RESTART, _SEND, flags))
    {
        status = _lwswitch_i2c_i2cSendStartViaHw_lr10(device, port);
        if (status == LWL_SUCCESS)
        {
            (void)_lwswitch_i2c_i2cSendAddressViaHw_lr10(device, pParams, LW_TRUE);
        }
    }

    // Issue the first byte of the message.
    if (pParams->messageLength == 0)
    {
        pCmd->data = 0;
        pCmd->cntl = DRF_DEF(_PMGR, _I2C_CNTL, _GEN_STOP, _YES) |
                     DRF_DEF(_PMGR, _I2C_CNTL, _INTR_WHEN_DONE, _YES);
        pCmd->bytesRemaining = 0;
    }
    else // pParams->messageLength != 0
    {
        if (FLD_TEST_DRF(SWITCH_CTRL, _I2C_FLAGS, _BLOCK_PROTOCOL, _ENABLED, flags))
        {
            // Actually write the size
            pCmd->data = pParams->messageLength;
            pCmd->cntl = DRF_DEF(_PMGR, _I2C_CNTL, _CMD           , _WRITE) |
                         DRF_DEF(_PMGR, _I2C_CNTL, _INTR_WHEN_DONE, _YES)   |
                         DRF_NUM(_PMGR, _I2C_CNTL, _BURST_SIZE    , 1);

            LWSWITCH_REG_WR32(device, _PMGR, _I2C_DATA(port), pCmd->data);
            LWSWITCH_REG_WR32(device, _PMGR, _I2C_CNTL(port), pCmd->cntl);

            //
            // Wait for write to complete.  Ignoring the status for now because
            // these writes below are always happening, even if everything above
            // failed!
            //
            (void)_lwswitch_i2c_i2cPollHwUntilDoneOrTimeout_lr10(device, port);
        }

        pCmd->data = pParams->message[0];
        pCmd->cntl = DRF_DEF(_PMGR, _I2C_CNTL, _CMD           , _WRITE) |
                     DRF_DEF(_PMGR, _I2C_CNTL, _INTR_WHEN_DONE, _YES)   |
                     DRF_NUM(_PMGR, _I2C_CNTL, _BURST_SIZE    , 1);
        if (pParams->messageLength == 1)
        {
            pCmd->cntl = FLD_SET_DRF(_PMGR, _I2C_CNTL, _GEN_STOP, _YES,
                                     pCmd->cntl);
        }
        else // pParams->messageLength > 1
        {
            pCmd->cntl = FLD_SET_DRF(_PMGR, _I2C_CNTL, _GEN_STOP, _NO,
                                     pCmd->cntl);
        }
        pCmd->bytesRemaining = pParams->messageLength - 1;
        pCmd->pMessage++;
    }

    LWSWITCH_REG_WR32(device, _PMGR, _I2C_DATA(port), pCmd->data);
    LWSWITCH_REG_WR32(device, _PMGR, _I2C_CNTL(port), pCmd->cntl);

    if (pCmd->bytesRemaining > 0)
    {
        pCmd->data = *pCmd->pMessage;
        pCmd->pMessage++;
    }

    return LWL_SUCCESS;
}

static LwU32
_lwswitch_i2c_i2cHwWriteNext_lr10
(
    lwswitch_device *device,
    PLWSWITCH_I2C_HW_CMD pCmd
)
{
    LwlStatus status = -LWL_ERR_GENERIC;
    LwU32 cntl = LWSWITCH_REG_RD32(device, _PMGR, _I2C_CNTL(pCmd->port));

    status = _lwswitch_i2c_i2cHwStatusToI2cStatus_lr10(device,
        DRF_VAL(_PMGR, _I2C_CNTL, _STATUS, cntl));

    // Get the next command running ASAP to reduce latency.
    if ((status == LWL_SUCCESS) && (pCmd->bytesRemaining > 0))
    {
        LWSWITCH_REG_WR32(device, _PMGR, _I2C_DATA(pCmd->port), pCmd->data);
        LWSWITCH_REG_WR32(device, _PMGR, _I2C_CNTL(pCmd->port), pCmd->cntl);

        pCmd->bytesRemaining--;
    }

    if (status == LWL_SUCCESS)
    {
        if (pCmd->bytesRemaining > 0)
        {
            pCmd->data = *pCmd->pMessage;
            pCmd->pMessage++;
        }
        else
        {
            pCmd->status = status;
        }
    }

    return status;
}

static LwlStatus
_lwswitch_i2c_i2cHwWrite_lr10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS *pParams,
    PLWSWITCH_I2C_HW_CMD          pCmd
)
{
    LwlStatus status  = _lwswitch_i2c_i2cHwInitCmdWrite_lr10(device, pParams, pCmd);
    LwlStatus status2 = -LWL_ERR_GENERIC;

    LWSWITCH_CHECK_STATUS(device, status);
    while ((status == LWL_SUCCESS) && (pCmd->status == -LWL_MORE_PROCESSING_REQUIRED))
    {
        status = _lwswitch_i2c_i2cPollHwUntilDoneOrTimeoutForStatus_lr10(device, pCmd->port);
        LWSWITCH_CHECK_STATUS(device, status);
        if (status == LWL_SUCCESS)
        {
            status = _lwswitch_i2c_i2cHwWriteNext_lr10(device, pCmd);
            LWSWITCH_CHECK_STATUS(device, status);
        }
    }

    if (status == LWL_SUCCESS)
    {
        status = _lwswitch_i2c_i2cPollHwUntilDoneOrTimeoutForStatus_lr10(device, pCmd->port);
        LWSWITCH_CHECK_STATUS(device, status);
    }

    // Success or failure, for writes we need to send a manual stop.
    status2 = _lwswitch_i2c_i2cSendStopViaHw_lr10(device, pCmd->port);
    LWSWITCH_CHECK_STATUS(device, status2);
    if (status == LWL_SUCCESS)
    {
        status = status2;
    }

    LWSWITCH_CHECK_STATUS(device, status);
    return status;
}

/*!
 * @brief   Perform a generic I2C transaction using the HW controller
 *
 * @param[in, out] pParams
 *          Transaction data; please see @ref ./inc/task4_i2c.h "task4_i2c.h".
 *
 * @return  LWL_SUCCESS
 *          The transaction completed successfully.
 */
static LwlStatus
_lwswitch_i2c_i2cPerformTransactiolwiaHw_lr10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS *pParams
)
{
    LwU32 status     = -LWL_ERR_GENERIC;
    LwU32 flags      = pParams->flags;
    LwU32 port     = pParams->port;
    PLWSWITCH_I2C_HW_CMD pCmd = &device->pI2c->Ports[port].hwCmd;
    LwU32 speedMode  = DRF_VAL(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, flags);

    lwswitch_i2c_set_hw_speed_mode(device, port, speedMode);

    if (pParams->bIsRead)
    {
        status = _lwswitch_i2c_i2cHwRead_lr10(device, pParams, pCmd);
    }
    else
    {
        status = _lwswitch_i2c_i2cHwWrite_lr10(device, pParams, pCmd);
        LWSWITCH_CHECK_STATUS(device, status);
    }

    return status;
}

/*!
 * RM Control command to determine port information.
 */
LwlStatus
lwswitch_ctrl_i2c_get_port_info_lr10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_I2C_GET_PORT_INFO_PARAMS *pParams
)
{
    LwU32   port;

    for (port = 0; port < LWSWITCH_CTRL_NUM_I2C_PORTS; port++)
    {
        pParams->info[port] = _lwswitch_i2c_get_port_info_lr10(device, port);
    }

    return (LWL_SUCCESS);
}

/*!
 * RM Control command to perform indexed I2C.
 */
LwlStatus
lwswitch_ctrl_i2c_indexed_lr10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS *pParams
)
{
    LwlStatus status = (-LWL_ERR_GENERIC);
    LwBool bSwImpl   = LW_TRUE;
    LwBool bWasBb    = LW_TRUE;
    LwU32  flags     = pParams->flags;
    LwU32  port    = pParams->port;

    bSwImpl = FLD_TEST_DRF(SWITCH_CTRL, _I2C_FLAGS, _FLAVOR, _SW, flags);

    // Do argument checks that don't require the I2C lock.
    status = _lwswitch_ctrl_i2c_indexedCheckPreconditions_lr10(device, pParams);
    LWSWITCH_CHECK_STATUS(device, status);
    if (status != (LWL_SUCCESS))
    {
        goto done;
    }

    status = _lwswitch_i2c_LockObjI2c_lr10(device, pParams->port, pParams->acquirer);
    LWSWITCH_CHECK_STATUS(device, status);
    if (status != (LWL_SUCCESS))
    {
        goto done;
    }

    // Brackets indicate period where LWSWITCH_OBJI2C lock is held.
    {
        // Now finish checking the arguments and state with the lock held.
        status = _lwswitch_i2c_indexed_check_state_lr10(device, pParams->port);
        LWSWITCH_CHECK_STATUS(device, status);

        if (status == LWL_SUCCESS)
        {
            //
            // When just starting an i2c transaction and just after successfully
            // acquiring the I2C mutex, it is never expected that the bus will be
            // busy.  If it is busy, consider it as being in an invalid state and
            // attempt to recover it.
            //
            if (!_lwswitch_i2c_i2cIsBusReady_lr10(device, port))
            {
                // set the i2c mode to bit-banging mode (should never fail!)
                status = _lwswitch_i2c_i2cSetMode_lr10(device, port, LW_TRUE, &bWasBb);
                LWSWITCH_CHECK_STATUS(device, status);

                // now attempt to recover the bus
                status = _lwswitch_i2c_i2cRecoverBusViaSw_lr10(device, pParams);
                LWSWITCH_CHECK_STATUS(device, status);
                _lwswitch_i2c_i2cRestoreMode_lr10(device, port, bWasBb);

                // bail-out if we couldn't recover it, nothing else to do
                if (status != LWL_SUCCESS)
                {
                    LWSWITCH_CHECK_STATUS(device, status);
                    goto i2cPerformTransaction_done;
                }
            }

            //
            // With the bus ready, set the desired operating mode and perform the
            // transaction.
            //
            status = _lwswitch_i2c_i2cSetMode_lr10(device, port, bSwImpl, &bWasBb);
            LWSWITCH_CHECK_STATUS(device, status);
            if (status == LWL_SUCCESS)
            {
                if (bSwImpl)
                {
                    status = _lwswitch_i2c_i2cPerformTransactiolwiaSw_lr10(device, pParams);
                }
                else
                {
                    status = _lwswitch_i2c_i2cPerformTransactiolwiaHw_lr10(device, pParams);
                }
                _lwswitch_i2c_i2cRestoreMode_lr10(device, port, bWasBb);
            }
        }

i2cPerformTransaction_done:
        _lwswitch_i2c_UnlockObjI2c_lr10(device, pParams->port);
    }

done:
    return status;
}

LwlStatus
lwswitch_get_rom_info_lr10
(
    lwswitch_device *device,
    LWSWITCH_EEPROM_TYPE *eeprom
)
{
    if (IS_RTLSIM(device) || IS_EMULATION(device) || IS_FMODEL(device))
    {
        LWSWITCH_PRINT(device, SETUP,
            "ROM configuration not supported on Fmodel/RTL/emulation\n");
        return -LWL_ERR_NOT_SUPPORTED;
    }

    return -LWL_ERR_NOT_SUPPORTED;
}

/*!
 * Set the speed of the HW I2C controller on a given port.
 *
 * @param[in] port          The port identifying the controller.
 *
 * @param[in] speedMode     The speed mode to run at.
 */
void
lwswitch_i2c_set_hw_speed_mode_lr10
(
    lwswitch_device *device,
    LwU32 port,
    LwU32 speedMode
)
{
    LwU32 timing = DRF_DEF(_PMGR, _I2C_TIMING, _IGNORE_ACK, _DISABLE) |
                   DRF_DEF(_PMGR, _I2C_TIMING, _TIMEOUT_CHECK, _ENABLE);

    switch (speedMode)
    {
        // Default should not be hit if above layers work correctly.
        default:
            LWSWITCH_PRINT(device, ERROR,
                "%s: undefined speed\n",
                __FUNCTION__);
            // Deliberate fallthrough
        case LWSWITCH_CTRL_I2C_FLAGS_SPEED_MODE_100KHZ:
            timing = FLD_SET_DRF(_PMGR, _I2C_TIMING, _SCL_PERIOD, _100KHZ, timing);
            timing = FLD_SET_DRF_NUM(_PMGR, _I2C_TIMING, _TIMEOUT_CLK_CNT, LWSWITCH_I2C_SCL_CLK_TIMEOUT_100KHZ, timing);
            break;

        case LWSWITCH_CTRL_I2C_FLAGS_SPEED_MODE_200KHZ:
            timing = FLD_SET_DRF(_PMGR, _I2C_TIMING, _SCL_PERIOD, _200KHZ, timing);
            timing = FLD_SET_DRF_NUM(_PMGR, _I2C_TIMING, _TIMEOUT_CLK_CNT, LWSWITCH_I2C_SCL_CLK_TIMEOUT_200KHZ, timing);
            break;

        case LWSWITCH_CTRL_I2C_FLAGS_SPEED_MODE_300KHZ:
            timing = FLD_SET_DRF(_PMGR, _I2C_TIMING, _SCL_PERIOD, _300KHZ, timing);
            timing = FLD_SET_DRF_NUM(_PMGR, _I2C_TIMING, _TIMEOUT_CLK_CNT, LWSWITCH_I2C_SCL_CLK_TIMEOUT_300KHZ, timing);
            break;

        case LWSWITCH_CTRL_I2C_FLAGS_SPEED_MODE_400KHZ:
            timing = FLD_SET_DRF(_PMGR, _I2C_TIMING, _SCL_PERIOD, _400KHZ, timing);
            timing = FLD_SET_DRF_NUM(_PMGR, _I2C_TIMING, _TIMEOUT_CLK_CNT, LWSWITCH_I2C_SCL_CLK_TIMEOUT_400KHZ, timing);
            break;

        case LWSWITCH_CTRL_I2C_FLAGS_SPEED_MODE_1000KHZ:
            timing = FLD_SET_DRF(_PMGR, _I2C_TIMING, _SCL_PERIOD, _1000KHZ, timing);
            timing = FLD_SET_DRF_NUM(_PMGR, _I2C_TIMING, _TIMEOUT_CLK_CNT, LWSWITCH_I2C_SCL_CLK_TIMEOUT_1000KHZ, timing);
            break;
    }

    LWSWITCH_REG_WR32(device, _PMGR, _I2C_TIMING(port), timing);
}

/*!
 * Return if I2C transactions are supported.
 *
 * @param[in] device        The LwSwitch Device.
 *
 */
LwBool
lwswitch_is_i2c_supported_lr10
(
    lwswitch_device *device
)
{
    return LW_TRUE;
}

/*!
 * Return if I2C device and port is allowed access
 *
 * @param[in] device        The LwSwitch Device.
 * @param[in] port          The I2C Port.
 * @param[in] addr          The I2C device to access.
 * @param[in] bIsRead       Boolean if I2C transaction is a read.
 *
 */
LwBool
lwswitch_i2c_is_device_access_allowed_lr10
(
    lwswitch_device *device,
    LwU32 port,
    LwU8 addr,
    LwBool bIsRead
)
{
    LwU32 i;
    LwU32 device_allow_list_size;
    LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE *device_allow_list;
    LwBool bAllow = LW_FALSE;
    PLWSWITCH_OBJI2C pI2c = device->pI2c;

    device_allow_list = pI2c->i2c_allow_list;
    device_allow_list_size = pI2c->i2c_allow_list_size;

    for (i = 0; i < device_allow_list_size; i++)
    {
        LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE i2c_device = device_allow_list[i];

        if ((port == i2c_device.i2cPortLogical) &&
            (addr == i2c_device.i2cAddress))
        {
            bAllow = bIsRead ?
                         FLD_TEST_DRF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL,
                                      _PUBLIC, i2c_device.i2cRdWrAccessMask) :
                         FLD_TEST_DRF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL,
                                      _PUBLIC, i2c_device.i2cRdWrAccessMask);
            break;
        }
    }

    return bAllow;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
/*!
 * RM Control to get I2C device info from the DCB I2C Devices Table.
 */
LwlStatus
lwswitch_ctrl_i2c_get_dev_info_lr10
(
    lwswitch_device *device,
    LWSWITCH_CTRL_I2C_GET_DEV_INFO_PARAMS *pParams
)
{
#ifdef LW_MODS
    LwU8    numDevices = 0;
    LwU32   idx_i2c_device;
    PLWSWITCH_OBJI2C pI2c = device->pI2c;

    // LR10 can manually add specific devices to pI2c->device_list that are not in the DCB
    // so capture those first
    for (idx_i2c_device=0; (idx_i2c_device < pI2c->device_list_size) && (numDevices < LWSWITCH_CTRL_I2C_MAX_DEVICES); idx_i2c_device++)
    {
        pParams->i2cDevInfo[numDevices].i2cAddress     = pI2c->device_list[idx_i2c_device].i2cAddress;
        pParams->i2cDevInfo[numDevices].i2cPortLogical = (LwU32) pI2c->device_list[idx_i2c_device].i2cPortLogical;
        pParams->i2cDevInfo[numDevices].type           = (LwU32) pI2c->device_list[idx_i2c_device].i2cDeviceType;
        numDevices++;
    }

    // If there are any DCB I2C devices defined include them as well
    if (device->firmware.dcb.i2c_device_count != 0)
    {
        LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE *fwI2cDevices = &(device->firmware.dcb.i2c_device[0]);

        for (idx_i2c_device = 0; (idx_i2c_device < device->firmware.dcb.i2c_device_count) && (numDevices < LWSWITCH_CTRL_I2C_MAX_DEVICES); idx_i2c_device++)
        {
            pParams->i2cDevInfo[numDevices].i2cAddress     = fwI2cDevices[idx_i2c_device].i2cAddress;
            pParams->i2cDevInfo[numDevices].i2cPortLogical = (LwU32) fwI2cDevices[idx_i2c_device].i2cPortLogical;
            pParams->i2cDevInfo[numDevices].type           = (LwU32) fwI2cDevices[idx_i2c_device].i2cDeviceType;
            numDevices++;
        }
    }

    pParams->i2cDevCount = numDevices;
    return (LWL_SUCCESS);
#else
    return -LWL_ERR_NOT_SUPPORTED;
#endif
}
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
