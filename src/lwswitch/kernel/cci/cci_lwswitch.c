/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "cci/cci_lwswitch.h"
#include "cci/cci_priv_lwswitch.h"

#include "common_lwswitch.h"
#include "boards_lwswitch.h"

#include "lwlink_export.h"
#include "lr10/lr10.h"
#include "regkey_lwswitch.h"
#include "rom_lwswitch.h"

#include "lwswitch/lr10/dev_pmgr.h"

#include "cci/led_tca6507.h"

#include "oob/smbpbi.h"

#include "ipmi/fru_lwswitch.h"

#include "lr10/cci_osfp_txeq_lr10.h"
#include "lr10/cci_osfp_cage_bezel_markings_lr10.h"

#define OSFP_LANE_MASK(lane0, lane1, lane2, lane3) \
    (LWBIT(lane0) | LWBIT(lane1) | LWBIT(lane2) | LWBIT(lane3))

#define OSFP_LED_PER_DRVR_PCA9685BS     4
#define OSFP_LED_PER_DRVR_TCA6507       2

#define OSFP_LED_TCA6507_GREEN_PORT(idx)    ((idx) * 2)
#define OSFP_LED_TCA6507_AMBER_PORT(idx)    ((idx) * 2 + 1)

#define CCI_LED_UPDATE_RATE_HZ  1

static void _lwswitch_find_led_drivers_and_rom(lwswitch_device *device, LwU32 client, PCCI pCci);
static LwlStatus _lwswitch_cci_init_xcvr_leds(lwswitch_device *device);

static void _lwswitch_cci_update_link_state_led(lwswitch_device *device);

static LwlStatus _lwswitch_cci_detect_board(lwswitch_device *device);
static LwlStatus _lwswitch_cci_detect_partitions(lwswitch_device *device);

static LwlStatus _lwswitch_cci_reset_links(lwswitch_device *device, LwU64 linkMask);

//
// Internal Macros
//
#define NUM_CCI_LINKS_LR10      16
#define NUM_CCI_OSFP_LR10        8
#define NUM_CCI_OSFP_LANES_LR10  4

/* -------------------- Object construction/initialization ------------------- */

#define LWSWITCH_BOARD_PARTITION_E4760_A00_PART_NUM "699-14760"
#define LWSWITCH_BOARD_PARTITION_E4761_A00_PART_NUM "699-14761"
#define LWSWITCH_BOARD_PARTITION_P4790_B00_PART_NUM "699-14790"
#define LWSWITCH_BOARD_PARTITION_P4791_B00_PART_NUM "699-14791"
#define LWSWITCH_BOARD_PARTITION_E3597_B00_PART_NUM "699-13597"

typedef enum
{
    LWSWITCH_BOARD_PARTITION_UNKNOWN = 0,
    LWSWITCH_BOARD_PARTITION_E4760_A00,
    LWSWITCH_BOARD_PARTITION_E4761_A00,
    LWSWITCH_BOARD_PARTITION_P4790_B00,
    LWSWITCH_BOARD_PARTITION_P4791_B00,
    LWSWITCH_BOARD_PARTITION_E3597_B00
} LWSWITCH_BOARD_PARTITION_TYPE;

typedef struct {
    const char* part_num;
    LWSWITCH_BOARD_PARTITION_TYPE type;
} LWSWITCH_BOARD_PARTITION_ENTRY;

static LwBool
_lwswitch_cci_board_supported
(
    lwswitch_device *device
)
{
    LwU64 enabledLinkMask;
    LwU8 linkId;
    
    enabledLinkMask = lwswitch_get_enabled_link_mask(device);

    FOR_EACH_INDEX_IN_MASK(64, linkId, enabledLinkMask)
    {
        LWSWITCH_ASSERT(linkId < LWSWITCH_LINK_COUNT(device));
        
        // CCI supported if any active repeater is present
        if (device->link[linkId].bActiveRepeaterPresent)
        {
            return LW_TRUE;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    LWSWITCH_PRINT(device, INFO, "%s Active repeaters are not present in bios so board does not support CCI. \n",
            __FUNCTION__);

    return LW_FALSE;
}

static LwlStatus
_lwswitch_cci_discovery
(
    lwswitch_device *device
)
{
    PCCI pCci = device->pCci;
    LwlStatus retval;

    if (!_lwswitch_cci_board_supported(device))
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    //
    // Determine FRU roms available
    // TODO: should be done by looking at bios
    //
    _lwswitch_find_led_drivers_and_rom(device, LWSWITCH_I2C_ACQUIRER_CCI_INITIALIZE, pCci);

    retval = _lwswitch_cci_detect_board(device);
    if (retval != LWL_SUCCESS)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    return LWL_SUCCESS;
}

LwBool
cciSupported
(
    lwswitch_device *device
)
{
    PCCI pCci = device->pCci;
    LwlStatus retval;

    if (pCci == NULL)
    {
        return LW_FALSE;
    }

    if (pCci->bDiscovered)
    {
        return pCci->bSupported;
    }

    // Discover if CCI supported board
    retval = _lwswitch_cci_discovery(device);
    if (retval == LWL_SUCCESS)
    {
        pCci->bSupported = LW_TRUE;
    }
    else
    {
        pCci->bSupported = LW_FALSE;
    }
    
    pCci->bDiscovered = LW_TRUE;

    return pCci->bSupported;
}

CCI *
cciAllocNew(void)
{
    CCI *pCci = lwswitch_os_malloc(sizeof(*pCci));
    if (pCci != NULL)
    {
        lwswitch_os_memset(pCci, 0, sizeof(*pCci));
    }

    return pCci;
}

static void
_lwswitch_cci_poll_callback
(
    lwswitch_device *device
)
{
    PCCI pCci = device->pCci;
    LwU32 i;

    // call all functions at specified frequencies
    for (i = 0; i < LWSWITCH_CCI_CALLBACK_NUM_MAX; i++)
    {
        if ((pCci->callbackList[i].functionPtr != NULL) &&
            ((pCci->callbackCounter % pCci->callbackList[i].interval) == 0))
        {
            pCci->callbackList[i].functionPtr(device);
        }
    }
    pCci->callbackCounter++;
}

LwlStatus
cciInit
(
    lwswitch_device    *device,
    CCI                *pCci,
    LwU32               pci_device_id
)
{
    lwswitch_task_create(device, _lwswitch_cci_poll_callback,
                         LWSWITCH_INTERVAL_1SEC_IN_NS / LWSWITCH_CCI_POLLING_RATE_HZ,
                         0);
    return LWL_SUCCESS;
}

// reverse of cciInit()
void
cciDestroy
(
    lwswitch_device    *device,
    CCI                *pCci
)
{
    unsigned    idx;

    for (idx = 0; idx < pCci->rom_num; ++idx)
    {
        lwswitch_os_free(pCci->romCache[idx]);
    }
}

static void
_lwswitch_cci_get_all_links
(
    lwswitch_device *device,
    LwU64 *pLinkMaskAll
)
{
    PCCI pCci = device->pCci;
    LwU64 linkMaskAll;
    LwU32 i;

    linkMaskAll = 0;
    for (i = 0; i < pCci->osfp_map_size; i++)
    {
        linkMaskAll |= LWBIT64(pCci->osfp_map[i].linkId);
    }

    if (pLinkMaskAll != NULL)
    {
        *pLinkMaskAll = linkMaskAll;
    }
}

LwlStatus 
cciResetAllPartitions
(
    lwswitch_device *device
)
{
    PCCI pCci = device->pCci;
    LwlStatus retval;
    LwU32 intn0, rstn0, intn1, rstn1;
    LwU32 intn0Val, rstn0Val, intn1Val, rstn1Val;
    LwU64 linkMaskAll;

    if ((pCci == NULL) || (!pCci->bSupported))
    {
        LWSWITCH_PRINT(device, ERROR,
                "%s: CCI not supported\n",
                __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    intn0 = 15; //GPIO15
    rstn0 = 18; //GPIO18
    intn1 = 14; //GPIO14
    rstn1 = 17; //GPIO17

    intn0Val = LWSWITCH_REG_RD32(device, _PMGR, _GPIO_OUTPUT_CNTL(intn0));
    rstn0Val = LWSWITCH_REG_RD32(device, _PMGR, _GPIO_OUTPUT_CNTL(rstn0));
    intn1Val = LWSWITCH_REG_RD32(device, _PMGR, _GPIO_OUTPUT_CNTL(intn1));
    rstn1Val = LWSWITCH_REG_RD32(device, _PMGR, _GPIO_OUTPUT_CNTL(rstn1));

    // Gpio set direction intn0 to input and rstn0 to output
    intn0Val = FLD_SET_DRF(_PMGR, _GPIO_OUTPUT_CNTL, _SEL, _NORMAL, intn0Val);
    intn0Val = FLD_SET_DRF_NUM(_PMGR, _GPIO_OUTPUT_CNTL, _IO_OUT_EN, 0, intn0Val);
    LWSWITCH_REG_WR32(device, _PMGR, _GPIO_OUTPUT_CNTL(intn0), intn0Val);

    rstn0Val = FLD_SET_DRF(_PMGR, _GPIO_OUTPUT_CNTL, _SEL, _NORMAL, rstn0Val);
    rstn0Val = FLD_SET_DRF_NUM(_PMGR, _GPIO_OUTPUT_CNTL, _IO_OUT_EN, 1, rstn0Val);
    LWSWITCH_REG_WR32(device, _PMGR, _GPIO_OUTPUT_CNTL(rstn0), rstn0Val);

    // Gpio set direction intn1 to input and rstn1 to output
    intn1Val = FLD_SET_DRF(_PMGR, _GPIO_OUTPUT_CNTL, _SEL, _NORMAL, intn1Val);
    intn1Val = FLD_SET_DRF_NUM(_PMGR, _GPIO_OUTPUT_CNTL, _IO_OUT_EN, 0, intn1Val);
    LWSWITCH_REG_WR32(device, _PMGR, _GPIO_OUTPUT_CNTL(intn1), intn1Val);

    rstn1Val = FLD_SET_DRF(_PMGR, _GPIO_OUTPUT_CNTL, _SEL, _NORMAL, rstn1Val);
    rstn1Val = FLD_SET_DRF_NUM(_PMGR, _GPIO_OUTPUT_CNTL, _IO_OUT_EN, 1, rstn1Val);
    LWSWITCH_REG_WR32(device, _PMGR, _GPIO_OUTPUT_CNTL(rstn1), rstn1Val);

    // Gpio set rstn0 and rstn1 output to low and trigger
    rstn0Val = FLD_SET_DRF(_PMGR, _GPIO_OUTPUT_CNTL, _IO_OUTPUT, _0, rstn0Val);
    LWSWITCH_REG_WR32(device, _PMGR, _GPIO_OUTPUT_CNTL(rstn0), rstn0Val);

    LWSWITCH_REG_WR32(device, _PMGR, _GPIO_OUTPUT_CNTL_TRIGGER,
        DRF_DEF(_PMGR, _GPIO_OUTPUT_CNTL, _TRIGGER_UPDATE, _TRIGGER));

    rstn1Val = FLD_SET_DRF(_PMGR, _GPIO_OUTPUT_CNTL, _IO_OUTPUT, _0, rstn1Val);
    LWSWITCH_REG_WR32(device, _PMGR, _GPIO_OUTPUT_CNTL(rstn1), rstn1Val);

    LWSWITCH_REG_WR32(device, _PMGR, _GPIO_OUTPUT_CNTL_TRIGGER,
        DRF_DEF(_PMGR, _GPIO_OUTPUT_CNTL, _TRIGGER_UPDATE, _TRIGGER));

    lwswitch_os_sleep(1000);

    // Gpio set rstn0 and rstn1 output to high and trigger
    rstn0Val = FLD_SET_DRF(_PMGR, _GPIO_OUTPUT_CNTL, _IO_OUTPUT, _1, rstn0Val);
    LWSWITCH_REG_WR32(device, _PMGR, _GPIO_OUTPUT_CNTL(rstn0), rstn0Val);

    LWSWITCH_REG_WR32(device, _PMGR, _GPIO_OUTPUT_CNTL_TRIGGER,
        DRF_DEF(_PMGR, _GPIO_OUTPUT_CNTL, _TRIGGER_UPDATE, _TRIGGER));

    rstn1Val = FLD_SET_DRF(_PMGR, _GPIO_OUTPUT_CNTL, _IO_OUTPUT, _1, rstn1Val);
    LWSWITCH_REG_WR32(device, _PMGR, _GPIO_OUTPUT_CNTL(rstn1), rstn1Val);

    LWSWITCH_REG_WR32(device, _PMGR, _GPIO_OUTPUT_CNTL_TRIGGER,
        DRF_DEF(_PMGR, _GPIO_OUTPUT_CNTL, _TRIGGER_UPDATE, _TRIGGER));

    lwswitch_os_sleep(2000);

    if (pCci->boardPartitionType == LWSWITCH_BOARD_PARTITION_P4790_B00 ||
        pCci->boardPartitionType == LWSWITCH_BOARD_PARTITION_P4791_B00 ||
        pCci->boardPartitionType == LWSWITCH_BOARD_PARTITION_E3597_B00)
    {
        _lwswitch_cci_get_all_links(device, &linkMaskAll);

        retval = _lwswitch_cci_reset_links(device, linkMaskAll);
        if (retval != LWL_SUCCESS)
        {
            return retval;
        }
    }

    return LWL_SUCCESS;
}

static LwBool
_lwswitch_cci_module_present
(
    lwswitch_device *device,
    LwU32           osfp
)
{
    return !!(device->pCci->osfpMaskPresent & LWBIT(osfp));
}

static LwlStatus
_lwswitch_cci_get_module_id
(
    lwswitch_device *device,
    LwU32           linkId,
    LwU32           *osfp
)
{
    PCCI pCci = device->pCci;
    LwU32 i;

    for (i = 0; i < pCci->osfp_map_size; i++)
    {
        if (pCci->osfp_map[i].linkId == linkId)
        {
            *osfp = pCci->osfp_map[i].moduleId;

            if (!(device->pCci->cagesMask & LWBIT(*osfp)))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: osfp %d associated with link %d is not supported\n",
                    __FUNCTION__, linkId, *osfp);
                return -LWL_NOT_FOUND;
            }

            return LWL_SUCCESS;
        }
    }

    return -LWL_NOT_FOUND;
}

#define P479X_IO_EXPANDER_PRESENT_INTERRUPT      0
#define P479X_IO_EXPANDER_RESET_LOW_POWER        1

// P4790 B00 board (io expander device)
static LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE lwswitch_io_expander_list_P4790_B00[] =
{
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xE0, _PCAL9538,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xE2, _PCAL9538,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
};

// P4791 B00 board (io expander device)
static LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE lwswitch_io_expander_list_P4791_B00[] =
{
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xE0, _PCAL9538,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xE2, _PCAL9538,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
};

#define WOLF_IO_EXPANDER_RESET         0
#define WOLF_IO_EXPANDER_LOW_POWER     1
#define WOLF_IO_EXPANDER_PRESENT       2
#define WOLF_IO_EXPANDER_INTERRUPT     3
#define WOLF_IO_EXPANDER_POWER_ENABLE  4
#define WOLF_IO_EXPANDER_POWER_GOOD    5
#define WOLF_IO_EXPANDER_LED           6

// Wolf board (io expander devices)
static LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE lwswitch_io_expander_list_wolf_B[] =
{
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xE0, _PCAL9538,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xE2, _PCAL9538,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),   
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xE4, _PCAL9538,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),   
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xE6, _PCAL9538,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),   
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xE8, _PCAL9538,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),   
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xEA, _PCAL9538,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),   
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xEC, _PCAL9538,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),      
};

// Wolf board (io expander devices)
static LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE lwswitch_io_expander_list_wolf_C[] =
{
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xE0, _PCAL9538,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xE2, _PCAL9538,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xE4, _PCAL9538,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xE6, _PCAL9538,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xE8, _PCAL9538,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xEA, _PCAL9538,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xEC, _PCAL9538,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
};

#define IO_EXPANDER_INPUT_REG                    0x0
#define IO_EXPANDER_OUTPUT_REG                   0x1
#define IO_EXPANDER_POLARITY_REG                 0x2
#define IO_EXPANDER_CONFIG_REG                   0x3
#define IO_EXPANDER_CONFIG_ASSERT_RESET(mask)    (mask)
#define IO_EXPANDER_CONFIG_DEASSERT_RESET        0x0

#define CCI_TO_PCS_MODULE_MASK_MAP_P479X_B00(mask, pcsId)   ((mask >> (pcsId << 2)) & 0xF)
#define PCS_TO_CCI_MODULE_MASK_MAP_P479X_B00(mask, pcsId)   ((LwU32)mask << (pcsId << 2))
#define CCI_TO_PCS_MODULE_MASK_MAP_E3597_B00(mask, pcsId)   ((mask >> (pcsId << 3)) & 0xFF)
#define PCS_TO_CCI_MODULE_MASK_MAP_E3597_B00(mask, pcsId)   ((LwU32)mask << (pcsId << 3))

static LwlStatus 
_cci_pcs_write_P479X_B00
(
    lwswitch_device *device,
    LwU32 pcsId,
    LwU8  expander,
    LwU8  reg,
    LwU8  value
)
{
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS i2c_params = { 0 };
    LwlStatus retval;

    i2c_params.bIsRead       = LW_FALSE;
    i2c_params.messageLength = 1;
    i2c_params.flags =
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _START, _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _STOP, _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _ADDRESS_MODE, _7BIT) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, _ONE) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _FLAVOR, _HW) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, _100KHZ);

    i2c_params.index[0]      = reg;
    
    // Select carrier board
    switch (pcsId)
    {
        case 0:
        {
            // Assert reset on IO expander for P4791_B00 parts
            i2c_params.port    = lwswitch_io_expander_list_P4791_B00[expander].i2cPortLogical;
            i2c_params.address = lwswitch_io_expander_list_P4791_B00[expander].i2cAddress;
            break;
        }
        case 1:
        {
            // Assert reset on IO expander for P4790_B00 parts
            i2c_params.port    = lwswitch_io_expander_list_P4790_B00[expander].i2cPortLogical;
            i2c_params.address = lwswitch_io_expander_list_P4790_B00[expander].i2cAddress;
            break;
        }
        default:
        {
            LWSWITCH_ASSERT(0);
            break;
        }
    }

    i2c_params.message[0] = value;
    
    retval = lwswitch_ctrl_i2c_indexed(device, &i2c_params);
    if (retval != LWL_SUCCESS)
    {
        return retval;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_cci_pcs_read_P479X_B00
(
    lwswitch_device *device,
    LwU32 pcsId,
    LwU8  expander,
    LwU8  reg,
    LwU8* pValue
)
{
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS i2c_params = { 0 };
    LwlStatus retval;

    i2c_params.bIsRead       = LW_TRUE;
    i2c_params.messageLength = 1;
    i2c_params.flags =
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _START, _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _STOP, _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _ADDRESS_MODE, _7BIT) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, _ONE) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _FLAVOR, _HW) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, _100KHZ);

    i2c_params.index[0]      = reg;
    
    switch (pcsId)
    {
        case 0:
        {
            // IO expander for P4791_B00 parts
            i2c_params.port    = lwswitch_io_expander_list_P4791_B00[expander].i2cPortLogical;
            i2c_params.address = lwswitch_io_expander_list_P4791_B00[expander].i2cAddress;
            break;
        }
        case 1:
        {
            // IO expander for P4790_B00 parts
            i2c_params.port    = lwswitch_io_expander_list_P4790_B00[expander].i2cPortLogical;
            i2c_params.address = lwswitch_io_expander_list_P4790_B00[expander].i2cAddress;
            break;
        }
        default:
        {
            LWSWITCH_ASSERT(0);
            break;
        }
    }
    
    retval = lwswitch_ctrl_i2c_indexed(device, &i2c_params);
    if (retval != LWL_SUCCESS)
    {
        return retval;
    }

    if (pValue != NULL)
    {
        *pValue = i2c_params.message[0];
    }

    return LWL_SUCCESS;
}

static LwlStatus
_cci_pcs_reset_P479X_B00
(
    lwswitch_device *device,
    LwU32 cciModuleMask
)
{
    PCCI pCci = device->pCci;
    LwU32 pcsModuleMask;
    LwU8 pcsId;
    LwlStatus retval;

    pcsModuleMask = 0;

    FOR_EACH_INDEX_IN_MASK(64, pcsId, pCci->pcsMask)
    {   
        // Present bits are set low if OSFPs are present, so ilwert polarity of data
        retval = _cci_pcs_write_P479X_B00(device, pcsId, P479X_IO_EXPANDER_PRESENT_INTERRUPT,
                                          IO_EXPANDER_POLARITY_REG, 0x0F);
        if (retval != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: PCS %d polarity IO Expander write failed\n",
                __FUNCTION__, pcsId);
            return retval;
        }

        pcsModuleMask = CCI_TO_PCS_MODULE_MASK_MAP_P479X_B00(cciModuleMask, pcsId);
        retval = _cci_pcs_write_P479X_B00(device, pcsId, P479X_IO_EXPANDER_RESET_LOW_POWER, 
                            IO_EXPANDER_CONFIG_REG, IO_EXPANDER_CONFIG_ASSERT_RESET(pcsModuleMask));
        if (retval != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: PCS %d reset assert failed\n",
                __FUNCTION__, pcsId);
            return retval;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    lwswitch_os_sleep(500);

    FOR_EACH_INDEX_IN_MASK(64, pcsId, pCci->pcsMask)
    {    
        retval = _cci_pcs_write_P479X_B00(device, pcsId, P479X_IO_EXPANDER_RESET_LOW_POWER, 
                            IO_EXPANDER_CONFIG_REG, IO_EXPANDER_CONFIG_DEASSERT_RESET);
        if (retval != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: PCS %d reset de-assert failed\n",
                __FUNCTION__, pcsId);
            return retval;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    lwswitch_os_sleep(3000);
    
    return LWL_SUCCESS;
}

static LwlStatus 
_cci_pcs_write_E3597_B00
(
    lwswitch_device *device,
    LwU32 pcsId,
    LwU8  expander,
    LwU8  configReg,
    LwU8  value
)
{
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS i2c_params = { 0 };
    LwlStatus retval;

    i2c_params.bIsRead       = LW_FALSE;
    i2c_params.messageLength = 1;
    i2c_params.flags =
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _START, _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _STOP, _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _ADDRESS_MODE, _7BIT) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, _ONE) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _FLAVOR, _HW) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, _100KHZ);

    i2c_params.index[0]      = configReg;
    
    // Select carrier board
    switch (pcsId)
    {
        case 0:
        {
            i2c_params.port    = lwswitch_io_expander_list_wolf_B[expander].i2cPortLogical;
            i2c_params.address = lwswitch_io_expander_list_wolf_B[expander].i2cAddress;
            break;
        }
        case 1:
        {
            i2c_params.port    = lwswitch_io_expander_list_wolf_C[expander].i2cPortLogical;
            i2c_params.address = lwswitch_io_expander_list_wolf_C[expander].i2cAddress;
            break;
        }
        default:
        {
            LWSWITCH_ASSERT(0);
            break;
        }
    }
    
    i2c_params.message[0] = value;
    
    retval = lwswitch_ctrl_i2c_indexed(device, &i2c_params);
    if (retval != LWL_SUCCESS)
    {
        return retval;
    }

    return LWL_SUCCESS;
}

static LwlStatus 
_cci_pcs_read_E3597_B00
(
    lwswitch_device *device,
    LwU32 pcsId,
    LwU8  expander,
    LwU8  reg,
    LwU8* pValue
)
{
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS i2c_params = { 0 };
    LwlStatus retval;

    i2c_params.bIsRead       = LW_TRUE;
    i2c_params.messageLength = 1;
    i2c_params.flags =
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _START, _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _STOP, _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _ADDRESS_MODE, _7BIT) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, _ONE) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _FLAVOR, _HW) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, _100KHZ);

    i2c_params.index[0]      = reg;
    
    // Select carrier board
    switch (pcsId)
    {
        case 0:
        {
            i2c_params.port    = lwswitch_io_expander_list_wolf_B[expander].i2cPortLogical;
            i2c_params.address = lwswitch_io_expander_list_wolf_B[expander].i2cAddress;
            break;
        }
        case 1:
        {
            i2c_params.port    = lwswitch_io_expander_list_wolf_C[expander].i2cPortLogical;
            i2c_params.address = lwswitch_io_expander_list_wolf_C[expander].i2cAddress;
            break;
        }
        default:
        {
            LWSWITCH_ASSERT(0);
            break;
        }
    }
    
    retval = lwswitch_ctrl_i2c_indexed(device, &i2c_params);
    if (retval != LWL_SUCCESS)
    {
        return retval;
    }

    if (pValue != NULL)
    {
        *pValue = i2c_params.message[0];
    }

    return LWL_SUCCESS;
}

static LwlStatus
_cci_pcs_reset_E3597_B00
(
    lwswitch_device *device,
    LwU32 cciModuleMask
)
{
    PCCI pCci = device->pCci;
    LwU32 pcsModuleMask;
    LwU8 pcsId;
    LwU8 pcsWriteVal;
    LwlStatus retval;

    pcsModuleMask = 0;
    FOR_EACH_INDEX_IN_MASK(64, pcsId, pCci->pcsMask)
    {   
        pcsModuleMask = CCI_TO_PCS_MODULE_MASK_MAP_E3597_B00(cciModuleMask, pcsId);

        // TODO: for individual module reset, configure operations can be moved since they only need to be done once 

        // Configure power enable IO expander as output
        retval = _cci_pcs_write_E3597_B00(device, pcsId, WOLF_IO_EXPANDER_POWER_ENABLE,
                                          IO_EXPANDER_CONFIG_REG, 0);
        if (retval != LWL_SUCCESS)
        {
            return retval;
        }

        // Configure reset IO expander as output
        retval = _cci_pcs_write_E3597_B00(device, pcsId, WOLF_IO_EXPANDER_RESET,
                                          IO_EXPANDER_CONFIG_REG, 0);
        if (retval != LWL_SUCCESS)
        {
            return retval;
        }

        // Configure low power IO expander as output
        retval = _cci_pcs_write_E3597_B00(device, pcsId, WOLF_IO_EXPANDER_LOW_POWER,
                                          IO_EXPANDER_CONFIG_REG, 0);
        if (retval != LWL_SUCCESS)
        {
            return retval;
        }

        // Configure present IO expander as input
        retval = _cci_pcs_write_E3597_B00(device, pcsId, WOLF_IO_EXPANDER_PRESENT,
                                          IO_EXPANDER_CONFIG_REG, 0xFF);
        if (retval != LWL_SUCCESS)
        {
            return retval;
        }

        //
        // Present bits are set low if OSFPs are present, so ilwert polarity of data
        //
        retval = _cci_pcs_write_E3597_B00(device, pcsId, WOLF_IO_EXPANDER_PRESENT,
                                          IO_EXPANDER_POLARITY_REG, 0xFF);
        if (retval != LWL_SUCCESS)
        {
            return retval;
        }

        // Configure interrupt IO expander as input
        retval = _cci_pcs_write_E3597_B00(device, pcsId, WOLF_IO_EXPANDER_INTERRUPT,
                                          IO_EXPANDER_CONFIG_REG, 0xFF);
        if (retval != LWL_SUCCESS)
        {
            return retval;
        }

        // Configure power good IO expander as input
        retval = _cci_pcs_write_E3597_B00(device, pcsId, WOLF_IO_EXPANDER_POWER_GOOD,
                                          IO_EXPANDER_CONFIG_REG, 0xFF);
        if (retval != LWL_SUCCESS)
        {
            return retval;
        }

        // Set power enable output to high
        retval = _cci_pcs_write_E3597_B00(device, pcsId, WOLF_IO_EXPANDER_POWER_ENABLE,
                                          IO_EXPANDER_OUTPUT_REG, 0xFF);
        if (retval != LWL_SUCCESS)
        {
            return retval;
        }

        // Assert reset(active low)
        pcsWriteVal = ~pcsModuleMask;
        retval = _cci_pcs_write_E3597_B00(device, pcsId, WOLF_IO_EXPANDER_RESET,
                                          IO_EXPANDER_OUTPUT_REG, pcsWriteVal);
        if (retval != LWL_SUCCESS)
        {
            return retval;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    lwswitch_os_sleep(500);

    FOR_EACH_INDEX_IN_MASK(64, pcsId, pCci->pcsMask)
    {    
        pcsModuleMask = CCI_TO_PCS_MODULE_MASK_MAP_E3597_B00(cciModuleMask, pcsId);
        pcsWriteVal = pcsModuleMask;

        // Deassert reset
        retval = _cci_pcs_write_E3597_B00(device, pcsId, WOLF_IO_EXPANDER_RESET,
                                          IO_EXPANDER_OUTPUT_REG, pcsWriteVal);
        if (retval != LWL_SUCCESS)
        {
            return retval;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    lwswitch_os_sleep(500);

    FOR_EACH_INDEX_IN_MASK(64, pcsId, pCci->pcsMask)
    {    
        // Set low power output to false
        retval = _cci_pcs_write_E3597_B00(device, pcsId, WOLF_IO_EXPANDER_LOW_POWER,
                                          IO_EXPANDER_OUTPUT_REG, 0xFF);
        if (retval != LWL_SUCCESS)
        {
            return retval;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    lwswitch_os_sleep(3000);

    return LWL_SUCCESS;
}

static LwlStatus 
_lwswitch_cci_get_module_mask
(
    lwswitch_device *device,
    LwU64 linkMask,
    LwU32 *pModuleMask  
)
{
    LwU32 cciModuleMask;
    LwU32 moduleId;
    LwU8 linkId;
    LwlStatus retval;

    cciModuleMask = 0;

    FOR_EACH_INDEX_IN_MASK(64, linkId, linkMask)
    {    
        retval = _lwswitch_cci_get_module_id(device, linkId, &moduleId);
        if (retval != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Failed to get moduleId associated with link %d\n",
                __FUNCTION__, linkId);
            return retval;
        }

        cciModuleMask |= LWBIT32(moduleId);
    }
    FOR_EACH_INDEX_IN_MASK_END;

    *pModuleMask = cciModuleMask;
    
    return LWL_SUCCESS;
}

static LwlStatus 
_lwswitch_cci_reset_links
(
    lwswitch_device *device,
    LwU64 linkMask
)
{
    PCCI pCci = device->pCci;
    LwU32 cciModuleMask;
    LwlStatus retval;

    cciModuleMask = 0;

    if (pCci->boardPartitionType == LWSWITCH_BOARD_PARTITION_E4760_A00 ||
        pCci->boardPartitionType == LWSWITCH_BOARD_PARTITION_E4761_A00)
    {
        LWSWITCH_PRINT(device, ERROR,
                "%s: Individual link reset not supported\n",
                __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    // Generate a mask of modules that need to be reset
    retval = _lwswitch_cci_get_module_mask(device, linkMask, &cciModuleMask);
    if (retval != LWL_SUCCESS)
    {
        return retval;
    }

    LWSWITCH_PRINT(device, INFO,
                "%s: Resetting CCI: module mask 0x%x, link mask 0x%llx\n",
                __FUNCTION__, cciModuleMask, linkMask);

    // I2C reset for B00 carriers
    if (pCci->boardPartitionType == LWSWITCH_BOARD_PARTITION_P4790_B00 ||
        pCci->boardPartitionType == LWSWITCH_BOARD_PARTITION_P4791_B00 ||
        pCci->boardPartitionType == LWSWITCH_BOARD_PARTITION_E3597_B00)
    {
        switch(device->pCci->boardId)
        {
            case LWSWITCH_BOARD_ID_DELTA:
            {
                retval = _cci_pcs_reset_P479X_B00(device, cciModuleMask);
                if (retval != LWL_SUCCESS)
                {
                    LWSWITCH_PRINT(device, ERROR,
                        "%s: PCS reset failed\n",
                        __FUNCTION__);
                    return retval;
                }
                break;
            }
            case LWSWITCH_BOARD_ID_WOLF:
            {
                retval = _cci_pcs_reset_E3597_B00(device, cciModuleMask);
                if (retval != LWL_SUCCESS)
                {
                    LWSWITCH_PRINT(device, ERROR,
                        "%s: PCS reset failed\n",
                        __FUNCTION__);
                    return retval;
                }
                break;
            }
            default:
                break;
        }
    } 

    return LWL_SUCCESS;
}

LwlStatus 
cciResetLinks
(
    lwswitch_device *device,
    LwU64 linkMask
)
{
    LwU32 cciModuleMask;
    LwlStatus retval;

    cciModuleMask = 0;

    if ((device->pCci == NULL) || (!device->pCci->bInitialized))    
    {
        LWSWITCH_PRINT(device, ERROR,
                "%s: CCI not supported\n",
                __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    retval = _lwswitch_cci_reset_links(device, linkMask);
    if (retval != LWL_SUCCESS)
    {
        return retval;
    }

    retval = _lwswitch_cci_get_module_mask(device, linkMask, &cciModuleMask);
    if (retval != LWL_SUCCESS)
    {
        return retval;
    }

    // Re-apply control set values for modules that were reset
    retval = cciApplyControlSetValues(device, 0, cciModuleMask);
    if (retval != LWL_SUCCESS)
    {
        return retval;
    }

    return retval;
}

LwlStatus 
cciGetLinkPartners
(
    lwswitch_device *device,
    LwU8 linkId,
    LwU64* pLinkMask
)
{
    PCCI pCci = device->pCci;
    LwU64 linkMask;
    LwU32 moduleId;
    LwU32 i;
    LwlStatus retval;

    if ((pCci == NULL) || (!pCci->bSupported))
    {
        LWSWITCH_PRINT(device, ERROR,
                "%s: CCI not supported\n",
                __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    linkMask = 0;

    retval = _lwswitch_cci_get_module_id(device, linkId, &moduleId);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get moduleId associated with link %d\n",
            __FUNCTION__, linkId);
        return retval;
    }

    // Return mask of all links associated with given module
    for (i = 0; i < pCci->osfp_map_size; i++)
    {
        if (pCci->osfp_map[i].moduleId == moduleId)
        {
            linkMask |= LWBIT64(pCci->osfp_map[i].linkId);
        }
    }

    if (pLinkMask != NULL)
    {
        *pLinkMask = linkMask;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_cciProcessCmd
(
    lwswitch_device *device,
    LwU32            client,
    LwU32            osfp,
    LwU32            addr,
    LwU32            length,
    LwU8            *pValArray,
    LwBool           bRead,
    LwBool           bBlk
)
{
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS i2cIndexed = { 0 };
    LwlStatus retval;
    PCCI pCci = device->pCci;

    i2cIndexed.bIsRead = bRead;
    i2cIndexed.port = pCci->osfp_i2c_info[osfp].i2cPortLogical;
    i2cIndexed.address = (LwU16) pCci->osfp_i2c_info[osfp].i2cAddress;
    i2cIndexed.messageLength = length;
    i2cIndexed.acquirer = client; 
    i2cIndexed.flags =
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _START, _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _STOP, _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _ADDRESS_MODE, _7BIT) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, _ONE) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _FLAVOR, _HW) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, _100KHZ) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _TRANSACTION_MODE, _NORMAL);
    i2cIndexed.flags |= bBlk ?
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _BLOCK_PROTOCOL, _ENABLED) :
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _BLOCK_PROTOCOL, _DISABLED);
    i2cIndexed.flags |= bRead ?
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _RESTART, _SEND) :
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _RESTART, _NONE);
    i2cIndexed.index[0] = addr;

    if (!bRead)
    {
        lwswitch_os_memcpy(i2cIndexed.message, pValArray, length);
    }

    retval = lwswitch_ctrl_i2c_indexed(device, &i2cIndexed);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, MMIO,
            "%s: I2C command to osfp[%d](addr : 0x%x, port : %d) failed\n",
            __FUNCTION__, osfp,
            pCci->osfp_i2c_info[osfp].i2cAddress,
            pCci->osfp_i2c_info[osfp].i2cPortLogical);
        return retval;
    }

    if (bRead)
    {
        lwswitch_os_memcpy(pValArray, i2cIndexed.message, length);
    }

    return LWL_SUCCESS;
}

LwlStatus
cciWrite
(
    lwswitch_device *device,
    LwU32 client,
    LwU32 osfp,
    LwU32 addr,
    LwU32 length,
    LwU8 *pVal
)
{
    LwlStatus status = LWL_SUCCESS;
    LWSWITCH_TIMEOUT timeout;
    LwBool bRead = LW_FALSE;
    LwBool bBlk = LW_FALSE;

    if (!pVal)
    {
        LWSWITCH_PRINT(device, ERROR,
             "%s: Bad Args!\n",
            __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    if (!device->pCci->bInitialized)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI is not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (!_lwswitch_cci_module_present(device, osfp))
    {
        return -LWL_NOT_FOUND;
    }

    lwswitch_timeout_create(5 * LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);

    do
    {
        status = _cciProcessCmd(device, client, osfp, addr, length, pVal, bRead, bBlk);
        if (status == LWL_SUCCESS)
        {
            return status;
        }

        if (lwswitch_timeout_check(&timeout))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Timeout waiting for CMIS write to complete! rc = %d.\n",
                __FUNCTION__, status);

            LWSWITCH_PRINT(device, ERROR,
                "%s: Write to register 0x%x failed on osfp[%d](addr : 0x%x, port : %d)\n",
                __FUNCTION__, addr, osfp,
                device->pCci->osfp_i2c_info[osfp].i2cAddress,
                device->pCci->osfp_i2c_info[osfp].i2cPortLogical);
            return -LWL_IO_ERROR;
        }

        lwswitch_os_sleep(10);
    } while (LW_TRUE);

    return LWL_SUCCESS;
}

LwlStatus
cciWriteBlk
(
    lwswitch_device *device,
    LwU32 client,
    LwU32 osfp,
    LwU32 addr,
    LwU32 length,
    LwU8 *pValArray
)
{
    LwlStatus status = LWL_SUCCESS;
    LWSWITCH_TIMEOUT timeout;
    LwBool bRead = LW_FALSE;
    LwBool bBlk = LW_TRUE;

    if (!pValArray)
    {
        LWSWITCH_PRINT(device, ERROR,
             "%s: Bad Args!\n",
            __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    if (!device->pCci->bInitialized)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI is not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (!_lwswitch_cci_module_present(device, osfp))
    {
        return -LWL_NOT_FOUND;
    }

    lwswitch_timeout_create(5 * LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);

    do
    {
        status = _cciProcessCmd(device, client, osfp, addr, length, pValArray, bRead, bBlk);
        if (status == LWL_SUCCESS)
        {
            return status;
        }

        if (lwswitch_timeout_check(&timeout))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Timeout waiting for CMIS write blk to complete! rc = %d\n",
                __FUNCTION__, status);

            LWSWITCH_PRINT(device, ERROR,
                "%s: Write Blk to register 0x%x failed on osfp[%d](addr : 0x%x, port : %d)\n",
                __FUNCTION__, addr, osfp,
                device->pCci->osfp_i2c_info[osfp].i2cAddress,
                device->pCci->osfp_i2c_info[osfp].i2cPortLogical);
            return -LWL_IO_ERROR;
        }

        lwswitch_os_sleep(10);
    } while (LW_TRUE);

    return status;
}

LwlStatus
cciRead
(
    lwswitch_device *device,
    LwU32 client,
    LwU32 osfp,
    LwU32 addr,
    LwU32 length,
    LwU8 *pVal
)
{
    LwlStatus status = LWL_SUCCESS;
    LWSWITCH_TIMEOUT timeout;
    LwBool bRead = LW_TRUE;
    LwBool bBlk = LW_FALSE;

    if (!pVal)
    {
        LWSWITCH_PRINT(device, ERROR,
             "%s: Bad Args!\n",
            __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    if (!device->pCci->bInitialized)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI is not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (!_lwswitch_cci_module_present(device, osfp))
    {
        return -LWL_NOT_FOUND;
    }

    lwswitch_timeout_create(5 * LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);

    do
    {
        status = _cciProcessCmd(device, client, osfp, addr, length, pVal, bRead, bBlk);
        if (status == LWL_SUCCESS)
        {
            return status;
        }

        if (lwswitch_timeout_check(&timeout))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Timeout waiting for CMIS read to complete! rc = %d\n",
                __FUNCTION__, status);

            LWSWITCH_PRINT(device, ERROR,
                "%s: Read to register 0x%x failed on osfp[%d](addr : 0x%x, port : %d)\n",
                __FUNCTION__, addr, osfp,
                device->pCci->osfp_i2c_info[osfp].i2cAddress,
                device->pCci->osfp_i2c_info[osfp].i2cPortLogical);
            return -LWL_IO_ERROR;
        }

        lwswitch_os_sleep(10);
    } while (LW_TRUE);

    return LWL_SUCCESS;
}

LwlStatus
cciReadBlk
(
    lwswitch_device *device,
    LwU32 client,
    LwU32 osfp,
    LwU32 addr,
    LwU32 length,
    LwU8 *pValArray
)
{
    LwlStatus status = LWL_SUCCESS;
    LWSWITCH_TIMEOUT timeout;
    LwBool bRead = LW_TRUE;
    LwBool bBlk = LW_TRUE;

    if (!pValArray)
    {
        LWSWITCH_PRINT(device, ERROR,
             "%s: Bad Args!\n",
            __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    if (!device->pCci->bInitialized)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI is not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (!_lwswitch_cci_module_present(device, osfp))
    {
        return -LWL_NOT_FOUND;
    }

    lwswitch_timeout_create(5 * LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);

    do
    {
        status = _cciProcessCmd(device, client, osfp, addr, length, pValArray, bRead, bBlk);
        if (status == LWL_SUCCESS)
        {
            return status;
        }

        if (lwswitch_timeout_check(&timeout))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Timeout waiting for CMIS read blk to complete! rc = %d\n",
                __FUNCTION__, status);

            LWSWITCH_PRINT(device, ERROR,
                "%s: Read to register 0x%x failed on osfp[%d](addr : 0x%x, port : %d)\n",
                __FUNCTION__, addr, osfp,
                device->pCci->osfp_i2c_info[osfp].i2cAddress,
                device->pCci->osfp_i2c_info[osfp].i2cPortLogical);
            return -LWL_IO_ERROR;
        }

        lwswitch_os_sleep(10);
    } while (LW_TRUE);

    return LWL_SUCCESS;
}

/*
 * @brief Set bank an page in the CMIS memory table.
 *
 * CMIS4 states, "For a bank change, the host shall write the Bank Select
 * and Page Select registers in the same TWI transaction".
 *
 * Write to Page 0h, byte 126 sets the bank and page.
 */
LwlStatus
cciSetBankAndPage
(
    lwswitch_device *device,
    LwU32 client,
    LwU32 osfp,
    LwU8 bank,
    LwU8 page
)
{
    LwU8 temp[2];

    temp[0] = bank;
    temp[1] = page;

    return cciWrite(device, client, osfp, 126, 2, temp);
}

/*
 * @brief Gets the current bank and page in the CMIS memory table.
 *
 * Read from Page 0h, byte 126 to get the bank and page.
 */
LwlStatus
cciGetBankAndPage
(
    lwswitch_device *device,
    LwU32 client,
    LwU32 osfp,
    LwU8 *pBank,
    LwU8 *pPage
)
{
    LwlStatus status;
    LwU8 temp[2];

    status = cciRead(device, client, osfp, 126, 2, temp);

    if (pBank != NULL)
    {
        *pBank = temp[0];
    }

    if (pPage != NULL)
    {
        *pPage = temp[1];
    }

    return status;
}

#define LWSWITCH_CCI_MAX_CDB_LENGTH 128

/*
 * @brief Send commands for Command Data Block(CDB) communication.
 *
 * CDB reads and writes are performed on memory map pages 9Fh-AFh.
 *
 * Page 9Fh is used to specify the CDB command and use
 * local payload (LPL) of 120 bytes.
 *
 * Pages A0h-AFh contain up to 2048 bytes of extended payload (EPL).
 *
 * Payload may be a zero-length array if no payload to send.
 *
 * (ref CMIS rev4.0, sections 7.2, 8.13, 8.2.7, 8.4.11).
 */
LwlStatus
cciSendCDBCommand
(
    lwswitch_device *device,
    LwU32 client,
    LwU32 osfp,
    LwU32 command,
    LwU32 length,
    LwU8 *payload,
    LwBool padding
)
{
    LwU8 cmd_msb = (command >> 8) & 0xff;
    LwU8 cmd_lsb = command & 0xff;
    LwU32 chkcode;
    LwU32 i;
    LwU8 temp[8];

    if (length > LWSWITCH_CCI_MAX_CDB_LENGTH)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Command length %d exceeded max length %d. "
            "CDB yet to support extended payloads\n",
            __FUNCTION__, length, LWSWITCH_CCI_MAX_CDB_LENGTH);
        return -LWL_ERR_GENERIC;
    }

    //
    // Compute checksum over payload, including header bytes(command, length,...)
    //
    // CdbChkCode is the ones complement of the summation of page 9Fh,
    // bytes 128 to (135+LPL_Length) with bytes 133,134 and 135 equal to 0.
    // (ref CMIS 4.0, sec 8.13)
    //
    chkcode = cmd_msb + cmd_lsb + length;

    // now add payload bytes
    if (length > 0)
    {
        for (i = 0; i < length; i++)
        {
            chkcode += payload[i];
        }
    }
    chkcode = (~(chkcode & 0xff)) & 0xff;

    // # nw: page 198 of cmis4 spec.
    // # nw:

    // Set page to 0x9F to setup CDB command
    cciSetBankAndPage(device, client, osfp, 0, 0x9f);

    //
    // Send CDB message
    //
    // Fill page 9Fh bytes 128-135 in the order -
    // [cmd_msb, cmd_lsb, epl_length msb, epl_length lsb, lpl_length,
    //  chkcode, resp_length = 0, resp_chkcode = 0]
    //
    // LPL starts from Bytes 136 upto 120 bytes.
    // The command is triggered when byte 129 is written. So fill bytes 128, 129 at the end.
    //

    // #1. Write bytes 130-135. The "header portion", minus the first two bytes
    // which is the command code.
    temp[0] = 0;       // epl_length msb
    temp[1] = 0;       // epl_length lsb
    temp[2] = length;  // lpl_length
    temp[3] = chkcode; // cdb chkcode
    temp[4] = 0;       // response length
    temp[5] = 0;       // response chkcode
    cciWrite(device, client, osfp, 130, 6, temp);

    // #2. If payload's not empty, write the payload (bytes 136 to 255).
    // If payload is empty, infer the command is payload-less and skip.
    if ((length > 0) && padding)
    {
        for (i = length; i < 120; i++)
        {
            payload[i] = 0;
        }
        cciWrite(device, client, osfp, 136, 120, payload);
    }
    else if ((length > 0) && !padding)
    {
        cciWrite(device, client, osfp, 136, length, payload);
    }

    // # 3. Write the command code (bytes 128,129), which additionally
    // kicks off processing of the command by the module.
    temp[0] = cmd_msb;
    temp[1] = cmd_lsb;
    cciWrite(device, client, osfp, 128, 2, temp);

    return LWL_SUCCESS;
}

/*!
 * @brief Waits for CDB command completion and returns status.
 *
 * Page 00h byte 37 contains status bits.
 * (see CMIS rev4.0, Table 9-3, CDB Command 0000h: QUERY-Status)
 */
LwlStatus
cciGetCDBStatus
(
    lwswitch_device *device,
    LwU32 client,
    LwU32 osfp,
    LwU8 *pStatus
)
{
    LwBool status_busy;
    LwBool status_fail;
    LwU8 cdb_result;
    LwU8 status;
    LWSWITCH_TIMEOUT timeout;

    lwswitch_timeout_create(10 * LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);

    do
    {
        cciRead(device, client, osfp, 37, 1, &status);
        *pStatus = status;

        // Quit when the STS_BUSY bit goes to 0
        if ((status & 0x80) == 0)
        {
            break;
        }

        if (lwswitch_timeout_check(&timeout))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Timeout waiting for CDB command to complete! "
                "STATUS = 0x%x\n",
                __FUNCTION__, status);
            break;
        }

        lwswitch_os_sleep(10);
    } while (LW_TRUE);

    status_busy = (status >> 7) & 0x1;
    status_fail = (status >> 6) & 0x1;
    cdb_result =  status & 0x3f;

    if (status_busy) // status is busy
    {
        if (cdb_result == 0x01)
        {
            LWSWITCH_PRINT(device, INFO,
                "%s: CDB status = BUSY. Last Command Result: "
                "'Command is captured but not processed'\n",
                __FUNCTION__);
        }
        else if (cdb_result == 0x02)
        {
            LWSWITCH_PRINT(device, INFO,
                "%s: CDB status = BUSY. Last Command Result: "
                "'Command checking is in progress'\n",
                __FUNCTION__);
        }
        else if (cdb_result == 0x03)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: CDB status = BUSY. Last Command Result: "
                "'Command exelwtion is in progress'\n",
                __FUNCTION__);
        }
        else
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: CDB status = BUSY. Last Command Result: "
                "Unknown (0x%x)\n",
                __FUNCTION__, cdb_result);
        }

        return -LWL_ERR_GENERIC;
    }

    if (status_fail) // status failed
    {
        if (cdb_result == 0x01)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: CDB status = FAIL. Last Command Result: "
                "'CMD Code unknown'\n",
                __FUNCTION__);
        }
        else if (cdb_result == 0x02)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: CDB status = FAIL. Last Command Result: "
                "'Parameter range error or not supported'\n",
                __FUNCTION__);
        }
        else if (cdb_result == 0x03)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: CDB status = FAIL. Last Command Result: "
                "'Previous CMD was not ABORTED by CMD Abort'\n",
                __FUNCTION__);
        }
        else if (cdb_result == 0x04)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: CDB status = FAIL. Last Command Result: "
                "'Command checking time out'\n",
                __FUNCTION__);
        }
        else if (cdb_result == 0x05)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: CDB status = FAIL. Last Command Result: "
                "'CdbCheckCode Error'\n",
                __FUNCTION__);
        }
        else if (cdb_result == 0x06)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: CDB status = FAIL. Last Command Result: "
                "'Password Error'\n",
                __FUNCTION__);
        }
        else
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: CDB status = FAIL. Last Command Result: "
                "Unknown (0x%x)\n",
                __FUNCTION__, cdb_result);
        }

        return -LWL_ERR_GENERIC;
    }

    return LWL_SUCCESS;
}

/*!
 * @brief Waits for CDB command completion.
 *
 * Page 00h byte 37 contains status bits. BIT 7 is the busy bit.
 * (see CMIS rev4.0, Table 9-3, CDB Command 0000h: QUERY-Status)
 */
LwlStatus
cciWaitForCDBComplete
(
    lwswitch_device *device,
    LwU32 client,
    LwU32 osfp
)
{
    LWSWITCH_TIMEOUT timeout;
    LwU8 status;

    lwswitch_timeout_create(LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);

    do
    {
        cciRead(device, client, osfp, 37, 1, &status);

        // Return when the STS_BUSY bit goes to 0
        if ((status & 0x80) == 0)
        {
            return LWL_SUCCESS;
        }

        if (lwswitch_timeout_check(&timeout))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Timeout waiting for CDB command to complete! STATUS = 0x%x\n",
                __FUNCTION__, status);
            return -LWL_ERR_GENERIC;
        }

        lwswitch_os_sleep(10);
    } while (LW_TRUE);
}

/*!
 * @brief Get the CDB response data.
 *
 * This function must be sent after CDB status is success.
 *
 * Page 9Fh, bytes 134-255 contains response data
 *   Byte 134 : Response LPL Length of the data returned by CDB command code.
 *   Byte 135 : Response LPL ChkCode
 *   Bytes 136-255 : Local payload of the module response.
 */
LwlStatus
cciGetCDBResponse
(
    lwswitch_device *device,
    LwU32 client,
    LwU32 osfp,
    LwU8 *response,
    LwU32 *resLength
)
{
    LwU8 header[8];
    LwU32 rlpllen;    // Response local payload length
    LwU8 rlplchkcode; // Response local payload check code
    LwU8 chksum = 0;
    LwU32 i;
    LwBool bSkipChecksum = LW_FALSE;

    cciSetBankAndPage(device, client, osfp, 0, 0x9f);

    // get header
    cciRead(device, client, osfp, 128, 8, header);

    // get reported response length
    rlpllen = header[6];
    rlplchkcode = header[7];

    // TODO : Remove this once FW support improves
    if (rlpllen == 0)
    {
        // bug with earlier Stallion FW, presume hit it an read maximum-sized lpl.
        // Assume the maximum length of 120 and skip checksum because it will also
        // be zero
        rlpllen = 120;
        bSkipChecksum = LW_TRUE;
    }

    if (rlpllen != 0)
    {
        // get response
        cciRead(device, client, osfp, 136, rlpllen, response);

        if (!bSkipChecksum)
        {
            // compute checksum of response
            for (i = 0; i < rlpllen; i++)
            {
                chksum += response[i];
            }

            // and compare against rlplchkcode (byte 7 of page 9Fh)
            if ((~chksum & 0xff) != rlplchkcode)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: Error: RLPLChkCode incorrect for returned data\n",
                    __FUNCTION__);
                return -LWL_ERR_GENERIC;
            }
        }
    }

    *resLength = rlpllen;
    
    return LWL_SUCCESS;
}

/*!
 * @brief Get the CDB command and get response.
 */
LwlStatus
cciSendCDBCommandAndGetResponse
(
    lwswitch_device *device,
    LwU32 client,
    LwU32 osfp,
    LwU32 command,
    LwU32 payLength,
    LwU8 *payload,
    LwU32 *resLength,
    LwU8 *response,
    LwBool padding
)
{
    LwlStatus retval;
    LwU8 status = 0;

    if (!_lwswitch_cci_module_present(device, osfp))
    {
        LWSWITCH_PRINT(device, INFO,
                "%s: osfp %d is missing\n",
                __FUNCTION__, osfp);
        return -LWL_NOT_FOUND;
    }

    // Wait for CDB status to be free
    retval = cciWaitForCDBComplete(device, client, osfp);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, INFO,
            "%s: CDB is busy!!\n",
            __FUNCTION__);
        return -LWL_ERR_GENERIC;
    }

    retval = cciSendCDBCommand(device, client, osfp, command, payLength, payload, padding);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, INFO,
            "%s: Failed to send CDB Command: 0x%x\n",
            __FUNCTION__, command);
        return -LWL_ERR_GENERIC;
    }

    retval = cciGetCDBStatus(device, client, osfp, &status);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CDB command failed! result = 0x%x\n",
            __FUNCTION__, status);
        return -LWL_ERR_GENERIC;
    }

    retval = cciGetCDBResponse(device, client, osfp, response, resLength);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get CDB command response\n",
            __FUNCTION__);
        return -LWL_ERR_GENERIC;
    }

    return LWL_SUCCESS;
}

LwlStatus cciRegisterCallback
(
    lwswitch_device *device,
    LwU32 callbackId,
    void (*functionPtr)(lwswitch_device*),
    LwU32 rateHz
)
{
    PCCI pCci = device->pCci;

    if ((callbackId >= LWSWITCH_CCI_CALLBACK_NUM_MAX) ||
        (functionPtr == NULL))
    {
        return -LWL_BAD_ARGS;
    }

    if ((rateHz == 0) || ((LWSWITCH_CCI_POLLING_RATE_HZ % rateHz) != 0))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Input rate must divide main polling rate: %d\n",
            __FUNCTION__, LWSWITCH_CCI_POLLING_RATE_HZ);
        return -LWL_BAD_ARGS;
    }

    if (pCci->callbackList[callbackId].functionPtr != NULL)
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s: CCI callback previously set.\n",
            __FUNCTION__);
    }

    pCci->callbackList[callbackId].interval = LWSWITCH_CCI_POLLING_RATE_HZ/rateHz;
    pCci->callbackList[callbackId].functionPtr = functionPtr;

    return LWL_SUCCESS;
}

// CCI CONTROL CALLS
LwlStatus
lwswitch_ctrl_get_cci_capabilities
(
    lwswitch_device *device,
    LWSWITCH_CCI_GET_CAPABILITIES_PARAMS *pParams
)
{
    return cciGetCapabilities(device, 0, pParams->linkId,
                              &pParams->capabilities);
}

LwlStatus
lwswitch_ctrl_get_cci_temperature
(
    lwswitch_device *device,
    LWSWITCH_CCI_GET_TEMPERATURE *pParams
)
{
    return cciGetTemperature(device, 0, pParams->linkId,
                             &pParams->temperature);
}

LwlStatus
lwswitch_ctrl_get_cci_fw_revisions
(
    lwswitch_device *device,
    LWSWITCH_CCI_GET_FW_REVISION_PARAMS *pParams
)
{
    return cciGetFWRevisions(device, 0, pParams->linkId,
                             pParams->revisions);
}

LwlStatus
lwswitch_ctrl_get_grading_values
(
    lwswitch_device *device,
    LWSWITCH_CCI_GET_GRADING_VALUES_PARAMS *pParams
)
{
    return cciGetGradingValues(device, 0, pParams->linkId,
                               &pParams->laneMask, &pParams->grading);
}

LwlStatus
lwswitch_ctrl_get_module_state
(
    lwswitch_device *device,
    LWSWITCH_CCI_GET_MODULE_STATE *pParams
)
{
    return cciGetModuleState(device, 0, pParams->linkId, &pParams->info);
}

LwlStatus
lwswitch_ctrl_get_module_flags
(
    lwswitch_device *device,
    LWSWITCH_CCI_GET_MODULE_FLAGS *pParams
)
{
    return cciGetModuleFlags(device, 0, pParams->linkId, &pParams->flags);
}

LwlStatus
lwswitch_ctrl_get_voltage
(
    lwswitch_device *device,
    LWSWITCH_CCI_GET_VOLTAGE *pParams
)
{
    return cciGetVoltage(device, 0, pParams->linkId, &pParams->voltage);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * @Brief : Reset CCI on this device.
 *
 * This is temporary and eventually moved to bios.
 * Tracking this in bug 200668898.
 *
 */
static LwlStatus
_lwswitch_reset_cci
(
    lwswitch_device *device
)
{
    return cciResetAllPartitions(device);
}

/*
 * @Brief : Execute CCI pre-reset sequence for secure reset.
 */
static LwlStatus
_lwswitch_cci_prepare_for_reset
(
    lwswitch_device *device
)
{
    return LWL_SUCCESS;
}


//
// E4760 A00 bringup board
//
//
// E4760 A00 bringup board
//
static LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE lwswitch_i2c_device_list_E4760_A00[] =
{
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
};

static const LwU32 lwswitch_i2c_device_list_E4760_A00_size =
    LW_ARRAY_ELEMENTS(lwswitch_i2c_device_list_E4760_A00);

//
// E4761 A00 bringup board
//
static LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE lwswitch_i2c_device_list_E4761_A00[] =
{
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
};

static const LwU32 lwswitch_i2c_device_list_E4761_A00_size =
    LW_ARRAY_ELEMENTS(lwswitch_i2c_device_list_E4761_A00);

//
// Delta board
//
static LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE lwswitch_i2c_device_list_delta[] =
{
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
};

static const LwU32 lwswitch_i2c_device_list_delta_size =
    LW_ARRAY_ELEMENTS(lwswitch_i2c_device_list_delta);

//
// Wolf board
//
LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE lwswitch_i2c_device_list_wolf[] =
{
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xA0, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xC0, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xA2, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xC2, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xA4, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xC4, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xA6, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xC6, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xA0, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xC0, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xA2, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xC2, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xA4, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xC4, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xA6, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xC6, _CMIS4_MODULE,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
};

const LwU32 lwswitch_i2c_device_list_wolf_size =
    LW_ARRAY_ELEMENTS(lwswitch_i2c_device_list_wolf);

//
//  All possible locations of the LED driver modules
//
static LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE lwswitch_led_drivers_and_rom[] =
{
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0x80, _PCA9685BS,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0x80, _PCA9685BS,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0x8A, _TCA6507,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0x8E, _TCA6507,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0x8A, _TCA6507,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0x8E, _TCA6507,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CC, 0xAE, _AT24C02D,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
    LWSWITCH_DESCRIBE_I2C_DEVICE(_I2CB, 0xAE, _AT24C02D,
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _READ_ACCESS_LEVEL, _PUBLIC) |
        DRF_DEF(_LWSWITCH, _I2C_DEVICE, _WRITE_ACCESS_LEVEL, _PUBLIC)),
};

static const LwU32 lwswitch_led_drivers_and_rom_size =
    LW_ARRAY_ELEMENTS(lwswitch_led_drivers_and_rom);

/*
 * Mapping between osfp, linkId, and osfp-lane-mask for E476X.
 *
 * This is temporary and will be removed once PCS and BSP framework
 * is in place
 */
static LWSWITCH_CCI_MODULE_LINK_LANE_MAP lwswitch_cci_osfp_map_E476X[] =
{
    { 1, 12, OSFP_LANE_MASK(0, 1, 2, 3) },
    { 1, 13, OSFP_LANE_MASK(4, 5, 6, 7) },
    { 2, 14, OSFP_LANE_MASK(0, 1, 2, 3) },
    { 2, 15, OSFP_LANE_MASK(4, 5, 6, 7) },
    { 3, 32, OSFP_LANE_MASK(0, 1, 2, 3) },
    { 3, 33, OSFP_LANE_MASK(4, 5, 6, 7) },
    { 5, 34, OSFP_LANE_MASK(0, 1, 2, 3) },
    { 5, 35, OSFP_LANE_MASK(4, 5, 6, 7) },
    { 6, 31, OSFP_LANE_MASK(0, 1, 2, 3) },
    { 6, 30, OSFP_LANE_MASK(4, 5, 6, 7) },
    { 7, 27, OSFP_LANE_MASK(0, 1, 2, 3) },
    { 7, 26, OSFP_LANE_MASK(4, 5, 6, 7) },
};

static const LwU32 lwswitch_cci_osfp_map_E476X_size =
    LW_ARRAY_ELEMENTS(lwswitch_cci_osfp_map_E476X);


/*
 * Mapping between osfp, linkId, and osfp-lane-mask for Delta.
 *
 * This is temporary and will be removed once PCS and BSP framework
 * is in place
 */
static LWSWITCH_CCI_MODULE_LINK_LANE_MAP lwswitch_cci_osfp_map_delta[] =
{
    { 0,  6, OSFP_LANE_MASK(0, 1, 2, 3) },
    { 0,  7, OSFP_LANE_MASK(4, 5, 6, 7) },
    { 1,  4, OSFP_LANE_MASK(0, 1, 2, 3) },
    { 1,  5, OSFP_LANE_MASK(4, 5, 6, 7) },
    { 2,  0, OSFP_LANE_MASK(0, 1, 2, 3) },
    { 2,  1, OSFP_LANE_MASK(4, 5, 6, 7) },
    { 3,  2, OSFP_LANE_MASK(0, 1, 2, 3) },
    { 3,  3, OSFP_LANE_MASK(4, 5, 6, 7) },
    { 4, 20, OSFP_LANE_MASK(0, 1, 2, 3) },
    { 4, 21, OSFP_LANE_MASK(4, 5, 6, 7) },
    { 5, 22, OSFP_LANE_MASK(0, 1, 2, 3) },
    { 5, 23, OSFP_LANE_MASK(4, 5, 6, 7) },
    { 6, 18, OSFP_LANE_MASK(0, 1, 2, 3) },
    { 6, 19, OSFP_LANE_MASK(4, 5, 6, 7) },
    { 7, 16, OSFP_LANE_MASK(0, 1, 2, 3) },
    { 7, 17, OSFP_LANE_MASK(4, 5, 6, 7) },
};

static const LwU32 lwswitch_cci_osfp_map_delta_size =
    LW_ARRAY_ELEMENTS(lwswitch_cci_osfp_map_delta);

/*
 * Mapping between osfp, linkId, and osfp-lane-mask for Wolf.
 *
 * This is temporary and will be removed once PCS and BSP framework
 * is in place
 */
LWSWITCH_CCI_MODULE_LINK_LANE_MAP lwswitch_cci_osfp_map_wolf[] =
{
    {  0, 26, OSFP_LANE_MASK(0, 1, 2, 3) },
    {  0, 27, OSFP_LANE_MASK(4, 5, 6, 7) },
    {  1, 31, OSFP_LANE_MASK(0, 1, 2, 3) },
    {  1, 30, OSFP_LANE_MASK(4, 5, 6, 7) },
    {  2, 33, OSFP_LANE_MASK(0, 1, 2, 3) },
    {  2, 32, OSFP_LANE_MASK(4, 5, 6, 7) },
    {  3, 34, OSFP_LANE_MASK(0, 1, 2, 3) },
    {  3, 35, OSFP_LANE_MASK(4, 5, 6, 7) },
    {  4, 11, OSFP_LANE_MASK(0, 1, 2, 3) },
    {  4, 10, OSFP_LANE_MASK(4, 5, 6, 7) },
    {  5, 14, OSFP_LANE_MASK(0, 1, 2, 3) },
    {  5, 15, OSFP_LANE_MASK(4, 5, 6, 7) },
    {  6,  9, OSFP_LANE_MASK(0, 1, 2, 3) },
    {  6,  8, OSFP_LANE_MASK(4, 5, 6, 7) },
    {  7, 12, OSFP_LANE_MASK(0, 1, 2, 3) },
    {  7, 13, OSFP_LANE_MASK(4, 5, 6, 7) },
    {  8,  0, OSFP_LANE_MASK(0, 1, 2, 3) },
    {  8,  1, OSFP_LANE_MASK(4, 5, 6, 7) },
    {  9,  5, OSFP_LANE_MASK(0, 1, 2, 3) },
    {  9,  4, OSFP_LANE_MASK(4, 5, 6, 7) },
    { 10,  2, OSFP_LANE_MASK(0, 1, 2, 3) },
    { 10,  3, OSFP_LANE_MASK(4, 5, 6, 7) },
    { 11,  7, OSFP_LANE_MASK(0, 1, 2, 3) },
    { 11,  6, OSFP_LANE_MASK(4, 5, 6, 7) },
    { 12, 19, OSFP_LANE_MASK(0, 1, 2, 3) },
    { 12, 18, OSFP_LANE_MASK(4, 5, 6, 7) },
    { 13, 22, OSFP_LANE_MASK(0, 1, 2, 3) },
    { 13, 23, OSFP_LANE_MASK(4, 5, 6, 7) },
    { 14, 17, OSFP_LANE_MASK(0, 1, 2, 3) },
    { 14, 16, OSFP_LANE_MASK(4, 5, 6, 7) },
    { 15, 20, OSFP_LANE_MASK(0, 1, 2, 3) },
    { 15, 21, OSFP_LANE_MASK(4, 5, 6, 7) },
};

const LwU32 lwswitch_cci_osfp_map_wolf_size =
    LW_ARRAY_ELEMENTS(lwswitch_cci_osfp_map_wolf);

static LwlStatus
_lwswitch_cci_setup_link_mask
(
    lwswitch_device *device
)
{
    LwlStatus status;
    LwU64 activeRepeaterMask;
    LwU64 presentLinks;
    LwU64 linkMask;
    LwU8  osfp;

    presentLinks = 0;
    activeRepeaterMask = lwswitch_get_active_repeater_mask(device);

    FOR_EACH_INDEX_IN_MASK(32, osfp, device->pCci->osfpMaskPresent)
    {    
        status = cciGetCageMapping(device, osfp, &linkMask, NULL);
        if (status != LWL_SUCCESS)
        {
            return -LWL_ERR_GENERIC;
        }

        presentLinks |= linkMask;
    }
    FOR_EACH_INDEX_IN_MASK_END;

    device->pCci->linkMask = (activeRepeaterMask & presentLinks);

    LWSWITCH_PRINT(device, INFO,
         "%s: Initial CCI link mask 0x%llx\n", __FUNCTION__,
            device->pCci->linkMask);

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_cci_get_lanemask
(
    lwswitch_device *device,
    LwU32           linkId,
    LwU8           *laneMask
)
{
    LWSWITCH_CCI_MODULE_LINK_LANE_MAP * module_map = device->pCci->osfp_map;
    LwU32 module_map_size = device->pCci->osfp_map_size;
    LwU32 osfp, i;

    for (i = 0; i < module_map_size; i++)
    {
        if (module_map[i].linkId == linkId)
        {
            osfp = module_map[i].moduleId;

            if (!(device->pCci->osfpMaskPresent & LWBIT(osfp)))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: osfp %d associated with link %d is missing\n",
                    __FUNCTION__, linkId, osfp);
                return -LWL_NOT_FOUND;
            }

            *laneMask = module_map[i].laneMask;

            return LWL_SUCCESS;
        }
    }

    return -LWL_NOT_FOUND;
}

/*
 * @brief Compile the list of LED driver modules by pinging all possible slave addresses
 *
 */
static void
_lwswitch_find_led_drivers_and_rom
(
    lwswitch_device *device,
    LwU32            client,
    PCCI             pCci
)
{
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS     i2c_params = { 0 };
    LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE *pDesc;
    unsigned                             idx_src;
    unsigned                             idx_dst_led = 0;
    unsigned                             idx_dst_rom = 0;

    for (idx_src = 0; idx_src < lwswitch_led_drivers_and_rom_size; ++idx_src)
    {
        LwBool  bLed;

        pDesc = &lwswitch_led_drivers_and_rom[idx_src];

        bLed = (pDesc->i2cDeviceType == LWSWITCH_I2C_DEVICE_TCA6507) ||
               (pDesc->i2cDeviceType == LWSWITCH_I2C_DEVICE_PCA9685BS);

        if ((bLed && (idx_dst_led >= LWSWITCH_CCI_LED_DRV_NUM_MAX)) ||
           (!bLed && (idx_dst_rom >= LWSWITCH_CCI_ROM_NUM_MAX)))
        {
            continue;
        }

        i2c_params.bIsRead       = LW_FALSE;
        i2c_params.port          = pDesc->i2cPortLogical;
        i2c_params.acquirer      = client;
        i2c_params.address       = pDesc->i2cAddress;
        i2c_params.messageLength = 0;
        i2c_params.flags =
            DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _START, _SEND) |
            DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _STOP, _SEND) |
            DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _ADDRESS_MODE, _7BIT) |
            DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, _ZERO) |
            DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _FLAVOR, _HW) |
            DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, _100KHZ);

        if (lwswitch_ctrl_i2c_indexed(device, &i2c_params) == LWL_SUCCESS)
        {
            if (bLed)
            {
                pCci->led_drv_i2c_info[idx_dst_led++] = pDesc;
            }
            else
            {
                pCci->rom_i2c_info[idx_dst_rom++] = pDesc;
            }

            LWSWITCH_PRINT(device, INFO,
                "%s: %s %s found at %d/%02x\n",
                __FUNCTION__,
                bLed ? "LED driver" : "ROM",
                bLed ?
                    (pDesc->i2cDeviceType == LWSWITCH_I2C_DEVICE_TCA6507 ?
                        "TCA6507" : "PCA9685BS") :
                    "AT24C02D",
                pDesc->i2cPortLogical,
                pDesc->i2cAddress);
        }
    }

    pCci->led_drv_num = idx_dst_led;
    pCci->rom_num = idx_dst_rom;

    LWSWITCH_PRINT(device, INFO,
         "%s: %d LED driver modules and %d ROMs found\n", __FUNCTION__,
            idx_dst_led, idx_dst_rom);
}

static LwlStatus
_lwswitch_fetch_rom_image
(
    lwswitch_device *device,
    LwU32 rom_idx,
    LwU32 romSize,
    LwU8 *pRomBuf
)
{
    PCCI                                 pCci = device->pCci;
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS     i2c_params = { 0 };
    struct LWSWITCH_I2C_DEVICE_DESCRIPTOR *pDesc;
    LwlStatus                            status;

    if (pCci == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (pCci->rom_num == 0)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (rom_idx >= pCci->rom_num || pRomBuf == NULL)
    {
        return -LWL_BAD_ARGS;
    }

    i2c_params.bIsRead       = LW_TRUE;
    i2c_params.acquirer      = 0;
    i2c_params.messageLength = romSize;
    i2c_params.index[0]      = 0;
    i2c_params.flags =
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _START, _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _STOP, _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _ADDRESS_MODE, _7BIT) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, _ONE) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _FLAVOR, _HW) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, _400KHZ);


    pDesc = pCci->rom_i2c_info[rom_idx];

    if (pDesc->i2cDeviceType != LWSWITCH_I2C_DEVICE_AT24C02D)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: ROM at %d/%02x type %02x is not supported.\n",
            __FUNCTION__, pDesc->i2cPortLogical,
            pDesc->i2cAddress, pDesc->i2cDeviceType);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    i2c_params.port          = pDesc->i2cPortLogical;
    i2c_params.address       = pDesc->i2cAddress;

    if ((status = lwswitch_ctrl_i2c_indexed(device, &i2c_params)) == LWL_SUCCESS)
    {
        lwswitch_os_memcpy(pRomBuf, i2c_params.message, romSize);
    }
    else
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: reading ROM at %d/%02x failed: %d\n",
            __FUNCTION__, pDesc->i2cPortLogical,
            pDesc->i2cAddress, status);
    }

    return status;
}

static void
_lwswitch_update_cages_mask
(
    lwswitch_device *device
)
{
    PCCI pCci = device->pCci;
    LwU32 idx_i2cdevice;
    LwU32 new_cages;

    new_cages = 0;

    //
    // osfp modules will be enumerated based on the order they are listed in
    // the CMIS modules device list(which is based on the order FRU
    // partitions are listed in bios)
    // Ex. lwswitch_i2c_device_list_delta
    //
    for(idx_i2cdevice = 0; idx_i2cdevice < pCci->osfp_num; idx_i2cdevice++)
    {
        new_cages |= LWBIT(idx_i2cdevice);   
    }

    device->pCci->cagesMask |= new_cages;
}

/*
 * @brief Detect board
 *
 */
static LwlStatus
_lwswitch_cci_detect_board
(
    lwswitch_device *device
)
{
    PCCI pCci = device->pCci;
    LwlStatus retval = LWL_SUCCESS;
    LwU16 boardId;

    retval = lwswitch_get_board_id(device, &boardId);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "%s Failed to get board ID. rc:%d\n",
                       __FUNCTION__, retval);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    switch (boardId)
    {
        // Rest of E4700 discovery will be done in _lwswitch_identify_cci_devices        
        case LWSWITCH_BOARD_LR10_4700_0000_PC0:
        case LWSWITCH_BOARD_LR10_4700_0000_STA:
        {
            pCci->boardId = LWSWITCH_BOARD_ID_UNKNOWN;
            pCci->osfp_num = lwswitch_i2c_device_list_E4760_A00_size;
            pCci->osfp_map = lwswitch_cci_osfp_map_E476X;
            pCci->osfp_map_size = lwswitch_cci_osfp_map_E476X_size;
            break;
        }
        case LWSWITCH_BOARD_LR10_3517_0300_890:
        {
            pCci->boardId = LWSWITCH_BOARD_ID_DELTA;
            pCci->osfp_i2c_info = lwswitch_i2c_device_list_delta;
            pCci->osfp_num = lwswitch_i2c_device_list_delta_size;
            pCci->osfp_map = lwswitch_cci_osfp_map_delta;
            pCci->osfp_map_size = lwswitch_cci_osfp_map_delta_size;
            pCci->numLinks = 16;

            if (pCci->rom_num == 0)
            {
                return -LWL_ERR_NOT_SUPPORTED;
            }

            // Determine partition ID using FRU contents
            retval = _lwswitch_cci_detect_partitions(device);
            if (retval != LWL_SUCCESS)
            {
                pCci->boardPartitionType = LWSWITCH_BOARD_PARTITION_E4760_A00;
                LWSWITCH_PRINT(device, ERROR, "%s Unable to determine partition. rc:%d\n",
                            __FUNCTION__, retval);
                LWSWITCH_PRINT(device, ERROR, "%s Inferring E476X_A00 carrier board\n",
                            __FUNCTION__);
            }

            _lwswitch_update_cages_mask(device);
            break;
        }
        case LWSWITCH_BOARD_LR10_3597_0000_891:
        {
            pCci->boardId = LWSWITCH_BOARD_ID_WOLF;
            pCci->boardPartitionType = LWSWITCH_BOARD_PARTITION_E3597_B00;
            pCci->osfp_i2c_info = lwswitch_i2c_device_list_wolf;
            pCci->osfp_num = lwswitch_i2c_device_list_wolf_size;
            pCci->osfp_map = lwswitch_cci_osfp_map_wolf;
            pCci->osfp_map_size = lwswitch_cci_osfp_map_wolf_size;
            pCci->numLinks = 32;
            pCci->pcsMask = 0x3;
            _lwswitch_update_cages_mask(device);
            break;
        }
        default:
        {
            LWSWITCH_PRINT(device, ERROR, "%s Setup code for board Id: 0x%x needed.\n",
                       __FUNCTION__, boardId);
            LWSWITCH_ASSERT(0);
            return -LWL_ERR_NOT_SUPPORTED;
            break;
        }
    }

    return LWL_SUCCESS;
}

static const LWSWITCH_BOARD_PARTITION_ENTRY lwswitch_part_num_map[] =
{
    { LWSWITCH_BOARD_PARTITION_E4760_A00_PART_NUM, LWSWITCH_BOARD_PARTITION_E4760_A00},
    { LWSWITCH_BOARD_PARTITION_E4761_A00_PART_NUM, LWSWITCH_BOARD_PARTITION_E4761_A00},
    { LWSWITCH_BOARD_PARTITION_P4790_B00_PART_NUM, LWSWITCH_BOARD_PARTITION_P4790_B00},
    { LWSWITCH_BOARD_PARTITION_P4791_B00_PART_NUM, LWSWITCH_BOARD_PARTITION_P4791_B00},
    { LWSWITCH_BOARD_PARTITION_E3597_B00_PART_NUM, LWSWITCH_BOARD_PARTITION_E3597_B00}
};

static const LwU32 lwswitch_board_partition_num_supported =
    LW_ARRAY_ELEMENTS(lwswitch_part_num_map);

/*
 * @brief Determine board part number using FRU data
 *
 */
static LwlStatus
_lwswitch_cci_detect_partitions
(
    lwswitch_device *device
)
{
    LwlStatus retval = LWL_SUCCESS;
    PCCI pCci = device->pCci;
    LWSWITCH_IPMI_FRU_BOARD_INFO fru_board_info;
    LWSWITCH_BOARD_PARTITION_TYPE partition = LWSWITCH_BOARD_PARTITION_UNKNOWN;
    LwU8 rom_image[1 << AT24C02D_INDEX_SIZE];
    LwU32 rom_idx, i;

    // look at each rom to obtain board type
    for (rom_idx = 0; rom_idx < pCci->rom_num; rom_idx++)
    {
        retval = _lwswitch_fetch_rom_image(device, rom_idx, (1 << AT24C02D_INDEX_SIZE), rom_image);
        if (retval != LW_OK)
        {
            LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_CCI_INIT,
                       "FRU ROM fetch failed \n");
            return retval;
        }

        retval = lwswitch_read_partition_fru_board_info(device, &fru_board_info, rom_image);

        // Parse board part number
        if (retval == LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, INFO, "%s Found in FRU - Board Part Number: %s, Product Name: %s\n",
                __FUNCTION__, fru_board_info.partNum, fru_board_info.productName);

            // search table, only look at first 8 digits eg. 699-14791-XXXX-YYY-Z.Z
            for (i = 0; i < lwswitch_board_partition_num_supported; ++i)
            {
                if (lwswitch_os_strncmp(lwswitch_part_num_map[i].part_num,
                                        fru_board_info.partNum,
                                        lwswitch_os_strlen(lwswitch_part_num_map[i].part_num)) == 0)
                {
                    partition = lwswitch_part_num_map[i].type;
                    break;
                }
            }

            pCci->pcsMask |= LWBIT64(rom_idx);
            pCci->boardPartitionType = partition;
        }
        else
        {
            LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_CCI_INIT,
                        "FRU ROM parsing error \n");
        }
    }

    return retval;
}

static LwlStatus
_lwswitch_identify_cci_devices_E4700
(
    lwswitch_device *device
)
{
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS i2c_params = { 0 };
    LwU32 idx_i2cdevice;
    PCCI pCci = device->pCci;

    for (idx_i2cdevice = 0; idx_i2cdevice < lwswitch_i2c_device_list_E4760_A00_size; idx_i2cdevice++)
    {
        i2c_params.port = lwswitch_i2c_device_list_E4760_A00[idx_i2cdevice].i2cPortLogical;
        i2c_params.address = (LwU16) lwswitch_i2c_device_list_E4760_A00[idx_i2cdevice].i2cAddress;
        i2c_params.bIsRead = LW_FALSE;
        i2c_params.flags =
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

        if (lwswitch_ctrl_i2c_indexed(device, &i2c_params) == LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, INFO,
                "%s: Identified osfp = %d, port = %d, addr = 0x%x\n",
                __FUNCTION__, idx_i2cdevice, i2c_params.port, i2c_params.address);

            pCci->osfpMaskPresent |= LWBIT(idx_i2cdevice);
            pCci->boardId = LWSWITCH_BOARD_ID_E4760_A00;
        }

        pCci->osfpMaskAll |= LWBIT(idx_i2cdevice);
    }

    // If E4760 board is recognized, no need to check for E4761.
    if (pCci->boardId == LWSWITCH_BOARD_ID_E4760_A00)
    {
        LWSWITCH_PRINT(device, INFO,
            "%s: E4760_A00 borad Identified. OSFP device mask = 0x%x\n",
            __FUNCTION__, pCci->osfpMaskPresent);
        pCci->osfp_i2c_info = lwswitch_i2c_device_list_E4760_A00;
        pCci->numLinks = 12;
        pCci->bInitialized = LW_TRUE;

        _lwswitch_find_led_drivers_and_rom(device, LWSWITCH_I2C_ACQUIRER_CCI_INITIALIZE, pCci);

        return LWL_SUCCESS;
    }

    for (idx_i2cdevice = 0; idx_i2cdevice < lwswitch_i2c_device_list_E4761_A00_size; idx_i2cdevice++)
    {
        i2c_params.port = lwswitch_i2c_device_list_E4761_A00[idx_i2cdevice].i2cPortLogical;
        i2c_params.address = (LwU16) lwswitch_i2c_device_list_E4761_A00[idx_i2cdevice].i2cAddress;

        if (lwswitch_ctrl_i2c_indexed(device, &i2c_params) == LWL_SUCCESS)
        {
            pCci->osfpMaskPresent |= LWBIT(idx_i2cdevice);
            pCci->boardId = LWSWITCH_BOARD_ID_E4761_A00;
        }
        pCci->osfpMaskAll |= LWBIT(idx_i2cdevice);
    }

    if (pCci->boardId == LWSWITCH_BOARD_ID_E4761_A00)
    {
        LWSWITCH_PRINT(device, INFO,
            "%s: E4761_A00 borad Identified. OSFP device mask = 0x%x\n",
            __FUNCTION__, pCci->osfpMaskPresent);
        pCci->osfp_i2c_info = lwswitch_i2c_device_list_E4761_A00;
        pCci->numLinks = 12;
        pCci->bInitialized = LW_TRUE;

        _lwswitch_find_led_drivers_and_rom(device, LWSWITCH_I2C_ACQUIRER_CCI_INITIALIZE, pCci);

        return LWL_SUCCESS;
    }

    LWSWITCH_PRINT(device, ERROR,
        "%s: Failed to identify any osfp devices.\n",
        __FUNCTION__);

    return -LWL_NOT_FOUND;
}

static void
_lwswitch_detect_presence_cci_devices_P479X_B00
(
    lwswitch_device *device,
    LwU32 *pMaskPresent
)
{
    PCCI pCci = device->pCci;
    LwlStatus retval;
    LwU32 presentMask;
    LwU8 pcsId;
    LwU8  val;

    presentMask = 0;

    FOR_EACH_INDEX_IN_MASK(64, pcsId, pCci->pcsMask)
    {    
        retval = _cci_pcs_read_P479X_B00(device, pcsId, P479X_IO_EXPANDER_PRESENT_INTERRUPT,
                                         IO_EXPANDER_INPUT_REG, &val);
        if (retval != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Present IO Expander read failed from pcs %d\n",
                __FUNCTION__, pcsId);
            continue;
        }

        // Bits 0-3 are present, bits 4-7 are interrupt
        val = val & 0xF;
        presentMask |= PCS_TO_CCI_MODULE_MASK_MAP_P479X_B00(val, pcsId);
    }
    FOR_EACH_INDEX_IN_MASK_END;

    if (pMaskPresent != NULL)
    {
        *pMaskPresent = presentMask;
    }
}

static void
_lwswitch_detect_presence_cci_devices_E3597_B00
(
    lwswitch_device *device,
    LwU32 *pMaskPresent
)
{
    PCCI pCci = device->pCci;
    LwlStatus retval;
    LwU64 pcsId;
    LwU32 presentMask;
    LwU8  val;

    presentMask = 0;

    FOR_EACH_INDEX_IN_MASK(64, pcsId, pCci->pcsMask)
    {    
        retval = _cci_pcs_read_E3597_B00(device, pcsId, WOLF_IO_EXPANDER_PRESENT,
                                          IO_EXPANDER_INPUT_REG, &val);
        if (retval != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Present IO Expander read failed from pcs %llu\n",
                __FUNCTION__, pcsId);
            continue;
        }

        presentMask |= PCS_TO_CCI_MODULE_MASK_MAP_E3597_B00(val, pcsId);
    }
    FOR_EACH_INDEX_IN_MASK_END;

    if (pMaskPresent != NULL)
    {
        *pMaskPresent = presentMask;
    }
}

static void
_lwswitch_ping_cci_devices
(
    lwswitch_device *device,
    LwU32 *pMaskPresent
)
{
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS i2c_params = { 0 };
    LwU32 idx_i2cdevice;
    PCCI pCci;
    LwU32 presentMask;

    pCci = device->pCci;
    presentMask = 0;

    for (idx_i2cdevice = 0; idx_i2cdevice < pCci->osfp_num; idx_i2cdevice++)
    {
        i2c_params.port = pCci->osfp_i2c_info[idx_i2cdevice].i2cPortLogical;
        i2c_params.address = (LwU16) pCci->osfp_i2c_info[idx_i2cdevice].i2cAddress;
        i2c_params.bIsRead = LW_FALSE;
        i2c_params.flags =
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

        if (lwswitch_ctrl_i2c_indexed(device, &i2c_params) == LWL_SUCCESS)
        {
            // Only print if newly present OSFP
            if (!_lwswitch_cci_module_present(device, idx_i2cdevice))
            {
                LWSWITCH_PRINT(device, INFO,
                    "%s: Identified osfp = %d, port = %d, addr = 0x%x\n",
                    __FUNCTION__, idx_i2cdevice, i2c_params.port, i2c_params.address);
            }

            presentMask |= LWBIT(idx_i2cdevice);
        }        
    }

    if (pMaskPresent != NULL)
    {
        *pMaskPresent = presentMask;
    }
}

/*
 * Check for CMIS boards by pinging on OSFP devices
 */
static LwlStatus
_lwswitch_identify_cci_devices
(
    lwswitch_device *device
)
{
    LwlStatus retval = LWL_SUCCESS;
    PCCI pCci = device->pCci;
    LwU32 presentMask;

    switch (pCci->boardId)
    {
        case LWSWITCH_BOARD_ID_UNKNOWN:
        {
            retval = _lwswitch_identify_cci_devices_E4700(device);
            break;
        }
        case LWSWITCH_BOARD_ID_DELTA:
        case LWSWITCH_BOARD_ID_WOLF:
        {
            _lwswitch_ping_cci_devices(device, &presentMask);
            pCci->osfpMaskPresent = presentMask;
            pCci->osfpMaskAll = pCci->cagesMask;
            break;
        }
        default:
        {
            LWSWITCH_ASSERT(0);
            retval = -LWL_NOT_FOUND;
            break;
        }
    }

    if (retval == LWL_SUCCESS)
    {
        pCci->bInitialized = LW_TRUE;
    }

    return retval;    
}

LwBool
cciIsLinkManaged
(
    lwswitch_device *device,
    LwU32 linkNumber
)
{
    if ((device->pCci == NULL) || (!device->pCci->bInitialized))
    {
        return LW_FALSE;
    }

    return  !!(device->pCci->linkMask & LWBIT64(linkNumber));
}

LwlStatus
cciOpticalPretrain
(
    lwswitch_device *device
)
{
    LwlStatus status;
    LwlStatus loopStatus = LWL_SUCCESS;
    LwU32 linkId;
    LwBool freeze_maintenance;
    LwBool restart_training;
    LwBool lwlink_mode;
    LwBool bTx;

    if ((device->pCci == NULL) || (!device->pCci->bInitialized))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    // enable IOBIST (PRBS31)
    FOR_EACH_INDEX_IN_MASK(64, linkId, device->pCci->linkMask)
    {
        lwswitch_cci_initialization_sequence_lr10(device, linkId);
        lwswitch_cci_enable_iobist_lr10(device, linkId, LW_TRUE);
    }
    FOR_EACH_INDEX_IN_MASK_END;

    // enable maintainence and restart training for TX
    freeze_maintenance = LW_FALSE;
    restart_training = LW_TRUE;
    lwlink_mode = LW_TRUE;
    bTx = LW_TRUE;

    loopStatus = LWL_SUCCESS;
    FOR_EACH_INDEX_IN_MASK(64, linkId, device->pCci->linkMask)
    {
        status = cciConfigureLwlinkMode(device, LWSWITCH_I2C_ACQUIRER_CCI_TRAIN,
                     linkId, bTx, freeze_maintenance,
                     restart_training, lwlink_mode);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Failed to enable TX maintenance on link %d\n",
                __FUNCTION__, linkId);
            loopStatus = status;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    if (loopStatus != LWL_SUCCESS)
    {
        return loopStatus;
    }

    lwswitch_os_sleep(3000);

    // poll for pre-training for TX
    loopStatus = LWL_SUCCESS;
    FOR_EACH_INDEX_IN_MASK(64, linkId, device->pCci->linkMask)
    {
        status = cciPollForPreTraining(device,
                     LWSWITCH_I2C_ACQUIRER_CCI_TRAIN, linkId, bTx);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Failed to poll for TX pre-training on link %d\n",
                __FUNCTION__, linkId);
            loopStatus = status;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    if (loopStatus != LWL_SUCCESS)
    {
        return loopStatus;
    }

    lwswitch_os_sleep(1000);

    // enable maintainence and restart training for RX
    bTx = LW_FALSE;

    loopStatus = LWL_SUCCESS;
    FOR_EACH_INDEX_IN_MASK(64, linkId, device->pCci->linkMask)
    {
        status = cciConfigureLwlinkMode(device, LWSWITCH_I2C_ACQUIRER_CCI_TRAIN,
                     linkId, bTx, freeze_maintenance,
                     restart_training, lwlink_mode);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Failed to enable RX maintenance on link %d\n",
                __FUNCTION__, linkId);
            loopStatus = status;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    if (loopStatus != LWL_SUCCESS)
    {
        return loopStatus;
    }

    // Wait 10sec for pre-training to complete along with 2 maintenance cycles
    lwswitch_os_sleep(10000);

    // poll for pre-training for RX
    loopStatus = LWL_SUCCESS;
    FOR_EACH_INDEX_IN_MASK(64, linkId, device->pCci->linkMask)
    {
        status = cciPollForPreTraining(device,
                     LWSWITCH_I2C_ACQUIRER_CCI_TRAIN, linkId, bTx);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Failed to poll for RX pre-training on link %d\n",
                __FUNCTION__, linkId);
            loopStatus = status;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    if (loopStatus != LWL_SUCCESS)
    {
        return loopStatus;
    }

    // Disable maintainence on TX
    freeze_maintenance = LW_TRUE;
    restart_training = LW_FALSE;
    lwlink_mode = LW_TRUE;
    bTx = LW_TRUE;

    loopStatus = LWL_SUCCESS;
    FOR_EACH_INDEX_IN_MASK(64, linkId, device->pCci->linkMask)
    {
        status = cciConfigureLwlinkMode(device, LWSWITCH_I2C_ACQUIRER_CCI_TRAIN,
                     linkId, bTx, freeze_maintenance,
                     restart_training, lwlink_mode);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Failed to disable TX maintenance on link %d\n",
                __FUNCTION__, linkId);
            loopStatus = status;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    if (loopStatus != LWL_SUCCESS)
    {
        return loopStatus;
    }

    // Disable maintainence on RX
    bTx = LW_FALSE;

    loopStatus = LWL_SUCCESS;
    FOR_EACH_INDEX_IN_MASK(64, linkId, device->pCci->linkMask)
    {
        status = cciConfigureLwlinkMode(device, LWSWITCH_I2C_ACQUIRER_CCI_TRAIN,
                     linkId, bTx, freeze_maintenance,
                     restart_training, lwlink_mode);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Failed to disable RX maintenance on link %d\n",
                __FUNCTION__, linkId);
            loopStatus = status;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    if (loopStatus != LWL_SUCCESS)
    {
        return loopStatus;
    }

    lwswitch_os_sleep(500);

    // Disable IOBIST
    FOR_EACH_INDEX_IN_MASK(64, linkId, device->pCci->linkMask)
    {
        lwswitch_cci_enable_iobist_lr10(device, linkId, LW_FALSE);
    }
    FOR_EACH_INDEX_IN_MASK_END;

    LWSWITCH_PRINT(device, INFO,
        "%s: Pretraining complete\n",
        __FUNCTION__);

    return LWL_SUCCESS;
}

/*
 * @Brief : Bootstrap CCI on the specified device
 *
 * @param[in] device Bootstrap CCI on this device
 */
LwlStatus
cciLoad
(
    lwswitch_device *device
)
{
    LwlStatus status;

    if (IS_FMODEL(device) || IS_RTLSIM(device))
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s: Skipping CCI init on preSilicon.\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (device->pCci == NULL)
    {
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_CCI_INIT,
            "Failed to init CCI(0)\n");
        return -LWL_BAD_ARGS;
    }

    // Prepare CCI for reset.
    status = _lwswitch_cci_prepare_for_reset(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_CCI_RESET,
            "Failed to reset CCI(0)\n");
        return status;
    }

    // Reset CCI
    status = _lwswitch_reset_cci(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_CCI_RESET,
            "Failed to reset CCI(1)\n");
        return status;
    }

    // Identify CCI devices
    status = _lwswitch_identify_cci_devices(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_CCI_INIT,
            "Failed to init CCI(1)\n");
        return status;
    }

    // Update Link Mask
    status = _lwswitch_cci_setup_link_mask(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_CCI_INIT,
            "Failed to init CCI(2)\n");
        return status;
    }

    status = lwswitch_cci_setup_optical_links_lr10(device, device->pCci->linkMask);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Setup CCI failed\n",
            __FUNCTION__);
    }

    status = cciApplyControlSetValues(device, 0, device->pCci->osfpMaskPresent);
    if (status != LWL_SUCCESS)
    {
        return status;
    }

    if (device->pCci->led_drv_num > 0)
    {
        status = _lwswitch_cci_init_xcvr_leds(device);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_CCI_INIT,
                "Failed to init CCI xcvr LEDs\n");
            return status;
        }

        status = cciRegisterCallback(device, LWSWITCH_CCI_CALLBACK_LED_UPDATE,
                                        _lwswitch_cci_update_link_state_led,
                                        CCI_LED_UPDATE_RATE_HZ);
        if (status != LWL_SUCCESS)
        {
            return status;
        }
    }

    LWSWITCH_PRINT(device, SETUP,
                   "%s: CCI load successful.\n",
                   __FUNCTION__);

    return status;
}

/*!
 * @brief Top level service routine for CCI
 *
 * @param[in] device         lwswitch_device  pointer
 * @param[in] pCci           CCI  pointer
 *
 * @return 32-bit interrupt status AFTER all known interrupt-sources were
 *         serviced.
 */
LwU32
cciService
(
    lwswitch_device *device,
    PCCI             pCci
)
{
    // TODO: bug 3367585
    return 0;
}

/*
 * @brief Get Temperature of the osfp device by link Id
 *
 * @param[in]  device         lwswitch_device  pointer
 * @param[in]  linkId         link Id
 * @param[out] pTemperature   Module temperature
 *
 *  Module temperature is obtained by reading the 14 & 15 bytes of Page 00h.
 *  (ref CMIS rev4.0, Table 8-6)
 */
LwlStatus
cciGetTemperature
(
    lwswitch_device *device,
    LwU32           client,
    LwU32           linkId,
    LwTemp          *pTemperature
)
{
    LwU32 addr = 14;
    LwU32 length = 2;
    LwU32 osfp;
    LwU8 temp[2];
    LwS16 temperature;

    if ((device->pCci == NULL) || (!device->pCci->bInitialized))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (_lwswitch_cci_get_module_id(device, linkId, &osfp) != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get moduleId associated with link %d\n",
            __FUNCTION__, linkId);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (cciRead(device, client, osfp, addr, length, temp) != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get temperature of osfp device %d\n",
            __FUNCTION__, osfp);
        return -LWL_ERR_GENERIC;
    }

    temperature = (LwS16) ((temp[0] << 8) | temp[1]);


    LWSWITCH_PRINT(device, INFO,
        "%s: Temperature of OSFP device %d : %dC\n",
        __FUNCTION__, osfp, temperature/256);

    *pTemperature = temperature;

    return LWL_SUCCESS;
}

/*
 * @brief Get Temperature of the osfp device
 *
 * @param[in]  device         lwswitch_device  pointer
 * @param[in]  osfp           osfp device
 * @param[out] pTemperature   Module temperature
 *
 *  Module temperature is obtained by reading the 14 & 15 bytes of Page 00h.
 *  (ref CMIS rev4.0, Table 8-6)
 */
LwlStatus
cciGetXcvrTemperature
(
    lwswitch_device *device,
    LwU32           client,
    LwU32           osfp,
    LwTemp          *pTemperature
)
{
    LwU32 addr = 14;
    LwU32 length = 2;
    LwU8 temp[2];
    LwS16 temperature;

    if ((device->pCci == NULL) || (!device->pCci->bInitialized))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (osfp >= (sizeof(device->pCci->osfpMaskAll) * 8))
    {
        return -LWL_BAD_ARGS;
    }

    //
    // This function is different from other similar ones in that
    // we allow it to proceed even if the OSFP module is not present
    // by testing against osfpMaskAll and not osfpMaskPresent.
    // This way we would be able to detect the module reinsertion,
    // essentially using this function in lieu of a ping, since it
    // is being ilwoked periodically from the polling loop.
    //
    if (!(device->pCci->osfpMaskAll & LWBIT(osfp)))
    {
        return -LWL_UNBOUND_DEVICE;
    }

    if (cciRead(device, client, osfp, addr, length, temp) != LWL_SUCCESS)
    {
        return -LWL_ERR_GENERIC;
    }

    temperature = (LwS16) ((temp[0] << 8) | temp[1]);

    *pTemperature = temperature;

    return LWL_SUCCESS;
}

/*
 * @brief Get CCI capabilities of the osfp device
 *
 * @param[in] device         lwswitch_device  pointer
 * @param[in] osfp           osfp device
 * @param[in] pCapabilities  CCI capabilities
 *
 *  CCI ID and status are obtained by reading bytes 0-2 & 86-88 of Page 00h
 *  (ref CMIS rev4.0, table 8-2, Table 6-1)
 *
 *  Control/status registers : Page 0 byte 0-3
 *  Byte  Bits  Field Name
 *    0   7-0   Identifier
 *    1   7-0   Revision Compliance
 *    2   7     Flat_mem
 *        3-2   TWI Maximum speed
 *  Application advertising: Page 0 byte 86-88
 *   Byte Bits  Field Name
 *    86  7-0   Host Electrical Interface ID
 *    87  7-0   Module Media Interface ID
 *    88  7-4   Host Lane Count
 *        3-0   Media Lane Count
 *
 *  @returns LWL_SUCCESS
 */
LwlStatus
cciGetCapabilities
(
    lwswitch_device *device,
    LwU32           client,
    LwU32           linkId,
    LWSWITCH_CCI_CAPABILITIES *pCapabilities
)
{
    LwU32 addr;
    LwU32 length;
    LwU32 osfp;
    LwU8 temp[3];

    if ((device->pCci == NULL) || (!device->pCci->bInitialized))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (_lwswitch_cci_get_module_id(device, linkId, &osfp) != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get moduleId associated with link %d\n",
            __FUNCTION__, linkId);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    //  ID/Status: Page 0 byte 0-2
    addr = 0x0;
    length = 3;

    if (cciRead(device, client, osfp, addr, length, temp) != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get CCI capabilities of osfp device %d\n",
            __FUNCTION__, osfp);
        return -LWL_ERR_GENERIC;
    }

    pCapabilities->identifier = temp[0];
    pCapabilities->rev_compliance = temp[1];
    pCapabilities->flat_mem = temp[2] & LWBIT(7);
    pCapabilities->twi_max_speed_khz = (temp[2] >> 2) & 0x3;

    if (pCapabilities->twi_max_speed_khz == 0x0)
    {
        pCapabilities->twi_max_speed_khz = 400;
    }
    else if (pCapabilities->twi_max_speed_khz == 0x1)
    {
        pCapabilities->twi_max_speed_khz = 1000;
    }
    else
    {
        pCapabilities->twi_max_speed_khz = 0xFFFF;
    }

    // Application advertising: Page 0 byte 86-88
    addr = 86;
    length = 3;

    if (cciRead(device, client, osfp, addr, length, temp) != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get CCI capabilities of osfp device %d\n",
            __FUNCTION__, osfp);
        return -LWL_ERR_GENERIC;
    }

    pCapabilities->host_interface_id = temp[0];
    pCapabilities->module_interface_id = temp[1];
    pCapabilities->host_lane_count = temp[2] >> 4;
    pCapabilities->module_lane_count = temp[2] & 0xf;

    return LWL_SUCCESS;
}

/*
 * @brief Get FW revisions of the osfp device by link Id
 *
 *  Module FW revision is obtained from CDB command 0x100.
 *  (ref CMIS rev4.0, Table 9-16 CDB Command 0100h: Get firmware Info)
 */
LwlStatus
cciGetFWRevisions
(
    lwswitch_device *device,
    LwU32            client,
    LwU32            linkId,
    LWSWITCH_CCI_GET_FW_REVISIONS *pRevisions
)
{
    LwU32 osfp;

    if ((device->pCci == NULL) || (!device->pCci->bInitialized))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (_lwswitch_cci_get_module_id(device, linkId, &osfp) != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get moduleid associated with link %d\n",
            __FUNCTION__, linkId);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    return cciGetXcvrFWRevisions(device, client, osfp, pRevisions);
}

/*
 * @brief Get FW revisions of the osfp device by OSFP xceiver index
 *
 *  Module FW revision is obtained from CDB command 0x100.
 *  (ref CMIS rev4.0, Table 9-16 CDB Command 0100h: Get firmware Info)
 */
LwlStatus
cciGetXcvrFWRevisions
(
    lwswitch_device *device,
    LwU32           client,
    LwU32            osfp,
    LWSWITCH_CCI_GET_FW_REVISIONS *pRevisions
)
{
    LwU8 response[120];
    LwU32 resLength;
    LwlStatus retVal;
    LwU8 status;

    if ((device->pCci == NULL) || (!device->pCci->bInitialized))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (osfp >= (sizeof(device->pCci->osfpMaskAll) * 8))
    {
        return -LWL_BAD_ARGS;
    }

    if (!(device->pCci->osfpMaskPresent & LWBIT(osfp)))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: osfp %d is missing\n",
            __FUNCTION__, osfp);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    retVal = cciSendCDBCommandAndGetResponse(device, client, osfp,
        0x100, 0, NULL, &resLength, response, LW_FALSE);
    if (retVal != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get FW revisions\n",
            __FUNCTION__);
        return -LWL_ERR_GENERIC;
    }

    lwswitch_os_memset(pRevisions, 0, sizeof(LWSWITCH_CCI_GET_FW_REVISIONS));

    // Byte 0(or 136) contains FW status
    status = response[0];

    pRevisions[LWSWITCH_CCI_FW_IMAGE_FACTORY].flags = 0;
    if (status == 0)
    {
        LWSWITCH_PRINT(device, INFO,
            "%s: Factory Boot Image is Running\n",
            __FUNCTION__);

       pRevisions[LWSWITCH_CCI_FW_IMAGE_FACTORY].flags =
           FLD_SET_DRF(SWITCH, _CCI_FW_FLAGS, _ACTIVE, _YES,
                       pRevisions[LWSWITCH_CCI_FW_IMAGE_FACTORY].flags);
    }

    if (response[1] & LWBIT(2))
    {
        //
        // For Factory Image,
        //   Byte 74(or 210) contains major revision
        //   Byte 75(or 211) contains minor revision
        //   Byte 76, 77(or 212, 213) contains build number
        //
        pRevisions[LWSWITCH_CCI_FW_IMAGE_FACTORY].flags =
            FLD_SET_DRF(SWITCH, _CCI_FW_FLAGS, _PRESENT, _YES,
                        pRevisions[LWSWITCH_CCI_FW_IMAGE_FACTORY].flags);
        pRevisions[LWSWITCH_CCI_FW_IMAGE_FACTORY].major = response[74];
        pRevisions[LWSWITCH_CCI_FW_IMAGE_FACTORY].minor = response[75];
        pRevisions[LWSWITCH_CCI_FW_IMAGE_FACTORY].build = (response[76] << 4 | response[77]);
    }

    pRevisions[LWSWITCH_CCI_FW_IMAGE_A].flags = 0;
    if (status & LWBIT(0))
    {
        LWSWITCH_PRINT(device, INFO,
            "%s: Image A is Running\n",
            __FUNCTION__);

        pRevisions[LWSWITCH_CCI_FW_IMAGE_A].flags =
            FLD_SET_DRF(SWITCH, _CCI_FW_FLAGS, _ACTIVE, _YES,
                        pRevisions[LWSWITCH_CCI_FW_IMAGE_A].flags);
    }
    if (status & LWBIT(1))
    {
        LWSWITCH_PRINT(device, INFO,
            "%s: Image A is committed, module boots from Image A\n",
            __FUNCTION__);
        pRevisions[LWSWITCH_CCI_FW_IMAGE_A].flags =
            FLD_SET_DRF(SWITCH, _CCI_FW_FLAGS, _COMMITED, _YES,
                        pRevisions[LWSWITCH_CCI_FW_IMAGE_A].flags);
    }
    if (status & LWBIT(2))
    {
        LWSWITCH_PRINT(device, INFO,
            "%s: Image A is erased/empty\n",
            __FUNCTION__);
        pRevisions[LWSWITCH_CCI_FW_IMAGE_A].flags =
            FLD_SET_DRF(SWITCH, _CCI_FW_FLAGS, _EMPTY, _YES,
                        pRevisions[LWSWITCH_CCI_FW_IMAGE_A].flags);
    }

    if (response[1] & LWBIT(0))
    {
        pRevisions[LWSWITCH_CCI_FW_IMAGE_A].flags =
            FLD_SET_DRF(SWITCH, _CCI_FW_FLAGS, _PRESENT, _YES,
                        pRevisions[LWSWITCH_CCI_FW_IMAGE_A].flags);
        //
        // For Image A,
        //   Byte 2(or 138) contains major revision
        //   Byte 3(or 139) contains minor revision
        //   Byte 4, 5(or 140, 141) contains build number
        //
        pRevisions[LWSWITCH_CCI_FW_IMAGE_A].major = response[2];
        pRevisions[LWSWITCH_CCI_FW_IMAGE_A].minor = response[3];
        pRevisions[LWSWITCH_CCI_FW_IMAGE_A].build = (response[4] << 4 | response[5]);
    }

    pRevisions[LWSWITCH_CCI_FW_IMAGE_B].flags = 0;
    if (status & LWBIT(4))
    {
        LWSWITCH_PRINT(device, INFO,
            "%s: Image B is Running\n",
            __FUNCTION__);

        pRevisions[LWSWITCH_CCI_FW_IMAGE_B].flags =
            FLD_SET_DRF(SWITCH, _CCI_FW_FLAGS, _ACTIVE, _YES,
                        pRevisions[LWSWITCH_CCI_FW_IMAGE_B].flags);
    }
    if (status & LWBIT(5))
    {
        LWSWITCH_PRINT(device, INFO,
            "%s: Image B is committed, module boots from Image B\n",
            __FUNCTION__);
        pRevisions[LWSWITCH_CCI_FW_IMAGE_B].flags =
            FLD_SET_DRF(SWITCH, _CCI_FW_FLAGS, _COMMITED, _YES,
                        pRevisions[LWSWITCH_CCI_FW_IMAGE_B].flags);
    }
    if (status & LWBIT(6))
    {
        LWSWITCH_PRINT(device, INFO,
            "%s: Image B is erased/empty\n",
            __FUNCTION__);
        pRevisions[LWSWITCH_CCI_FW_IMAGE_B].flags =
            FLD_SET_DRF(SWITCH, _CCI_FW_FLAGS, _EMPTY, _YES,
                        pRevisions[LWSWITCH_CCI_FW_IMAGE_B].flags);
    }

    if (response[1] & LWBIT(1))
    {
        pRevisions[LWSWITCH_CCI_FW_IMAGE_B].flags =
            FLD_SET_DRF(SWITCH, _CCI_FW_FLAGS, _PRESENT, _YES,
                        pRevisions[LWSWITCH_CCI_FW_IMAGE_B].flags);
        //
        // For Image B,
        //   Byte 38(or 174) contains major revision
        //   Byte 39(or 175) contains minor revision
        //   Byte 40, 41(or 176, 177) contains build number
        //
        pRevisions[LWSWITCH_CCI_FW_IMAGE_B].major = response[38];
        pRevisions[LWSWITCH_CCI_FW_IMAGE_B].minor = response[39];
        pRevisions[LWSWITCH_CCI_FW_IMAGE_B].build = (response[40] << 8 | response[41]);
    }

    return LWL_SUCCESS;
}

/*
 * @brief Get Serial number, Part number, HW revision and FRU EEPROM
 * of the osfp device by OSFP xceiver index.
 *
 *  Source:
 *      ref CMIS rev4.0, Table 8-15 Upper Memory Page 00h,
 *      Administrative Information
 *          Offsets: 166-181, 148-163, 164-165 respectively
 */
LwlStatus
cciGetXcvrStaticIdInfo
(
    lwswitch_device *device,
    LwU32           client,
    LwU32            osfp,
    LwU8            *pSerial,
    LwU8            *pPart,
    LwU8            *pHwRev,
    LwU8            **ppFru
)
{
    struct
    {
        LwU8    offset;
        LwU8    size;
        LwU8   *pDst;
    }           dataDesc[3] =
    {
        {166,   16, pSerial},
        {148,   16, pPart},
        {164,   16, pHwRev},
    };
    unsigned    idx;

    if ((device->pCci == NULL) || (!device->pCci->bInitialized))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (osfp >= (sizeof(device->pCci->osfpMaskAll) * 8))
    {
        return -LWL_BAD_ARGS;
    }

    if (!(device->pCci->osfpMaskPresent & LWBIT(osfp)))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: osfp %d is missing\n",
            __FUNCTION__, osfp);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    cciSetBankAndPage(device, client, osfp, 0, 0);

    for (idx = 0; idx < LW_ARRAY_ELEMENTS(dataDesc); ++idx)
    {
        if (dataDesc[idx].pDst == NULL)
        {
            continue;
        }

        if (cciRead(device, client, osfp, dataDesc[idx].offset, dataDesc[idx].size,
                    dataDesc[idx].pDst) != LWL_SUCCESS)
        {
            return -LWL_ERR_GENERIC;
        }
    }

    if (ppFru != NULL)
    {
        unsigned osfpPerRom;

        if (device->pCci->rom_num != 0)
        {
            osfpPerRom = device->pCci->osfp_num / device->pCci->rom_num;
            *ppFru = device->pCci->romCache[osfp / osfpPerRom];
        }
        else
        {
            *ppFru = NULL;
        }
    }

    return LWL_SUCCESS;
}

/*
 * @brief Read a byte from an I2C slave using the SMBus Read Byte protocol
 *
 */
static LwlStatus
_cciReadSlaveRegisterByte
(
    lwswitch_device                     *device,
    LwU32                                client,
    LWSWITCH_I2C_PORT_TYPE               port,
    LwU8                                 slaveAddr,
    LwU8                                 command,
    LwU8                                *pData
)
{
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS     i2cIndexed = {0};
    LwlStatus                            status;

    i2cIndexed.acquirer = client; 
    i2cIndexed.port = port;
    i2cIndexed.bIsRead = LW_TRUE;
    i2cIndexed.address = slaveAddr;
    i2cIndexed.flags =
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _START,        _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _RESTART,      _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _STOP,         _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _ADDRESS_MODE, _7BIT) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _FLAVOR, _HW)         |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, _400KHZ) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _BLOCK_PROTOCOL, _DISABLED) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _TRANSACTION_MODE, _NORMAL) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, _ONE);
    i2cIndexed.index[0] = command;
    i2cIndexed.messageLength = 1;

    status = lwswitch_ctrl_i2c_indexed(device, &i2cIndexed);

    if (status == LWL_SUCCESS)
    {
        *pData = i2cIndexed.message[0];
    }

    return status;
}

/*
 * @brief Write a byte to an I2C slave using the SMBus Write Byte protocol
 *
 */
static LwlStatus
_cciWriteSlaveRegisterByte
(
    lwswitch_device                     *device,
    LwU32                                client,
    LWSWITCH_I2C_PORT_TYPE               port,
    LwU8                                 slaveAddr,
    LwU8                                 command,
    LwU8                                 data
)
{
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS     i2cIndexed = {0};

    i2cIndexed.acquirer = client;
    i2cIndexed.port = port;
    i2cIndexed.bIsRead = LW_FALSE;
    i2cIndexed.address = slaveAddr;
    i2cIndexed.flags =
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _START,        _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _RESTART,      _NONE) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _STOP,         _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _ADDRESS_MODE, _7BIT) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _FLAVOR, _HW)         |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, _400KHZ) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _BLOCK_PROTOCOL, _DISABLED) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _TRANSACTION_MODE, _NORMAL) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, _ZERO) |
        0;
    i2cIndexed.messageLength = 2;
    i2cIndexed.message[0] = command;
    i2cIndexed.message[1] = data;

    return lwswitch_ctrl_i2c_indexed(device, &i2cIndexed);
}

/*
 * @brief Get xceiver LED state from a TCA6507 driver
 *
 *  Source:
 *      https://www.ti.com/lit/gpn/tca6507
 */
static LwlStatus
_cciGetXcvrLedState_Tca6507
(
    lwswitch_device                     *device,
    LwU32                                client,
    LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE *pI2cInfo,
    LwU32                                ledIdx,
    LwU8                                *pLedState
)
{
    LwU8        select[LED_TCA6507_SELECT_REG_NUM];
    LwlStatus   status;
    LwBool      greenOn    = LW_FALSE;
    LwBool      amberOn    = LW_FALSE;
    LwBool      greenBlink = LW_FALSE;
    LwBool      amberBlink = LW_FALSE;
    unsigned    idx;

    for (idx = 0; idx < LED_TCA6507_SELECT_REG_NUM; ++idx)
    {
        status = _cciReadSlaveRegisterByte(device, client,
                                           pI2cInfo->i2cPortLogical,
                                           pI2cInfo->i2cAddress,
                                           LED_TCA6507_REG_SELECT0 + idx,
                                           select + idx);
        if (status != LWL_SUCCESS)
        {
            goto _cciGetXcvrLedState_Tca6507_exit;
        }
    }

    switch (LED_TCA6507_LED_STATE(select, OSFP_LED_TCA6507_GREEN_PORT(ledIdx)))
    {
        case LED_TCA6507_LED_STATE_OFF:
        case LED_TCA6507_LED_STATE_OFF_ALT:
        {
            break;
        }
        case LED_TCA6507_LED_STATE_ON_PWM0:
        case LED_TCA6507_LED_STATE_ON_PWM1:
        case LED_TCA6507_LED_STATE_ON_MAX:
        case LED_TCA6507_LED_STATE_ON_1_SHOT:
        {
            greenOn = LW_TRUE;
            break;
        }
        case LED_TCA6507_LED_STATE_BLINK_BANK0:
        case LED_TCA6507_LED_STATE_BLINK_BANK1:
        {
            greenBlink = LW_TRUE;
            break;
        }
    }

    switch (LED_TCA6507_LED_STATE(select, OSFP_LED_TCA6507_AMBER_PORT(ledIdx)))
    {
        case LED_TCA6507_LED_STATE_OFF:
        case LED_TCA6507_LED_STATE_OFF_ALT:
        {
            break;
        }
        case LED_TCA6507_LED_STATE_ON_PWM0:
        case LED_TCA6507_LED_STATE_ON_PWM1:
        case LED_TCA6507_LED_STATE_ON_MAX:
        case LED_TCA6507_LED_STATE_ON_1_SHOT:
        {
            amberOn = LW_TRUE;
            break;
        }
        case LED_TCA6507_LED_STATE_BLINK_BANK0:
        case LED_TCA6507_LED_STATE_BLINK_BANK1:
        {
            amberBlink = LW_TRUE;
            break;
        }
    }

    if (amberBlink && !greenOn && !greenBlink)
    {
        *pLedState = LW_MSGBOX_DATA_OPTICAL_XCEIVER_CTL_LED_STATE_BLINKING_AMBER;
    }
    else if (amberOn && !greenOn && !greenBlink)
    {
        *pLedState = LW_MSGBOX_DATA_OPTICAL_XCEIVER_CTL_LED_STATE_SOLID_AMBER;
    }
    else if (greenBlink && !amberOn && !amberBlink)
    {
        *pLedState = LW_MSGBOX_DATA_OPTICAL_XCEIVER_CTL_LED_STATE_BLINKING_GREEN;
    }
    else if (greenOn && !amberBlink && !amberOn)
    {
        *pLedState = LW_MSGBOX_DATA_OPTICAL_XCEIVER_CTL_LED_STATE_GREEN;
    }
    else if (!(amberOn || amberBlink || greenOn || greenBlink))
    {
        *pLedState = LW_MSGBOX_DATA_OPTICAL_XCEIVER_CTL_LED_STATE_OFF;
    }
    else
    {
        *pLedState = 0xf;   // Not an expected state;
    }

_cciGetXcvrLedState_Tca6507_exit:
    return status;
}

static LwlStatus
_cciInitXcvrLedBlinkRate_Tca6507
(
    lwswitch_device                     *device,
    LwU32                                client,
    LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE *pI2cInfo
)
{
    LwU8        timeRegs[LED_TCA6507_TIME_REG_NUM];
    LwlStatus   status;
    unsigned    idx;

    for (idx = 0; idx < LED_TCA6507_TIME_REG_NUM; ++idx)
    {
        status = _cciReadSlaveRegisterByte(device, client,
                                           pI2cInfo->i2cPortLogical,
                                           pI2cInfo->i2cAddress,
                                           LED_TCA6507_REG_FADE_ON_TIME + idx,
                                           timeRegs + idx);
        if (status != LWL_SUCCESS)
        {
            goto _cciInitXcvrLedBlinkRate_Tca6507_exit;
        }
    }

    // Set BANK0 LED to blink at 1 Hz
    LED_TCA6507_REG_SET_TIME(timeRegs, LED_TCA6507_REG_FADE_ON_TIME,
                             LED_TCA6507_REG_TIME_BANK0, 0);
    LED_TCA6507_REG_SET_TIME(timeRegs, LED_TCA6507_REG_FULLY_ON_TIME,
                             LED_TCA6507_REG_TIME_BANK0, 6);
    LED_TCA6507_REG_SET_TIME(timeRegs, LED_TCA6507_REG_FADE_OFF_TIME,
                             LED_TCA6507_REG_TIME_BANK0, 0);
    LED_TCA6507_REG_SET_TIME(timeRegs, LED_TCA6507_REG_FIRST_FULLY_OFF_TIME,
                             LED_TCA6507_REG_TIME_BANK0, 6);
    LED_TCA6507_REG_SET_TIME(timeRegs, LED_TCA6507_REG_SECOND_FULLY_OFF_TIME,
                             LED_TCA6507_REG_TIME_BANK0, 0);

    // Set BANK1 LED to maximum blinking rate of 8 Hz
    LED_TCA6507_REG_SET_TIME(timeRegs, LED_TCA6507_REG_FADE_ON_TIME,
                             LED_TCA6507_REG_TIME_BANK1, 0);
    LED_TCA6507_REG_SET_TIME(timeRegs, LED_TCA6507_REG_FULLY_ON_TIME,
                             LED_TCA6507_REG_TIME_BANK1, 1);
    LED_TCA6507_REG_SET_TIME(timeRegs, LED_TCA6507_REG_FADE_OFF_TIME,
                             LED_TCA6507_REG_TIME_BANK1, 0);
    LED_TCA6507_REG_SET_TIME(timeRegs, LED_TCA6507_REG_FIRST_FULLY_OFF_TIME,
                             LED_TCA6507_REG_TIME_BANK1, 1);
    LED_TCA6507_REG_SET_TIME(timeRegs, LED_TCA6507_REG_SECOND_FULLY_OFF_TIME,
                             LED_TCA6507_REG_TIME_BANK1, 0);

    for (idx = 0; idx < LED_TCA6507_TIME_REG_NUM; ++idx)
    {
        status = _cciWriteSlaveRegisterByte(device, client,
                                           pI2cInfo->i2cPortLogical,
                                           pI2cInfo->i2cAddress,
                                           LED_TCA6507_REG_FADE_ON_TIME + idx,
                                           timeRegs[idx]);
        if (status != LWL_SUCCESS)
        {
            goto _cciInitXcvrLedBlinkRate_Tca6507_exit;
        }
    }

_cciInitXcvrLedBlinkRate_Tca6507_exit:
    return status;
}

static LwlStatus
_lwswitch_cci_init_xcvr_leds
(
    lwswitch_device *device
)
{
    LWSWITCH_I2C_DEVICE_TYPE ledDriverType;
    LwlStatus   status;
    LwU32 i;

    for (i = 0; i < device->pCci->led_drv_num; i++)
    {
        ledDriverType = device->pCci->led_drv_i2c_info[i]->i2cDeviceType;
        if (ledDriverType == LWSWITCH_I2C_DEVICE_TCA6507)
        {
            status = _cciInitXcvrLedBlinkRate_Tca6507(device,
                        LWSWITCH_I2C_ACQUIRER_CCI_INITIALIZE,
                        device->pCci->led_drv_i2c_info[i]);
            if (status != LWL_SUCCESS)
            {
                return status;
            }
        }
    }

    return LWL_SUCCESS;
}


/*
 * @brief Set xceiver LED state in a TCA6507 driver
 *
 *  Source:
 *      https://www.ti.com/lit/gpn/tca6507
 */
static LwlStatus
_cciSetXcvrLedState_Tca6507
(
    lwswitch_device                     *device,
    LwU32                                client,
    LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE *pI2cInfo,
    LwU32                                ledIdx,
    LwU8                                 ledState
)
{
    LwU8        select[LED_TCA6507_SELECT_REG_NUM];
    LwlStatus   status;
    LwU8        greenPort;
    LwU8        amberPort;
    unsigned    idx;

    switch (ledState)
    {
        case LW_MSGBOX_DATA_OPTICAL_XCEIVER_CTL_LED_STATE_OFF:
        {
            greenPort = LED_TCA6507_LED_STATE_OFF;
            amberPort = LED_TCA6507_LED_STATE_OFF;
            break;
        }

        case LW_MSGBOX_DATA_OPTICAL_XCEIVER_CTL_LED_STATE_BLINKING_GREEN:
        {
            greenPort = LED_TCA6507_LED_STATE_BLINK_BANK1;
            amberPort = LED_TCA6507_LED_STATE_OFF;
            break;
        }

        case LW_MSGBOX_DATA_OPTICAL_XCEIVER_CTL_LED_STATE_GREEN:
        {
            greenPort = LED_TCA6507_LED_STATE_ON_MAX;
            amberPort = LED_TCA6507_LED_STATE_OFF;
            break;
        }

        case LW_MSGBOX_DATA_OPTICAL_XCEIVER_CTL_LED_STATE_BLINKING_AMBER:
        {
            greenPort = LED_TCA6507_LED_STATE_OFF;
            amberPort = LED_TCA6507_LED_STATE_BLINK_BANK0;
            break;
        }

        case LW_MSGBOX_DATA_OPTICAL_XCEIVER_CTL_LED_STATE_SOLID_AMBER:
        {
            greenPort = LED_TCA6507_LED_STATE_OFF;
            amberPort = LED_TCA6507_LED_STATE_ON_MAX;
            break;
        }

        default:
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Illegal LED state %02x requested\n",
                __FUNCTION__, ledState);
            status = -LWL_BAD_ARGS;
            goto _cciSetXcvrLedState_Tca6507_exit;
        }
    }

    for (idx = 0; idx < LED_TCA6507_SELECT_REG_NUM; ++idx)
    {
        status = _cciReadSlaveRegisterByte(device, client,
                                           pI2cInfo->i2cPortLogical,
                                           pI2cInfo->i2cAddress,
                                           LED_TCA6507_REG_SELECT0 + idx,
                                           select + idx);
        if (status != LWL_SUCCESS)
        {
            goto _cciSetXcvrLedState_Tca6507_exit;
        }
    }

    LED_TCA6507_LED_SET_STATE(select, OSFP_LED_TCA6507_GREEN_PORT(ledIdx), greenPort);
    LED_TCA6507_LED_SET_STATE(select, OSFP_LED_TCA6507_AMBER_PORT(ledIdx), amberPort);

    for (idx = 0; idx < LED_TCA6507_SELECT_REG_NUM; ++idx)
    {
        status = _cciWriteSlaveRegisterByte(device, client,
                                           pI2cInfo->i2cPortLogical,
                                           pI2cInfo->i2cAddress,
                                           LED_TCA6507_REG_SELECT0 + idx,
                                           select[idx]);
        if (status != LWL_SUCCESS)
        {
            goto _cciSetXcvrLedState_Tca6507_exit;
        }
    }

_cciSetXcvrLedState_Tca6507_exit:
    return status;
}

/*
 * @brief Get xceiver LED state from a PCA9685BS driver
 *
 *  Source:
 *      https://www.nxp.com/docs/en/data-sheet/PCA9685.pdf
 */
static LwlStatus
_cciGetXcvrLedState_Pca9685BS
(
    lwswitch_device                     *device,
    LwU32                                client,
    LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE *pI2cInfo,
    LwU32                                ledIdx,
    LwU8                                *pLedState
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

/*
 * @brief Set xceiver LED state in a PCA9685BS driver
 *
 *  Source:
 *      https://www.nxp.com/docs/en/data-sheet/PCA9685.pdf
 */
static LwlStatus
_cciSetXcvrLedState_Pca9685BS
(
    lwswitch_device                     *device,
    LwU32                                client,
    LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE *pI2cInfo,
    LwU32                                ledIdx,
    LwU8                                 ledState
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

/*
 * @brief Get xceiver LED state by OSFP xceiver index.
 *
 *  Source:
 *      Prospector Optical Carrier (confluence page ID: 783120628)
 */
LwlStatus
cciGetXcvrLedState
(
    lwswitch_device *device,
    LwU32            client,
    LwU32            osfp,
    LwU8            *pLedState
)
{
    unsigned    ledPerDrvr;
    unsigned    drvrIdx;
    LwBool      bDrvrIsTca6507;

    if (!device->pCci->bInitialized)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (device->pCci->led_drv_num == 0)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: no LED driver devices\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    bDrvrIsTca6507 = device->pCci->led_drv_i2c_info[0]->i2cDeviceType ==
                        LWSWITCH_I2C_DEVICE_TCA6507;
    ledPerDrvr = bDrvrIsTca6507 ?
                    OSFP_LED_PER_DRVR_TCA6507 : OSFP_LED_PER_DRVR_PCA9685BS;

    drvrIdx = osfp / ledPerDrvr;

    if ((osfp >= (sizeof(device->pCci->osfpMaskAll) * 8)) ||
        (osfp >= device->pCci->led_drv_num * ledPerDrvr) ||
        (drvrIdx >= LWSWITCH_CCI_LED_DRV_NUM_MAX))
    {
        return -LWL_BAD_ARGS;
    }

    //
    // Since the LED driver is not a part of the OSFP module,
    // we do not require that the module is present
    //
    if (!(device->pCci->osfpMaskAll & LWBIT(osfp)))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: osfp %d is not identified\n",
            __FUNCTION__, osfp);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    return (bDrvrIsTca6507 ?
            _cciGetXcvrLedState_Tca6507 : _cciGetXcvrLedState_Pca9685BS)
                (device, client,
                device->pCci->led_drv_i2c_info[drvrIdx],
                osfp % ledPerDrvr, pLedState);
}

static LwBool
_cciCheckXcvrForLinkTraffic
(
    lwswitch_device *device,
    LwU32 osfp,
    LwU64 linkMask
)
{
    LWSWITCH_GET_THROUGHPUT_COUNTERS_PARAMS *pCounterParams = NULL;
    LwU64 *pCounterValues;
    LwU64 tpCounterPreviousSum;
    LwU64 tpCounterLwrrentSum;
    LwBool bTraffic = LW_FALSE;
    LwU8 linkNum;

    pCounterParams = lwswitch_os_malloc(sizeof(*pCounterParams));
    if (pCounterParams == NULL)
        goto out;

    pCounterParams->counterMask = LWSWITCH_THROUGHPUT_COUNTERS_TYPE_DATA_TX |
                                  LWSWITCH_THROUGHPUT_COUNTERS_TYPE_DATA_RX;
    pCounterParams->linkMask = linkMask;
    if (lwswitch_ctrl_get_throughput_counters_lr10(device,
        pCounterParams) != LWL_SUCCESS)
    {
        goto out;
    }

    // Sum TX/RX traffic for each link
    FOR_EACH_INDEX_IN_MASK(64, linkNum, linkMask)
    {
        pCounterValues = pCounterParams->counters[linkNum].values;

        tpCounterPreviousSum = device->pCci->tpCounterPreviousSum[linkNum];

        // Sum taken to save space as it is unlikely to overflow before system is reset
        tpCounterLwrrentSum = pCounterValues[LWSWITCH_THROUGHPUT_COUNTERS_TYPE_DATA_TX] +
                              pCounterValues[LWSWITCH_THROUGHPUT_COUNTERS_TYPE_DATA_RX];

        device->pCci->tpCounterPreviousSum[linkNum] = tpCounterLwrrentSum;

        // Skip traffic check in first call on system start up
        if (device->pCci->callbackCounter == 0)
        {
            continue;
        }

        if (tpCounterLwrrentSum > tpCounterPreviousSum)
        {
            bTraffic = LW_TRUE;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

out:
    lwswitch_os_free(pCounterParams);
    return bTraffic;
}


/*
 * @brief
 *
 *    T = true, F = false, X = don't care
 *   +--------------+--------------+--------------------+-----------------+-----------+----------------+
 *   | LINKSTATE HS | LINK TRAFFIC | LINKSTATE RECOVERY | LINKSTATE FAULT | LOCATE ON | LED COLOR      |
 *   +--------------+--------------+--------------------+-----------------+-----------+----------------+
 *   | T            | T            | F                  | F               | F         | BLINKING GREEN |
 *   +--------------+--------------+--------------------+-----------------+-----------+----------------+
 *   | T            | F            | F                  | F               | F         | GREEN          |
 *   +--------------+--------------+--------------------+-----------------+-----------+----------------+
 *   | F            | F            | T                  | F               | F         | GREEN          |
 *   +--------------+--------------+--------------------+-----------------+-----------+----------------+
 *   | F            | F            | F                  | T               | F         | AMBER          |
 *   +--------------+--------------+--------------------+-----------------+-----------+----------------+
 *   | X            | X            | X                  | X               | T         | BLINKING AMBER |
 *   +--------------+--------------+--------------------+-----------------+-----------+----------------+
 *   | F            | F            | F                  | F               | F         | OFF            |
 *   +--------------+--------------+--------------------+-----------------+-----------+----------------+
 *
 */
static LwU8
_cciGetXcvrNextLedState
(
    lwswitch_device *device,
    LwU32            osfp
)
{
    lwlink_link *link;
    LwU64 linkState;
    LwU64 linkMask;
    LwU8 linkNum;
    LwlStatus   status;

    status = cciGetCageMapping(device, osfp, &linkMask, NULL);
    if (status != LWL_SUCCESS)
    {
        return LW_MSGBOX_DATA_OPTICAL_XCEIVER_CTL_LED_STATE_OFF;
    }

    // Check if all links on osfp are in active or in recovery
    FOR_EACH_INDEX_IN_MASK(64, linkNum, linkMask)
    {
        link = lwswitch_get_link(device, linkNum);

        if ((link == NULL) ||
            (device->hal.lwswitch_corelib_get_dl_link_mode(link, &linkState) != LWL_SUCCESS))
        {
            return LW_MSGBOX_DATA_OPTICAL_XCEIVER_CTL_LED_STATE_OFF;
        }

        switch (linkState)
        {
            case LWLINK_LINKSTATE_HS:
            case LWLINK_LINKSTATE_RECOVERY:
            {
                break;
            }
            case LWLINK_LINKSTATE_FAULT:
            {
                return LW_MSGBOX_DATA_OPTICAL_XCEIVER_CTL_LED_STATE_SOLID_AMBER;
            }
            default:
            {
                return LW_MSGBOX_DATA_OPTICAL_XCEIVER_CTL_LED_STATE_OFF;
            }
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    if (_cciCheckXcvrForLinkTraffic(device, osfp, linkMask))
    {
        return LW_MSGBOX_DATA_OPTICAL_XCEIVER_CTL_LED_STATE_BLINKING_GREEN;
    }

    return LW_MSGBOX_DATA_OPTICAL_XCEIVER_CTL_LED_STATE_GREEN;
}

static void
_lwswitch_cci_update_link_state_led
(
    lwswitch_device         *device
)
{
    LwU32 cagesMask;
    LwU8  osfp;
    LwU8  lwrrentLedState;
    LwU8  nextLedState;

    // LEDs are soldered to the carrier PCBA
    if (cciGetXcvrMask(device, &cagesMask, NULL) == LWL_SUCCESS)
    {
        // Loop over all cages and update leds
        FOR_EACH_INDEX_IN_MASK(32, osfp, cagesMask)
        {
            // xcvrLwrrentLedState[] is only updated when LED HW state is set
            lwrrentLedState = device->pCci->xcvrLwrrentLedState[osfp];
            nextLedState = _cciGetXcvrNextLedState(device, osfp);

            //
            // This is the next state that the LED will be set to.
            // When locate is turned off by SMBPBI, the LED
            // will be set to the most recent next state that was determined.
            //
            device->pCci->xcvrNextLedState[osfp] = nextLedState;

            // Only update HW if required
            if ((lwrrentLedState != LW_MSGBOX_DATA_OPTICAL_XCEIVER_CTL_LED_STATE_BLINKING_AMBER) &&
                (lwrrentLedState != nextLedState))
            {
                cciSetXcvrLedState(device, LWSWITCH_I2C_ACQUIRER_CCI_UX,
                                       osfp, LW_FALSE);
            }
        }
        FOR_EACH_INDEX_IN_MASK_END;
    }
}

/*
 * @brief Set xceiver LED state (Locate On/Off) by OSFP xceiver index.
 *        If Locate is off then set LED state based on link state.
 *
 *  Source:
 *      Prospector Optical Carrier (confluence page ID: 783120628)
 */
LwlStatus
cciSetXcvrLedState
(
    lwswitch_device *device,
    LwU32            client,
    LwU32            osfp,
    LwBool           bSetLocate
)
{
    unsigned    ledPerDrvr;
    unsigned    drvrIdx;
    LwBool      bDrvrIsTca6507;
    LwU8        ledState;
    LwU8        nextLedState;

    if (!device->pCci->bInitialized)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (device->pCci->led_drv_num == 0)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: no LED driver devices\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    bDrvrIsTca6507 = device->pCci->led_drv_i2c_info[0]->i2cDeviceType ==
                        LWSWITCH_I2C_DEVICE_TCA6507;
    ledPerDrvr = bDrvrIsTca6507 ?
                    OSFP_LED_PER_DRVR_TCA6507 : OSFP_LED_PER_DRVR_PCA9685BS;

    drvrIdx = osfp / ledPerDrvr;

    if ((osfp >= (sizeof(device->pCci->osfpMaskAll) * 8)) ||
        (osfp >= device->pCci->led_drv_num * ledPerDrvr) ||
        (drvrIdx >= LWSWITCH_CCI_LED_DRV_NUM_MAX))
    {
        return -LWL_BAD_ARGS;
    }

    //
    // Since the LED driver is not a part of the OSFP module,
    // we do not require that the module is present
    //
    if (!(device->pCci->osfpMaskAll & LWBIT(osfp)))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: osfp %d is not identified\n",
            __FUNCTION__, osfp);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    nextLedState = device->pCci->xcvrNextLedState[osfp];
    ledState = bSetLocate ?
            LW_MSGBOX_DATA_OPTICAL_XCEIVER_CTL_LED_STATE_BLINKING_AMBER :
            nextLedState;

    // save HW state
    device->pCci->xcvrLwrrentLedState[osfp] = ledState;

    return (bDrvrIsTca6507 ?
            _cciSetXcvrLedState_Tca6507 : _cciSetXcvrLedState_Pca9685BS)
                (device, client,
                device->pCci->led_drv_i2c_info[drvrIdx],
                osfp % ledPerDrvr, ledState);
}

/*
 * @brief Determine which OSFP transceivers are connected
 *
 */
void
cciDetectXcvrsPresent
(
    lwswitch_device *device
)
{
    LwU32 maskPresent;

    if ((device->pCci == NULL) || (!device->pCci->bInitialized))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return;
    }

    maskPresent = 0;

    switch (device->pCci->boardId)
    {
        case LWSWITCH_BOARD_ID_DELTA:
        {
            // A00 carriers do not have Present IO Expander
            if (device->pCci->boardPartitionType == LWSWITCH_BOARD_PARTITION_E4760_A00 ||
                device->pCci->boardPartitionType == LWSWITCH_BOARD_PARTITION_E4761_A00)
            {
                _lwswitch_ping_cci_devices(device, &maskPresent);
            }
            else
            {
                _lwswitch_detect_presence_cci_devices_P479X_B00(device, &maskPresent);
            }
            break;
        }
        case LWSWITCH_BOARD_ID_WOLF:
        {
            _lwswitch_detect_presence_cci_devices_E3597_B00(device, &maskPresent);
            break;
        }
        default:
        {
            _lwswitch_ping_cci_devices(device, &maskPresent);
            break;
        }
    }

    // Only add here to avoid removing modules that are present but intermittently busy
    device->pCci->osfpMaskPresent |= maskPresent;
}

/*
 * @brief Get the bitset mask of connected OSFP transceivers
 *
 */
LwlStatus
cciGetXcvrMask
(
    lwswitch_device *device,
    LwU32           *pMaskAll,
    LwU32           *pMaskPresent
)
{
    if ((device->pCci == NULL) || (!device->pCci->bInitialized))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (pMaskAll != NULL)
    {
        // Maintain behavior for boards where cage mask is made with I2C pings
        if ((device->pCci->cagesMask != device->pCci->osfpMaskAll) &&
            ((device->pCci->boardId == LWSWITCH_BOARD_ID_E4760_A00) ||
             (device->pCci->boardId == LWSWITCH_BOARD_ID_E4761_A00) ||
             (device->pCci->boardId == LWSWITCH_BOARD_ID_DELTA)))
        {
            LWSWITCH_PRINT(device, ERROR,
            "%s: Mismatch between cages mask and osfp mask all. Returning osfp mask all.\n",
            __FUNCTION__);
            device->pCci->cagesMask = device->pCci->osfpMaskAll;
        }
        *pMaskAll = device->pCci->cagesMask;
    }

    if (pMaskPresent != NULL)
    {
        device->pCci->modulesMask = device->pCci->osfpMaskPresent;
        *pMaskPresent = device->pCci->modulesMask;
    }

    return LWL_SUCCESS;
}

/*
 * @brief Register the OSFP transceivers as present/missing
 *
 */
LwlStatus
cciSetXcvrPresent
(
    lwswitch_device *device,
    LwU32            osfp,
    LwBool           bPresent
)
{
    if ((device->pCci == NULL) || (!device->pCci->bInitialized))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (!(device->pCci->osfpMaskAll & LWBIT(osfp)))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: osfp %d is not identified\n",
            __FUNCTION__, osfp);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (bPresent)
    {
        device->pCci->osfpMaskPresent |= LWBIT(osfp);
    }
    else
    {
        device->pCci->osfpMaskPresent &= ~LWBIT(osfp);
    }

    return LWL_SUCCESS;
}

/*
 * @brief Cache the FRU EEPROM contents
 *
 */
LwlStatus
cciRomCache
(
    lwswitch_device *device,
    LwU32            client
)
{
    PCCI                                 pCci = device->pCci;
    unsigned                             idx;
    unsigned                             romSize = 1 << AT24C02D_INDEX_SIZE;
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS     i2c_params = { 0 };
    LwlStatus                            status;

    if ((pCci == NULL) || (!device->pCci->bInitialized))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (pCci->rom_num == 0)
    {
        return -LWL_ERR_NOT_SUPPORTED;
    }

    i2c_params.bIsRead       = LW_TRUE;
    i2c_params.acquirer      = client;
    i2c_params.messageLength = romSize;
    i2c_params.index[0]      = 0;
    i2c_params.flags =
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _START, _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _STOP, _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _ADDRESS_MODE, _7BIT) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, _ONE) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _FLAVOR, _HW) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, _400KHZ);

    for (idx = 0; idx < pCci->rom_num; ++idx)
    {
        struct LWSWITCH_I2C_DEVICE_DESCRIPTOR   *pDesc = pCci->rom_i2c_info[idx];
        LwU8                                    *pBuf;

        if (pDesc->i2cDeviceType != LWSWITCH_I2C_DEVICE_AT24C02D)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: ROM at %d/%02x type %02x is not supported.\n",
                __FUNCTION__, pDesc->i2cPortLogical,
                pDesc->i2cAddress, pDesc->i2cDeviceType);
            LWSWITCH_ASSERT(0);
            continue;
        }

        pBuf = lwswitch_os_malloc(romSize);
        if (pBuf == NULL)
        {
            return -LWL_NO_MEM;
        }

        i2c_params.port          = pDesc->i2cPortLogical;
        i2c_params.address       = pDesc->i2cAddress;

        if ((status = lwswitch_ctrl_i2c_indexed(device, &i2c_params)) == LWL_SUCCESS)
        {
            lwswitch_os_memcpy(pBuf, i2c_params.message, romSize);
            pCci->romCache[idx] = pBuf;
        }
        else
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: reading ROM at %d/%02x failed: %d\n",
                __FUNCTION__, pDesc->i2cPortLogical,
                pDesc->i2cAddress, status);

            lwswitch_os_free(pBuf);
        }
    }

    return LWL_SUCCESS;
}

/*
 * @brief Setup TX/RX module lanes
 *
 *  Enable or Disable TX/RX lanes/channels on an optical module
 *  (ref CMIS rev4.0, Table 8-46 Lane-specific Control Fields (Page 10h))
 */
LwlStatus
cciSetupLanes
(
    lwswitch_device *device,
    LwU32            client,
    LwBool           bTx,
    LwBool           bEnable
)
{
    LwU8 temp;
    LwU32 osfp;

    if ((device->pCci == NULL) || (!device->pCci->bInitialized))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    FOR_EACH_INDEX_IN_MASK(32, osfp, device->pCci->osfpMaskPresent)
    {
        cciSetBankAndPage(device, client, osfp, 0, 0x10);

        if (bEnable)
        {
            temp = 0;
        }
        else
        {
            temp = 0xff;
        }

        if (bTx)
        {
            cciWrite(device, client, osfp, 130, 1, &temp);
        }
        else
        {
            cciWrite(device, client, osfp, 138, 1, &temp);
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return LWL_SUCCESS;
}

/*
 * @brief Wait for CDR to Lock on the RX Input
 *
 * The PLL in the CDR circuit maintains that lock by continuously monitoring the received signal.
 * If the lock cannot be maintained, the Loss of Lock (LOL) alarm is generated.
 *
 * This function waits for remote end to enable IOBIST and polls on LOL flag to clear on RX-Input.
 *
 * The read operation clears the latched flag if CDR is locked.
 *
 * (ref CMIS rev4.0, Table 8-61 Rx Flags (Page 11h))
 */
LwlStatus
cciPollRxCdrLock
(
    lwswitch_device *device,
    LwU32           client
)
{
    LWSWITCH_TIMEOUT timeout;
    LwU8 temp;
    LwU32 osfp;

    if ((device->pCci == NULL) || (!device->pCci->bInitialized))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    lwswitch_timeout_create(LWSWITCH_INTERVAL_1SEC_IN_NS, &timeout);

    FOR_EACH_INDEX_IN_MASK(32, osfp, device->pCci->osfpMaskPresent)
    {
        cciSetBankAndPage(device, client, osfp, 0, 0x11); // page 11h, lane status

        do
        {
            cciRead(device, client, osfp, 148, 1, &temp); // RX input LOL

            if (temp == 0)
            {
                break;
            }

            if (lwswitch_timeout_check(&timeout))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: Timeout waiting for RX input CDRs to be locked (0x%x)!\n",
                     __FUNCTION__, temp);
                return -LWL_ERR_GENERIC;
            }

            lwswitch_os_sleep(10);

        } while (LW_TRUE);
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return LWL_SUCCESS;
}

/*
 * @brief Configures individual RX and TX osfp lanes to LWLink mode
 *
 * Freeze Maintenance(FM) : When pre-training starts this should be set, and later
 *   cleared once link becomes active.
 *
 * Restart Training(RT) : must be set only when linkup flow is reset and pre-training
 *   should be performed.
 *
 * Lwlink Mode(LW) : Must be set to 0x1 for LWLink Mode.
 *
 * CDB address 0xCD19. Write sequence -
 *  [0, (0,0,0,0,FM,0,0,RT,LW), TX CH Mask 15..8(00h), TX CH Mask 7..0(FFh), RX CH Mask 15..8 (00h), RX CH Mask 7..0(FFh)]
 *
 * (ref cdb_prospector.pdf)
 */
LwlStatus
cciConfigureLwlinkMode
(
    lwswitch_device *device,
    LwU32            client,
    LwU32            linkId,
    LwBool           bTx,
    LwBool           freeze_maintenance,
    LwBool           restart_training,
    LwBool           lwlink_mode
)
{
    LwU8 payload[120];
    LwU8 response[120];
    LwU32 resLength;
    LwlStatus status;
    LwU32 osfp;
    LwU8 laneMask;

    if ((device->pCci == NULL) || (!device->pCci->bInitialized))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (_lwswitch_cci_get_module_id(device, linkId, &osfp) != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get moduleId associated with link %d\n",
            __FUNCTION__, linkId);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (_lwswitch_cci_get_lanemask(device, linkId, &laneMask) != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get osfp lanemask associated with link %d\n",
            __FUNCTION__, linkId);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    payload[0] = 0;
    payload[1] = (freeze_maintenance<<4)+(restart_training<<1)+lwlink_mode;
    payload[2] = 0;
    payload[3] = bTx ? laneMask : 0;
    payload[4] = 0;
    payload[5] = bTx ? 0 : laneMask;

    status = cciSendCDBCommandAndGetResponse(device, client, osfp,
        0xcd19, 6, payload, &resLength, response, LW_FALSE);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to configure lwlink mode\n",
            __FUNCTION__);
        return status;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_cciPollForPreTraining
(
    lwswitch_device *device,
    LwU32           client,
    LwU32           linkId,
    LwBool          bTx
)
{
    LwlStatus status;
    LwU8 response[120];
    LwU32 resLength;
    LwU32 osfp;
    LwU8 train_mask;
    LwU8 lane_mask;

    if ((device->pCci == NULL) || (!device->pCci->bInitialized))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (_lwswitch_cci_get_module_id(device, linkId, &osfp) != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get moduleId associated with link %d\n",
            __FUNCTION__, linkId);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (_lwswitch_cci_get_lanemask(device, linkId, &lane_mask) != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get osfp lanemask associated with link %d\n",
            __FUNCTION__, linkId);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    status = cciSendCDBCommandAndGetResponse(device, client, osfp,
        0xcd20, 0, NULL, &resLength, response, LW_FALSE);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get lwlink status\n",
            __FUNCTION__);
        return status;
    }

    train_mask = bTx ? response[1] : response[3];
    if ((lane_mask & train_mask) == lane_mask)
    {
        LWSWITCH_PRINT(device, INFO,
            "%s: pre-training completed on link %d!\n",
            __FUNCTION__, linkId);
        return LWL_SUCCESS;
    }
    else 
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: pre-training failed on link %d!\n",
            __FUNCTION__, linkId);
    }

    if (bTx)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: OSFP %d - TX-Input trained lanemask : 0x%x, untrained lanemask : 0x%x on link %d\n",
            __FUNCTION__, osfp, train_mask, (~train_mask & 0xff), linkId);
    }
    else
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: OSFP %d - RX-Input trained linkmask : 0x%x, untrained linkmask : 0x%x\n on link %d",
            __FUNCTION__, osfp, train_mask, (~train_mask & 0xff), linkId);
    }

    return -LWL_ERR_GENERIC;
}

/*
 * @brief Poll for optical pre training
 *
 * CDB address CD20h - Retrieves individual RX and TX channels training status.
 * Read sequence -
 *  [TX CH Mask 15..8(00h), TX CH Mask 7..0(FFh), RX CH Mask 15..8 (00h), RX CH Mask 7..0(FFh)]
 *
 * (ref cdb_prospector.pdf)
 */
LwlStatus
cciPollForPreTraining
(
    lwswitch_device *device,
    LwU32           client,
    LwU32           linkId,
    LwBool          bTx
)
{
    LwlStatus status;

    status = _cciPollForPreTraining(device, client, linkId, bTx);

    LWSWITCH_PRINT(device, INFO,
                "%s: Printing grading values after poll for pre-training.\n",
                __FUNCTION__); 
    
    cciPrintGradingValues(device, client, linkId);

    return status;
}

LwlStatus
cciApplyControlSetValues
(
    lwswitch_device *device,
    LwU32           client,
    LwU32           moduleMask
)
{
    LwU8 temp[8];
    LwU32 osfp;
    LwU32 missingOsfps;
    LwU32 i;
    LwU32 deviceId;
    const LwU8 *arr = NULL;
    LWSWITCH_BIOS_LWLINK_CONFIG    *bios_config;

    if ((device->pCci == NULL) || (!device->pCci->bInitialized))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    bios_config = lwswitch_get_bios_lwlink_config(device);
    if ((bios_config == NULL) || (bios_config->bit_address == 0))
    {
        LWSWITCH_PRINT(device, WARN,
            "%s: VBIOS LwLink configuration table not found\n",
            __FUNCTION__);
        return -LWL_ERR_GENERIC;
    }

    deviceId = bios_config->link_base_entry_assigned;

    if (!((device->pCci->osfpMaskPresent & moduleMask) == moduleMask))
    {
        missingOsfps = ~(device->pCci->osfpMaskPresent & moduleMask) &
                      moduleMask;
        (void) missingOsfps;
        LWSWITCH_PRINT(device, WARN,
            "%s: osfps 0x%x missing\n",
            __FUNCTION__, missingOsfps);
    }

    LWSWITCH_PRINT(device, INFO,
         "%s: the device ID is %d\n",
         __FUNCTION__, deviceId);
    
    switch(device->pCci->boardId)
    {
        case LWSWITCH_BOARD_ID_E4760_A00:
        case LWSWITCH_BOARD_ID_E4761_A00:
        {
            arr = cci_osfp_txeq_e4700_0_lr10;
            break;
        }
        case LWSWITCH_BOARD_ID_DELTA:
        {
            switch (deviceId)
            {
                case 0:
                    arr = cci_osfp_txeq_prospector_0_lr10;
                    break;
                case 1:
                    arr = cci_osfp_txeq_prospector_1_lr10;
                    break;
                case 2:
                    arr = cci_osfp_txeq_prospector_2_lr10;
                    break;
                case 3:
                    arr = cci_osfp_txeq_prospector_3_lr10;
                    break;
                case 4:
                    arr = cci_osfp_txeq_prospector_4_lr10;
                    break;
                case 5:
                    arr = cci_osfp_txeq_prospector_5_lr10;
                    break;
                default:
                    LWSWITCH_PRINT(device, ERROR,
                        "%s: Invalid device id %d\n",
                        __FUNCTION__, deviceId);
                    return -LWL_ERR_GENERIC;
            }
            break;
        }
        case LWSWITCH_BOARD_ID_WOLF:
        {
            switch (deviceId)
            {
                case 0:
                    arr = cci_osfp_txeq_wolf_0_lr10;
                    break;
                case 1:
                    arr = cci_osfp_txeq_wolf_1_lr10;
                    break;
                default:
                    LWSWITCH_PRINT(device, ERROR,
                        "%s: Invalid device id %d\n",
                        __FUNCTION__, deviceId);
                    return -LWL_ERR_GENERIC;
            }
            break;
        }
        default:
            break;
    }

    if ((device->pCci->boardId == LWSWITCH_BOARD_ID_E4760_A00) ||
        (device->pCci->boardId == LWSWITCH_BOARD_ID_E4761_A00))
    {
        FOR_EACH_INDEX_IN_MASK(32, osfp, moduleMask)
        {
            cciSetBankAndPage(device, client, osfp, 0, 0x10);

            // Page 10h, bytes 162-169 (Staged Control Set 0) [pre/post cursor]
            for (i = 0; i < 4; i++)
                temp[i] = arr[(osfp * 12) + i];
            cciWrite(device, client, osfp, 162, 4, temp); //pre-cursor, 0-7 in 0.5dB increments

            for (i = 0; i < 4; i++)
                temp[i] = arr[(osfp * 12) + i + 4];
            cciWrite(device, client, osfp, 166, 4, temp); // post-cursor, 0-7 in 1dB increments

            // output amplitude, 650mVpp desired
            // 0: 100-400mVpp,
            // 1: 300-600mVpp,
            // 2: 400-800mVpp,
            // 3: 600-1200mVpp
            // 4-14: reserved
            for (i = 0; i < 4; i++)
                temp[i] = arr[(osfp * 12) + i + 8];
            cciWrite(device, client, osfp, 170, 4, temp);

            // Updating explicit control bit by RMW.
            for (i = 0; i < 8; i++)
            {
                cciRead(device, client, osfp, 145+i, 1, temp);
                temp[0] = temp[0] | 0x1;
                cciWrite(device, client, osfp, 145+i, 1, temp);
            }

            // apply control set (apply_datapathinit)
            // apply control set (apply_immediate)
            temp[0] = 0xff;
            cciWrite(device, client, osfp, 144, 1, temp);
        }FOR_EACH_INDEX_IN_MASK_END;
    }

    else if (device->pCci->boardId == LWSWITCH_BOARD_ID_DELTA ||
             device->pCci->boardId == LWSWITCH_BOARD_ID_WOLF)
    {  
        FOR_EACH_INDEX_IN_MASK(32, osfp, moduleMask)
        {
            cciSetBankAndPage(device, client, osfp, 0, 0x10);

            // Page 10h, bytes 162-169 (Staged Control Set 0) [pre/post cursor]
            for (i = 0; i < 4; i++)
                temp[i] = arr[(osfp * 12) + i];

            cciWrite(device, client, osfp, 162, 4, temp); //pre-cursor, 0-7 in 0.5dB increments

            LWSWITCH_PRINT(device, INFO,
                "%s: precursor value for device ID %d, osfp %d are %x %x %x %x %x %x %x %x\n",
                __FUNCTION__, deviceId, osfp, temp[0] & 0x0f, temp[0] >> 4, temp[1] & 0x0f,
                temp[1] >> 4, temp[2] & 0x0f, temp[2] >> 4, temp[3] & 0x0f, temp[3] >> 4);

            for (i = 0; i < 4; i++)
                temp[i] = arr[(osfp * 12) + i + 4];

            cciWrite(device, client, osfp, 166, 4, temp); // post-cursor, 0-7 in 1dB increments

            LWSWITCH_PRINT(device, INFO,
                "%s: postlwrsor value for device ID %d, osfp %d are %x %x %x %x %x %x %x %x\n",
                __FUNCTION__, deviceId, osfp, temp[0] & 0x0f, temp[0] >> 4, temp[1] & 0x0f,
                temp[1] >> 4, temp[2] & 0x0f, temp[2] >> 4, temp[3] & 0x0f, temp[3] >> 4);

            // output amplitude, 650mVpp desired
            // 0: 100-400mVpp,
            // 1: 300-600mVpp,
            // 2: 400-800mVpp,
            // 3: 600-1200mVpp
            // 4-14: reserved
            for (i = 0; i < 4; i++)
                temp[i] = arr[(osfp * 12) + i + 8];

            cciWrite(device, client, osfp, 170, 4, temp);

            LWSWITCH_PRINT(device, INFO,
                "%s: amplitude value for device ID %d, osfp %d are %x %x %x %x %x %x %x %x\n",
                __FUNCTION__, deviceId, osfp, temp[0] & 0x0f, temp[0] >> 4, temp[1] & 0x0f,
                temp[1] >> 4, temp[2] & 0x0f, temp[2] >> 4, temp[3] & 0x0f, temp[3] >> 4);

            // Updating explicit control bit by RMW.
            for (i = 0; i < 8; i++)
            {
                cciRead(device, client, osfp, 145+i, 1, temp);
                temp[0] = temp[0] | 0x1;
                cciWrite(device, client, osfp, 145+i, 1, temp);
            }

            // apply control set (apply_datapathinit)
            // apply control set (apply_immediate)
            temp[0] = 0xff;
            cciWrite(device, client, osfp, 144, 1, temp);
        }FOR_EACH_INDEX_IN_MASK_END;
    }

    return LWL_SUCCESS;
}

LwlStatus
cciGetGradingValues
(
    lwswitch_device *device,
    LwU32           client,
    LwU32           linkId,
    LwU8            *laneMask,
    LWSWITCH_CCI_GRADING_VALUES *pGrading
)
{
    LwlStatus status;
    LwU8 response[120];
    LwU32 resLength;
    LwU32 osfp;
    LwU8 lane_mask;
    LwU32 lane;
    LwU32 i;

    if ((device->pCci == NULL) || (!device->pCci->bInitialized))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (_lwswitch_cci_get_module_id(device, linkId, &osfp) != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get moduleId associated with link %d\n",
            __FUNCTION__, linkId);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (_lwswitch_cci_get_lanemask(device, linkId, &lane_mask) != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get osfp lanemask associated with link %d\n",
            __FUNCTION__, linkId);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    status = cciSendCDBCommandAndGetResponse(device, client, osfp,
        0xdb00, 0, NULL, &resLength, response, LW_FALSE);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get xvcr grading values\n",
            __FUNCTION__);
        return status;
    }

    if (resLength != 32)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Grading length expected to be 32, actual length %d\n",
            __FUNCTION__, resLength);
        return -LWL_ERR_GENERIC;
    }

    *laneMask = lane_mask;
    for (i = 0; i < 8; i++)
    {
        lane = i;
        if (lane_mask & LWBIT(lane))
        {
            pGrading->tx_init[lane] = response[i];
        }
    }

    for (i = 16; i < 24; i++)
    {
        lane = i - 16;
        if (lane_mask & LWBIT(lane))
        {
            pGrading->rx_init[lane] = response[i];
        }
    }

    for (i = 8; i < 16; i++)
    {
        lane = i - 8;
        if (lane_mask & LWBIT(lane))
        {
            pGrading->tx_maint[lane] = response[i];
        }
    }

    for (i = 24; i < 32; i++)
    {
        lane = i - 24;
        if (lane_mask & LWBIT(lane))
        {
            pGrading->rx_maint[lane] = response[i];
        }
    }

    return LWL_SUCCESS;
}

/*
 * @brief Get the module state of qsfp
 *
 * (ref QSFP DD CMIS ref manual)
 * byte 3, bits 3-1 indicate the module state
 */
LwlStatus
cciGetModuleState
(
    lwswitch_device *device,
    LwU32           client,
    LwU32           linkId,
    LWSWITCH_CCI_MODULE_STATE *pInfo
)
{
    LwlStatus status;
    LwU32 osfp;
    LwU8 temp;

    if ((device->pCci == NULL) || (!device->pCci->bInitialized))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (_lwswitch_cci_get_module_id(device, linkId, &osfp) != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get moduleId associated with link %d\n",
            __FUNCTION__, linkId);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    // get status
    status = cciRead(device, client, osfp, 3, 1, &temp);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get fault status on link %d\n",
            __FUNCTION__, linkId);
        return -LWL_ERR_GENERIC;
    }

    temp = temp >> 1;
    temp = temp & 7;

    switch(temp)
    {
        case 1:
        pInfo->bLModuleLowPwrState = LW_TRUE;
        LWSWITCH_PRINT(device, INFO,
                "%s:  Module is in LowPwr state on link %d\n",
                __FUNCTION__, linkId);
        break;

    case 2:
        pInfo->bLModulePwrUpState = LW_TRUE;
        LWSWITCH_PRINT(device, INFO,
                "%s:  Module is in PwrUp state on link %d\n",
                __FUNCTION__, linkId);
        break;

    case 3:
        pInfo->bLModuleReadyState = LW_TRUE;
        LWSWITCH_PRINT(device, INFO,
                "%s:  Module is in Ready state on link %d\n",
                __FUNCTION__, linkId);
        break;

    case 4:
        pInfo->bLModulePwrDnState = LW_TRUE;
        LWSWITCH_PRINT(device, INFO,
                "%s:  Module is in PwrDn state on link %d\n",
                __FUNCTION__, linkId);
        break;

    case 5:
        pInfo->bLFaultState = LW_TRUE;
        LWSWITCH_PRINT(device, INFO,
                "%s:  Module is in Fault state on link %d\n",
                __FUNCTION__, linkId);
        break;

    default:
        pInfo->bLReserved = LW_TRUE;
        LWSWITCH_PRINT(device, INFO,
                "%s:  Module is in Reserved state on link %d\n",
                __FUNCTION__, linkId);
        break;

    }

    return LWL_SUCCESS;
}

/*
 * @brief Module Flags for Alarm and Fault Bits
 * (ref QSFP DD CMIS ref manual)
 * Page 0, bytes 8-11 indicate voltage and
 * temperature alarm and fault bits
 */
LwlStatus
cciGetModuleFlags
(
    lwswitch_device *device,
    LwU32           client,
    LwU32           linkId,
    LWSWITCH_CCI_MODULE_FLAGS *pModuleFlags
)
{
    LwlStatus status;
    LwU32 osfp;
    LwU8 temp[4];

    if ((device->pCci == NULL) || (!device->pCci->bInitialized))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (_lwswitch_cci_get_module_id(device, linkId, &osfp) != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get moduleId associated with link %d\n",
            __FUNCTION__, linkId);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    status = cciRead(device, client, osfp, 8, 4, temp);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get module flags on link %d\n",
            __FUNCTION__, linkId);
        return -LWL_ERR_GENERIC;
    }

    // temp 0 gets the 8th byte of page 0
    // 8th byte contains L cdb block and faults
    if (LWBIT(7) & temp[0])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-CDB Block 2 complete on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLCDBBlock2Complete = LW_TRUE;
    }

    if (LWBIT(6) & temp[0])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-CDB Block 1 complete on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLCDBBlock1Complete = LW_TRUE;
    }

    if (LWBIT(0) & temp[0])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Module State Changed on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLModuleStateChange = LW_TRUE;
    }

    if (LWBIT(1) & temp[0])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Module Firmware fault on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLModuleFirmwareFault = LW_TRUE;
    }

    if (LWBIT(2) & temp[0])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Datapath Firmware fault on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bDatapathFirmwareFault = LW_TRUE;
    }

    // temp 1 gets the byte 9 of page 0
    // Vcc and Temp Warning and alarm bits are present in byte 9
    if (LWBIT(0) & temp[1])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Temp High Alarm on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLTempHighAlarm = LW_TRUE;
    }

    if (LWBIT(1) & temp[1])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Temp Low Alarm on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLTempLowAlarm = LW_TRUE;
    }

    if (LWBIT(2) & temp[1])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Temp High Warning on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLTempHighWarn = LW_TRUE;
    }

    if (LWBIT(3) & temp[1])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Temp Low Warning on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLTempLowWarn = LW_TRUE;
    }

    if (LWBIT(4) & temp[1])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Vcc3.3v High Alarm on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLVccHighAlarm = LW_TRUE;
    }

    if (LWBIT(5) & temp[1])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Vcc3.3v Low Alarm on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLVccLowAlarm = LW_TRUE;
    }

    if (LWBIT(6) & temp[1])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Vcc3.3v High Warning on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLVccHighWarn = LW_TRUE;
    }

    if (LWBIT(7) & temp[1])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Vcc3.3v Low Warning on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLVccLowWarn = LW_TRUE;
    }

    // temp 2gets the byte 10 of page 0
    // Byte 10 has the Aux 2 and Aux 1 warning and alarm information
    if (LWBIT(0) & temp[2])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Aux 1 High Alarm on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLAux1HighAlarm = LW_TRUE;
    }

    if (LWBIT(1) & temp[2])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Aux 1 Low Alarm on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLAux1LowAlarm = LW_TRUE;
    }

    if (LWBIT(2) & temp[2])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Aux 1 High Warning on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLAux1HighWarn = LW_TRUE;
    }

    if (LWBIT(3) & temp[2])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Aux 1 Low Warning on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLAux1LowWarn = LW_TRUE;
    }

    if (LWBIT(4) & temp[2])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Aux 2 High Alarm on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLAux2HighAlarm = LW_TRUE;
    }

    if (LWBIT(5) & temp[2])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Aux 2 Low Alarm on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLAux2LowAlarm = LW_TRUE;
    }

    if (LWBIT(6) & temp[2])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Aux 2 High Warning on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLAux2HighWarn = LW_TRUE;
    }

    if (LWBIT(7) & temp[2])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Aux 2 Low Warning on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLAux2LowWarn = LW_TRUE;
    }

    // temp 3 gets the value of byte 11 of page 0
    // byte 11 has Vendor defined warning and alarm, along with
    // Aux 3 warning and alarm
    if (LWBIT(0) & temp[3])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Aux 3 High Alarm on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLAux3HighAlarm = LW_TRUE;
    }

    if (LWBIT(1) & temp[3])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Aux 3 Low Alarm on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLAux3LowAlarm = LW_TRUE;
    }

    if (LWBIT(2) & temp[3])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Aux 3 High Warning on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLAux3HighWarn = LW_TRUE;
    }

    if (LWBIT(3) & temp[3])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Aux 3 Low Warning on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLAux3LowWarn = LW_TRUE;
    }

    if (LWBIT(4) & temp[3])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Vendor High Alarm on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLVendorHighAlarm = LW_TRUE;
    }

    if (LWBIT(5) & temp[3])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Vendor Low Alarm on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLVendorLowAlarm = LW_TRUE;
    }

    if (LWBIT(6) & temp[3])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Vendor High Warning on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLVendorHighWarn = LW_TRUE;
    }

    if (LWBIT(7) & temp[3])
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: L-Vendor Low Warning on osfp %d, link %d.\n",
            __FUNCTION__, osfp, linkId);
        pModuleFlags->bLVendorLowWarn = LW_TRUE;
    }

    return LWL_SUCCESS;
}

/*
 * @brief Gets the mapping between cageIndex and link Ids
 * Returns a bitmask containing all links mapped to the given
 * cage. Also returns a value that encodes other information
 * including the mapping between OSFP link lane and Lwlink link
 */
LwlStatus
cciGetCageMapping
(
    lwswitch_device *device,
    LwU8            cageIndex,
    LwU64           *pLinkMask,
    LwU64           *pEncodedValue
)
{
    LWSWITCH_CCI_MODULE_LINK_LANE_MAP *p_lwswitch_cci_osfp_map;
    LwU64 linkMask;
    LwU64 encodedValue;
    LwU8 *pEncodedByte;
    LwU32 lwswitch_cci_osfp_map_size;
    LwU32 i;
    LwU8 linkId;
    LwU8 osfpLane;

    if ((device->pCci == NULL) || (!device->pCci->bInitialized))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not initialized\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    linkMask = 0;
    encodedValue = 0;
    pEncodedByte = (LwU8 *)&encodedValue;
    p_lwswitch_cci_osfp_map = device->pCci->osfp_map;
    lwswitch_cci_osfp_map_size = device->pCci->osfp_map_size;

    for (i = 0; i < lwswitch_cci_osfp_map_size; i++)
    {
        if (p_lwswitch_cci_osfp_map[i].moduleId == cageIndex)
        {
            linkId = p_lwswitch_cci_osfp_map[i].linkId;
            LWSWITCH_ASSERT(linkId <= 63);

            linkMask |= LWBIT64(linkId);
            FOR_EACH_INDEX_IN_MASK(8, osfpLane, p_lwswitch_cci_osfp_map[i].laneMask)
            {
                pEncodedByte[osfpLane] =
                    REF_NUM(LWSWITCH_CCI_CMIS_LWLINK_MAPPING_ENCODED_VALUE_LINK_ID, linkId);
            }
            FOR_EACH_INDEX_IN_MASK_END;
        }
    }

    if (pLinkMask != NULL)
    {
        *pLinkMask = linkMask;
    }

    if (pEncodedValue != NULL)
    {
        *pEncodedValue = encodedValue;
    }

    return LWL_SUCCESS;
}

/*
 * @brief Get voltage of the module
 * Returns the voltage in mV
 * refer to Table 8-6 Module Monitors (Lower Page, active modules only)
 * voltage is a 16 bit value stored in 2 registers
 */
LwlStatus
cciGetVoltage
(
    lwswitch_device *device,
    LwU32           client,
    LwU32           linkId,
    LWSWITCH_CCI_VOLTAGE *pVoltage
)
{
    LwlStatus status;
    LwU32 osfp;
    LwU8 temp[2];
    LwU16 voltage;

    if ((device->pCci == NULL) || (!device->pCci->bInitialized))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (_lwswitch_cci_get_module_id(device, linkId, &osfp) != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get moduleId associated with link %d\n",
            __FUNCTION__, linkId);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    // get status
    status = cciRead(device, client, osfp, 16, 2, temp);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to get voltage on link %d\n",
            __FUNCTION__, linkId);
        return -LWL_ERR_GENERIC;
    }

    voltage = (temp[0] << 8) | (temp[1] & 0xff);
    pVoltage->voltage_mV = voltage/10;
    LWSWITCH_PRINT(device, INFO,
        "%s: voltage is : %d mV on link %d\n",
        __FUNCTION__, voltage, linkId);

    return LWL_SUCCESS;
}

static LwlStatus
_cciCmisAccessSetup
(
    lwswitch_device *device,
    LwU8            cageIndex,
    LwU8            bank,
    LwU8            page,
    LwU8            address,
    LwU8            count
)
{
    LwlStatus status;
    LwU32 cagesMask;

    status = cciGetXcvrMask(device, &cagesMask, NULL);
    if (status != LWL_SUCCESS)
    {
        return status;
    }

    if (!(cagesMask & LWBIT(cageIndex)))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Provided cage index does not exist.\n",
            __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    if (count > LWSWITCH_CCI_CMIS_MEMORY_ACCESS_BUF_SIZE)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Count exceeds buffer size.\n",
            __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    if (address >= 0x80)
    {
        status = cciSetBankAndPage(device, LWSWITCH_I2C_ACQUIRER_CCI_UX,
                                   cageIndex, bank, page);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Failed to set bank and page.\n",
                __FUNCTION__);
            return status;
        }
    }

    return status;
}

/*!
 * @brief Read from specified module cage.
 *        Sets the bank and page if necessary.
 */
LwlStatus
cciCmisRead
(
    lwswitch_device *device,
    LwU8            cageIndex,
    LwU8            bank,
    LwU8            page,
    LwU8            address,
    LwU8            count,
    LwU8            *pData
)
{
    LwU8 savedBank;
    LwU8 savedPage;
    LwlStatus status;

    if ((device->pCci == NULL) || (!device->pCci->bInitialized))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    // Save previous bank and page
    if (address >= 0x80)
    {
        status = cciGetBankAndPage(device, LWSWITCH_I2C_ACQUIRER_CCI_UX,
                                   cageIndex, &savedBank, &savedPage);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Failed to save bank and page.\n",
                __FUNCTION__);
            return status;
        }
    }

    status = _cciCmisAccessSetup(device, cageIndex, bank, page,
                                 address, count);
    if (status != LWL_SUCCESS)
    {
        return status;
    }

    status = cciRead(device, LWSWITCH_I2C_ACQUIRER_CCI_UX,
                     cageIndex, address, count, pData);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to read from module cage %d\n",
            __FUNCTION__, cageIndex);
        return status;
    }

    // Restore previous bank and page
    if (address >= 0x80)
    {
        status = cciSetBankAndPage(device, LWSWITCH_I2C_ACQUIRER_CCI_UX,
                                   cageIndex, savedBank, savedPage);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Failed to restore bank and page.\n",
                __FUNCTION__);
            return status;
        }
    }

    return LWL_SUCCESS;
}

/*!
 * @brief Write to specified module cage.
 *        Sets the bank and page if necessary.
 */
LwlStatus
cciCmisWrite
(
    lwswitch_device *device,
    LwU8            cageIndex,
    LwU8            bank,
    LwU8            page,
    LwU8            address,
    LwU8            count,
    LwU8            *pData
)
{
    LwU8 savedBank;
    LwU8 savedPage;
    LwlStatus status;

    if ((device->pCci == NULL) || (!device->pCci->bInitialized))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    // Save previous bank and page
    if (address >= 0x80)
    {
        status = cciGetBankAndPage(device, LWSWITCH_I2C_ACQUIRER_CCI_UX,
                                   cageIndex, &savedBank, &savedPage);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Failed to save bank and page.\n",
                __FUNCTION__);
            return status;
        }
    }

    status = _cciCmisAccessSetup(device, cageIndex, bank, page,
                                 address, count);
    if (status != LWL_SUCCESS)
    {
        return status;
    }

    status = cciWrite(device, LWSWITCH_I2C_ACQUIRER_CCI_UX,
                     cageIndex, address, count, pData);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to write to module cage %d\n",
            __FUNCTION__, cageIndex);
        return status;
    }

    // Restore previous bank and page
    if (address >= 0x80)
    {
        status = cciSetBankAndPage(device, LWSWITCH_I2C_ACQUIRER_CCI_UX,
                                   cageIndex, savedBank, savedPage);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Failed to restore bank and page.\n",
                __FUNCTION__);
            return status;
        }
    }

    return LWL_SUCCESS;
}

void
cciPrintGradingValues 
(
    lwswitch_device *device,
    LwU32 client,
    LwU32 linkId
)
{
    #if defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)

        LwlStatus status;
        LwU8 response[120];
        LwU32 resLength;
        LwU32 retry;
        LwU32 osfp;

        if ((device->pCci == NULL) || (!device->pCci->bInitialized))
        {
            return;
        }

        if (_lwswitch_cci_get_module_id(device, linkId, &osfp) != LWL_SUCCESS)
        {
            return;
        }

        status = cciSendCDBCommandAndGetResponse(device, client, osfp,
                0xdb00, 0, NULL, &resLength, response, LW_FALSE);

        retry = 5;
        while ((status != LWL_SUCCESS) && (retry > 0))
        {
            status = cciSendCDBCommandAndGetResponse(device, client, osfp,
                0xdb00, 0, NULL, &resLength, response, LW_FALSE);
            retry--;
        }

        if (status != LWL_SUCCESS)    
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Error: Reading PRBS BER failed!\n",
                __FUNCTION__);
            return;
        }

        if (resLength != 32)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Error: StallionGetChannelBER payload length expected 32, actual %d!\n",
                __FUNCTION__, resLength);
        }
        else
        {
            LwU32 i = 0;
            (void)i;

            LWSWITCH_PRINT(device, INFO,
                "%s: link %d\n",
                __FUNCTION__, linkId);
            
            i = 0;
            LWSWITCH_PRINT(device, INFO,
                "%s: OSFP %d CH0-7 TX-Input Initial Tuning BER: %d, %d, %d, %d, %d, %d, %d, %d\n",
                __FUNCTION__, osfp, response[i], response[i+1], response[i+2], response[i+3],
                                    response[i+4], response[i+5], response[i+6], response[i+7]);

            i = 16;
            LWSWITCH_PRINT(device, INFO,
                "%s: OSFP %d CH0-7 RX-Input Initial Tuning BER: %d, %d, %d, %d, %d, %d, %d, %d\n\n",
                __FUNCTION__, osfp, response[i], response[i+1], response[i+2], response[i+3],
                                    response[i+4], response[i+5], response[i+6], response[i+7]);
            
            i = 8;
            LWSWITCH_PRINT(device, INFO,
                "%s: OSFP %d CH0-7 TX-Input Maintenance BER:    %d, %d, %d, %d, %d, %d, %d, %d\n",
                __FUNCTION__, osfp, response[i], response[i+1], response[i+2], response[i+3],
                                    response[i+4], response[i+5], response[i+6], response[i+7]);

            i = 24;
            LWSWITCH_PRINT(device, INFO,
                "%s: OSFP %d CH0-7 RX-Input Maintenance BER:    %d, %d, %d, %d, %d, %d, %d, %d\n\n",
                __FUNCTION__, osfp, response[i], response[i+1], response[i+2], response[i+3],
                                    response[i+4], response[i+5], response[i+6], response[i+7]);
        }

    #endif
}

LwlStatus
cciCmisCageBezelMarking
(
    lwswitch_device *device,
    LwU8 cageIndex, 
    char *pBezelMarking
)
{
    LwU32 cagesMask;
    LwU32 lwswitchNum;
    const char* bezelMarking;
    LwlStatus status;
    LWSWITCH_BIOS_LWLINK_CONFIG *bios_config;

    if ((device->pCci == NULL) || (!device->pCci->bInitialized))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: CCI not supported\n",
            __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    status = cciGetXcvrMask(device, &cagesMask, NULL);
    if (status != LWL_SUCCESS)
    {
        return status;
    }

    if (!(cagesMask & LWBIT(cageIndex)))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Provided cage index does not exist.\n",
            __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    bios_config = lwswitch_get_bios_lwlink_config(device);
    if ((bios_config == NULL) || (bios_config->bit_address == 0))
    {
        LWSWITCH_PRINT(device, WARN,
            "%s: VBIOS LwLink configuration table not found\n",
            __FUNCTION__);
        return -LWL_ERR_GENERIC;
    }

    lwswitchNum = bios_config->link_base_entry_assigned;
    
    switch (device->pCci->boardId)
    {
        case LWSWITCH_BOARD_ID_E4760_A00:
        case LWSWITCH_BOARD_ID_E4761_A00:
        {
            bezelMarking = cci_osfp_cage_bezel_markings_e4700_lr10[cageIndex];
            break;
        }
        case LWSWITCH_BOARD_ID_DELTA:
        {
            bezelMarking = cci_osfp_cage_bezel_markings_prospector_lr10[lwswitchNum][cageIndex];
            break;
        }
        default:
        {
            bezelMarking = NULL;
            LWSWITCH_PRINT(device, ERROR,
                "%s: Unsupported board.\n",
                __FUNCTION__);
            LWSWITCH_ASSERT(0);
            break;
        }
    }

    if (pBezelMarking != NULL)
    {                                   
        lwswitch_os_snprintf(pBezelMarking, LWSWITCH_CCI_CMIS_CAGE_BEZEL_MARKING_LEN + 1,
                            "%s", bezelMarking);
    }

    return LWL_SUCCESS;
}
