/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2019 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "g_lwconfig.h"
#include "export_lwswitch.h"
#include "ctrl_dev_internal_lwswitch.h"
#include "rom_lwswitch.h"
#include "common_lwswitch.h"
#include "haldef_lwswitch.h"

static LwU8
_lwswitch_callwlate_checksum
(
    LwU8 *data,
    LwU32 size
)
{
    LwU32 i;
    LwU8 checksum = 0;

    for (i = 0; i < size; i++)
    {
        checksum += data[i];
    }
    return -checksum;
}

static LwlStatus
_lwswitch_read_rom_bytes
(
    lwswitch_device *device,
    LWSWITCH_EEPROM_TYPE *eeprom,
    LwU32   offset,
    LwU8    *buffer,
    LwU32   buffer_size
)
{
    LWSWITCH_CTRL_I2C_INDEXED_PARAMS  i2cIndexed = {0};
    LwU32   i;
    LwlStatus retval;

    if (offset + buffer_size > (LwU32)(1 << eeprom->index_size))
    {
        LWSWITCH_PRINT(device, SETUP,
            "EEPROM offset 0x%x..0x%x out of range\n",
            offset, offset + buffer_size - 1);
        return -LWL_BAD_ARGS;
    }

    if (buffer_size > LWSWITCH_CTRL_I2C_MESSAGE_LENGTH_MAX)
    {
        LWSWITCH_PRINT(device, SETUP,
            "EEPROM read buffer (0x%x bytes) larger than max (0x%x bytes)\n",
            buffer_size, LWSWITCH_CTRL_I2C_MESSAGE_LENGTH_MAX);
        return -LWL_BAD_ARGS;
    }

    i2cIndexed.port = (LwU8)eeprom->i2c_port;
    i2cIndexed.bIsRead = LW_TRUE;
    i2cIndexed.address = (LwU16)eeprom->i2c_address;
    i2cIndexed.flags =
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _START,        _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _RESTART,      _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _STOP,         _SEND) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _ADDRESS_MODE, _7BIT) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _FLAVOR, _HW)         |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _SPEED_MODE, _400KHZ) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _BLOCK_PROTOCOL, _DISABLED) |
        DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _TRANSACTION_MODE, _NORMAL) |
        0;

    if (eeprom->index_size <= 8)
    {
        i2cIndexed.flags |=
            DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, _ONE);
        i2cIndexed.index[0] =  offset & 0x000FF;        // Read [eeprom_offset]
    }
    else
    {
        i2cIndexed.address |= ((offset & 0x30000) >> 15);
        i2cIndexed.flags |=
            DRF_DEF(SWITCH_CTRL, _I2C_FLAGS, _INDEX_LENGTH, _TWO);
        i2cIndexed.index[0] = (offset & 0x0FF00) >> 8;  // Read [eeprom_offset]
        i2cIndexed.index[1] = (offset & 0x000FF);
    }

    i2cIndexed.messageLength = LW_MIN(buffer_size, LWSWITCH_CTRL_I2C_MESSAGE_LENGTH_MAX);

    retval = lwswitch_ctrl_i2c_indexed(device, &i2cIndexed);
    if (retval != LWL_SUCCESS)
    {
        return retval;
    }

    for (i = 0; i < i2cIndexed.messageLength; i++)
    {
        buffer[i] = i2cIndexed.message[i];
    }

    return retval;
}

//
// Parse EEPROM header, if present
//
static LwlStatus
_lwswitch_read_rom_header
(
    lwswitch_device *device,
    LWSWITCH_EEPROM_TYPE *eeprom,
    LWSWITCH_FIRMWARE *firmware,
    LwU32 *offset
)
{
    LWSWITCH_EEPROM_HEADER eeprom_header = {{0}};
    LwlStatus retval;

    firmware->firmware_size = 0;
    *offset = 0x0000;

    retval = _lwswitch_read_rom_bytes(device,
        eeprom, *offset,
        (LwU8 *) &eeprom_header, sizeof(eeprom_header));
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, SETUP,
            "Unable to read ROM header\n");
        return retval;
    }

    if ((eeprom_header.signature[0] == 'N') &&
        (eeprom_header.signature[1] == 'V') &&
        (eeprom_header.signature[2] == 'L') &&
        (eeprom_header.signature[3] == 'S') &&
        (_lwswitch_callwlate_checksum((LwU8 *) &eeprom_header, sizeof(eeprom_header)) == 0x00))
    {
        // Assume eeprom_header is version 1

        *offset += eeprom_header.header_size;

        firmware->pci_vendor_id = eeprom_header.pci_vendor_id;
        firmware->pci_device_id = eeprom_header.pci_device_id;
        firmware->pci_system_vendor_id = eeprom_header.pci_system_vendor_id;
        firmware->pci_system_device_id = eeprom_header.pci_system_device_id;

        // EEPROM header firmware size field is in 512 byte blocks
        firmware->firmware_size = eeprom_header.firmware_size * 512;
    }
    else
    {
        LWSWITCH_PRINT(device, SETUP,
            "Firmware header not found\n");
        return -LWL_NOT_FOUND;
    }

    return LWL_SUCCESS;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
//
// 0x52: BIT_TOKEN_BRIDGE_FW_DATA
// https://wiki.lwpu.com/engwiki/index.php/VBIOS/Data_Structures/BIT#BIT_BRIDGE_FW_DATA
//
#endif
static LwlStatus
_lwswitch_rom_parse_bit_bridge_fw_data
(
    lwswitch_device *device,
    LWSWITCH_EEPROM_TYPE *eeprom,
    LWSWITCH_FIRMWARE *firmware,
    LWSWITCH_BIT_TOKEN *bit_token
)
{
    LWSWITCH_BIT_BRIDGE_FW_DATA bit_bridge_fw = {0};
    LwU32 copy_size;
    LwU32 bridge_fw_size;
    LwlStatus retval;

    firmware->bridge.bridge_fw_found = LW_FALSE;

    if (bit_token->data_size != sizeof(bit_bridge_fw))
    {
        LWSWITCH_PRINT(device, SETUP,
            "BIT_BRIDGE_FW_DATA: Expected data size 0x%x but found 0x%x\n",
            (LwU32) sizeof(bit_bridge_fw), bit_token->data_size);
    }

    bridge_fw_size = LW_MIN(bit_token->data_size, sizeof(bit_bridge_fw));

    // Get basic bridge-specific firmware info
    retval = _lwswitch_read_rom_bytes(device, eeprom, bit_token->data_offset,
        (LwU8 *) &bit_bridge_fw, bridge_fw_size);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, SETUP,
            "Failed to read BIT_BRIDGE_FW_DATA\n");
        return -LWL_ERR_NOT_SUPPORTED;
    }

    firmware->bridge.bridge_fw_found = LW_TRUE;

    firmware->bridge.firmware_version =
        LWSWITCH_ELEMENT_READ(&bit_bridge_fw, firmware_version, bridge_fw_size, 0);

    firmware->bridge.oem_version =
        LWSWITCH_ELEMENT_READ(&bit_bridge_fw, oem_version, bridge_fw_size, 0);

    LWSWITCH_ELEMENT_VALIDATE(&bit_bridge_fw, firmware_size, bridge_fw_size, 0,
        firmware->firmware_size/512);

    if (LWSWITCH_ELEMENT_PRESENT(&bit_bridge_fw, BIOS_MOD_date, bridge_fw_size))
    {
        lwswitch_os_memcpy(firmware->bridge.BIOS_MOD_date, bit_bridge_fw.BIOS_MOD_date,
            sizeof(firmware->bridge.BIOS_MOD_date));
    }

    firmware->bridge.fw_release_build =
        (LWSWITCH_ELEMENT_PRESENT(&bit_bridge_fw, firmware_flags, bridge_fw_size) ?
            FLD_TEST_DRF(SWITCH_BIT_BRIDGE_FW_DATA, _FLAGS, _BUILD, _REL,
                bit_bridge_fw.firmware_flags) :
            LW_FALSE);

    copy_size = LW_MIN(LWSWITCH_PRODUCT_NAME_MAX_LEN,
        LWSWITCH_ELEMENT_READ(&bit_bridge_fw, eng_product_name_size, bridge_fw_size, 0));
    if (copy_size > 0)
    {
        retval = _lwswitch_read_rom_bytes(device, eeprom,
            bit_bridge_fw.eng_product_name,
            (LwU8 *) firmware->bridge.product_name, copy_size);
        if (retval != LWL_SUCCESS)
        {
            // Failed to read product name string
            copy_size = 0;
        }
    }
    firmware->bridge.product_name[copy_size] = 0;

    firmware->bridge.instance_id = LWSWITCH_ELEMENT_READ(
        &bit_bridge_fw,
        lwswitch_instance_id,
        bridge_fw_size,
        LWSWITCH_FIRMWARE_BRIDGE_INSTANCE_ID_UNKNOWN);

    return retval;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
//
// 0x43: BIT_TOKEN_CLOCK_PTRS
// https://wiki.lwpu.com/engwiki/index.php/VBIOS/Data_Structures/BIT#BIT_CLOCK_PTRS_.28Version_2.29
// https://wiki.lwpu.com/engwiki/index.php/Resman/PLLs/PLL_Information_Table_5.0_Specification
//
#endif
static LwlStatus
_lwswitch_rom_parse_bit_clock_ptrs
(
    lwswitch_device *device,
    LWSWITCH_EEPROM_TYPE *eeprom,
    LWSWITCH_FIRMWARE *firmware,
    LWSWITCH_BIT_TOKEN *bit_token
)
{
    LWSWITCH_BIT_CLOCK_PTRS bit_clock_ptrs = {0};
    LWSWITCH_PLL_INFO_HEADER pll_info_header;
    LWSWITCH_PLL_INFO_ENTRY pll_info;
    LwU32 pll_info_offset;
    LwU32 idx_pll;
    LwU32 clock_ptrs_size;
    LwU32 pll_info_table;
    LwlStatus retval;

    firmware->clocks.clocks_found = LW_FALSE;

    if (bit_token->data_size != sizeof(bit_clock_ptrs))
    {
        LWSWITCH_PRINT(device, SETUP,
            "CLOCK_PTRS: Expected data size 0x%x but found 0x%x\n",
            (LwU32) sizeof(bit_clock_ptrs), bit_token->data_size);
    }

    clock_ptrs_size = LW_MIN(bit_token->data_size, sizeof(bit_clock_ptrs));

     // Get PLL limits
    retval = _lwswitch_read_rom_bytes(device, eeprom, bit_token->data_offset,
        (LwU8 *) &bit_clock_ptrs, clock_ptrs_size);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, SETUP,
            "LWINIT_PTRS: Failed to read BIT_TOKEN_CLOCK_PTRS\n");
        return -LWL_ERR_NOT_SUPPORTED;
    }

    pll_info_table = LWSWITCH_ELEMENT_READ(&bit_clock_ptrs, pll_info_table, clock_ptrs_size, 0);

    if ((pll_info_table == 0) ||
        (pll_info_table + sizeof(pll_info_header) > firmware->firmware_size))
    {
        LWSWITCH_PRINT(device, SETUP,
            "LWINIT_PTRS: BIT_TOKEN_CLOCK_PTRS not preset or out of range (0x%x)\n",
            bit_clock_ptrs.pll_info_table);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    retval = _lwswitch_read_rom_bytes(device, eeprom, pll_info_table,
        (LwU8 *) &pll_info_header, sizeof(pll_info_header));
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, SETUP,
            "CLOCK_PTRS: Failed to read LWSWITCH_PLL_INFO_HEADER\n");
        return -LWL_ERR_NOT_SUPPORTED;
    }

    if (pll_info_header.version != LWSWITCH_CLOCK_PTRS_PLL_INFO_VERSION)
    {
        LWSWITCH_PRINT(device, SETUP,
            "PLL_INFO version (0x%x) != expected version (0x%x)\n",
            pll_info_header.version, LWSWITCH_CLOCK_PTRS_PLL_INFO_VERSION);
        return -LWL_ERR_NOT_SUPPORTED;
    }
    if (pll_info_header.header_size != sizeof(LWSWITCH_PLL_INFO_HEADER))
    {
        LWSWITCH_PRINT(device, SETUP,
            "PLL_INFO header size (0x%x) != expected (0x%x)\n",
            pll_info_header.header_size, (LwU32) sizeof(LWSWITCH_PLL_INFO_HEADER));
        return -LWL_ERR_NOT_SUPPORTED;
    }
    if (pll_info_header.entry_size != sizeof(LWSWITCH_PLL_INFO_ENTRY))
    {
        LWSWITCH_PRINT(device, SETUP,
            "PLL_INFO: Expected entry size 0x%x but found 0x%x\n",
            (LwU32) sizeof(LWSWITCH_PLL_INFO_ENTRY), pll_info_header.entry_size);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    firmware->clocks.clocks_found = LW_TRUE;
    firmware->clocks.sys_pll.valid = LW_FALSE;

    for (idx_pll = 0; idx_pll < pll_info_header.entry_count; idx_pll++)
    {
        pll_info_offset =
            bit_clock_ptrs.pll_info_table + pll_info_header.header_size +
            idx_pll*pll_info_header.entry_size;
        if (pll_info_offset + sizeof(pll_info) > firmware->firmware_size)
        {
            LWSWITCH_PRINT(device, SETUP,
                "PLL info #%d out of range (%x+%x > %x)\n", idx_pll,
                pll_info_offset, (LwU32) sizeof(pll_info), firmware->firmware_size);
            retval = -LWL_NOT_FOUND;
            break;
        }

        retval = _lwswitch_read_rom_bytes(device, eeprom, pll_info_offset,
            (LwU8 *) &pll_info, sizeof(pll_info));
        if (retval != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, SETUP,
                "CLOCK_PTRS: Failed to read LWSWITCH_PLL_INFO_ENTRY\n");
            retval = -LWL_ERR_NOT_SUPPORTED;
            break;
        }

        if (pll_info.pll_id == LWSWITCH_PLL_ID_SYSPLL)
        {
            if (firmware->clocks.sys_pll.valid)
            {
                LWSWITCH_PRINT(device, SETUP,
                    "LWINIT_PTRS: More than 1 SYSPLL entry found.  Skipping\n");
            }
            else
            {
                firmware->clocks.sys_pll.valid = LW_TRUE;
                firmware->clocks.sys_pll.ref_min_mhz = pll_info.ref_min_mhz;
                firmware->clocks.sys_pll.ref_max_mhz = pll_info.ref_max_mhz;
                firmware->clocks.sys_pll.vco_min_mhz = pll_info.vco_min_mhz;
                firmware->clocks.sys_pll.vco_max_mhz = pll_info.vco_max_mhz;
                firmware->clocks.sys_pll.update_min_mhz = pll_info.update_min_mhz;
                firmware->clocks.sys_pll.update_max_mhz = pll_info.update_max_mhz;
                firmware->clocks.sys_pll.m_min = pll_info.m_min;
                firmware->clocks.sys_pll.m_max = pll_info.m_max;
                firmware->clocks.sys_pll.n_min = pll_info.n_min;
                firmware->clocks.sys_pll.n_max = pll_info.n_max;
                firmware->clocks.sys_pll.pl_min = pll_info.pl_min;
                firmware->clocks.sys_pll.pl_max = pll_info.pl_max;
            }
        }
        else
        {
            LWSWITCH_PRINT(device, SETUP,
                "Ignoring PLL ID 0x%x\n", pll_info.pll_id);
        }
    }

    return retval;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
//
// 0x69: BIT_TOKEN_INTERNAL_USE_ONLY_PTRS
// https://wiki.lwpu.com/engwiki/index.php/VBIOS/Data_Structures/BIT#BIT_INTERNAL_USE_ONLY_.28Version_2.29
//
#endif
static LwlStatus
_lwswitch_rom_parse_bit_internal_ptrs
(
    lwswitch_device *device,
    LWSWITCH_EEPROM_TYPE *eeprom,
    LWSWITCH_FIRMWARE *firmware,
    LWSWITCH_BIT_TOKEN *bit_token
)
{
    LWSWITCH_BIT_INTERNAL_USE_ONLY_PTRS bit_internal = {0};
    LwlStatus retval;
    LwU32 internal_size;

    firmware->internal.internal_found = LW_FALSE;

    if (bit_token->data_size != sizeof(bit_internal))
    {
        LWSWITCH_PRINT(device, SETUP,
            "INTERNAL_USE_ONLY_PTRS: Expected data size 0x%x but found 0x%x\n",
            (LwU32) sizeof(bit_internal), bit_token->data_size);
    }

    internal_size = LW_MIN(bit_token->data_size, sizeof(bit_internal));

    // Get board ID, build, and flash info
    retval = _lwswitch_read_rom_bytes(device, eeprom, bit_token->data_offset,
        (LwU8 *) &bit_internal, internal_size);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, SETUP,
            "Failed to read INTERNAL_USE_ONLY_PTRS\n");
        return -LWL_ERR_NOT_SUPPORTED;
    }

    firmware->internal.internal_found = LW_TRUE;

    firmware->internal.int_firmware_version =
        LWSWITCH_ELEMENT_READ(&bit_internal, bios_version, internal_size, 0);
    firmware->internal.int_oem_version =
        LWSWITCH_ELEMENT_READ(&bit_internal, oem_version, internal_size, 0);
    firmware->internal.int_p4_cl =
        LWSWITCH_ELEMENT_READ(&bit_internal, perforce, internal_size, 0);
    firmware->internal.int_board_id =
        LWSWITCH_ELEMENT_READ(&bit_internal, board_id, internal_size, 0);
    firmware->internal.int_chip_sku_modifier =
        LWSWITCH_ELEMENT_READ(&bit_internal, chip_sku_modifier, internal_size, 0);
    firmware->internal.int_project_sku_modifier =
        LWSWITCH_ELEMENT_READ(&bit_internal, lw_project_sku_modifier, internal_size, 0);

    if (LWSWITCH_ELEMENT_PRESENT(&bit_internal, build_date, internal_size))
    {
        lwswitch_os_memcpy(firmware->internal.int_build_date, bit_internal.build_date,
            sizeof(firmware->internal.int_build_date));
    }

    if (LWSWITCH_ELEMENT_PRESENT(&bit_internal, chip_sku, internal_size))
    {
        lwswitch_os_memcpy(firmware->internal.int_chip_sku, bit_internal.chip_sku,
            sizeof(firmware->internal.int_chip_sku));
    }

    if (LWSWITCH_ELEMENT_PRESENT(&bit_internal, lw_project, internal_size))
    {
        lwswitch_os_memcpy(firmware->internal.int_project, &bit_internal.lw_project,
            sizeof(firmware->internal.int_project));
    }

    if (LWSWITCH_ELEMENT_PRESENT(&bit_internal, lw_project_sku, internal_size))
    {
        lwswitch_os_memcpy(firmware->internal.int_project_sku, &bit_internal.lw_project_sku,
            sizeof(firmware->internal.int_project_sku));
    }

    return retval;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
//
// 0x49: BIT_TOKEN_LWINIT_PTRS
// https://wiki.lwpu.com/engwiki/index.php/VBIOS/Data_Structures/BIT#BIT_LWINIT_PTRS
// https://wiki.lwpu.com/engwiki/index.php/VBIOS/Data_Structures/LWLink_Configuration_Data/Spec_V1
//
#endif
static LwlStatus
_lwswitch_rom_parse_bit_lwinit_ptrs
(
    lwswitch_device *device,
    LWSWITCH_EEPROM_TYPE *eeprom,
    LWSWITCH_FIRMWARE *firmware,
    LWSWITCH_BIT_TOKEN *bit_token
)
{
    LWSWITCH_BIT_LWINIT_PTRS bit_lwinit_ptrs = {0};
    LWSWITCH_LWLINK_CONFIG lwlink_config;
    LwU32 lwinit_ptrs_size;
    LwU32 lwlink_config_offset;
    LwU32 lwlink_config_size;
    LwlStatus retval;

    firmware->lwlink.link_config_found = LW_FALSE;

    if (bit_token->data_size != sizeof(bit_lwinit_ptrs))
    {
        LWSWITCH_PRINT(device, SETUP,
            "LWINIT_PTRS: Expected data size 0x%x but found 0x%x\n",
            (LwU32) sizeof(bit_lwinit_ptrs), bit_token->data_size);
    }

    lwinit_ptrs_size = LW_MIN(bit_token->data_size, sizeof(bit_lwinit_ptrs));

    // Get basic LWLink settings
    retval = _lwswitch_read_rom_bytes(device, eeprom, bit_token->data_offset,
        (LwU8 *) &bit_lwinit_ptrs, lwinit_ptrs_size);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, SETUP,
            "LWINIT_PTRS: Failed to read LWSWITCH_BIT_LWINIT_PTRS\n");
        return -LWL_ERR_NOT_SUPPORTED;
    }

    lwlink_config_offset = LWSWITCH_ELEMENT_READ(&bit_lwinit_ptrs, lwlink_config, lwinit_ptrs_size, 0);
    if ((lwlink_config_offset == 0) ||
        (lwlink_config_offset + sizeof(lwlink_config) > firmware->firmware_size))
    {
        LWSWITCH_PRINT(device, SETUP,
            "LWINIT_PTRS: LWSWITCH_BIT_LWINIT_PTRS LWLink config absent or out of range (0x%x)\n",
            bit_lwinit_ptrs.lwlink_config);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    retval = _lwswitch_read_rom_bytes(device, eeprom, lwlink_config_offset,
        (LwU8 *) &lwlink_config, sizeof(lwlink_config));
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, SETUP,
            "LWINIT_PTRS: Failed to read LWSWITCH_LWLINK_CONFIG\n");
        return -LWL_ERR_NOT_SUPPORTED;
    }

    lwlink_config_size = LW_MIN(lwlink_config.size, sizeof(lwlink_config));

    if (0x01 != LWSWITCH_ELEMENT_READ(&lwlink_config, version, lwlink_config_size, 0))
    {
        LWSWITCH_PRINT(device, SETUP,
            "LWINIT_PTRS: LWLINK_CONFIG version mismatch (0x01 != 0x%x)\n",
            LWSWITCH_ELEMENT_READ(&lwlink_config, version, lwlink_config_size, 0));
        return -LWL_ERR_NOT_SUPPORTED;
    }

    LWSWITCH_ELEMENT_CHECK(&lwlink_config, flags, lwlink_config_size, 0x0);
    LWSWITCH_ELEMENT_CHECK(&lwlink_config, link_speed_mask, lwlink_config_size, 0x0);
    LWSWITCH_ELEMENT_CHECK(&lwlink_config, link_refclk_mask, lwlink_config_size, 0x0);

    firmware->lwlink.link_config_found = LW_TRUE;

    //
    // If lwlink_config is incomplete, assume:
    //  1) all links enabled
    //  2) DC coupled
    //
    firmware->lwlink.link_enable_mask = ~LWSWITCH_ELEMENT_READ(&lwlink_config, link_disable_mask, lwlink_config_size, 0);
    firmware->lwlink.link_ac_coupled_mask = LWSWITCH_ELEMENT_READ(&lwlink_config, ac_coupled_mask, lwlink_config_size, 0);

    return retval;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
//
// 0x6E: BIT_TOKEN_DCB_PTRS
// https://wiki.lwpu.com/engwiki/index.php/Resman/Display_and_DCB_Dolwmentation/DCB_4.x_Specification
// https://wiki.lwpu.com/engwiki/index.php/Resman/Display_and_DCB_Dolwmentation/DCB_4.x_Specification/DCB_4.2
//
#endif
static void
_lwswitch_rom_parse_bit_dcb_ccb_block
(
    lwswitch_device *device,
    LWSWITCH_EEPROM_TYPE *eeprom,
    LWSWITCH_FIRMWARE *firmware,
    LwU32 ccb_block_offset
)
{
    LWSWITCH_CCB_TABLE ccb;
    LWSWITCH_CCB_ENTRY ccb_entry;
    LwU32 ccb_table_offset;
    LwU32 idx_ccb;
    LwU32 retval;

    // dcb:ccb_block_ptr
    if ((ccb_block_offset == 0) ||
        (ccb_block_offset + sizeof(LWSWITCH_CCB_TABLE) > firmware->firmware_size))
    {
        LWSWITCH_PRINT(device, SETUP,
            "DCB_PTRS: CCB_BLOCK absent or out of range (0x%x)\n",
            ccb_block_offset);
        return;
    }

    retval = _lwswitch_read_rom_bytes(device, eeprom, ccb_block_offset,
        (LwU8 *) &ccb, sizeof(ccb));
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, SETUP, "DCB_PTRS: CCB header read failure\n");
        return;
    }

    if ((ccb.version != LWSWITCH_CCB_VERSION) ||
        (ccb.header_size != sizeof(ccb)))
    {
        LWSWITCH_PRINT(device, SETUP,
            "DCB_PTRS: CCB_BLOCK version (0x%x) or size mismatch (0x%x)\n",
            ccb.version, ccb.header_size);
        return;
    }

    ccb_table_offset = ccb_block_offset + ccb.header_size;

    for (idx_ccb = 0; idx_ccb < ccb.entry_count; idx_ccb++)
    {
        LwU32 ccb_entry_offset = ccb_table_offset + idx_ccb*ccb.entry_size;
        LwU32 i2c_bus_idx;
        LwU32 idx_i2c_port;

        if (ccb_entry_offset + sizeof(ccb_entry) > firmware->firmware_size)
        {
            LWSWITCH_PRINT(device, SETUP,
                "DCB_PTRS: CCB out of range\n");
            break;
        }

        retval = _lwswitch_read_rom_bytes(device, eeprom, ccb_entry_offset,
            (LwU8 *) &ccb_entry, sizeof(ccb_entry));
        if (retval != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, SETUP,
                "DCB_PTRS: CCB entry[%d] read failure\n",
                idx_ccb);
            break;
        }

        i2c_bus_idx = DRF_VAL(SWITCH_CCB, _DEVICE, _I2C_PORT,  ccb_entry.device);
        if (i2c_bus_idx >= LWSWITCH_MAX_I2C_PORTS)
        {
            continue;
        }

        for (idx_i2c_port = 0; idx_i2c_port < LWSWITCH_MAX_I2C_PORTS; idx_i2c_port++)
        {
            if (ccb.comm_port[idx_i2c_port] == i2c_bus_idx)
            {
                break;
            }
        }

        if (idx_i2c_port >= LWSWITCH_MAX_I2C_PORTS)
        {
            LWSWITCH_PRINT(device, SETUP,
                "DCB_PTRS: CCB entry[%d] I2C port %x out of range\n",
                idx_ccb, idx_i2c_port);
            continue;
        }

        firmware->dcb.i2c[idx_i2c_port].valid = LW_TRUE;
        firmware->dcb.i2c[idx_i2c_port].i2c_speed = DRF_VAL(SWITCH_CCB, _DEVICE, _I2C_SPEED, ccb_entry.device);
        firmware->dcb.i2c[idx_i2c_port].i2c_33v = DRF_VAL(SWITCH_CCB, _DEVICE, _VOLTAGE, ccb_entry.device);
    }
}

static void
_lwswitch_rom_parse_bit_dcb_gpio_table
(
    lwswitch_device *device,
    LWSWITCH_EEPROM_TYPE *eeprom,
    LWSWITCH_FIRMWARE *firmware,
    LwU32 gpio_table_offset
)
{
    LWSWITCH_GPIO_TABLE gpio;
    LWSWITCH_GPIO_ENTRY gpio_entry;
    LwU32 idx_gpio;
    LwU32 retval;

    // gpio_tables
    if ((gpio_table_offset == 0) ||
        (gpio_table_offset + sizeof(LWSWITCH_GPIO_TABLE) > firmware->firmware_size))
    {
        LWSWITCH_PRINT(device, SETUP,
            "DCB_PTRS: GPIO_TABLE absent or out of range (0x%x)\n",
            gpio_table_offset);
        return;
    }

    retval = _lwswitch_read_rom_bytes(device, eeprom, gpio_table_offset,
        (LwU8 *) &gpio, sizeof(gpio));
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, SETUP,
            "DCB_PTRS: GPIO table read failure\n");
        return;
    }

    if ((gpio.version != LWSWITCH_GPIO_TABLE_VERSION_42) ||
        (gpio.header_size != sizeof(gpio)))
    {
        LWSWITCH_PRINT(device, SETUP,
            "DCB_PTRS: GPIO_TABLE version (0x%x) or size mismatch (0x%x)\n",
            gpio.version, gpio.header_size);
        return;
    }

    if (gpio.entry_size != sizeof(gpio_entry))
    {
        LWSWITCH_PRINT(device, SETUP,
            "DCB_PTRS: GPIO_ENTRY size mismatch (0x%x != 0x%x)\n",
            gpio.entry_size, (LwU32) sizeof(gpio_entry));
        return;
    }

    LWSWITCH_ELEMENT_CHECK(&gpio, ext_gpio_master, gpio.header_size, 0x0000);

    gpio_table_offset += gpio.header_size;
    firmware->dcb.gpio_pin_count = 0;

    for (idx_gpio = 0; idx_gpio < gpio.entry_count; idx_gpio++)
    {
        LWSWITCH_GPIO_INFO *gpio_pin;
        LwU32 gpio_entry_offset = gpio_table_offset + idx_gpio*gpio.entry_size;

        if (gpio_entry_offset + gpio.entry_size > firmware->firmware_size)
        {
            LWSWITCH_PRINT(device, SETUP,
                "DCB_PTRS: GPIO entry[%d] out of range\n",
                idx_gpio);
            break;
        }

        if (firmware->dcb.gpio_pin_count == LWSWITCH_MAX_GPIO_PINS)
        {
            LWSWITCH_PRINT(device, SETUP,
                "DCB_PTRS: Too many GPIO pins listed\n");
            break;
        }

        retval = _lwswitch_read_rom_bytes(device, eeprom, gpio_entry_offset,
            (LwU8 *) &gpio_entry, sizeof(gpio_entry));
        if (retval != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, SETUP,
                "DCB_PTRS: GPIO entry read failure\n");
            break;
        }

        if (!FLD_TEST_DRF(SWITCH_GPIO_ENTRY, , _FUNCTION, _SKIP_ENTRY, gpio_entry.function))
        {
            gpio_pin = &firmware->dcb.gpio_pin[firmware->dcb.gpio_pin_count];
            firmware->dcb.gpio_pin_count++;

            gpio_pin->pin = DRF_VAL(SWITCH_GPIO_ENTRY, _PIN, _NUM, gpio_entry.pin);
            gpio_pin->function = DRF_VAL(SWITCH_GPIO_ENTRY, , _FUNCTION, gpio_entry.function);
            gpio_pin->hw_select = DRF_VAL(SWITCH_GPIO_ENTRY, _INPUT, _HW_SELECT, gpio_entry.input);
            gpio_pin->misc = DRF_VAL(SWITCH_GPIO_ENTRY, _MISC, _IO, gpio_entry.misc);
        }
    }
}

static void
_lwswitch_rom_parse_bit_dcb_i2c_devices
(
    lwswitch_device *device,
    LWSWITCH_EEPROM_TYPE *eeprom,
    LWSWITCH_FIRMWARE *firmware,
    LwU32 i2c_devices_offset
)
{
    LWSWITCH_I2C_TABLE i2c;
    LWSWITCH_I2C_ENTRY i2c_entry;
    LwU32 i2c_table_offset;
    LwU32 idx_i2c;
    LwU32 retval;

    // i2c_devices
    if ((i2c_devices_offset == 0) ||
        (i2c_devices_offset + sizeof(LWSWITCH_I2C_TABLE) > firmware->firmware_size))
    {
        LWSWITCH_PRINT(device, SETUP,
            "DCB_PTRS: I2C_DEVICES absent or out of range (0x%x)\n",
            i2c_devices_offset);
        return;
    }

    retval = _lwswitch_read_rom_bytes(device, eeprom, i2c_devices_offset,
        (LwU8 *) &i2c, sizeof(i2c));
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, SETUP,
            "DCB_PTRS: I2C device read failure\n");
        return;
    }

    if ((i2c.version != LWSWITCH_I2C_VERSION) ||
        (i2c.header_size != sizeof(i2c)) ||
        (i2c.entry_size != sizeof(i2c_entry)))
    {
        LWSWITCH_PRINT(device, SETUP,
            "DCB_PTRS: I2C header version (0x%x) or header/entry size mismatch (0x%x/0x%x)\n",
            i2c.version, i2c.header_size, i2c.entry_size);
        return;
    }

    i2c_table_offset = i2c_devices_offset + i2c.header_size;

    firmware->dcb.i2c_device_count = 0;

    for (idx_i2c = 0; idx_i2c < i2c.entry_count; idx_i2c++)
    {
        LwU32 i2c_entry_offset = i2c_table_offset + idx_i2c*i2c.entry_size;
        LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE *i2c_device;

        if (i2c_entry_offset + sizeof(i2c_entry) > firmware->firmware_size)
        {
            LWSWITCH_PRINT(device, SETUP,
                "DCB_PTRS: I2C[%d] out of range\n",
                idx_i2c);
            break;
        }

        if (firmware->dcb.i2c_device_count >= LWSWITCH_MAX_I2C_DEVICES)
        {
            LWSWITCH_PRINT(device, SETUP,
                "DCB_PTRS: Too many I2C devices listed\n");
            break;
        }

        retval = _lwswitch_read_rom_bytes(device, eeprom, i2c_entry_offset,
            (LwU8 *) &i2c_entry, sizeof(i2c_entry));
        if (retval != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, SETUP,
                "DCB_PTRS: I2C read failure\n");
            break;
        }

        if (LWSWITCH_I2C_DEVICE_SKIP != DRF_VAL(SWITCH_I2C, _ENTRY, _TYPE, i2c_entry.device))
        {
            i2c_device = &firmware->dcb.i2c_device[firmware->dcb.i2c_device_count];
            firmware->dcb.i2c_device_count++;

            i2c_device->i2cDeviceType = DRF_VAL(SWITCH_I2C, _ENTRY, _TYPE, i2c_entry.device);
            i2c_device->i2cAddress = DRF_VAL(SWITCH_I2C, _ENTRY, _ADDRESS, i2c_entry.device);
            i2c_device->i2cPortLogical =
                (DRF_VAL(SWITCH_I2C, _ENTRY, _PORT_2, i2c_entry.device) << 1) |
                DRF_VAL(SWITCH_I2C, _ENTRY, _PORT_1, i2c_entry.device);
        }
    }
}

static LwlStatus
_lwswitch_rom_parse_bit_dcb_ptrs
(
    lwswitch_device *device,
    LWSWITCH_EEPROM_TYPE *eeprom,
    LWSWITCH_FIRMWARE *firmware,
    LWSWITCH_BIT_TOKEN *bit_token
)
{
    LWSWITCH_BIT_DCB_PTRS dcb_ptrs;
    LWSWITCH_DCB_HEADER dcb;
    LwU32 dcb_ptrs_size;
    LwU32 dcb_version;
    LwU32 dcb_signature;
    LwlStatus retval = LWL_SUCCESS;

    firmware->dcb.dcb_found = LW_FALSE;

    if (bit_token->data_size != sizeof(dcb_ptrs))
    {
        LWSWITCH_PRINT(device, SETUP,
            "DCB_PTRS: Expected data size 0x%x but found 0x%x\n",
            (LwU32) sizeof(dcb_ptrs), bit_token->data_size);
    }

    dcb_ptrs_size = LW_MIN(bit_token->data_size, sizeof(dcb_ptrs));

    // Get I2C & GPIO tables
    retval = _lwswitch_read_rom_bytes(device, eeprom, bit_token->data_offset, 
        (LwU8 *) &dcb_ptrs, dcb_ptrs_size);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, SETUP,
            "DCB_PTRS: Failed to read LWSWITCH_BIT_DCB_PTRS\n");
        return retval;
    }

    if ((dcb_ptrs.dcb_header_ptr == 0) ||
        (dcb_ptrs.dcb_header_ptr >= firmware->firmware_size))
    {
        LWSWITCH_PRINT(device, SETUP,
            "DCB_PTRS: DCB header absent or out of range (0x%x)\n",
            dcb_ptrs.dcb_header_ptr);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    retval = _lwswitch_read_rom_bytes(device, eeprom, dcb_ptrs.dcb_header_ptr,
        (LwU8 *) &dcb, sizeof(dcb));
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, SETUP,
            "DCB_PTRS: DCB header read failure\n");
        return retval;
    }

    dcb_version = LWSWITCH_ELEMENT_READ(&dcb, version, dcb.header_size, 0x0);
    dcb_signature = LWSWITCH_ELEMENT_READ(&dcb, dcb_signature, dcb.header_size, 0x0);
    if ((dcb_version != LWSWITCH_DCB_HEADER_VERSION_41) ||
        (dcb_signature != LWSWITCH_DCB_HEADER_SIGNATURE))
    {
        LWSWITCH_PRINT(device, SETUP,
            "DCB_PTRS: DCB header version (0x%x) or signature mismatch (0x%x)\n",
            dcb_version, dcb_signature);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    _lwswitch_rom_parse_bit_dcb_ccb_block(device, eeprom, firmware, dcb.ccb_block_ptr);
    _lwswitch_rom_parse_bit_dcb_i2c_devices(device, eeprom, firmware, dcb.i2c_devices);
    _lwswitch_rom_parse_bit_dcb_gpio_table(device, eeprom, firmware, dcb.gpio_table);

    return retval;
}

//
// Parse BIT tokens, if present
//
static LwlStatus
_lwswitch_read_bit_tokens
(
    lwswitch_device *device,
    LWSWITCH_EEPROM_TYPE *eeprom,
    LWSWITCH_FIRMWARE *firmware,
    LWSWITCH_BIT_HEADER *bit_header,
    LwU32 *offset
)
{
    LwU32 idx_token;
    LwU32 bit_entry_offset;
    LWSWITCH_BIT_TOKEN bit_token;
    LwlStatus retval = LWL_SUCCESS;

    for (idx_token = 0; idx_token < bit_header->token_entries; idx_token++)
    {
        bit_entry_offset = *offset + idx_token*bit_header->token_size;
        if (bit_entry_offset >= firmware->firmware_size)
        {
            LWSWITCH_PRINT(device, SETUP,
                "BIT token out of range (%x >= %x)\n",
                bit_entry_offset, firmware->firmware_size);
            return -LWL_NOT_FOUND;
        }

        retval = _lwswitch_read_rom_bytes(device,
            eeprom, bit_entry_offset,
            (LwU8 *) &bit_token, sizeof(bit_token));
        if (retval != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, SETUP,
                "Error reading BIT token[%d]\n", idx_token);
            return -LWL_NOT_FOUND;
        }

        if (bit_token.data_offset >= firmware->firmware_size)
        {
            LWSWITCH_PRINT(device, SETUP,
                "BIT 0x%x target data out of range (%x >= %x)\n",
                bit_token.id,
                bit_token.data_offset, firmware->firmware_size);
            // Soldier on to the next one.  Hopefully it's valid
            continue;
        }

        switch (bit_token.id)
        {
            case LWSWITCH_BIT_TOKEN_CLOCK_PTRS:
            {
                retval = _lwswitch_rom_parse_bit_clock_ptrs(device, eeprom, firmware, &bit_token);
                break;
            }
            case LWSWITCH_BIT_TOKEN_LWINIT_PTRS:
            {
                retval = _lwswitch_rom_parse_bit_lwinit_ptrs(device, eeprom, firmware, &bit_token);
                break;
            }
            case LWSWITCH_BIT_TOKEN_NOP:
            {
                // Ignore
                break;
            }
            case LWSWITCH_BIT_TOKEN_PERF_PTRS:
            {
                LWSWITCH_PRINT(device, INFO, "Skipping parsing BIT_TOKEN_PERF_PTRS\n");
                break;
            }
            case LWSWITCH_BIT_TOKEN_BRIDGE_FW_DATA:
            {
                retval = _lwswitch_rom_parse_bit_bridge_fw_data(device, eeprom, firmware, &bit_token);
                break;
            }
            case LWSWITCH_BIT_TOKEN_INTERNAL_USE_ONLY_PTRS:
            {
                retval = _lwswitch_rom_parse_bit_internal_ptrs(device, eeprom, firmware, &bit_token);
                break;
            }
            case LWSWITCH_BIT_TOKEN_DCB_PTRS:
            {
                retval = _lwswitch_rom_parse_bit_dcb_ptrs(device, eeprom, firmware, &bit_token);
                break;
            }
            default:
            {
                LWSWITCH_PRINT(device, SETUP,
                    "Unrecognized BIT_TOKEN 0x%02x\n", bit_token.id);
                break;
            }
        }
    }

    return retval;
}

//
// Parse BIT table, if present
//
static LwlStatus
_lwswitch_read_bit_table
(
    lwswitch_device *device,
    LWSWITCH_EEPROM_TYPE *eeprom,
    LWSWITCH_FIRMWARE *firmware,
    LwU32 *offset
)
{
    LWSWITCH_BIT_HEADER  bit_header = {0};
    LwlStatus retval;

    retval = _lwswitch_read_rom_bytes(device,
        eeprom, *offset,
        (LwU8 *) &bit_header, sizeof(bit_header));
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, SETUP,
            "Unable to read BIT header @%04x\n",
            *offset);
        return retval;
    }

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    //
    // BIT table definitions:
    // https://wiki.lwpu.com/engwiki/index.php/VBIOS/Data_Structures/BIT
    //
#endif
    if ((bit_header.id           == 0xB8FF) &&
        (bit_header.signature[0] == 'B') &&
        (bit_header.signature[1] == 'I') &&
        (bit_header.signature[2] == 'T') &&
        (bit_header.signature[3] == 0x00) &&
        (bit_header.bcd_version  == 0x0100) &&
        (_lwswitch_callwlate_checksum((LwU8 *) &bit_header, sizeof(bit_header)) == 0x00))
    {
        *offset += bit_header.header_size;
        if (*offset >= firmware->firmware_size)
        {
            LWSWITCH_PRINT(device, SETUP,
                "BIT token table out of range (%x >= %x)\n",
                *offset, firmware->firmware_size);
            return -LWL_NOT_FOUND;
        }
    }
    else
    {
        LWSWITCH_PRINT(device, SETUP,
            "BIT header not found @%04x\n",
            *offset);
        return -LWL_NOT_FOUND;
    }

    retval = _lwswitch_read_bit_tokens(device, eeprom, firmware, &bit_header, offset);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, SETUP,
            "Unable to read BIT tokens\n");
        return retval;
    }

    return LWL_SUCCESS;
}

//
// Print BIT table information
//
static void
_lwswitch_print_bit_table_info
(
    lwswitch_device *device,
    LWSWITCH_FIRMWARE *firmware
)
{
    if (firmware->firmware_size > 0)
    {
        LWSWITCH_PRINT(device, SETUP, "PCI ID:           %04x/%04x\n",
            firmware->pci_vendor_id,
            firmware->pci_device_id);
        LWSWITCH_PRINT(device, SETUP, "Subsystem PCI ID: %04x/%04x\n",
            firmware->pci_system_vendor_id,
            firmware->pci_system_device_id);

        if (firmware->bridge.bridge_fw_found)
        {
            LWSWITCH_PRINT(device, SETUP, "firmware_version: %08x\n",
                firmware->bridge.firmware_version);
            LWSWITCH_PRINT(device, SETUP, "oem_version:      %02x\n",
                firmware->bridge.oem_version);
            LWSWITCH_PRINT(device, SETUP, "BIOS_MOD_date:    '%.8s'\n",
                firmware->bridge.BIOS_MOD_date);
            LWSWITCH_PRINT(device, SETUP, "fw_release_build: %s\n",
                (firmware->bridge.fw_release_build ? "REL" : "ENG"));
            LWSWITCH_PRINT(device, SETUP, "product_name:     '%s'\n",
                firmware->bridge.product_name);
            if (firmware->bridge.instance_id != LWSWITCH_FIRMWARE_BRIDGE_INSTANCE_ID_UNKNOWN)
            {
                LWSWITCH_PRINT(device, SETUP, "instance_id:      %04x\n",
                    firmware->bridge.instance_id);
            }
        }

        if (firmware->internal.internal_found)
        {
            LWSWITCH_PRINT(device, SETUP, "int firmware_version: %08x\n", firmware->internal.int_firmware_version);
            LWSWITCH_PRINT(device, SETUP, "int oem_version:      %02x\n", firmware->internal.int_oem_version);
            LWSWITCH_PRINT(device, SETUP, "P4 CL:                %d\n",   firmware->internal.int_p4_cl);
            LWSWITCH_PRINT(device, SETUP, "int board ID:         %04x\n", firmware->internal.int_board_id);
            LWSWITCH_PRINT(device, SETUP, "int build date:       '%.8s'\n", firmware->internal.int_build_date);

            if (firmware->internal.int_chip_sku[0] != 0)
            {
                LWSWITCH_PRINT(device, SETUP, "chip SKU:             '%.3s%c'\n",
                    firmware->internal.int_chip_sku,
                    firmware->internal.int_chip_sku_modifier);
            }
            if (firmware->internal.int_project[0] != 0)
            {
                LWSWITCH_PRINT(device, SETUP, "project:              '%.4s'\n",
                    firmware->internal.int_project);
            }
            if (firmware->internal.int_project_sku[0] != 0)
            {
                LWSWITCH_PRINT(device, SETUP, "project SKU:          '%.4s%c'\n",
                    firmware->internal.int_project_sku,
                    firmware->internal.int_project_sku_modifier);
            }
        }

        if (firmware->lwlink.link_config_found)
        {
            LWSWITCH_PRINT(device, SETUP, "link_enable: %016llx\n", firmware->lwlink.link_enable_mask);
            LWSWITCH_PRINT(device, SETUP, "ac_coupled:  %016llx\n", firmware->lwlink.link_ac_coupled_mask);
        }
    }
}

//
// Parse EEPROM BIT tables, if present
//
void
lwswitch_read_rom_tables
(
    lwswitch_device *device,
    LWSWITCH_FIRMWARE *firmware
)
{
    LWSWITCH_EEPROM_TYPE eeprom = {0};
    LwU32 offset;
    LwlStatus retval;

    retval = lwswitch_get_rom_info(device, &eeprom);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, SETUP,
            "ROM configuration not supported\n");
        return;
    }

    retval = _lwswitch_read_rom_header(device, &eeprom, firmware, &offset);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, SETUP,
            "Unable to read ROM header\n");
        return;
    }

    retval = _lwswitch_read_bit_table(device, &eeprom, firmware, &offset);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, SETUP,
            "Unable to read BIT table\n");
        return;
    }

    _lwswitch_print_bit_table_info(device, firmware);

    return;
}

