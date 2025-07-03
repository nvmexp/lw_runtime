/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "common_lwswitch.h"
#include "error_lwswitch.h"
#include "regkey_lwswitch.h"
#include "haldef_lwswitch.h"
#include "lr10/lr10.h"
#include "lr10/clock_lr10.h"
#include "lr10/bus_lr10.h"
#include "lr10/minion_lr10.h"
#include "lr10/soe_lr10.h"
#include "lr10/fuse_lr10.h"
#include "lr10/pmgr_lr10.h"
#include "lr10/therm_lr10.h"
#include "lr10/inforom_lr10.h"
#include "lr10/smbpbi_lr10.h"
#include "flcn/flcnable_lwswitch.h"
#include "soe/soe_lwswitch.h"
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#include "lr10/jtag_lr10.h"
#include "boards_lwswitch.h"
#include "cci/cci_lwswitch.h"
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#include "lwswitch/lr10/dev_lws_top.h"
#include "lwswitch/lr10/dev_pri_ringmaster.h"
#include "lwswitch/lr10/dev_pri_ringstation_sys.h"
#include "lwswitch/lr10/dev_lwlsaw_ip.h"
#include "lwswitch/lr10/dev_lwlsaw_ip_addendum.h"
#include "lwswitch/lr10/dev_lws_master.h"
#include "lwswitch/lr10/dev_fuse.h"
#include "lwswitch/lr10/dev_lwltlc_ip.h"
#include "lwswitch/lr10/dev_lwldl_ip.h"
#include "lwswitch/lr10/dev_lwlipt_lnk_ip.h"
#include "lwswitch/lr10/dev_lwlctrl_ip.h"
#include "lwswitch/lr10/dev_npg_ip.h"
#include "lwswitch/lr10/dev_npgperf_ip.h"
#include "lwswitch/lr10/dev_nport_ip.h"
#include "lwswitch/lr10/dev_ingress_ip.h"
#include "lwswitch/lr10/dev_tstate_ip.h"
#include "lwswitch/lr10/dev_egress_ip.h"
#include "lwswitch/lr10/dev_route_ip.h"
#include "lwswitch/lr10/dev_therm.h"
#include "lwswitch/lr10/dev_soe_ip.h"
#include "lwswitch/lr10/dev_route_ip_addendum.h"
#include "lwswitch/lr10/dev_minion_ip.h"
#include "lwswitch/lr10/dev_minion_ip_addendum.h"
#include "lwswitch/lr10/dev_nport_ip_addendum.h"
#include "lwswitch/lr10/dev_nxbar_tile_ip.h"
#include "lwswitch/lr10/dev_nxbar_tc_global_ip.h"
#include "lwswitch/lr10/dev_sourcetrack_ip.h"

#include "oob/smbpbi.h"

#define DMA_ADDR_WIDTH_LR10     64
#define ROUTE_GANG_TABLE_SIZE (1 << DRF_SIZE(LW_ROUTE_REG_TABLE_ADDRESS_INDEX))

static void
_lwswitch_deassert_link_resets_lr10
(
    lwswitch_device *device
)
{
    LwU32 val, i;
    LWSWITCH_TIMEOUT timeout;
    LwBool           keepPolling;

    LWSWITCH_PRINT(device, WARN,
        "%s: LWSwitch Driver is taking the links out of reset. This should only happen during forced config.\n",
        __FUNCTION__);

    for (i = 0; i < LWSWITCH_LINK_COUNT(device); i++)
    {
        if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLIPT_LNK, i)) continue;

        val = LWSWITCH_LINK_RD32_LR10(device, i,
                LWLIPT_LNK, _LWLIPT_LNK, _RESET_RSTSEQ_LINK_RESET);
        val = FLD_SET_DRF_NUM(_LWLIPT_LNK, _RESET_RSTSEQ_LINK_RESET, _LINK_RESET,
                          LW_LWLIPT_LNK_RESET_RSTSEQ_LINK_RESET_LINK_RESET_DEASSERT, val);

        LWSWITCH_LINK_WR32_LR10(device, i,
                LWLIPT_LNK, _LWLIPT_LNK, _RESET_RSTSEQ_LINK_RESET, val);
    }

    for (i = 0; i < LWSWITCH_LINK_COUNT(device); i++)
    {
        if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLIPT_LNK, i)) continue;

        // Poll for _RESET_STATUS == _DEASSERTED
        lwswitch_timeout_create(25*LWSWITCH_INTERVAL_1MSEC_IN_NS, &timeout);

        do
        {
            keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

            val = LWSWITCH_LINK_RD32_LR10(device, i,
                    LWLIPT_LNK, _LWLIPT_LNK, _RESET_RSTSEQ_LINK_RESET);
            if (FLD_TEST_DRF(_LWLIPT_LNK, _RESET_RSTSEQ_LINK_RESET,
                        _LINK_RESET_STATUS, _DEASSERTED, val))
            {
                break;
            }

            lwswitch_os_sleep(1);
        }
        while (keepPolling);

        if (!FLD_TEST_DRF(_LWLIPT_LNK, _RESET_RSTSEQ_LINK_RESET,
                    _LINK_RESET_STATUS, _DEASSERTED, val))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Timeout waiting for link %d_LINK_RESET_STATUS == _DEASSERTED\n",
                __FUNCTION__, i);
                // Bug 2974064: Review this timeout handling (fall through)
        }
    }
}

static void
_lwswitch_train_forced_config_link_lr10
(
    lwswitch_device *device,
    LwU32            linkId
)
{
    LwU32 data, i;
    lwlink_link *link;

    link = lwswitch_get_link(device, linkId);

    if ((link == NULL) ||
        !LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLDL, link->linkNumber) ||
        (linkId >= LWSWITCH_LWLINK_MAX_LINKS))
    {
        return;
    }

    data = LWSWITCH_LINK_RD32_LR10(device, linkId, LWLDL, _LWLDL_TOP, _LINK_TEST);
    data = FLD_SET_DRF(_LWLDL_TOP, _LINK_TEST, _AUTO_HWCFG, _ENABLE, data);
    LWSWITCH_LINK_WR32_LR10(device, linkId, LWLDL, _LWLDL_TOP, _LINK_TEST, data);

    // Add some delay to let the sim/emu go to SAFE
    LWSWITCH_NSEC_DELAY(400 * LWSWITCH_INTERVAL_1USEC_IN_NS);

    data = LWSWITCH_LINK_RD32_LR10(device, linkId, LWLDL, _LWLDL_TOP, _LINK_TEST);
    data = FLD_SET_DRF(_LWLDL_TOP, _LINK_TEST, _AUTO_LWHS, _ENABLE, data);
    LWSWITCH_LINK_WR32_LR10(device, linkId, LWLDL, _LWLDL_TOP, _LINK_TEST, data);

    // Add some delay to let the sim/emu go to HS
    LWSWITCH_NSEC_DELAY(400 * LWSWITCH_INTERVAL_1USEC_IN_NS);

    data = LWSWITCH_LINK_RD32_LR10(device, linkId, LWLDL, _LWLDL_TOP, _LINK_CHANGE);
    data = FLD_SET_DRF(_LWLDL_TOP, _LINK_CHANGE, _NEWSTATE,      _ACTIVE, data);
    data = FLD_SET_DRF(_LWLDL_TOP, _LINK_CHANGE, _OLDSTATE_MASK, _DONTCARE, data);
    data = FLD_SET_DRF(_LWLDL_TOP, _LINK_CHANGE, _ACTION,        _LTSSM_CHANGE, data);
    LWSWITCH_LINK_WR32_LR10(device, linkId, LWLDL, _LWLDL_TOP, _LINK_CHANGE, data);

    i = 0;

    // Poll until LINK_CHANGE[1:0] != 2b01.
    while (i < 5)
    {
        data = LWSWITCH_LINK_RD32_LR10(device, linkId, LWLDL, _LWLDL_TOP, _LINK_CHANGE);

        if (FLD_TEST_DRF(_LWLDL_TOP, _LINK_CHANGE, _STATUS, _BUSY, data))
        {
            LWSWITCH_PRINT(device, INFO,
                "%s : Waiting for link %d to go to ACTIVE\n",
                __FUNCTION__, linkId);
        }
        else if (FLD_TEST_DRF(_LWLDL_TOP, _LINK_CHANGE, _STATUS, _FAULT, data))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s : Fault while changing LINK to ACTIVE. Link = %d\n",
                __FUNCTION__, linkId);
            break;
        }
        else
        {
            break;
        }

        LWSWITCH_NSEC_DELAY(5 * LWSWITCH_INTERVAL_1USEC_IN_NS);
        i++;
    }

    data = LWSWITCH_LINK_RD32_LR10(device, linkId, LWLDL, _LWLDL_TOP, _LINK_STATE);

    if (FLD_TEST_DRF(_LWLDL_TOP, _LINK_STATE, _STATE, _ACTIVE, data))
    {
        LWSWITCH_PRINT(device, INFO,
            "%s : Link %d is in ACTIVE state, setting BUFFER_READY\n",
            __FUNCTION__, linkId);

        // Set buffer ready only for lwlink TLC and not NPORT
        lwswitch_init_buffer_ready(device, link, LW_FALSE);
    }
    else
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s : Timeout while waiting for link %d to go to ACTIVE\n",
            __FUNCTION__, linkId);
        LWSWITCH_PRINT(device, ERROR,
            "%s : Link %d is in 0x%x state\n",
            __FUNCTION__, linkId,DRF_VAL(_LWLDL_TOP, _LINK_STATE, _STATE, data));
    }

}

void
_lwswitch_setup_chiplib_forced_config_lr10
(
    lwswitch_device *device
)
{
    LwU64 links = ((LwU64)device->regkeys.chiplib_forced_config_link_mask) +
                  ((LwU64)device->regkeys.chiplib_forced_config_link_mask2 << 32);
    LwU32 i;

    if (links == 0)
    {
        return;
    }

    //
    // First, take the links out of reset
    //
    // NOTE: On LR10, MINION will take the links out of reset during INITPHASE1
    // On platforms where MINION is not present and/or we want to run with forced
    // config, the driver must de-assert the link reset
    //
    _lwswitch_deassert_link_resets_lr10(device);

    // Next, train the links to ACTIVE/LWHS
    FOR_EACH_INDEX_IN_MASK(64, i, links)
    {
        if (device->link[i].valid)
        {
            _lwswitch_train_forced_config_link_lr10(device, i);
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;
}

static LwU32
_lwswitch_get_lwlink_linerate
(
    lwswitch_device *device,
    LwU32            val
)
{
    LwU32  lineRate = 0;
    switch (val)
    {
        case LW_SWITCH_REGKEY_SPEED_CONTROL_SPEED_16G:
            lineRate = LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_CLK_CTRL_LINE_RATE_16_00000_GBPS;
            break;
        case LW_SWITCH_REGKEY_SPEED_CONTROL_SPEED_20G:
            lineRate = LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_CLK_CTRL_LINE_RATE_20_00000_GBPS;
            break;
        case LW_SWITCH_REGKEY_SPEED_CONTROL_SPEED_25G:
            lineRate = LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_CLK_CTRL_LINE_RATE_25_00000_GBPS;
            break;
        case LW_SWITCH_REGKEY_SPEED_CONTROL_SPEED_32G:
            lineRate = LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_CLK_CTRL_LINE_RATE_32_00000_GBPS;
            break;
        case LW_SWITCH_REGKEY_SPEED_CONTROL_SPEED_40G:
            lineRate = LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_CLK_CTRL_LINE_RATE_40_00000_GBPS;
            break;
        case LW_SWITCH_REGKEY_SPEED_CONTROL_SPEED_50G:
            lineRate = LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_CLK_CTRL_LINE_RATE_50_00000_GBPS;
            break;
        case LW_SWITCH_REGKEY_SPEED_CONTROL_SPEED_53_12500G:
            lineRate = LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_CLK_CTRL_LINE_RATE_53_12500_GBPS;
            break;
        default:
            LWSWITCH_PRINT(device, SETUP, "%s:ERROR LINE_RATE = 0x%x requested by regkey\n",
                       __FUNCTION__, lineRate);
            lineRate = LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_CLK_CTRL_LINE_RATE_ILLEGAL_LINE_RATE;
    }
    return lineRate;
}

static void
_lwswitch_setup_link_system_registers_lr10
(
    lwswitch_device *device,
    LWSWITCH_BIOS_LWLINK_CONFIG *bios_config,
    lwlink_link *link
)
{
    LwU32 regval, fldval;
    LwU32 lineRate = 0;
    LWLINK_CONFIG_DATA_LINKENTRY *vbios_link_entry = NULL;

    //
    // Identify the valid link entry to update. If not, proceed with the default settings
    //
    if ((bios_config == NULL) || (bios_config->bit_address == 0))
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s: No override with VBIOS - VBIOS LwLink configuration table not found\n",
            __FUNCTION__);
    }
    else
    {
        vbios_link_entry = &bios_config->link_vbios_entry[bios_config->link_base_entry_assigned][link->linkNumber];
    }

    // LINE_RATE SYSTEM register
    if (device->regkeys.lwlink_speed_control != LW_SWITCH_REGKEY_SPEED_CONTROL_SPEED_DEFAULT)
    {
        regval   = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLIPT_LNK,
                                           _LWLIPT_LNK_CTRL_SYSTEM_LINK, _CLK_CTRL);
        lineRate = _lwswitch_get_lwlink_linerate(device, device->regkeys.lwlink_speed_control);
        regval   = FLD_SET_DRF_NUM(_LWLIPT_LNK_CTRL_SYSTEM_LINK, _CLK_CTRL,
                                    _LINE_RATE, lineRate, regval);
        LWSWITCH_PRINT(device, SETUP, "%s: LINE_RATE = 0x%x requested by regkey\n",
                       __FUNCTION__, lineRate);
        LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLIPT_LNK,
                            _LWLIPT_LNK_CTRL_SYSTEM_LINK, _CLK_CTRL, regval);
    }

    // TXTRAIN SYSTEM register
    regval = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLIPT_LNK,
                                     _LWLIPT_LNK_CTRL_SYSTEM_LINK, _CHANNEL_CTRL);

    fldval = DRF_VAL(_SWITCH_REGKEY, _TXTRAIN_CONTROL, _FOM_FORMAT,
                     device->regkeys.txtrain_control);
    if (fldval != LW_SWITCH_REGKEY_TXTRAIN_CONTROL_FOM_FORMAT_NOP)
    {
        LWSWITCH_PRINT(device, SETUP, "%s: FOM_FORMAT = 0x%x requested by regkey\n",
                       __FUNCTION__, fldval);
        regval = FLD_SET_DRF_NUM(_LWLIPT_LNK_CTRL_SYSTEM_LINK, _CHANNEL_CTRL,
                                 _TXTRAIN_FOM_FORMAT, fldval, regval);
    }
    else if (vbios_link_entry != NULL)
    {
        regval = FLD_SET_DRF_NUM(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL, _TXTRAIN_FOM_FORMAT,
                                    DRF_VAL(_LWLINK_VBIOS,_PARAM5,_TXTRAIN_FOM_FORMAT, vbios_link_entry->lwLinkparam5),
                                    regval);
    }

    fldval = DRF_VAL(_SWITCH_REGKEY, _TXTRAIN_CONTROL, _OPTIMIZATION_ALGORITHM,
                     device->regkeys.txtrain_control);
    if (fldval != LW_SWITCH_REGKEY_TXTRAIN_CONTROL_OPTIMIZATION_ALGORITHM_NOP)
    {
        LWSWITCH_PRINT(device, SETUP, "%s: OPTIMIZATION_ALGORITHM = 0x%x requested by regkey\n",
                       __FUNCTION__, fldval);
        regval = FLD_SET_DRF_NUM(_LWLIPT_LNK_CTRL_SYSTEM_LINK, _CHANNEL_CTRL,
                                 _TXTRAIN_OPTIMIZATION_ALGORITHM, fldval, regval);
    }
    else if (vbios_link_entry != NULL)
    {
        regval = FLD_SET_DRF_NUM(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL, _TXTRAIN_OPTIMIZATION_ALGORITHM,
                                 vbios_link_entry->lwLinkparam4, regval);
    }

    fldval = DRF_VAL(_SWITCH_REGKEY, _TXTRAIN_CONTROL, _ADJUSTMENT_ALGORITHM,
                     device->regkeys.txtrain_control);
    if (fldval != LW_SWITCH_REGKEY_TXTRAIN_CONTROL_ADJUSTMENT_ALGORITHM_NOP)
    {
        LWSWITCH_PRINT(device, SETUP, "%s: ADJUSTMENT_ALGORITHM = 0x%x requested by regkey\n",
                       __FUNCTION__, fldval);
        regval = FLD_SET_DRF_NUM(_LWLIPT_LNK_CTRL_SYSTEM_LINK, _CHANNEL_CTRL,
                                 _TXTRAIN_ADJUSTMENT_ALGORITHM, fldval, regval);
    }
    else if (vbios_link_entry != NULL)
    {
        regval = FLD_SET_DRF_NUM(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL, _TXTRAIN_ADJUSTMENT_ALGORITHM,
                                     DRF_VAL(_LWLINK_VBIOS,_PARAM5,_TXTRAIN_ADJUSTMENT_ALGORITHM, vbios_link_entry->lwLinkparam5),
                                     regval);
    }

    fldval = DRF_VAL(_SWITCH_REGKEY, _TXTRAIN_CONTROL, _MINIMUM_TRAIN_TIME_MANTISSA,
                     device->regkeys.txtrain_control);
    if (fldval != LW_SWITCH_REGKEY_TXTRAIN_CONTROL_MINIMUM_TRAIN_TIME_MANTISSA_NOP)
    {
        LWSWITCH_PRINT(device, SETUP, "%s: MINIMUM_TRAIN_TIME_MANTISSA = 0x%x requested by regkey\n",
                       __FUNCTION__, fldval);
        regval = FLD_SET_DRF_NUM(_LWLIPT_LNK_CTRL_SYSTEM_LINK, _CHANNEL_CTRL,
                                 _TXTRAIN_MINIMUM_TRAIN_TIME_MANTISSA, fldval, regval);
    }
    else if (vbios_link_entry != NULL)
    {
        regval = FLD_SET_DRF_NUM(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL, _TXTRAIN_MINIMUM_TRAIN_TIME_MANTISSA,
                                 DRF_VAL(_LWLINK_VBIOS,_PARAM6,_TXTRAIN_MINIMUM_TRAIN_TIME_MANTISSA, vbios_link_entry->lwLinkparam6),
                                 regval);
    }
    else
    {
        //
        // Default working configuration for LR10
        // This will be provided by VBIOS once support available (bug 2767390)
        //
        LWSWITCH_PRINT(device, SETUP, "%s: MINIMUM_TRAIN_TIME_MANTISSA = 0x5 forced by driver\n",
                       __FUNCTION__);
        regval = FLD_SET_DRF_NUM(_LWLIPT_LNK_CTRL_SYSTEM_LINK, _CHANNEL_CTRL,
                                 _TXTRAIN_MINIMUM_TRAIN_TIME_MANTISSA, 0x5, regval);
    }

    fldval = DRF_VAL(_SWITCH_REGKEY, _TXTRAIN_CONTROL, _MINIMUM_TRAIN_TIME_EXPONENT,
                     device->regkeys.txtrain_control);
    if (fldval != LW_SWITCH_REGKEY_TXTRAIN_CONTROL_MINIMUM_TRAIN_TIME_EXPONENT_NOP)
    {
        LWSWITCH_PRINT(device, SETUP, "%s: MINIMUM_TRAIN_TIME_EXPONENT = 0x%x requested by regkey\n",
                       __FUNCTION__, fldval);
        regval = FLD_SET_DRF_NUM(_LWLIPT_LNK_CTRL_SYSTEM_LINK, _CHANNEL_CTRL,
                                 _TXTRAIN_MINIMUM_TRAIN_TIME_EXPONENT, fldval, regval);
    }
    else if (vbios_link_entry != NULL)
    {
        regval = FLD_SET_DRF_NUM(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL, _TXTRAIN_MINIMUM_TRAIN_TIME_EXPONENT,
                                 DRF_VAL(_LWLINK_VBIOS,_PARAM6,_TXTRAIN_MINIMUM_TRAIN_TIME_EXPONENT, vbios_link_entry->lwLinkparam6),
                                 regval);
    }
    else
    {
        //
        // Default working configuration for LR10
        // This will be provided by VBIOS once support available  (bug 2767390)
        //
        LWSWITCH_PRINT(device, SETUP, "%s: MINIMUM_TRAIN_TIME_EXPONENT = 0x4 forced by driver\n",
                       __FUNCTION__);
        regval = FLD_SET_DRF_NUM(_LWLIPT_LNK_CTRL_SYSTEM_LINK, _CHANNEL_CTRL,
                                 _TXTRAIN_MINIMUM_TRAIN_TIME_EXPONENT, 0x4, regval);
    }

    LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLIPT_LNK,
                            _LWLIPT_LNK_CTRL_SYSTEM_LINK, _CHANNEL_CTRL, regval);

    // Disable L2 (Bug 3176196)
    regval = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLIPT_LNK, _LWLIPT_LNK, _CTRL_SYSTEM_LINK_AN1_CTRL);
    regval = FLD_SET_DRF(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_AN1_CTRL, _PWRM_L2_ENABLE, _DISABLE, regval);
    LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLIPT_LNK, _LWLIPT_LNK, _CTRL_SYSTEM_LINK_AN1_CTRL, regval);

    // SW WAR: Bug 3364420
    lwswitch_apply_recal_settings(device, link);
}

/*!
 * @brief Parse packed little endian data and unpack into padded structure
 *
 * @param[in]   format          Data format
 * @param[in]   packedData      Packed little endian data
 * @param[out]  unpackedData    Unpacked padded structure
 * @param[out]  unpackedSize    Unpacked data size
 * @param[out]  fieldsCount     Number of fields
 *
 * @return 'LW_OK'
 */
LW_STATUS
_lwswitch_devinit_unpack_structure
(
    const char *format,
    const LwU8 *packedData,
    LwU32      *unpackedData,
    LwU32      *unpackedSize,
    LwU32      *fieldsCount
)
{
    LwU32 unpkdSize = 0;
    LwU32 fields = 0;
    LwU32 count;
    LwU32 data;
    char fmt;

    while ((fmt = *format++))
    {
        count = 0;
        while ((fmt >= '0') && (fmt <= '9'))
        {
            count *= 10;
            count += fmt - '0';
            fmt = *format++;
        }
        if (count == 0)
            count = 1;

        while (count--)
        {
            switch (fmt)
            {
                case 'b':
                    data = *packedData++;
                    unpkdSize += 1;
                    break;

                case 's':    // signed byte
                    data = *packedData++;
                    if (data & 0x80)
                        data |= ~0xff;
                    unpkdSize += 1;
                    break;

                case 'w':
                    data  = *packedData++;
                    data |= *packedData++ << 8;
                    unpkdSize += 2;
                    break;

                case 'd':
                    data  = *packedData++;
                    data |= *packedData++ << 8;
                    data |= *packedData++ << 16;
                    data |= *packedData++ << 24;
                    unpkdSize += 4;
                    break;

                default:
                    return LW_ERR_GENERIC;
            }
            *unpackedData++ = data;
            fields++;
        }
    }

    if (unpackedSize != NULL)
        *unpackedSize = unpkdSize;

    if (fieldsCount != NULL)
        *fieldsCount = fields;

    return LW_OK;
}

/*!
 * @brief Callwlate packed and unpacked data size based on given data format
 *
 * @param[in]   format          Data format
 * @param[out]  packedSize      Packed data size
 * @param[out]  unpackedSize    Unpacked data size
 *
 */
void
_lwswitch_devinit_callwlate_sizes
(
    const char *format,
    LwU32      *packedSize,
    LwU32      *unpackedSize
)
{
    LwU32 unpkdSize = 0;
    LwU32 pkdSize = 0;
    LwU32 count;
    char fmt;

    while ((fmt = *format++))
    {
        count = 0;
        while ((fmt >= '0') && (fmt <= '9'))
        {
            count *= 10;
            count += fmt - '0';
            fmt = *format++;
        }
        if (count == 0)
            count = 1;

        switch (fmt)
        {
            case 'b':
                pkdSize += count * 1;
                unpkdSize += count * sizeof(bios_U008);
                break;

            case 's':    // signed byte
                pkdSize += count * 1;
                unpkdSize += count * sizeof(bios_S008);
                break;

            case 'w':
                pkdSize += count * 2;
                unpkdSize += count * sizeof(bios_U016);
                break;

            case 'd':
                pkdSize += count * 4;
                unpkdSize += count * sizeof(bios_U032);
                break;
        }
    }

    if (packedSize != NULL)
        *packedSize = pkdSize;

    if (unpackedSize != NULL)
        *unpackedSize = unpkdSize;
}

/*!
 * @brief Callwlate packed and unpacked data size based on given data format
 *
 * @param[in]   format          Data format
 * @param[out]  packedSize      Packed data size
 * @param[out]  unpackedSize    Unpacked data size
 *
 */

LW_STATUS
_lwswitch_vbios_read_structure
(
    lwswitch_device *device,
    void            *structure,
    LwU32           offset,
    LwU32           *ppacked_size,
    const char      *format
)
{
    LwU32  packed_size;
    LwU8  *packed_data;
    LwU32  unpacked_bytes;

    // callwlate the size of the data as indicated by its packed format.
    _lwswitch_devinit_callwlate_sizes(format, &packed_size, &unpacked_bytes);

    if (ppacked_size)
        *ppacked_size = packed_size;

    //
    // is 'offset' too big?
    // happens when we read bad ptrs from fixed addrs in image frequently
    //
    if ((offset + packed_size) > device->biosImage.size)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Bad offset in bios read: 0x%x, max is 0x%x, fmt is '%s'\n",
                       __FUNCTION__, offset, device->biosImage.size, format);
        return LW_ERR_GENERIC;
    }

    packed_data = &device->biosImage.pImage[offset];
    return _lwswitch_devinit_unpack_structure(format, packed_data, structure,
                                              &unpacked_bytes, NULL);
}

LwU8
_lwswitch_vbios_read8
(
    lwswitch_device *device,
    LwU32           offset
)
{
    bios_U008 data;     // BiosReadStructure expects 'bios' types

    _lwswitch_vbios_read_structure(device, &data, offset, (LwU32 *) 0, "b");

    return (LwU8) data;
}

LwU16
_lwswitch_vbios_read16
(
    lwswitch_device *device,
    LwU32           offset
)
{
    bios_U016 data;     // BiosReadStructure expects 'bios' types

    _lwswitch_vbios_read_structure(device, &data, offset, (LwU32 *) 0, "w");

    return (LwU16) data;
}


LwU32
_lwswitch_vbios_read32
(
    lwswitch_device *device,
    LwU32           offset
)
{
    bios_U032 data;     // BiosReadStructure expects 'bios' types

    _lwswitch_vbios_read_structure(device, &data, offset, (LwU32 *) 0, "d");

    return (LwU32) data;
}

LW_STATUS
_lwswitch_verify_BIT_Version
(
    lwswitch_device *device,
    LWSWITCH_BIOS_LWLINK_CONFIG *bios_config
)
{
    BIT_HEADER_V1_00         bitHeader;
    BIT_TOKEN_V1_00          bitToken;
    BIT_DATA_BIOSDATA_V1     biosDataV1;
    BIT_DATA_BIOSDATA_V2     biosDataV2;
    BIT_DATA_INTERNAL_USE_V1 intUseTable;
    LW_STATUS                rmStatus;
    LwU32                    dataPointerOffset;
    LwU32 i, bflag, iflag, bBiosDataV2;

    // XXX can I checksum this?
    rmStatus = _lwswitch_vbios_read_structure(device,
                                              (LwU8*) &bitHeader,
                                              bios_config->bit_address,
                                              (LwU32 *) 0,
                                              BIT_HEADER_V1_00_FMT);

    if(rmStatus != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR,
                       "%s: Failed to read BIT table structure!.\n",
                       __FUNCTION__);
        return rmStatus;
    }

    bflag = iflag = 0;
    bBiosDataV2 = 0;
    //
    // parse through the bit tokens quickly and isolate the biosdata and intdata tokens
    // once we have both, compare the versions and either return success or failure
    //
    for(i=0; i < bitHeader.TokenEntries; i++)
    {
        LwU32 BitTokenLocation = bios_config->bit_address + bitHeader.HeaderSize + (i * bitHeader.TokenSize);
        rmStatus = _lwswitch_vbios_read_structure(device,
                                                 (LwU8*) &bitToken,
                                                 BitTokenLocation,
                                                 (LwU32 *) 0,
                                                 BIT_TOKEN_V1_00_FMT);
        if(rmStatus != LW_OK)
        {
            LWSWITCH_PRINT(device, WARN,
                "%s: Failed to read BIT token %d!\n",
                __FUNCTION__, i);
            return LW_ERR_GENERIC;
        }

        dataPointerOffset = (bios_config->pci_image_address + bitToken.DataPtr);
        switch(bitToken.TokenId)
        {
            case BIT_TOKEN_BIOSDATA:
            {
                switch(bitToken.DataVersion)
                {
                    case 2:
                        rmStatus = _lwswitch_vbios_read_structure(device,
                                                                  (LwU8*) &biosDataV2,
                                                                  dataPointerOffset,
                                                                  (LwU32 *) 0,
                                                                  BIT_DATA_BIOSDATA_V2_FMT);
                        if (rmStatus != LW_OK)
                        {
                            LWSWITCH_PRINT(device, WARN,
                                "%s: Failed to read bios data format 2 structure\n",
                                __FUNCTION__);
                            return LW_ERR_GENERIC;
                        }
                        bflag       = 1;
                        bBiosDataV2 = 1;
                        break;
                    case 1:
                    case 0:
                        rmStatus = _lwswitch_vbios_read_structure(device,
                                                                  (LwU8*) &biosDataV1,
                                                                  dataPointerOffset,
                                                                  (LwU32 *) 0,
                                                                  BIT_DATA_BIOSDATA_V1_FMT);
                        if (rmStatus != LW_OK)
                        {
                            LWSWITCH_PRINT(device, WARN,
                                "%s: Failed to read bios data format 1 structure\n",
                                __FUNCTION__);
                            return LW_ERR_GENERIC;
                        }
                        bflag = 1;
                }
                break;
            }
            break;

            case BIT_TOKEN_LWINIT_PTRS:
            {
                BIT_DATA_LWINIT_PTRS_V1 lwInitTablePtrs;
                rmStatus = _lwswitch_vbios_read_structure(device,
                                                          (LwU8*) &lwInitTablePtrs,
                                                          dataPointerOffset,
                                                          (LwU32 *) 0,
                                                          BIT_DATA_LWINIT_PTRS_V1_30_FMT);
                if (rmStatus != LW_OK)
                {
                    LWSWITCH_PRINT(device, WARN,
                                   "%s: Failed to read internal data structure\n",
                                   __FUNCTION__);
                    return LW_ERR_GENERIC;
                }
                // Update the retrived info with device info
                bios_config->lwlink_config_table_address = (lwInitTablePtrs.LwlinkConfigDataPtr + bios_config->pci_image_address);
            }
            break;

            case BIT_TOKEN_INTERNAL_USE:
            {
                if(bitToken.DataVersion <= 2)
                {
                    rmStatus = _lwswitch_vbios_read_structure(device,
                                                              (LwU8*) &intUseTable,
                                                              dataPointerOffset,
                                                              (LwU32 *) 0,
                                                              BIT_DATA_INTERNAL_USE_V1_FMT);
                    if (rmStatus != LW_OK)
                    {
                        LWSWITCH_PRINT(device, WARN,
                                       "%s: Failed to read internal data structure\n",
                                       __FUNCTION__);
                        return LW_ERR_GENERIC;
                    }
                    iflag = 1;
                }
            }
        }
    }

    if (bflag && iflag)
    {
        if (bBiosDataV2)
        {
            if ((biosDataV2.Version & 0xffff0000) == (intUseTable.Version & 0xffff0000))
            {
                return LW_OK;
            }
        }
        else
        {
            if ((biosDataV1.Version & 0xffff0000) == (intUseTable.Version & 0xffff0000))
            {
                return LW_OK;
            }
        }
    }
    LWSWITCH_PRINT(device, ERROR,
                   "%s: Failed BIT version check\n",
                   __FUNCTION__);
    return LW_ERR_GENERIC;
}

LW_STATUS
_lwswitch_validate_BIT_header
(
    lwswitch_device *device,
    LwU32            bit_address
)
{
    LwU32    headerSize = 0;
    LwU32    chkSum = 0;
    LwU32    i;

    //
    // For now let's assume the Header Size is always at the same place.
    // We can create something more complex if needed later.
    //
    headerSize = (LwU32)_lwswitch_vbios_read8(device, bit_address + BIT_HEADER_SIZE_OFFSET);

    // Now perform checksum
    for (i = 0; i < headerSize; i++)
        chkSum += (LwU32)_lwswitch_vbios_read8(device, bit_address + i);

    //Byte checksum removes upper bytes
    chkSum = chkSum & 0xFF;

    if (chkSum)
        return LW_ERR_GENERIC;

    return LW_OK;
}


LW_STATUS
lwswitch_verify_header
(
    lwswitch_device *device,
    LWSWITCH_BIOS_LWLINK_CONFIG *bios_config
)
{
    LwU32       i;
    LW_STATUS   status = LW_ERR_GENERIC;

    if ((bios_config == NULL) || (!bios_config->pci_image_address))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: PCI Image offset is not identified\n",
            __FUNCTION__);
        return status;
    }

    // attempt to find the init info in the BIOS
    for (i = bios_config->pci_image_address; i < device->biosImage.size - 3; i++)
    {
        LwU16 bitheaderID = _lwswitch_vbios_read16(device, i);
        if (bitheaderID == BIT_HEADER_ID)
        {
            LwU32 signature = _lwswitch_vbios_read32(device, i + 2);
            if (signature == BIT_HEADER_SIGNATURE)
            {
                bios_config->bit_address = i;

                // Checksum BIT to prove accuracy
                if (LW_OK != _lwswitch_validate_BIT_header(device, bios_config->bit_address))
                {
                    device->biosImage.pImage = 0;
                    device->biosImage.size = 0;
                }
            }
        }
        // only if we find the bit address do we break
        if (bios_config->bit_address)
            break;
    }
    if (bios_config->bit_address)
    {
        status = LW_OK;
    }

    return status;
}

LW_STATUS
_lwswitch_vbios_update_bit_Offset
(
    lwswitch_device *device,
    LWSWITCH_BIOS_LWLINK_CONFIG *bios_config
)
{
    LW_STATUS   status = LW_OK;

    if (bios_config->bit_address)
    {
        goto vbios_update_bit_Offset_done;
    }

    status = lwswitch_verify_header(device, bios_config);
    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: *** BIT header is not found in vbios!\n",
            __FUNCTION__);
        goto vbios_update_bit_Offset_done;
    }

    if (bios_config->bit_address)
    {
        status = _lwswitch_verify_BIT_Version(device, bios_config);
        if (status != LW_OK)
            goto vbios_update_bit_Offset_done;
    }

vbios_update_bit_Offset_done:
    return status;
}


LW_STATUS
_lwswitch_vbios_identify_pci_image_loc
(
    lwswitch_device         *device,
    LWSWITCH_BIOS_LWLINK_CONFIG *bios_config
)
{
    LW_STATUS   status = LW_OK;
    LwU32       i;

    if (bios_config->pci_image_address)
    {
        goto vbios_identify_pci_image_loc_done;
    }

    // Match the PCI_EXP_ROM_SIGNATURE and followed by the PCI Data structure
    // with PCIR and matching vendor ID
    LWSWITCH_PRINT(device, SETUP,
        "%s: Verifying and extracting PCI Data.\n",
        __FUNCTION__);

    // attempt to find the init info in the BIOS
    for (i = 0; i < (device->biosImage.size - PCI_ROM_HEADER_PCI_DATA_SIZE); i++)
    {
        LwU16 pci_rom_sigature = _lwswitch_vbios_read16(device, i);

        if (pci_rom_sigature == PCI_EXP_ROM_SIGNATURE)
        {
            LwU32 pcir_data_dffSet  = _lwswitch_vbios_read16(device, i + PCI_ROM_HEADER_SIZE);  // 0x16 -> 0x18 i.e, including the ROM Signature bytes

            if (((i + pcir_data_dffSet) + PCI_DATA_STRUCT_SIZE) < device->biosImage.size)
            {
                LwU32 pcirSigature = _lwswitch_vbios_read32(device, (i + pcir_data_dffSet));

                if (pcirSigature == PCI_DATA_STRUCT_SIGNATURE)
                {
                    PCI_DATA_STRUCT pciData;
                    status = _lwswitch_vbios_read_structure(device,
                                                           (LwU8*) &pciData,
                                                            i + pcir_data_dffSet,
                                                            (LwU32 *) 0,
                                                            PCI_DATA_STRUCT_FMT);
                    if (status != LW_OK)
                    {
                        LWSWITCH_PRINT(device, WARN,
                                       "%s: Failed to PCI Data for validation\n",
                                       __FUNCTION__);
                        goto vbios_identify_pci_image_loc_done;
                    }

                    // Validate the vendor details as well
                    if (pciData.vendorID == PCI_VENDOR_ID_LWIDIA)
                    {
                        bios_config->pci_image_address = i;
                        break;
                    }
                }
            }
        }
    }

vbios_identify_pci_image_loc_done:
    return status;
}

LwU32 _lwswitch_get_lwlink_config_address
(
    lwswitch_device         *device,
    LWSWITCH_BIOS_LWLINK_CONFIG *bios_config
)
{
    return bios_config->lwlink_config_table_address;
}

LW_STATUS
_lwswitch_read_vbios_link_base_entry
(
    lwswitch_device *device,
    LwU32            tblPtr,
    LWLINK_CONFIG_DATA_BASEENTRY  *link_base_entry
)
{
    LW_STATUS status = LW_ERR_ILWALID_PARAMETER;
    LWLINK_VBIOS_CONFIG_DATA_BASEENTRY vbios_link_base_entry;

    status = _lwswitch_vbios_read_structure(device, &vbios_link_base_entry, tblPtr, (LwU32 *)0, LWLINK_CONFIG_DATA_BASEENTRY_FMT);
    if (status != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Error on reading lwlink base entry\n",
            __FUNCTION__);
        return status;
    }

    link_base_entry->positionId = vbios_link_base_entry.positionId;

    return status;
}

LW_STATUS
_lwswitch_read_vbios_link_entries
(
    lwswitch_device *device,
    LwU32            tblPtr,
    LwU32            expected_link_entriesCount,
    LWLINK_CONFIG_DATA_LINKENTRY  *link_entries,
    LwU32            *identified_link_entriesCount
)
{
    LW_STATUS status = LW_ERR_ILWALID_PARAMETER;
    LwU32 i;
    LWLINK_VBIOS_CONFIG_DATA_LINKENTRY vbios_link_entry;
    *identified_link_entriesCount = 0;

    for (i = 0; i < expected_link_entriesCount; i++)
    {
        status = _lwswitch_vbios_read_structure(device,
                                                &vbios_link_entry,
                                                tblPtr, (LwU32 *)0,
                                                LWLINK_CONFIG_DATA_LINKENTRY_FMT);
        if (status != LW_OK)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Error on reading lwlink entry\n",
                __FUNCTION__);
            return status;
        }
        link_entries[i].lwLinkparam0 = (LwU8)vbios_link_entry.lwLinkparam0;
        link_entries[i].lwLinkparam1 = (LwU8)vbios_link_entry.lwLinkparam1;
        link_entries[i].lwLinkparam2 = (LwU8)vbios_link_entry.lwLinkparam2;
        link_entries[i].lwLinkparam3 = (LwU8)vbios_link_entry.lwLinkparam3;
        link_entries[i].lwLinkparam4 = (LwU8)vbios_link_entry.lwLinkparam4;
        link_entries[i].lwLinkparam5 = (LwU8)vbios_link_entry.lwLinkparam5;
        link_entries[i].lwLinkparam6 = (LwU8)vbios_link_entry.lwLinkparam6;
        tblPtr += sizeof(LWLINK_CONFIG_DATA_LINKENTRY);

        LWSWITCH_PRINT(device, SETUP,
            "<<<---- LwLink ID 0x%x ---->>>\n", i);
        LWSWITCH_PRINT(device, SETUP,
            "LWLink Params 0 \t0x%x \tBinary:"BYTE_TO_BINARY_PATTERN"\n", vbios_link_entry.lwLinkparam0, BYTE_TO_BINARY(vbios_link_entry.lwLinkparam0));
        LWSWITCH_PRINT(device, SETUP,
            "LWLink Params 1 \t0x%x \tBinary:"BYTE_TO_BINARY_PATTERN"\n", vbios_link_entry.lwLinkparam1, BYTE_TO_BINARY(vbios_link_entry.lwLinkparam1));
        LWSWITCH_PRINT(device, SETUP,
            "LWLink Params 2 \t0x%x \tBinary:"BYTE_TO_BINARY_PATTERN"\n", vbios_link_entry.lwLinkparam2, BYTE_TO_BINARY(vbios_link_entry.lwLinkparam2));
        LWSWITCH_PRINT(device, SETUP,
            "LWLink Params 3 \t0x%x \tBinary:"BYTE_TO_BINARY_PATTERN"\n", vbios_link_entry.lwLinkparam3, BYTE_TO_BINARY(vbios_link_entry.lwLinkparam3));
        LWSWITCH_PRINT(device, SETUP,
            "LWLink Params 4 \t0x%x \tBinary:"BYTE_TO_BINARY_PATTERN"\n", vbios_link_entry.lwLinkparam4, BYTE_TO_BINARY(vbios_link_entry.lwLinkparam4));
        LWSWITCH_PRINT(device, SETUP,
            "LWLink Params 5 \t0x%x \tBinary:"BYTE_TO_BINARY_PATTERN"\n", vbios_link_entry.lwLinkparam5, BYTE_TO_BINARY(vbios_link_entry.lwLinkparam5));
        LWSWITCH_PRINT(device, SETUP,
            "LWLink Params 6 \t0x%x \tBinary:"BYTE_TO_BINARY_PATTERN"\n", vbios_link_entry.lwLinkparam6, BYTE_TO_BINARY(vbios_link_entry.lwLinkparam6));
        LWSWITCH_PRINT(device, SETUP,
            "<<<---- LwLink ID 0x%x ---->>>\n\n", i);
    }
    *identified_link_entriesCount = i;
    return status;
}

LW_STATUS
_lwswitch_vbios_fetch_lwlink_entries
(
    lwswitch_device         *device,
    LWSWITCH_BIOS_LWLINK_CONFIG    *bios_config
)
{
    LwU32                       tblPtr;
    LwU8                        version;
    LwU8                        size;
    LW_STATUS                   status = LW_ERR_GENERIC;
    LWLINK_CONFIG_DATA_HEADER   header;
    LwU32                       base_entry_index;
    LwU32                       expected_base_entry_count;

    tblPtr = _lwswitch_get_lwlink_config_address(device, bios_config);
    if (!tblPtr)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: No LwLink Config table set\n",
            __FUNCTION__);
        goto vbios_fetch_lwlink_entries_done;
    }

    // Read the table version number
    version = _lwswitch_vbios_read8(device, tblPtr);
    switch (version)
    {
        case LWLINK_CONFIG_DATA_HEADER_VER_20:
            size = _lwswitch_vbios_read8(device, tblPtr + 1);
            if (size == LWLINK_CONFIG_DATA_HEADER_20_SIZE)
            {
                // Grab Lwlink Config Data Header
                status = _lwswitch_vbios_read_structure(device, &header.ver_20, tblPtr, (LwU32 *) 0, LWLINK_CONFIG_DATA_HEADER_20_FMT);

                if (status != LW_OK)
                {
                    LWSWITCH_PRINT(device, ERROR,
                        "%s: Error on reading the lwlink config header\n",
                        __FUNCTION__);
                }
            }
            break;
        default:
            LWSWITCH_PRINT(device, ERROR,
                "%s: Invalid version 0x%x\n",
                __FUNCTION__, version);
    }
    if (status != LW_OK)
    {
        goto vbios_fetch_lwlink_entries_done;
    }

    LWSWITCH_PRINT(device, SETUP,
        "<<<---- LwLink Header ---->>>\n\n");
    LWSWITCH_PRINT(device, SETUP,
        "Version \t\t 0x%x\n", header.ver_20.Version);
    LWSWITCH_PRINT(device, SETUP,
        "Header Size \t0x%x\n", header.ver_20.HeaderSize);
    LWSWITCH_PRINT(device, SETUP,
        "Base Entry Size \t0x%x\n", header.ver_20.BaseEntrySize);
    LWSWITCH_PRINT(device, SETUP,
        "Base Entry count \t0x%x\n", header.ver_20.BaseEntryCount);
    LWSWITCH_PRINT(device, SETUP,
        "Link Entry Size \t0x%x\n", header.ver_20.LinkEntrySize);
    LWSWITCH_PRINT(device, SETUP,
        "Link Entry Count \t0x%x\n", header.ver_20.LinkEntryCount);
    LWSWITCH_PRINT(device, SETUP,
        "Reserved \t0x%x\n", header.ver_20.Reserved);
    LWSWITCH_PRINT(device, SETUP,
        "<<<---- LwLink Header ---->>>\n");

    expected_base_entry_count = header.ver_20.BaseEntryCount;
    if (expected_base_entry_count > LWSWITCH_NUM_BIOS_LWLINK_CONFIG_BASE_ENTRY)
    {
        LWSWITCH_PRINT(device, WARN,
            "%s: Greater than expected base entry count 0x%x - Restricting to count 0x%x\n",
            __FUNCTION__, expected_base_entry_count, LWSWITCH_NUM_BIOS_LWLINK_CONFIG_BASE_ENTRY);
        expected_base_entry_count = LWSWITCH_NUM_BIOS_LWLINK_CONFIG_BASE_ENTRY;
    }

    tblPtr += header.ver_20.HeaderSize;
    for (base_entry_index = 0; base_entry_index < expected_base_entry_count; base_entry_index++)
    {
        LwU32 expected_link_entriesCount = header.ver_20.LinkEntryCount;
        if (expected_link_entriesCount > LWSWITCH_LINK_COUNT(device))
        {
            LWSWITCH_PRINT(device, WARN,
                "%s: Greater than expected link count 0x%x - Restricting to count 0x%x\n",
                __FUNCTION__, expected_link_entriesCount, LWSWITCH_LINK_COUNT(device));
            expected_link_entriesCount = LWSWITCH_LINK_COUNT(device);
        }

        // Grab Lwlink Config Data Base Entry
        _lwswitch_read_vbios_link_base_entry(device, tblPtr, &bios_config->link_vbios_base_entry[base_entry_index]);
        tblPtr += header.ver_20.BaseEntrySize;
        
        _lwswitch_read_vbios_link_entries(device,
                                          tblPtr,
                                          expected_link_entriesCount,
                                          bios_config->link_vbios_entry[base_entry_index],
                                          &bios_config->identified_Link_entries[base_entry_index]);
        tblPtr += (expected_link_entriesCount * sizeof(LWLINK_CONFIG_DATA_LINKENTRY));
    }
vbios_fetch_lwlink_entries_done:
    return status;
}

LW_STATUS
_lwswitch_vbios_assign_base_entry
(
    lwswitch_device         *device,
    LWSWITCH_BIOS_LWLINK_CONFIG    *bios_config
)
{
    LwU32 physical_id;
    LwU32 entry_index;

    physical_id = lwswitch_read_physical_id(device);

    for (entry_index = 0; entry_index < LWSWITCH_NUM_BIOS_LWLINK_CONFIG_BASE_ENTRY; entry_index++)
    {
        if (physical_id == bios_config->link_vbios_base_entry[entry_index].positionId)
        {
            bios_config->link_base_entry_assigned = entry_index;
            return LW_OK;
        }
    }

    // TODO: Bug 3507948
    LWSWITCH_PRINT(device, ERROR,
            "%s: Error on assigning base entry. Setting base entry index = 0\n",
            __FUNCTION__);
    bios_config->link_base_entry_assigned = 0;

    return LW_OK;
}

LW_STATUS
_lwswitch_setup_link_vbios_overrides
(
    lwswitch_device *device,
    LWSWITCH_BIOS_LWLINK_CONFIG *bios_config
)
{
    LW_STATUS    status         = LW_OK;

    if (bios_config == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
                "%s: BIOS config override not supported\n",
                __FUNCTION__);
         return -LWL_ERR_NOT_SUPPORTED;
    }

    bios_config->vbios_disabled_link_mask = 0;

    bios_config->bit_address                 = 0;
    bios_config->pci_image_address           = 0;
    bios_config->lwlink_config_table_address = 0;

    if ((device->biosImage.size == 0) || (device->biosImage.pImage == NULL))
    {
#if !defined(LW_MODS)
        LWSWITCH_PRINT(device, ERROR,
                "%s: VBIOS not exist size:0x%x\n",
                __FUNCTION__, device->biosImage.size);
#else
        LWSWITCH_PRINT(device, SETUP,
                "%s: VBIOS not exist - Need to confirm on SPI interface support size:0x%x\n",
                __FUNCTION__, device->biosImage.size);
#endif
         return -LWL_ERR_NOT_SUPPORTED;
    }

    //
    // Locate the PCI ROM Image
    //
    if (_lwswitch_vbios_identify_pci_image_loc(device, bios_config)  != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Error on identifying pci image loc\n",
            __FUNCTION__);
        status = LW_ERR_GENERIC;
        goto setup_link_vbios_overrides_done;
    }

    //
    // Locate and fetch BIT offset
    //
    if (_lwswitch_vbios_update_bit_Offset(device, bios_config) != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Error on identifying pci image loc\n",
            __FUNCTION__);
        status = LW_ERR_GENERIC;
        goto setup_link_vbios_overrides_done;
    }

    //
    // Fetch LwLink Entries
    //
    if (_lwswitch_vbios_fetch_lwlink_entries(device, bios_config) != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Error on identifying pci image loc\n",
            __FUNCTION__);
        status = LW_ERR_GENERIC;
        goto setup_link_vbios_overrides_done;
    }

    //
    // Assign Base Entry for this device
    //
    if (_lwswitch_vbios_assign_base_entry(device, bios_config) != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Error on assigning base entry\n",
            __FUNCTION__);
        status = LW_ERR_GENERIC;
        goto setup_link_vbios_overrides_done;
    }

setup_link_vbios_overrides_done:
    if (status != LW_OK)
    {
        bios_config->bit_address                = 0;
        bios_config->pci_image_address          = 0;
        bios_config->lwlink_config_table_address =0;
    }
    return status;
}

static void
_lwswitch_load_link_disable_settings_lr10
(
    lwswitch_device *device,
    LWSWITCH_BIOS_LWLINK_CONFIG *bios_config,
    lwlink_link *link
)
{
    LwU32 val;
    LWLINK_CONFIG_DATA_LINKENTRY *vbios_link_entry = NULL;

    // SW CTRL - clear out LINK_DISABLE on driver load
    val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber,
            LWLIPT_LNK, _LWLIPT_LNK, _CTRL_SW_LINK_MODE_CTRL);
    val = FLD_SET_DRF(_LWLIPT_LNK, _CTRL_SW_LINK_MODE_CTRL, _LINK_DISABLE,
                      _ENABLED, val);
    LWSWITCH_LINK_WR32_LR10(device, link->linkNumber,
            LWLIPT_LNK, _LWLIPT_LNK, _CTRL_SW_LINK_MODE_CTRL, val);

    //
    // SYSTEM CTRL
    // If the SYSTEM_CTRL setting had been overidden by another entity,
    // it should also be locked, so this write would not take effect.
    //
    if (bios_config != NULL)
    {
        vbios_link_entry = &bios_config->link_vbios_entry[bios_config->link_base_entry_assigned][link->linkNumber];
    }

    val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber,
            LWLIPT_LNK, _LWLIPT_LNK, _CTRL_SYSTEM_LINK_MODE_CTRL);

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    if ((vbios_link_entry != NULL) &&
         (FLD_TEST_DRF(_LWLINK_VBIOS,_PARAM0, _LINK, _DISABLE, vbios_link_entry->lwLinkparam0) ||
         (FLD_TEST_DRF(_LWLINK_VBIOS,_PARAM0, _ACTIVE_REPEATER, _PRESENT, vbios_link_entry->lwLinkparam0) && 
          !cciIsLinkManaged(device, link->linkNumber))))
#else
    if ((vbios_link_entry != NULL) &&
         (FLD_TEST_DRF(_LWLINK_VBIOS,_PARAM0, _LINK, _DISABLE, vbios_link_entry->lwLinkparam0)))
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    {
        if (!lwswitch_is_link_in_reset(device, link))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: link #%d is not in reset, cannot set LINK_DISABLE\n",
                __FUNCTION__, link->linkNumber);
            return;
        }
        val = FLD_SET_DRF(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_MODE_CTRL, _LINK_DISABLE,
                          _DISABLED, val);
        LWSWITCH_LINK_WR32_LR10(device, link->linkNumber,
                LWLIPT_LNK, _LWLIPT_LNK, _CTRL_SYSTEM_LINK_MODE_CTRL, val);

        // Set link to invalid and unregister from corelib
        device->link[link->linkNumber].valid = LW_FALSE;
        lwlink_lib_unregister_link(link);
        lwswitch_destroy_link(link);

        return;
    }
    else
    {
        val = FLD_SET_DRF(_LWLIPT_LNK, _CTRL_SYSTEM_LINK_MODE_CTRL, _LINK_DISABLE,
                          _ENABLED, val);
        LWSWITCH_LINK_WR32_LR10(device, link->linkNumber,
                LWLIPT_LNK, _LWLIPT_LNK, _CTRL_SYSTEM_LINK_MODE_CTRL, val);
    }
}

/*
 * @Brief : Setting up system registers after device initialization
 *
 * @Description :
 *
 * @param[in] device        a reference to the device to initialize
 */
LwlStatus
lwswitch_setup_link_system_registers_lr10
(
    lwswitch_device *device
)
{
    lwlink_link *link;
    LwU8 i;
    LwU32 val;
    LwU64 enabledLinkMask;
    LWSWITCH_BIOS_LWLINK_CONFIG *bios_config;

    bios_config = lwswitch_get_bios_lwlink_config(device);
    if ((bios_config == NULL) || (bios_config->bit_address == 0))
    {
        LWSWITCH_PRINT(device, WARN,
            "%s: VBIOS LwLink configuration table not found\n",
            __FUNCTION__);
    }

    enabledLinkMask = lwswitch_get_enabled_link_mask(device);

    FOR_EACH_INDEX_IN_MASK(64, i, enabledLinkMask)
    {
        LWSWITCH_ASSERT(i < LWSWITCH_LINK_COUNT(device));

        link = lwswitch_get_link(device, i);

        if ((link == NULL) ||
            !LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLDL, link->linkNumber) ||
            (i >= LWSWITCH_LWLINK_MAX_LINKS))
        {
            continue;
        }

        // AC vs DC mode SYSTEM register
        if (link->ac_coupled)
        {
            //
            // In LWL3.0, ACMODE is handled by MINION in the INITPHASE1 command
            // Here we just setup the register with the proper info
            //
            val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLIPT_LNK,
                    _LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL);
            val = FLD_SET_DRF(_LWLIPT_LNK,
                    _CTRL_SYSTEM_LINK_CHANNEL_CTRL, _AC_DC_MODE, _AC, val);
            LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLIPT_LNK,
                    _LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL, val);
        }

        _lwswitch_setup_link_system_registers_lr10(device, bios_config, link);
        _lwswitch_load_link_disable_settings_lr10(device, bios_config, link);
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_deassert_link_reset_lr10
(
    lwswitch_device *device,
    lwlink_link     *link
)
{
    LwU64 mode;
    LwlStatus status = LWL_SUCCESS;

    status = device->hal.lwswitch_corelib_get_dl_link_mode(link, &mode);

    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
                "%s:DL link mode failed on link %d\n",
                __FUNCTION__, link->linkNumber);
        return status;
    }

    // Check if the link is RESET
    if (mode != LWLINK_LINKSTATE_RESET)
    {
        return LWL_SUCCESS;
    }

    // Send INITPHASE1 to bring link out of reset
    status = link->link_handlers->set_dl_link_mode(link,
                                        LWLINK_LINKSTATE_INITPHASE1,
                                        LWLINK_STATE_CHANGE_ASYNC);

    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
                "%s: INITPHASE1 failed on link %d\n",
                __FUNCTION__, link->linkNumber);
    }

    return status;
}

void
lwswitch_store_seed_data_from_inforom_to_corelib_lr10
(
    lwswitch_device *device
)
{
    lwlink_link *link;
    LwU8 i;
    LwU64 enabledLinkMask;
    LwU32 seedData[LWLINK_MAX_SEED_BUFFER_SIZE];

    if (device->regkeys.minion_cache_seeds == LW_SWITCH_REGKEY_MINION_CACHE_SEEDS_DISABLE)
    {
        return;
    }

    enabledLinkMask = lwswitch_get_enabled_link_mask(device);

    FOR_EACH_INDEX_IN_MASK(64, i, enabledLinkMask)
    {
        LWSWITCH_ASSERT(i < LWSWITCH_LINK_COUNT(device));

        link = lwswitch_get_link(device, i);
        if (link == NULL)
        {
            continue;
        }

        lwswitch_inforom_lwlink_get_minion_data(device, i, seedData);
        lwlink_lib_save_training_seeds(link, seedData);
    }
    FOR_EACH_INDEX_IN_MASK_END;
}

static LwU32
_lwswitch_get_num_vcs_lr10
(
    lwswitch_device *device
)
{
    return LWSWITCH_NUM_VCS_LR10;
}

void
lwswitch_determine_platform_lr10
(
    lwswitch_device *device
)
{
    LwU32 value;

    //
    // Determine which model we are using SMC_BOOT_2 and OS query
    //
    value = LWSWITCH_REG_RD32(device, _PSMC, _BOOT_2);
    device->is_emulation = FLD_TEST_DRF(_PSMC, _BOOT_2, _EMULATION, _YES, value);

    if (!IS_EMULATION(device))
    {
        // If we are not on fmodel, we must be on RTL sim or silicon
        if (FLD_TEST_DRF(_PSMC, _BOOT_2, _FMODEL, _YES, value))
        {
            device->is_fmodel = LW_TRUE;
        }
        else
        {
            device->is_rtlsim = LW_TRUE;

            // Let OS code finalize RTL sim vs silicon setting
            lwswitch_os_override_platform(device->os_handle, &device->is_rtlsim);
        }
    }

#if defined(LWLINK_PRINT_ENABLED)
    {
        const char *build;
        const char *mode;

        build = "HW";
        if (IS_FMODEL(device))
            mode = "fmodel";
        else if (IS_RTLSIM(device))
            mode = "rtlsim";
        else if (IS_EMULATION(device))
            mode = "emulation";
        else
            mode = "silicon";

        LWSWITCH_PRINT(device, SETUP,
            "%s: build: %s platform: %s\n",
             __FUNCTION__, build, mode);
    }
#endif // LWLINK_PRINT_ENABLED
}

static void
_lwswitch_portstat_reset_latency_counters
(
    lwswitch_device *device
)
{
    // Set SNAPONDEMAND from 0->1 to reset the counters
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _NPORT, _PORTSTAT_SNAP_CONTROL,
        DRF_DEF(_NPORT, _PORTSTAT_SNAP_CONTROL, _STARTCOUNTER, _ENABLE) |
        DRF_DEF(_NPORT, _PORTSTAT_SNAP_CONTROL, _SNAPONDEMAND, _ENABLE));

    // Set SNAPONDEMAND back to 0.
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _NPORT, _PORTSTAT_SNAP_CONTROL,
        DRF_DEF(_NPORT, _PORTSTAT_SNAP_CONTROL, _STARTCOUNTER, _ENABLE) |
        DRF_DEF(_NPORT, _PORTSTAT_SNAP_CONTROL, _SNAPONDEMAND, _DISABLE));
}

//
// Data collector which runs on a background thread, collecting latency stats.
//
// The latency counters have a maximum window period of 3.299 seconds
// (2^32 clk cycles). The counters reset after this period. So SW snaps
// the bins and records latencies every 3 seconds. Setting SNAPONDEMAND from 0->1
// snaps the  latency counters and updates them to PRI registers for
// the SW to read. It then resets the counters to start collecting fresh latencies.
//

void
lwswitch_internal_latency_bin_log_lr10
(
    lwswitch_device *device
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LwU32 idx_nport;
    LwU32 idx_vc;
    LwBool vc_valid;
    LwU32 latency;
    LwU64 time_nsec;
    LwU32 link_type;    // Access or trunk link
    LwU64 last_visited_time_nsec;

    if (chip_device->latency_stats == NULL)
    {
        // Latency stat buffers not allocated yet
        return;
    }

    time_nsec = lwswitch_os_get_platform_time();
    last_visited_time_nsec = chip_device->latency_stats->last_visited_time_nsec;

    // Update last visited time
    chip_device->latency_stats->last_visited_time_nsec = time_nsec;

    // Compare time stamp and reset the counters if the snap is missed
    if (!IS_RTLSIM(device) || !IS_FMODEL(device))
    {
        if ((last_visited_time_nsec != 0) &&
            ((time_nsec - last_visited_time_nsec) > 3 * LWSWITCH_INTERVAL_1SEC_IN_NS))
        {
            LWSWITCH_PRINT(device, ERROR,
                "Latency metrics recording interval missed.  Resetting counters.\n");
            _lwswitch_portstat_reset_latency_counters(device);
            return;
        }
    }

    for (idx_nport=0; idx_nport < LWSWITCH_LINK_COUNT(device); idx_nport++)
    {
        if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, NPORT, idx_nport))
        {
            continue;
        }

        // Setting SNAPONDEMAND from 0->1 snaps the latencies and resets the counters
        LWSWITCH_LINK_WR32_LR10(device, idx_nport, NPORT, _NPORT, _PORTSTAT_SNAP_CONTROL,
            DRF_DEF(_NPORT, _PORTSTAT_SNAP_CONTROL, _STARTCOUNTER, _ENABLE) |
            DRF_DEF(_NPORT, _PORTSTAT_SNAP_CONTROL, _SNAPONDEMAND, _ENABLE));

        //
        // TODO: Check _STARTCOUNTER and don't log if counter not enabled.
        // Lwrrently all counters are always enabled
        //

        link_type = LWSWITCH_LINK_RD32_LR10(device, idx_nport, NPORT, _NPORT, _CTRL);
        for (idx_vc = 0; idx_vc < LWSWITCH_NUM_VCS_LR10; idx_vc++)
        {
            vc_valid = LW_FALSE;

            // VC's CREQ0(0) and RSP0(5) are relevant on access links.
            if (FLD_TEST_DRF(_NPORT, _CTRL, _TRUNKLINKENB, _ACCESSLINK, link_type) &&
                ((idx_vc == LW_NPORT_VC_MAPPING_CREQ0) ||
                (idx_vc == LW_NPORT_VC_MAPPING_RSP0)))
            {
                vc_valid = LW_TRUE;
            }

            // VC's CREQ0(0), RSP0(5), CREQ1(6) and RSP1(7) are relevant on trunk links.
            if (FLD_TEST_DRF(_NPORT, _CTRL, _TRUNKLINKENB, _TRUNKLINK, link_type) &&
                ((idx_vc == LW_NPORT_VC_MAPPING_CREQ0)  ||
                 (idx_vc == LW_NPORT_VC_MAPPING_RSP0)   ||
                 (idx_vc == LW_NPORT_VC_MAPPING_CREQ1)  ||
                 (idx_vc == LW_NPORT_VC_MAPPING_RSP1)))
            {
                vc_valid = LW_TRUE;
            }

            // If the VC is not being used, skip reading it
            if (!vc_valid)
            {
                continue;
            }

            latency = LWSWITCH_NPORT_PORTSTAT_RD32_LR10(device, idx_nport, _COUNT, _LOW, idx_vc);
            chip_device->latency_stats->latency[idx_vc].aclwm_latency[idx_nport].low += latency;

            latency = LWSWITCH_NPORT_PORTSTAT_RD32_LR10(device, idx_nport, _COUNT, _MEDIUM, idx_vc);
            chip_device->latency_stats->latency[idx_vc].aclwm_latency[idx_nport].medium += latency;

            latency = LWSWITCH_NPORT_PORTSTAT_RD32_LR10(device, idx_nport, _COUNT, _HIGH, idx_vc);
            chip_device->latency_stats->latency[idx_vc].aclwm_latency[idx_nport].high += latency;

            latency = LWSWITCH_NPORT_PORTSTAT_RD32_LR10(device, idx_nport, _COUNT, _PANIC, idx_vc);
            chip_device->latency_stats->latency[idx_vc].aclwm_latency[idx_nport].panic += latency;

            latency = LWSWITCH_NPORT_PORTSTAT_RD32_LR10(device, idx_nport, _PACKET, _COUNT, idx_vc);
            chip_device->latency_stats->latency[idx_vc].aclwm_latency[idx_nport].count += latency;

            // Note the time of this snap
            chip_device->latency_stats->latency[idx_vc].last_read_time_nsec = time_nsec;
            chip_device->latency_stats->latency[idx_vc].count++;
        }

        // Disable SNAPONDEMAND after fetching the latencies
        LWSWITCH_LINK_WR32_LR10(device, idx_nport, NPORT, _NPORT, _PORTSTAT_SNAP_CONTROL,
            DRF_DEF(_NPORT, _PORTSTAT_SNAP_CONTROL, _STARTCOUNTER, _ENABLE) |
            DRF_DEF(_NPORT, _PORTSTAT_SNAP_CONTROL, _SNAPONDEMAND, _DISABLE));
    }
}

void
lwswitch_ecc_writeback_task_lr10
(
    lwswitch_device *device
)
{
}

void
lwswitch_set_ganged_link_table_lr10
(
    lwswitch_device *device,
    LwU32            firstIndex,
    LwU64           *ganged_link_table,
    LwU32            numEntries
)
{
    LwU32 i;

    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _ROUTE, _REG_TABLE_ADDRESS,
        DRF_NUM(_ROUTE, _REG_TABLE_ADDRESS, _INDEX, firstIndex) |
        DRF_NUM(_ROUTE, _REG_TABLE_ADDRESS, _AUTO_INCR, 1));

    for (i = 0; i < numEntries; i++)
    {
        LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _ROUTE, _REG_TABLE_DATA0,
            LwU64_LO32(ganged_link_table[i]));

        LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _ROUTE, _REG_TABLE_DATA0,
            LwU64_HI32(ganged_link_table[i]));
    }
}

static LwlStatus
_lwswitch_init_ganged_link_routing
(
    lwswitch_device *device
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LwU32        gang_index, gang_size;
    LwU64        gang_entry;
    LwU32        block_index;
    LwU32        block_count = 16;
    LwU32        glt_entries = 16;
    LwU32        glt_size = ROUTE_GANG_TABLE_SIZE / 2;
    LwU64        *ganged_link_table = NULL;
    LwU32        block_size = ROUTE_GANG_TABLE_SIZE / block_count;
    LwU32        table_index = 0;
    LwU32        i;

    //
    // Refer to switch IAS 11.2 Figure 82. Limerock Ganged RAM Table Format
    //
    // The ganged link routing table is composed of 512 entries divided into 16 sections.
    // Each section specifies how requests should be routed through the ganged links.
    // Each 32-bit entry is composed of eight 4-bit fields specifying the set of of links
    // to distribute through.  More complex spray patterns could be constructed, but for
    // now initialize it with a uniform distribution pattern.
    //
    // The ganged link routing table will be loaded with following values:
    // Typically the first section would be filled with (0,1,2,3,4,5,6,7), (8,9,10,11,12,13,14,15),...
    // Typically the second section would be filled with (0,0,0,0,0,0,0,0), (0,0,0,0,0,0,0,0),...
    // Typically the third section would be filled with (0,1,0,1,0,1,0,1), (0,1,0,1,0,1,0,1),...
    // Typically the third section would be filled with (0,1,2,0,1,2,0,1), (2,0,1,2,0,1,2,0),...
    //  :
    // The last section would typically be filled with (0,1,2,3,4,5,6,7), (8,9,10,11,12,13,14,0),...
    //
    // Refer table 20: Definition of size bits used with Ganged Link Number Table.
    // Note that section 0 corresponds with 16 ganged links.  Section N corresponds with
    // N ganged links.
    //

    //Alloc memory for Ganged Link Table
    ganged_link_table = lwswitch_os_malloc(glt_size * sizeof(gang_entry));
    if (ganged_link_table == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "Failed to allocate memory for GLT!!\n");
        return -LWL_NO_MEM;
    }

    for (block_index = 0; block_index < block_count; block_index++)
    {
        gang_size = ((block_index==0) ? 16 : block_index);

        for (gang_index = 0; gang_index < block_size/2; gang_index++)
        {
            gang_entry = 0;
            LWSWITCH_ASSERT(table_index < glt_size);

            for (i = 0; i < glt_entries; i++)
            {
                gang_entry |=
                    DRF_NUM64(_ROUTE, _REG_TABLE_DATA0, _GLX(i), (16 * gang_index + i) % gang_size);
            }

            ganged_link_table[table_index++] = gang_entry;
        }
    }

    lwswitch_set_ganged_link_table_lr10(device, 0, ganged_link_table, glt_size);

    chip_device->ganged_link_table = ganged_link_table;

    return LWL_SUCCESS;
}

static LwlStatus
lwswitch_initialize_ip_wrappers_lr10
(
    lwswitch_device *device
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LwU32 engine_enable_mask;
    LwU32 engine_disable_mask;
    LwU32 i, j;
    LwU32 idx_link;

    //
    // Now that software knows the devices and addresses, it must take all
    // the wrapper modules out of reset.  It does this by writing to the
    // PMC module enable registers.
    //

// Temporary - bug 2069764
//    LWSWITCH_REG_WR32(device, _PSMC, _ENABLE,
//        DRF_DEF(_PSMC, _ENABLE, _SAW, _ENABLE) |
//        DRF_DEF(_PSMC, _ENABLE, _PRIV_RING, _ENABLE) |
//        DRF_DEF(_PSMC, _ENABLE, _PERFMON, _ENABLE));

    LWSWITCH_SAW_WR32_LR10(device, _LWLSAW_LWSPMC, _ENABLE,
        DRF_DEF(_LWLSAW_LWSPMC, _ENABLE, _NXBAR, _ENABLE));

    //
    // At this point the list of discovered devices has been cross-referenced
    // with the ROM configuration, platform configuration, and regkey override.
    // The LWLIPT & NPORT enable filtering done here further updates the MMIO
    // information based on KVM.
    //

    // Enable the LWLIPT units that have been discovered
    engine_enable_mask = 0;
    for (i = 0; i < LWSWITCH_ENG_COUNT(device, LWLW, ); i++)
    {
        if (LWSWITCH_ENG_IS_VALID(device, LWLW, i))
        {
            engine_enable_mask |= LWBIT(i);
        }
    }
    LWSWITCH_SAW_WR32_LR10(device, _LWLSAW_LWSPMC, _ENABLE_LWLIPT, engine_enable_mask);

    //
    // In bare metal we write ENABLE_LWLIPT to enable the units that aren't
    // disabled by ROM configuration, platform configuration, or regkey override.
    // If we are running inside a VM, the hypervisor has already set ENABLE_LWLIPT
    // and write protected it.  Reading ENABLE_LWLIPT tells us which units we
    // are allowed to use inside this VM.
    //
    engine_disable_mask = ~LWSWITCH_SAW_RD32_LR10(device, _LWLSAW_LWSPMC, _ENABLE_LWLIPT);
    if (engine_enable_mask != ~engine_disable_mask)
    {
        LWSWITCH_PRINT(device, WARN,
            "LW_LWLSAW_LWSPMC_ENABLE_LWLIPT mismatch: wrote 0x%x, read 0x%x\n",
            engine_enable_mask,
            ~engine_disable_mask);
        LWSWITCH_PRINT(device, WARN,
            "Ignoring LW_LWLSAW_LWSPMC_ENABLE_LWLIPT readback until supported on fmodel\n");
        engine_disable_mask = ~engine_enable_mask;
    }
    engine_disable_mask &= LWBIT(LWSWITCH_ENG_COUNT(device, LWLW, )) - 1;
    FOR_EACH_INDEX_IN_MASK(32, i, engine_disable_mask)
    {
        chip_device->engLWLW[i].valid = LW_FALSE;
        for (j = 0; j < LWSWITCH_LINKS_PER_LWLW; j++)
        {
            idx_link = i * LWSWITCH_LINKS_PER_LWLW + j;
            if (idx_link < LWSWITCH_LINK_COUNT(device))
            {
                device->link[idx_link].valid = LW_FALSE;
                //
                // TODO: This ilwalidate used to also ilwalidate all the
                // associated LWLW engFOO units. This is probably not necessary
                // but code that bypasses the link valid check might touch the
                // underlying units when they are not supposed to.
                //
            }
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    // Enable the NPORT units that have been discovered
    engine_enable_mask = 0;
    for (i = 0; i < LWSWITCH_ENG_COUNT(device, NPG, ); i++)
    {
        if (LWSWITCH_ENG_IS_VALID(device, NPG, i))
        {
            engine_enable_mask |= LWBIT(i);
        }
    }
    LWSWITCH_SAW_WR32_LR10(device, _LWLSAW_LWSPMC, _ENABLE_NPG, engine_enable_mask);

    //
    // In bare metal we write ENABLE_NPG to enable the units that aren't
    // disabled by ROM configuration, platform configuration, or regkey override.
    // If we are running inside a VM, the hypervisor has already set ENABLE_NPG
    // and write protected it.  Reading ENABLE_NPG tells us which units we
    // are allowed to use inside this VM.
    //
    engine_disable_mask = ~LWSWITCH_SAW_RD32_LR10(device, _LWLSAW_LWSPMC, _ENABLE_NPG);
    if (engine_enable_mask != ~engine_disable_mask)
    {
        LWSWITCH_PRINT(device, WARN,
            "LW_LWLSAW_LWSPMC_ENABLE_NPG mismatch: wrote 0x%x, read 0x%x\n",
            engine_enable_mask,
            ~engine_disable_mask);
        LWSWITCH_PRINT(device, WARN,
            "Ignoring LW_LWLSAW_LWSPMC_ENABLE_NPG readback until supported on fmodel\n");
        engine_disable_mask = ~engine_enable_mask;
    }
    engine_disable_mask &= LWBIT(LWSWITCH_ENG_COUNT(device, NPG, )) - 1;
    FOR_EACH_INDEX_IN_MASK(32, i, engine_disable_mask)
    {
        chip_device->engNPG[i].valid = LW_FALSE;
        for (j = 0; j < LWSWITCH_LINKS_PER_NPG; j++)
        {
            idx_link = i * LWSWITCH_LINKS_PER_NPG + j;

            if (idx_link < LWSWITCH_LINK_COUNT(device))
            {
                device->link[idx_link].valid = LW_FALSE;
                //
                // TODO: This ilwalidate used to also ilwalidate all the
                // associated NPG engFOO units. This is probably not necessary
                // but code that bypasses the link valid check might touch the
                // underlying units when they are not supposed to.
                //
            }
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return LWL_SUCCESS;
}

//
// Bring units out of warm reset on boot.  Used by driver load.
//
static void
_lwswitch_init_warm_reset_lr10
(
    lwswitch_device *device
)
{
    LwU32 idx_npg;
    LwU32 idx_nport;
    LwU32 nport_mask;
    LwU32 nport_disable = 0;

#if defined(LW_NPG_WARMRESET_NPORTDISABLE)
    nport_disable = DRF_NUM(_NPG, _WARMRESET, _NPORTDISABLE, ~nport_mask);
#endif

    //
    // Walk the NPGs and build the mask of extant NPORTs
    //
    for (idx_npg = 0; idx_npg < LWSWITCH_ENG_COUNT(device, NPG, ); idx_npg++)
    {
        if (LWSWITCH_ENG_IS_VALID(device, NPG, idx_npg))
        {
            nport_mask = 0;
            for (idx_nport = 0; idx_nport < LWSWITCH_NPORT_PER_NPG; idx_nport++)
            {
                nport_mask |=
                    (LWSWITCH_ENG_IS_VALID(device, NPORT, idx_npg*LWSWITCH_NPORT_PER_NPG + idx_nport) ?
                    LWBIT(idx_nport) : 0x0);
            }

            LWSWITCH_NPG_WR32_LR10(device, idx_npg,
                _NPG, _WARMRESET,
                nport_disable |
                DRF_NUM(_NPG, _WARMRESET, _NPORTWARMRESET, nport_mask));
        }
    }
}

/*
 * CTRL_LWSWITCH_SET_REMAP_POLICY
 */

LwlStatus
lwswitch_get_remap_table_selector_lr10
(
    lwswitch_device *device,
    LWSWITCH_TABLE_SELECT_REMAP table_selector,
    LwU32 *remap_ram_sel
)
{
    LwU32 ram_sel = 0;

    switch (table_selector)
    {
        case LWSWITCH_TABLE_SELECT_REMAP_PRIMARY:
            ram_sel = LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECTSREMAPPOLICYRAM;
            break;
        default:
            // Unsupported remap table selector
            return -LWL_ERR_NOT_SUPPORTED;
            break;
    }

    if (remap_ram_sel)
    {
        *remap_ram_sel = ram_sel;
    }

    return LWL_SUCCESS;
}

LwU32
lwswitch_get_ingress_ram_size_lr10
(
    lwswitch_device *device,
    LwU32 ingress_ram_selector      // LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECT*
)
{
    LwU32 ram_size = 0;

    switch (ingress_ram_selector)
    {
        case LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECTSREMAPPOLICYRAM:
            ram_size = LW_INGRESS_REQRSPMAPADDR_RAM_ADDRESS_REMAPTAB_DEPTH + 1;
            break;
        case LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECTSRIDROUTERAM:
            ram_size = LW_INGRESS_REQRSPMAPADDR_RAM_ADDRESS_RID_TAB_DEPTH + 1;
            break;
        case LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECTSRLANROUTERAM:
            ram_size = LW_INGRESS_REQRSPMAPADDR_RAM_ADDRESS_RLAN_TAB_DEPTH + 1;
            break;
        default:
            // Unsupported ingress RAM selector
            break;
    }

    return ram_size;
}

static void
_lwswitch_set_remap_policy_lr10
(
    lwswitch_device *device,
    LwU32 portNum,
    LwU32 firstIndex,
    LwU32 numEntries,
    LWSWITCH_REMAP_POLICY_ENTRY *remap_policy
)
{
    LwU32 i;
    LwU32 remap_address;
    LwU32 address_offset;
    LwU32 address_base;
    LwU32 address_limit;

    LWSWITCH_LINK_WR32_LR10(device, portNum, NPORT, _INGRESS, _REQRSPMAPADDR,
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _RAM_ADDRESS, firstIndex) |
        DRF_DEF(_INGRESS, _REQRSPMAPADDR, _RAM_SEL, _SELECTSREMAPPOLICYRAM) |
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _AUTO_INCR, 1));

    for (i = 0; i < numEntries; i++)
    {
        // Set each field if enabled, else set it to 0.
        remap_address = DRF_VAL64(_INGRESS, _REMAP, _ADDR_PHYS_LR10, remap_policy[i].address);
        address_offset = DRF_VAL64(_INGRESS, _REMAP, _ADR_OFFSET_PHYS_LR10, remap_policy[i].addressOffset);
        address_base = DRF_VAL64(_INGRESS, _REMAP, _ADR_BASE_PHYS_LR10, remap_policy[i].addressBase);
        address_limit = DRF_VAL64(_INGRESS, _REMAP, _ADR_LIMIT_PHYS_LR10, remap_policy[i].addressLimit);

        LWSWITCH_LINK_WR32_LR10(device, portNum, NPORT, _INGRESS, _REMAPTABDATA1,
            DRF_NUM(_INGRESS, _REMAPTABDATA1, _REQCTXT_MSK, remap_policy[i].reqCtxMask) |
            DRF_NUM(_INGRESS, _REMAPTABDATA1, _REQCTXT_CHK, remap_policy[i].reqCtxChk));
        LWSWITCH_LINK_WR32_LR10(device, portNum, NPORT, _INGRESS, _REMAPTABDATA2,
            DRF_NUM(_INGRESS, _REMAPTABDATA2, _REQCTXT_REP, remap_policy[i].reqCtxRep) |
            DRF_NUM(_INGRESS, _REMAPTABDATA2, _ADR_OFFSET, address_offset));
        LWSWITCH_LINK_WR32_LR10(device, portNum, NPORT, _INGRESS, _REMAPTABDATA3,
            DRF_NUM(_INGRESS, _REMAPTABDATA3, _ADR_BASE, address_base) |
            DRF_NUM(_INGRESS, _REMAPTABDATA3, _ADR_LIMIT, address_limit));
        LWSWITCH_LINK_WR32_LR10(device, portNum, NPORT, _INGRESS, _REMAPTABDATA4,
            DRF_NUM(_INGRESS, _REMAPTABDATA4, _TGTID, remap_policy[i].targetId) |
            DRF_NUM(_INGRESS, _REMAPTABDATA4, _RFUNC, remap_policy[i].flags));

        // Write last and auto-increment
        LWSWITCH_LINK_WR32_LR10(device, portNum, NPORT, _INGRESS, _REMAPTABDATA0,
            DRF_NUM(_INGRESS, _REMAPTABDATA0, _RMAP_ADDR, remap_address) |
            DRF_NUM(_INGRESS, _REMAPTABDATA0, _IRL_SEL, remap_policy[i].irlSelect) |
            DRF_NUM(_INGRESS, _REMAPTABDATA0, _ACLVALID, remap_policy[i].entryValid));
    }
}

LwlStatus
lwswitch_ctrl_set_remap_policy_lr10
(
    lwswitch_device *device,
    LWSWITCH_SET_REMAP_POLICY *p
)
{
    LwU32 i;
    LwU32 rfunc;
    LwU32 ram_size;
    LwlStatus retval = LWL_SUCCESS;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, NPORT, p->portNum))
    {
        LWSWITCH_PRINT(device, ERROR,
            "NPORT port #%d not valid\n",
            p->portNum);
        return -LWL_BAD_ARGS;
    }

    if (p->tableSelect != LWSWITCH_TABLE_SELECT_REMAP_PRIMARY)
    {
        LWSWITCH_PRINT(device, ERROR,
            "Remap table #%d not supported\n",
            p->tableSelect);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    ram_size = lwswitch_get_ingress_ram_size(device, LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECTSREMAPPOLICYRAM);
    if ((p->firstIndex >= ram_size) ||
        (p->numEntries > LWSWITCH_REMAP_POLICY_ENTRIES_MAX) ||
        (p->firstIndex + p->numEntries > ram_size))
    {
        LWSWITCH_PRINT(device, ERROR,
            "remapPolicy[%d..%d] overflows range %d..%d or size %d.\n",
            p->firstIndex, p->firstIndex + p->numEntries - 1,
            0, ram_size - 1,
            LWSWITCH_REMAP_POLICY_ENTRIES_MAX);
        return -LWL_BAD_ARGS;
    }

    for (i = 0; i < p->numEntries; i++)
    {
        if (p->remapPolicy[i].targetId &
            ~DRF_MASK(LW_INGRESS_REMAPTABDATA4_TGTID))
        {
            LWSWITCH_PRINT(device, ERROR,
                "remapPolicy[%d].targetId 0x%x out of valid range (0x%x..0x%x)\n",
                i, p->remapPolicy[i].targetId,
                0, DRF_MASK(LW_INGRESS_REMAPTABDATA4_TGTID));
            return -LWL_BAD_ARGS;
        }

        if (p->remapPolicy[i].irlSelect &
            ~DRF_MASK(LW_INGRESS_REMAPTABDATA0_IRL_SEL))
        {
            LWSWITCH_PRINT(device, ERROR,
                "remapPolicy[%d].irlSelect 0x%x out of valid range (0x%x..0x%x)\n",
                i, p->remapPolicy[i].irlSelect,
                0, DRF_MASK(LW_INGRESS_REMAPTABDATA0_IRL_SEL));
            return -LWL_BAD_ARGS;
        }

        rfunc = p->remapPolicy[i].flags &
            (
                LWSWITCH_REMAP_POLICY_FLAGS_REMAP_ADDR |
                LWSWITCH_REMAP_POLICY_FLAGS_REQCTXT_CHECK |
                LWSWITCH_REMAP_POLICY_FLAGS_REQCTXT_REPLACE |
                LWSWITCH_REMAP_POLICY_FLAGS_ADR_BASE |
                LWSWITCH_REMAP_POLICY_FLAGS_ADR_OFFSET
            );
        if (rfunc != p->remapPolicy[i].flags)
        {
            LWSWITCH_PRINT(device, ERROR,
                "remapPolicy[%d].flags 0x%x has undefined flags (0x%x)\n",
                i, p->remapPolicy[i].flags,
                p->remapPolicy[i].flags ^ rfunc);
            return -LWL_BAD_ARGS;
        }

        // Validate that only bits 46:36 are used
        if (p->remapPolicy[i].address &
            ~DRF_SHIFTMASK64(LW_INGRESS_REMAP_ADDR_PHYS_LR10))
        {
            LWSWITCH_PRINT(device, ERROR,
                "remapPolicy[%d].address 0x%llx & ~0x%llx != 0\n",
                i, p->remapPolicy[i].address,
                DRF_SHIFTMASK64(LW_INGRESS_REMAP_ADDR_PHYS_LR10));
            return -LWL_BAD_ARGS;
        }

        if (p->remapPolicy[i].reqCtxMask &
           ~DRF_MASK(LW_INGRESS_REMAPTABDATA1_REQCTXT_MSK))
        {
            LWSWITCH_PRINT(device, ERROR,
                "remapPolicy[%d].reqCtxMask 0x%x out of valid range (0x%x..0x%x)\n",
                i, p->remapPolicy[i].reqCtxMask,
                0, DRF_MASK(LW_INGRESS_REMAPTABDATA1_REQCTXT_MSK));
            return -LWL_BAD_ARGS;
        }

        if (p->remapPolicy[i].reqCtxChk &
            ~DRF_MASK(LW_INGRESS_REMAPTABDATA1_REQCTXT_CHK))
        {
            LWSWITCH_PRINT(device, ERROR,
                "remapPolicy[%d].reqCtxChk 0x%x out of valid range (0x%x..0x%x)\n",
                i, p->remapPolicy[i].reqCtxChk,
                0, DRF_MASK(LW_INGRESS_REMAPTABDATA1_REQCTXT_CHK));
            return -LWL_BAD_ARGS;
        }

        if (p->remapPolicy[i].reqCtxRep &
            ~DRF_MASK(LW_INGRESS_REMAPTABDATA2_REQCTXT_REP))
        {
            LWSWITCH_PRINT(device, ERROR,
                "remapPolicy[%d].reqCtxRep 0x%x out of valid range (0x%x..0x%x)\n",
                i, p->remapPolicy[i].reqCtxRep,
                0, DRF_MASK(LW_INGRESS_REMAPTABDATA2_REQCTXT_REP));
            return -LWL_BAD_ARGS;
        }

        if ((p->remapPolicy[i].flags & LWSWITCH_REMAP_POLICY_FLAGS_ADR_OFFSET) &&
            !(p->remapPolicy[i].flags & LWSWITCH_REMAP_POLICY_FLAGS_ADR_BASE))
        {
            LWSWITCH_PRINT(device, ERROR,
                "remapPolicy[%d].flags: _FLAGS_ADR_OFFSET should not be set if "
                "_FLAGS_ADR_BASE is not set\n",
                i);
            return -LWL_BAD_ARGS;
        }

        // Validate that only bits 35:20 are used
        if (p->remapPolicy[i].addressBase &
            ~DRF_SHIFTMASK64(LW_INGRESS_REMAP_ADR_BASE_PHYS_LR10))
        {
            LWSWITCH_PRINT(device, ERROR,
                "remapPolicy[%d].addressBase 0x%llx & ~0x%llx != 0\n",
                i, p->remapPolicy[i].addressBase,
                DRF_SHIFTMASK64(LW_INGRESS_REMAP_ADR_BASE_PHYS_LR10));
            return -LWL_BAD_ARGS;
        }

        // Validate that only bits 35:20 are used
        if (p->remapPolicy[i].addressLimit &
            ~DRF_SHIFTMASK64(LW_INGRESS_REMAP_ADR_LIMIT_PHYS_LR10))
        {
            LWSWITCH_PRINT(device, ERROR,
                 "remapPolicy[%d].addressLimit 0x%llx & ~0x%llx != 0\n",
                 i, p->remapPolicy[i].addressLimit,
                 DRF_SHIFTMASK64(LW_INGRESS_REMAP_ADR_LIMIT_PHYS_LR10));
            return -LWL_BAD_ARGS;
        }

        // Validate base & limit describe a region
        if (p->remapPolicy[i].addressBase > p->remapPolicy[i].addressLimit)
        {
            LWSWITCH_PRINT(device, ERROR,
                 "remapPolicy[%d].addressBase/Limit invalid: 0x%llx > 0x%llx\n",
                 i, p->remapPolicy[i].addressBase, p->remapPolicy[i].addressLimit);
            return -LWL_BAD_ARGS;
        }

        // Validate that only bits 35:20 are used
        if (p->remapPolicy[i].addressOffset &
            ~DRF_SHIFTMASK64(LW_INGRESS_REMAP_ADR_OFFSET_PHYS_LR10))
        {
            LWSWITCH_PRINT(device, ERROR,
                "remapPolicy[%d].addressOffset 0x%llx & ~0x%llx != 0\n",
                i, p->remapPolicy[i].addressOffset,
                DRF_SHIFTMASK64(LW_INGRESS_REMAP_ADR_OFFSET_PHYS_LR10));
            return -LWL_BAD_ARGS;
        }

        // Validate limit - base + offset doesn't overflow 64G
        if ((p->remapPolicy[i].addressLimit - p->remapPolicy[i].addressBase +
                p->remapPolicy[i].addressOffset) &
            ~DRF_SHIFTMASK64(LW_INGRESS_REMAP_ADR_OFFSET_PHYS_LR10))
        {
            LWSWITCH_PRINT(device, ERROR,
                "remapPolicy[%d].addressLimit 0x%llx - addressBase 0x%llx + "
                "addressOffset 0x%llx overflows 64GB\n",
                i, p->remapPolicy[i].addressLimit, p->remapPolicy[i].addressBase,
                p->remapPolicy[i].addressOffset);
            return -LWL_BAD_ARGS;
        }
    }

    _lwswitch_set_remap_policy_lr10(device, p->portNum, p->firstIndex, p->numEntries, p->remapPolicy);

    return retval;
}

/*
 * CTRL_LWSWITCH_GET_REMAP_POLICY
 */

#define LWSWITCH_NUM_REMAP_POLICY_REGS_LR10 5

LwlStatus
lwswitch_ctrl_get_remap_policy_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_REMAP_POLICY_PARAMS *params
)
{
    LWSWITCH_REMAP_POLICY_ENTRY *remap_policy;
    LwU32 remap_policy_data[LWSWITCH_NUM_REMAP_POLICY_REGS_LR10]; // 5 REMAP tables
    LwU32 table_index;
    LwU32 remap_count;
    LwU32 remap_address;
    LwU32 address_offset;
    LwU32 address_base;
    LwU32 address_limit;
    LwU32 ram_size;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, NPORT, params->portNum))
    {
        LWSWITCH_PRINT(device, ERROR,
            "NPORT port #%d not valid\n",
            params->portNum);
        return -LWL_BAD_ARGS;
    }

    if (params->tableSelect != LWSWITCH_TABLE_SELECT_REMAP_PRIMARY)
    {
        LWSWITCH_PRINT(device, ERROR,
            "Remap table #%d not supported\n",
            params->tableSelect);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    ram_size = lwswitch_get_ingress_ram_size(device, LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECTSREMAPPOLICYRAM);
    if ((params->firstIndex >= ram_size))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: remapPolicy first index %d out of range[%d..%d].\n",
            __FUNCTION__, params->firstIndex, 0, ram_size - 1);
        return -LWL_BAD_ARGS;
    }

    lwswitch_os_memset(params->entry, 0, (LWSWITCH_REMAP_POLICY_ENTRIES_MAX *
        sizeof(LWSWITCH_REMAP_POLICY_ENTRY)));

    table_index = params->firstIndex;
    remap_policy = params->entry;
    remap_count = 0;

    /* set table offset */
    LWSWITCH_LINK_WR32_LR10(device, params->portNum, NPORT, _INGRESS, _REQRSPMAPADDR,
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _RAM_ADDRESS, params->firstIndex) |
        DRF_DEF(_INGRESS, _REQRSPMAPADDR, _RAM_SEL, _SELECTSREMAPPOLICYRAM) |
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _AUTO_INCR, 1));

    while (remap_count < LWSWITCH_REMAP_POLICY_ENTRIES_MAX &&
        table_index < ram_size)
    {
        remap_policy_data[0] = LWSWITCH_LINK_RD32_LR10(device, params->portNum, NPORT, _INGRESS, _REMAPTABDATA0);
        remap_policy_data[1] = LWSWITCH_LINK_RD32_LR10(device, params->portNum, NPORT, _INGRESS, _REMAPTABDATA1);
        remap_policy_data[2] = LWSWITCH_LINK_RD32_LR10(device, params->portNum, NPORT, _INGRESS, _REMAPTABDATA2);
        remap_policy_data[3] = LWSWITCH_LINK_RD32_LR10(device, params->portNum, NPORT, _INGRESS, _REMAPTABDATA3);
        remap_policy_data[4] = LWSWITCH_LINK_RD32_LR10(device, params->portNum, NPORT, _INGRESS, _REMAPTABDATA4);

        /* add to remap_entries list if nonzero */
        if (remap_policy_data[0] || remap_policy_data[1] || remap_policy_data[2] ||
            remap_policy_data[3] || remap_policy_data[4])
        {
            remap_policy[remap_count].irlSelect =
                DRF_VAL(_INGRESS, _REMAPTABDATA0, _IRL_SEL, remap_policy_data[0]);

            remap_policy[remap_count].entryValid =
                DRF_VAL(_INGRESS, _REMAPTABDATA0, _ACLVALID, remap_policy_data[0]);

            remap_address =
                DRF_VAL(_INGRESS, _REMAPTABDATA0, _RMAP_ADDR, remap_policy_data[0]);

            remap_policy[remap_count].address =
                DRF_NUM64(_INGRESS, _REMAP, _ADDR_PHYS_LR10, remap_address);

            remap_policy[remap_count].reqCtxMask =
                DRF_VAL(_INGRESS, _REMAPTABDATA1, _REQCTXT_MSK, remap_policy_data[1]);

            remap_policy[remap_count].reqCtxChk =
                DRF_VAL(_INGRESS, _REMAPTABDATA1, _REQCTXT_CHK, remap_policy_data[1]);

            remap_policy[remap_count].reqCtxRep =
                DRF_VAL(_INGRESS, _REMAPTABDATA2, _REQCTXT_REP, remap_policy_data[2]);

            address_offset =
                DRF_VAL(_INGRESS, _REMAPTABDATA2, _ADR_OFFSET, remap_policy_data[2]);

            remap_policy[remap_count].addressOffset =
                DRF_NUM64(_INGRESS, _REMAP, _ADR_OFFSET_PHYS_LR10, address_offset);

            address_base =
                DRF_VAL(_INGRESS, _REMAPTABDATA3, _ADR_BASE, remap_policy_data[3]);

            remap_policy[remap_count].addressBase =
                DRF_NUM64(_INGRESS, _REMAP, _ADR_BASE_PHYS_LR10, address_base);

            address_limit =
                DRF_VAL(_INGRESS, _REMAPTABDATA3, _ADR_LIMIT, remap_policy_data[3]);

            remap_policy[remap_count].addressLimit =
                DRF_NUM64(_INGRESS, _REMAP, _ADR_LIMIT_PHYS_LR10, address_limit);

            remap_policy[remap_count].targetId =
                DRF_VAL(_INGRESS, _REMAPTABDATA4, _TGTID, remap_policy_data[4]);

            remap_policy[remap_count].flags =
                DRF_VAL(_INGRESS, _REMAPTABDATA4, _RFUNC, remap_policy_data[4]);

            remap_count++;
        }

        table_index++;
    }

    params->nextIndex = table_index;
    params->numEntries = remap_count;

    return LWL_SUCCESS;
}

/*
 * CTRL_LWSWITCH_SET_REMAP_POLICY_VALID
 */
LwlStatus
lwswitch_ctrl_set_remap_policy_valid_lr10
(
    lwswitch_device *device,
    LWSWITCH_SET_REMAP_POLICY_VALID *p
)
{
    LwU32 remap_ram;
    LwU32 ram_address = p->firstIndex;
    LwU32 remap_policy_data[LWSWITCH_NUM_REMAP_POLICY_REGS_LR10]; // 5 REMAP tables
    LwU32 i;
    LwU32 ram_size;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, NPORT, p->portNum))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: NPORT port #%d not valid\n",
            __FUNCTION__, p->portNum);
        return -LWL_BAD_ARGS;
    }

    if (p->tableSelect != LWSWITCH_TABLE_SELECT_REMAP_PRIMARY)
    {
        LWSWITCH_PRINT(device, ERROR,
            "Remap table #%d not supported\n",
            p->tableSelect);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    ram_size = lwswitch_get_ingress_ram_size(device, LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECTSREMAPPOLICYRAM);
    if ((p->firstIndex >= ram_size) ||
        (p->numEntries > LWSWITCH_REMAP_POLICY_ENTRIES_MAX) ||
        (p->firstIndex + p->numEntries > ram_size))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: remapPolicy[%d..%d] overflows range %d..%d or size %d.\n",
            __FUNCTION__, p->firstIndex, p->firstIndex + p->numEntries - 1,
            0, ram_size - 1,
            LWSWITCH_REMAP_POLICY_ENTRIES_MAX);
        return -LWL_BAD_ARGS;
    }

    // Select REMAPPOLICY RAM and disable Auto Increament.
    remap_ram =
        DRF_DEF(_INGRESS, _REQRSPMAPADDR, _RAM_SEL, _SELECTSREMAPPOLICYRAM) |
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _AUTO_INCR, 0);

    for (i = 0; i < p->numEntries; i++)
    {
        /* set the ram address */
        remap_ram = FLD_SET_DRF_NUM(_INGRESS, _REQRSPMAPADDR, _RAM_ADDRESS, ram_address++, remap_ram);
        LWSWITCH_LINK_WR32_LR10(device, p->portNum, NPORT, _INGRESS, _REQRSPMAPADDR, remap_ram);

        remap_policy_data[0] = LWSWITCH_LINK_RD32_LR10(device, p->portNum, NPORT, _INGRESS, _REMAPTABDATA0);
        remap_policy_data[1] = LWSWITCH_LINK_RD32_LR10(device, p->portNum, NPORT, _INGRESS, _REMAPTABDATA1);
        remap_policy_data[2] = LWSWITCH_LINK_RD32_LR10(device, p->portNum, NPORT, _INGRESS, _REMAPTABDATA2);
        remap_policy_data[3] = LWSWITCH_LINK_RD32_LR10(device, p->portNum, NPORT, _INGRESS, _REMAPTABDATA3);
        remap_policy_data[4] = LWSWITCH_LINK_RD32_LR10(device, p->portNum, NPORT, _INGRESS, _REMAPTABDATA4);

        // Set valid bit in REMAPTABDATA0.
        remap_policy_data[0] = FLD_SET_DRF_NUM(_INGRESS, _REMAPTABDATA0, _ACLVALID, p->entryValid[i], remap_policy_data[0]);

        LWSWITCH_LINK_WR32_LR10(device, p->portNum, NPORT, _INGRESS, _REMAPTABDATA4, remap_policy_data[4]);
        LWSWITCH_LINK_WR32_LR10(device, p->portNum, NPORT, _INGRESS, _REMAPTABDATA3, remap_policy_data[3]);
        LWSWITCH_LINK_WR32_LR10(device, p->portNum, NPORT, _INGRESS, _REMAPTABDATA2, remap_policy_data[2]);
        LWSWITCH_LINK_WR32_LR10(device, p->portNum, NPORT, _INGRESS, _REMAPTABDATA1, remap_policy_data[1]);
        LWSWITCH_LINK_WR32_LR10(device, p->portNum, NPORT, _INGRESS, _REMAPTABDATA0, remap_policy_data[0]);
    }

    return LWL_SUCCESS;
}

//
// Programming invalid entries to 0x3F causes Route block to detect an invalid port number
// and flag a PRIV error to the FM. (See Table 14.RID RAM Programming, IAS 3.3.4)
//

#define LWSWITCH_ILWALID_PORT_VAL_LR10   0x3F
#define LWSWITCH_ILWALID_VC_VAL_LR10     0x0

#define LWSWITCH_PORTLIST_PORT_LR10(_entry, _idx) \
    ((_idx < _entry.numEntries) ? _entry.portList[_idx].destPortNum : LWSWITCH_ILWALID_PORT_VAL_LR10)

#define LWSWITCH_PORTLIST_VC_LR10(_entry, _idx) \
    ((_idx < _entry.numEntries) ? _entry.portList[_idx].vcMap : LWSWITCH_ILWALID_VC_VAL_LR10)

/*
 * CTRL_LWSWITCH_SET_ROUTING_ID
 */

static void
_lwswitch_set_routing_id_lr10
(
    lwswitch_device *device,
    LwU32 portNum,
    LwU32 firstIndex,
    LwU32 numEntries,
    LWSWITCH_ROUTING_ID_ENTRY *routing_id
)
{
    LwU32 i;
    LwU32 rmod;

    LWSWITCH_LINK_WR32_LR10(device, portNum, NPORT, _INGRESS, _REQRSPMAPADDR,
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _RAM_ADDRESS, firstIndex) |
        DRF_DEF(_INGRESS, _REQRSPMAPADDR, _RAM_SEL, _SELECTSRIDROUTERAM) |
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _AUTO_INCR, 1));

    for (i = 0; i < numEntries; i++)
    {
        LWSWITCH_LINK_WR32_LR10(device, portNum, NPORT, _INGRESS, _RIDTABDATA1,
            DRF_NUM(_INGRESS, _RIDTABDATA1, _PORT3,    LWSWITCH_PORTLIST_PORT_LR10(routing_id[i], 3)) |
            DRF_NUM(_INGRESS, _RIDTABDATA1, _VC_MODE3, LWSWITCH_PORTLIST_VC_LR10(routing_id[i], 3))   |
            DRF_NUM(_INGRESS, _RIDTABDATA1, _PORT4,    LWSWITCH_PORTLIST_PORT_LR10(routing_id[i], 4)) |
            DRF_NUM(_INGRESS, _RIDTABDATA1, _VC_MODE4, LWSWITCH_PORTLIST_VC_LR10(routing_id[i], 4))   |
            DRF_NUM(_INGRESS, _RIDTABDATA1, _PORT5,    LWSWITCH_PORTLIST_PORT_LR10(routing_id[i], 5)) |
            DRF_NUM(_INGRESS, _RIDTABDATA1, _VC_MODE5, LWSWITCH_PORTLIST_VC_LR10(routing_id[i], 5)));

        LWSWITCH_LINK_WR32_LR10(device, portNum, NPORT, _INGRESS, _RIDTABDATA2,
            DRF_NUM(_INGRESS, _RIDTABDATA2, _PORT6,    LWSWITCH_PORTLIST_PORT_LR10(routing_id[i], 6)) |
            DRF_NUM(_INGRESS, _RIDTABDATA2, _VC_MODE6, LWSWITCH_PORTLIST_VC_LR10(routing_id[i], 6))   |
            DRF_NUM(_INGRESS, _RIDTABDATA2, _PORT7,    LWSWITCH_PORTLIST_PORT_LR10(routing_id[i], 7)) |
            DRF_NUM(_INGRESS, _RIDTABDATA2, _VC_MODE7, LWSWITCH_PORTLIST_VC_LR10(routing_id[i], 7))   |
            DRF_NUM(_INGRESS, _RIDTABDATA2, _PORT8,    LWSWITCH_PORTLIST_PORT_LR10(routing_id[i], 8)) |
            DRF_NUM(_INGRESS, _RIDTABDATA2, _VC_MODE8, LWSWITCH_PORTLIST_VC_LR10(routing_id[i], 8)));

        LWSWITCH_LINK_WR32_LR10(device, portNum, NPORT, _INGRESS, _RIDTABDATA3,
            DRF_NUM(_INGRESS, _RIDTABDATA3, _PORT9,     LWSWITCH_PORTLIST_PORT_LR10(routing_id[i],  9)) |
            DRF_NUM(_INGRESS, _RIDTABDATA3, _VC_MODE9,  LWSWITCH_PORTLIST_VC_LR10(routing_id[i],  9))   |
            DRF_NUM(_INGRESS, _RIDTABDATA3, _PORT10,    LWSWITCH_PORTLIST_PORT_LR10(routing_id[i], 10)) |
            DRF_NUM(_INGRESS, _RIDTABDATA3, _VC_MODE10, LWSWITCH_PORTLIST_VC_LR10(routing_id[i], 10))   |
            DRF_NUM(_INGRESS, _RIDTABDATA3, _PORT11,    LWSWITCH_PORTLIST_PORT_LR10(routing_id[i], 11)) |
            DRF_NUM(_INGRESS, _RIDTABDATA3, _VC_MODE11, LWSWITCH_PORTLIST_VC_LR10(routing_id[i], 11)));

        LWSWITCH_LINK_WR32_LR10(device, portNum, NPORT, _INGRESS, _RIDTABDATA4,
            DRF_NUM(_INGRESS, _RIDTABDATA4, _PORT12,    LWSWITCH_PORTLIST_PORT_LR10(routing_id[i], 12)) |
            DRF_NUM(_INGRESS, _RIDTABDATA4, _VC_MODE12, LWSWITCH_PORTLIST_VC_LR10(routing_id[i], 12))   |
            DRF_NUM(_INGRESS, _RIDTABDATA4, _PORT13,    LWSWITCH_PORTLIST_PORT_LR10(routing_id[i], 13)) |
            DRF_NUM(_INGRESS, _RIDTABDATA4, _VC_MODE13, LWSWITCH_PORTLIST_VC_LR10(routing_id[i], 13))   |
            DRF_NUM(_INGRESS, _RIDTABDATA4, _PORT14,    LWSWITCH_PORTLIST_PORT_LR10(routing_id[i], 14)) |
            DRF_NUM(_INGRESS, _RIDTABDATA4, _VC_MODE14, LWSWITCH_PORTLIST_VC_LR10(routing_id[i], 14)));

        rmod =
            (routing_id[i].useRoutingLan ? LWBIT(6) : 0) |
            (routing_id[i].enableIrlErrResponse ? LWBIT(9) : 0);

        LWSWITCH_LINK_WR32_LR10(device, portNum, NPORT, _INGRESS, _RIDTABDATA5,
            DRF_NUM(_INGRESS, _RIDTABDATA5, _PORT15,    LWSWITCH_PORTLIST_PORT_LR10(routing_id[i], 15)) |
            DRF_NUM(_INGRESS, _RIDTABDATA5, _VC_MODE15, LWSWITCH_PORTLIST_VC_LR10(routing_id[i], 15))   |
            DRF_NUM(_INGRESS, _RIDTABDATA5, _RMOD,      rmod)                                           |
            DRF_NUM(_INGRESS, _RIDTABDATA5, _ACLVALID,  routing_id[i].entryValid));

        LWSWITCH_ASSERT(routing_id[i].numEntries <= 16);
        // Write last and auto-increment
        LWSWITCH_LINK_WR32_LR10(device, portNum, NPORT, _INGRESS, _RIDTABDATA0,
            DRF_NUM(_INGRESS, _RIDTABDATA0, _GSIZE,
                (routing_id[i].numEntries == 16) ? 0x0 : routing_id[i].numEntries) |
            DRF_NUM(_INGRESS, _RIDTABDATA0, _PORT0,    LWSWITCH_PORTLIST_PORT_LR10(routing_id[i], 0)) |
            DRF_NUM(_INGRESS, _RIDTABDATA0, _VC_MODE0, LWSWITCH_PORTLIST_VC_LR10(routing_id[i], 0))   |
            DRF_NUM(_INGRESS, _RIDTABDATA0, _PORT1,    LWSWITCH_PORTLIST_PORT_LR10(routing_id[i], 1)) |
            DRF_NUM(_INGRESS, _RIDTABDATA0, _VC_MODE1, LWSWITCH_PORTLIST_VC_LR10(routing_id[i], 1))   |
            DRF_NUM(_INGRESS, _RIDTABDATA0, _PORT2,    LWSWITCH_PORTLIST_PORT_LR10(routing_id[i], 2)) |
            DRF_NUM(_INGRESS, _RIDTABDATA0, _VC_MODE2, LWSWITCH_PORTLIST_VC_LR10(routing_id[i], 2)));
    }
}

#define LWSWITCH_NUM_RIDTABDATA_REGS_LR10 6

LwlStatus
lwswitch_ctrl_get_routing_id_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_ROUTING_ID_PARAMS *params
)
{
    LWSWITCH_ROUTING_ID_IDX_ENTRY *rid_entries;
    LwU32 table_index;
    LwU32 rid_tab_data[LWSWITCH_NUM_RIDTABDATA_REGS_LR10]; // 6 RID tables
    LwU32 rid_count;
    LwU32 rmod;
    LwU32 gsize;
    LwU32 ram_size;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, NPORT, params->portNum))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: NPORT port #%d not valid\n",
            __FUNCTION__, params->portNum);
        return -LWL_BAD_ARGS;
    }

    ram_size = lwswitch_get_ingress_ram_size(device, LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECTSRIDROUTERAM);
    if (params->firstIndex >= ram_size)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: routingId first index %d out of range[%d..%d].\n",
            __FUNCTION__, params->firstIndex, 0, ram_size - 1);
        return -LWL_BAD_ARGS;
    }

    lwswitch_os_memset(params->entries, 0, sizeof(params->entries));

    table_index = params->firstIndex;
    rid_entries = params->entries;
    rid_count = 0;

    /* set table offset */
    LWSWITCH_LINK_WR32_LR10(device, params->portNum, NPORT, _INGRESS, _REQRSPMAPADDR,
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _RAM_ADDRESS, params->firstIndex) |
        DRF_DEF(_INGRESS, _REQRSPMAPADDR, _RAM_SEL, _SELECTSRIDROUTERAM) |
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _AUTO_INCR, 1));

    while (rid_count < LWSWITCH_ROUTING_ID_ENTRIES_MAX &&
           table_index < ram_size)
    {
        rid_tab_data[0] = LWSWITCH_LINK_RD32_LR10(device, params->portNum, NPORT, _INGRESS, _RIDTABDATA0);
        rid_tab_data[1] = LWSWITCH_LINK_RD32_LR10(device, params->portNum, NPORT, _INGRESS, _RIDTABDATA1);
        rid_tab_data[2] = LWSWITCH_LINK_RD32_LR10(device, params->portNum, NPORT, _INGRESS, _RIDTABDATA2);
        rid_tab_data[3] = LWSWITCH_LINK_RD32_LR10(device, params->portNum, NPORT, _INGRESS, _RIDTABDATA3);
        rid_tab_data[4] = LWSWITCH_LINK_RD32_LR10(device, params->portNum, NPORT, _INGRESS, _RIDTABDATA4);
        rid_tab_data[5] = LWSWITCH_LINK_RD32_LR10(device, params->portNum, NPORT, _INGRESS, _RIDTABDATA5);

        /* add to rid_entries list if nonzero */
        if (rid_tab_data[0] || rid_tab_data[1] || rid_tab_data[2] ||
            rid_tab_data[3] || rid_tab_data[4] || rid_tab_data[5])
        {
            rid_entries[rid_count].entry.portList[0].destPortNum  = DRF_VAL(_INGRESS, _RIDTABDATA0, _PORT0, rid_tab_data[0]);
            rid_entries[rid_count].entry.portList[0].vcMap        = DRF_VAL(_INGRESS, _RIDTABDATA0, _VC_MODE0, rid_tab_data[0]);

            rid_entries[rid_count].entry.portList[1].destPortNum  = DRF_VAL(_INGRESS, _RIDTABDATA0, _PORT1, rid_tab_data[0]);
            rid_entries[rid_count].entry.portList[1].vcMap        = DRF_VAL(_INGRESS, _RIDTABDATA0, _VC_MODE1, rid_tab_data[0]);

            rid_entries[rid_count].entry.portList[2].destPortNum  = DRF_VAL(_INGRESS, _RIDTABDATA0, _PORT2, rid_tab_data[0]);
            rid_entries[rid_count].entry.portList[2].vcMap        = DRF_VAL(_INGRESS, _RIDTABDATA0, _VC_MODE2, rid_tab_data[0]);

            rid_entries[rid_count].entry.portList[3].destPortNum  = DRF_VAL(_INGRESS, _RIDTABDATA1, _PORT3, rid_tab_data[1]);
            rid_entries[rid_count].entry.portList[3].vcMap        = DRF_VAL(_INGRESS, _RIDTABDATA1, _VC_MODE3, rid_tab_data[1]);

            rid_entries[rid_count].entry.portList[4].destPortNum  = DRF_VAL(_INGRESS, _RIDTABDATA1, _PORT4, rid_tab_data[1]);
            rid_entries[rid_count].entry.portList[4].vcMap        = DRF_VAL(_INGRESS, _RIDTABDATA1, _VC_MODE4, rid_tab_data[1]);

            rid_entries[rid_count].entry.portList[5].destPortNum  = DRF_VAL(_INGRESS, _RIDTABDATA1, _PORT5, rid_tab_data[1]);
            rid_entries[rid_count].entry.portList[5].vcMap        = DRF_VAL(_INGRESS, _RIDTABDATA1, _VC_MODE5, rid_tab_data[1]);

            rid_entries[rid_count].entry.portList[6].destPortNum  = DRF_VAL(_INGRESS, _RIDTABDATA2, _PORT6, rid_tab_data[2]);
            rid_entries[rid_count].entry.portList[6].vcMap        = DRF_VAL(_INGRESS, _RIDTABDATA2, _VC_MODE6, rid_tab_data[2]);

            rid_entries[rid_count].entry.portList[7].destPortNum  = DRF_VAL(_INGRESS, _RIDTABDATA2, _PORT7, rid_tab_data[2]);
            rid_entries[rid_count].entry.portList[7].vcMap        = DRF_VAL(_INGRESS, _RIDTABDATA2, _VC_MODE7, rid_tab_data[2]);

            rid_entries[rid_count].entry.portList[8].destPortNum  = DRF_VAL(_INGRESS, _RIDTABDATA2, _PORT8, rid_tab_data[2]);
            rid_entries[rid_count].entry.portList[8].vcMap        = DRF_VAL(_INGRESS, _RIDTABDATA2, _VC_MODE8, rid_tab_data[2]);

            rid_entries[rid_count].entry.portList[9].destPortNum  = DRF_VAL(_INGRESS, _RIDTABDATA3, _PORT9, rid_tab_data[3]);
            rid_entries[rid_count].entry.portList[9].vcMap        = DRF_VAL(_INGRESS, _RIDTABDATA3, _VC_MODE9, rid_tab_data[3]);

            rid_entries[rid_count].entry.portList[10].destPortNum = DRF_VAL(_INGRESS, _RIDTABDATA3, _PORT10, rid_tab_data[3]);
            rid_entries[rid_count].entry.portList[10].vcMap       = DRF_VAL(_INGRESS, _RIDTABDATA3, _VC_MODE10, rid_tab_data[3]);

            rid_entries[rid_count].entry.portList[11].destPortNum = DRF_VAL(_INGRESS, _RIDTABDATA3, _PORT11, rid_tab_data[3]);
            rid_entries[rid_count].entry.portList[11].vcMap       = DRF_VAL(_INGRESS, _RIDTABDATA3, _VC_MODE11, rid_tab_data[3]);

            rid_entries[rid_count].entry.portList[12].destPortNum = DRF_VAL(_INGRESS, _RIDTABDATA4, _PORT12, rid_tab_data[4]);
            rid_entries[rid_count].entry.portList[12].vcMap       = DRF_VAL(_INGRESS, _RIDTABDATA4, _VC_MODE12, rid_tab_data[4]);

            rid_entries[rid_count].entry.portList[13].destPortNum = DRF_VAL(_INGRESS, _RIDTABDATA4, _PORT13, rid_tab_data[4]);
            rid_entries[rid_count].entry.portList[13].vcMap       = DRF_VAL(_INGRESS, _RIDTABDATA4, _VC_MODE13, rid_tab_data[4]);

            rid_entries[rid_count].entry.portList[14].destPortNum = DRF_VAL(_INGRESS, _RIDTABDATA4, _PORT14, rid_tab_data[4]);
            rid_entries[rid_count].entry.portList[14].vcMap       = DRF_VAL(_INGRESS, _RIDTABDATA4, _VC_MODE14, rid_tab_data[4]);

            rid_entries[rid_count].entry.portList[15].destPortNum = DRF_VAL(_INGRESS, _RIDTABDATA5, _PORT15, rid_tab_data[5]);
            rid_entries[rid_count].entry.portList[15].vcMap       = DRF_VAL(_INGRESS, _RIDTABDATA5, _VC_MODE15, rid_tab_data[5]);
            rid_entries[rid_count].entry.entryValid               = DRF_VAL(_INGRESS, _RIDTABDATA5, _ACLVALID, rid_tab_data[5]);

            rmod = DRF_VAL(_INGRESS, _RIDTABDATA5, _RMOD, rid_tab_data[5]);
            rid_entries[rid_count].entry.useRoutingLan = (LWBIT(6) & rmod) ? 1 : 0;
            rid_entries[rid_count].entry.enableIrlErrResponse = (LWBIT(9) & rmod) ? 1 : 0;

            // Gsize of 16 falls into the 0th entry of GLT region. The _GSIZE field must be mapped accordingly
            // to the number of port entries (See IAS, Table 20, Sect 3.4.2.2. Packet Routing).
            gsize = DRF_VAL(_INGRESS, _RIDTABDATA0, _GSIZE, rid_tab_data[0]);
            rid_entries[rid_count].entry.numEntries = ((gsize == 0) ? 16 : gsize);

            rid_entries[rid_count].idx = table_index;
            rid_count++;
        }

        table_index++;
    }

    params->nextIndex = table_index;
    params->numEntries = rid_count;

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_ctrl_set_routing_id_valid_lr10
(
    lwswitch_device *device,
    LWSWITCH_SET_ROUTING_ID_VALID *p
)
{
    LwU32 rid_ctrl;
    LwU32 rid_tab_data0;
    LwU32 rid_tab_data1;
    LwU32 rid_tab_data2;
    LwU32 rid_tab_data3;
    LwU32 rid_tab_data4;
    LwU32 rid_tab_data5;
    LwU32 ram_address = p->firstIndex;
    LwU32 i;
    LwU32 ram_size;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, NPORT, p->portNum))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: NPORT port #%d not valid\n",
            __FUNCTION__, p->portNum);
        return -LWL_BAD_ARGS;
    }

    ram_size = lwswitch_get_ingress_ram_size(device, LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECTSRIDROUTERAM);
    if ((p->firstIndex >= ram_size) ||
        (p->numEntries > LWSWITCH_ROUTING_ID_ENTRIES_MAX) ||
        (p->firstIndex + p->numEntries > ram_size))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: routingId[%d..%d] overflows range %d..%d or size %d.\n",
            __FUNCTION__, p->firstIndex, p->firstIndex + p->numEntries - 1,
            0, ram_size - 1,
            LWSWITCH_ROUTING_ID_ENTRIES_MAX);
        return -LWL_BAD_ARGS;
    }

    // Select RID RAM and disable Auto Increment.
    rid_ctrl =
        DRF_DEF(_INGRESS, _REQRSPMAPADDR, _RAM_SEL, _SELECTSRIDROUTERAM) |
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _AUTO_INCR, 0);


    for (i = 0; i < p->numEntries; i++)
    {
        /* set the ram address */
        rid_ctrl = FLD_SET_DRF_NUM(_INGRESS, _REQRSPMAPADDR, _RAM_ADDRESS, ram_address++, rid_ctrl);
        LWSWITCH_LINK_WR32_LR10(device, p->portNum, NPORT, _INGRESS, _REQRSPMAPADDR, rid_ctrl);

        rid_tab_data0 = LWSWITCH_LINK_RD32_LR10(device, p->portNum, NPORT, _INGRESS, _RIDTABDATA0);
        rid_tab_data1 = LWSWITCH_LINK_RD32_LR10(device, p->portNum, NPORT, _INGRESS, _RIDTABDATA1);
        rid_tab_data2 = LWSWITCH_LINK_RD32_LR10(device, p->portNum, NPORT, _INGRESS, _RIDTABDATA2);
        rid_tab_data3 = LWSWITCH_LINK_RD32_LR10(device, p->portNum, NPORT, _INGRESS, _RIDTABDATA3);
        rid_tab_data4 = LWSWITCH_LINK_RD32_LR10(device, p->portNum, NPORT, _INGRESS, _RIDTABDATA4);
        rid_tab_data5 = LWSWITCH_LINK_RD32_LR10(device, p->portNum, NPORT, _INGRESS, _RIDTABDATA5);

        // Set the valid bit in _RIDTABDATA5
        rid_tab_data5 = FLD_SET_DRF_NUM(_INGRESS, _RIDTABDATA5, _ACLVALID,
            p->entryValid[i], rid_tab_data5);

        LWSWITCH_LINK_WR32_LR10(device, p->portNum, NPORT, _INGRESS, _RIDTABDATA1, rid_tab_data1);
        LWSWITCH_LINK_WR32_LR10(device, p->portNum, NPORT, _INGRESS, _RIDTABDATA2, rid_tab_data2);
        LWSWITCH_LINK_WR32_LR10(device, p->portNum, NPORT, _INGRESS, _RIDTABDATA3, rid_tab_data3);
        LWSWITCH_LINK_WR32_LR10(device, p->portNum, NPORT, _INGRESS, _RIDTABDATA4, rid_tab_data4);
        LWSWITCH_LINK_WR32_LR10(device, p->portNum, NPORT, _INGRESS, _RIDTABDATA5, rid_tab_data5);
        LWSWITCH_LINK_WR32_LR10(device, p->portNum, NPORT, _INGRESS, _RIDTABDATA0, rid_tab_data0);
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_ctrl_set_routing_id_lr10
(
    lwswitch_device *device,
    LWSWITCH_SET_ROUTING_ID *p
)
{
    LwU32 i, j;
    LwlStatus retval = LWL_SUCCESS;
    LwU32 ram_size;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, NPORT, p->portNum))
    {
        LWSWITCH_PRINT(device, ERROR,
            "NPORT port #%d not valid\n",
            p->portNum);
        return -LWL_BAD_ARGS;
    }

    ram_size = lwswitch_get_ingress_ram_size(device, LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECTSRIDROUTERAM);
    if ((p->firstIndex >= ram_size) ||
        (p->numEntries > LWSWITCH_ROUTING_ID_ENTRIES_MAX) ||
        (p->firstIndex + p->numEntries > ram_size))
    {
        LWSWITCH_PRINT(device, ERROR,
            "routingId[%d..%d] overflows range %d..%d or size %d.\n",
            p->firstIndex, p->firstIndex + p->numEntries - 1,
            0, ram_size - 1,
            LWSWITCH_ROUTING_ID_ENTRIES_MAX);
        return -LWL_BAD_ARGS;
    }

    for (i = 0; i < p->numEntries; i++)
    {
        if ((p->routingId[i].numEntries < 1) ||
            (p->routingId[i].numEntries > LWSWITCH_ROUTING_ID_DEST_PORT_LIST_MAX))
        {
            LWSWITCH_PRINT(device, ERROR,
                "routingId[%d].portList[] size %d overflows range %d..%d\n",
                i, p->routingId[i].numEntries,
                1, LWSWITCH_ROUTING_ID_DEST_PORT_LIST_MAX);
            return -LWL_BAD_ARGS;
        }

        for (j = 0; j < p->routingId[i].numEntries; j++)
        {
            if (p->routingId[i].portList[j].vcMap > DRF_MASK(LW_INGRESS_RIDTABDATA0_VC_MODE0))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "routingId[%d].portList[%d] vcMap 0x%x out of valid range (0x%x..0x%x)\n",
                    i, j,
                    p->routingId[i].portList[j].vcMap,
                    0, DRF_MASK(LW_INGRESS_RIDTABDATA0_VC_MODE0));
                return -LWL_BAD_ARGS;
            }

            if (p->routingId[i].portList[j].destPortNum > DRF_MASK(LW_INGRESS_RIDTABDATA0_PORT0))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "routingId[%d].portList[%d] destPortNum 0x%x out of valid range (0x%x..0x%x)\n",
                    i, j,
                    p->routingId[i].portList[j].destPortNum,
                    0, DRF_MASK(LW_INGRESS_RIDTABDATA0_PORT0));
                return -LWL_BAD_ARGS;
            }
        }
    }

    _lwswitch_set_routing_id_lr10(device, p->portNum, p->firstIndex, p->numEntries, p->routingId);

    return retval;
}

/*
 * CTRL_LWSWITCH_SET_ROUTING_LAN
 */

//
// Check the data field is present in the list.  Return either the data field
// or default if not present.
//
#define LWSWITCH_PORTLIST_VALID_LR10(_entry, _idx, _field, _default) \
    ((_idx < _entry.numEntries) ? _entry.portList[_idx]._field  : _default)

static void
_lwswitch_set_routing_lan_lr10
(
    lwswitch_device *device,
    LwU32 portNum,
    LwU32 firstIndex,
    LwU32 numEntries,
    LWSWITCH_ROUTING_LAN_ENTRY *routing_lan
)
{
    LwU32 i;

    LWSWITCH_LINK_WR32_LR10(device, portNum, NPORT, _INGRESS, _REQRSPMAPADDR,
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _RAM_ADDRESS, firstIndex) |
        DRF_DEF(_INGRESS, _REQRSPMAPADDR, _RAM_SEL, _SELECTSRLANROUTERAM) |
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _AUTO_INCR, 1));

    for (i = 0; i < numEntries; i++)
    {
        //
        // NOTE: The GRP_SIZE field is 4-bits.  A subgroup is size 1 through 16
        // with encoding 0x0=16 and 0x1=1, ..., 0xF=15.
        // Programming of GRP_SIZE takes advantage of the inherent masking of
        // DRF_NUM to truncate 16 to 0.
        // See bug #3300673
        //

        LWSWITCH_LINK_WR32_LR10(device, portNum, NPORT, _INGRESS, _RLANTABDATA1,
            DRF_NUM(_INGRESS, _RLANTABDATA1, _GRP_SEL_3, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 3, groupSelect, 0)) |
            DRF_NUM(_INGRESS, _RLANTABDATA1, _GRP_SIZE_3, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 3, groupSize, 1)) |
            DRF_NUM(_INGRESS, _RLANTABDATA1, _GRP_SEL_4, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 4, groupSelect, 0)) |
            DRF_NUM(_INGRESS, _RLANTABDATA1, _GRP_SIZE_4, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 4, groupSize, 1)) |
            DRF_NUM(_INGRESS, _RLANTABDATA1, _GRP_SEL_5, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 5, groupSelect, 0)) |
            DRF_NUM(_INGRESS, _RLANTABDATA1, _GRP_SIZE_5, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 5, groupSize, 1)));

        LWSWITCH_LINK_WR32_LR10(device, portNum, NPORT, _INGRESS, _RLANTABDATA2,
            DRF_NUM(_INGRESS, _RLANTABDATA2, _GRP_SEL_6, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 6, groupSelect, 0)) |
            DRF_NUM(_INGRESS, _RLANTABDATA2, _GRP_SIZE_6, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 6, groupSize, 1)) |
            DRF_NUM(_INGRESS, _RLANTABDATA2, _GRP_SEL_7, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 7, groupSelect, 0)) |
            DRF_NUM(_INGRESS, _RLANTABDATA2, _GRP_SIZE_7, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 7, groupSize, 1)) |
            DRF_NUM(_INGRESS, _RLANTABDATA2, _GRP_SEL_8, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 8, groupSelect, 0)) |
            DRF_NUM(_INGRESS, _RLANTABDATA2, _GRP_SIZE_8, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 8, groupSize, 1)));

        LWSWITCH_LINK_WR32_LR10(device, portNum, NPORT, _INGRESS, _RLANTABDATA3,
            DRF_NUM(_INGRESS, _RLANTABDATA3, _GRP_SEL_9, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 9, groupSelect, 0)) |
            DRF_NUM(_INGRESS, _RLANTABDATA3, _GRP_SIZE_9, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 9, groupSize, 1)) |
            DRF_NUM(_INGRESS, _RLANTABDATA3, _GRP_SEL_10, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 10, groupSelect, 0)) |
            DRF_NUM(_INGRESS, _RLANTABDATA3, _GRP_SIZE_10, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 10, groupSize, 1)) |
            DRF_NUM(_INGRESS, _RLANTABDATA3, _GRP_SEL_11, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 11, groupSelect, 0)) |
            DRF_NUM(_INGRESS, _RLANTABDATA3, _GRP_SIZE_11, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 11, groupSize, 1)));

        LWSWITCH_LINK_WR32_LR10(device, portNum, NPORT, _INGRESS, _RLANTABDATA4,
            DRF_NUM(_INGRESS, _RLANTABDATA4, _GRP_SEL_12, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 12, groupSelect, 0)) |
            DRF_NUM(_INGRESS, _RLANTABDATA4, _GRP_SIZE_12, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 12, groupSize, 1)) |
            DRF_NUM(_INGRESS, _RLANTABDATA4, _GRP_SEL_13, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 13, groupSelect, 0)) |
            DRF_NUM(_INGRESS, _RLANTABDATA4, _GRP_SIZE_13, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 13, groupSize, 1)) |
            DRF_NUM(_INGRESS, _RLANTABDATA4, _GRP_SEL_14, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 14, groupSelect, 0)) |
            DRF_NUM(_INGRESS, _RLANTABDATA4, _GRP_SIZE_14, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 14, groupSize, 1)));

        LWSWITCH_LINK_WR32_LR10(device, portNum, NPORT, _INGRESS, _RLANTABDATA5,
            DRF_NUM(_INGRESS, _RLANTABDATA5, _GRP_SEL_15, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 15, groupSelect, 0)) |
            DRF_NUM(_INGRESS, _RLANTABDATA5, _GRP_SIZE_15, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 15, groupSize, 1)) |
            DRF_NUM(_INGRESS, _RLANTABDATA5, _ACLVALID,  routing_lan[i].entryValid));

        // Write last and auto-increment
        LWSWITCH_LINK_WR32_LR10(device, portNum, NPORT, _INGRESS, _RLANTABDATA0,
            DRF_NUM(_INGRESS, _RLANTABDATA0, _GRP_SEL_0, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 0, groupSelect, 0)) |
            DRF_NUM(_INGRESS, _RLANTABDATA0, _GRP_SIZE_0, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 0, groupSize, 1)) |
            DRF_NUM(_INGRESS, _RLANTABDATA0, _GRP_SEL_1, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 1, groupSelect, 0)) |
            DRF_NUM(_INGRESS, _RLANTABDATA0, _GRP_SIZE_1, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 1, groupSize, 1)) |
            DRF_NUM(_INGRESS, _RLANTABDATA0, _GRP_SEL_2, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 2, groupSelect, 0)) |
            DRF_NUM(_INGRESS, _RLANTABDATA0, _GRP_SIZE_2, LWSWITCH_PORTLIST_VALID_LR10(routing_lan[i], 2, groupSize, 1)));
    }
}

LwlStatus
lwswitch_ctrl_set_routing_lan_lr10
(
    lwswitch_device *device,
    LWSWITCH_SET_ROUTING_LAN *p
)
{
    LwU32 i, j;
    LwlStatus retval = LWL_SUCCESS;
    LwU32 ram_size;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, NPORT, p->portNum))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: NPORT port #%d not valid\n",
            __FUNCTION__, p->portNum);
        return -LWL_BAD_ARGS;
    }

    ram_size = lwswitch_get_ingress_ram_size(device, LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECTSRLANROUTERAM);
    if ((p->firstIndex >= ram_size) ||
        (p->numEntries > LWSWITCH_ROUTING_LAN_ENTRIES_MAX) ||
        (p->firstIndex + p->numEntries > ram_size))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: routingLan[%d..%d] overflows range %d..%d or size %d.\n",
            __FUNCTION__, p->firstIndex, p->firstIndex + p->numEntries - 1,
            0, ram_size - 1,
            LWSWITCH_ROUTING_LAN_ENTRIES_MAX);
        return -LWL_BAD_ARGS;
    }

    for (i = 0; i < p->numEntries; i++)
    {
        if (p->routingLan[i].numEntries > LWSWITCH_ROUTING_LAN_GROUP_SEL_MAX)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: routingLan[%d].portList[] size %d overflows range %d..%d\n",
                __FUNCTION__, i, p->routingLan[i].numEntries,
                0, LWSWITCH_ROUTING_LAN_GROUP_SEL_MAX);
            return -LWL_BAD_ARGS;
        }

        for (j = 0; j < p->routingLan[i].numEntries; j++)
        {
            if (p->routingLan[i].portList[j].groupSelect > DRF_MASK(LW_INGRESS_RLANTABDATA0_GRP_SEL_0))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: routingLan[%d].portList[%d] groupSelect 0x%x out of valid range (0x%x..0x%x)\n",
                    __FUNCTION__, i, j,
                    p->routingLan[i].portList[j].groupSelect,
                    0, DRF_MASK(LW_INGRESS_RLANTABDATA0_GRP_SEL_0));
                return -LWL_BAD_ARGS;
            }

            if ((p->routingLan[i].portList[j].groupSize == 0) ||
                (p->routingLan[i].portList[j].groupSize > DRF_MASK(LW_INGRESS_RLANTABDATA0_GRP_SIZE_0) + 1))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: routingLan[%d].portList[%d] groupSize 0x%x out of valid range (0x%x..0x%x)\n",
                    __FUNCTION__, i, j,
                    p->routingLan[i].portList[j].groupSize,
                    1, DRF_MASK(LW_INGRESS_RLANTABDATA0_GRP_SIZE_0) + 1);
                return -LWL_BAD_ARGS;
            }
        }
    }

    _lwswitch_set_routing_lan_lr10(device, p->portNum, p->firstIndex, p->numEntries, p->routingLan);

    return retval;
}

#define LWSWITCH_NUM_RLANTABDATA_REGS_LR10 6

LwlStatus
lwswitch_ctrl_get_routing_lan_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_ROUTING_LAN_PARAMS *params
)
{
    LWSWITCH_ROUTING_LAN_IDX_ENTRY *rlan_entries;
    LwU32 table_index;
    LwU32 rlan_tab_data[LWSWITCH_NUM_RLANTABDATA_REGS_LR10]; // 6 RLAN tables
    LwU32 rlan_count;
    LwU32 ram_size;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, NPORT, params->portNum))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: NPORT port #%d not valid\n",
            __FUNCTION__, params->portNum);
        return -LWL_BAD_ARGS;
    }

    ram_size = lwswitch_get_ingress_ram_size(device, LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECTSRLANROUTERAM);
    if ((params->firstIndex >= ram_size))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: routingLan first index %d out of range[%d..%d].\n",
            __FUNCTION__, params->firstIndex, 0, ram_size - 1);
        return -LWL_BAD_ARGS;
    }

    lwswitch_os_memset(params->entries, 0, (LWSWITCH_ROUTING_LAN_ENTRIES_MAX *
        sizeof(LWSWITCH_ROUTING_LAN_IDX_ENTRY)));

    table_index = params->firstIndex;
    rlan_entries = params->entries;
    rlan_count = 0;

    /* set table offset */
    LWSWITCH_LINK_WR32_LR10(device, params->portNum, NPORT, _INGRESS, _REQRSPMAPADDR,
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _RAM_ADDRESS, params->firstIndex) |
        DRF_DEF(_INGRESS, _REQRSPMAPADDR, _RAM_SEL, _SELECTSRLANROUTERAM)   |
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _AUTO_INCR, 1));

    while (rlan_count < LWSWITCH_ROUTING_LAN_ENTRIES_MAX &&
           table_index < ram_size)
    {
        /* read one entry */
        rlan_tab_data[0] = LWSWITCH_LINK_RD32_LR10(device, params->portNum, NPORT, _INGRESS, _RLANTABDATA0);
        rlan_tab_data[1] = LWSWITCH_LINK_RD32_LR10(device, params->portNum, NPORT, _INGRESS, _RLANTABDATA1);
        rlan_tab_data[2] = LWSWITCH_LINK_RD32_LR10(device, params->portNum, NPORT, _INGRESS, _RLANTABDATA2);
        rlan_tab_data[3] = LWSWITCH_LINK_RD32_LR10(device, params->portNum, NPORT, _INGRESS, _RLANTABDATA3);
        rlan_tab_data[4] = LWSWITCH_LINK_RD32_LR10(device, params->portNum, NPORT, _INGRESS, _RLANTABDATA4);
        rlan_tab_data[5] = LWSWITCH_LINK_RD32_LR10(device, params->portNum, NPORT, _INGRESS, _RLANTABDATA5);

        /* add to rlan_entries list if nonzero */
        if (rlan_tab_data[0] || rlan_tab_data[1] || rlan_tab_data[2] ||
            rlan_tab_data[3] || rlan_tab_data[4] || rlan_tab_data[5])
        {
            rlan_entries[rlan_count].entry.portList[0].groupSelect = DRF_VAL(_INGRESS, _RLANTABDATA0, _GRP_SEL_0, rlan_tab_data[0]);
            rlan_entries[rlan_count].entry.portList[0].groupSize   = DRF_VAL(_INGRESS, _RLANTABDATA0, _GRP_SIZE_0, rlan_tab_data[0]);
            if (rlan_entries[rlan_count].entry.portList[0].groupSize == 0)
            {
                rlan_entries[rlan_count].entry.portList[0].groupSize = 16;
            }

            rlan_entries[rlan_count].entry.portList[1].groupSelect = DRF_VAL(_INGRESS, _RLANTABDATA0, _GRP_SEL_1, rlan_tab_data[0]);
            rlan_entries[rlan_count].entry.portList[1].groupSize   = DRF_VAL(_INGRESS, _RLANTABDATA0, _GRP_SIZE_1, rlan_tab_data[0]);
            if (rlan_entries[rlan_count].entry.portList[1].groupSize == 0)
            {
                rlan_entries[rlan_count].entry.portList[1].groupSize = 16;
            }

            rlan_entries[rlan_count].entry.portList[2].groupSelect = DRF_VAL(_INGRESS, _RLANTABDATA0, _GRP_SEL_2, rlan_tab_data[0]);
            rlan_entries[rlan_count].entry.portList[2].groupSize   = DRF_VAL(_INGRESS, _RLANTABDATA0, _GRP_SIZE_2, rlan_tab_data[0]);
            if (rlan_entries[rlan_count].entry.portList[2].groupSize == 0)
            {
                rlan_entries[rlan_count].entry.portList[2].groupSize = 16;
            }

            rlan_entries[rlan_count].entry.portList[3].groupSelect = DRF_VAL(_INGRESS, _RLANTABDATA1, _GRP_SEL_3, rlan_tab_data[1]);
            rlan_entries[rlan_count].entry.portList[3].groupSize   = DRF_VAL(_INGRESS, _RLANTABDATA1, _GRP_SIZE_3, rlan_tab_data[1]);
            if (rlan_entries[rlan_count].entry.portList[3].groupSize == 0)
            {
                rlan_entries[rlan_count].entry.portList[3].groupSize = 16;
            }

            rlan_entries[rlan_count].entry.portList[4].groupSelect = DRF_VAL(_INGRESS, _RLANTABDATA1, _GRP_SEL_4, rlan_tab_data[1]);
            rlan_entries[rlan_count].entry.portList[4].groupSize   = DRF_VAL(_INGRESS, _RLANTABDATA1, _GRP_SIZE_4, rlan_tab_data[1]);
            if (rlan_entries[rlan_count].entry.portList[4].groupSize == 0)
            {
                rlan_entries[rlan_count].entry.portList[4].groupSize = 16;
            }

            rlan_entries[rlan_count].entry.portList[5].groupSelect = DRF_VAL(_INGRESS, _RLANTABDATA1, _GRP_SEL_5, rlan_tab_data[1]);
            rlan_entries[rlan_count].entry.portList[5].groupSize   = DRF_VAL(_INGRESS, _RLANTABDATA1, _GRP_SIZE_5, rlan_tab_data[1]);
            if (rlan_entries[rlan_count].entry.portList[5].groupSize == 0)
            {
                rlan_entries[rlan_count].entry.portList[5].groupSize = 16;
            }

            rlan_entries[rlan_count].entry.portList[6].groupSelect = DRF_VAL(_INGRESS, _RLANTABDATA2, _GRP_SEL_6, rlan_tab_data[2]);
            rlan_entries[rlan_count].entry.portList[6].groupSize   = DRF_VAL(_INGRESS, _RLANTABDATA2, _GRP_SIZE_6, rlan_tab_data[2]);
            if (rlan_entries[rlan_count].entry.portList[6].groupSize == 0)
            {
                rlan_entries[rlan_count].entry.portList[6].groupSize = 16;
            }

            rlan_entries[rlan_count].entry.portList[7].groupSelect = DRF_VAL(_INGRESS, _RLANTABDATA2, _GRP_SEL_7, rlan_tab_data[2]);
            rlan_entries[rlan_count].entry.portList[7].groupSize   = DRF_VAL(_INGRESS, _RLANTABDATA2, _GRP_SIZE_7, rlan_tab_data[2]);
            if (rlan_entries[rlan_count].entry.portList[7].groupSize == 0)
            {
                rlan_entries[rlan_count].entry.portList[7].groupSize = 16;
            }

            rlan_entries[rlan_count].entry.portList[8].groupSelect = DRF_VAL(_INGRESS, _RLANTABDATA2, _GRP_SEL_8, rlan_tab_data[2]);
            rlan_entries[rlan_count].entry.portList[8].groupSize   = DRF_VAL(_INGRESS, _RLANTABDATA2, _GRP_SIZE_8, rlan_tab_data[2]);
            if (rlan_entries[rlan_count].entry.portList[8].groupSize == 0)
            {
                rlan_entries[rlan_count].entry.portList[8].groupSize = 16;
            }

            rlan_entries[rlan_count].entry.portList[9].groupSelect = DRF_VAL(_INGRESS, _RLANTABDATA3, _GRP_SEL_9, rlan_tab_data[3]);
            rlan_entries[rlan_count].entry.portList[9].groupSize   = DRF_VAL(_INGRESS, _RLANTABDATA3, _GRP_SIZE_9, rlan_tab_data[3]);
            if (rlan_entries[rlan_count].entry.portList[9].groupSize == 0)
            {
                rlan_entries[rlan_count].entry.portList[9].groupSize = 16;
            }

            rlan_entries[rlan_count].entry.portList[10].groupSelect = DRF_VAL(_INGRESS, _RLANTABDATA3, _GRP_SEL_10, rlan_tab_data[3]);
            rlan_entries[rlan_count].entry.portList[10].groupSize   = DRF_VAL(_INGRESS, _RLANTABDATA3, _GRP_SIZE_10, rlan_tab_data[3]);
            if (rlan_entries[rlan_count].entry.portList[10].groupSize == 0)
            {
                rlan_entries[rlan_count].entry.portList[10].groupSize = 16;
            }

            rlan_entries[rlan_count].entry.portList[11].groupSelect = DRF_VAL(_INGRESS, _RLANTABDATA3, _GRP_SEL_11, rlan_tab_data[3]);
            rlan_entries[rlan_count].entry.portList[11].groupSize   = DRF_VAL(_INGRESS, _RLANTABDATA3, _GRP_SIZE_11, rlan_tab_data[3]);
            if (rlan_entries[rlan_count].entry.portList[11].groupSize == 0)
            {
                rlan_entries[rlan_count].entry.portList[11].groupSize = 16;
            }

            rlan_entries[rlan_count].entry.portList[12].groupSelect = DRF_VAL(_INGRESS, _RLANTABDATA4, _GRP_SEL_12, rlan_tab_data[4]);
            rlan_entries[rlan_count].entry.portList[12].groupSize   = DRF_VAL(_INGRESS, _RLANTABDATA4, _GRP_SIZE_12, rlan_tab_data[4]);
            if (rlan_entries[rlan_count].entry.portList[12].groupSize == 0)
            {
                rlan_entries[rlan_count].entry.portList[12].groupSize = 16;
            }

            rlan_entries[rlan_count].entry.portList[13].groupSelect = DRF_VAL(_INGRESS, _RLANTABDATA4, _GRP_SEL_13, rlan_tab_data[4]);
            rlan_entries[rlan_count].entry.portList[13].groupSize   = DRF_VAL(_INGRESS, _RLANTABDATA4, _GRP_SIZE_13, rlan_tab_data[4]);
            if (rlan_entries[rlan_count].entry.portList[13].groupSize == 0)
            {
                rlan_entries[rlan_count].entry.portList[13].groupSize = 16;
            }

            rlan_entries[rlan_count].entry.portList[14].groupSelect = DRF_VAL(_INGRESS, _RLANTABDATA4, _GRP_SEL_14, rlan_tab_data[4]);
            rlan_entries[rlan_count].entry.portList[14].groupSize   = DRF_VAL(_INGRESS, _RLANTABDATA4, _GRP_SIZE_14, rlan_tab_data[4]);
            if (rlan_entries[rlan_count].entry.portList[14].groupSize == 0)
            {
                rlan_entries[rlan_count].entry.portList[14].groupSize = 16;
            }

            rlan_entries[rlan_count].entry.portList[15].groupSelect = DRF_VAL(_INGRESS, _RLANTABDATA5, _GRP_SEL_15, rlan_tab_data[5]);
            rlan_entries[rlan_count].entry.portList[15].groupSize   = DRF_VAL(_INGRESS, _RLANTABDATA5, _GRP_SIZE_15, rlan_tab_data[5]);
            if (rlan_entries[rlan_count].entry.portList[15].groupSize == 0)
            {
                rlan_entries[rlan_count].entry.portList[15].groupSize = 16;
            }

            rlan_entries[rlan_count].entry.entryValid               = DRF_VAL(_INGRESS, _RLANTABDATA5, _ACLVALID, rlan_tab_data[5]);
            rlan_entries[rlan_count].entry.numEntries = LWSWITCH_ROUTING_ID_DEST_PORT_LIST_MAX;
            rlan_entries[rlan_count].idx  = table_index;

            rlan_count++;
        }

        table_index++;
    }

    params->nextIndex  = table_index;
    params->numEntries = rlan_count;

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_ctrl_set_routing_lan_valid_lr10
(
    lwswitch_device *device,
    LWSWITCH_SET_ROUTING_LAN_VALID *p
)
{
    LwU32 rlan_ctrl;
    LwU32 rlan_tab_data[LWSWITCH_NUM_RLANTABDATA_REGS_LR10]; // 6 RLAN tables
    LwU32 ram_address = p->firstIndex;
    LwU32 i;
    LwU32 ram_size;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, NPORT, p->portNum))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: NPORT port #%d not valid\n",
            __FUNCTION__, p->portNum);
        return -LWL_BAD_ARGS;
    }

    ram_size = lwswitch_get_ingress_ram_size(device, LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECTSRLANROUTERAM);
    if ((p->firstIndex >= ram_size) ||
        (p->numEntries > LWSWITCH_ROUTING_LAN_ENTRIES_MAX) ||
        (p->firstIndex + p->numEntries > ram_size))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: routingLan[%d..%d] overflows range %d..%d or size %d.\n",
            __FUNCTION__, p->firstIndex, p->firstIndex + p->numEntries - 1,
            0, ram_size - 1,
            LWSWITCH_ROUTING_LAN_ENTRIES_MAX);
        return -LWL_BAD_ARGS;
    }

    // Select RLAN RAM and disable Auto Increament.
    rlan_ctrl =
        DRF_DEF(_INGRESS, _REQRSPMAPADDR, _RAM_SEL, _SELECTSRLANROUTERAM) |
        DRF_NUM(_INGRESS, _REQRSPMAPADDR, _AUTO_INCR, 0);

    for (i = 0; i < p->numEntries; i++)
    {
        /* set the RAM address */
        rlan_ctrl = FLD_SET_DRF_NUM(_INGRESS, _REQRSPMAPADDR, _RAM_ADDRESS, ram_address++, rlan_ctrl);
        LWSWITCH_LINK_WR32_LR10(device, p->portNum, NPORT, _INGRESS, _REQRSPMAPADDR, rlan_ctrl);

        rlan_tab_data[0] = LWSWITCH_LINK_RD32_LR10(device, p->portNum, NPORT, _INGRESS, _RLANTABDATA0);
        rlan_tab_data[1] = LWSWITCH_LINK_RD32_LR10(device, p->portNum, NPORT, _INGRESS, _RLANTABDATA1);
        rlan_tab_data[2] = LWSWITCH_LINK_RD32_LR10(device, p->portNum, NPORT, _INGRESS, _RLANTABDATA2);
        rlan_tab_data[3] = LWSWITCH_LINK_RD32_LR10(device, p->portNum, NPORT, _INGRESS, _RLANTABDATA3);
        rlan_tab_data[4] = LWSWITCH_LINK_RD32_LR10(device, p->portNum, NPORT, _INGRESS, _RLANTABDATA4);
        rlan_tab_data[5] = LWSWITCH_LINK_RD32_LR10(device, p->portNum, NPORT, _INGRESS, _RLANTABDATA5);

        // Set the valid bit in _RLANTABDATA5
        rlan_tab_data[5] = FLD_SET_DRF_NUM(_INGRESS, _RLANTABDATA5, _ACLVALID,
            p->entryValid[i], rlan_tab_data[5]);

        LWSWITCH_LINK_WR32_LR10(device, p->portNum, NPORT, _INGRESS, _RLANTABDATA1, rlan_tab_data[1]);
        LWSWITCH_LINK_WR32_LR10(device, p->portNum, NPORT, _INGRESS, _RLANTABDATA2, rlan_tab_data[2]);
        LWSWITCH_LINK_WR32_LR10(device, p->portNum, NPORT, _INGRESS, _RLANTABDATA3, rlan_tab_data[3]);
        LWSWITCH_LINK_WR32_LR10(device, p->portNum, NPORT, _INGRESS, _RLANTABDATA4, rlan_tab_data[4]);
        LWSWITCH_LINK_WR32_LR10(device, p->portNum, NPORT, _INGRESS, _RLANTABDATA5, rlan_tab_data[5]);
        LWSWITCH_LINK_WR32_LR10(device, p->portNum, NPORT, _INGRESS, _RLANTABDATA0, rlan_tab_data[0]);
    }

    return LWL_SUCCESS;
}

/*
 * @Brief : Send priv ring command and wait for completion
 *
 * @Description :
 *
 * @param[in] device        a reference to the device to initialize
 * @param[in] cmd           encoded priv ring command
 */
LwlStatus
lwswitch_ring_master_cmd_lr10
(
    lwswitch_device *device,
    LwU32 cmd
)
{
    LwU32 value;
    LWSWITCH_TIMEOUT timeout;
    LwBool           keepPolling;

    LWSWITCH_REG_WR32(device, _PPRIV_MASTER, _RING_COMMAND, cmd);

    lwswitch_timeout_create(LWSWITCH_INTERVAL_5MSEC_IN_NS, &timeout);
    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

        value = LWSWITCH_REG_RD32(device, _PPRIV_MASTER, _RING_COMMAND);
        if (FLD_TEST_DRF(_PPRIV_MASTER, _RING_COMMAND, _CMD, _NO_CMD, value))
        {
            break;
        }

        lwswitch_os_sleep(1);
    }
    while (keepPolling);

    if (!FLD_TEST_DRF(_PPRIV_MASTER, _RING_COMMAND, _CMD, _NO_CMD, value))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Timeout waiting for RING_COMMAND == NO_CMD (cmd=0x%x).\n",
            __FUNCTION__, cmd);
        return -LWL_INITIALIZATION_TOTAL_FAILURE;
    }

    return LWL_SUCCESS;
}

/*
 * @brief Process the information read from ROM tables and apply it to device
 * settings.
 *
 * @param[in] device    a reference to the device to query
 * @param[in] firmware  Information parsed from ROM tables
 */
static void
_lwswitch_process_firmware_info_lr10
(
    lwswitch_device *device,
    LWSWITCH_FIRMWARE *firmware
)
{
    LwU32 idx_link;
    LwU64 link_enable_mask;

    if (device->firmware.firmware_size == 0)
    {
        return;
    }

    if (device->firmware.lwlink.link_config_found)
    {
        link_enable_mask = ((LwU64)device->regkeys.link_enable_mask2 << 32 |
                            (LwU64)device->regkeys.link_enable_mask);
        //
        // If the link enables were not already overridden by regkey, then
        // apply the ROM link enables
        //
        if (link_enable_mask == LW_U64_MAX)
        {
            for (idx_link = 0; idx_link < lwswitch_get_num_links(device); idx_link++)
            {
                if ((device->firmware.lwlink.link_enable_mask & LWBIT64(idx_link)) == 0)
                {
                    device->link[idx_link].valid = LW_FALSE;
                }
            }
        }
    }
}

static void
_lwswitch_init_npg_multicast_lr10
(
    lwswitch_device *device
)
{
    LwU32 idx_npg;
    LwU32 idx_nport;
    LwU32 nport_mask;

    //
    // Walk the NPGs and build the mask of extant NPORTs
    //
    for (idx_npg = 0; idx_npg < LWSWITCH_ENG_COUNT(device, NPG, ); idx_npg++)
    {
        if (LWSWITCH_ENG_IS_VALID(device, NPG, idx_npg))
        {
            nport_mask = 0;
            for (idx_nport = 0; idx_nport < LWSWITCH_NPORT_PER_NPG; idx_nport++)
            {
                nport_mask |=
                    (LWSWITCH_ENG_IS_VALID(device, NPORT, idx_npg*LWSWITCH_NPORT_PER_NPG + idx_nport) ?
                    LWBIT(idx_nport) : 0x0);
            }

            LWSWITCH_NPG_WR32_LR10(device, idx_npg,
                _NPG, _CTRL_PRI_MULTICAST,
                DRF_NUM(_NPG, _CTRL_PRI_MULTICAST, _NPORT_ENABLE, nport_mask) |
                DRF_DEF(_NPG, _CTRL_PRI_MULTICAST, _READ_MODE, _AND_ALL_BUSSES));

            LWSWITCH_NPGPERF_WR32_LR10(device, idx_npg,
                _NPGPERF, _CTRL_PRI_MULTICAST,
                DRF_NUM(_NPGPERF, _CTRL_PRI_MULTICAST, _NPORT_ENABLE, nport_mask) |
                DRF_DEF(_NPGPERF, _CTRL_PRI_MULTICAST, _READ_MODE, _AND_ALL_BUSSES));
        }
    }
}

static LwlStatus
lwswitch_clear_nport_rams_lr10
(
    lwswitch_device *device
)
{
    LwU32 idx_nport;
    LwU64 nport_mask = 0;
    LwU32 zero_init_mask;
    LwU32 val;
    LWSWITCH_TIMEOUT timeout;
    LwBool           keepPolling;
    LwlStatus retval = LWL_SUCCESS;

    // Build the mask of available NPORTs
    for (idx_nport = 0; idx_nport < LWSWITCH_ENG_COUNT(device, NPORT, ); idx_nport++)
    {
        if (LWSWITCH_ENG_IS_VALID(device, NPORT, idx_nport))
        {
            nport_mask |= LWBIT64(idx_nport);
        }
    }

    // Start the HW zero init
    zero_init_mask =
        DRF_DEF(_NPORT, _INITIALIZATION, _TAGPOOLINIT_0, _HWINIT) |
        DRF_DEF(_NPORT, _INITIALIZATION, _TAGPOOLINIT_1, _HWINIT) |
        DRF_DEF(_NPORT, _INITIALIZATION, _TAGPOOLINIT_2, _HWINIT) |
        DRF_DEF(_NPORT, _INITIALIZATION, _TAGPOOLINIT_3, _HWINIT) |
        DRF_DEF(_NPORT, _INITIALIZATION, _TAGPOOLINIT_4, _HWINIT) |
        DRF_DEF(_NPORT, _INITIALIZATION, _TAGPOOLINIT_5, _HWINIT) |
        DRF_DEF(_NPORT, _INITIALIZATION, _TAGPOOLINIT_6, _HWINIT) |
        DRF_DEF(_NPORT, _INITIALIZATION, _LINKTABLEINIT, _HWINIT) |
        DRF_DEF(_NPORT, _INITIALIZATION, _REMAPTABINIT,  _HWINIT) |
        DRF_DEF(_NPORT, _INITIALIZATION, _RIDTABINIT,    _HWINIT) |
        DRF_DEF(_NPORT, _INITIALIZATION, _RLANTABINIT,   _HWINIT);

    LWSWITCH_BCAST_WR32_LR10(device, NPORT, _NPORT, _INITIALIZATION,
        zero_init_mask);

    lwswitch_timeout_create(25*LWSWITCH_INTERVAL_1MSEC_IN_NS, &timeout);

    do
    {
        keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

        // Check each enabled NPORT that is still pending until all are done
        for (idx_nport = 0; idx_nport < LWSWITCH_ENG_COUNT(device, NPORT, ); idx_nport++)
        {
            if (LWSWITCH_ENG_IS_VALID(device, NPORT, idx_nport) && (nport_mask & LWBIT64(idx_nport)))
            {
                val = LWSWITCH_ENG_RD32_LR10(device, NPORT, idx_nport, _NPORT, _INITIALIZATION);
                if (val == zero_init_mask)
                {
                    nport_mask &= ~LWBIT64(idx_nport);
                }
            }
        }

        if (nport_mask == 0)
        {
            break;
        }

        lwswitch_os_sleep(1);
    }
    while (keepPolling);

    if (nport_mask != 0)
    {
        LWSWITCH_PRINT(device, WARN,
            "%s: Timeout waiting for LW_NPORT_INITIALIZATION (0x%llx)\n",
            __FUNCTION__, nport_mask);
        // Bug 2974064: Review this timeout handling (fall through)
        retval = -LWL_ERR_ILWALID_STATE;
    }

    //bug 2737147 requires SW To init this crumbstore setting for LR10
    val = DRF_NUM(_TSTATE, _RAM_ADDRESS, _ADDR, 0)             |
          DRF_DEF(_TSTATE, _RAM_ADDRESS, _SELECT, _CRUMBSTORE_RAM) |
          DRF_NUM(_TSTATE, _RAM_ADDRESS, _AUTO_INCR, 0)        |
          DRF_DEF(_TSTATE, _RAM_ADDRESS, _VC, _VC5_TRANSDONE);

    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _TSTATE, _RAM_ADDRESS, val);

    return retval;
}

static void
_lwswitch_init_nport_ecc_control_lr10
(
    lwswitch_device *device
)
{
    // Set ingress ECC error limits
    LWSWITCH_BCAST_WR32_LR10(device, NPORT, _INGRESS, _ERR_NCISOC_HDR_ECC_ERROR_COUNTER,
        DRF_NUM(_INGRESS, _ERR_NCISOC_HDR_ECC_ERROR_COUNTER, _ERROR_COUNT, 0x0));
    LWSWITCH_BCAST_WR32_LR10(device, NPORT, _INGRESS, _ERR_NCISOC_HDR_ECC_ERROR_COUNTER_LIMIT, 1);

    // Set egress ECC error limits
    LWSWITCH_BCAST_WR32_LR10(device, NPORT, _EGRESS, _ERR_NXBAR_ECC_ERROR_COUNTER,
        DRF_NUM(_EGRESS, _ERR_NXBAR_ECC_ERROR_COUNTER, _ERROR_COUNT, 0x0));
    LWSWITCH_BCAST_WR32_LR10(device, NPORT, _EGRESS, _ERR_NXBAR_ECC_ERROR_COUNTER_LIMIT, 1);

    LWSWITCH_BCAST_WR32_LR10(device, NPORT, _EGRESS, _ERR_RAM_OUT_ECC_ERROR_COUNTER,
        DRF_NUM(_EGRESS, _ERR_RAM_OUT_ECC_ERROR_COUNTER, _ERROR_COUNT, 0x0));
    LWSWITCH_BCAST_WR32_LR10(device, NPORT, _EGRESS, _ERR_RAM_OUT_ECC_ERROR_COUNTER_LIMIT, 1);

    // Set route ECC error limits
    LWSWITCH_BCAST_WR32_LR10(device, NPORT, _ROUTE, _ERR_LWS_ECC_ERROR_COUNTER,
        DRF_NUM(_ROUTE, _ERR_LWS_ECC_ERROR_COUNTER, _ERROR_COUNT, 0x0));
    LWSWITCH_BCAST_WR32_LR10(device, NPORT, _ROUTE, _ERR_LWS_ECC_ERROR_COUNTER_LIMIT, 1);

    // Set tstate ECC error limits
    LWSWITCH_BCAST_WR32_LR10(device, NPORT, _TSTATE, _ERR_CRUMBSTORE_ECC_ERROR_COUNTER,
        DRF_NUM(_TSTATE, _ERR_CRUMBSTORE_ECC_ERROR_COUNTER, _ERROR_COUNT, 0x0));
    LWSWITCH_BCAST_WR32_LR10(device, NPORT, _TSTATE, _ERR_CRUMBSTORE_ECC_ERROR_COUNTER_LIMIT, 1);

    LWSWITCH_BCAST_WR32_LR10(device, NPORT, _TSTATE, _ERR_TAGPOOL_ECC_ERROR_COUNTER,
        DRF_NUM(_TSTATE, _ERR_TAGPOOL_ECC_ERROR_COUNTER, _ERROR_COUNT, 0x0));
    LWSWITCH_BCAST_WR32_LR10(device, NPORT, _TSTATE, _ERR_TAGPOOL_ECC_ERROR_COUNTER_LIMIT, 1);

    // Set sourcetrack ECC error limits to _PROD value
    LWSWITCH_BCAST_WR32_LR10(device, NPORT, _SOURCETRACK, _ERR_CREQ_TCEN0_CRUMBSTORE_ECC_ERROR_COUNTER_LIMIT,
        DRF_NUM(_SOURCETRACK, _ERR_CREQ_TCEN0_CRUMBSTORE_ECC_ERROR_COUNTER, _ERROR_COUNT, 0x0));
    LWSWITCH_BCAST_WR32_LR10(device, NPORT, _SOURCETRACK, _ERR_CREQ_TCEN0_CRUMBSTORE_ECC_ERROR_COUNTER_LIMIT, 1);

    LWSWITCH_BCAST_WR32_LR10(device, NPORT, _SOURCETRACK, _ERR_CREQ_TCEN1_CRUMBSTORE_ECC_ERROR_COUNTER_LIMIT,
        DRF_NUM(_SOURCETRACK, _ERR_CREQ_TCEN1_CRUMBSTORE_ECC_ERROR_COUNTER, _ERROR_COUNT, 0x0));
    LWSWITCH_BCAST_WR32_LR10(device, NPORT, _SOURCETRACK, _ERR_CREQ_TCEN1_CRUMBSTORE_ECC_ERROR_COUNTER_LIMIT, 1);

    // Enable ECC/parity
    LWSWITCH_BCAST_WR32_LR10(device, NPORT, _INGRESS, _ERR_ECC_CTRL,
        DRF_DEF(_INGRESS, _ERR_ECC_CTRL, _NCISOC_HDR_ECC_ENABLE, __PROD) |
        DRF_DEF(_INGRESS, _ERR_ECC_CTRL, _NCISOC_PARITY_ENABLE, __PROD) |
        DRF_DEF(_INGRESS, _ERR_ECC_CTRL, _REMAPTAB_ECC_ENABLE, __PROD) |
        DRF_DEF(_INGRESS, _ERR_ECC_CTRL, _RIDTAB_ECC_ENABLE, __PROD) |
        DRF_DEF(_INGRESS, _ERR_ECC_CTRL, _RLANTAB_ECC_ENABLE, __PROD));

    LWSWITCH_BCAST_WR32_LR10(device, NPORT, _EGRESS, _ERR_ECC_CTRL,
        DRF_DEF(_EGRESS, _ERR_ECC_CTRL, _NXBAR_ECC_ENABLE, __PROD) |
        DRF_DEF(_EGRESS, _ERR_ECC_CTRL, _NXBAR_PARITY_ENABLE, __PROD) |
        DRF_DEF(_EGRESS, _ERR_ECC_CTRL, _RAM_OUT_ECC_ENABLE, __PROD) |
        DRF_DEF(_EGRESS, _ERR_ECC_CTRL, _NCISOC_ECC_ENABLE, __PROD) |
        DRF_DEF(_EGRESS, _ERR_ECC_CTRL, _NCISOC_PARITY_ENABLE, __PROD));

    LWSWITCH_BCAST_WR32_LR10(device, NPORT, _ROUTE, _ERR_ECC_CTRL,
        DRF_DEF(_ROUTE, _ERR_ECC_CTRL, _GLT_ECC_ENABLE, __PROD) |
        DRF_DEF(_ROUTE, _ERR_ECC_CTRL, _LWS_ECC_ENABLE, __PROD));

    LWSWITCH_BCAST_WR32_LR10(device, NPORT, _TSTATE, _ERR_ECC_CTRL,
        DRF_DEF(_TSTATE, _ERR_ECC_CTRL, _CRUMBSTORE_ECC_ENABLE, __PROD) |
        DRF_DEF(_TSTATE, _ERR_ECC_CTRL, _TAGPOOL_ECC_ENABLE, __PROD) |
        DRF_DEF(_TSTATE, _ERR_ECC_CTRL, _TD_TID_ECC_ENABLE, _DISABLE));

    LWSWITCH_BCAST_WR32_LR10(device, NPORT, _SOURCETRACK, _ERR_ECC_CTRL,
        DRF_DEF(_SOURCETRACK, _ERR_ECC_CTRL, _CREQ_TCEN0_CRUMBSTORE_ECC_ENABLE, __PROD) |
        DRF_DEF(_SOURCETRACK, _ERR_ECC_CTRL, _CREQ_TCEN0_TD_CRUMBSTORE_ECC_ENABLE, _DISABLE) |
        DRF_DEF(_SOURCETRACK, _ERR_ECC_CTRL, _CREQ_TCEN1_CRUMBSTORE_ECC_ENABLE, __PROD));
}

static void
_lwswitch_init_cmd_routing
(
    lwswitch_device *device
)
{
    LwU32 val;

    //Set Hash policy for the requests.
    val = DRF_DEF(_ROUTE, _CMD_ROUTE_TABLE0, _RFUN1, _SPRAY) |
          DRF_DEF(_ROUTE, _CMD_ROUTE_TABLE0, _RFUN2, _SPRAY) |
          DRF_DEF(_ROUTE, _CMD_ROUTE_TABLE0, _RFUN4, _SPRAY) |
          DRF_DEF(_ROUTE, _CMD_ROUTE_TABLE0, _RFUN7, _SPRAY);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _ROUTE, _CMD_ROUTE_TABLE0, val);

    // Set Random policy for reponses.
    val = DRF_DEF(_ROUTE, _CMD_ROUTE_TABLE2, _RFUN16, _RANDOM) |
          DRF_DEF(_ROUTE, _CMD_ROUTE_TABLE2, _RFUN17, _RANDOM);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _ROUTE, _CMD_ROUTE_TABLE2, val);
}

static LwlStatus
_lwswitch_init_portstat_counters
(
    lwswitch_device *device
)
{
    LwlStatus retval;
    LwU32 idx_channel;
    LWSWITCH_SET_LATENCY_BINS default_latency_bins;
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);

    chip_device->latency_stats = lwswitch_os_malloc(sizeof(LWSWITCH_LATENCY_STATS_LR10));
    if (chip_device->latency_stats == NULL)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Failed allocate memory for latency stats\n",
            __FUNCTION__);
        return -LWL_NO_MEM;
    }

    lwswitch_os_memset(chip_device->latency_stats, 0, sizeof(LWSWITCH_LATENCY_STATS_LR10));

    //
    // These bin thresholds are values provided by Arch based off
    // switch latency expectations.
    //
    for (idx_channel=0; idx_channel < LWSWITCH_NUM_VCS_LR10; idx_channel++)
    {
        default_latency_bins.bin[idx_channel].lowThreshold = 120;    // 120ns
        default_latency_bins.bin[idx_channel].medThreshold = 200;    // 200ns
        default_latency_bins.bin[idx_channel].hiThreshold  = 1000;   // 1us
    }

    chip_device->latency_stats->sample_interval_msec = 3000; // 3 second sample interval

    retval = lwswitch_ctrl_set_latency_bins(device, &default_latency_bins);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Failed to set latency bins\n",
            __FUNCTION__);
        LWSWITCH_ASSERT(0);
        return retval;
    }

    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _NPORT, _PORTSTAT_CONTROL,
        DRF_DEF(_NPORT, _PORTSTAT_CONTROL, _SWEEPMODE, _SWONDEMAND) |
        DRF_DEF(_NPORT, _PORTSTAT_CONTROL, _RANGESELECT, _BITS13TO0));

     LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _NPORT, _PORTSTAT_SOURCE_FILTER_0,
         DRF_NUM(_NPORT, _PORTSTAT_SOURCE_FILTER_0, _SRCFILTERBIT, 0xFFFFFFFF));

    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _NPORT, _PORTSTAT_SOURCE_FILTER_1,
        DRF_NUM(_NPORT, _PORTSTAT_SOURCE_FILTER_1, _SRCFILTERBIT, 0xF));

    // Set window limit to the maximum value
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _NPORT, _PORTSTAT_WINDOW_LIMIT, 0xffffffff);

     LWSWITCH_SAW_WR32_LR10(device, _LWLSAW, _GLBLLATENCYTIMERCTRL,
         DRF_DEF(_LWLSAW, _GLBLLATENCYTIMERCTRL, _ENABLE, _ENABLE));

     LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _NPORT, _PORTSTAT_SNAP_CONTROL,
         DRF_DEF(_NPORT, _PORTSTAT_SNAP_CONTROL, _STARTCOUNTER, _ENABLE) |
         DRF_DEF(_NPORT, _PORTSTAT_SNAP_CONTROL, _SNAPONDEMAND, _DISABLE));

     return LWL_SUCCESS;
}

LwlStatus
lwswitch_init_nxbar_lr10
(
    lwswitch_device *device
)
{
    LwU32 tileout;

    // Setting this bit will send error detection info to NPG.
    LWSWITCH_BCAST_WR32_LR10(device, TILE, _NXBAR, _TILE_ERR_CYA,
        DRF_DEF(_NXBAR, _TILE_ERR_CYA, _SRCID_UPDATE_AT_EGRESS_CTRL, __PROD));

    for (tileout = 0; tileout < NUM_NXBAR_TILEOUTS_PER_TC_LR10; tileout++)
    {
        LWSWITCH_BCAST_WR32_LR10(device, NXBAR, _NXBAR, _TC_TILEOUT_ERR_CYA(tileout),
            DRF_DEF(_NXBAR, _TC_TILEOUT0_ERR_CYA, _SRCID_UPDATE_AT_EGRESS_CTRL, __PROD));
    }

    // Enable idle-based clk gating and setup delay count.
    LWSWITCH_BCAST_WR32_LR10(device, TILE, _NXBAR, _TILE_PRI_NXBAR_TILE_CG,
        DRF_DEF(_NXBAR, _TILE_PRI_NXBAR_TILE_CG, _IDLE_CG_EN, __PROD) |
        DRF_DEF(_NXBAR, _TILE_PRI_NXBAR_TILE_CG, _IDLE_CG_DLY_CNT, __PROD));

    LWSWITCH_BCAST_WR32_LR10(device, NXBAR, _NXBAR, _TC_PRI_NXBAR_TC_CG,
        DRF_DEF(_NXBAR, _TC_PRI_NXBAR_TC_CG, _IDLE_CG_EN, __PROD) |
        DRF_DEF(_NXBAR, _TC_PRI_NXBAR_TC_CG, _IDLE_CG_DLY_CNT, __PROD));

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_init_nport_lr10
(
    lwswitch_device *device
)
{
    LwU32 data32, timeout;
    LwU32 idx_nport;
    LwU32 num_nports;

    num_nports = LWSWITCH_ENG_COUNT(device, NPORT, );

    for (idx_nport = 0; idx_nport < num_nports; idx_nport++)
    {
        // Find the first valid nport
        if (LWSWITCH_ENG_IS_VALID(device, NPORT, idx_nport))
        {
            break;
        }
    }

    // There were no valid nports
    if (idx_nport == num_nports)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: No valid nports found!\n", __FUNCTION__);
        return -LWL_ERR_ILWALID_STATE;
    }

    _lwswitch_init_nport_ecc_control_lr10(device);

    data32 = LWSWITCH_NPORT_RD32_LR10(device, idx_nport, _ROUTE, _ROUTE_CONTROL);
    data32 = FLD_SET_DRF(_ROUTE, _ROUTE_CONTROL, _URRESPENB, __PROD, data32);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _ROUTE, _ROUTE_CONTROL, data32);

    data32 = LWSWITCH_NPORT_RD32_LR10(device, idx_nport, _EGRESS, _CTRL);
    data32 = FLD_SET_DRF(_EGRESS, _CTRL, _DESTINATIONIDCHECKENB, __PROD, data32);
    data32 = FLD_SET_DRF(_EGRESS, _CTRL, _CTO_ENB, __PROD, data32);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _EGRESS, _CTRL, data32);

    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _EGRESS, _CTO_TIMER_LIMIT,
        DRF_DEF(_EGRESS, _CTO_TIMER_LIMIT, _LIMIT, __PROD));

    if (DRF_VAL(_SWITCH_REGKEY, _ATO_CONTROL, _DISABLE, device->regkeys.ato_control) ==
        LW_SWITCH_REGKEY_ATO_CONTROL_DISABLE_TRUE)
    {
        // ATO Disable
        data32 = LWSWITCH_NPORT_RD32_LR10(device, idx_nport, _TSTATE, _TAGSTATECONTROL);
        data32 = FLD_SET_DRF(_TSTATE, _TAGSTATECONTROL, _ATO_ENB, _OFF, data32);
        LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _TSTATE, _TAGSTATECONTROL, data32);
    }
    else
    {
        // ATO Enable
        data32 = LWSWITCH_NPORT_RD32_LR10(device, idx_nport, _TSTATE, _TAGSTATECONTROL);
        data32 = FLD_SET_DRF(_TSTATE, _TAGSTATECONTROL, _ATO_ENB, _ON, data32);
        LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _TSTATE, _TAGSTATECONTROL, data32);

        // ATO Timeout value
        timeout = DRF_VAL(_SWITCH_REGKEY, _ATO_CONTROL, _TIMEOUT, device->regkeys.ato_control);
        if (timeout != LW_SWITCH_REGKEY_ATO_CONTROL_TIMEOUT_DEFAULT)
        {
            LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _TSTATE, _ATO_TIMER_LIMIT,
                DRF_NUM(_TSTATE, _ATO_TIMER_LIMIT, _LIMIT, timeout));
        }
        else
        {
            LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _TSTATE, _ATO_TIMER_LIMIT,
                DRF_DEF(_TSTATE, _ATO_TIMER_LIMIT, _LIMIT, __PROD));
        }
    }

    if (DRF_VAL(_SWITCH_REGKEY, _STO_CONTROL, _DISABLE, device->regkeys.sto_control) ==
        LW_SWITCH_REGKEY_STO_CONTROL_DISABLE_TRUE)
    {
        // STO Disable
        data32 = LWSWITCH_NPORT_RD32_LR10(device, idx_nport, _SOURCETRACK, _CTRL);
        data32 = FLD_SET_DRF(_SOURCETRACK, _CTRL, _STO_ENB, _OFF, data32);
        LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _SOURCETRACK, _CTRL, data32);
    }
    else
    {
        // STO Enable
        data32 = LWSWITCH_NPORT_RD32_LR10(device, idx_nport, _SOURCETRACK, _CTRL);
        data32 = FLD_SET_DRF(_SOURCETRACK, _CTRL, _STO_ENB, _ON, data32);
        LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _SOURCETRACK, _CTRL, data32);

        // STO Timeout value
        timeout = DRF_VAL(_SWITCH_REGKEY, _STO_CONTROL, _TIMEOUT, device->regkeys.sto_control);
        if (timeout != LW_SWITCH_REGKEY_STO_CONTROL_TIMEOUT_DEFAULT)
        {
            LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _SOURCETRACK, _MULTISEC_TIMER0,
                DRF_NUM(_SOURCETRACK, _MULTISEC_TIMER0, _TIMERVAL0, timeout));
        }
        else
        {
            LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _SOURCETRACK, _MULTISEC_TIMER0,
                DRF_DEF(_SOURCETRACK, _MULTISEC_TIMER0, _TIMERVAL0, __PROD));
        }
    }

    //
    // WAR for bug 200606509
    // Disable CAM for entry 0 to prevent false ATO trigger
    //
    data32 = LWSWITCH_NPORT_RD32_LR10(device, idx_nport, _TSTATE, _CREQ_CAM_LOCK);
    data32 = DRF_NUM(_TSTATE, _CREQ_CAM_LOCK, _ON, 0x1);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _TSTATE, _CREQ_CAM_LOCK, data32);

    //
    // WAR for bug 3115824
    // Clear CONTAIN_AND_DRAIN during init for links in reset.
    // Since SBR does not clear CONTAIN_AND_DRAIN, this will clear the bit
    // when the driver is reloaded after an SBR. If the driver has been reloaded
    // without an SBR, then CONTAIN_AND_DRAIN will be re-triggered.
    //
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _NPORT, _CONTAIN_AND_DRAIN,
        DRF_DEF(_NPORT, _CONTAIN_AND_DRAIN, _CLEAR, _ENABLE));

    return LWL_SUCCESS;
}

void *
lwswitch_alloc_chipdevice_lr10
(
    lwswitch_device *device
)
{
    void *chip_device;

    chip_device = lwswitch_os_malloc(sizeof(lr10_device));
    if (NULL != chip_device)
    {
        lwswitch_os_memset(chip_device, 0, sizeof(lr10_device));
    }

    device->chip_id = LW_PSMC_BOOT_42_CHIP_ID_LR10;
    return(chip_device);
}

static LwlStatus
lwswitch_initialize_pmgr_lr10
(
    lwswitch_device *device
)
{
    lwswitch_init_pmgr_lr10(device);
    lwswitch_init_pmgr_devices_lr10(device);

    return LWL_SUCCESS;
}

static LwlStatus
lwswitch_initialize_route_lr10
(
    lwswitch_device *device
)
{
    LwlStatus retval;

    retval = _lwswitch_init_ganged_link_routing(device);
    if (LWL_SUCCESS != retval)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to initialize GLT\n",
            __FUNCTION__);
        goto lwswitch_initialize_route_exit;
    }

    _lwswitch_init_cmd_routing(device);

    // Initialize Portstat Counters
    retval = _lwswitch_init_portstat_counters(device);
    if (LWL_SUCCESS != retval)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to initialize portstat counters\n",
            __FUNCTION__);
        goto lwswitch_initialize_route_exit;
    }

lwswitch_initialize_route_exit:
    return retval;
}


LwlStatus
lwswitch_pri_ring_init_lr10
(
    lwswitch_device *device
)
{
    LwU32 i;
    LwU32 value;
    LwBool enumerated = LW_FALSE;
    LwlStatus retval = LWL_SUCCESS;

    //
    // Sometimes on RTL simulation we see the priv ring initialization fail.
    // Retry up to 3 times until this issue is root caused. Bug 1826216.
    //
    for (i = 0; !enumerated && (i < 3); i++)
    {
        value = DRF_DEF(_PPRIV_MASTER, _RING_COMMAND, _CMD, _ENUMERATE_AND_START_RING);
        retval = lwswitch_ring_master_cmd_lr10(device, value);
        if (retval != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: PRIV ring enumeration failed\n",
                __FUNCTION__);
            continue;
        }

        value = LWSWITCH_REG_RD32(device, _PPRIV_MASTER, _RING_START_RESULTS);
        if (!FLD_TEST_DRF(_PPRIV_MASTER, _RING_START_RESULTS, _CONNECTIVITY, _PASS, value))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: PRIV ring connectivity failed\n",
                __FUNCTION__);
            continue;
        }

        value = LWSWITCH_REG_RD32(device, _PPRIV_MASTER, _RING_INTERRUPT_STATUS0);
        if (value)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: LW_PPRIV_MASTER_RING_INTERRUPT_STATUS0 = %x\n",
                __FUNCTION__, value);

            if ((!FLD_TEST_DRF_NUM(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0,
                    _RING_START_CONN_FAULT, 0, value)) ||
                (!FLD_TEST_DRF_NUM(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0,
                    _DISCONNECT_FAULT, 0, value))      ||
                (!FLD_TEST_DRF_NUM(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0,
                    _OVERFLOW_FAULT, 0, value)))
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: PRIV ring error interrupt\n",
                    __FUNCTION__);
            }

            (void)lwswitch_ring_master_cmd_lr10(device,
                    DRF_DEF(_PPRIV_MASTER, _RING_COMMAND, _CMD, _ACK_INTERRUPT));

            continue;
        }

        enumerated = LW_TRUE;
    }

    if (!enumerated)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Cannot enumerate PRIV ring!\n",
            __FUNCTION__);
        retval = -LWL_INITIALIZATION_TOTAL_FAILURE;
    }

    return retval;
}

/*
 * @Brief : Initializes an LwSwitch hardware state
 *
 * @Description :
 *
 * @param[in] device        a reference to the device to initialize
 *
 * @returns                 LWL_SUCCESS if the action succeeded
 *                          -LWL_BAD_ARGS if bad arguments provided
 *                          -LWL_PCI_ERROR if bar info unable to be retrieved
 */
LwlStatus
lwswitch_initialize_device_state_lr10
(
    lwswitch_device *device
)
{
    LwlStatus retval = LWL_SUCCESS;

    // alloc chip-specific device structure
    device->chip_device = lwswitch_alloc_chipdevice(device);
    if (NULL == device->chip_device)
    {
        LWSWITCH_PRINT(device, ERROR,
            "lwswitch_os_malloc during chip_device creation failed!\n");
        retval = -LWL_NO_MEM;
        goto lwswitch_initialize_device_state_exit;
    }

    LWSWITCH_PRINT(device, SETUP,
        "%s: MMIO discovery\n",
        __FUNCTION__);
    retval = lwswitch_device_discovery(device, LW_SWPTOP_TABLE_BASE_ADDRESS_OFFSET);
    if (LWL_SUCCESS != retval)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Engine discovery failed\n",
            __FUNCTION__);
        goto lwswitch_initialize_device_state_exit;
    }

    lwswitch_filter_discovery(device);

    retval = lwswitch_process_discovery(device);
    if (LWL_SUCCESS != retval)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Discovery processing failed\n",
            __FUNCTION__);
        goto lwswitch_initialize_device_state_exit;
    }

    // now that we have completed discovery, perform initialization steps that
    // depend on engineDescriptors being initialized
    //
    // Temporary location, really needs to be done somewhere common to all flcnables
    if (lwswitch_is_soe_supported(device))
    {
        flcnablePostDiscoveryInit(device, device->pSoe);
    }
    else
    {
        LWSWITCH_PRINT(device, INFO, "%s: Skipping SOE post discovery init.\n",
            __FUNCTION__);
    }

    // Make sure interrupts are disabled before we enable interrupts with the OS.
    lwswitch_lib_disable_interrupts(device);

    retval = lwswitch_pri_ring_init(device);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: PRI init failed\n", __FUNCTION__);
        goto lwswitch_initialize_device_state_exit;
    }

    LWSWITCH_PRINT(device, SETUP,
        "%s: Enabled links: 0x%llx\n",
        __FUNCTION__,
        ((LwU64)device->regkeys.link_enable_mask2 << 32 |
        (LwU64)device->regkeys.link_enable_mask) &
        ((~0ULL) >> (64 - LWSWITCH_LINK_COUNT(device))));

    if (lwswitch_is_soe_supported(device))
    {
        retval = lwswitch_init_soe(device);
        if (LWL_SUCCESS != retval)
        {
            LWSWITCH_PRINT(device, ERROR, "%s: Init SOE failed\n",
                __FUNCTION__);
            goto lwswitch_initialize_device_state_exit;
        }
    }
    else
    {
        LWSWITCH_PRINT(device, INFO, "%s: Skipping SOE init.\n",
            __FUNCTION__);
    }

    // Read ROM configuration
    lwswitch_read_rom_tables(device, &device->firmware);
    _lwswitch_process_firmware_info_lr10(device, &device->firmware);

    // Init PMGR info
    retval = lwswitch_initialize_pmgr(device);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: PMGR init failed\n", __FUNCTION__);
        retval = -LWL_INITIALIZATION_TOTAL_FAILURE;
        goto lwswitch_initialize_device_state_exit;
    }

    retval = lwswitch_init_pll_config(device);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: failed\n", __FUNCTION__);
        retval = -LWL_INITIALIZATION_TOTAL_FAILURE;
        goto lwswitch_initialize_device_state_exit;
    }

    //
    // PLL init should be done *first* before other hardware init
    //
    retval = lwswitch_init_pll(device);
    if (LWL_SUCCESS != retval)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: PLL init failed\n",
            __FUNCTION__);
        goto lwswitch_initialize_device_state_exit;
    }

    //
    // Now that software knows the devices and addresses, it must take all
    // the wrapper modules out of reset.  It does this by writing to the
    // PMC module enable registers.
    //

    // Init IP wrappers
//    _lwswitch_init_mc_enable_lr10(device);
    retval = lwswitch_initialize_ip_wrappers(device);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: init failed\n", __FUNCTION__);
        retval = -LWL_INITIALIZATION_TOTAL_FAILURE;
        goto lwswitch_initialize_device_state_exit;
    }

    _lwswitch_init_warm_reset_lr10(device);
    _lwswitch_init_npg_multicast_lr10(device);
    retval = lwswitch_clear_nport_rams(device);
    if (LWL_SUCCESS != retval)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: NPORT RAM clear failed\n",
            __FUNCTION__);
        goto lwswitch_initialize_device_state_exit;
    }

    retval = lwswitch_init_nport(device);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Init NPORTs failed\n",
            __FUNCTION__);
        goto lwswitch_initialize_device_state_exit;
    }

    retval = lwswitch_init_nxbar(device);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Init NXBARs failed\n",
            __FUNCTION__);
        goto lwswitch_initialize_device_state_exit;
    }

    if (device->regkeys.minion_disable != LW_SWITCH_REGKEY_MINION_DISABLE_YES)
    {
        LWSWITCH_PRINT(device, WARN, "%s: Entering init minion\n", __FUNCTION__);

        retval = lwswitch_init_minion(device);
        if (LWL_SUCCESS != retval)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Init MINIONs failed\n",
                __FUNCTION__);
            goto lwswitch_initialize_device_state_exit;
        }
    }
    else
    {
        LWSWITCH_PRINT(device, INFO, "MINION is disabled via regkey.\n");

        LWSWITCH_PRINT(device, INFO, "%s: Skipping MINION init\n",
            __FUNCTION__);
    }

    _lwswitch_setup_chiplib_forced_config_lr10(device);

    // Init route
    retval = lwswitch_initialize_route(device);
    if (retval != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: route init failed\n", __FUNCTION__);
        retval = -LWL_INITIALIZATION_TOTAL_FAILURE;
        goto lwswitch_initialize_device_state_exit;
    }

    lwswitch_init_clock_gating(device);

    // Initialize SPI
    if (lwswitch_is_spi_supported(device))
    {
        retval = lwswitch_spi_init(device);
        if (LWL_SUCCESS != retval)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: SPI init failed!, rc: %d\n",
                __FUNCTION__, retval);
            goto lwswitch_initialize_device_state_exit;
        }
    }
    else
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Skipping SPI init.\n",
            __FUNCTION__);
    }

    // Initialize SMBPBI
    if (lwswitch_is_smbpbi_supported(device))
    {
        retval = lwswitch_smbpbi_init(device);
        if (LWL_SUCCESS != retval)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: SMBPBI init failed!, rc: %d\n",
                __FUNCTION__, retval);
            goto lwswitch_initialize_device_state_exit;
        }
    }
    else
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Skipping SMBPBI init.\n",
            __FUNCTION__);
    }

    lwswitch_initialize_interrupt_tree(device);

    // Initialize external thermal sensor
    retval = lwswitch_init_thermal(device);
    if (LWL_SUCCESS != retval)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: External Thermal init failed\n",
            __FUNCTION__);
    }

    return LWL_SUCCESS;

lwswitch_initialize_device_state_exit:
    lwswitch_destroy_device_state(device);

    return retval;
}

/*
 * @Brief : Destroys an LwSwitch hardware state
 *
 * @Description :
 *
 * @param[in] device        a reference to the device to initialize
 */
void
lwswitch_destroy_device_state_lr10
(
    lwswitch_device *device
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);

    if (lwswitch_is_soe_supported(device))
    {
        lwswitch_soe_unregister_events(device);
    }

    if (chip_device != NULL)
    {
        if ((chip_device->latency_stats) != NULL)
        {
            lwswitch_os_free(chip_device->latency_stats);
        }

        if ((chip_device->ganged_link_table) != NULL)
        {
            lwswitch_os_free(chip_device->ganged_link_table);
        }

        lwswitch_free_chipdevice(device);
    }

    lwswitch_i2c_destroy(device);

    return;
}

static void
_lwswitch_set_lwlink_caps_lr10
(
    LwU32 *pCaps
)
{
    LwU8 tempCaps[LWSWITCH_LWLINK_CAPS_TBL_SIZE];

    lwswitch_os_memset(tempCaps, 0, sizeof(tempCaps));

    LWSWITCH_SET_CAP(tempCaps, LWSWITCH_LWLINK_CAPS, _VALID);
    LWSWITCH_SET_CAP(tempCaps, LWSWITCH_LWLINK_CAPS, _SUPPORTED);
    LWSWITCH_SET_CAP(tempCaps, LWSWITCH_LWLINK_CAPS, _P2P_SUPPORTED);
    LWSWITCH_SET_CAP(tempCaps, LWSWITCH_LWLINK_CAPS, _P2P_ATOMICS);

    // Assume IBM P9 for PPC -- TODO Xavier support.
#if defined(LWCPU_PPC64LE)
    LWSWITCH_SET_CAP(tempCaps, LWSWITCH_LWLINK_CAPS, _SYSMEM_ACCESS);
    LWSWITCH_SET_CAP(tempCaps, LWSWITCH_LWLINK_CAPS, _SYSMEM_ATOMICS);
#endif

    lwswitch_os_memcpy(pCaps, tempCaps, sizeof(tempCaps));
}

/*
 * @brief Determines if a link's lanes are reversed
 *
 * @param[in] device    a reference to the device to query
 * @param[in] linkId    Target link ID
 *
 * @return LW_TRUE if a link's lanes are reversed
 */
LwBool
lwswitch_link_lane_reversed_lr10
(
    lwswitch_device *device,
    LwU32            linkId
)
{
    LwU32 regData;
    lwlink_link *link;

    link = lwswitch_get_link(device, linkId);
    if (lwswitch_is_link_in_reset(device, link))
    {
        return LW_FALSE;
    }

    regData = LWSWITCH_LINK_RD32_LR10(device, linkId, LWLDL, _LWLDL_RX, _CONFIG_RX);

    // HW may reverse the lane ordering or it may be overridden by SW.
    if (FLD_TEST_DRF(_LWLDL_RX, _CONFIG_RX, _REVERSAL_OVERRIDE, _ON, regData))
    {
        // Overridden
        if (FLD_TEST_DRF(_LWLDL_RX, _CONFIG_RX, _LANE_REVERSE, _ON, regData))
        {
            return LW_TRUE;
        }
        else
        {
            return LW_FALSE;
        }
    }
    else
    {
        // Sensed in HW
        if (FLD_TEST_DRF(_LWLDL_RX, _CONFIG_RX, _HW_LANE_REVERSE, _ON, regData))
        {
            return LW_TRUE;
        }
        else
        {
            return LW_FALSE;
        }
    }

    return LW_FALSE;
}

LwlStatus
lwswitch_ctrl_get_lwlink_status_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_STATUS_PARAMS *ret
)
{
    LwlStatus retval = LWL_SUCCESS;
    lwlink_link *link;
    LwU8 i;
    LwU32 linkState, txSublinkStatus, rxSublinkStatus;
    lwlink_conn_info conn_info = {0};
    LwU64 enabledLinkMask;
    LwU32 lwlink_caps_version;
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LwBool bIsRepeaterMode = LW_FALSE;
#endif

    enabledLinkMask = lwswitch_get_enabled_link_mask(device);
    ret->enabledLinkMask = enabledLinkMask;

    FOR_EACH_INDEX_IN_MASK(64, i, enabledLinkMask)
    {
        LWSWITCH_ASSERT(i < LWSWITCH_LINK_COUNT(device));

        link = lwswitch_get_link(device, i);

        if ((link == NULL) ||
            !LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLDL, link->linkNumber) ||
            (i >= LWSWITCH_LWLINK_MAX_LINKS))
        {
            continue;
        }

        //
        // Call the core library to get the remote end information. On the first
        // invocation this will also trigger link training, if link-training is
        // not externally managed by FM. Therefore it is necessary that this be
        // before link status on the link is populated since this call will
        // actually change link state.
        //
        if (device->regkeys.external_fabric_mgmt)
        {
            lwlink_lib_get_remote_conn_info(link, &conn_info);
        }
        else
        {
            lwlink_lib_discover_and_get_remote_conn_info(link, &conn_info, LWLINK_STATE_CHANGE_SYNC);
        }

        // Set LWLINK per-link caps
        _lwswitch_set_lwlink_caps_lr10(&ret->linkInfo[i].capsTbl);

        ret->linkInfo[i].phyType = LWSWITCH_LWLINK_STATUS_PHY_LWHS;
        ret->linkInfo[i].subLinkWidth = lwswitch_get_sublink_width(device, link->linkNumber);

        if (!lwswitch_is_link_in_reset(device, link))
        {
            linkState = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TOP, _LINK_STATE);
            linkState = DRF_VAL(_LWLDL_TOP, _LINK_STATE, _STATE, linkState);

            txSublinkStatus = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TX, _SLSM_STATUS_TX);
            txSublinkStatus = DRF_VAL(_LWLDL_TX, _SLSM_STATUS_TX, _PRIMARY_STATE, txSublinkStatus);

            rxSublinkStatus = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_RX, _SLSM_STATUS_RX);
            rxSublinkStatus = DRF_VAL(_LWLDL_RX, _SLSM_STATUS_RX, _PRIMARY_STATE, rxSublinkStatus);

            ret->linkInfo[i].bLaneReversal = lwswitch_link_lane_reversed_lr10(device, i);
        }
        else
        {
            linkState       = LWSWITCH_LWLINK_STATUS_LINK_STATE_INIT;
            txSublinkStatus = LWSWITCH_LWLINK_STATUS_SUBLINK_TX_STATE_OFF;
            rxSublinkStatus = LWSWITCH_LWLINK_STATUS_SUBLINK_RX_STATE_OFF;
        }

        ret->linkInfo[i].linkState       = linkState;
        ret->linkInfo[i].txSublinkStatus = txSublinkStatus;
        ret->linkInfo[i].rxSublinkStatus = rxSublinkStatus;

        lwlink_caps_version = lwswitch_get_caps_lwlink_version(device);
        if (lwlink_caps_version == LWSWITCH_LWLINK_CAPS_LWLINK_VERSION_3_0)
        {
            ret->linkInfo[i].lwlinkVersion = LWSWITCH_LWLINK_STATUS_LWLINK_VERSION_3_0;
            ret->linkInfo[i].nciVersion = LWSWITCH_LWLINK_STATUS_NCI_VERSION_3_0;
        }
        else if (lwlink_caps_version == LWSWITCH_LWLINK_CAPS_LWLINK_VERSION_4_0)
        {
            ret->linkInfo[i].lwlinkVersion = LWSWITCH_LWLINK_STATUS_LWLINK_VERSION_4_0;
            ret->linkInfo[i].nciVersion = LWSWITCH_LWLINK_STATUS_NCI_VERSION_4_0;
        }
        else
        {
            LWSWITCH_PRINT(device, WARN,
                "%s WARNING: Unknown LWSWITCH_LWLINK_CAPS_LWLINK_VERSION 0x%x\n",
                __FUNCTION__, lwlink_caps_version);
            ret->linkInfo[i].lwlinkVersion = LWSWITCH_LWLINK_STATUS_LWLINK_VERSION_ILWALID;
            ret->linkInfo[i].nciVersion = LWSWITCH_LWLINK_STATUS_NCI_VERSION_ILWALID;
        }

        ret->linkInfo[i].phyVersion = LWSWITCH_LWLINK_STATUS_LWHS_VERSION_1_0;

        if (conn_info.bConnected)
        {
            ret->linkInfo[i].connected = LWSWITCH_LWLINK_STATUS_CONNECTED_TRUE;
            ret->linkInfo[i].remoteDeviceLinkNumber = (LwU8)conn_info.linkNumber;

            ret->linkInfo[i].remoteDeviceInfo.domain = conn_info.domain;
            ret->linkInfo[i].remoteDeviceInfo.bus = conn_info.bus;
            ret->linkInfo[i].remoteDeviceInfo.device = conn_info.device;
            ret->linkInfo[i].remoteDeviceInfo.function = conn_info.function;
            ret->linkInfo[i].remoteDeviceInfo.pciDeviceId = conn_info.pciDeviceId;
            ret->linkInfo[i].remoteDeviceInfo.deviceType = conn_info.deviceType;

            ret->linkInfo[i].localLinkSid  = link->localSid;
            ret->linkInfo[i].remoteLinkSid = link->remoteSid;

            if (0 != conn_info.pciDeviceId)
            {
                ret->linkInfo[i].remoteDeviceInfo.deviceIdFlags =
                    FLD_SET_DRF(SWITCH_LWLINK, _DEVICE_INFO, _DEVICE_ID_FLAGS,
                         _PCI, ret->linkInfo[i].remoteDeviceInfo.deviceIdFlags);
            }

            // Does not use loopback
            ret->linkInfo[i].loopProperty =
                LWSWITCH_LWLINK_STATUS_LOOP_PROPERTY_NONE;
        }
        else
        {
            ret->linkInfo[i].connected =
                LWSWITCH_LWLINK_STATUS_CONNECTED_FALSE;
            ret->linkInfo[i].remoteDeviceInfo.deviceType =
                LWSWITCH_LWLINK_DEVICE_INFO_DEVICE_TYPE_NONE;
        }

        // Set the device information for the local end of the link
        ret->linkInfo[i].localDeviceInfo.domain = device->lwlink_device->pciInfo.domain;
        ret->linkInfo[i].localDeviceInfo.bus = device->lwlink_device->pciInfo.bus;
        ret->linkInfo[i].localDeviceInfo.device = device->lwlink_device->pciInfo.device;
        ret->linkInfo[i].localDeviceInfo.function = device->lwlink_device->pciInfo.function;
        ret->linkInfo[i].localDeviceInfo.pciDeviceId = 0xdeadbeef; // TODO
        ret->linkInfo[i].localDeviceLinkNumber = i;
        ret->linkInfo[i].laneRxdetStatusMask = device->link[i].lane_rxdet_status_mask;
        ret->linkInfo[i].localDeviceInfo.deviceType =
            LWSWITCH_LWLINK_DEVICE_INFO_DEVICE_TYPE_SWITCH;

        // Clock data
        ret->linkInfo[i].lwlinkLineRateMbps = lwswitch_minion_get_line_rate_Mbps_lr10(device, i);
        ret->linkInfo[i].lwlinkLinkDataRateKiBps = lwswitch_minion_get_data_rate_KiBps_lr10(device, i);
        ret->linkInfo[i].lwlinkLinkClockMhz = ret->linkInfo[i].lwlinkLineRateMbps / 32;
        ret->linkInfo[i].lwlinkRefClkSpeedMhz = 156;
        ret->linkInfo[i].lwlinkRefClkType = LWSWITCH_LWLINK_REFCLK_TYPE_LWHS;

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
        // Repeater mode
        retval = device->hal.lwswitch_is_link_in_repeater_mode(device, i, &bIsRepeaterMode);
        if (retval != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR, "%s: Error getting Repeater Mode for link %d.\n", __FUNCTION__, i);
            return retval;
        }
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        ret->linkInfo[i].bIsRepeaterMode = bIsRepeaterMode;
#endif
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    }
    FOR_EACH_INDEX_IN_MASK_END;

//    LWSWITCH_ASSERT(ret->enabledLinkMask == enabledLinkMask);

    return retval;
}

LwlStatus
lwswitch_ctrl_get_counters_lr10
(
    lwswitch_device *device,
    LWSWITCH_LWLINK_GET_COUNTERS_PARAMS *ret
)
{
    lwlink_link *link;
    LwU8   i;
    LwU32  counterMask;
    LwU32  data;
    LwU32  val;
    LwU64  tx0TlCount;
    LwU64  tx1TlCount;
    LwU64  rx0TlCount;
    LwU64  rx1TlCount;
    LwU32  laneId;
    LwBool bLaneReversed;
    LwlStatus status;
    LwBool minion_enabled;

    ct_assert(LWSWITCH_NUM_LANES_LR10 <= LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE__SIZE);

    link = lwswitch_get_link(device, ret->linkId);
    if ((link == NULL) ||
        !LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLDL, link->linkNumber))
    {
        return -LWL_BAD_ARGS;
    }

    minion_enabled = lwswitch_is_minion_initialized(device, LWSWITCH_GET_LINK_ENG_INST(device, link->linkNumber, MINION));

    counterMask = ret->counterMask;

    // Common usage allows one of these to stand for all of them
    if (counterMask & (LWSWITCH_LWLINK_COUNTER_TL_TX0 |
                       LWSWITCH_LWLINK_COUNTER_TL_TX1 |
                       LWSWITCH_LWLINK_COUNTER_TL_RX0 |
                       LWSWITCH_LWLINK_COUNTER_TL_RX1))
    {
        tx0TlCount = lwswitch_read_64bit_counter(device,
            LWSWITCH_LINK_OFFSET_LR10(device, link->linkNumber, LWLTLC, _LWLTLC_TX_LNK, _DEBUG_TP_CNTR_LO(0)),
            LWSWITCH_LINK_OFFSET_LR10(device, link->linkNumber, LWLTLC, _LWLTLC_TX_LNK, _DEBUG_TP_CNTR_HI(0)));
        if (LWBIT64(63) & tx0TlCount)
        {
            ret->bTx0TlCounterOverflow = LW_TRUE;
            tx0TlCount &= ~(LWBIT64(63));
        }

        tx1TlCount = lwswitch_read_64bit_counter(device,
            LWSWITCH_LINK_OFFSET_LR10(device, link->linkNumber, LWLTLC, _LWLTLC_TX_LNK, _DEBUG_TP_CNTR_LO(1)),
            LWSWITCH_LINK_OFFSET_LR10(device, link->linkNumber, LWLTLC, _LWLTLC_TX_LNK, _DEBUG_TP_CNTR_HI(1)));
        if (LWBIT64(63) & tx1TlCount)
        {
            ret->bTx1TlCounterOverflow = LW_TRUE;
            tx1TlCount &= ~(LWBIT64(63));
        }

        rx0TlCount = lwswitch_read_64bit_counter(device,
            LWSWITCH_LINK_OFFSET_LR10(device, link->linkNumber, LWLTLC, _LWLTLC_RX_LNK, _DEBUG_TP_CNTR_LO(0)),
            LWSWITCH_LINK_OFFSET_LR10(device, link->linkNumber, LWLTLC, _LWLTLC_RX_LNK, _DEBUG_TP_CNTR_HI(0)));
        if (LWBIT64(63) & rx0TlCount)
        {
            ret->bRx0TlCounterOverflow = LW_TRUE;
            rx0TlCount &= ~(LWBIT64(63));
        }

        rx1TlCount = lwswitch_read_64bit_counter(device,
            LWSWITCH_LINK_OFFSET_LR10(device, link->linkNumber, LWLTLC, _LWLTLC_RX_LNK, _DEBUG_TP_CNTR_LO(1)),
            LWSWITCH_LINK_OFFSET_LR10(device, link->linkNumber, LWLTLC, _LWLTLC_RX_LNK, _DEBUG_TP_CNTR_HI(1)));
        if (LWBIT64(63) & rx1TlCount)
        {
            ret->bRx1TlCounterOverflow = LW_TRUE;
            rx1TlCount &= ~(LWBIT64(63));
        }

        ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_TL_TX0)] = tx0TlCount;
        ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_TL_TX1)] = tx1TlCount;
        ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_TL_RX0)] = rx0TlCount;
        ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_TL_RX1)] = rx1TlCount;
    }

    if (counterMask & LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_FLIT)
    {
        if (minion_enabled)
        {
            status = lwswitch_minion_get_dl_status(device, link->linkNumber,
                                    LW_LWLSTAT_RX01, 0, &data);
            if (status != LWL_SUCCESS)
            {
                return status;
            }
            data = DRF_VAL(_LWLSTAT, _RX01, _FLIT_CRC_ERRORS_VALUE, data);
        }
        else
        {
            // MINION disabled
            data = 0;
        }

        ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_FLIT)]
            = data;
    }

    data = 0x0;
    bLaneReversed = lwswitch_link_lane_reversed_lr10(device, link->linkNumber);

    for (laneId = 0; laneId < LWSWITCH_NUM_LANES_LR10; laneId++)
    {
        //
        // HW may reverse the lane ordering or it may be overridden by SW.
        // If so, ilwert the interpretation of the lane CRC errors.
        //
        i = (LwU8)((bLaneReversed) ? (LWSWITCH_NUM_LANES_LR10 - 1) - laneId : laneId);

        if (minion_enabled)
        {
            status = lwswitch_minion_get_dl_status(device, link->linkNumber,
                                    LW_LWLSTAT_DB01, 0, &data);
            if (status != LWL_SUCCESS)
            {
                return status;
            }
        }
        else
        {
            // MINION disabled
            data = 0;
        }

        if (counterMask & LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L(laneId))
        {
            val = BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L(laneId));

            switch (i)
            {
                case 0:
                    ret->lwlinkCounters[val]
                        = DRF_VAL(_LWLSTAT, _DB01, _ERROR_COUNT_ERR_LANECRC_L0, data);
                    break;
                case 1:
                    ret->lwlinkCounters[val]
                        = DRF_VAL(_LWLSTAT, _DB01, _ERROR_COUNT_ERR_LANECRC_L1, data);
                    break;
                case 2:
                    ret->lwlinkCounters[val]
                        = DRF_VAL(_LWLSTAT, _DB01, _ERROR_COUNT_ERR_LANECRC_L2, data);
                    break;
                case 3:
                    ret->lwlinkCounters[val]
                        = DRF_VAL(_LWLSTAT, _DB01, _ERROR_COUNT_ERR_LANECRC_L3, data);
                    break;
            }
        }
    }

    if (counterMask & LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_REPLAY)
    {
        if (minion_enabled)
        {
            status = lwswitch_minion_get_dl_status(device, link->linkNumber,
                                    LW_LWLSTAT_TX09, 0, &data);
            if (status != LWL_SUCCESS)
            {
                return status;
            }
            data = DRF_VAL(_LWLSTAT, _TX09, _REPLAY_EVENTS_VALUE, data);
        }
        else
        {
            // MINION disabled
            data = 0;
        }

        ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_REPLAY)]
            = data;
    }

    if (counterMask & LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_RECOVERY)
    {
        if (minion_enabled)
        {
            status = lwswitch_minion_get_dl_status(device, link->linkNumber,
                                    LW_LWLSTAT_LNK1, 0, &data);
            if (status != LWL_SUCCESS)
            {
                return status;
            }
            data = DRF_VAL(_LWLSTAT, _LNK1, _ERROR_COUNT1_RECOVERY_EVENTS_VALUE, data);
        }
        else
        {
            // MINION disabled
            data = 0;
        }

        ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_RECOVERY)]
            = data;
    }

    if (counterMask & LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_REPLAY)
    {
        if (minion_enabled)
        {
            status = lwswitch_minion_get_dl_status(device, link->linkNumber,
                                    LW_LWLSTAT_RX00, 0, &data);
            if (status != LWL_SUCCESS)
            {
                return status;
            }
            data = DRF_VAL(_LWLSTAT, _RX00, _REPLAY_EVENTS_VALUE, data);
        }
        else
        {
            // MINION disabled
            data = 0;
        }

        ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_REPLAY)]
            = data;
    }

    if (counterMask & LWSWITCH_LWLINK_COUNTER_PHY_REFRESH_PASS)
    {
        ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_PHY_REFRESH_PASS)] = 0;
    }

    if (counterMask & LWSWITCH_LWLINK_COUNTER_PHY_REFRESH_FAIL)
    {
        ret->lwlinkCounters[BIT_IDX_32(LWSWITCH_LWLINK_COUNTER_PHY_REFRESH_FAIL)] = 0;
    }

    return LWL_SUCCESS;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
static void
lwswitch_ctrl_clear_throughput_counters_lr10
(
    lwswitch_device *device,
    lwlink_link     *link,
    LwU32            counterMask
)
{
    LwU32 data;

    // TX
    data = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLTLC, _LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL);
    if (counterMask & LWSWITCH_LWLINK_COUNTER_TL_TX0)
    {
        data = FLD_SET_DRF_NUM(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL, _RESETTX0, 0x1, data);
    }
    if (counterMask & LWSWITCH_LWLINK_COUNTER_TL_TX1)
    {
        data = FLD_SET_DRF_NUM(_LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL, _RESETTX1, 0x1, data);
    }
    LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLTLC, _LWLTLC_TX_LNK, _DEBUG_TP_CNTR_CTRL, data);

    // RX
    data = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLTLC, _LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL);
    if (counterMask & LWSWITCH_LWLINK_COUNTER_TL_RX0)
    {
        data = FLD_SET_DRF_NUM(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL, _RESETRX0, 0x1, data);
    }
    if (counterMask & LWSWITCH_LWLINK_COUNTER_TL_RX1)
    {
        data = FLD_SET_DRF_NUM(_LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL, _RESETRX1, 0x1, data);
    }
    LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLTLC, _LWLTLC_RX_LNK, _DEBUG_TP_CNTR_CTRL, data);
}

static LwlStatus
lwswitch_ctrl_clear_dl_error_counters_lr10
(
    lwswitch_device *device,
    lwlink_link     *link,
    LwU32            counterMask
)
{
    LwU32           data;

    if ((!counterMask) ||
        (!(counterMask & (LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L0 |
                          LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L1 |
                          LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L2 |
                          LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L3 |
                          LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L4 |
                          LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L5 |
                          LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L6 |
                          LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L7 |
                          LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_ECC_COUNTS |
                          LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_REPLAY |
                          LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_RECOVERY))))
    {
        LWSWITCH_PRINT(device, INFO,
            "%s: Link%d: No error count clear request, counterMask (0x%x). Returning!\n",
            __FUNCTION__, link->linkNumber, counterMask);
        return LWL_SUCCESS;
    }

    // With Minion initialized, send command to minion
    if (lwswitch_is_minion_initialized(device, LWSWITCH_GET_LINK_ENG_INST(device, link->linkNumber, MINION)))
    {
        return lwswitch_minion_clear_dl_error_counters_lr10(device, link->linkNumber);
    }

    // With Minion not-initialized, perform with the registers
    if (counterMask & LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_FLIT)
    {
        data = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_RX, _ERROR_COUNT_CTRL);
        data = FLD_SET_DRF(_LWLDL_RX, _ERROR_COUNT_CTRL, _CLEAR_FLIT_CRC, _CLEAR, data);
        data = FLD_SET_DRF(_LWLDL_RX, _ERROR_COUNT_CTRL, _CLEAR_RATES, _CLEAR, data);
        LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLDL, _LWLDL_RX, _ERROR_COUNT_CTRL, data);
    }

    if (counterMask & (LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L0 |
               LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L1 |
               LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L2 |
               LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L3 |
               LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L4 |
               LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L5 |
               LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L6 |
               LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L7 |
               LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_ECC_COUNTS))
    {
        data = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_RX, _ERROR_COUNT_CTRL);
        data = FLD_SET_DRF(_LWLDL_RX, _ERROR_COUNT_CTRL, _CLEAR_LANE_CRC, _CLEAR, data);
        data = FLD_SET_DRF(_LWLDL_RX, _ERROR_COUNT_CTRL, _CLEAR_RATES, _CLEAR, data);
        if (counterMask & LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_ECC_COUNTS)
        {
            data = FLD_SET_DRF(_LWLDL_RX, _ERROR_COUNT_CTRL, _CLEAR_ECC_COUNTS, _CLEAR, data);
        }
        LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLDL, _LWLDL_RX, _ERROR_COUNT_CTRL, data);
    }

    if (counterMask & LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_REPLAY)
    {
        data = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TX, _ERROR_COUNT_CTRL);
        data = FLD_SET_DRF(_LWLDL_TX, _ERROR_COUNT_CTRL, _CLEAR_REPLAY, _CLEAR, data);
        LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TX, _ERROR_COUNT_CTRL, data);
    }

    if (counterMask & LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_RECOVERY)
    {
        data = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TOP, _ERROR_COUNT_CTRL);
        data = FLD_SET_DRF(_LWLDL_TOP, _ERROR_COUNT_CTRL, _CLEAR_RECOVERY, _CLEAR, data);
        LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TOP, _ERROR_COUNT_CTRL, data);
    }
    return LWL_SUCCESS;
}
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

/*
 * CTRL_LWSWITCH_GET_INFO
 *
 * Query for miscellaneous information analogous to LW2080_CTRL_GPU_INFO
 * This provides a single API to query for multiple pieces of miscellaneous
 * information via a single call.
 *
 */

static LwU32
_lwswitch_get_info_chip_id
(
    lwswitch_device *device
)
{
    LwU32 val = LWSWITCH_REG_RD32(device, _PSMC, _BOOT_42);

    return (DRF_VAL(_PSMC, _BOOT_42, _CHIP_ID, val));
}

static LwU32
_lwswitch_get_info_revision_major
(
    lwswitch_device *device
)
{
    LwU32 val = LWSWITCH_REG_RD32(device, _PSMC, _BOOT_42);

    return (DRF_VAL(_PSMC, _BOOT_42, _MAJOR_REVISION, val));
}

static LwU32
_lwswitch_get_info_revision_minor
(
    lwswitch_device *device
)
{
    LwU32 val = LWSWITCH_REG_RD32(device, _PSMC, _BOOT_42);

    return (DRF_VAL(_PSMC, _BOOT_42, _MINOR_REVISION, val));
}

static LwU32
_lwswitch_get_info_revision_minor_ext
(
    lwswitch_device *device
)
{
    LwU32 val = LWSWITCH_REG_RD32(device, _PSMC, _BOOT_42);

    return (DRF_VAL(_PSMC, _BOOT_42, _MINOR_EXTENDED_REVISION, val));
}

static LwU32
_lwswitch_get_info_foundry
(
    lwswitch_device *device
)
{
    LwU32 data = LWSWITCH_REG_RD32(device, _PSMC, _BOOT_0);

    return (DRF_VAL(_PSMC, _BOOT_0, _FOUNDRY, data));
}

static LwU32
_lwswitch_get_info_voltage
(
    lwswitch_device *device
)
{
    LwU32 voltage = 0;

    return voltage;
}

static LwBool
_lwswitch_inforom_lwl_supported
(
    lwswitch_device *device
)
{
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    return LW_TRUE;
#else
    return LW_FALSE;
#endif //(!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
}

static LwBool
_lwswitch_inforom_bbx_supported
(
    lwswitch_device *device
)
{
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    return LW_TRUE;
#else
    return LW_FALSE;
#endif //(!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
}

/*
 * CTRL_LWSWITCH_GET_INFO
 *
 * Query for miscellaneous information analogous to LW2080_CTRL_GPU_INFO
 * This provides a single API to query for multiple pieces of miscellaneous
 * information via a single call.
 *
 */

LwlStatus
lwswitch_ctrl_get_info_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_INFO *p
)
{
    LwlStatus retval = LWL_SUCCESS;
    LwU32 i;

    if (p->count > LWSWITCH_GET_INFO_COUNT_MAX)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Invalid args\n",
            __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    lwswitch_os_memset(p->info, 0, sizeof(LwU32)*LWSWITCH_GET_INFO_COUNT_MAX);

    for (i = 0; i < p->count; i++)
    {
        switch (p->index[i])
        {
            case LWSWITCH_GET_INFO_INDEX_ARCH:
                p->info[i] = device->chip_arch;
                break;
            case LWSWITCH_GET_INFO_INDEX_PLATFORM:
                if (IS_RTLSIM(device))
                {
                    p->info[i] = LWSWITCH_GET_INFO_INDEX_PLATFORM_RTLSIM;
                }
                else if (IS_FMODEL(device))
                {
                    p->info[i] = LWSWITCH_GET_INFO_INDEX_PLATFORM_FMODEL;
                }
                else if (IS_EMULATION(device))
                {
                    p->info[i] = LWSWITCH_GET_INFO_INDEX_PLATFORM_EMULATION;
                }
                else
                {
                    p->info[i] = LWSWITCH_GET_INFO_INDEX_PLATFORM_SILICON;
                }
                break;
            case LWSWITCH_GET_INFO_INDEX_IMPL:
                p->info[i] = device->chip_impl;
                break;
            case LWSWITCH_GET_INFO_INDEX_CHIPID:
                p->info[i] = _lwswitch_get_info_chip_id(device);
                break;
            case LWSWITCH_GET_INFO_INDEX_REVISION_MAJOR:
                p->info[i] = _lwswitch_get_info_revision_major(device);
                break;
            case LWSWITCH_GET_INFO_INDEX_REVISION_MINOR:
                p->info[i] = _lwswitch_get_info_revision_minor(device);
                break;
            case LWSWITCH_GET_INFO_INDEX_REVISION_MINOR_EXT:
                p->info[i] = _lwswitch_get_info_revision_minor_ext(device);
                break;
            case LWSWITCH_GET_INFO_INDEX_FOUNDRY:
                p->info[i] = _lwswitch_get_info_foundry(device);
                break;
            case LWSWITCH_GET_INFO_INDEX_FAB:
                p->info[i] = lwswitch_fuse_opt_read_lr10(device, LW_FUSE_OPT_FAB_CODE);
                break;
            case LWSWITCH_GET_INFO_INDEX_LOT_CODE_0:
                p->info[i] = lwswitch_fuse_opt_read_lr10(device, LW_FUSE_OPT_LOT_CODE_0);
                break;
            case LWSWITCH_GET_INFO_INDEX_LOT_CODE_1:
                p->info[i] = lwswitch_fuse_opt_read_lr10(device, LW_FUSE_OPT_LOT_CODE_1);
                break;
            case LWSWITCH_GET_INFO_INDEX_WAFER:
                p->info[i] = lwswitch_fuse_opt_read_lr10(device, LW_FUSE_OPT_WAFER_ID);
                break;
            case LWSWITCH_GET_INFO_INDEX_XCOORD:
                {
                    LwS32 xcoord = lwswitch_fuse_opt_read_lr10(device, LW_FUSE_OPT_X_COORDINATE);
                    // X coordinate is two's complement, colwert to signed 32-bit
                    p->info[i] = xcoord -
                        2 * (xcoord & (1<<(DRF_SIZE(LW_FUSE_OPT_X_COORDINATE_DATA) - 1)));
                }
                break;
            case LWSWITCH_GET_INFO_INDEX_YCOORD:
                p->info[i] = lwswitch_fuse_opt_read_lr10(device, LW_FUSE_OPT_Y_COORDINATE);
                break;
            case LWSWITCH_GET_INFO_INDEX_SPEEDO_REV:
                p->info[i] = lwswitch_fuse_opt_read_lr10(device, LW_FUSE_OPT_SPEEDO_REV);
                break;
            case LWSWITCH_GET_INFO_INDEX_SPEEDO0:
                p->info[i] = lwswitch_fuse_opt_read_lr10(device, LW_FUSE_OPT_SPEEDO0);
                break;
            case LWSWITCH_GET_INFO_INDEX_SPEEDO1:
                p->info[i] = lwswitch_fuse_opt_read_lr10(device, LW_FUSE_OPT_SPEEDO1);
                break;
            case LWSWITCH_GET_INFO_INDEX_SPEEDO2:
                p->info[i] = lwswitch_fuse_opt_read_lr10(device, LW_FUSE_OPT_SPEEDO2);
                break;
            case LWSWITCH_GET_INFO_INDEX_IDDQ:
                p->info[i] = lwswitch_fuse_opt_read_lr10(device, LW_FUSE_OPT_IDDQ);
                break;
            case LWSWITCH_GET_INFO_INDEX_IDDQ_REV:
                p->info[i] = lwswitch_fuse_opt_read_lr10(device, LW_FUSE_OPT_IDDQ_REV);
                break;
            case LWSWITCH_GET_INFO_INDEX_IDDQ_DVDD:
                p->info[i] = lwswitch_read_iddq_dvdd(device);
                break;
            case LWSWITCH_GET_INFO_INDEX_ATE_REV:
                p->info[i] = lwswitch_fuse_opt_read_lr10(device, LW_FUSE_OPT_LWST_ATE_REV);
                break;
            case LWSWITCH_GET_INFO_INDEX_VENDOR_CODE:
                p->info[i] = lwswitch_fuse_opt_read_lr10(device, LW_FUSE_OPT_VENDOR_CODE);
                break;
            case LWSWITCH_GET_INFO_INDEX_OPS_RESERVED:
                p->info[i] = lwswitch_fuse_opt_read_lr10(device, LW_FUSE_OPT_OPS_RESERVED);
                break;
            case LWSWITCH_GET_INFO_INDEX_DEVICE_ID:
                p->info[i] = device->lwlink_device->pciInfo.pciDeviceId;
                break;
            case LWSWITCH_GET_INFO_INDEX_NUM_PORTS:
                p->info[i] = LWSWITCH_LINK_COUNT(device);
                break;
            case LWSWITCH_GET_INFO_INDEX_ENABLED_PORTS_MASK_31_0:
                p->info[i] = LwU64_LO32(lwswitch_get_enabled_link_mask(device));
                break;
            case LWSWITCH_GET_INFO_INDEX_ENABLED_PORTS_MASK_63_32:
                p->info[i] = LwU64_HI32(lwswitch_get_enabled_link_mask(device));
                break;
            case LWSWITCH_GET_INFO_INDEX_NUM_VCS:
                p->info[i] = _lwswitch_get_num_vcs_lr10(device);
                break;
            case LWSWITCH_GET_INFO_INDEX_REMAP_POLICY_TABLE_SIZE:
                {
                    LwU32 remap_ram_sel;
                    LwlStatus status;

                    status = lwswitch_get_remap_table_selector(device, LWSWITCH_TABLE_SELECT_REMAP_PRIMARY, &remap_ram_sel);
                    if (status == LWL_SUCCESS)
                    {
                        p->info[i] = lwswitch_get_ingress_ram_size(device, remap_ram_sel);
                    }
                    else
                    {
                        p->info[i] = 0;
                    }
                }
                break;
            case LWSWITCH_GET_INFO_INDEX_REMAP_POLICY_EXTA_TABLE_SIZE:
                {
                    LwU32 remap_ram_sel;
                    LwlStatus status;

                    status = lwswitch_get_remap_table_selector(device, LWSWITCH_TABLE_SELECT_REMAP_EXTA, &remap_ram_sel);
                    if (status == LWL_SUCCESS)
                    {
                        p->info[i] = lwswitch_get_ingress_ram_size(device, remap_ram_sel);
                    }
                    else
                    {
                        p->info[i] = 0;
                    }
                }
                break;
            case LWSWITCH_GET_INFO_INDEX_REMAP_POLICY_EXTB_TABLE_SIZE:
                {
                    LwU32 remap_ram_sel;
                    LwlStatus status;

                    status = lwswitch_get_remap_table_selector(device, LWSWITCH_TABLE_SELECT_REMAP_EXTB, &remap_ram_sel);
                    if (status == LWL_SUCCESS)
                    {
                        p->info[i] = lwswitch_get_ingress_ram_size(device, remap_ram_sel);
                    }
                    else
                    {
                        p->info[i] = 0;
                    }
                }
                break;
            case LWSWITCH_GET_INFO_INDEX_REMAP_POLICY_MULTICAST_TABLE_SIZE:
                {
                    LwU32 remap_ram_sel;
                    LwlStatus status;

                    status = lwswitch_get_remap_table_selector(device, LWSWITCH_TABLE_SELECT_REMAP_MULTICAST, &remap_ram_sel);
                    if (status == LWL_SUCCESS)
                    {
                        p->info[i] = lwswitch_get_ingress_ram_size(device, remap_ram_sel);
                    }
                    else
                    {
                        p->info[i] = 0;
                    }
                }
                break;
            case LWSWITCH_GET_INFO_INDEX_ROUTING_ID_TABLE_SIZE:
                p->info[i] = lwswitch_get_ingress_ram_size(device, LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECTSRIDROUTERAM);
                break;
            case LWSWITCH_GET_INFO_INDEX_ROUTING_LAN_TABLE_SIZE:
                p->info[i] = lwswitch_get_ingress_ram_size(device, LW_INGRESS_REQRSPMAPADDR_RAM_SEL_SELECTSRLANROUTERAM);
                break;
            case LWSWITCH_GET_INFO_INDEX_FREQ_KHZ:
                p->info[i] = device->switch_pll.freq_khz;
                break;
            case LWSWITCH_GET_INFO_INDEX_VCOFREQ_KHZ:
                p->info[i] = device->switch_pll.vco_freq_khz;
                break;
            case LWSWITCH_GET_INFO_INDEX_VOLTAGE_MVOLT:
                p->info[i] = _lwswitch_get_info_voltage(device);
                break;
            case LWSWITCH_GET_INFO_INDEX_PHYSICAL_ID:
                p->info[i] = lwswitch_read_physical_id(device);
                break;
            case LWSWITCH_GET_INFO_INDEX_PCI_DOMAIN:
                p->info[i] = device->lwlink_device->pciInfo.domain;
                break;
            case LWSWITCH_GET_INFO_INDEX_PCI_BUS:
                p->info[i] = device->lwlink_device->pciInfo.bus;
                break;
            case LWSWITCH_GET_INFO_INDEX_PCI_DEVICE:
                p->info[i] = device->lwlink_device->pciInfo.device;
                break;
            case LWSWITCH_GET_INFO_INDEX_PCI_FUNCTION:
                p->info[i] = device->lwlink_device->pciInfo.function;
                break;
            case LWSWITCH_GET_INFO_INDEX_INFOROM_LWL_SUPPORTED:
                p->info[i] = (LwU32)_lwswitch_inforom_lwl_supported(device);
                break;
            case LWSWITCH_GET_INFO_INDEX_INFOROM_BBX_SUPPORTED:
                p->info[i] = (LwU32)_lwswitch_inforom_bbx_supported(device);
                break;
            default:
                LWSWITCH_PRINT(device, ERROR,
                    "%s: Undefined LWSWITCH_GET_INFO_INDEX 0x%x\n",
                    __FUNCTION__,
                    p->index[i]);
                retval = -LWL_BAD_ARGS;
                break;
        }
    }

    return retval;
}

LwlStatus
lwswitch_set_nport_port_config_lr10
(
    lwswitch_device *device,
    LWSWITCH_SET_SWITCH_PORT_CONFIG *p
)
{
    LwU32   val;

    if (p->requesterLinkID > DRF_MASK(LW_NPORT_REQLINKID_REQROUTINGID))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Invalid requester RID 0x%x\n",
            __FUNCTION__, p->requesterLinkID);
        return -LWL_BAD_ARGS;
    }

    if (p->requesterLanID > DRF_MASK(LW_NPORT_REQLINKID_REQROUTINGLAN))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Invalid requester RLAN 0x%x\n",
            __FUNCTION__, p->requesterLanID);
        return -LWL_BAD_ARGS;
    }

    val = LWSWITCH_LINK_RD32_LR10(device, p->portNum, NPORT, _NPORT, _CTRL);
    switch (p->type)
    {
        case CONNECT_ACCESS_GPU:
        case CONNECT_ACCESS_CPU:
        case CONNECT_ACCESS_SWITCH:
            val = FLD_SET_DRF(_NPORT, _CTRL, _TRUNKLINKENB, _ACCESSLINK, val);
            break;
        case CONNECT_TRUNK_SWITCH:
            val = FLD_SET_DRF(_NPORT, _CTRL, _TRUNKLINKENB, _TRUNKLINK, val);
            break;
        default:
            LWSWITCH_PRINT(device, ERROR,
                "%s: invalid type #%d\n",
                __FUNCTION__, p->type);
            return -LWL_BAD_ARGS;
    }

    switch(p->count)
    {
        case CONNECT_COUNT_512:
            val = FLD_SET_DRF(_NPORT, _CTRL, _ENDPOINT_COUNT, _512, val);
            break;
        case CONNECT_COUNT_1024:
            val = FLD_SET_DRF(_NPORT, _CTRL, _ENDPOINT_COUNT, _1024, val);
            break;
        case CONNECT_COUNT_2048:
            val = FLD_SET_DRF(_NPORT, _CTRL, _ENDPOINT_COUNT, _2048, val);
            break;
        default:
            LWSWITCH_PRINT(device, ERROR,
                "%s: invalid count #%d\n",
                __FUNCTION__, p->count);
            return -LWL_BAD_ARGS;
    }
    LWSWITCH_LINK_WR32(device, p->portNum, NPORT, _NPORT, _CTRL, val);

    LWSWITCH_LINK_WR32(device, p->portNum, NPORT, _NPORT, _REQLINKID,
        DRF_NUM(_NPORT, _REQLINKID, _REQROUTINGID, p->requesterLinkID) |
        DRF_NUM(_NPORT, _REQLINKID, _REQROUTINGLAN, p->requesterLanID));

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_ctrl_set_switch_port_config_lr10
(
    lwswitch_device *device,
    LWSWITCH_SET_SWITCH_PORT_CONFIG *p
)
{
    lwlink_link *link;
    LwU32 val;
    LwlStatus status;

    if (!LWSWITCH_IS_LINK_ENG_VALID(device, p->portNum, NPORT))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: invalid link #%d\n",
            __FUNCTION__, p->portNum);
        return -LWL_BAD_ARGS;
    }

    if (p->enableVC1 && (p->type != CONNECT_TRUNK_SWITCH))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: VC1 only allowed on trunk links\n",
            __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    // Validate chip-specific NPORT settings and program port config settings.
    status = lwswitch_set_nport_port_config(device, p);
    if (status != LWL_SUCCESS)
    {
        return status;
    }

    link = lwswitch_get_link(device, (LwU8)p->portNum);
    if (link == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: invalid link\n",
            __FUNCTION__);
        return -LWL_ERR_ILWALID_STATE;
    }

    //
    // If ac_coupled_mask is configured during lwswitch_create_link,
    // give preference to it.
    //
    if (device->regkeys.ac_coupled_mask  ||
        device->regkeys.ac_coupled_mask2 ||
        device->firmware.lwlink.link_ac_coupled_mask)
    {
        if (link->ac_coupled != p->acCoupled)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: port[%d]: Unsupported AC coupled change (%s)\n",
                __FUNCTION__, p->portNum, p->acCoupled ? "AC" : "DC");
            return -LWL_BAD_ARGS;
        }
    }

    link->ac_coupled = p->acCoupled;

    // AC vs DC mode SYSTEM register
    if (link->ac_coupled)
    {
        //
        // In LWL3.0, ACMODE is handled by MINION in the INITPHASE1 command
        // Here we just setup the register with the proper info
        //
        val = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLIPT_LNK,
                _LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL);
        val = FLD_SET_DRF(_LWLIPT_LNK,
                _CTRL_SYSTEM_LINK_CHANNEL_CTRL, _AC_DC_MODE, _AC, val);
        LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLIPT_LNK,
                _LWLIPT_LNK, _CTRL_SYSTEM_LINK_CHANNEL_CTRL, val);
    }

    // If _BUFFER_RDY is asserted, credits are locked.
    val = LWSWITCH_LINK_RD32_LR10(device, p->portNum, NPORT, _NPORT, _CTRL_BUFFER_READY);
    if (FLD_TEST_DRF(_NPORT, _CTRL_BUFFER_READY, _BUFFERRDY, _ENABLE, val))
    {
        LWSWITCH_PRINT(device, SETUP,
            "%s: port[%d]: BUFFERRDY already enabled.\n",
            __FUNCTION__, p->portNum);
        return LWL_SUCCESS;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_ctrl_set_ingress_request_table_lr10
(
    lwswitch_device *device,
    LWSWITCH_SET_INGRESS_REQUEST_TABLE *p
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_ctrl_get_ingress_request_table_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_INGRESS_REQUEST_TABLE_PARAMS *params
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_ctrl_set_ingress_request_valid_lr10
(
    lwswitch_device *device,
    LWSWITCH_SET_INGRESS_REQUEST_VALID *p
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_ctrl_get_ingress_response_table_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_INGRESS_RESPONSE_TABLE_PARAMS *params
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}


LwlStatus
lwswitch_ctrl_set_ingress_response_table_lr10
(
    lwswitch_device *device,
    LWSWITCH_SET_INGRESS_RESPONSE_TABLE *p
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

static LwlStatus
lwswitch_ctrl_set_ganged_link_table_lr10
(
    lwswitch_device *device,
    LWSWITCH_SET_GANGED_LINK_TABLE *p
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

static LwlStatus
lwswitch_ctrl_get_internal_latency_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_INTERNAL_LATENCY *pLatency
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LwU32 vc_selector = pLatency->vc_selector;
    LwU32 idx_nport;

    // Validate VC selector
    if (vc_selector >= LWSWITCH_NUM_VCS_LR10)
    {
        return -LWL_BAD_ARGS;
    }

    lwswitch_os_memset(pLatency, 0, sizeof(*pLatency));
    pLatency->vc_selector = vc_selector;

    for (idx_nport=0; idx_nport < LWSWITCH_LINK_COUNT(device); idx_nport++)
    {
        if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, NPORT, idx_nport))
        {
            continue;
        }

        pLatency->egressHistogram[idx_nport].low =
            chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].low;
        pLatency->egressHistogram[idx_nport].medium =
            chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].medium;
        pLatency->egressHistogram[idx_nport].high =
           chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].high;
        pLatency->egressHistogram[idx_nport].panic =
           chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].panic;
        pLatency->egressHistogram[idx_nport].count =
           chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].count;
    }

    pLatency->elapsed_time_msec =
      (chip_device->latency_stats->latency[vc_selector].last_read_time_nsec -
       chip_device->latency_stats->latency[vc_selector].start_time_nsec)/1000000ULL;

    chip_device->latency_stats->latency[vc_selector].start_time_nsec =
        chip_device->latency_stats->latency[vc_selector].last_read_time_nsec;

    chip_device->latency_stats->latency[vc_selector].count = 0;

    // Clear aclwm_latency[]
    for (idx_nport = 0; idx_nport < LWSWITCH_LINK_COUNT(device); idx_nport++)
    {
        chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].low = 0;
        chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].medium = 0;
        chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].high = 0;
        chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].panic = 0;
        chip_device->latency_stats->latency[vc_selector].aclwm_latency[idx_nport].count = 0;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_ctrl_set_latency_bins_lr10
(
    lwswitch_device *device,
    LWSWITCH_SET_LATENCY_BINS *pLatency
)
{
    LwU32 vc_selector;
    const LwU32 freq_mhz = 1330;
    const LwU32 switchpll_hz = freq_mhz * 1000000ULL; // TODO: Update this with device->switch_pll.freq_khz after LR10 PLL update
    const LwU32 min_threshold = 10;   // Must be > zero to avoid div by zero
    const LwU32 max_threshold = 10000;

    // Quick input validation and ns to register value colwersion
    for (vc_selector = 0; vc_selector < LWSWITCH_NUM_VCS_LR10; vc_selector++)
    {
        if ((pLatency->bin[vc_selector].lowThreshold > max_threshold)                           ||
            (pLatency->bin[vc_selector].lowThreshold < min_threshold)                           ||
            (pLatency->bin[vc_selector].medThreshold > max_threshold)                           ||
            (pLatency->bin[vc_selector].medThreshold < min_threshold)                           ||
            (pLatency->bin[vc_selector].hiThreshold  > max_threshold)                           ||
            (pLatency->bin[vc_selector].hiThreshold  < min_threshold)                           ||
            (pLatency->bin[vc_selector].lowThreshold > pLatency->bin[vc_selector].medThreshold) ||
            (pLatency->bin[vc_selector].medThreshold > pLatency->bin[vc_selector].hiThreshold))
        {
            return -LWL_BAD_ARGS;
        }

        pLatency->bin[vc_selector].lowThreshold =
            switchpll_hz / (1000000000 / pLatency->bin[vc_selector].lowThreshold);
        pLatency->bin[vc_selector].medThreshold =
            switchpll_hz / (1000000000 / pLatency->bin[vc_selector].medThreshold);
        pLatency->bin[vc_selector].hiThreshold =
            switchpll_hz / (1000000000 / pLatency->bin[vc_selector].hiThreshold);

        LWSWITCH_PORTSTAT_BCAST_WR32_LR10(device, _LIMIT, _LOW,    vc_selector, pLatency->bin[vc_selector].lowThreshold);
        LWSWITCH_PORTSTAT_BCAST_WR32_LR10(device, _LIMIT, _MEDIUM, vc_selector, pLatency->bin[vc_selector].medThreshold);
        LWSWITCH_PORTSTAT_BCAST_WR32_LR10(device, _LIMIT, _HIGH,   vc_selector, pLatency->bin[vc_selector].hiThreshold);
    }

    return LWL_SUCCESS;
}

#define LW_NPORT_REQLINKID_REQROUTINGLAN_1024  18:18
#define LW_NPORT_REQLINKID_REQROUTINGLAN_2048  18:17

/*
 * @brief Returns the ingress requester link id.
 *
 * On LR10, REQROUTINGID only gives the endpoint but not the specific port of the response packet.
 * To identify the specific port, the routing_ID must be appended with the upper bits of REQROUTINGLAN.
 *
 * When LW_NPORT_CTRL_ENDPOINT_COUNT = 1024, the upper bit of LW_NPORT_REQLINKID_REQROUTINGLAN become REQROUTINGID[9].
 * When LW_NPORT_CTRL_ENDPOINT_COUNT = 2048, the upper two bits of LW_NPORT_REQLINKID_REQROUTINGLAN become REQROUTINGID[10:9].
 *
 * @param[in] device            lwswitch device
 * @param[in] params            LWSWITCH_GET_INGRESS_REQLINKID_PARAMS
 *
 * @returns                     LWL_SUCCESS if action succeeded,
 *                              -LWL_ERR_ILWALID_STATE invalid link
 */
LwlStatus
lwswitch_ctrl_get_ingress_reqlinkid_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_INGRESS_REQLINKID_PARAMS *params
)
{
    LwU32 regval;
    LwU32 reqRid;
    LwU32 reqRlan;
    LwU32 rlan_shift = DRF_SHIFT_RT(LW_NPORT_REQLINKID_REQROUTINGID) + 1;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, NPORT, params->portNum))
    {
        return -LWL_BAD_ARGS;
    }

    regval = LWSWITCH_NPORT_RD32_LR10(device, params->portNum, _NPORT, _REQLINKID);
    reqRid = DRF_VAL(_NPORT, _REQLINKID, _REQROUTINGID, regval);
    reqRlan = regval;

    regval = LWSWITCH_NPORT_RD32_LR10(device, params->portNum, _NPORT, _CTRL);
    if (FLD_TEST_DRF(_NPORT, _CTRL, _ENDPOINT_COUNT, _1024, regval))
    {
        reqRlan = DRF_VAL(_NPORT, _REQLINKID, _REQROUTINGLAN_1024, reqRlan);
        params->requesterLinkID = (reqRid | (reqRlan << rlan_shift));
    }
    else if (FLD_TEST_DRF(_NPORT, _CTRL, _ENDPOINT_COUNT, _2048, regval))
    {
        reqRlan = DRF_VAL(_NPORT, _REQLINKID, _REQROUTINGLAN_2048, reqRlan);
        params->requesterLinkID = (reqRid | (reqRlan << rlan_shift));
    }
    else
    {
        params->requesterLinkID = reqRid;
    }

    return LWL_SUCCESS;
}

//
// MODS-only IOCTLS
//

/*
 * REGISTER_READ/_WRITE
 * Provides direct access to the MMIO space for trusted clients like MODS.
 * This API should not be exposed to unselwre clients.
 */

/*
 * _lwswitch_get_engine_base
 * Used by REGISTER_READ/WRITE API.  Looks up an engine based on device/instance
 * and returns the base address in BAR0.
 *
 * register_rw_engine   [in] REGISTER_RW_ENGINE_*
 * instance             [in] physical instance of device
 * bcast                [in] FALSE: find unicast base address
 *                           TRUE:  find broadcast base address
 * base_addr            [out] base address in BAR0 of requested device
 *
 * Returns              LWL_SUCCESS: Device base address successfully found
 *                      else device lookup failed
 */

static LwlStatus
_lwswitch_get_engine_base_lr10
(
    lwswitch_device *device,
    LwU32   register_rw_engine,     // REGISTER_RW_ENGINE_*
    LwU32   instance,               // device instance
    LwBool  bcast,
    LwU32   *base_addr
)
{
    LwU32 base = 0;
    ENGINE_DESCRIPTOR_TYPE_LR10  *engine = NULL;
    LwlStatus retval = LWL_SUCCESS;
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);

    // Find the engine descriptor matching the request
    engine = NULL;

    switch (register_rw_engine)
    {
        case REGISTER_RW_ENGINE_RAW:
            // Special case raw IO
            if ((instance != 0) ||
                (bcast != LW_FALSE))
            {
                retval = -LWL_BAD_ARGS;
            }
        break;

        case REGISTER_RW_ENGINE_CLKS:
        case REGISTER_RW_ENGINE_FUSE:
        case REGISTER_RW_ENGINE_JTAG:
        case REGISTER_RW_ENGINE_PMGR:
        case REGISTER_RW_ENGINE_XP3G:
            //
            // Legacy devices are always single-instance, unicast-only.
            // These manuals are BAR0 offset-based, not IP-based.  Treat them
            // the same as RAW.
            //
            if ((instance != 0) ||
                (bcast != LW_FALSE))
            {
                retval = -LWL_BAD_ARGS;
            }
            register_rw_engine = REGISTER_RW_ENGINE_RAW;
        break;

        case REGISTER_RW_ENGINE_SAW:
            if (bcast)
            {
                retval = -LWL_BAD_ARGS;
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, SAW, instance))
                {
                    engine = &chip_device->engSAW[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_XVE:
            if (bcast)
            {
                retval = -LWL_BAD_ARGS;
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, XVE, instance))
                {
                    engine = &chip_device->engXVE[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_SOE:
            if (bcast)
            {
                retval = -LWL_BAD_ARGS;
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, SOE, instance))
                {
                    engine = &chip_device->engSOE[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_SE:
            if (bcast)
            {
                retval = -LWL_BAD_ARGS;
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, SE, instance))
                {
                    engine = &chip_device->engSE[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_LWLW:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, LWLW_BCAST, instance))
                {
                    engine = &chip_device->engLWLW_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, LWLW, instance))
                {
                    engine = &chip_device->engLWLW[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_MINION:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, MINION_BCAST, instance))
                {
                    engine = &chip_device->engMINION_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, MINION, instance))
                {
                    engine = &chip_device->engMINION[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_LWLIPT:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, LWLIPT_BCAST, instance))
                {
                    engine = &chip_device->engLWLIPT_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, LWLIPT, instance))
                {
                    engine = &chip_device->engLWLIPT[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_LWLTLC:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, LWLTLC_BCAST, instance))
                {
                    engine = &chip_device->engLWLTLC_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, LWLTLC, instance))
                {
                    engine = &chip_device->engLWLTLC[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_LWLTLC_MULTICAST:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, LWLTLC_MULTICAST_BCAST, instance))
                {
                    engine = &chip_device->engLWLTLC_MULTICAST_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, LWLTLC_MULTICAST, instance))
                {
                    engine = &chip_device->engLWLTLC_MULTICAST[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_TX_PERFMON:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, TX_PERFMON_BCAST, instance))
                {
                    engine = &chip_device->engTX_PERFMON_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, TX_PERFMON, instance))
                {
                    engine = &chip_device->engTX_PERFMON[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_TX_PERFMON_MULTICAST:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, TX_PERFMON_MULTICAST_BCAST, instance))
                {
                    engine = &chip_device->engTX_PERFMON_MULTICAST_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, TX_PERFMON_MULTICAST, instance))
                {
                    engine = &chip_device->engTX_PERFMON_MULTICAST[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_RX_PERFMON:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, RX_PERFMON_BCAST, instance))
                {
                    engine = &chip_device->engRX_PERFMON_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, RX_PERFMON, instance))
                {
                    engine = &chip_device->engRX_PERFMON[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_RX_PERFMON_MULTICAST:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, RX_PERFMON_MULTICAST_BCAST, instance))
                {
                    engine = &chip_device->engRX_PERFMON_MULTICAST_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, RX_PERFMON_MULTICAST, instance))
                {
                    engine = &chip_device->engRX_PERFMON_MULTICAST[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_NPG:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, NPG_BCAST, instance))
                {
                    engine = &chip_device->engNPG_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, NPG, instance))
                {
                    engine = &chip_device->engNPG[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_NPORT:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, NPORT_BCAST, instance))
                {
                    engine = &chip_device->engNPORT_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, NPORT, instance))
                {
                    engine = &chip_device->engNPORT[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_NPORT_MULTICAST:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, NPORT_MULTICAST_BCAST, instance))
                {
                    engine = &chip_device->engNPORT_MULTICAST_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, NPORT_MULTICAST, instance))
                {
                    engine = &chip_device->engNPORT_MULTICAST[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_NPG_PERFMON:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, NPG_PERFMON_BCAST, instance))
                {
                    engine = &chip_device->engNPG_PERFMON_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, NPG_PERFMON, instance))
                {
                    engine = &chip_device->engNPG_PERFMON[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_NPORT_PERFMON:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, NPORT_PERFMON_BCAST, instance))
                {
                    engine = &chip_device->engNPORT_PERFMON_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, NPORT_PERFMON, instance))
                {
                    engine = &chip_device->engNPORT_PERFMON[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_NPORT_PERFMON_MULTICAST:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, NPORT_PERFMON_MULTICAST_BCAST, instance))
                {
                    engine = &chip_device->engNPORT_PERFMON_MULTICAST_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, NPORT_PERFMON_MULTICAST, instance))
                {
                    engine = &chip_device->engNPORT_PERFMON_MULTICAST[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_LWLIPT_LNK:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, LWLIPT_LNK_BCAST, instance))
                {
                    engine = &chip_device->engLWLIPT_LNK_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, LWLIPT_LNK, instance))
                {
                    engine = &chip_device->engLWLIPT_LNK[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_LWLIPT_LNK_MULTICAST:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, LWLIPT_LNK_MULTICAST_BCAST, instance))
                {
                    engine = &chip_device->engLWLIPT_LNK_MULTICAST_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, LWLIPT_LNK_MULTICAST, instance))
                {
                    engine = &chip_device->engLWLIPT_LNK_MULTICAST[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_PLL:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, PLL_BCAST, instance))
                {
                    engine = &chip_device->engPLL_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, PLL, instance))
                {
                    engine = &chip_device->engPLL[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_LWLDL:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, LWLDL_BCAST, instance))
                {
                    engine = &chip_device->engLWLDL_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, LWLDL, instance))
                {
                    engine = &chip_device->engLWLDL[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_LWLDL_MULTICAST:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, LWLDL_MULTICAST_BCAST, instance))
                {
                    engine = &chip_device->engLWLDL_MULTICAST_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, LWLDL_MULTICAST, instance))
                {
                    engine = &chip_device->engLWLDL_MULTICAST[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_NXBAR:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, NXBAR_BCAST, instance))
                {
                    engine = &chip_device->engNXBAR_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, NXBAR, instance))
                {
                    engine = &chip_device->engNXBAR[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_NXBAR_PERFMON:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, NXBAR_PERFMON_BCAST, instance))
                {
                    engine = &chip_device->engNXBAR_PERFMON_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, NXBAR_PERFMON, instance))
                {
                    engine = &chip_device->engNXBAR_PERFMON[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_TILE:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, TILE_BCAST, instance))
                {
                    engine = &chip_device->engTILE_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, TILE, instance))
                {
                    engine = &chip_device->engTILE[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_TILE_MULTICAST:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, TILE_MULTICAST_BCAST, instance))
                {
                    engine = &chip_device->engTILE_MULTICAST_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, TILE_MULTICAST, instance))
                {
                    engine = &chip_device->engTILE_MULTICAST[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_TILE_PERFMON:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, TILE_PERFMON_BCAST, instance))
                {
                    engine = &chip_device->engTILE_PERFMON_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, TILE_PERFMON, instance))
                {
                    engine = &chip_device->engTILE_PERFMON[instance];
                }
            }
        break;

        case REGISTER_RW_ENGINE_TILE_PERFMON_MULTICAST:
            if (bcast)
            {
                if (LWSWITCH_ENG_VALID_LR10(device, TILE_PERFMON_MULTICAST_BCAST, instance))
                {
                    engine = &chip_device->engTILE_PERFMON_MULTICAST_BCAST[instance];
                }
            }
            else
            {
                if (LWSWITCH_ENG_VALID_LR10(device, TILE_PERFMON_MULTICAST, instance))
                {
                    engine = &chip_device->engTILE_PERFMON_MULTICAST[instance];
                }
            }
        break;

        default:
            LWSWITCH_PRINT(device, ERROR,
                "%s: unknown REGISTER_RW_ENGINE 0x%x\n",
                __FUNCTION__,
                register_rw_engine);
            engine = NULL;
        break;
    }

    if (register_rw_engine == REGISTER_RW_ENGINE_RAW)
    {
        // Raw IO -- client provides full BAR0 offset
        base = 0;
    }
    else
    {
        // Check engine descriptor was found and valid
        if (engine == NULL)
        {
            retval = -LWL_BAD_ARGS;
            LWSWITCH_PRINT(device, ERROR,
                "%s: invalid REGISTER_RW_ENGINE/instance 0x%x(%d)\n",
                __FUNCTION__,
                register_rw_engine,
                instance);
        }
        else if (!engine->valid)
        {
            retval = -LWL_UNBOUND_DEVICE;
            LWSWITCH_PRINT(device, ERROR,
                "%s: REGISTER_RW_ENGINE/instance 0x%x(%d) disabled or invalid\n",
                __FUNCTION__,
                register_rw_engine,
                instance);
        }
        else
        {
            if (bcast && (engine->disc_type == DISCOVERY_TYPE_BROADCAST))
            {
                //
                // Caveat emptor: A read of a broadcast register is
                // implementation-specific.
                //
                base = engine->info.bc.bc_addr;
            }
            else if ((!bcast) && (engine->disc_type == DISCOVERY_TYPE_UNICAST))
            {
                base = engine->info.uc.uc_addr;
            }

            if (base == 0)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: REGISTER_RW_ENGINE/instance 0x%x(%d) has %s base address 0!\n",
                    __FUNCTION__,
                    register_rw_engine,
                    instance,
                    (bcast ? "BCAST" : "UNICAST" ));
                retval = -LWL_IO_ERROR;
            }
        }
    }

    *base_addr = base;
    return retval;
}

/*
 * CTRL_LWSWITCH_REGISTER_READ
 *
 * This provides direct access to the MMIO space for trusted clients like
 * MODS.
 * This API should not be exposed to unselwre clients.
 */

static LwlStatus
lwswitch_ctrl_register_read_lr10
(
    lwswitch_device *device,
    LWSWITCH_REGISTER_READ *p
)
{
    LwU32 base;
    LwU32 data;
    LwlStatus retval = LWL_SUCCESS;

    retval = _lwswitch_get_engine_base_lr10(device, p->engine, p->instance, LW_FALSE, &base);
    if (retval != LWL_SUCCESS)
    {
        return retval;
    }

    // Make sure target offset isn't out-of-range
    if ((base + p->offset) >= device->lwlink_device->pciInfo.bars[0].barSize)
    {
        return -LWL_IO_ERROR;
    }

    //
    // Some legacy device manuals are not 0-based (IP style).
    //
    data = LWSWITCH_OFF_RD32(device, base + p->offset);
    p->val = data;

    return LWL_SUCCESS;
}

/*
 * CTRL_LWSWITCH_REGISTER_WRITE
 *
 * This provides direct access to the MMIO space for trusted clients like
 * MODS.
 * This API should not be exposed to unselwre clients.
 */

static LwlStatus
lwswitch_ctrl_register_write_lr10
(
    lwswitch_device *device,
    LWSWITCH_REGISTER_WRITE *p
)
{
    LwU32 base;
    LwlStatus retval = LWL_SUCCESS;

    retval = _lwswitch_get_engine_base_lr10(device, p->engine, p->instance, p->bcast, &base);
    if (retval != LWL_SUCCESS)
    {
        return retval;
    }

    // Make sure target offset isn't out-of-range
    if ((base + p->offset) >= device->lwlink_device->pciInfo.bars[0].barSize)
    {
        return -LWL_IO_ERROR;
    }

    //
    // Some legacy device manuals are not 0-based (IP style).
    //
    LWSWITCH_OFF_WR32(device, base + p->offset, p->val);

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_ctrl_get_bios_info_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_BIOS_INFO_PARAMS *p
)
{
    LwU32 biosVersionBytes;
    LwU32 biosOemVersionBytes;
    LwU32 biosMagic = 0x9210;

    //
    // Example: 92.10.09.00.00 is the formatted version string
    //          |         |  |
    //          |         |  |__ BIOS OEM version byte
    //          |         |
    //          |_________|_____ BIOS version bytes
    //
    biosVersionBytes = LWSWITCH_SAW_RD32_LR10(device, _LWLSAW_SW, _SCRATCH_6);
    biosOemVersionBytes = LWSWITCH_SAW_RD32_LR10(device, _LWLSAW_SW, _SCRATCH_7);

    //
    // LR10 is built out of core92 and the BIOS version will always begin with
    // 92.10.xx.xx.xx
    //
    if ((biosVersionBytes >> 16) != biosMagic)
    {
        LWSWITCH_PRINT(device, ERROR,
                "BIOS version not found in scratch register\n");
        return -LWL_ERR_ILWALID_STATE;
    }

    p->version = (((LwU64)biosVersionBytes) << 8) | (biosOemVersionBytes & 0xff);

    return LWL_SUCCESS;
}

static void
_lwlink_clear_corelib_state
(
    lwlink_link *link
)
{
    // Receiver Detect needs to happen again
    link->bRxDetected = LW_FALSE;

    // INITNEGOTIATE needs to happen again
    link->bInitnegotiateConfigGood = LW_FALSE;

    // TxCommonMode needs to happen again
    link->bTxCommonModeFail = LW_FALSE;

    // SAFE transition needs to happen again
    link->bSafeTransitionFail = LW_FALSE;

    // Reset the SW state tracking the link and sublink states
    link->state            = LWLINK_LINKSTATE_OFF;
    link->tx_sublink_state = LWLINK_SUBLINK_STATE_TX_OFF;
    link->rx_sublink_state = LWLINK_SUBLINK_STATE_RX_OFF;
}

const static LwU32 nport_reg_addr[] =
{
    LW_NPORT_CTRL,
    LW_NPORT_CTRL_SLCG,
    LW_NPORT_REQLINKID,
    LW_NPORT_PORTSTAT_CONTROL,
    LW_NPORT_PORTSTAT_SNAP_CONTROL,
    LW_NPORT_PORTSTAT_WINDOW_LIMIT,
    LW_NPORT_PORTSTAT_LIMIT_LOW_0,
    LW_NPORT_PORTSTAT_LIMIT_MEDIUM_0,
    LW_NPORT_PORTSTAT_LIMIT_HIGH_0,
    LW_NPORT_PORTSTAT_LIMIT_LOW_1,
    LW_NPORT_PORTSTAT_LIMIT_MEDIUM_1,
    LW_NPORT_PORTSTAT_LIMIT_HIGH_1,
    LW_NPORT_PORTSTAT_LIMIT_LOW_2,
    LW_NPORT_PORTSTAT_LIMIT_MEDIUM_2,
    LW_NPORT_PORTSTAT_LIMIT_HIGH_2,
    LW_NPORT_PORTSTAT_LIMIT_LOW_3,
    LW_NPORT_PORTSTAT_LIMIT_MEDIUM_3,
    LW_NPORT_PORTSTAT_LIMIT_HIGH_3,
    LW_NPORT_PORTSTAT_LIMIT_LOW_4,
    LW_NPORT_PORTSTAT_LIMIT_MEDIUM_4,
    LW_NPORT_PORTSTAT_LIMIT_HIGH_4,
    LW_NPORT_PORTSTAT_LIMIT_LOW_5,
    LW_NPORT_PORTSTAT_LIMIT_MEDIUM_5,
    LW_NPORT_PORTSTAT_LIMIT_HIGH_5,
    LW_NPORT_PORTSTAT_LIMIT_LOW_6,
    LW_NPORT_PORTSTAT_LIMIT_MEDIUM_6,
    LW_NPORT_PORTSTAT_LIMIT_HIGH_6,
    LW_NPORT_PORTSTAT_LIMIT_LOW_7,
    LW_NPORT_PORTSTAT_LIMIT_MEDIUM_7,
    LW_NPORT_PORTSTAT_LIMIT_HIGH_7,
    LW_NPORT_PORTSTAT_SOURCE_FILTER_0,
    LW_NPORT_PORTSTAT_SOURCE_FILTER_1,
    LW_ROUTE_ROUTE_CONTROL,
    LW_ROUTE_CMD_ROUTE_TABLE0,
    LW_ROUTE_CMD_ROUTE_TABLE1,
    LW_ROUTE_CMD_ROUTE_TABLE2,
    LW_ROUTE_CMD_ROUTE_TABLE3,
    LW_ROUTE_ERR_LOG_EN_0,
    LW_ROUTE_ERR_CONTAIN_EN_0,
    LW_ROUTE_ERR_ECC_CTRL,
    LW_ROUTE_ERR_GLT_ECC_ERROR_COUNTER_LIMIT,
    LW_ROUTE_ERR_LWS_ECC_ERROR_COUNTER_LIMIT,
    LW_INGRESS_ERR_LOG_EN_0,
    LW_INGRESS_ERR_CONTAIN_EN_0,
    LW_INGRESS_ERR_ECC_CTRL,
    LW_INGRESS_ERR_REMAPTAB_ECC_ERROR_COUNTER_LIMIT,
    LW_INGRESS_ERR_RIDTAB_ECC_ERROR_COUNTER_LIMIT,
    LW_INGRESS_ERR_RLANTAB_ECC_ERROR_COUNTER_LIMIT,
    LW_INGRESS_ERR_NCISOC_HDR_ECC_ERROR_COUNTER_LIMIT,
    LW_EGRESS_CTRL,
    LW_EGRESS_CTO_TIMER_LIMIT,
    LW_EGRESS_ERR_LOG_EN_0,
    LW_EGRESS_ERR_CONTAIN_EN_0,
    LW_EGRESS_ERR_ECC_CTRL,
    LW_EGRESS_ERR_NXBAR_ECC_ERROR_COUNTER_LIMIT,
    LW_EGRESS_ERR_RAM_OUT_ECC_ERROR_COUNTER_LIMIT,
    LW_TSTATE_TAGSTATECONTROL,
    LW_TSTATE_ATO_TIMER_LIMIT,
    LW_TSTATE_CREQ_CAM_LOCK,
    LW_TSTATE_ERR_LOG_EN_0,
    LW_TSTATE_ERR_CONTAIN_EN_0,
    LW_TSTATE_ERR_ECC_CTRL,
    LW_TSTATE_ERR_CRUMBSTORE_ECC_ERROR_COUNTER_LIMIT,
    LW_TSTATE_ERR_TAGPOOL_ECC_ERROR_COUNTER_LIMIT,
    LW_TSTATE_ERR_TD_TID_RAM_ECC_ERROR_COUNTER_LIMIT,
    LW_SOURCETRACK_CTRL,
    LW_SOURCETRACK_MULTISEC_TIMER0,
    LW_SOURCETRACK_ERR_LOG_EN_0,
    LW_SOURCETRACK_ERR_CONTAIN_EN_0,
    LW_SOURCETRACK_ERR_ECC_CTRL,
    LW_SOURCETRACK_ERR_CREQ_TCEN0_CRUMBSTORE_ECC_ERROR_COUNTER_LIMIT,
    LW_SOURCETRACK_ERR_CREQ_TCEN0_TD_CRUMBSTORE_ECC_ERROR_COUNTER_LIMIT,
    LW_SOURCETRACK_ERR_CREQ_TCEN1_CRUMBSTORE_ECC_ERROR_COUNTER_LIMIT,
};

/*
 *  Disable interrupts comming from NPG & LWLW blocks.
 */
static void
_lwswitch_link_disable_interrupts_lr10
(
    lwswitch_device *device,
    LwU32 link
)
{
    LwU32 i;

    LWSWITCH_NPORT_WR32_LR10(device, link, _NPORT, _ERR_CONTROL_COMMON_NPORT,
        DRF_NUM(_NPORT, _ERR_CONTROL_COMMON_NPORT, _CORRECTABLEENABLE, 0x0) |
        DRF_NUM(_NPORT, _ERR_CONTROL_COMMON_NPORT, _FATALENABLE, 0x0) |
        DRF_NUM(_NPORT, _ERR_CONTROL_COMMON_NPORT, _NONFATALENABLE, 0x0));

    for (i = 0; i < LW_LWLCTRL_LINK_INTR_0_STATUS__SIZE_1; i++)
    {
        LWSWITCH_LINK_WR32_LR10(device, link, LWLW, _LWLCTRL, _LINK_INTR_0_MASK(i),
            DRF_NUM(_LWLCTRL, _LINK_INTR_0_MASK, _FATAL, 0x0) |
            DRF_NUM(_LWLCTRL, _LINK_INTR_0_MASK, _NONFATAL, 0x0) |
            DRF_NUM(_LWLCTRL, _LINK_INTR_0_MASK, _CORRECTABLE, 0x0));

        LWSWITCH_LINK_WR32_LR10(device, link, LWLW, _LWLCTRL, _LINK_INTR_1_MASK(i),
            DRF_NUM(_LWLCTRL, _LINK_INTR_1_MASK, _FATAL, 0x0) |
            DRF_NUM(_LWLCTRL, _LINK_INTR_1_MASK, _NONFATAL, 0x0) |
            DRF_NUM(_LWLCTRL, _LINK_INTR_1_MASK, _CORRECTABLE, 0x0));

        LWSWITCH_LINK_WR32_LR10(device, link, LWLW, _LWLCTRL, _LINK_INTR_2_MASK(i),
            DRF_NUM(_LWLCTRL, _LINK_INTR_2_MASK, _FATAL, 0x0) |
            DRF_NUM(_LWLCTRL, _LINK_INTR_2_MASK, _NONFATAL, 0x0) |
            DRF_NUM(_LWLCTRL, _LINK_INTR_2_MASK, _CORRECTABLE, 0x0));
    }
}

/*
 *  Reset NPG & LWLW interrupt state.
 */
static void
_lwswitch_link_reset_interrupts_lr10
(
    lwswitch_device *device,
    LwU32 link
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LwU32 i;

    LWSWITCH_NPORT_WR32_LR10(device, link, _NPORT, _ERR_CONTROL_COMMON_NPORT,
        DRF_NUM(_NPORT, _ERR_CONTROL_COMMON_NPORT, _CORRECTABLEENABLE, 0x1) |
        DRF_NUM(_NPORT, _ERR_CONTROL_COMMON_NPORT, _FATALENABLE, 0x1) |
        DRF_NUM(_NPORT, _ERR_CONTROL_COMMON_NPORT, _NONFATALENABLE, 0x1));

    for (i = 0; i < LW_LWLCTRL_LINK_INTR_0_STATUS__SIZE_1; i++)
    {
        LWSWITCH_LINK_WR32_LR10(device, link, LWLW, _LWLCTRL, _LINK_INTR_0_MASK(i),
            DRF_NUM(_LWLCTRL, _LINK_INTR_0_MASK, _FATAL, 0x1) |
            DRF_NUM(_LWLCTRL, _LINK_INTR_0_MASK, _NONFATAL, 0x1) |
            DRF_NUM(_LWLCTRL, _LINK_INTR_0_MASK, _CORRECTABLE, 0x1));

        LWSWITCH_LINK_WR32_LR10(device, link, LWLW, _LWLCTRL, _LINK_INTR_1_MASK(i),
            DRF_NUM(_LWLCTRL, _LINK_INTR_1_MASK, _FATAL, 0x1) |
            DRF_NUM(_LWLCTRL, _LINK_INTR_1_MASK, _NONFATAL, 0x1) |
            DRF_NUM(_LWLCTRL, _LINK_INTR_1_MASK, _CORRECTABLE, 0x1));

        LWSWITCH_LINK_WR32_LR10(device, link, LWLW, _LWLCTRL, _LINK_INTR_2_MASK(i),
            DRF_NUM(_LWLCTRL, _LINK_INTR_2_MASK, _FATAL, 0x1) |
            DRF_NUM(_LWLCTRL, _LINK_INTR_2_MASK, _NONFATAL, 0x1) |
            DRF_NUM(_LWLCTRL, _LINK_INTR_2_MASK, _CORRECTABLE, 0x1));
    }

    // Enable interrupts which are disabled to prevent interrupt storm.
    LWSWITCH_NPORT_WR32_LR10(device, link, _ROUTE, _ERR_FATAL_REPORT_EN_0, chip_device->intr_mask.route.fatal);
    LWSWITCH_NPORT_WR32_LR10(device, link, _ROUTE, _ERR_NON_FATAL_REPORT_EN_0, chip_device->intr_mask.route.nonfatal);
    LWSWITCH_NPORT_WR32_LR10(device, link, _INGRESS, _ERR_FATAL_REPORT_EN_0, chip_device->intr_mask.ingress.fatal);
    LWSWITCH_NPORT_WR32_LR10(device, link, _INGRESS, _ERR_NON_FATAL_REPORT_EN_0, chip_device->intr_mask.ingress.nonfatal);
    LWSWITCH_NPORT_WR32_LR10(device, link, _EGRESS, _ERR_FATAL_REPORT_EN_0, chip_device->intr_mask.egress.fatal);
    LWSWITCH_NPORT_WR32_LR10(device, link, _EGRESS, _ERR_NON_FATAL_REPORT_EN_0, chip_device->intr_mask.egress.nonfatal);
    LWSWITCH_NPORT_WR32_LR10(device, link, _TSTATE, _ERR_FATAL_REPORT_EN_0, chip_device->intr_mask.tstate.fatal);
    LWSWITCH_NPORT_WR32_LR10(device, link, _TSTATE, _ERR_NON_FATAL_REPORT_EN_0, chip_device->intr_mask.tstate.nonfatal);
    LWSWITCH_NPORT_WR32_LR10(device, link, _SOURCETRACK, _ERR_FATAL_REPORT_EN_0, chip_device->intr_mask.sourcetrack.fatal);
    LWSWITCH_NPORT_WR32_LR10(device, link, _SOURCETRACK, _ERR_NON_FATAL_REPORT_EN_0, chip_device->intr_mask.sourcetrack.nonfatal);

    // Clear fatal error status
    device->link[link].fatal_error_oclwrred = LW_FALSE;
}

/*
 * @Brief : Control to reset and drain the links.
 *
 * @param[in] device        A reference to the device to initialize
 * @param[in] linkMask      A mask of link(s) to be reset.
 *
 * @returns :               LWL_SUCCESS if there were no errors
 *                         -LWL_BAD_PARAMS if input parameters are wrong.
 *                         -LWL_ERR_ILWALID_STATE if other errors are present and a full-chip reset is required.
 *                         -LWL_INITIALIZATION_TOTAL_FAILURE if NPORT initialization failed and a retry is required.
 */

LwlStatus
lwswitch_reset_and_drain_links_lr10
(
    lwswitch_device *device,
    LwU64 link_mask
)
{
    LwlStatus status = -LWL_ERR_GENERIC;
    lwlink_link *link_info;
    LwU32 val;
    LwU32 link;
    LwU32 idx_nport;
    LwU32 npg;
    LWSWITCH_TIMEOUT timeout;
    LwBool           keepPolling;
    LwU32 i;
    LwU64 link_mode, tx_sublink_mode, rx_sublink_mode;
    LwU32 tx_sublink_submode, rx_sublink_submode;
    LwU32 *nport_reg_val = NULL;
    LwU32 reg_count = LW_ARRAY_ELEMENTS(nport_reg_addr);
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LwU64 managedLinkMask;
    LwU64 linkPartners;

    managedLinkMask = 0;

    if (lwswitch_is_cci_supported(device))
    {
        // For managed links, we need to determine all links that need to be reset
        FOR_EACH_INDEX_IN_MASK(64, link, link_mask)
        {    
            if (!cciIsLinkManaged(device, link))
            {
                continue;
            }

            status = cciGetLinkPartners(device, link, &linkPartners);
            if (status != LWL_SUCCESS)
            {
                return status;
            }

            managedLinkMask |= linkPartners;
        }
        FOR_EACH_INDEX_IN_MASK_END;

        LWSWITCH_PRINT(device, INFO,
            "%s: Input link mask 0x%llx.\n",
            __FUNCTION__, link_mask);
        
        link_mask |= managedLinkMask;  
        LWSWITCH_PRINT(device, INFO,
            "%s: Links 0x%llx will be reset.\n",
            __FUNCTION__, link_mask);  
    }
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

    if ((link_mask == 0) ||
        (link_mask >> LWSWITCH_LINK_COUNT(device)))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Invalid link_mask = 0x%llx\n",
            __FUNCTION__, link_mask);

        return -LWL_BAD_ARGS;
    }

    // Check for in-active links
    FOR_EACH_INDEX_IN_MASK(64, link, link_mask)
    {
        if (!lwswitch_is_link_valid(device, link))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: link #%d invalid\n",
                __FUNCTION__, link);

            continue;
        }
        if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, NPORT, link))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: NPORT #%d invalid\n",
                __FUNCTION__, link);

            continue;
        }

        if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLW, link))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: LWLW #%d invalid\n",
                __FUNCTION__, link);

            continue;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    // Buffer to backup NPORT state
    nport_reg_val = lwswitch_os_malloc(sizeof(nport_reg_addr));
    if (nport_reg_val == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Failed to allocate memory\n",
            __FUNCTION__);

        return -LWL_NO_MEM;
    }

    FOR_EACH_INDEX_IN_MASK(64, link, link_mask)
    {
        // Unregister links to make them unusable while reset is in progress.
        link_info = lwswitch_get_link(device, link);
        if (link_info == NULL)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: invalid link %d\n",
                __FUNCTION__, link);
            continue;
        }

        lwlink_lib_unregister_link(link_info);

        //
        // Step 0 :
        // Prior to starting port reset, FM must shutdown the LWlink links
        // it wishes to reset.
        // However, with shared-virtualization, FM is unable to shut down the links
        // since the GPU is no longer attached to the service VM.
        // In this case, we must perform unilateral shutdown on the LR10 side
        // of the link.
        //
        // If links are in OFF or RESET, we don't need to perform shutdown
        // If links already went through a proper pseudo-clean shutdown sequence,
        // they'll be in SAFE + sublinks in OFF
        //

        status = lwswitch_corelib_get_dl_link_mode_lr10(link_info, &link_mode);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Unable to get link mode from link %d\n",
                __FUNCTION__, link);
            goto lwswitch_reset_and_drain_links_exit;
        }
        status = lwswitch_corelib_get_tx_mode_lr10(link_info, &tx_sublink_mode, &tx_sublink_submode);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Unable to get tx sublink mode from link %d\n",
                __FUNCTION__, link);
            goto lwswitch_reset_and_drain_links_exit;
        }
        status = lwswitch_corelib_get_rx_mode_lr10(link_info, &rx_sublink_mode, &rx_sublink_submode);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Unable to get rx sublink mode from link %d\n",
                __FUNCTION__, link);
            goto lwswitch_reset_and_drain_links_exit;
        }

        if (!((link_mode == LWLINK_LINKSTATE_RESET) ||
              (link_mode == LWLINK_LINKSTATE_OFF) ||
              ((link_mode == LWLINK_LINKSTATE_SAFE) &&
               (tx_sublink_mode == LWLINK_SUBLINK_STATE_TX_OFF) &&
               (rx_sublink_mode == LWLINK_SUBLINK_STATE_RX_OFF))))
        {
            lwswitch_exelwte_unilateral_link_shutdown_lr10(link_info);
            _lwlink_clear_corelib_state(link_info);
        }

        //
        // Step 1 : Perform surgical reset
        // Refer to switch IAS 11.5.2 Link Reset.
        //

        // Step 1.a : Backup NPORT state before reset
        for (i = 0; i < reg_count; i++)
        {
            nport_reg_val[i] = LWSWITCH_ENG_OFF_RD32(device, NPORT, _UNICAST, link,
                nport_reg_addr[i]);
        }

        // Step 1.b : Assert INGRESS_STOP / EGRESS_STOP
        val = LWSWITCH_NPORT_RD32_LR10(device, link, _NPORT, _CTRL_STOP);
        val = FLD_SET_DRF(_NPORT, _CTRL_STOP, _INGRESS_STOP, _STOP, val);
        val = FLD_SET_DRF(_NPORT, _CTRL_STOP, _EGRESS_STOP, _STOP, val);
        LWSWITCH_NPORT_WR32_LR10(device, link, _NPORT, _CTRL_STOP, val);

        // Wait for stop operation to take effect at TLC.
        // Expected a minimum of 256 clk cycles.
        lwswitch_os_sleep(1);

        //
        // Step 1.c : Disable NPG & LWLW interrupts
        //
        _lwswitch_link_disable_interrupts_lr10(device, link);

        // Step 1.d : Assert NPortWarmReset
        npg = link / LWSWITCH_LINKS_PER_NPG;
        val = LWSWITCH_NPG_RD32_LR10(device, npg, _NPG, _WARMRESET);

        idx_nport = link % LWSWITCH_LINKS_PER_NPG;
        LWSWITCH_NPG_WR32_LR10(device, npg, _NPG, _WARMRESET,
            DRF_NUM(_NPG, _WARMRESET, _NPORTWARMRESET, ~LWBIT(idx_nport)));

        // Step 1.e : Initiate Minion reset sequence.
        status = lwswitch_request_tl_link_state_lr10(link_info,
            LW_LWLIPT_LNK_CTRL_LINK_STATE_REQUEST_REQUEST_RESET, LW_TRUE);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: LwLink Reset has failed for link %d\n",
                __FUNCTION__, link);
            goto lwswitch_reset_and_drain_links_exit;
        }

        // Step 1.e : De-assert NPortWarmReset
        LWSWITCH_NPG_WR32_LR10(device, npg, _NPG, _WARMRESET, val);

        // Step 1.f : Assert and De-assert NPort debug_clear
        // to clear the error status
        LWSWITCH_NPG_WR32_LR10(device, npg, _NPG, _DEBUG_CLEAR,
            DRF_NUM(_NPG, _DEBUG_CLEAR, _CLEAR, LWBIT(idx_nport)));

        LWSWITCH_NPG_WR32_LR10(device, npg, _NPG, _DEBUG_CLEAR,
            DRF_DEF(_NPG, _DEBUG_CLEAR, _CLEAR, _DEASSERT));

        // Step 1.g : Clear CONTAIN_AND_DRAIN to clear contain state (Bug 3115824)
        LWSWITCH_NPORT_WR32_LR10(device, link, _NPORT, _CONTAIN_AND_DRAIN,
            DRF_DEF(_NPORT, _CONTAIN_AND_DRAIN, _CLEAR, _ENABLE));

        val = LWSWITCH_NPORT_RD32_LR10(device, link, _NPORT, _CONTAIN_AND_DRAIN);
        if (FLD_TEST_DRF(_NPORT, _CONTAIN_AND_DRAIN, _CLEAR, _ENABLE, val))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: NPORT Contain and Drain Clear has failed for link %d\n",
                __FUNCTION__, link);
            status = LWL_ERR_ILWALID_STATE;
            goto lwswitch_reset_and_drain_links_exit;
        }

        //
        // Step 2 : Assert NPORT Reset after Control & Drain routine.
        //  Clear Tagpool, CrumbStore and CAM RAMs
        //

        // Step 2.a Clear Tagpool RAM
        LWSWITCH_NPORT_WR32_LR10(device, link, _NPORT, _INITIALIZATION,
            DRF_DEF(_NPORT, _INITIALIZATION, _TAGPOOLINIT_0, _HWINIT));

        lwswitch_timeout_create(25 * LWSWITCH_INTERVAL_1MSEC_IN_NS, &timeout);

        do
        {
            keepPolling = (lwswitch_timeout_check(&timeout)) ? LW_FALSE : LW_TRUE;

            // Check if NPORT initialization is done
            val = LWSWITCH_NPORT_RD32_LR10(device, link, _NPORT, _INITIALIZATION);
            if (FLD_TEST_DRF(_NPORT, _INITIALIZATION, _TAGPOOLINIT_0, _HWINIT, val))
            {
                break;
            }

            lwswitch_os_sleep(1);
        }
        while (keepPolling);

        if (!FLD_TEST_DRF(_NPORT, _INITIALIZATION, _TAGPOOLINIT_0, _HWINIT, val))
        {
            LWSWITCH_PRINT(device, ERROR,
                "%s: Timeout waiting for TAGPOOL Initialization on link %d)\n",
                __FUNCTION__, link);

            status = -LWL_INITIALIZATION_TOTAL_FAILURE;
            goto lwswitch_reset_and_drain_links_exit;
        }

        // Step 2.b Clear CrumbStore RAM
        val = DRF_NUM(_TSTATE, _RAM_ADDRESS, _ADDR, 0) |
              DRF_DEF(_TSTATE, _RAM_ADDRESS, _SELECT, _CRUMBSTORE_RAM) |
              DRF_NUM(_TSTATE, _RAM_ADDRESS, _AUTO_INCR, 1);

        LWSWITCH_NPORT_WR32_LR10(device, link, _TSTATE, _RAM_ADDRESS, val);
        LWSWITCH_NPORT_WR32_LR10(device, link, _TSTATE, _RAM_DATA1, 0x0);

        val = DRF_NUM(_TSTATE, _RAM_DATA0, _ECC, 0x7f);
        for (i = 0; i <= LW_TSTATE_RAM_ADDRESS_ADDR_TAGPOOL_CRUMBSTORE_TDTID_DEPTH; i++)
        {
            LWSWITCH_NPORT_WR32_LR10(device, link, _TSTATE, _RAM_DATA0, val);
        }

        // Step 2.c Clear CAM RAM
        val = DRF_NUM(_TSTATE, _RAM_ADDRESS, _ADDR, 0) |
              DRF_DEF(_TSTATE, _RAM_ADDRESS, _SELECT, _CREQ_CAM) |
              DRF_NUM(_TSTATE, _RAM_ADDRESS, _AUTO_INCR, 1);

        LWSWITCH_NPORT_WR32_LR10(device, link, _TSTATE, _RAM_ADDRESS, val);
        LWSWITCH_NPORT_WR32_LR10(device, link, _TSTATE, _RAM_DATA1, 0x0);
        LWSWITCH_NPORT_WR32_LR10(device, link, _TSTATE, _RAM_DATA2, 0x0);

        for (i = 0; i <= LW_TSTATE_RAM_ADDRESS_ADDR_CREQ_CAM_DEPTH; i++)
        {
            LWSWITCH_NPORT_WR32_LR10(device, link, _TSTATE, _RAM_DATA0, 0x0);
        }

        //
        // Step 3 : Restore link state
        //

        // Restore NPORT state after reset
        for (i = 0; i < reg_count; i++)
        {
            LWSWITCH_ENG_OFF_WR32(device, NPORT, _UNICAST, link,
                                  nport_reg_addr[i], nport_reg_val[i]);
        }

        // Initialize GLT
        lwswitch_set_ganged_link_table_lr10(device, 0, chip_device->ganged_link_table,
                                            ROUTE_GANG_TABLE_SIZE/2);

        // Initialize select scratch registers to 0x0
        lwswitch_init_scratch_lr10(device);

        // Reset LWLW and NPORT interrupt state
        _lwswitch_link_reset_interrupts_lr10(device, link);

        // Re-register links.
        status = lwlink_lib_register_link(device->lwlink_device, link_info);
        if (status != LWL_SUCCESS)
        {
            lwswitch_destroy_link(link_info);
            goto lwswitch_reset_and_drain_links_exit;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    if (lwswitch_is_cci_supported(device))
    {
        status = cciResetLinks(device, managedLinkMask);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR, 
                "%s: Xcvr reset failed.\n",
                __FUNCTION__);
            goto lwswitch_reset_and_drain_links_exit;
        }
    }

#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    // Launch ALI training if applicable
    (void)lwswitch_launch_ALI(device);
#endif

lwswitch_reset_and_drain_links_exit:
    lwswitch_os_free(nport_reg_val);
    return status;
}

LwlStatus
lwswitch_get_lwlink_ecc_errors_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_ECC_ERRORS_PARAMS *params
)
{
    LwU32 statData;
    LwU8 i, j;
    LwlStatus status;
    LwBool bLaneReversed;

    lwswitch_os_memset(params->errorLink, 0, sizeof(params->errorLink));

    FOR_EACH_INDEX_IN_MASK(64, i, params->linkMask)
    {
        lwlink_link         *link;
        LWSWITCH_LANE_ERROR *errorLane;
        LwU8                offset;
        LwBool              minion_enabled;
        LwU32               sublinkWidth;

        link = lwswitch_get_link(device, i);
        sublinkWidth = device->hal.lwswitch_get_sublink_width(device, i);

        if ((link == NULL) ||
            !LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLDL, link->linkNumber) ||
            (i >= LWSWITCH_LINK_COUNT(device)))
        {
            return -LWL_BAD_ARGS;
        }

        minion_enabled = lwswitch_is_minion_initialized(device,
            LWSWITCH_GET_LINK_ENG_INST(device, link->linkNumber, MINION));

        bLaneReversed = lwswitch_link_lane_reversed_lr10(device, link->linkNumber);

        for (j = 0; j < LWSWITCH_LWLINK_MAX_LANES; j++)
        {
            if (minion_enabled && (j < sublinkWidth))
            {
                status = lwswitch_minion_get_dl_status(device, i,
                                        (LW_LWLSTAT_RX12 + j), 0, &statData);

                if (status != LWL_SUCCESS)
                {
                    return status;
                }
                offset = bLaneReversed ? ((sublinkWidth - 1) - j) : j;
                errorLane                = &params->errorLink[i].errorLane[offset];
                errorLane->valid         = LW_TRUE;
            }
            else
            {
                // MINION disabled
                statData                 = 0;
                offset                   = j;
                errorLane                = &params->errorLink[i].errorLane[offset];
                errorLane->valid         = LW_FALSE;
            }

            errorLane->eccErrorValue = DRF_VAL(_LWLSTAT, _RX12, _ECC_CORRECTED_ERR_L0_VALUE, statData);
            errorLane->overflowed    = DRF_VAL(_LWLSTAT, _RX12, _ECC_CORRECTED_ERR_L0_OVER, statData);
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return LWL_SUCCESS;
}

static LwU32
lwswitch_get_num_links_lr10
(
    lwswitch_device *device
)
{
    LwU32 num_links = LWSWITCH_NUM_LINKS_LR10;
    return num_links;
}

LwBool
lwswitch_is_link_valid_lr10
(
    lwswitch_device *device,
    LwU32            link_id
)
{
    if (link_id >= lwswitch_get_num_links(device))
    {
        return LW_FALSE;
    }
    return device->link[link_id].valid;
}

LwlStatus
lwswitch_ctrl_get_fom_values_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_FOM_VALUES_PARAMS *p
)
{
    LwlStatus status;
    LwU32     statData;

    LWSWITCH_ASSERT(p->linkId < lwswitch_get_num_links(device));

    status = lwswitch_minion_get_dl_status(device, p->linkId,
                                        LW_LWLSTAT_TR16, 0, &statData);
    p->figureOfMeritValues[0] = (LwU16) (statData & 0xFFFF);
    p->figureOfMeritValues[1] = (LwU16) ((statData >> 16) & 0xFFFF);

    status = lwswitch_minion_get_dl_status(device, p->linkId,
                                        LW_LWLSTAT_TR17, 0, &statData);
    p->figureOfMeritValues[2] = (LwU16) (statData & 0xFFFF);
    p->figureOfMeritValues[3] = (LwU16) ((statData >> 16) & 0xFFFF);

    p->numLanes = lwswitch_get_sublink_width(device, p->linkId);

    return status;
}

void
lwswitch_set_fatal_error_lr10
(
    lwswitch_device *device,
    LwBool           device_fatal,
    LwU32            link_id
)
{
    LwU32 reg;

    LWSWITCH_ASSERT(link_id < lwswitch_get_num_links(device));

    // On first fatal error, notify PORT_DOWN
    if (!device->link[link_id].fatal_error_oclwrred)
    {
        if (lwswitch_lib_notify_client_events(device,
                    LWSWITCH_DEVICE_EVENT_PORT_DOWN) != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR, "%s: Failed to notify PORT_DOWN event\n",
                         __FUNCTION__);
        }
    }

    device->link[link_id].fatal_error_oclwrred = LW_TRUE;

    if (device_fatal)
    {
        reg = LWSWITCH_SAW_RD32_LR10(device, _LWLSAW, _SW_SCRATCH_12);
        reg = FLD_SET_DRF_NUM(_LWLSAW, _SW_SCRATCH_12, _DEVICE_RESET_REQUIRED,
                              1, reg);

        LWSWITCH_SAW_WR32_LR10(device, _LWLSAW, _SW_SCRATCH_12, reg);
    }
    else
    {
        reg = LWSWITCH_LINK_RD32_LR10(device, link_id, NPORT, _NPORT, _SCRATCH_WARM);
        reg = FLD_SET_DRF_NUM(_NPORT, _SCRATCH_WARM, _PORT_RESET_REQUIRED,
                              1, reg);

        LWSWITCH_LINK_WR32_LR10(device, link_id, NPORT, _NPORT, _SCRATCH_WARM, reg);
    }
}

static LwU32
lwswitch_get_latency_sample_interval_msec_lr10
(
    lwswitch_device *device
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    return chip_device->latency_stats->sample_interval_msec;
}

LwU32
lwswitch_get_swap_clk_default_lr10
(
    lwswitch_device *device
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwBool
lwswitch_is_link_in_use_lr10
(
    lwswitch_device *device,
    LwU32 link_id
)
{
    LwU32 data;
    lwlink_link *link;

    link = lwswitch_get_link(device, link_id);
    if (link == NULL)
    {
        // A query on an invalid link should never occur
        LWSWITCH_ASSERT(link != NULL);
        return LW_FALSE;
    }

    if (lwswitch_is_link_in_reset(device, link))
    {
        return LW_FALSE;
    }

    data = LWSWITCH_LINK_RD32_LR10(device, link_id,
                                   LWLDL, _LWLDL_TOP, _LINK_STATE);

    return (DRF_VAL(_LWLDL_TOP, _LINK_STATE, _STATE, data) !=
            LW_LWLDL_TOP_LINK_STATE_STATE_INIT);
}

static LwU32
lwswitch_get_device_dma_width_lr10
(
    lwswitch_device *device
)
{
    return DMA_ADDR_WIDTH_LR10;
}

LwU32
lwswitch_get_link_ip_version_lr10
(
    lwswitch_device *device,
    LwU32            link_id
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LwU32 lwldl_instance;

    lwldl_instance = LWSWITCH_GET_LINK_ENG_INST(device, link_id, LWLDL);
    if (LWSWITCH_ENG_IS_VALID(device, LWLDL, lwldl_instance))
    {
        return chip_device->engLWLDL[lwldl_instance].version;
    }
    else
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: LWLink[0x%x] LWLDL instance invalid\n",
            __FUNCTION__, link_id);
        return 0;
    }
}

static LwlStatus
lwswitch_test_soe_dma_lr10
(
    lwswitch_device *device
)
{
    return soeTestDma_HAL(device, (PSOE)device->pSoe);
}

static LwlStatus
_lwswitch_get_reserved_throughput_counters
(
    lwswitch_device *device,
    lwlink_link     *link,
    LwU16           counter_mask,
    LwU64           *counter_values
)
{
    LwU16 counter = 0;

    //
    // LR10 to use counters 0 & 2 for monitoring
    // (Same as GPU behavior)
    // Counter 0 counts data flits
    // Counter 2 counts all flits
    //
    FOR_EACH_INDEX_IN_MASK(16, counter, counter_mask)
    {
        LwU32 counter_type = LWBIT(counter);
        LwU64 data = 0;

        switch (counter_type)
        {
            case LWSWITCH_THROUGHPUT_COUNTERS_TYPE_DATA_TX:
            {
                data = lwswitch_read_64bit_counter(device,
                           LWSWITCH_LINK_OFFSET_LR10(device, link->linkNumber,
                           LWLTLC, _LWLTLC_TX_LNK, _DEBUG_TP_CNTR_LO(0)),
                           LWSWITCH_LINK_OFFSET_LR10(device, link->linkNumber,
                           LWLTLC, _LWLTLC_TX_LNK, _DEBUG_TP_CNTR_HI(0)));
                break;
            }
            case LWSWITCH_THROUGHPUT_COUNTERS_TYPE_DATA_RX:
            {
                data = lwswitch_read_64bit_counter(device,
                           LWSWITCH_LINK_OFFSET_LR10(device, link->linkNumber,
                           LWLTLC, _LWLTLC_RX_LNK, _DEBUG_TP_CNTR_LO(0)),
                           LWSWITCH_LINK_OFFSET_LR10(device, link->linkNumber,
                           LWLTLC, _LWLTLC_RX_LNK, _DEBUG_TP_CNTR_HI(0)));
                break;
            }
            case LWSWITCH_THROUGHPUT_COUNTERS_TYPE_RAW_TX:
            {
                data = lwswitch_read_64bit_counter(device,
                           LWSWITCH_LINK_OFFSET_LR10(device, link->linkNumber,
                           LWLTLC, _LWLTLC_TX_LNK, _DEBUG_TP_CNTR_LO(2)),
                           LWSWITCH_LINK_OFFSET_LR10(device, link->linkNumber,
                           LWLTLC, _LWLTLC_TX_LNK, _DEBUG_TP_CNTR_HI(2)));
                break;
            }
            case LWSWITCH_THROUGHPUT_COUNTERS_TYPE_RAW_RX:
            {
                data = lwswitch_read_64bit_counter(device,
                           LWSWITCH_LINK_OFFSET_LR10(device, link->linkNumber,
                           LWLTLC, _LWLTLC_RX_LNK, _DEBUG_TP_CNTR_LO(2)),
                           LWSWITCH_LINK_OFFSET_LR10(device, link->linkNumber,
                           LWLTLC, _LWLTLC_RX_LNK, _DEBUG_TP_CNTR_HI(2)));
                break;
            }
            default:
            {
                return -LWL_ERR_NOT_SUPPORTED;
            }
        }
        counter_values[counter] = data;
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_ctrl_get_throughput_counters_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_THROUGHPUT_COUNTERS_PARAMS *p
)
{
    LwlStatus status;
    lwlink_link *link;
    LwU16 i = 0;

    lwswitch_os_memset(p->counters, 0, sizeof(p->counters));

    FOR_EACH_INDEX_IN_MASK(64, i, p->linkMask)
    {
        link = lwswitch_get_link(device, i);
        if ((link == NULL) || (link->linkNumber >= LWSWITCH_MAX_PORTS) ||
            (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLTLC, link->linkNumber)))
        {
            continue;
        }

        status = _lwswitch_get_reserved_throughput_counters(device, link, p->counterMask,
                        p->counters[link->linkNumber].values);
        if (status != LWL_SUCCESS)
        {
            LWSWITCH_PRINT(device, ERROR,
                "Failed to get reserved LWLINK throughput counters on link %d\n",
                link->linkNumber);
            return status;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return LWL_SUCCESS;
}

static LwBool
lwswitch_is_soe_supported_lr10
(
    lwswitch_device *device
)
{
    return LW_TRUE;
}

LwBool
lwswitch_is_inforom_supported_lr10
(
    lwswitch_device *device
)
{
    if (IS_RTLSIM(device) || IS_EMULATION(device) || IS_FMODEL(device))
    {
        LWSWITCH_PRINT(device, INFO,
            "INFOROM is not supported on non-silicon platform\n");
        return LW_FALSE;
    }

    if (!lwswitch_is_soe_supported(device))
    {
        LWSWITCH_PRINT(device, INFO,
            "INFOROM is not supported since SOE is not supported\n");
        return LW_FALSE;
    }

    return LW_TRUE;
}

LwBool
lwswitch_is_spi_supported_lr10
(
    lwswitch_device *device
)
{
    return lwswitch_is_soe_supported(device);
}

LwBool
lwswitch_is_smbpbi_supported_lr10
(
    lwswitch_device *device
)
{
    if (IS_RTLSIM(device) || IS_FMODEL(device))
    {
        LWSWITCH_PRINT(device, INFO,
            "SMBPBI is not supported on RTLSIM/FMODEL platforms\n");
        return LW_FALSE;
    }

    if (!lwswitch_is_soe_supported(device))
    {
        LWSWITCH_PRINT(device, INFO,
            "SMBPBI is not supported since SOE is not supported\n");
        return LW_FALSE;
    }

    return LW_TRUE;
}

/*
 * @Brief : Additional setup needed after device initialization
 *
 * @Description :
 *
 * @param[in] device        a reference to the device to initialize
 */
LwlStatus
lwswitch_post_init_device_setup_lr10
(
    lwswitch_device *device
)
{
    LwlStatus retval;

    if (device->regkeys.soe_dma_self_test ==
            LW_SWITCH_REGKEY_SOE_DMA_SELFTEST_DISABLE)
    {
        LWSWITCH_PRINT(device, INFO,
            "Skipping SOE DMA selftest as requested using regkey\n");
    }
    else if (IS_RTLSIM(device) || IS_FMODEL(device))
    {
        LWSWITCH_PRINT(device, SETUP,
            "Skipping DMA selftest on FMODEL/RTLSIM platforms\n");
    }
    else if (!lwswitch_is_soe_supported(device))
    {
        LWSWITCH_PRINT(device, SETUP,
            "Skipping DMA selftest since SOE is not supported\n");
    }
    else
    {
        retval = lwswitch_test_soe_dma_lr10(device);
        if (retval != LWL_SUCCESS)
        {
            return retval;
        }
    }

    if (lwswitch_is_inforom_supported(device))
    {
        lwswitch_inforom_post_init(device);
    }
    else
    {
        LWSWITCH_PRINT(device, SETUP, "Skipping INFOROM init\n");
    }

    return LWL_SUCCESS;
}

/*
 * @Brief : Additional setup needed after blacklisted device initialization
 *
 * @Description :
 *
 * @param[in] device        a reference to the device to initialize
 */
void
lwswitch_post_init_blacklist_device_setup_lr10
(
    lwswitch_device *device
)
{
    LwlStatus status;

    if (lwswitch_is_inforom_supported(device))
    {
        lwswitch_inforom_post_init(device);
    }

    //
    // Initialize the driver state monitoring callback.
    // This is still needed for SOE to report correct driver state.
    //
    status = lwswitch_smbpbi_post_init(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Smbpbi post init failed, rc:%d\n",
                       status);
        return;
    }

    //
    // This internally will only flush if OMS value has changed
    //
    status = device->hal.lwswitch_oms_inforom_flush(device);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR, "Flushing OMS failed, rc:%d\n",
                       status);
        return;
    }
}

void
lwswitch_load_uuid_lr10
(
    lwswitch_device *device
)
{
    LwU32 regData[4];

    //
    // Read 128-bit UUID from secure scratch registers which must be
    // populated by firmware.
    //
    regData[0] = LWSWITCH_SAW_RD32_LR10(device, _LWLSAW_SW, _SCRATCH_8);
    regData[1] = LWSWITCH_SAW_RD32_LR10(device, _LWLSAW_SW, _SCRATCH_9);
    regData[2] = LWSWITCH_SAW_RD32_LR10(device, _LWLSAW_SW, _SCRATCH_10);
    regData[3] = LWSWITCH_SAW_RD32_LR10(device, _LWLSAW_SW, _SCRATCH_11);

    lwswitch_os_memcpy(&device->uuid.uuid, (LwU8 *)regData, LW_UUID_LEN);
}

LwlStatus
lwswitch_read_oob_blacklist_state_lr10
(
    lwswitch_device *device
)
{
    LwU32 reg;
    LwBool is_oob_blacklist;
    LwlStatus status;

    if (device == NULL)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Called with invalid argument\n", __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    reg = LWSWITCH_SAW_RD32_LR10(device, _LWLSAW, _SCRATCH_COLD);

    // Check for uninitialized SCRATCH_COLD before declaring the device blacklisted
    if (reg == LW_LWLSAW_SCRATCH_COLD_DATA_INIT)
        is_oob_blacklist = LW_FALSE;
    else
        is_oob_blacklist = DRF_VAL(_LWLSAW, _SCRATCH_COLD, _OOB_BLACKLIST_DEVICE_REQUESTED, reg);

    status = lwswitch_inforom_oms_set_device_disable(device, is_oob_blacklist);
    if (status != LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
            "Failed to set device disable to %d, rc:%d\n",
            is_oob_blacklist, status);
    }

    if (is_oob_blacklist)
    {
        device->device_fabric_state = LWSWITCH_DEVICE_FABRIC_STATE_BLACKLISTED;
        device->device_blacklist_reason = LWSWITCH_DEVICE_BLACKLIST_REASON_MANUAL_OUT_OF_BAND;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_write_fabric_state_lr10
(
    lwswitch_device *device
)
{
    LwU32 reg;

    if (device == NULL)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Called with invalid argument\n", __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    // bump the sequence number for each write
    device->fabric_state_sequence_number++;

    reg = LWSWITCH_SAW_RD32_LR10(device, _LWLSAW, _SW_SCRATCH_12);

    reg = FLD_SET_DRF_NUM(_LWLSAW, _SW_SCRATCH_12, _DEVICE_BLACKLIST_REASON,
                          device->device_blacklist_reason, reg);
    reg = FLD_SET_DRF_NUM(_LWLSAW, _SW_SCRATCH_12, _DEVICE_FABRIC_STATE,
                          device->device_fabric_state, reg);
    reg = FLD_SET_DRF_NUM(_LWLSAW, _SW_SCRATCH_12, _DRIVER_FABRIC_STATE,
                          device->driver_fabric_state, reg);
    reg = FLD_SET_DRF_NUM(_LWLSAW, _SW_SCRATCH_12, _EVENT_MESSAGE_COUNT,
                          device->fabric_state_sequence_number, reg);

    LWSWITCH_SAW_WR32_LR10(device, _LWLSAW, _SW_SCRATCH_12, reg);

    return LWL_SUCCESS;
}

static LWSWITCH_ENGINE_DESCRIPTOR_TYPE *
_lwswitch_get_eng_descriptor_lr10
(
    lwswitch_device *device,
    LWSWITCH_ENGINE_ID eng_id
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LWSWITCH_ENGINE_DESCRIPTOR_TYPE  *engine = NULL;

    if (eng_id >= LWSWITCH_ENGINE_ID_SIZE)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Engine_ID 0x%x out of range 0..0x%x\n",
            __FUNCTION__,
            eng_id, LWSWITCH_ENGINE_ID_SIZE-1);
        return NULL;
    }

    engine = &(chip_device->io.common[eng_id]);
    LWSWITCH_ASSERT(eng_id == engine->eng_id);

    return engine;
}

LwU32
lwswitch_get_eng_base_lr10
(
    lwswitch_device *device,
    LWSWITCH_ENGINE_ID eng_id,
    LwU32 eng_bcast,
    LwU32 eng_instance
)
{
    LWSWITCH_ENGINE_DESCRIPTOR_TYPE  *engine;
    LwU32 base_addr = LWSWITCH_BASE_ADDR_ILWALID;

    engine = _lwswitch_get_eng_descriptor_lr10(device, eng_id);
    if (engine == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: ID 0x%x[%d] %s not found\n",
            __FUNCTION__,
            eng_id, eng_instance,
            (
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_UNICAST) ? "UC" :
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_BCAST) ? "BC" :
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_MULTICAST) ? "MC" :
                "??"
            ));
        return LWSWITCH_BASE_ADDR_ILWALID;
    }

    if ((eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_UNICAST) &&
        (eng_instance < engine->eng_count))
    {
        base_addr = engine->uc_addr[eng_instance];
    }
    else if (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_BCAST)
    {
        base_addr = engine->bc_addr;
    }
    else if ((eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_MULTICAST) &&
        (eng_instance < engine->mc_addr_count))
    {
        base_addr = engine->mc_addr[eng_instance];
    }
    else
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Unknown address space type 0x%x (not UC, BC, or MC)\n",
            __FUNCTION__,
            eng_bcast);
    }

    if (base_addr == LWSWITCH_BASE_ADDR_ILWALID)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: ID 0x%x[%d] %s invalid address\n",
            __FUNCTION__,
            eng_id, eng_instance,
            (
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_UNICAST) ? "UC" :
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_BCAST) ? "BC" :
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_MULTICAST) ? "MC" :
                "??"
            ));
    }

    return base_addr;
}

LwU32
lwswitch_get_eng_count_lr10
(
    lwswitch_device *device,
    LWSWITCH_ENGINE_ID eng_id,
    LwU32 eng_bcast
)
{
    LWSWITCH_ENGINE_DESCRIPTOR_TYPE  *engine;
    LwU32 eng_count = 0;

    engine = _lwswitch_get_eng_descriptor_lr10(device, eng_id);
    if (engine == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: ID 0x%x %s not found\n",
            __FUNCTION__,
            eng_id,
            (
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_UNICAST) ? "UC" :
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_BCAST) ? "BC" :
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_MULTICAST) ? "MC" :
                "??"
            ));
        return 0;
    }

    if (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_UNICAST)
    {
        eng_count = engine->eng_count;
    }
    else if (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_BCAST)
    {
        if (engine->bc_addr == LWSWITCH_BASE_ADDR_ILWALID)
        {
            eng_count = 0;
        }
        else
        {
            eng_count = 1;
        }
    }
    else if (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_MULTICAST)
    {
        eng_count = engine->mc_addr_count;
    }
    else
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Unknown address space type 0x%x (not UC, BC, or MC)\n",
            __FUNCTION__,
            eng_bcast);
    }

    return eng_count;
}

LwU32
lwswitch_eng_rd_lr10
(
    lwswitch_device *device,
    LWSWITCH_ENGINE_ID eng_id,
    LwU32 eng_bcast,
    LwU32 eng_instance,
    LwU32 offset
)
{
    LwU32 base_addr = LWSWITCH_BASE_ADDR_ILWALID;
    LwU32 data;

    base_addr = lwswitch_get_eng_base_lr10(device, eng_id, eng_bcast, eng_instance);
    if (base_addr == LWSWITCH_BASE_ADDR_ILWALID)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: ID 0x%x[%d] %s invalid address\n",
            __FUNCTION__,
            eng_id, eng_instance,
            (
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_UNICAST) ? "UC" :
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_BCAST) ? "BC" :
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_MULTICAST) ? "MC" :
                "??"
            ));
        LWSWITCH_ASSERT(base_addr != LWSWITCH_BASE_ADDR_ILWALID);
        return 0xBADFBADF;
    }

    data = lwswitch_reg_read_32(device, base_addr + offset);

#if defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)
    {
        LWSWITCH_ENGINE_DESCRIPTOR_TYPE  *engine = _lwswitch_get_eng_descriptor_lr10(device, eng_id);

        LWSWITCH_PRINT(device, MMIO,
            "%s: ENG_RD %s(0x%x)[%d] @0x%08x+0x%06x = 0x%08x\n",
            __FUNCTION__,
            engine->eng_name, engine->eng_id,
            eng_instance,
            base_addr, offset,
            data);
    }
#endif  //defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)

    return data;
}

void
lwswitch_eng_wr_lr10
(
    lwswitch_device *device,
    LWSWITCH_ENGINE_ID eng_id,
    LwU32 eng_bcast,
    LwU32 eng_instance,
    LwU32 offset,
    LwU32 data
)
{
    LwU32 base_addr = LWSWITCH_BASE_ADDR_ILWALID;

    base_addr = lwswitch_get_eng_base_lr10(device, eng_id, eng_bcast, eng_instance);
    if (base_addr == LWSWITCH_BASE_ADDR_ILWALID)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: ID 0x%x[%d] %s invalid address\n",
            __FUNCTION__,
            eng_id, eng_instance,
            (
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_UNICAST) ? "UC" :
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_BCAST) ? "BC" :
                (eng_bcast == LWSWITCH_GET_ENG_DESC_TYPE_MULTICAST) ? "MC" :
                "??"
            ));
        LWSWITCH_ASSERT(base_addr != LWSWITCH_BASE_ADDR_ILWALID);
        return;
    }

    lwswitch_reg_write_32(device, base_addr + offset,  data);

#if defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)
    {
        LWSWITCH_ENGINE_DESCRIPTOR_TYPE  *engine = _lwswitch_get_eng_descriptor_lr10(device, eng_id);

        LWSWITCH_PRINT(device, MMIO,
            "%s: ENG_WR %s(0x%x)[%d] @0x%08x+0x%06x = 0x%08x\n",
            __FUNCTION__,
            engine->eng_name, engine->eng_id,
            eng_instance,
            base_addr, offset,
            data);
    }
#endif  //defined(DEVELOP) || defined(DEBUG) || defined(LW_MODS)
}

LwU32
lwswitch_get_link_eng_inst_lr10
(
    lwswitch_device *device,
    LwU32 link_id,
    LWSWITCH_ENGINE_ID eng_id
)
{
    LwU32   eng_instance = LWSWITCH_ENGINE_INSTANCE_ILWALID;

    if (link_id >= LWSWITCH_LINK_COUNT(device))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: link ID 0x%x out-of-range [0x0..0x%x]\n",
            __FUNCTION__,
            link_id, LWSWITCH_LINK_COUNT(device)-1);
        return LWSWITCH_ENGINE_INSTANCE_ILWALID;
    }

    switch (eng_id)
    {
        case LWSWITCH_ENGINE_ID_NPG:
            eng_instance = link_id / LWSWITCH_LINKS_PER_NPG;
            break;
        case LWSWITCH_ENGINE_ID_LWLIPT:
            eng_instance = link_id / LWSWITCH_LINKS_PER_LWLIPT;
            break;
        case LWSWITCH_ENGINE_ID_LWLW:
        case LWSWITCH_ENGINE_ID_LWLW_PERFMON:
            eng_instance = link_id / LWSWITCH_LINKS_PER_LWLW;
            break;
        case LWSWITCH_ENGINE_ID_MINION:
            eng_instance = link_id / LWSWITCH_LINKS_PER_MINION;
            break;
        case LWSWITCH_ENGINE_ID_NPORT:
        case LWSWITCH_ENGINE_ID_LWLTLC:
        case LWSWITCH_ENGINE_ID_LWLDL:
        case LWSWITCH_ENGINE_ID_LWLIPT_LNK:
        case LWSWITCH_ENGINE_ID_NPORT_PERFMON:
        case LWSWITCH_ENGINE_ID_RX_PERFMON:
        case LWSWITCH_ENGINE_ID_TX_PERFMON:
            eng_instance = link_id;
            break;
        default:
            LWSWITCH_PRINT(device, ERROR,
                "%s: link ID 0x%x has no association with EngID 0x%x\n",
                __FUNCTION__,
                link_id, eng_id);
            eng_instance = LWSWITCH_ENGINE_INSTANCE_ILWALID;
            break;
    }

    return eng_instance;
}

LwU32
lwswitch_get_caps_lwlink_version_lr10
(
    lwswitch_device *device
)
{
    return LWSWITCH_LWLINK_CAPS_LWLINK_VERSION_3_0;
}

LWSWITCH_BIOS_LWLINK_CONFIG *
lwswitch_get_bios_lwlink_config_lr10
(
    lwswitch_device *device
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);

    return (chip_device != NULL) ? &chip_device->bios_config : NULL;
}

/*
 * CTRL_LWSWITCH_SET_RESIDENCY_BINS
 */
static LwlStatus
lwswitch_ctrl_set_residency_bins_lr10
(
    lwswitch_device *device,
    LWSWITCH_SET_RESIDENCY_BINS *p
)
{
    LWSWITCH_PRINT(device, ERROR,
        "SET_RESIDENCY_BINS should not be called on LR10\n");
    return -LWL_ERR_NOT_SUPPORTED;
}

/*
 * CTRL_LWSWITCH_GET_RESIDENCY_BINS
 */
static LwlStatus
lwswitch_ctrl_get_residency_bins_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_RESIDENCY_BINS *p
)
{
    LWSWITCH_PRINT(device, ERROR,
        "GET_RESIDENCY_BINS should not be called on LR10\n");
    return -LWL_ERR_NOT_SUPPORTED;
}

/*
 * CTRL_LWSWITCH_GET_RB_STALL_BUSY
 */
static LwlStatus
lwswitch_ctrl_get_rb_stall_busy_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_RB_STALL_BUSY *p
)
{
    LWSWITCH_PRINT(device, ERROR,
        "GET_RB_STALL_BUSY should not be called on LR10\n");
    return -LWL_ERR_NOT_SUPPORTED;
}

/*
 * CTRL_LWSWITCH_INBAND_SEND_DATA
 */
LwlStatus
lwswitch_ctrl_inband_send_data_lr10
(
    lwswitch_device *device,
    LWSWITCH_INBAND_SEND_DATA_PARAMS *p
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

/*
 * CTRL_LWSWITCH_INBAND_RECEIVE_DATA
 */
LwlStatus
lwswitch_ctrl_inband_read_data_lr10
(
    lwswitch_device *device,
    LWSWITCH_INBAND_READ_DATA_PARAMS *p
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

/*
 * CTRL_LWSWITCH_INBAND_FLUSH_DATA
 */
LwlStatus
lwswitch_ctrl_inband_flush_data_lr10
(
    lwswitch_device *device,
    LWSWITCH_INBAND_FLUSH_DATA_PARAMS *p
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

/*
 * CTRL_LWSWITCH_INBAND_PENDING_DATA_STATS
 */
LwlStatus
lwswitch_ctrl_inband_pending_data_stats_lr10
(
    lwswitch_device *device,
    LWSWITCH_INBAND_PENDING_DATA_STATS_PARAMS *p
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwU32
lwswitch_read_iddq_dvdd_lr10
(
    lwswitch_device *device
)
{
    return lwswitch_fuse_opt_read_lr10(device, LW_FUSE_OPT_SPEEDO2);
}

/*
* @brief: This function will try to save the last valid seeds from MINION into InfoROM
* @params[in] device        reference to current lwswitch device
* @params[in] linkId        link we want to save seed data for
*/
void lwswitch_save_lwlink_seed_data_from_minion_to_inforom_lr10
(
    lwswitch_device *device,
    LwU32 linkId
)
{
    LwU32 seedDataCopy[LWLINK_MAX_SEED_BUFFER_SIZE];
    lwlink_link *link = lwswitch_get_link(device, linkId);

    if (link == NULL)
    {
        return;
    }

    if (device->regkeys.minion_cache_seeds == LW_SWITCH_REGKEY_MINION_CACHE_SEEDS_DISABLE)
    {
        return;
    }

    if (lwlink_lib_copy_training_seeds(link, seedDataCopy) != LWL_SUCCESS)
    {
         LWSWITCH_PRINT(device, INFO, "%s : Failed to get seed data for (%s):(%d).\n",
            __FUNCTION__, device->name, linkId);
    }

    //
    // (seedDataCopy+1) is the pointer to beginning of the actual parameters
    // seedDataCopy[0] is size of buffer
    //
    lwswitch_inforom_lwlink_set_minion_data(device, linkId, seedDataCopy + 1, seedDataCopy[0]);
}

/*
* @brief: This function retrieves the LWLIPT public ID for a given global link idx
* @params[in]  device        reference to current lwswitch device
* @params[in]  linkId        link to retrieve LWLIPT public ID from
* @params[out] publicId      Public ID of LWLIPT owning linkId
*/
LwlStatus lwswitch_get_link_public_id_lr10
(
    lwswitch_device *device,
    LwU32 linkId,
    LwU32 *publicId
)
{
    if (!device->hal.lwswitch_is_link_valid(device, linkId) ||
        (publicId == NULL))
    {
        return -LWL_BAD_ARGS;
    }

    *publicId = LWSWITCH_LWLIPT_GET_PUBLIC_ID_LR10(linkId);


    return (LWSWITCH_ENG_VALID_LR10(device, LWLIPT, *publicId)) ?
                LWL_SUCCESS : -LWL_BAD_ARGS;
}

/*
* @brief: This function retrieves the internal link idx for a given global link idx
* @params[in]  device        reference to current lwswitch device
* @params[in]  linkId        link to retrieve LWLIPT public ID from
* @params[out] localLinkIdx  Internal link index of linkId
*/
LwlStatus lwswitch_get_link_local_idx_lr10
(
    lwswitch_device *device,
    LwU32 linkId,
    LwU32 *localLinkIdx
)
{
    if (!device->hal.lwswitch_is_link_valid(device, linkId) ||
        (localLinkIdx == NULL))
    {
        return -LWL_BAD_ARGS;
    }

    *localLinkIdx = LWSWITCH_LWLIPT_GET_LOCAL_LINK_ID_LR10(linkId);

    return LWL_SUCCESS;
}

LwlStatus lwswitch_set_training_error_info_lr10
(
    lwswitch_device *device,
    LWSWITCH_SET_TRAINING_ERROR_INFO_PARAMS *pLinkTrainingErrorInfoParams
)
{
    LWSWITCH_LINK_TRAINING_ERROR_INFO linkTrainingErrorInfo;
    LWSWITCH_LINK_RUNTIME_ERROR_INFO linkRuntimeErrorInfo;

    linkTrainingErrorInfo.isValid = LW_TRUE;
    linkTrainingErrorInfo.attemptedTrainingMask0 =
        pLinkTrainingErrorInfoParams->attemptedTrainingMask0;
    linkTrainingErrorInfo.trainingErrorMask0 =
        pLinkTrainingErrorInfoParams->trainingErrorMask0;

    linkRuntimeErrorInfo.isValid = LW_FALSE;
    linkRuntimeErrorInfo.mask0   = 0;

    return lwswitch_smbpbi_set_link_error_info(device,
                                               &linkTrainingErrorInfo,
                                               &linkRuntimeErrorInfo);
}

LwlStatus lwswitch_ctrl_get_fatal_error_scope_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_FATAL_ERROR_SCOPE_PARAMS *pParams
)
{
    LwU32 linkId;
    LwU32 reg = LWSWITCH_SAW_RD32_LR10(device, _LWLSAW, _SW_SCRATCH_12);
    pParams->device = FLD_TEST_DRF_NUM(_LWLSAW, _SW_SCRATCH_12, _DEVICE_RESET_REQUIRED,
                                       1, reg);

    for (linkId = 0; linkId < LWSWITCH_MAX_PORTS; linkId++)
    {
        if (!lwswitch_is_link_valid(device, linkId))
        {
            pParams->port[linkId] = LW_FALSE;
            continue;
        }

        reg = LWSWITCH_LINK_RD32_LR10(device, linkId, NPORT, _NPORT, _SCRATCH_WARM);
        pParams->port[linkId] = FLD_TEST_DRF_NUM(_NPORT, _SCRATCH_WARM,
                                                 _PORT_RESET_REQUIRED, 1, reg);
    }

    return LWL_SUCCESS;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
LwlStatus lwswitch_ctrl_set_mc_rid_table_lr10
(
    lwswitch_device *device,
    LWSWITCH_SET_MC_RID_TABLE_PARAMS *p
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus lwswitch_ctrl_get_mc_rid_table_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_MC_RID_TABLE_PARAMS *p
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

void lwswitch_init_scratch_lr10
(
    lwswitch_device *device
)
{
    LwU32 linkId;
    LwU32 reg;

    for (linkId = 0; linkId < lwswitch_get_num_links(device); linkId++)
    {
        if (!lwswitch_is_link_valid(device, linkId))
        {
            continue;
        }

        reg = LWSWITCH_LINK_RD32_LR10(device, linkId, NPORT, _NPORT, _SCRATCH_WARM);
        if (reg == LW_NPORT_SCRATCH_WARM_DATA_INIT)
        {
            LWSWITCH_LINK_WR32_LR10(device, linkId, NPORT, _NPORT, _SCRATCH_WARM, 0);
        }
    }
}

#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
LwlStatus
lwswitch_launch_ALI_lr10
(
    lwswitch_device *device
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}
#endif

LwlStatus
lwswitch_set_training_mode_lr10
(
    lwswitch_device *device
)
{
    return LWL_SUCCESS;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
static LwlStatus
_lwswitch_get_internal_use_table_v2
(
    lwswitch_device *device,
    BIT_DATA_INTERNAL_USE_V2 *pInternalUseTable
)
{
    LWSWITCH_BIOS_LWLINK_CONFIG *bios_config;
    BIT_HEADER_V1_00         bitHeader;
    BIT_TOKEN_V1_00          bitToken;
    LW_STATUS                rmStatus;
    LwU32                    dataPointerOffset;
    LwU32 i;

    bios_config = lwswitch_get_bios_lwlink_config(device);
    if ((bios_config == NULL) || (bios_config->bit_address == 0))
    {
        LWSWITCH_PRINT(device, WARN,
            "%s: VBIOS LwLink configuration table not found\n",
            __FUNCTION__);
        return -LWL_ERR_GENERIC;
    }

    rmStatus = _lwswitch_vbios_read_structure(device,
                                              (LwU8*) &bitHeader,
                                              bios_config->bit_address,
                                              (LwU32 *) 0,
                                              BIT_HEADER_V1_00_FMT);

    if(rmStatus != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR,
                       "%s: Failed to read BIT table structure!.\n",
                       __FUNCTION__);
        return -LWL_ERR_GENERIC;
    }

    // parse through bit tokens
    for(i=0; i < bitHeader.TokenEntries; i++)
    {
        LwU32 BitTokenLocation = bios_config->bit_address + bitHeader.HeaderSize + (i * bitHeader.TokenSize);
        rmStatus = _lwswitch_vbios_read_structure(device,
                                                 (LwU8*) &bitToken,
                                                 BitTokenLocation,
                                                 (LwU32 *) 0,
                                                 BIT_TOKEN_V1_00_FMT);
        if(rmStatus != LW_OK)
        {
            LWSWITCH_PRINT(device, WARN,
                "%s: Failed to read BIT token %d!\n",
                __FUNCTION__, i);
            return -LWL_ERR_GENERIC;
        }

        dataPointerOffset = (bios_config->pci_image_address + bitToken.DataPtr);

        if (bitToken.TokenId == BIT_TOKEN_INTERNAL_USE)
        {
            if(bitToken.DataVersion <= 2)
            {
                rmStatus = _lwswitch_vbios_read_structure(device,
                                                            (LwU8*) pInternalUseTable,
                                                            dataPointerOffset,
                                                            (LwU32 *) 0,
                                                            BIT_DATA_INTERNAL_USE_V2_FMT);
                if (rmStatus != LW_OK)
                {
                    LWSWITCH_PRINT(device, WARN,
                                    "%s: Failed to read internal data structure\n",
                                    __FUNCTION__);
                    return -LWL_ERR_GENERIC;
                }
                return LWL_SUCCESS;
            }
        }
    }

    return -LWL_ERR_GENERIC;
}
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#ifdef LW_MODS
LW_STATUS
_lwswitch_get_i2c_devices_from_dcb
(
    lwswitch_device *device
)
{
    LW_STATUS rmStatus;
    LwU32 i;
    LwU16 dcbHeaderPtr;
    LwU32 i2cEntryPtr;
    LwU32 i2cDeviceType;
    LWSWITCH_VBIOS_DCB_HEADER dcbHeader;
    LWSWITCH_VBIOS_CCB_TABLE ccbTable;
    LWSWITCH_VBIOS_I2C_TABLE i2cTable;
    LWSWITCH_VBIOS_I2C_ENTRY i2cEntry;
    LWSWITCH_I2C_DEVICE_DESCRIPTOR_TYPE * i2c_device;
    LWSWITCH_BIOS_LWLINK_CONFIG *bios_config;

    if ((device->biosImage.size == 0) || (device->biosImage.pImage == NULL))
    {
        LWSWITCH_PRINT(device, SETUP,
                "%s: VBIOS does not exist - Need to confirm on SPI interface support size:0x%x\n",
                __FUNCTION__, device->biosImage.size);

         if (device->pSoe)
         {
             return -LWL_ERR_NOT_SUPPORTED;
         }
         LWSWITCH_PRINT(device, SETUP,
                 "%s: Skipping DCB setup because SOE is not supported\n",
                 __FUNCTION__);
         return LWL_SUCCESS;
    }

    bios_config = lwswitch_get_bios_lwlink_config(device);

    if (_lwswitch_vbios_identify_pci_image_loc(device, bios_config)  != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Error on identifying pci image loc\n",
            __FUNCTION__);
        return LW_ERR_GENERIC;
    }

    dcbHeaderPtr = _lwswitch_vbios_read16(device,
                                          bios_config->pci_image_address + LWSWITCH_DCB_PTR_OFFSET);

    rmStatus = _lwswitch_vbios_read_structure(device,
                                              (LwU8*) &dcbHeader,
                                              bios_config->pci_image_address + dcbHeaderPtr,
                                              (LwU32 *) 0,
                                              LWSWITCH_VBIOS_DCB_HEADER_FMT);
    if (rmStatus != LW_OK)
    {
        LWSWITCH_PRINT(device, WARN,
                       "%s: Failed to read internal data structure\n",
                       __FUNCTION__);
        return LW_ERR_GENERIC;
    }

    if ((dcbHeader.version != LWSWITCH_DCB_HEADER_VERSION_41) || (dcbHeader.header_size != 35))
    {
        LWSWITCH_PRINT(device, WARN,
                       "%s: Unknown DCB header version or size\n",
                       __FUNCTION__);
        return LW_ERR_GENERIC;
    }

    rmStatus = _lwswitch_vbios_read_structure(device,
                                              (LwU8*) &ccbTable,
                                              bios_config->pci_image_address + dcbHeader.ccb_block_ptr,
                                              (LwU32 *) 0,
                                              LWSWITCH_VBIOS_CCB_TABLE_FMT);
    if (rmStatus != LW_OK)
    {
        LWSWITCH_PRINT(device, WARN,
                       "%s: Failed to read internal data structure\n",
                       __FUNCTION__);
        return LW_ERR_GENERIC;
    }

    if (ccbTable.version != LWSWITCH_CCB_VERSION)
    {
        LWSWITCH_PRINT(device, WARN,
                       "%s: Unknown CCB table version\n",
                       __FUNCTION__);
        return LW_ERR_GENERIC;
    }

    rmStatus = _lwswitch_vbios_read_structure(device,
                                              (LwU8*) &i2cTable,
                                              bios_config->pci_image_address + dcbHeader.i2c_devices,
                                              (LwU32 *) 0,
                                              LWSWITCH_I2C_TABLE_FMT);
    if (rmStatus != LW_OK)
    {
        LWSWITCH_PRINT(device, WARN,
                       "%s: Failed to read internal data structure\n",
                       __FUNCTION__);
        return LW_ERR_GENERIC;
    }

    if ((i2cTable.version != LWSWITCH_I2C_VERSION) || (i2cTable.header_size != 5))
    {
        LWSWITCH_PRINT(device, WARN,
                       "%s: Unknown I2C DCB table version or size\n",
                       __FUNCTION__);
        return LW_ERR_GENERIC;
    }

    i2cEntryPtr = bios_config->pci_image_address + dcbHeader.i2c_devices + i2cTable.header_size;
    device->firmware.dcb.i2c_device_count = 0;
    for (i = 0; i < i2cTable.entry_count; i++, i2cEntryPtr += i2cTable.entry_size)
    {

        rmStatus = _lwswitch_vbios_read_structure(device,
                                                  (LwU8*) &i2cEntry,
                                                  i2cEntryPtr,
                                                  (LwU32 *) 0,
                                                  LWSWITCH_I2C_ENTRY_FMT);
        if (rmStatus != LW_OK)
        {
            LWSWITCH_PRINT(device, WARN,
                           "%s: Failed to read internal data structure\n",
                           __FUNCTION__);
            return LW_ERR_GENERIC;
        }

        i2cDeviceType = DRF_VAL(SWITCH_I2C, _ENTRY, _TYPE, i2cEntry.device);

        if (i2cDeviceType != 0xFF)
        {
            LwU8 ccbIdx;

            if (device->firmware.dcb.i2c_device_count == LWSWITCH_MAX_I2C_DEVICES)
            {
                LWSWITCH_PRINT(device, ERROR, "%s: Too many I2C devices listed\n", __FUNCTION__);
                return LW_ERR_GENERIC;
            }

            i2c_device = &device->firmware.dcb.i2c_device[device->firmware.dcb.i2c_device_count];
            device->firmware.dcb.i2c_device_count++;

            i2c_device->i2cDeviceType = DRF_VAL(SWITCH_I2C, _ENTRY, _TYPE, i2cEntry.device);
            i2c_device->i2cAddress = DRF_VAL(SWITCH_I2C, _ENTRY, _ADDRESS, i2cEntry.device);

            ccbIdx = (DRF_VAL(SWITCH_I2C, _ENTRY, _PORT_2, i2cEntry.device) << 1) |
                     DRF_VAL(SWITCH_I2C, _ENTRY, _PORT_1, i2cEntry.device);
            i2c_device->i2cPortLogical = ccbTable.comm_port[ccbIdx];
        }
    }

    return rmStatus;
}
#endif

LwlStatus
lwswitch_parse_bios_image_lr10
(
    lwswitch_device *device
)
{
    LWSWITCH_BIOS_LWLINK_CONFIG *bios_config;
    LW_STATUS status = LW_OK;

    // check if spi is supported
    if (!lwswitch_is_spi_supported(device))
    {
        LWSWITCH_PRINT(device, ERROR,
                "%s: SPI is not supported\n",
                __FUNCTION__);
        return -LWL_ERR_NOT_SUPPORTED;
    }

    bios_config = lwswitch_get_bios_lwlink_config(device);

    // Parse and retrieve the VBIOS info
    status = _lwswitch_setup_link_vbios_overrides(device, bios_config);
    if ((status != LW_OK) && device->pSoe)
    {
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
        //To enable LS10 bringup (VBIOS is not ready and SOE is disabled), fail the device init only when SOE is enabled and vbios overrides has failed
#endif
        LWSWITCH_PRINT(device, ERROR,
                "%s: error=0x%x\n",
                __FUNCTION__, status);

        return -LWL_ERR_GENERIC;
    }

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    {
        BIT_DATA_INTERNAL_USE_V2 intUseTable;

        if (_lwswitch_get_internal_use_table_v2(device, &intUseTable) == LWL_SUCCESS)
        {
            device->int_board_id = intUseTable.BoardID;
        }
        else
        {
            device->int_board_id = LWSWITCH_BOARD_UNKNOWN;
        }    
    }
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#ifdef LW_MODS
    if (_lwswitch_get_i2c_devices_from_dcb(device)  != LW_OK)
    {
        LWSWITCH_PRINT(device, ERROR, "%s: Error on getting DCB\n", __FUNCTION__);
        return LW_ERR_GENERIC;
    }
#endif

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_ctrl_get_lwlink_lp_counters_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_LP_COUNTERS_PARAMS *params
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
/* TRY TO ADD NEW UNPUBLISHED CODE BELOW THIS LINE */

LwBool
lwswitch_is_cci_supported_lr10
(
    lwswitch_device *device
)
{
    //
    // Lwrrently, device needs to be initialized before
    // board ID can be obtained from bios
    //
    if (LWSWITCH_IS_DEVICE_INITIALIZED(device) &&
        !cciSupported(device))
    {
        LWSWITCH_PRINT(device, INFO,
                      "%s: CCI is not supported on current board.\n",
                       __FUNCTION__);
        return LW_FALSE;
    }

    return LW_TRUE;
}

LwlStatus
lwswitch_get_board_id_lr10
(
    lwswitch_device *device,
    LwU16 *pBoardId
)
{
    if (pBoardId == NULL)
    {
        return -LWL_BAD_ARGS;
    }

    *pBoardId = device->int_board_id;

    return LWL_SUCCESS;
}

/*
 * @brief: This function returns current link repeater mode state for a given global link idx
 * @params[in]  device          reference to current lwswitch device
 * @params[in]  linkId          link to retrieve repeater mode state from
 * @params[out] isRepeaterMode  pointer to Repeater Mode boolean
 */
LwlStatus
lwswitch_is_link_in_repeater_mode_lr10
(
    lwswitch_device *device,
    LwU32 link_id,
    LwBool *isRepeaterMode
)
{
    *isRepeaterMode = LW_FALSE;
    return LWL_SUCCESS;
}

void
lwswitch_fetch_active_repeater_mask_lr10
(
    lwswitch_device *device
)
{
    LWSWITCH_BIOS_LWLINK_CONFIG *bios_config = NULL;
    LwU64 enabledLinkMask;
    LwU8 i;
    LWLINK_CONFIG_DATA_LINKENTRY *vbios_link_entry = NULL;

    bios_config = lwswitch_get_bios_lwlink_config(device);
    if ((bios_config == NULL) || (bios_config->bit_address == 0))
    {
        LWSWITCH_PRINT(device, WARN,
            "%s: VBIOS LwLink configuration table not found\n",
            __FUNCTION__);
        return;
    }

    enabledLinkMask = lwswitch_get_enabled_link_mask(device);

    FOR_EACH_INDEX_IN_MASK(64, i, enabledLinkMask)
    {
        LWSWITCH_ASSERT(i < LWSWITCH_LINK_COUNT(device));

        if (!LWSWITCH_IS_LINK_ENG_VALID(device, i, LWLDL) ||
            (i >= LWSWITCH_LWLINK_MAX_LINKS))
        {
            continue;
        }

        //
        // TODO: Only base entry 0 has active repeater bits set
        //       Should use bios_config->link_base_entry_assigned once
        //        bios is updated
        // 
        vbios_link_entry = &bios_config->link_vbios_entry[0][i];
        if ((vbios_link_entry != NULL) &&
             FLD_TEST_DRF(_LWLINK_VBIOS,_PARAM0, _ACTIVE_REPEATER, _PRESENT,
                          vbios_link_entry->lwLinkparam0))
        {
            device->link[i].bActiveRepeaterPresent = LW_TRUE;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;
}

LwU64
lwswitch_get_active_repeater_mask_lr10
(
    lwswitch_device *device
)
{
    LwU64 enabledLinkMask, mask = 0x0;
    LwU8 i;

    enabledLinkMask = lwswitch_get_enabled_link_mask(device);

    FOR_EACH_INDEX_IN_MASK(64, i, enabledLinkMask)
    {
        LWSWITCH_ASSERT(i < LWSWITCH_LINK_COUNT(device));

        if (device->link[i].bActiveRepeaterPresent)
        {
            mask |= LWBIT64(i);
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return mask;
}

LwlStatus
lwswitch_ctrl_set_port_test_mode_lr10
(
    lwswitch_device *device,
    LWSWITCH_SET_PORT_TEST_MODE *p
)
{
    lwlink_link *link;

    if (!LWSWITCH_IS_LINK_ENG_VALID_LR10(device, MINION, p->portNum))
    {
        return -LWL_BAD_ARGS;
    }

    link = lwswitch_get_link(device, (LwU8)p->portNum);
    if (link == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: invalid link\n",
            __FUNCTION__);
        return -LWL_ERR_ILWALID_STATE;
    }

    if ((p->nea) && (p->ned))
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: NEA and NED can not both be enabled simultaneously.\n",
            __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    // Near End Analog
    device->link[p->portNum].nea = p->nea;

    // Near End Digital
    device->link[p->portNum].ned = p->ned;

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_ctrl_jtag_chain_read_lr10
(
    lwswitch_device *device,
    LWSWITCH_JTAG_CHAIN_PARAMS *jtag_chain
)
{
    LwlStatus retval = LWL_SUCCESS;

    // Compute the amount of the buffer that we will actually use.
    LwU32 dataArrayLen = jtag_chain->chainLen / 32 + !!(jtag_chain->chainLen % 32);

    // The buffer must be at least large enough to hold the chain.
    if (jtag_chain->dataArrayLen < dataArrayLen)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Buffer too small for chain.  buffer=%u  required=%u  chain=%u.\n",
            __FUNCTION__,
            jtag_chain->dataArrayLen, dataArrayLen, jtag_chain->chainLen);
        return -LWL_BAD_ARGS;
    }

    // We need a buffer from which to read.
    if (jtag_chain->data == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: data: Required parameter is NULL.\n",
            __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    retval = lwswitch_jtag_read_seq_lr10(device,
                 jtag_chain->chainLen,
                 jtag_chain->chipletSel,
                 jtag_chain->instrId,
                 jtag_chain->data,
                 dataArrayLen);

    return retval;
}

LwlStatus
lwswitch_ctrl_jtag_chain_write_lr10
(
    lwswitch_device *device,
    LWSWITCH_JTAG_CHAIN_PARAMS *jtag_chain
)
{
    LwlStatus retval = LWL_SUCCESS;

    // Compute the amount of the bufer that we will actuall use.
    LwU32 dataArrayLen = jtag_chain->chainLen / 32 + !!(jtag_chain->chainLen % 32);

    // The buffer must be at least large enough to hold the chain.
    if (jtag_chain->dataArrayLen < dataArrayLen)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: Buffer too small for chain.  buffer=%u  required=%u  chain=%u.\n",
            __FUNCTION__,
            jtag_chain->dataArrayLen, dataArrayLen, jtag_chain->chainLen);
        return -LWL_BAD_ARGS;
    }

    // We need a buffer from which to read.
    if (jtag_chain->data == NULL)
    {
        LWSWITCH_PRINT(device, ERROR,
            "%s: data: Required parameter is NULL.\n",
            __FUNCTION__);
        return -LWL_BAD_ARGS;
    }

    retval = lwswitch_jtag_write_seq_lr10(device,
                  jtag_chain->chainLen,
                  jtag_chain->chipletSel,
                  jtag_chain->instrId,
                  jtag_chain->data,
                  dataArrayLen);

    return retval;
}

/*
 * @brief Inject an LWLink error on a link or links
 *
 * Errors are injected asynchronously.  This is a MODS test-only API and is an
 * exception to the locking rules regarding LWLink corelib callbacks (lr10_link.c).
 *
 * @param[in] device            LwSwitch device to contain this link
 * @param[in] p                 LWSWITCH_INJECT_LINK_ERROR
 *
 * @returns                     LWL_SUCCESS if action succeeded,
 *                              -LWL_ERR_ILWALID_STATE invalid link
 */
LwlStatus
lwswitch_ctrl_inject_link_error_lr10
(
    lwswitch_device *device,
    LWSWITCH_INJECT_LINK_ERROR *p
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_ctrl_get_lwlink_caps_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_LWLINK_CAPS_PARAMS *ret
)
{
    LwlStatus retval = LWL_SUCCESS;

    _lwswitch_set_lwlink_caps_lr10(&ret->capsTbl);

    ret->lowestLwlinkVersion = lwswitch_get_caps_lwlink_version(device);
    ret->highestLwlinkVersion = ret->lowestLwlinkVersion;

    if (ret->lowestLwlinkVersion == LWSWITCH_LWLINK_CAPS_LWLINK_VERSION_3_0)
    {
        ret->lowestNciVersion       = LWSWITCH_LWLINK_CAPS_NCI_VERSION_3_0;
        ret->highestNciVersion      = LWSWITCH_LWLINK_CAPS_NCI_VERSION_3_0;
    }
    else if (ret->lowestLwlinkVersion == LWSWITCH_LWLINK_CAPS_LWLINK_VERSION_4_0)
    {
        ret->lowestNciVersion       = LWSWITCH_LWLINK_CAPS_NCI_VERSION_4_0;
        ret->highestNciVersion      = LWSWITCH_LWLINK_CAPS_NCI_VERSION_4_0;
    }
    else
    {
        LWSWITCH_PRINT(device, WARN,
            "%s WARNING: Unknown LWSWITCH_LWLINK_CAPS_LWLINK_VERSION 0x%x\n",
            __FUNCTION__, ret->lowestLwlinkVersion);
        ret->lowestNciVersion       = LWSWITCH_LWLINK_CAPS_NCI_VERSION_ILWALID;
        ret->highestNciVersion      = LWSWITCH_LWLINK_CAPS_NCI_VERSION_ILWALID;
    }

    ret->enabledLinkMask    = lwswitch_get_enabled_link_mask(device);
    ret->activeRepeaterMask = lwswitch_get_active_repeater_mask(device);

    return retval;
}

static LwlStatus
lwswitch_ctrl_clear_counters_lr10
(
    lwswitch_device *device,
    LWSWITCH_LWLINK_CLEAR_COUNTERS_PARAMS *ret
)
{
    lwlink_link *link;
    LwU8 i;
    LwU32 counterMask;
    LwlStatus status = LWL_SUCCESS;

    counterMask = ret->counterMask;

    // Common usage allows one of these to stand for all of them
    if ((counterMask) & ( LWSWITCH_LWLINK_COUNTER_TL_TX0
                        | LWSWITCH_LWLINK_COUNTER_TL_TX1
                        | LWSWITCH_LWLINK_COUNTER_TL_RX0
                        | LWSWITCH_LWLINK_COUNTER_TL_RX1
                        ))
    {
        counterMask |= ( LWSWITCH_LWLINK_COUNTER_TL_TX0
                       | LWSWITCH_LWLINK_COUNTER_TL_TX1
                       | LWSWITCH_LWLINK_COUNTER_TL_RX0
                       | LWSWITCH_LWLINK_COUNTER_TL_RX1
                       );
    }

    // Common usage allows one of these to stand for all of them
    if ((counterMask) & ( LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_FLIT
                        | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L0
                        | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L1
                        | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L2
                        | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L3
                        | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L4
                        | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L5
                        | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L6
                        | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L7
                        | LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_REPLAY
                        | LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_RECOVERY
                        ))
    {
        counterMask |= ( LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_FLIT
                       | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L0
                       | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L1
                       | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L2
                       | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L3
                       | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L4
                       | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L5
                       | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L6
                       | LWSWITCH_LWLINK_COUNTER_DL_RX_ERR_CRC_LANE_L7
                       | LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_REPLAY
                       | LWSWITCH_LWLINK_COUNTER_DL_TX_ERR_RECOVERY
                       );
    }

    FOR_EACH_INDEX_IN_MASK(64, i, ret->linkMask)
    {
        link = lwswitch_get_link(device, i);
        if (link == NULL)
        {
            continue;
        }

        if (LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLTLC, link->linkNumber))
        {
            lwswitch_ctrl_clear_throughput_counters_lr10(device, link, counterMask);
        }
        if (LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLDL, link->linkNumber))
        {
            status = lwswitch_ctrl_clear_dl_error_counters_lr10(device, link, counterMask);
            // Return early with failure on clearing through minion
            if (status != LWL_SUCCESS)
            {
                LWSWITCH_PRINT(device, ERROR,
                    "%s: Failure on clearing link counter mask 0x%x on link %d\n",
                    __FUNCTION__, counterMask, link->linkNumber);
                break;
            }
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return status;
}

LwlStatus
lwswitch_ctrl_get_err_info_lr10
(
    lwswitch_device *device,
    LWSWITCH_LWLINK_GET_ERR_INFO_PARAMS *ret
)
{
    lwlink_link *link;
    LwU32 data;
    LwU8 i;

     ret->linkMask = lwswitch_get_enabled_link_mask(device);

    FOR_EACH_INDEX_IN_MASK(64, i, ret->linkMask)
    {
        link = lwswitch_get_link(device, i);

        if ((link == NULL) ||
            !LWSWITCH_IS_LINK_ENG_VALID_LR10(device, LWLDL, link->linkNumber) ||
            (i >= LWSWITCH_LWLINK_MAX_LINKS))
        {
            continue;
        }

        // TODO LWpu TL not supported
        LWSWITCH_PRINT(device, WARN,
            "%s WARNING: Lwpu %s register %s does not exist!\n",
            __FUNCTION__, "LWLTL", "LW_LWLTL_TL_ERRLOG_REG");

        LWSWITCH_PRINT(device, WARN,
            "%s WARNING: Lwpu %s register %s does not exist!\n",
            __FUNCTION__, "LWLTL", "LW_LWLTL_TL_INTEN_REG");

        ret->linkErrInfo[i].TLErrlog = 0x0;
        ret->linkErrInfo[i].TLIntrEn = 0x0;

        data = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TX, _SLSM_STATUS_TX);
        ret->linkErrInfo[i].DLSpeedStatusTx =
            DRF_VAL(_LWLDL_TX, _SLSM_STATUS_TX, _PRIMARY_STATE, data);

        data = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_RX, _SLSM_STATUS_RX);
        ret->linkErrInfo[i].DLSpeedStatusRx =
            DRF_VAL(_LWLDL_RX, _SLSM_STATUS_RX, _PRIMARY_STATE, data);

        data = LWSWITCH_LINK_RD32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TOP, _INTR);
        ret->linkErrInfo[i].bExcessErrorDL =
            !!DRF_VAL(_LWLDL_TOP, _INTR, _RX_SHORT_ERROR_RATE, data);

        if (ret->linkErrInfo[i].bExcessErrorDL)
        {
            LWSWITCH_LINK_WR32_LR10(device, link->linkNumber, LWLDL, _LWLDL_TOP, _INTR,
                DRF_NUM(_LWLDL_TOP, _INTR, _RX_SHORT_ERROR_RATE, 0x1));
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return LWL_SUCCESS;
}

static LwlStatus
lwswitch_ctrl_get_irq_info_lr10
(
    lwswitch_device *device,
    LWSWITCH_GET_IRQ_INFO_PARAMS *p
)
{
    // Set the mask to AND out during servicing in order to avoid int storm.
    p->maskInfoList[0].irqPendingOffset = LW_PSMC_INTR_LEGACY;
    p->maskInfoList[0].irqEnabledOffset = LW_PSMC_INTR_EN_LEGACY;
    p->maskInfoList[0].irqEnableOffset  = LW_PSMC_INTR_EN_SET_LEGACY;
    p->maskInfoList[0].irqDisableOffset = LW_PSMC_INTR_EN_CLR_LEGACY;
    p->maskInfoCount                    = 1;

    return LWL_SUCCESS;
}
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

//
// This function auto creates the lr10 HAL connectivity from the LWSWITCH_INIT_HAL
// macro in haldef_lwswitch.h
//
// Note: All hal fns must be implemented for each chip.
//       There is no automatic stubbing here.
//
void lwswitch_setup_hal_lr10(lwswitch_device *device)
{
    device->chip_arch = LWSWITCH_GET_INFO_INDEX_ARCH_LR10;

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    if (IS_FMODEL(device))
    {
        device->chip_impl = LWSWITCH_GET_INFO_INDEX_IMPL_S000;
    }
    else
#endif
    {
        device->chip_impl = LWSWITCH_GET_INFO_INDEX_IMPL_LR10;
    }

    LWSWITCH_INIT_HAL(device, lr10);
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
    LWSWITCH_INIT_HAL_UNPUBLISHED(device, lr10);                             
#endif //(!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#if LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
    LWSWITCH_INIT_HAL_LWCFG_LS10(device, lr10);                             
#endif //LWCFG(GLOBAL_LWSWITCH_IMPL_LS10)
}
