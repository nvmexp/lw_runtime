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
#include "intr_lwswitch.h"
#include "lr10/lr10.h"
#include "lr10/minion_lr10.h"
#include "regkey_lwswitch.h"
#include "soe/soe_lwswitch.h"
#include "inforom/inforom_lwswitch.h"

#include "lwswitch/lr10/dev_lws.h"
#include "lwswitch/lr10/dev_lws_master.h"
#include "lwswitch/lr10/dev_timer.h"
#include "lwswitch/lr10/dev_lwlsaw_ip.h"
#include "lwswitch/lr10/dev_pri_ringmaster.h"
#include "lwswitch/lr10/dev_pri_ringstation_sys.h"
#include "lwswitch/lr10/dev_pri_ringstation_prt.h"
#include "lwswitch/lr10/dev_lw_xve.h"
#include "lwswitch/lr10/dev_npg_ip.h"
#include "lwswitch/lr10/dev_nport_ip.h"
#include "lwswitch/lr10/dev_route_ip.h"
#include "lwswitch/lr10/dev_ingress_ip.h"
#include "lwswitch/lr10/dev_sourcetrack_ip.h"
#include "lwswitch/lr10/dev_egress_ip.h"
#include "lwswitch/lr10/dev_tstate_ip.h"
#include "lwswitch/lr10/dev_nxbar_tc_global_ip.h"
#include "lwswitch/lr10/dev_nxbar_tile_ip.h"
#include "lwswitch/lr10/dev_lwlipt_ip.h"
#include "lwswitch/lr10/dev_lwltlc_ip.h"
#include "lwswitch/lr10/dev_lwlipt_lnk_ip.h"
#include "lwswitch/lr10/dev_minion_ip.h"
#include "lwswitch/lr10/dev_lwldl_ip.h"
#include "lwswitch/lr10/dev_lwltlc_ip.h"
#include "lwswitch/lr10/dev_lwlctrl_ip.h"

static void
_lwswitch_construct_ecc_error_event
(
    INFOROM_LWS_ECC_ERROR_EVENT *err_event,
    LwU32  sxid,
    LwU32  linkId,
    LwBool bAddressValid,
    LwU32  address,
    LwBool bUncErr,
    LwU32  errorCount
)
{
    err_event->sxid          = sxid;
    err_event->linkId        = linkId;
    err_event->bAddressValid = bAddressValid;
    err_event->address       = address;
    err_event->bUncErr       = bUncErr;
    err_event->errorCount    = errorCount;
}

/*
 * @Brief : Enable top level HW interrupts.
 *
 * @Description :
 *
 * @param[in] device        operate on this device
 */
void
lwswitch_lib_enable_interrupts_lr10
(
    lwswitch_device *device
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LwU32 saw_legacy_intr_enable = 0;

    if (FLD_TEST_DRF_NUM(_PSMC, _INTR_EN_SET_LEGACY, _PTIMER, 1, chip_device->intr_enable_legacy))
    {
        saw_legacy_intr_enable = FLD_SET_DRF_NUM(_LWLSAW_LWSPMC, _INTR_EN_SET_LEGACY, _PTIMER_0, 1, saw_legacy_intr_enable);
        saw_legacy_intr_enable = FLD_SET_DRF_NUM(_LWLSAW_LWSPMC, _INTR_EN_SET_LEGACY, _PTIMER_1, 1, saw_legacy_intr_enable);
    }
    if (FLD_TEST_DRF_NUM(_PSMC, _INTR_EN_SET_LEGACY, _PMGR, 1, chip_device->intr_enable_legacy))
    {
        saw_legacy_intr_enable = FLD_SET_DRF_NUM(_LWLSAW_LWSPMC, _INTR_EN_SET_LEGACY, _PMGR_0, 1, saw_legacy_intr_enable);
        saw_legacy_intr_enable = FLD_SET_DRF_NUM(_LWLSAW_LWSPMC, _INTR_EN_SET_LEGACY, _PMGR_1, 1, saw_legacy_intr_enable);
    }

    LWSWITCH_REG_WR32(device, _PSMC, _INTR_EN_SET_LEGACY, chip_device->intr_enable_legacy);
    LWSWITCH_SAW_WR32_LR10(device, _LWLSAW_LWSPMC, _INTR_EN_SET_LEGACY, saw_legacy_intr_enable);
}

/*
 * @Brief : Disable top level HW interrupts.
 *
 * @Description :
 *
 * @param[in] device        operate on this device
 */
void
lwswitch_lib_disable_interrupts_lr10
(
    lwswitch_device *device
)
{
    if (LWSWITCH_GET_CHIP_DEVICE_LR10(device) == NULL)
    {
        LWSWITCH_PRINT(device, WARN,
            "%s: Can not disable interrupts.  Chip device==NULL\n",
            __FUNCTION__);
        return;
    }

    LWSWITCH_REG_WR32(device, _PSMC, _INTR_EN_CLR_LEGACY, 0xffffffff);

    //
    // Need a bit more time to ensure interrupt de-asserts, on
    // RTL simulation. Part of BUG 1869204 and 1881361.
    //
    if (IS_RTLSIM(device))
    {
        (void)LWSWITCH_SAW_RD32_LR10(device, _LWLSAW_LWSPMC, _INTR_EN_CLR_CORRECTABLE);
    }
}

static void
_lwswitch_build_top_interrupt_mask_lr10
(
    lwswitch_device *device
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LwU32 intr_bit;
    LwU32 i;

    chip_device->intr_enable_legacy =
//        DRF_NUM(_PSMC, _INTR_EN_SET_LEGACY, _PTIMER, 1) |
//        DRF_NUM(_PSMC, _INTR_EN_SET_LEGACY, _PMGR, 1) |
        DRF_NUM(_PSMC, _INTR_EN_SET_LEGACY, _SAW, 1) |
//        DRF_NUM(_PSMC, _INTR_EN_SET_LEGACY, _DECODE_TRAP_PRIV_LEVEL_VIOLATION, 1) |
//        DRF_NUM(_PSMC, _INTR_EN_SET_LEGACY, _DECODE_TRAP_WRITE_DROPPED, 1) |
//        DRF_NUM(_PSMC, _INTR_EN_SET_LEGACY, _RING_MANAGE_SUCCESS, 1) |
        DRF_NUM(_PSMC, _INTR_EN_SET_LEGACY, _PBUS, 1) |
//        DRF_NUM(_PSMC, _INTR_EN_SET_LEGACY, _XVE, 1) |
        DRF_NUM(_PSMC, _INTR_EN_SET_LEGACY, _PRIV_RING, 1) |
        0;

    chip_device->intr_enable_fatal = 0;
    chip_device->intr_enable_nonfatal = 0;
    chip_device->intr_enable_corr = 0;

    for (i = 0; i < NUM_NXBAR_ENGINE_LR10; i++)
    {
        if (LWSWITCH_ENG_VALID_LR10(device, NXBAR, i))
        {
            intr_bit = DRF_NUM(_LWLSAW_LWSPMC, _INTR_EN_SET_FATAL, _NXBAR_0, 1) << i;

            // NXBAR only has fatal interrupts
            chip_device->intr_enable_fatal |= intr_bit;
        }
    }

    for (i = 0; i < NUM_NPG_ENGINE_LR10; i++)
    {
        if (LWSWITCH_ENG_VALID_LR10(device, NPG, i))
        {
            intr_bit = DRF_NUM(_LWLSAW_LWSPMC, _INTR_EN_SET_FATAL, _NPG_0, 1) << i;
            chip_device->intr_enable_fatal |= intr_bit;

            intr_bit = DRF_NUM(_LWLSAW_LWSPMC, _INTR_EN_SET_NONFATAL, _NPG_0, 1) << i;
            chip_device->intr_enable_nonfatal |= intr_bit;
        }
    }

    for (i = 0; i < NUM_LWLW_ENGINE_LR10; i++)
    {
        if (LWSWITCH_ENG_VALID_LR10(device, LWLW, i))
        {
            intr_bit = DRF_NUM(_LWLSAW_LWSPMC, _INTR_EN_SET_FATAL, _LWLIPT_0, 1) << i;
            chip_device->intr_enable_fatal |= intr_bit;

            intr_bit = DRF_NUM(_LWLSAW_LWSPMC, _INTR_EN_SET_NONFATAL, _LWLIPT_0, 1) << i;
            chip_device->intr_enable_nonfatal |= intr_bit;
        }
    }

#if defined(LW_LWLSAW_LWSPMC_INTR_EN_SET_FATAL_SOE)
    if (lwswitch_is_soe_supported(device))
    {
        intr_bit = DRF_NUM(_LWLSAW_LWSPMC, _INTR_EN_SET_FATAL, _SOE, 1);
        chip_device->intr_enable_fatal |= intr_bit;
    }
#endif
}

static void
_lwswitch_initialize_minion_interrupts
(
    lwswitch_device *device,
    LwU32 instance
)
{
    LwU32 intrEn, localDiscoveredLinks, globalLink, i;
    localDiscoveredLinks = 0;

    // Tree 1 (non-stall) is disabled until there is a need
    LWSWITCH_MINION_WR32_LR10(device, instance, _MINION, _MINION_INTR_NONSTALL_EN, 0);

     // Tree 0 (stall) is where we route _all_ MINION interrupts for now
    intrEn = DRF_DEF(_MINION, _MINION_INTR_STALL_EN, _FATAL,          _ENABLE) |
             DRF_DEF(_MINION, _MINION_INTR_STALL_EN, _NONFATAL,       _ENABLE) |
             DRF_DEF(_MINION, _MINION_INTR_STALL_EN, _FALCON_STALL,   _ENABLE) |
             DRF_DEF(_MINION, _MINION_INTR_STALL_EN, _FALCON_NOSTALL, _DISABLE);

    for (i = 0; i < LWSWITCH_LINKS_PER_MINION; ++i)
    {
        // get the global link number of the link we are iterating over
        globalLink = (instance * LWSWITCH_LINKS_PER_MINION) + i;

        // the link is valid place bit in link mask
        if (device->link[globalLink].valid)
        {
            localDiscoveredLinks |= LWBIT(i);
        }
    }

    intrEn = FLD_SET_DRF_NUM(_MINION, _MINION_INTR_STALL_EN, _LINK,
                            localDiscoveredLinks, intrEn);

    LWSWITCH_MINION_WR32_LR10(device, instance, _MINION, _MINION_INTR_STALL_EN, intrEn);
}

static void
_lwswitch_initialize_lwlipt_interrupts_lr10
(
    lwswitch_device *device
)
{
    LwU32 i;
    LwU32 regval = 0;

    //
    // LWLipt interrupt routing (LWLIPT_COMMON, LWLIPT_LNK, LWLDL, LWLTLC)
    // will be initialized by MINION LWLPROD flow
    //
    // We must enable interrupts at the top levels in LWLW, LWLIPT_COMMON,
    // LWLIPT_LNK and MINION
    //

    // LWLW
    regval = DRF_NUM(_LWLCTRL_COMMON, _INTR_0_MASK, _FATAL,       0x1) |
             DRF_NUM(_LWLCTRL_COMMON, _INTR_0_MASK, _NONFATAL,    0x1) |
             DRF_NUM(_LWLCTRL_COMMON, _INTR_0_MASK, _CORRECTABLE, 0x1) |
             DRF_NUM(_LWLCTRL_COMMON, _INTR_0_MASK, _INTR0,       0x1) |
             DRF_NUM(_LWLCTRL_COMMON, _INTR_0_MASK, _INTR1,       0x1);
    LWSWITCH_BCAST_WR32_LR10(device, LWLW, _LWLCTRL_COMMON, _INTR_0_MASK, regval);

    regval = DRF_NUM(_LWLCTRL_COMMON, _INTR_1_MASK, _FATAL,       0x1) |
             DRF_NUM(_LWLCTRL_COMMON, _INTR_1_MASK, _NONFATAL,    0x1) |
             DRF_NUM(_LWLCTRL_COMMON, _INTR_1_MASK, _CORRECTABLE, 0x1) |
             DRF_NUM(_LWLCTRL_COMMON, _INTR_1_MASK, _INTR0,       0x1) |
             DRF_NUM(_LWLCTRL_COMMON, _INTR_1_MASK, _INTR1,       0x1);
    LWSWITCH_BCAST_WR32_LR10(device, LWLW, _LWLCTRL_COMMON, _INTR_1_MASK, regval);

    regval = DRF_NUM(_LWLCTRL_COMMON, _INTR_2_MASK, _FATAL,       0x1) |
             DRF_NUM(_LWLCTRL_COMMON, _INTR_2_MASK, _NONFATAL,    0x1) |
             DRF_NUM(_LWLCTRL_COMMON, _INTR_2_MASK, _CORRECTABLE, 0x1) |
             DRF_NUM(_LWLCTRL_COMMON, _INTR_2_MASK, _INTR0,       0x1) |
             DRF_NUM(_LWLCTRL_COMMON, _INTR_2_MASK, _INTR1,       0x1);
    LWSWITCH_BCAST_WR32_LR10(device, LWLW, _LWLCTRL_COMMON, _INTR_2_MASK, regval);

    // LWLW link
    for (i = 0; i < LW_LWLCTRL_LINK_INTR_0_MASK__SIZE_1; i++)
    {
        regval = DRF_NUM(_LWLCTRL_LINK, _INTR_0_MASK, _FATAL,       0x1) |
                 DRF_NUM(_LWLCTRL_LINK, _INTR_0_MASK, _NONFATAL,    0x1) |
                 DRF_NUM(_LWLCTRL_LINK, _INTR_0_MASK, _CORRECTABLE, 0x1) |
                 DRF_NUM(_LWLCTRL_LINK, _INTR_0_MASK, _INTR0,       0x1) |
                 DRF_NUM(_LWLCTRL_LINK, _INTR_0_MASK, _INTR1,       0x1);
        LWSWITCH_BCAST_WR32_LR10(device, LWLW, _LWLCTRL_LINK, _INTR_0_MASK(i), regval);

        regval = DRF_NUM(_LWLCTRL_LINK, _INTR_1_MASK, _FATAL,       0x1) |
                 DRF_NUM(_LWLCTRL_LINK, _INTR_1_MASK, _NONFATAL,    0x1) |
                 DRF_NUM(_LWLCTRL_LINK, _INTR_1_MASK, _CORRECTABLE, 0x1) |
                 DRF_NUM(_LWLCTRL_LINK, _INTR_1_MASK, _INTR0,       0x1) |
                 DRF_NUM(_LWLCTRL_LINK, _INTR_1_MASK, _INTR1,       0x1);
        LWSWITCH_BCAST_WR32_LR10(device, LWLW, _LWLCTRL_LINK, _INTR_1_MASK(i), regval);

        regval = DRF_NUM(_LWLCTRL_LINK, _INTR_2_MASK, _FATAL,       0x1) |
                 DRF_NUM(_LWLCTRL_LINK, _INTR_2_MASK, _NONFATAL,    0x1) |
                 DRF_NUM(_LWLCTRL_LINK, _INTR_2_MASK, _CORRECTABLE, 0x1) |
                 DRF_NUM(_LWLCTRL_LINK, _INTR_2_MASK, _INTR0,       0x1) |
                 DRF_NUM(_LWLCTRL_LINK, _INTR_2_MASK, _INTR1,       0x1);
        LWSWITCH_BCAST_WR32_LR10(device, LWLW, _LWLCTRL_LINK, _INTR_2_MASK(i), regval);
    }

    // LWLIPT_COMMON
    regval = DRF_NUM(_LWLIPT_COMMON, _INTR_CONTROL_COMMON, _INT0_EN, 0x1) |
             DRF_NUM(_LWLIPT_COMMON, _INTR_CONTROL_COMMON, _INT1_EN, 0x1);

    LWSWITCH_BCAST_WR32_LR10(device, LWLIPT, _LWLIPT_COMMON, _INTR_CONTROL_COMMON, regval);

    // LWLIPT_LNK
    regval = DRF_NUM(_LWLIPT_LNK, _INTR_CONTROL_LINK, _INT0_EN, 0x1) |
             DRF_NUM(_LWLIPT_LNK, _INTR_CONTROL_LINK, _INT1_EN, 0x1);
    LWSWITCH_BCAST_WR32_LR10(device, LWLIPT_LNK, _LWLIPT_LNK, _INTR_CONTROL_LINK, regval);

    // MINION
    for (i = 0; i < NUM_MINION_ENGINE_LR10; ++i)
    {
        if (!LWSWITCH_ENG_VALID_LR10(device, MINION, i))
        {
            continue;
        }

        _lwswitch_initialize_minion_interrupts(device,i);
    }
}

static void
_lwswitch_initialize_nxbar_tileout_interrupts
(
    lwswitch_device *device
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LwU32 report_fatal;
    LwU32 tileout;

    report_fatal =
        DRF_NUM(_NXBAR, _TC_TILEOUT0_ERR_FATAL_INTR_EN, _INGRESS_BUFFER_OVERFLOW, 1)     |
        DRF_NUM(_NXBAR, _TC_TILEOUT0_ERR_FATAL_INTR_EN, _INGRESS_BUFFER_UNDERFLOW, 1)    |
        DRF_NUM(_NXBAR, _TC_TILEOUT0_ERR_FATAL_INTR_EN, _EGRESS_CREDIT_OVERFLOW, 1)      |
        DRF_NUM(_NXBAR, _TC_TILEOUT0_ERR_FATAL_INTR_EN, _EGRESS_CREDIT_UNDERFLOW, 1)     |
        DRF_NUM(_NXBAR, _TC_TILEOUT0_ERR_FATAL_INTR_EN, _INGRESS_NON_BURSTY_PKT, 1)      |
        DRF_NUM(_NXBAR, _TC_TILEOUT0_ERR_FATAL_INTR_EN, _INGRESS_NON_STICKY_PKT, 1)      |
        DRF_NUM(_NXBAR, _TC_TILEOUT0_ERR_FATAL_INTR_EN, _INGRESS_BURST_GT_9_DATA_VC, 1)  |
        DRF_NUM(_NXBAR, _TC_TILEOUT0_ERR_FATAL_INTR_EN, _EGRESS_CDT_PARITY_ERROR, 1);

    for (tileout = 0; tileout < NUM_NXBAR_TILEOUTS_PER_TC_LR10; tileout++)
    {
        LWSWITCH_BCAST_WR32_LR10(device, NXBAR, _NXBAR, _TC_TILEOUT_ERR_FATAL_INTR_EN(tileout), report_fatal);
    }

    chip_device->intr_mask.tileout.fatal = report_fatal;
}

static void
_lwswitch_initialize_nxbar_tile_interrupts
(
    lwswitch_device *device
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LwU32 report_fatal;

    report_fatal =
        DRF_NUM(_NXBAR, _TILE_ERR_FATAL_INTR_EN, _INGRESS_BUFFER_OVERFLOW, 1)     |
        DRF_NUM(_NXBAR, _TILE_ERR_FATAL_INTR_EN, _INGRESS_BUFFER_UNDERFLOW, 1)    |
        DRF_NUM(_NXBAR, _TILE_ERR_FATAL_INTR_EN, _EGRESS_CREDIT_OVERFLOW, 1)      |
        DRF_NUM(_NXBAR, _TILE_ERR_FATAL_INTR_EN, _EGRESS_CREDIT_UNDERFLOW, 1)     |
        DRF_NUM(_NXBAR, _TILE_ERR_FATAL_INTR_EN, _INGRESS_NON_BURSTY_PKT, 1)      |
        DRF_NUM(_NXBAR, _TILE_ERR_FATAL_INTR_EN, _INGRESS_NON_STICKY_PKT, 1)      |
        DRF_NUM(_NXBAR, _TILE_ERR_FATAL_INTR_EN, _INGRESS_BURST_GT_9_DATA_VC, 1)  |
        DRF_NUM(_NXBAR, _TILE_ERR_FATAL_INTR_EN, _INGRESS_PKT_ILWALID_DST, 1)     |
        DRF_NUM(_NXBAR, _TILE_ERR_FATAL_INTR_EN, _INGRESS_PKT_PARITY_ERROR, 1);

    LWSWITCH_BCAST_WR32_LR10(device, TILE, _NXBAR, _TILE_ERR_FATAL_INTR_EN, report_fatal);

    chip_device->intr_mask.tile.fatal = report_fatal;
}

static void
_lwswitch_initialize_nxbar_interrupts
(
    lwswitch_device *device
)
{
    _lwswitch_initialize_nxbar_tile_interrupts(device);
    _lwswitch_initialize_nxbar_tileout_interrupts(device);
}

static void
_lwswitch_initialize_route_interrupts
(
    lwswitch_device *device
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LwU32 enable;
    LwU32 report_fatal;
    LwU32 report_nonfatal;
    LwU32 contain;

    report_fatal =
        DRF_DEF(_ROUTE, _ERR_FATAL_REPORT_EN_0, _ROUTEBUFERR, _ENABLE)            |
        DRF_DEF(_ROUTE, _ERR_FATAL_REPORT_EN_0, _NOPORTDEFINEDERR, _DISABLE)      |
        DRF_DEF(_ROUTE, _ERR_FATAL_REPORT_EN_0, _ILWALIDROUTEPOLICYERR, _DISABLE) |
        DRF_DEF(_ROUTE, _ERR_FATAL_REPORT_EN_0, _GLT_ECC_LIMIT_ERR, _DISABLE)     |
        DRF_DEF(_ROUTE, _ERR_FATAL_REPORT_EN_0, _GLT_ECC_DBE_ERR, _ENABLE)        |
        DRF_DEF(_ROUTE, _ERR_FATAL_REPORT_EN_0, _TRANSDONERESVERR, _DISABLE)      |
        DRF_DEF(_ROUTE, _ERR_FATAL_REPORT_EN_0, _PDCTRLPARERR, _ENABLE)           |
        DRF_DEF(_ROUTE, _ERR_FATAL_REPORT_EN_0, _LWS_ECC_LIMIT_ERR, _DISABLE)     |
        DRF_DEF(_ROUTE, _ERR_FATAL_REPORT_EN_0, _LWS_ECC_DBE_ERR, _ENABLE)        |
        DRF_DEF(_ROUTE, _ERR_FATAL_REPORT_EN_0, _CDTPARERR, _ENABLE);

    report_nonfatal =
        DRF_DEF(_ROUTE, _ERR_NON_FATAL_REPORT_EN_0, _ROUTEBUFERR, _DISABLE)          |
        DRF_DEF(_ROUTE, _ERR_NON_FATAL_REPORT_EN_0, _NOPORTDEFINEDERR, _ENABLE)      |
        DRF_DEF(_ROUTE, _ERR_NON_FATAL_REPORT_EN_0, _ILWALIDROUTEPOLICYERR, _ENABLE) |
        DRF_DEF(_ROUTE, _ERR_NON_FATAL_REPORT_EN_0, _GLT_ECC_LIMIT_ERR, _DISABLE)    |
        DRF_DEF(_ROUTE, _ERR_NON_FATAL_REPORT_EN_0, _GLT_ECC_DBE_ERR, _DISABLE)      |
        DRF_DEF(_ROUTE, _ERR_NON_FATAL_REPORT_EN_0, _TRANSDONERESVERR, _DISABLE)     |
        DRF_DEF(_ROUTE, _ERR_NON_FATAL_REPORT_EN_0, _PDCTRLPARERR, _DISABLE)         |
        DRF_DEF(_ROUTE, _ERR_NON_FATAL_REPORT_EN_0, _LWS_ECC_LIMIT_ERR, _ENABLE)     |
        DRF_DEF(_ROUTE, _ERR_NON_FATAL_REPORT_EN_0, _LWS_ECC_DBE_ERR, _DISABLE)      |
        DRF_DEF(_ROUTE, _ERR_NON_FATAL_REPORT_EN_0, _CDTPARERR, _DISABLE);

    contain =
        DRF_DEF(_ROUTE, _ERR_CONTAIN_EN_0, _ROUTEBUFERR, __PROD)           |
        DRF_DEF(_ROUTE, _ERR_CONTAIN_EN_0, _NOPORTDEFINEDERR, __PROD)      |
        DRF_DEF(_ROUTE, _ERR_CONTAIN_EN_0, _ILWALIDROUTEPOLICYERR, __PROD) |
        DRF_DEF(_ROUTE, _ERR_CONTAIN_EN_0, _GLT_ECC_LIMIT_ERR, __PROD)     |
        DRF_DEF(_ROUTE, _ERR_CONTAIN_EN_0, _GLT_ECC_DBE_ERR, __PROD)       |
        DRF_DEF(_ROUTE, _ERR_CONTAIN_EN_0, _TRANSDONERESVERR, __PROD)      |
        DRF_DEF(_ROUTE, _ERR_CONTAIN_EN_0, _PDCTRLPARERR, __PROD)          |
        DRF_DEF(_ROUTE, _ERR_CONTAIN_EN_0, _LWS_ECC_LIMIT_ERR, __PROD)     |
        DRF_DEF(_ROUTE, _ERR_CONTAIN_EN_0, _LWS_ECC_DBE_ERR, __PROD)       |
        DRF_DEF(_ROUTE, _ERR_CONTAIN_EN_0, _CDTPARERR, __PROD);

    enable = report_fatal | report_nonfatal;
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _ROUTE, _ERR_LOG_EN_0, enable);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _ROUTE, _ERR_FATAL_REPORT_EN_0, report_fatal);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _ROUTE, _ERR_NON_FATAL_REPORT_EN_0, report_nonfatal);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _ROUTE, _ERR_CORRECTABLE_REPORT_EN_0, 0);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _ROUTE, _ERR_CONTAIN_EN_0, contain);

    chip_device->intr_mask.route.fatal = report_fatal;
    chip_device->intr_mask.route.nonfatal = report_nonfatal;
}

static void
_lwswitch_initialize_ingress_interrupts
(
    lwswitch_device *device
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LwU32 enable;
    LwU32 report_fatal;
    LwU32 report_nonfatal;
    LwU32 contain;

    report_fatal =
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _CMDDECODEERR, _ENABLE)              |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _NCISOC_HDR_ECC_DBE_ERR, _ENABLE)    |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _ILWALIDVCSET, _ENABLE)              |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _REMAPTAB_ECC_DBE_ERR, _ENABLE)      |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _RIDTAB_ECC_DBE_ERR, _ENABLE)        |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _RLANTAB_ECC_DBE_ERR, _ENABLE)       |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _NCISOC_PARITY_ERR, _ENABLE)         |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _REQCONTEXTMISMATCHERR, _DISABLE)    |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _ACLFAIL, _DISABLE)                  |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _NCISOC_HDR_ECC_LIMIT_ERR, _DISABLE) |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _ADDRBOUNDSERR, _DISABLE)            |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _RIDTABCFGERR, _DISABLE)             |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _RLANTABCFGERR, _DISABLE)            |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _REMAPTAB_ECC_LIMIT_ERR, _DISABLE)   |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _RIDTAB_ECC_LIMIT_ERR, _DISABLE)     |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _RLANTAB_ECC_LIMIT_ERR, _DISABLE)    |
        DRF_DEF(_INGRESS, _ERR_FATAL_REPORT_EN_0, _ADDRTYPEERR, _DISABLE);

    report_nonfatal =
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _REQCONTEXTMISMATCHERR, _ENABLE)    |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _ACLFAIL, _ENABLE)                  |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _NCISOC_HDR_ECC_LIMIT_ERR, _ENABLE) |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _ADDRBOUNDSERR, _ENABLE)            |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _RIDTABCFGERR, _ENABLE)             |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _RLANTABCFGERR, _ENABLE)            |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _REMAPTAB_ECC_LIMIT_ERR, _DISABLE)  |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _RIDTAB_ECC_LIMIT_ERR, _DISABLE)    |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _RLANTAB_ECC_LIMIT_ERR, _DISABLE)   |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _ADDRTYPEERR, _ENABLE)              |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _CMDDECODEERR, _DISABLE)            |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _NCISOC_HDR_ECC_DBE_ERR, _DISABLE)  |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _ILWALIDVCSET, _DISABLE)            |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _REMAPTAB_ECC_DBE_ERR, _DISABLE)    |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _RIDTAB_ECC_DBE_ERR, _DISABLE)      |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _RLANTAB_ECC_DBE_ERR, _DISABLE)     |
        DRF_DEF(_INGRESS, _ERR_NON_FATAL_REPORT_EN_0, _NCISOC_PARITY_ERR, _DISABLE);

    contain =
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _REQCONTEXTMISMATCHERR, __PROD)    |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _ACLFAIL, __PROD)                  |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _NCISOC_HDR_ECC_LIMIT_ERR, __PROD) |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _ADDRBOUNDSERR, __PROD)            |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _RIDTABCFGERR, __PROD)             |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _RLANTABCFGERR, __PROD)            |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _REMAPTAB_ECC_LIMIT_ERR, __PROD)   |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _RIDTAB_ECC_LIMIT_ERR, __PROD)     |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _RLANTAB_ECC_LIMIT_ERR, __PROD)    |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _ADDRTYPEERR, __PROD)              |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _CMDDECODEERR, __PROD)             |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _NCISOC_HDR_ECC_DBE_ERR, __PROD)   |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _ILWALIDVCSET, __PROD)             |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _REMAPTAB_ECC_DBE_ERR, __PROD)     |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _RIDTAB_ECC_DBE_ERR, __PROD)       |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _RLANTAB_ECC_DBE_ERR, __PROD)      |
        DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _NCISOC_PARITY_ERR, __PROD);

    enable = report_fatal | report_nonfatal;
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _INGRESS, _ERR_LOG_EN_0, enable);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _INGRESS, _ERR_FATAL_REPORT_EN_0, report_fatal);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _INGRESS, _ERR_NON_FATAL_REPORT_EN_0, report_nonfatal);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _INGRESS, _ERR_CORRECTABLE_REPORT_EN_0, 0);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _INGRESS, _ERR_CONTAIN_EN_0, contain);

    chip_device->intr_mask.ingress.fatal = report_fatal;
    chip_device->intr_mask.ingress.nonfatal = report_nonfatal;
}

static void
_lwswitch_initialize_egress_interrupts
(
    lwswitch_device *device
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LwU32 enable;
    LwU32 report_fatal;
    LwU32 report_nonfatal;
    LwU32 contain;

    report_fatal =
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _EGRESSBUFERR, _ENABLE)                 |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _PKTROUTEERR, _ENABLE)                  |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _SEQIDERR, _ENABLE)                     |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _NXBAR_HDR_ECC_LIMIT_ERR, _DISABLE)     |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _NXBAR_HDR_ECC_DBE_ERR, _ENABLE)        |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _RAM_OUT_HDR_ECC_LIMIT_ERR, _DISABLE)   |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _RAM_OUT_HDR_ECC_DBE_ERR, _ENABLE)      |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _NCISOCCREDITOVFL, _ENABLE)             |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _REQTGTIDMISMATCHERR, _ENABLE)          |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _RSPREQIDMISMATCHERR, _ENABLE)          |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _URRSPERR, _DISABLE)                    |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _PRIVRSPERR, _DISABLE)                  |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _HWRSPERR, _DISABLE)                    |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _NXBAR_HDR_PARITY_ERR, _ENABLE)         |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _NCISOC_CREDIT_PARITY_ERR, _ENABLE)     |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _NXBAR_FLITTYPE_MISMATCH_ERR, _ENABLE)  |
        DRF_DEF(_EGRESS, _ERR_FATAL_REPORT_EN_0, _CREDIT_TIME_OUT_ERR, _ENABLE);

    report_nonfatal =
        DRF_DEF(_EGRESS, _ERR_NON_FATAL_REPORT_EN_0, _EGRESSBUFERR, _DISABLE)                 |
        DRF_DEF(_EGRESS, _ERR_NON_FATAL_REPORT_EN_0, _PKTROUTEERR, _DISABLE)                  |
        DRF_DEF(_EGRESS, _ERR_NON_FATAL_REPORT_EN_0, _SEQIDERR, _DISABLE)                     |
        DRF_DEF(_EGRESS, _ERR_NON_FATAL_REPORT_EN_0, _NXBAR_HDR_ECC_LIMIT_ERR, _ENABLE)       |
        DRF_DEF(_EGRESS, _ERR_NON_FATAL_REPORT_EN_0, _NXBAR_HDR_ECC_DBE_ERR, _DISABLE)        |
        DRF_DEF(_EGRESS, _ERR_NON_FATAL_REPORT_EN_0, _RAM_OUT_HDR_ECC_LIMIT_ERR, _ENABLE)     |
        DRF_DEF(_EGRESS, _ERR_NON_FATAL_REPORT_EN_0, _RAM_OUT_HDR_ECC_DBE_ERR, _DISABLE)      |
        DRF_DEF(_EGRESS, _ERR_NON_FATAL_REPORT_EN_0, _NCISOCCREDITOVFL, _DISABLE)             |
        DRF_DEF(_EGRESS, _ERR_NON_FATAL_REPORT_EN_0, _REQTGTIDMISMATCHERR, _DISABLE)          |
        DRF_DEF(_EGRESS, _ERR_NON_FATAL_REPORT_EN_0, _RSPREQIDMISMATCHERR, _DISABLE)          |
        DRF_DEF(_EGRESS, _ERR_NON_FATAL_REPORT_EN_0, _URRSPERR, _ENABLE)                      |
        DRF_DEF(_EGRESS, _ERR_NON_FATAL_REPORT_EN_0, _PRIVRSPERR, _ENABLE)                    |
        DRF_DEF(_EGRESS, _ERR_NON_FATAL_REPORT_EN_0, _HWRSPERR, _ENABLE)                      |
        DRF_DEF(_EGRESS, _ERR_NON_FATAL_REPORT_EN_0, _NXBAR_HDR_PARITY_ERR, _DISABLE)         |
        DRF_DEF(_EGRESS, _ERR_NON_FATAL_REPORT_EN_0, _NCISOC_CREDIT_PARITY_ERR, _DISABLE)     |
        DRF_DEF(_EGRESS, _ERR_NON_FATAL_REPORT_EN_0, _NXBAR_FLITTYPE_MISMATCH_ERR, _DISABLE)  |
        DRF_DEF(_EGRESS, _ERR_NON_FATAL_REPORT_EN_0, _CREDIT_TIME_OUT_ERR, _DISABLE);

    contain =
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _EGRESSBUFERR, __PROD)                 |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _PKTROUTEERR, __PROD)                  |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _SEQIDERR, __PROD)                     |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _NXBAR_HDR_ECC_LIMIT_ERR, __PROD)      |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _NXBAR_HDR_ECC_DBE_ERR, __PROD)        |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _RAM_OUT_HDR_ECC_LIMIT_ERR, __PROD)    |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _RAM_OUT_HDR_ECC_DBE_ERR, __PROD)      |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _NCISOCCREDITOVFL, __PROD)             |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _REQTGTIDMISMATCHERR, __PROD)          |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _RSPREQIDMISMATCHERR, __PROD)          |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _URRSPERR, __PROD)                     |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _PRIVRSPERR, __PROD)                   |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _HWRSPERR, __PROD)                     |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _NXBAR_HDR_PARITY_ERR, __PROD)         |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _NCISOC_CREDIT_PARITY_ERR, __PROD)     |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _NXBAR_FLITTYPE_MISMATCH_ERR, __PROD)  |
        DRF_DEF(_EGRESS, _ERR_CONTAIN_EN_0, _CREDIT_TIME_OUT_ERR, __PROD);

    enable = report_fatal | report_nonfatal;

    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _EGRESS, _ERR_LOG_EN_0, enable);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _EGRESS, _ERR_FATAL_REPORT_EN_0, report_fatal);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _EGRESS, _ERR_NON_FATAL_REPORT_EN_0, report_nonfatal);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _EGRESS, _ERR_CORRECTABLE_REPORT_EN_0, 0);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _EGRESS, _ERR_CONTAIN_EN_0, contain);

    chip_device->intr_mask.egress.fatal = report_fatal;
    chip_device->intr_mask.egress.nonfatal = report_nonfatal;
}

static void
_lwswitch_initialize_tstate_interrupts
(
    lwswitch_device *device
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LwU32 enable;
    LwU32 report_fatal;
    LwU32 report_nonfatal;
    LwU32 contain;

    // TD_TID errors are disbaled on both fatal & non-fatal trees since TD_TID RAM is no longer used.

    report_fatal =
        DRF_DEF(_TSTATE, _ERR_FATAL_REPORT_EN_0, _TAGPOOLBUFERR, _ENABLE)              |
        DRF_DEF(_TSTATE, _ERR_FATAL_REPORT_EN_0, _TAGPOOL_ECC_LIMIT_ERR, _DISABLE)     |
        DRF_DEF(_TSTATE, _ERR_FATAL_REPORT_EN_0, _TAGPOOL_ECC_DBE_ERR, _ENABLE)        |
        DRF_DEF(_TSTATE, _ERR_FATAL_REPORT_EN_0, _CRUMBSTOREBUFERR, _ENABLE)           |
        DRF_DEF(_TSTATE, _ERR_FATAL_REPORT_EN_0, _CRUMBSTORE_ECC_LIMIT_ERR, _DISABLE)  |
        DRF_DEF(_TSTATE, _ERR_FATAL_REPORT_EN_0, _CRUMBSTORE_ECC_DBE_ERR, _ENABLE)     |
        DRF_DEF(_TSTATE, _ERR_FATAL_REPORT_EN_0, _TD_TID_RAMBUFERR, _DISABLE)          |
        DRF_DEF(_TSTATE, _ERR_FATAL_REPORT_EN_0, _TD_TID_RAM_ECC_LIMIT_ERR, _DISABLE)  |
        DRF_DEF(_TSTATE, _ERR_FATAL_REPORT_EN_0, _TD_TID_RAM_ECC_DBE_ERR, _DISABLE)    |
        DRF_DEF(_TSTATE, _ERR_FATAL_REPORT_EN_0, _ATO_ERR, _ENABLE)                    |
        DRF_DEF(_TSTATE, _ERR_FATAL_REPORT_EN_0, _CAMRSP_ERR, _ENABLE);

    report_nonfatal =
        DRF_DEF(_TSTATE, _ERR_NON_FATAL_REPORT_EN_0, _TAGPOOLBUFERR, _DISABLE)             |
        DRF_DEF(_TSTATE, _ERR_NON_FATAL_REPORT_EN_0, _TAGPOOL_ECC_LIMIT_ERR, _ENABLE)      |
        DRF_DEF(_TSTATE, _ERR_NON_FATAL_REPORT_EN_0, _TAGPOOL_ECC_DBE_ERR, _DISABLE)       |
        DRF_DEF(_TSTATE, _ERR_NON_FATAL_REPORT_EN_0, _CRUMBSTOREBUFERR, _DISABLE)          |
        DRF_DEF(_TSTATE, _ERR_NON_FATAL_REPORT_EN_0, _CRUMBSTORE_ECC_LIMIT_ERR, _ENABLE)   |
        DRF_DEF(_TSTATE, _ERR_NON_FATAL_REPORT_EN_0, _CRUMBSTORE_ECC_DBE_ERR, _DISABLE)    |
        DRF_DEF(_TSTATE, _ERR_NON_FATAL_REPORT_EN_0, _TD_TID_RAMBUFERR, _DISABLE)          |
        DRF_DEF(_TSTATE, _ERR_NON_FATAL_REPORT_EN_0, _TD_TID_RAM_ECC_LIMIT_ERR, _DISABLE)  |
        DRF_DEF(_TSTATE, _ERR_NON_FATAL_REPORT_EN_0, _TD_TID_RAM_ECC_DBE_ERR, _DISABLE)    |
        DRF_DEF(_TSTATE, _ERR_NON_FATAL_REPORT_EN_0, _ATO_ERR, _DISABLE)                   |
        DRF_DEF(_TSTATE, _ERR_NON_FATAL_REPORT_EN_0, _CAMRSP_ERR, _DISABLE);

    contain =
        DRF_DEF(_TSTATE, _ERR_CONTAIN_EN_0, _TAGPOOLBUFERR, __PROD)             |
        DRF_DEF(_TSTATE, _ERR_CONTAIN_EN_0, _TAGPOOL_ECC_LIMIT_ERR, __PROD)     |
        DRF_DEF(_TSTATE, _ERR_CONTAIN_EN_0, _TAGPOOL_ECC_DBE_ERR, __PROD)       |
        DRF_DEF(_TSTATE, _ERR_CONTAIN_EN_0, _CRUMBSTOREBUFERR, __PROD)          |
        DRF_DEF(_TSTATE, _ERR_CONTAIN_EN_0, _CRUMBSTORE_ECC_LIMIT_ERR, __PROD)  |
        DRF_DEF(_TSTATE, _ERR_CONTAIN_EN_0, _CRUMBSTORE_ECC_DBE_ERR, __PROD)    |
        DRF_DEF(_TSTATE, _ERR_CONTAIN_EN_0, _TD_TID_RAMBUFERR, __PROD)          |
        DRF_DEF(_TSTATE, _ERR_CONTAIN_EN_0, _TD_TID_RAM_ECC_LIMIT_ERR, __PROD)  |
        DRF_DEF(_TSTATE, _ERR_CONTAIN_EN_0, _TD_TID_RAM_ECC_DBE_ERR, __PROD)    |
        DRF_DEF(_TSTATE, _ERR_CONTAIN_EN_0, _ATO_ERR, __PROD)                   |
        DRF_DEF(_TSTATE, _ERR_CONTAIN_EN_0, _CAMRSP_ERR, __PROD);

    enable = report_fatal | report_nonfatal;
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _TSTATE, _ERR_LOG_EN_0, enable);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _TSTATE, _ERR_FATAL_REPORT_EN_0, report_fatal);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _TSTATE, _ERR_NON_FATAL_REPORT_EN_0, report_nonfatal);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _TSTATE, _ERR_CORRECTABLE_REPORT_EN_0, 0);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _TSTATE, _ERR_CONTAIN_EN_0, contain);

    chip_device->intr_mask.tstate.fatal = report_fatal;
    chip_device->intr_mask.tstate.nonfatal = report_nonfatal;
}

static void
_lwswitch_initialize_sourcetrack_interrupts
(
    lwswitch_device *device
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LwU32 enable;
    LwU32 report_fatal;
    LwU32 report_nonfatal;
    LwU32 contain;

    report_fatal =
        DRF_DEF(_SOURCETRACK, _ERR_FATAL_REPORT_EN_0, _CREQ_TCEN0_CRUMBSTORE_ECC_LIMIT_ERR, _DISABLE)    |
        DRF_DEF(_SOURCETRACK, _ERR_FATAL_REPORT_EN_0, _CREQ_TCEN0_TD_CRUMBSTORE_ECC_LIMIT_ERR, _DISABLE) |
        DRF_DEF(_SOURCETRACK, _ERR_FATAL_REPORT_EN_0, _CREQ_TCEN1_CRUMBSTORE_ECC_LIMIT_ERR, _DISABLE)    |
        DRF_DEF(_SOURCETRACK, _ERR_FATAL_REPORT_EN_0, _CREQ_TCEN0_CRUMBSTORE_ECC_DBE_ERR, _ENABLE)       |
        DRF_DEF(_SOURCETRACK, _ERR_FATAL_REPORT_EN_0, _CREQ_TCEN0_TD_CRUMBSTORE_ECC_DBE_ERR, _DISABLE)   |
        DRF_DEF(_SOURCETRACK, _ERR_FATAL_REPORT_EN_0, _CREQ_TCEN1_CRUMBSTORE_ECC_DBE_ERR, _ENABLE)       |
        DRF_DEF(_SOURCETRACK, _ERR_FATAL_REPORT_EN_0, _SOURCETRACK_TIME_OUT_ERR, _ENABLE);

    report_nonfatal =
        DRF_DEF(_SOURCETRACK, _ERR_NON_FATAL_REPORT_EN_0, _CREQ_TCEN0_CRUMBSTORE_ECC_LIMIT_ERR, _ENABLE)     |
        DRF_DEF(_SOURCETRACK, _ERR_NON_FATAL_REPORT_EN_0, _CREQ_TCEN0_TD_CRUMBSTORE_ECC_LIMIT_ERR, _DISABLE) |
        DRF_DEF(_SOURCETRACK, _ERR_NON_FATAL_REPORT_EN_0, _CREQ_TCEN1_CRUMBSTORE_ECC_LIMIT_ERR, _ENABLE)     |
        DRF_DEF(_SOURCETRACK, _ERR_NON_FATAL_REPORT_EN_0, _CREQ_TCEN0_CRUMBSTORE_ECC_DBE_ERR, _DISABLE)      |
        DRF_DEF(_SOURCETRACK, _ERR_NON_FATAL_REPORT_EN_0, _CREQ_TCEN0_TD_CRUMBSTORE_ECC_DBE_ERR, _DISABLE)   |
        DRF_DEF(_SOURCETRACK, _ERR_NON_FATAL_REPORT_EN_0, _CREQ_TCEN1_CRUMBSTORE_ECC_DBE_ERR, _DISABLE)      |
        DRF_DEF(_SOURCETRACK, _ERR_NON_FATAL_REPORT_EN_0, _SOURCETRACK_TIME_OUT_ERR, _DISABLE);

    contain =
        DRF_DEF(_SOURCETRACK, _ERR_CONTAIN_EN_0, _CREQ_TCEN0_CRUMBSTORE_ECC_LIMIT_ERR, __PROD)    |
        DRF_DEF(_SOURCETRACK, _ERR_CONTAIN_EN_0, _CREQ_TCEN0_TD_CRUMBSTORE_ECC_LIMIT_ERR, __PROD) |
        DRF_DEF(_SOURCETRACK, _ERR_CONTAIN_EN_0, _CREQ_TCEN1_CRUMBSTORE_ECC_LIMIT_ERR, __PROD)    |
        DRF_DEF(_SOURCETRACK, _ERR_CONTAIN_EN_0, _CREQ_TCEN0_CRUMBSTORE_ECC_DBE_ERR, __PROD)      |
        DRF_DEF(_SOURCETRACK, _ERR_CONTAIN_EN_0, _CREQ_TCEN0_TD_CRUMBSTORE_ECC_DBE_ERR, __PROD)   |
        DRF_DEF(_SOURCETRACK, _ERR_CONTAIN_EN_0, _CREQ_TCEN1_CRUMBSTORE_ECC_DBE_ERR, __PROD)      |
        DRF_DEF(_SOURCETRACK, _ERR_CONTAIN_EN_0, _SOURCETRACK_TIME_OUT_ERR, __PROD);

    enable = report_fatal | report_nonfatal;
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _SOURCETRACK, _ERR_LOG_EN_0, enable);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _SOURCETRACK, _ERR_FATAL_REPORT_EN_0, report_fatal);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _SOURCETRACK, _ERR_NON_FATAL_REPORT_EN_0, report_nonfatal);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _SOURCETRACK, _ERR_CORRECTABLE_REPORT_EN_0, 0);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _SOURCETRACK, _ERR_CONTAIN_EN_0, contain);

    chip_device->intr_mask.sourcetrack.fatal = report_fatal;
    chip_device->intr_mask.sourcetrack.nonfatal = report_nonfatal;

}

void
_lwswitch_initialize_nport_interrupts
(
    lwswitch_device *device
)
{
    LwU32 val;

    val =
        DRF_NUM(_NPORT, _ERR_CONTROL_COMMON_NPORT, _CORRECTABLEENABLE, 1) |
        DRF_NUM(_NPORT, _ERR_CONTROL_COMMON_NPORT, _FATALENABLE, 1) |
        DRF_NUM(_NPORT, _ERR_CONTROL_COMMON_NPORT, _NONFATALENABLE, 1);
    LWSWITCH_NPORT_MC_BCAST_WR32_LR10(device, _NPORT, _ERR_CONTROL_COMMON_NPORT, val);

    _lwswitch_initialize_route_interrupts(device);
    _lwswitch_initialize_ingress_interrupts(device);
    _lwswitch_initialize_egress_interrupts(device);
    _lwswitch_initialize_tstate_interrupts(device);
    _lwswitch_initialize_sourcetrack_interrupts(device);
}

static void
_lwswitch_initialize_saw_interrupts
(
    lwswitch_device *device
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);

    LWSWITCH_SAW_WR32_LR10(device, _LWLSAW_LWSPMC, _INTR_EN_SET_CORRECTABLE, chip_device->intr_enable_corr);
    LWSWITCH_SAW_WR32_LR10(device, _LWLSAW_LWSPMC, _INTR_EN_SET_FATAL,       chip_device->intr_enable_fatal);
    LWSWITCH_SAW_WR32_LR10(device, _LWLSAW_LWSPMC, _INTR_EN_SET_NONFATAL,    chip_device->intr_enable_nonfatal);
}

/*
 * Initialize interrupt tree HW for all units.
 *
 * Init and servicing both depend on bits matching across STATUS/MASK
 * and IErr STATUS/LOG/REPORT/CONTAIN registers.
 */
void
lwswitch_initialize_interrupt_tree_lr10
(
    lwswitch_device *device
)
{
    _lwswitch_build_top_interrupt_mask_lr10(device);

    // Initialize legacy interrupt tree - depends on reset to disable
    // unused interrupts
    LWSWITCH_REG_WR32(device, _PBUS, _INTR_0, 0xffffffff);

    // Clear prior saved PRI error data
    LWSWITCH_REG_WR32(device, _PBUS, _PRI_TIMEOUT_SAVE_0,
        DRF_DEF(_PBUS, _PRI_TIMEOUT_SAVE_0, _TO, _CLEAR));

    LWSWITCH_REG_WR32(device, _PBUS, _INTR_EN_0,
            DRF_DEF(_PBUS, _INTR_EN_0, _PRI_SQUASH, _ENABLED) |
            DRF_DEF(_PBUS, _INTR_EN_0, _PRI_FECSERR, _ENABLED) |
            DRF_DEF(_PBUS, _INTR_EN_0, _PRI_TIMEOUT, _ENABLED) |
            DRF_DEF(_PBUS, _INTR_EN_0, _SW, _ENABLED));

    // SAW block
    _lwswitch_initialize_saw_interrupts(device);

    // NPG/NPORT
    _lwswitch_initialize_nport_interrupts(device);

    // LWLIPT interrupts
    _lwswitch_initialize_lwlipt_interrupts_lr10(device);

    // NXBAR interrupts
    _lwswitch_initialize_nxbar_interrupts(device);
}

/*
 * @brief Service MINION Falcon interrupts on the requested interrupt tree
 *        Falcon Interrupts are a little in unqiue in how they are handled:#include <assert.h>
 *        IRQSTAT is used to read in interrupt status from FALCON
 *        IRQMASK is used to read in mask of interrupts
 *        IRQDEST is used to read in enabled interrupts that are routed to the HOST
 *
 *        IRQSTAT & IRQMASK gives the pending interrupting on this minion 
 *
 * @param[in] device   MINION on this device
 * @param[in] instance MINION instance
 *
 */
LwlStatus
lwswitch_minion_service_falcon_interrupts_lr10
(
    lwswitch_device *device,
    LwU32           instance
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, unhandled, intr, link;

    link = instance * LWSWITCH_LINKS_PER_MINION;
    report.raw_pending = LWSWITCH_MINION_RD32_LR10(device, instance, _CMINION, _FALCON_IRQSTAT);
    report.raw_enable = chip_device->intr_minion_dest;
    report.mask = LWSWITCH_MINION_RD32_LR10(device, instance, _CMINION, _FALCON_IRQMASK);

    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending; 

    bit = DRF_NUM(_CMINION_FALCON, _IRQSTAT, _WDTMR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_MINION_WATCHDOG, "MINION Watchdog timer ran out", LW_TRUE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_CMINION_FALCON, _IRQSTAT, _HALT, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_MINION_HALT, "MINION HALT", LW_TRUE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_CMINION_FALCON, _IRQSTAT, _EXTERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_MINION_EXTERR, "MINION EXTERR", LW_TRUE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_CMINION_FALCON, _IRQSTAT, _SWGEN0, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_PRINT(device, INFO,
                      "%s: Received MINION Falcon SWGEN0 interrupt on MINION %d.\n",
                      __FUNCTION__, instance);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_CMINION_FALCON, _IRQSTAT, _SWGEN1, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_PRINT(device, INFO,
                       "%s: Received MINION Falcon SWGEN1 interrupt on MINION %d.\n",
                      __FUNCTION__, instance);
        lwswitch_clear_flags(&unhandled, bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    if (device->link[link].fatal_error_oclwrred)
    {
        intr = LWSWITCH_MINION_RD32_LR10(device, instance, _MINION, _MINION_INTR_STALL_EN);
        intr = FLD_SET_DRF(_MINION, _MINION_INTR_STALL_EN, _FATAL, _DISABLE, intr);
        intr = FLD_SET_DRF(_MINION, _MINION_INTR_STALL_EN, _FALCON_STALL, _DISABLE, intr);
        LWSWITCH_MINION_WR32_LR10(device, instance, _MINION, _MINION_INTR_STALL_EN, intr);
    }

    // Write to IRQSCLR to clear status of interrupt
    LWSWITCH_MINION_WR32_LR10(device, instance, _CMINION, _FALCON_IRQSCLR, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

//
// Check if there are interrupts pending.
//
// On silicon/emulation we only use MSIs which are not shared, so this
// function does not need to be called.
//
// FSF/RTMsim does not model interrupts correctly.  The interrupt is shared
// with USB so we must check the HW status.  In addition we must disable
// interrupts to run the interrupt thread. On silicon this is done
// automatically in XVE.
//
// This is called in the ISR context by the Linux driver.  The WAR does
// access more of device outside the Linux mutex than it should. Sim only
// supports 1 device lwrrently so these fields are safe while interrupts
// are enabled.
//
// TODO: Bug 1881361 to remove the FSF WAR
//
LwlStatus
lwswitch_lib_check_interrupts_lr10
(
    lwswitch_device *device
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LwU32 saw_legacy_intr_enable = 0;
    LwU32 pending;

    if (IS_RTLSIM(device) || IS_FMODEL(device))
    {
        pending = LWSWITCH_REG_RD32(device, _PSMC, _INTR_LEGACY);
        pending &= chip_device->intr_enable_legacy;
        if (pending)
        {
            LWSWITCH_PRINT(device, WARN,
                "%s: _PSMC, _INTR_LEGACY pending (0x%0x)\n",
                __FUNCTION__, pending);
            return -LWL_MORE_PROCESSING_REQUIRED;
        }

        if (FLD_TEST_DRF_NUM(_PSMC, _INTR_EN_SET_LEGACY, _PTIMER, 1, chip_device->intr_enable_legacy))
        {
            saw_legacy_intr_enable = FLD_SET_DRF_NUM(_LWLSAW_LWSPMC, _INTR_EN_SET_LEGACY, _PTIMER_0, 1, saw_legacy_intr_enable);
            saw_legacy_intr_enable = FLD_SET_DRF_NUM(_LWLSAW_LWSPMC, _INTR_EN_SET_LEGACY, _PTIMER_1, 1, saw_legacy_intr_enable);
        }
        if (FLD_TEST_DRF_NUM(_PSMC, _INTR_EN_SET_LEGACY, _PMGR, 1, chip_device->intr_enable_legacy))
        {
            saw_legacy_intr_enable = FLD_SET_DRF_NUM(_LWLSAW_LWSPMC, _INTR_EN_SET_LEGACY, _PMGR_0, 1, saw_legacy_intr_enable);
            saw_legacy_intr_enable = FLD_SET_DRF_NUM(_LWLSAW_LWSPMC, _INTR_EN_SET_LEGACY, _PMGR_1, 1, saw_legacy_intr_enable);
        }

        pending = LWSWITCH_SAW_RD32_LR10(device, _LWLSAW_LWSPMC, _INTR_LEGACY);
        pending &= saw_legacy_intr_enable;
        if (pending)
        {
            LWSWITCH_PRINT(device, WARN,
                "%s: _LWLSAW_LWSPMC, _INTR_LEGACY pending (0x%0x)\n",
                __FUNCTION__, pending);
            return -LWL_MORE_PROCESSING_REQUIRED;
        }

        // Fatal Interrupts
        pending = LWSWITCH_SAW_RD32_LR10(device, _LWLSAW_LWSPMC, _INTR_FATAL);
        pending &= chip_device->intr_enable_fatal;
        if (pending)
        {
            LWSWITCH_PRINT(device, WARN,
                "%s: _LWLSAW_LWSPMC, _INTR_FATAL pending (0x%0x)\n",
                __FUNCTION__, pending);
            return -LWL_MORE_PROCESSING_REQUIRED;
        }

        // Non-Fatal interrupts
        pending = LWSWITCH_SAW_RD32_LR10(device, _LWLSAW_LWSPMC, _INTR_NONFATAL);
        pending &= chip_device->intr_enable_nonfatal;
        if (pending)
        {
            LWSWITCH_PRINT(device, WARN,
                "%s: _LWLSAW_LWSPMC, _INTR_NONFATAL pending (0x%0x)\n",
                __FUNCTION__, pending);
            return -LWL_MORE_PROCESSING_REQUIRED;
        }

        return LWL_SUCCESS;
    }
    else
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }
}

/*
 * The MSI interrupt block must be re-armed after servicing interrupts. This
 * write generates an EOI, which allows further MSIs to be triggered.
 */
static void
_lwswitch_rearm_msi_lr10
(
    lwswitch_device *device
)
{
    LWSWITCH_ENG_WR32_LR10(device, XVE, , 0, _XVE_CYA, _2, 0xff);
}

static LwlStatus
_lwswitch_service_pbus_lr10
(
    lwswitch_device *device
)
{
    LwU32 pending, mask, bit, unhandled;
    LwU32 save0, save1, save3, errCode;
    LWSWITCH_PRI_TIMEOUT_ERROR_LOG_TYPE pri_timeout = { 0 };

    pending = LWSWITCH_REG_RD32(device, _PBUS, _INTR_0);
    mask = LWSWITCH_REG_RD32(device, _PBUS, _INTR_EN_0);
    pending &= mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;

    bit = DRF_DEF(_PBUS, _INTR_0, _PRI_SQUASH, _PENDING) |
          DRF_DEF(_PBUS, _INTR_0, _PRI_FECSERR, _PENDING) |
          DRF_DEF(_PBUS, _INTR_0, _PRI_TIMEOUT, _PENDING);

    if (lwswitch_test_flags(pending, bit))
    {
        // PRI timeout is likely not recoverable
        LWSWITCH_REG_WR32(device, _PBUS, _INTR_0,
            DRF_DEF(_PBUS, _INTR_0, _PRI_TIMEOUT, _RESET));

        save0 = LWSWITCH_REG_RD32(device, _PBUS, _PRI_TIMEOUT_SAVE_0);
        save1 = LWSWITCH_REG_RD32(device, _PBUS, _PRI_TIMEOUT_SAVE_1);
        save3 = LWSWITCH_REG_RD32(device, _PBUS, _PRI_TIMEOUT_SAVE_3);
        errCode = LWSWITCH_REG_RD32(device, _PBUS, _PRI_TIMEOUT_FECS_ERRCODE);

        pri_timeout.addr    = DRF_VAL(_PBUS, _PRI_TIMEOUT_SAVE_0, _ADDR, save0) * 4;
        pri_timeout.data    = DRF_VAL(_PBUS, _PRI_TIMEOUT_SAVE_1, _DATA, save1);
        pri_timeout.write   = DRF_VAL(_PBUS, _PRI_TIMEOUT_SAVE_0, _WRITE, save0);
        pri_timeout.dest    = DRF_VAL(_PBUS, _PRI_TIMEOUT_SAVE_0, _TO, save0);
        pri_timeout.subId   = DRF_VAL(_PBUS, _PRI_TIMEOUT_SAVE_3, _SUBID, save3);
        pri_timeout.errCode = DRF_VAL(_PBUS, _PRI_TIMEOUT_FECS_ERRCODE, _DATA, errCode);

        // Dump register values as well
        pri_timeout.raw_data[0] = save0;
        pri_timeout.raw_data[1] = save1;
        pri_timeout.raw_data[2] = save3;
        pri_timeout.raw_data[3] = errCode;

        LWSWITCH_PRINT(device, ERROR,
                    "PBUS PRI error: %s offset: 0x%x data: 0x%x to: %d, "
                    "subId: 0x%x, FECS errCode: 0x%x\n",
                    pri_timeout.write ? "write" : "read",
                    pri_timeout.addr,
                    pri_timeout.data,
                    pri_timeout.dest,
                    pri_timeout.subId,
                    pri_timeout.errCode);

        if (FLD_TEST_DRF(_PBUS, _INTR_0, _PRI_SQUASH, _PENDING, bit))
        {
            LWSWITCH_REPORT_PRI_ERROR_NONFATAL(_HW_HOST_PRIV_TIMEOUT, "PBUS PRI SQUASH error", LWSWITCH_PBUS_PRI_SQUASH, 0, pri_timeout);
            LWSWITCH_PRINT(device, ERROR, "PRI_SQUASH: "
                "PBUS PRI error due to pri access while target block is in reset\n");
        }

        if (FLD_TEST_DRF(_PBUS, _INTR_0, _PRI_FECSERR, _PENDING, bit))
        {
            LWSWITCH_REPORT_PRI_ERROR_NONFATAL(_HW_HOST_PRIV_TIMEOUT, "PBUS PRI FECSERR error", LWSWITCH_PBUS_PRI_FECSERR, 0, pri_timeout);
            LWSWITCH_PRINT(device, ERROR, "PRI_FECSERR: "
                "FECS detected the error while processing a PRI request\n");
        }

        if (FLD_TEST_DRF(_PBUS, _INTR_0, _PRI_TIMEOUT, _PENDING, bit))
        {
            LWSWITCH_REPORT_PRI_ERROR_NONFATAL(_HW_HOST_PRIV_TIMEOUT, "PBUS PRI TIMEOUT error", LWSWITCH_PBUS_PRI_TIMEOUT, 0, pri_timeout);
            LWSWITCH_PRINT(device, ERROR, "PRI_TIMEOUT: "
                "PBUS PRI error due non-existent host register or timeout waiting for FECS\n");
        }

        // allow next error to latch
        LWSWITCH_REG_WR32(device, _PBUS, _PRI_TIMEOUT_SAVE_0,
            FLD_SET_DRF(_PBUS, _PRI_TIMEOUT_SAVE_0, _TO, _CLEAR, save0));

        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_DEF(_PBUS, _INTR_0, _SW, _PENDING);
    if (lwswitch_test_flags(pending, bit))
    {
        // Useful for debugging SW interrupts
        LWSWITCH_PRINT(device, INFO, "SW intr\n");
        lwswitch_clear_flags(&unhandled, bit);
    }

    LWSWITCH_REG_WR32(device, _PBUS, _INTR_0, pending);  // W1C with _RESET

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_priv_ring_lr10
(
    lwswitch_device *device
)
{
    LwU32 pending, i;
    LWSWITCH_PRI_ERROR_LOG_TYPE pri_error;

    pending = LWSWITCH_REG_RD32(device, _PPRIV_MASTER, _RING_INTERRUPT_STATUS0);

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    if (FLD_TEST_DRF_NUM(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0,
            _GBL_WRITE_ERROR_SYS, 1, pending))
    {
        pri_error.addr = LWSWITCH_REG_RD32(device, _PPRIV_SYS, _PRIV_ERROR_ADR);
        pri_error.data = LWSWITCH_REG_RD32(device, _PPRIV_SYS, _PRIV_ERROR_WRDAT);
        pri_error.info = LWSWITCH_REG_RD32(device, _PPRIV_SYS, _PRIV_ERROR_INFO);
        pri_error.code = LWSWITCH_REG_RD32(device, _PPRIV_SYS, _PRIV_ERROR_CODE);

        LWSWITCH_REPORT_PRI_ERROR_NONFATAL(_HW_HOST_PRIV_ERROR, "PRI WRITE SYS error", LWSWITCH_PPRIV_WRITE_SYS, 0, pri_error);

        LWSWITCH_PRINT(device, ERROR,
            "SYS PRI write error addr: 0x%08x data: 0x%08x info: 0x%08x code: 0x%08x\n",
            pri_error.addr, pri_error.data,
            pri_error.info, pri_error.code);

        pending = FLD_SET_DRF_NUM(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0,
            _GBL_WRITE_ERROR_SYS, 0, pending);
    }

    for (i = 0; i < LWSWITCH_NUM_PRIV_PRT_LR10; i++)
    {
        if (DRF_VAL(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0,
            _GBL_WRITE_ERROR_FBP, pending) & LWBIT(i))
        {
            pri_error.addr = LWSWITCH_REG_RD32(device, _PPRIV_PRT_PRT, _PRIV_ERROR_ADR(i));
            pri_error.data = LWSWITCH_REG_RD32(device, _PPRIV_PRT_PRT, _PRIV_ERROR_WRDAT(i));
            pri_error.info = LWSWITCH_REG_RD32(device, _PPRIV_PRT_PRT, _PRIV_ERROR_INFO(i));
            pri_error.code = LWSWITCH_REG_RD32(device, _PPRIV_PRT_PRT, _PRIV_ERROR_CODE(i));

            LWSWITCH_REPORT_PRI_ERROR_NONFATAL(_HW_HOST_PRIV_ERROR, "PRI WRITE PRT error", LWSWITCH_PPRIV_WRITE_PRT, i, pri_error);

            LWSWITCH_PRINT(device, ERROR,
                "PRT%d PRI write error addr: 0x%08x data: 0x%08x info: 0x%08x code: 0x%08x\n",
                i, pri_error.addr, pri_error.data, pri_error.info, pri_error.code);

            pending &= ~DRF_NUM(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0,
                _GBL_WRITE_ERROR_FBP, LWBIT(i));
        }
    }

    if (pending != 0)
    {
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_HOST_PRIV_ERROR, 
            "Fatal, Unexpected PRI error\n");
        LWSWITCH_LOG_FATAL_DATA(device, _HW, _HW_HOST_PRIV_ERROR, 2, 0, LW_FALSE, &pending);

        LWSWITCH_PRINT(device, ERROR,
            "Unexpected PRI error 0x%08x\n", pending);
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    // TODO reset the priv ring like GPU driver?

    // acknowledge the interrupt to the ringmaster
    lwswitch_ring_master_cmd_lr10(device,
        DRF_DEF(_PPRIV_MASTER, _RING_COMMAND, _CMD, _ACK_INTERRUPT));

    return LWL_SUCCESS;
}

static void
_lwswitch_save_route_err_header_lr10
(
    lwswitch_device    *device,
    LwU32               link,
    LWSWITCH_RAW_ERROR_LOG_TYPE *data
)
{
    LwU32 val;
    LwU32 i = 0;

    data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _ROUTE, _ERR_TIMESTAMP_LOG);
    val = LWSWITCH_NPORT_RD32_LR10(device, link, _ROUTE, _ERR_HEADER_LOG_VALID);

    if (FLD_TEST_DRF_NUM(_ROUTE, _ERR_HEADER_LOG_VALID, _HEADERVALID0, 1, val))
    {
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _ROUTE, _ERR_MISC_LOG_0);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _ROUTE, _ERR_HEADER_LOG_0);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _ROUTE, _ERR_HEADER_LOG_1);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _ROUTE, _ERR_HEADER_LOG_2);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _ROUTE, _ERR_HEADER_LOG_3);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _ROUTE, _ERR_HEADER_LOG_4);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _ROUTE, _ERR_HEADER_LOG_5);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _ROUTE, _ERR_HEADER_LOG_6);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _ROUTE, _ERR_HEADER_LOG_7);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _ROUTE, _ERR_HEADER_LOG_8);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _ROUTE, _ERR_HEADER_LOG_9);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _ROUTE, _ERR_HEADER_LOG_10);
    }
}

static LwlStatus
_lwswitch_service_route_fatal_lr10
(
    lwswitch_device *device,
    LwU32            link
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, contain, unhandled;
    LWSWITCH_RAW_ERROR_LOG_TYPE data = {{ 0 }};
    INFOROM_LWS_ECC_ERROR_EVENT err_event = {0};

    report.raw_pending = LWSWITCH_NPORT_RD32_LR10(device, link, _ROUTE, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_NPORT_RD32_LR10(device, link, _ROUTE, _ERR_FATAL_REPORT_EN_0);
    report.mask = report.raw_enable & chip_device->intr_mask.route.fatal;
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;

    report.raw_first = LWSWITCH_NPORT_RD32_LR10(device, link, _ROUTE, _ERR_FIRST_0);
    contain = LWSWITCH_NPORT_RD32_LR10(device, link, _ROUTE, _ERR_CONTAIN_EN_0);
    _lwswitch_save_route_err_header_lr10(device, link, &data);

    bit = DRF_NUM(_ROUTE, _ERR_STATUS_0, _ROUTEBUFERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_ROUTE_ROUTEBUFERR, "route buffer over/underflow", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_ROUTE_ROUTEBUFERR, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_ROUTE, _ERR_STATUS_0, _GLT_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LwBool bAddressValid = LW_FALSE;
        LwU32 address = 0;
        LwU32 addressValid = LWSWITCH_NPORT_RD32_LR10(device, link, _ROUTE,
                _ERR_GLT_ECC_ERROR_ADDRESS_VALID);

        if (FLD_TEST_DRF(_ROUTE_ERR_GLT, _ECC_ERROR_ADDRESS_VALID, _VALID, _VALID,
                         addressValid))
        {
            address = LWSWITCH_NPORT_RD32_LR10(device, link, _ROUTE,
                                               _ERR_GLT_ECC_ERROR_ADDRESS);
            bAddressValid = LW_TRUE;
        }

        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_ROUTE_GLT_ECC_DBE_ERR, "route GLT DBE", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_ROUTE_GLT_ECC_DBE_ERR, data);
        lwswitch_clear_flags(&unhandled, bit);

        _lwswitch_construct_ecc_error_event(&err_event,
            LWSWITCH_ERR_HW_NPORT_ROUTE_GLT_ECC_DBE_ERR, link, bAddressValid,
            address, LW_TRUE, 1);

        lwswitch_inforom_ecc_log_err_event(device, &err_event);
    }

    bit = DRF_NUM(_ROUTE, _ERR_STATUS_0, _TRANSDONERESVERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_ROUTE_TRANSDONERESVERR, "route transdone over/underflow", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_ROUTE_TRANSDONERESVERR, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_ROUTE, _ERR_STATUS_0, _PDCTRLPARERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_ROUTE_PDCTRLPARERR, "route parity", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_ROUTE_PDCTRLPARERR, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_ROUTE, _ERR_STATUS_0, _LWS_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_ROUTE_LWS_ECC_DBE_ERR, "route incoming DBE", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_ROUTE_LWS_ECC_DBE_ERR, data);
        lwswitch_clear_flags(&unhandled, bit);

        _lwswitch_construct_ecc_error_event(&err_event,
            LWSWITCH_ERR_HW_NPORT_ROUTE_LWS_ECC_DBE_ERR, link, LW_FALSE, 0,
            LW_TRUE, 1);

        lwswitch_inforom_ecc_log_err_event(device, &err_event);

        // Clear associated LIMIT_ERR interrupt
        if (report.raw_pending & DRF_NUM(_ROUTE, _ERR_STATUS_0, _LWS_ECC_LIMIT_ERR, 1))
        {
            LWSWITCH_NPORT_WR32_LR10(device, link, _ROUTE, _ERR_STATUS_0,
                DRF_NUM(_ROUTE, _ERR_STATUS_0, _LWS_ECC_LIMIT_ERR, 1));
        }
    }

    bit = DRF_NUM(_ROUTE, _ERR_STATUS_0, _CDTPARERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_ROUTE_CDTPARERR, "route credit parity", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_ROUTE_CDTPARERR, data);
        lwswitch_clear_flags(&unhandled, bit);

        _lwswitch_construct_ecc_error_event(&err_event,
            LWSWITCH_ERR_HW_NPORT_ROUTE_CDTPARERR, link, LW_FALSE, 0,
            LW_TRUE, 1);

        lwswitch_inforom_ecc_log_err_event(device, &err_event);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    // This helps prevent an interrupt storm if HW keeps triggering unnecessary stream of interrupts.
    if (device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_NPORT_WR32_LR10(device, link, _ROUTE, _ERR_FATAL_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    if (report.raw_first & report.mask)
    {
        LWSWITCH_NPORT_WR32_LR10(device, link, _ROUTE, _ERR_FIRST_0,
                report.raw_first & report.mask);
    }
    LWSWITCH_NPORT_WR32_LR10(device, link, _ROUTE, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_route_nonfatal_lr10
(
    lwswitch_device *device,
    LwU32            link
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, unhandled;
    LWSWITCH_RAW_ERROR_LOG_TYPE data = {{ 0 }};
    INFOROM_LWS_ECC_ERROR_EVENT err_event = {0};

    report.raw_pending = LWSWITCH_NPORT_RD32_LR10(device, link, _ROUTE, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_NPORT_RD32_LR10(device, link, _ROUTE, _ERR_NON_FATAL_REPORT_EN_0);
    report.mask = report.raw_enable & chip_device->intr_mask.route.nonfatal;
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;
    report.raw_first = LWSWITCH_NPORT_RD32_LR10(device, link, _ROUTE, _ERR_FIRST_0);
    _lwswitch_save_route_err_header_lr10(device, link, &data);

    bit = DRF_NUM(_ROUTE, _ERR_STATUS_0, _NOPORTDEFINEDERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_NPORT_ROUTE_NOPORTDEFINEDERR, "route undefined route");
        LWSWITCH_REPORT_DATA(_HW_NPORT_ROUTE_NOPORTDEFINEDERR, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_ROUTE, _ERR_STATUS_0, _ILWALIDROUTEPOLICYERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_NPORT_ROUTE_ILWALIDROUTEPOLICYERR, "route invalid policy");
        LWSWITCH_REPORT_DATA(_HW_NPORT_ROUTE_ILWALIDROUTEPOLICYERR, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_ROUTE, _ERR_STATUS_0, _LWS_ECC_LIMIT_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        // Ignore LIMIT error if DBE is pending
        if (!(lwswitch_test_flags(report.raw_pending,
                DRF_NUM(_ROUTE, _ERR_STATUS_0, _LWS_ECC_DBE_ERR, 1))))
        {
            report.data[0] = LWSWITCH_NPORT_RD32_LR10(device, link, _ROUTE, _ERR_LWS_ECC_ERROR_COUNTER);
            LWSWITCH_REPORT_NONFATAL(_HW_NPORT_ROUTE_LWS_ECC_LIMIT_ERR, "route incoming ECC limit");
            LWSWITCH_REPORT_DATA(_HW_NPORT_ROUTE_LWS_ECC_LIMIT_ERR, data);

            _lwswitch_construct_ecc_error_event(&err_event,
                LWSWITCH_ERR_HW_NPORT_ROUTE_LWS_ECC_LIMIT_ERR, link, LW_FALSE, 0,
                LW_FALSE, 1);

            lwswitch_inforom_ecc_log_err_event(device, &err_event);
        }

        lwswitch_clear_flags(&unhandled, bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    // This helps prevent an interrupt storm if HW keeps triggering unnecessary stream of interrupts.
    if (device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_NPORT_WR32_LR10(device, link, _ROUTE, _ERR_NON_FATAL_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    if (report.raw_first & report.mask)
    {
        LWSWITCH_NPORT_WR32_LR10(device, link, _ROUTE, _ERR_FIRST_0,
                report.raw_first & report.mask);
    }

    LWSWITCH_NPORT_WR32_LR10(device, link, _ROUTE, _ERR_STATUS_0, pending);

    //
    // Note, when traffic is flowing, if we reset ERR_COUNT before ERR_STATUS
    // register, we won't see an interrupt again until counter wraps around.
    // In that case, we will miss writing back many ECC victim entries. Hence,
    // always clear _ERR_COUNT only after _ERR_STATUS register is cleared!
    //
    LWSWITCH_NPORT_WR32_LR10(device, link, _ROUTE, _ERR_LWS_ECC_ERROR_COUNTER, 0x0);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

//
// Ingress
//

static void
_lwswitch_save_ingress_err_header_lr10
(
    lwswitch_device    *device,
    LwU32               link,
    LWSWITCH_RAW_ERROR_LOG_TYPE *data
)
{
    LwU32 val;
    LwU32 i = 0;

    data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_TIMESTAMP_LOG);

    val = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_HEADER_LOG_VALID);
    if (FLD_TEST_DRF_NUM(_INGRESS, _ERR_HEADER_LOG_VALID, _HEADERVALID0, 1, val))
    {
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_MISC_LOG_0);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_HEADER_LOG_0);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_HEADER_LOG_1);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_HEADER_LOG_2);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_HEADER_LOG_3);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_HEADER_LOG_4);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_HEADER_LOG_5);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_HEADER_LOG_6);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_HEADER_LOG_7);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_HEADER_LOG_8);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_HEADER_LOG_9);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_HEADER_LOG_10);
    }
    else
    {
        data->data[i++] = 0xdeadbeef;
    }
}

static LwlStatus
_lwswitch_service_ingress_fatal_lr10
(
    lwswitch_device *device,
    LwU32            link
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, contain, unhandled;
    LWSWITCH_RAW_ERROR_LOG_TYPE data = {{ 0 }};
    INFOROM_LWS_ECC_ERROR_EVENT err_event = {0};

    report.raw_pending = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_FATAL_REPORT_EN_0);
    report.mask = report.raw_enable & chip_device->intr_mask.ingress.fatal;
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;
    report.raw_first = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_FIRST_0);
    contain = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_CONTAIN_EN_0);
    _lwswitch_save_ingress_err_header_lr10(device, link, &data);

    bit = DRF_NUM(_INGRESS, _ERR_STATUS_0, _CMDDECODEERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_INGRESS_CMDDECODEERR, "ingress invalid command", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_INGRESS_CMDDECODEERR, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_INGRESS, _ERR_STATUS_0, _NCISOC_HDR_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        report.data[0] = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_NCISOC_HDR_ECC_ERROR_COUNTER);
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_INGRESS_NCISOC_HDR_ECC_DBE_ERR, "ingress header DBE", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_INGRESS_NCISOC_HDR_ECC_DBE_ERR, data);
        lwswitch_clear_flags(&unhandled, bit);

        _lwswitch_construct_ecc_error_event(&err_event,
            LWSWITCH_ERR_HW_NPORT_INGRESS_NCISOC_HDR_ECC_DBE_ERR, link, LW_FALSE, 0,
            LW_TRUE, 1);

        lwswitch_inforom_ecc_log_err_event(device, &err_event);

        // Clear associated LIMIT_ERR interrupt
        if (report.raw_pending & DRF_NUM(_INGRESS, _ERR_STATUS_0, _NCISOC_HDR_ECC_LIMIT_ERR, 1))
        {
            LWSWITCH_NPORT_WR32_LR10(device, link, _INGRESS, _ERR_STATUS_0,
                DRF_NUM(_INGRESS, _ERR_STATUS_0, _NCISOC_HDR_ECC_LIMIT_ERR, 1));
        }
    }

    bit = DRF_NUM(_INGRESS, _ERR_STATUS_0, _ILWALIDVCSET, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_INGRESS_ILWALIDVCSET, "ingress invalid VCSet", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_INGRESS_ILWALIDVCSET, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_INGRESS, _ERR_STATUS_0, _REMAPTAB_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LwBool bAddressValid = LW_FALSE;
        LwU32 address = 0;
        LwU32 addressValid = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS,
                _ERR_REMAPTAB_ECC_ERROR_ADDRESS);

        if (FLD_TEST_DRF(_INGRESS_ERR_REMAPTAB, _ECC_ERROR_ADDRESS_VALID, _VALID, _VALID,
                         addressValid))
        {
            address = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS,
                                               _ERR_REMAPTAB_ECC_ERROR_ADDRESS);
            bAddressValid = LW_TRUE;
        }

        report.data[0] = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_REMAPTAB_ECC_ERROR_COUNTER);
        report.data[1] = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_REMAPTAB_ECC_ERROR_ADDRESS);
        report.data[2] = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_REMAPTAB_ECC_ERROR_ADDRESS_VALID);
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_INGRESS_REMAPTAB_ECC_DBE_ERR, "ingress Remap DBE", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_INGRESS_REMAPTAB_ECC_DBE_ERR, data);
        lwswitch_clear_flags(&unhandled, bit);

        _lwswitch_construct_ecc_error_event(&err_event,
            LWSWITCH_ERR_HW_NPORT_INGRESS_REMAPTAB_ECC_DBE_ERR, link, bAddressValid,
            address, LW_TRUE, 1);

        lwswitch_inforom_ecc_log_err_event(device, &err_event);
    }

    bit = DRF_NUM(_INGRESS, _ERR_STATUS_0, _RIDTAB_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LwBool bAddressValid = LW_FALSE;
        LwU32 address = 0;
        LwU32 addressValid = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS,
                _ERR_RIDTAB_ECC_ERROR_ADDRESS_VALID);

        if (FLD_TEST_DRF(_INGRESS_ERR_RIDTAB, _ECC_ERROR_ADDRESS_VALID, _VALID, _VALID,
                         addressValid))
        {
            address = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS,
                                               _ERR_RIDTAB_ECC_ERROR_ADDRESS);
            bAddressValid = LW_TRUE;
        }

        report.data[0] = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_RIDTAB_ECC_ERROR_COUNTER);
        report.data[1] = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_RIDTAB_ECC_ERROR_ADDRESS);
        report.data[2] = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_RIDTAB_ECC_ERROR_ADDRESS_VALID);
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_INGRESS_RIDTAB_ECC_DBE_ERR, "ingress RID DBE", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_INGRESS_RIDTAB_ECC_DBE_ERR, data);
        lwswitch_clear_flags(&unhandled, bit);

        _lwswitch_construct_ecc_error_event(&err_event,
            LWSWITCH_ERR_HW_NPORT_INGRESS_RIDTAB_ECC_DBE_ERR, link, bAddressValid,
            address, LW_TRUE, 1);

        lwswitch_inforom_ecc_log_err_event(device, &err_event);
    }

    bit = DRF_NUM(_INGRESS, _ERR_STATUS_0, _RLANTAB_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LwBool bAddressValid = LW_FALSE;
        LwU32 address = 0;
        LwU32 addressValid = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS,
                _ERR_RLANTAB_ECC_ERROR_ADDRESS_VALID);

        if (FLD_TEST_DRF(_INGRESS_ERR_RLANTAB, _ECC_ERROR_ADDRESS_VALID, _VALID, _VALID,
                         addressValid))
        {
            address = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS,
                                               _ERR_RLANTAB_ECC_ERROR_ADDRESS);
            bAddressValid = LW_TRUE;
        }

        report.data[0] = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_RLANTAB_ECC_ERROR_COUNTER);
        report.data[1] = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_RLANTAB_ECC_ERROR_ADDRESS);
        report.data[2] = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_RLANTAB_ECC_ERROR_ADDRESS_VALID);
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_INGRESS_RLANTAB_ECC_DBE_ERR, "ingress RLAN DBE", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_INGRESS_RLANTAB_ECC_DBE_ERR, data);
        lwswitch_clear_flags(&unhandled, bit);

        _lwswitch_construct_ecc_error_event(&err_event,
            LWSWITCH_ERR_HW_NPORT_INGRESS_RLANTAB_ECC_DBE_ERR, link, bAddressValid,
            address, LW_TRUE, 1);

        lwswitch_inforom_ecc_log_err_event(device, &err_event);
    }

    bit = DRF_NUM(_INGRESS, _ERR_STATUS_0, _NCISOC_PARITY_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_INGRESS_NCISOC_PARITY_ERR, "ingress control parity", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_INGRESS_NCISOC_PARITY_ERR, data);
        lwswitch_clear_flags(&unhandled, bit);

        _lwswitch_construct_ecc_error_event(&err_event,
            LWSWITCH_ERR_HW_NPORT_INGRESS_NCISOC_PARITY_ERR, link, LW_FALSE, 0,
            LW_TRUE, 1);

        lwswitch_inforom_ecc_log_err_event(device, &err_event);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    // This helps prevent an interrupt storm if HW keeps triggering unnecessary stream of interrupts.
    if (device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_NPORT_WR32_LR10(device, link, _INGRESS, _ERR_FATAL_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    if (report.raw_first & report.mask)
    {
        LWSWITCH_NPORT_WR32_LR10(device, link, _INGRESS, _ERR_FIRST_0,
            report.raw_first & report.mask);
    }

    LWSWITCH_NPORT_WR32_LR10(device, link, _INGRESS, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_ingress_nonfatal_lr10
(
    lwswitch_device *device,
    LwU32            link
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, unhandled;
    LWSWITCH_RAW_ERROR_LOG_TYPE data = {{ 0 }};
    INFOROM_LWS_ECC_ERROR_EVENT err_event = {0};

    report.raw_pending = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_NON_FATAL_REPORT_EN_0);
    report.mask = report.raw_enable & chip_device->intr_mask.ingress.nonfatal;
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;
    report.raw_first = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_FIRST_0);
    _lwswitch_save_ingress_err_header_lr10(device, link, &data);

    bit = DRF_NUM(_INGRESS, _ERR_STATUS_0, _REQCONTEXTMISMATCHERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_NPORT_INGRESS_REQCONTEXTMISMATCHERR, "ingress request context mismatch");
        LWSWITCH_REPORT_DATA(_HW_NPORT_INGRESS_REQCONTEXTMISMATCHERR, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_INGRESS, _ERR_STATUS_0, _ACLFAIL, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_NPORT_INGRESS_ACLFAIL, "ingress invalid ACL");
        LWSWITCH_REPORT_DATA(_HW_NPORT_INGRESS_ACLFAIL, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_INGRESS, _ERR_STATUS_0, _NCISOC_HDR_ECC_LIMIT_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        // Ignore LIMIT error if DBE is pending
        if (!(lwswitch_test_flags(report.raw_pending,
                DRF_NUM(_INGRESS, _ERR_STATUS_0, _NCISOC_HDR_ECC_DBE_ERR, 1))))
        {
            report.data[0] = LWSWITCH_NPORT_RD32_LR10(device, link, _INGRESS, _ERR_NCISOC_HDR_ECC_ERROR_COUNTER);
            LWSWITCH_REPORT_NONFATAL(_HW_NPORT_INGRESS_NCISOC_HDR_ECC_LIMIT_ERR, "ingress header ECC");
            LWSWITCH_REPORT_DATA(_HW_NPORT_INGRESS_NCISOC_HDR_ECC_LIMIT_ERR, data);

            _lwswitch_construct_ecc_error_event(&err_event,
                LWSWITCH_ERR_HW_NPORT_INGRESS_NCISOC_HDR_ECC_LIMIT_ERR, link, LW_FALSE, 0,
                LW_FALSE, 1);

            lwswitch_inforom_ecc_log_err_event(device, &err_event);
        }

        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_INGRESS, _ERR_STATUS_0, _ADDRBOUNDSERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_NPORT_INGRESS_ADDRBOUNDSERR, "ingress address bounds");
        LWSWITCH_REPORT_DATA(_HW_NPORT_INGRESS_ADDRBOUNDSERR, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_INGRESS, _ERR_STATUS_0, _RIDTABCFGERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_NPORT_INGRESS_RIDTABCFGERR, "ingress RID packet");
        LWSWITCH_REPORT_DATA(_HW_NPORT_INGRESS_RIDTABCFGERR, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_INGRESS, _ERR_STATUS_0, _RLANTABCFGERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_NPORT_INGRESS_RLANTABCFGERR, "ingress RLAN packet");
        LWSWITCH_REPORT_DATA(_HW_NPORT_INGRESS_RLANTABCFGERR, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_INGRESS, _ERR_STATUS_0, _ADDRTYPEERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_NPORT_INGRESS_ADDRTYPEERR, "ingress illegal address");
        LWSWITCH_REPORT_DATA(_HW_NPORT_INGRESS_ADDRTYPEERR, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    // This helps prevent an interrupt storm if HW keeps triggering unnecessary stream of interrupts.
    if (device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_NPORT_WR32_LR10(device, link, _INGRESS, _ERR_NON_FATAL_REPORT_EN_0,
            report.raw_enable ^ pending);
    }

    if (report.raw_first & report.mask)
    {
        LWSWITCH_NPORT_WR32_LR10(device, link, _INGRESS, _ERR_FIRST_0,
            report.raw_first & report.mask);
    }
    
    LWSWITCH_NPORT_WR32_LR10(device, link, _INGRESS, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

//
// Egress
//

static void
_lwswitch_save_egress_err_header_lr10
(
    lwswitch_device    *device,
    LwU32               link,
    LWSWITCH_RAW_ERROR_LOG_TYPE *data
)
{
    LwU32 val;
    LwU32 i = 0;

    data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _ERR_TIMESTAMP_LOG);

    val = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _ERR_HEADER_LOG_VALID);
    if (FLD_TEST_DRF_NUM(_EGRESS, _ERR_HEADER_LOG_VALID, _HEADERVALID0, 1, val))
    {
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _ERR_MISC_LOG_0);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _ERR_HEADER_LOG_0);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _ERR_HEADER_LOG_1);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _ERR_HEADER_LOG_2);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _ERR_HEADER_LOG_3);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _ERR_HEADER_LOG_4);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _ERR_HEADER_LOG_5);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _ERR_HEADER_LOG_6);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _ERR_HEADER_LOG_7);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _ERR_HEADER_LOG_8);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _ERR_HEADER_LOG_9);
        data->data[i++] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _ERR_HEADER_LOG_10);
    }
    else
    {
        data->data[i++] = 0xdeadbeef;
    }
}

static LwlStatus
_lwswitch_service_tstate_nonfatal_lr10
(
    lwswitch_device *device,
    LwU32            link
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, unhandled;
    LWSWITCH_RAW_ERROR_LOG_TYPE data = {{ 0 }};
    INFOROM_LWS_ECC_ERROR_EVENT err_event = {0};

    report.raw_pending = LWSWITCH_NPORT_RD32_LR10(device, link, _TSTATE, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_NPORT_RD32_LR10(device, link, _TSTATE, _ERR_NON_FATAL_REPORT_EN_0);
    report.mask = report.raw_enable & chip_device->intr_mask.tstate.nonfatal;
    report.data[0] = LWSWITCH_NPORT_RD32_LR10(device, link, _TSTATE, _ERR_MISC_LOG_0);
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;
    report.raw_first = LWSWITCH_NPORT_RD32_LR10(device, link, _TSTATE, _ERR_FIRST_0);

    bit = DRF_NUM(_TSTATE, _ERR_STATUS_0, _TAGPOOL_ECC_LIMIT_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        // Ignore LIMIT error if DBE is pending
        if(!(lwswitch_test_flags(report.raw_pending,
                DRF_NUM(_TSTATE, _ERR_STATUS_0, _TAGPOOL_ECC_DBE_ERR, 1))))
        {
            LwBool bAddressValid = LW_FALSE;
            LwU32 address = 0;
            LwU32 addressValid = LWSWITCH_NPORT_RD32_LR10(device, link, _TSTATE,
                    _ERR_TAGPOOL_ECC_ERROR_ADDRESS_VALID);

            if (FLD_TEST_DRF(_TSTATE_ERR_TAGPOOL, _ECC_ERROR_ADDRESS_VALID, _VALID, _VALID,
                             addressValid))
            {
                address = LWSWITCH_NPORT_RD32_LR10(device, link, _TSTATE,
                                                   _ERR_TAGPOOL_ECC_ERROR_ADDRESS);
                bAddressValid = LW_TRUE;
            }

            report.data[1] = LWSWITCH_NPORT_RD32_LR10(device, link, _TSTATE, _ERR_TAGPOOL_ECC_ERROR_COUNTER);
            LWSWITCH_NPORT_WR32_LR10(device, link, _TSTATE, _ERR_TAGPOOL_ECC_ERROR_COUNTER,
                DRF_DEF(_TSTATE, _ERR_TAGPOOL_ECC_ERROR_COUNTER, _ERROR_COUNT, _INIT));
            LWSWITCH_REPORT_NONFATAL(_HW_NPORT_TSTATE_TAGPOOL_ECC_LIMIT_ERR, "TS tag store single-bit threshold");
            _lwswitch_save_egress_err_header_lr10(device, link, &data);
            LWSWITCH_REPORT_DATA(_HW_NPORT_TSTATE_TAGPOOL_ECC_LIMIT_ERR, data);

            _lwswitch_construct_ecc_error_event(&err_event,
                LWSWITCH_ERR_HW_NPORT_TSTATE_TAGPOOL_ECC_LIMIT_ERR, link,
                bAddressValid, address, LW_FALSE, 1);

            lwswitch_inforom_ecc_log_err_event(device, &err_event);
        }

        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_TSTATE, _ERR_STATUS_0, _CRUMBSTORE_ECC_LIMIT_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        // Ignore LIMIT error if DBE is pending
        if(!(lwswitch_test_flags(report.raw_pending,
                DRF_NUM(_TSTATE, _ERR_STATUS_0, _CRUMBSTORE_ECC_DBE_ERR, 1))))
        {
            LwBool bAddressValid = LW_FALSE;
            LwU32 address = 0;
            LwU32 addressValid = LWSWITCH_NPORT_RD32_LR10(device, link, _TSTATE,
                    _ERR_CRUMBSTORE_ECC_ERROR_ADDRESS_VALID);

            if (FLD_TEST_DRF(_TSTATE_ERR_CRUMBSTORE, _ECC_ERROR_ADDRESS_VALID, _VALID, _VALID,
                             addressValid))
            {
                address = LWSWITCH_NPORT_RD32_LR10(device, link, _TSTATE,
                                                   _ERR_CRUMBSTORE_ECC_ERROR_ADDRESS);
                bAddressValid = LW_TRUE;
            }

            report.data[1] = LWSWITCH_NPORT_RD32_LR10(device, link, _TSTATE, _ERR_CRUMBSTORE_ECC_ERROR_COUNTER);
            LWSWITCH_NPORT_WR32_LR10(device, link, _TSTATE, _ERR_CRUMBSTORE_ECC_ERROR_COUNTER,
                DRF_DEF(_TSTATE, _ERR_CRUMBSTORE_ECC_ERROR_COUNTER, _ERROR_COUNT, _INIT));
            LWSWITCH_REPORT_NONFATAL(_HW_NPORT_TSTATE_CRUMBSTORE_ECC_LIMIT_ERR, "TS crumbstore single-bit threshold");
            _lwswitch_save_ingress_err_header_lr10(device, link, &data);
            LWSWITCH_REPORT_DATA(_HW_NPORT_TSTATE_CRUMBSTORE_ECC_LIMIT_ERR, data);

            _lwswitch_construct_ecc_error_event(&err_event,
                LWSWITCH_ERR_HW_NPORT_TSTATE_CRUMBSTORE_ECC_LIMIT_ERR, link,
                bAddressValid, address, LW_FALSE, 1);

            lwswitch_inforom_ecc_log_err_event(device, &err_event);
        }

        lwswitch_clear_flags(&unhandled, bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    // This helps prevent an interrupt storm if HW keeps triggering unnecessary stream of interrupts.
    if (device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_NPORT_WR32_LR10(device, link, _TSTATE, _ERR_NON_FATAL_REPORT_EN_0,
            report.raw_enable ^ pending);
    }

    if (report.raw_first & report.mask)
    {
        LWSWITCH_NPORT_WR32_LR10(device, link, _TSTATE, _ERR_FIRST_0,
            report.raw_first & report.mask);
    }

    LWSWITCH_NPORT_WR32_LR10(device, link, _TSTATE, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_tstate_fatal_lr10
(
    lwswitch_device *device,
    LwU32            link
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, contain, unhandled;
    LWSWITCH_RAW_ERROR_LOG_TYPE data = {{ 0 }};
    INFOROM_LWS_ECC_ERROR_EVENT err_event = {0};

    report.raw_pending = LWSWITCH_NPORT_RD32_LR10(device, link, _TSTATE, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_NPORT_RD32_LR10(device, link, _TSTATE, _ERR_FATAL_REPORT_EN_0);
    report.mask = report.raw_enable & chip_device->intr_mask.tstate.fatal;
    report.data[0] = LWSWITCH_NPORT_RD32_LR10(device, link, _TSTATE, _ERR_MISC_LOG_0);
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;
    report.raw_first = LWSWITCH_NPORT_RD32_LR10(device, link, _TSTATE, _ERR_FIRST_0);
    contain = LWSWITCH_NPORT_RD32_LR10(device, link, _TSTATE, _ERR_CONTAIN_EN_0);

    bit = DRF_NUM(_TSTATE, _ERR_STATUS_0, _TAGPOOLBUFERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_TSTATE_TAGPOOLBUFERR, "TS pointer crossover", LW_FALSE);
        _lwswitch_save_egress_err_header_lr10(device, link, &data);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_TSTATE_TAGPOOLBUFERR, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_TSTATE, _ERR_STATUS_0, _TAGPOOL_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LwBool bAddressValid = LW_FALSE;
        LwU32 address = 0;
        LwU32 addressValid = LWSWITCH_NPORT_RD32_LR10(device, link, _TSTATE,
                _ERR_TAGPOOL_ECC_ERROR_ADDRESS_VALID);

        if (FLD_TEST_DRF(_TSTATE_ERR_TAGPOOL, _ECC_ERROR_ADDRESS_VALID, _VALID, _VALID,
                         addressValid))
        {
            address = LWSWITCH_NPORT_RD32_LR10(device, link, _TSTATE,
                                               _ERR_TAGPOOL_ECC_ERROR_ADDRESS);
            bAddressValid = LW_TRUE;
        }

        report.data[1] = LWSWITCH_NPORT_RD32_LR10(device, link, _TSTATE, _ERR_TAGPOOL_ECC_ERROR_COUNTER);
        LWSWITCH_NPORT_WR32_LR10(device, link, _TSTATE, _ERR_TAGPOOL_ECC_ERROR_COUNTER,
            DRF_DEF(_TSTATE, _ERR_TAGPOOL_ECC_ERROR_COUNTER, _ERROR_COUNT, _INIT));
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_TSTATE_TAGPOOL_ECC_DBE_ERR, "TS tag store fatal ECC", LW_FALSE);
        _lwswitch_save_egress_err_header_lr10(device, link, &data);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_TSTATE_TAGPOOL_ECC_DBE_ERR, data);
        lwswitch_clear_flags(&unhandled, bit);

        _lwswitch_construct_ecc_error_event(&err_event,
            LWSWITCH_ERR_HW_NPORT_TSTATE_TAGPOOL_ECC_DBE_ERR, link, bAddressValid,
            address, LW_TRUE, 1);

        lwswitch_inforom_ecc_log_err_event(device, &err_event);

        // Clear associated LIMIT_ERR interrupt
        if (report.raw_pending & DRF_NUM(_TSTATE, _ERR_STATUS_0, _TAGPOOL_ECC_LIMIT_ERR, 1))
        {
            LWSWITCH_NPORT_WR32_LR10(device, link, _TSTATE, _ERR_STATUS_0,
                DRF_NUM(_TSTATE, _ERR_STATUS_0, _TAGPOOL_ECC_LIMIT_ERR, 1));
        }
    }

    bit = DRF_NUM(_TSTATE, _ERR_STATUS_0, _CRUMBSTOREBUFERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_TSTATE_CRUMBSTOREBUFERR, "TS crumbstore", LW_FALSE);
        _lwswitch_save_egress_err_header_lr10(device, link, &data);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_TSTATE_CRUMBSTOREBUFERR, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_TSTATE, _ERR_STATUS_0, _CRUMBSTORE_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LwBool bAddressValid = LW_FALSE;
        LwU32 address = 0;
        LwU32 addressValid = LWSWITCH_NPORT_RD32_LR10(device, link, _TSTATE,
                _ERR_CRUMBSTORE_ECC_ERROR_ADDRESS_VALID);

        if (FLD_TEST_DRF(_TSTATE_ERR_CRUMBSTORE, _ECC_ERROR_ADDRESS_VALID, _VALID, _VALID,
                         addressValid))
        {
            address = LWSWITCH_NPORT_RD32_LR10(device, link, _TSTATE,
                                               _ERR_CRUMBSTORE_ECC_ERROR_ADDRESS);
            bAddressValid = LW_TRUE;
        }

        report.data[1] = LWSWITCH_NPORT_RD32_LR10(device, link, _TSTATE, _ERR_CRUMBSTORE_ECC_ERROR_COUNTER);
        LWSWITCH_NPORT_WR32_LR10(device, link, _TSTATE, _ERR_CRUMBSTORE_ECC_ERROR_COUNTER,
            DRF_DEF(_TSTATE, _ERR_CRUMBSTORE_ECC_ERROR_COUNTER, _ERROR_COUNT, _INIT));
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_TSTATE_CRUMBSTORE_ECC_DBE_ERR, "TS crumbstore fatal ECC", LW_FALSE);
        _lwswitch_save_ingress_err_header_lr10(device, link, &data);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_TSTATE_CRUMBSTORE_ECC_DBE_ERR, data);
        lwswitch_clear_flags(&unhandled, bit);

        _lwswitch_construct_ecc_error_event(&err_event,
            LWSWITCH_ERR_HW_NPORT_TSTATE_CRUMBSTORE_ECC_DBE_ERR, link, bAddressValid,
            address, LW_TRUE, 1);

        lwswitch_inforom_ecc_log_err_event(device, &err_event);

        // Clear associated LIMIT_ERR interrupt
        if (report.raw_pending & DRF_NUM(_TSTATE, _ERR_STATUS_0, _CRUMBSTORE_ECC_LIMIT_ERR, 1))
        {
            LWSWITCH_NPORT_WR32_LR10(device, link, _TSTATE, _ERR_STATUS_0,
                DRF_NUM(_TSTATE, _ERR_STATUS_0, _CRUMBSTORE_ECC_LIMIT_ERR, 1));
        }
    }

    bit = DRF_NUM(_TSTATE, _ERR_STATUS_0, _ATO_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        if (FLD_TEST_DRF_NUM(_TSTATE, _ERR_FIRST_0, _ATO_ERR, 1, report.raw_first))
        {
            report.data[1] = LWSWITCH_NPORT_RD32_LR10(device, link, _TSTATE, _ERR_DEBUG);
        }
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_TSTATE_ATO_ERR, "TS ATO timeout", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_TSTATE, _ERR_STATUS_0, _CAMRSP_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_TSTATE_CAMRSP_ERR, "Rsp Tag value out of range", LW_FALSE);
        _lwswitch_save_ingress_err_header_lr10(device, link, &data);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_TSTATE_CAMRSP_ERR, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    // This helps prevent an interrupt storm if HW keeps triggering unnecessary stream of interrupts.
    if (device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_NPORT_WR32_LR10(device, link, _TSTATE, _ERR_FATAL_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    if (report.raw_first & report.mask)
    {
        LWSWITCH_NPORT_WR32_LR10(device, link, _TSTATE, _ERR_FIRST_0,
            report.raw_first & report.mask);
    }

    LWSWITCH_NPORT_WR32_LR10(device, link, _TSTATE, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_egress_nonfatal_lr10
(
    lwswitch_device *device,
    LwU32            link
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, unhandled;
    LWSWITCH_RAW_ERROR_LOG_TYPE data = { { 0 } };
    INFOROM_LWS_ECC_ERROR_EVENT err_event = {0};

    report.raw_pending = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _ERR_NON_FATAL_REPORT_EN_0);
    report.mask = report.raw_enable & chip_device->intr_mask.egress.nonfatal;
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;
    report.raw_first = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _ERR_FIRST_0);
    _lwswitch_save_egress_err_header_lr10(device, link, &data);

    bit = DRF_NUM(_EGRESS, _ERR_STATUS_0, _NXBAR_HDR_ECC_LIMIT_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        // Ignore LIMIT error if DBE is pending
        if (!(lwswitch_test_flags(report.raw_pending,
                DRF_NUM(_EGRESS, _ERR_STATUS_0, _NXBAR_HDR_ECC_DBE_ERR, 1))))
        {
            report.data[0] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _ERR_NXBAR_ECC_ERROR_COUNTER);
            LWSWITCH_REPORT_NONFATAL(_HW_NPORT_EGRESS_NXBAR_HDR_ECC_LIMIT_ERR, "egress input ECC error limit");
            LWSWITCH_REPORT_DATA(_HW_NPORT_EGRESS_NXBAR_HDR_ECC_LIMIT_ERR, data);

            _lwswitch_construct_ecc_error_event(&err_event,
                LWSWITCH_ERR_HW_NPORT_EGRESS_NXBAR_HDR_ECC_LIMIT_ERR, link, LW_FALSE, 0,
                LW_FALSE, 1);

            lwswitch_inforom_ecc_log_err_event(device, &err_event);
        }

        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_EGRESS, _ERR_STATUS_0, _RAM_OUT_HDR_ECC_LIMIT_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        // Ignore LIMIT error if DBE is pending
        if(!(lwswitch_test_flags(report.raw_pending,
                DRF_NUM(_EGRESS, _ERR_STATUS_0, _RAM_OUT_HDR_ECC_DBE_ERR, 1))))
        {
            LwBool bAddressValid = LW_FALSE;
            LwU32 address = 0;
            LwU32 addressValid = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS,
                    _ERR_RAM_OUT_ECC_ERROR_ADDRESS_VALID);

            if (FLD_TEST_DRF(_EGRESS_ERR_RAM_OUT, _ECC_ERROR_ADDRESS_VALID, _VALID, _VALID,
                             addressValid))
            {
                address = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS,
                                                   _ERR_RAM_OUT_ECC_ERROR_ADDRESS);
                bAddressValid = LW_TRUE;
            }

            report.data[0] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _ERR_RAM_OUT_ECC_ERROR_COUNTER);
            report.data[1] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _ERR_RAM_OUT_ECC_ERROR_ADDRESS);
            LWSWITCH_REPORT_NONFATAL(_HW_NPORT_EGRESS_RAM_OUT_HDR_ECC_LIMIT_ERR, "egress output ECC error limit");
            LWSWITCH_REPORT_DATA(_HW_NPORT_EGRESS_RAM_OUT_HDR_ECC_LIMIT_ERR, data);

            _lwswitch_construct_ecc_error_event(&err_event,
                LWSWITCH_ERR_HW_NPORT_EGRESS_RAM_OUT_HDR_ECC_LIMIT_ERR, link, bAddressValid, address,
                LW_FALSE, 1);

            lwswitch_inforom_ecc_log_err_event(device, &err_event);
        }

        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_EGRESS, _ERR_STATUS_0, _URRSPERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_NPORT_EGRESS_DROPNPURRSPERR, "egress non-posted UR");
        LWSWITCH_REPORT_DATA(_HW_NPORT_EGRESS_DROPNPURRSPERR, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_EGRESS, _ERR_STATUS_0, _PRIVRSPERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_NPORT_EGRESS_PRIVRSPERR, "egress non-posted PRIV error");
        LWSWITCH_REPORT_DATA(_HW_NPORT_EGRESS_PRIVRSPERR, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_EGRESS, _ERR_STATUS_0, _HWRSPERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_NPORT_EGRESS_HWRSPERR, "egress non-posted HW error");
        LWSWITCH_REPORT_DATA(_HW_NPORT_EGRESS_HWRSPERR, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    // This helps prevent an interrupt storm if HW keeps triggering unnecessary stream of interrupts.
    if (device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_NPORT_WR32_LR10(device, link, _EGRESS, _ERR_NON_FATAL_REPORT_EN_0,
            report.raw_enable ^ pending);
    }

    if (report.raw_first & report.mask)
    {
        LWSWITCH_NPORT_WR32_LR10(device, link, _EGRESS, _ERR_FIRST_0,
            report.raw_first & report.mask);
    }

    LWSWITCH_NPORT_WR32_LR10(device, link, _EGRESS, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_egress_fatal_lr10
(
    lwswitch_device *device,
    LwU32            link
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, contain, unhandled;
    LWSWITCH_RAW_ERROR_LOG_TYPE data = {{ 0 }};
    LWSWITCH_RAW_ERROR_LOG_TYPE credit_data = { { 0 } };
    LWSWITCH_RAW_ERROR_LOG_TYPE buffer_data = { { 0 } };
    INFOROM_LWS_ECC_ERROR_EVENT err_event = {0};

    report.raw_pending = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _ERR_FATAL_REPORT_EN_0);
    report.mask = report.raw_enable & chip_device->intr_mask.egress.fatal;
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;
    report.raw_first = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _ERR_FIRST_0);
    contain = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _ERR_CONTAIN_EN_0);
    _lwswitch_save_egress_err_header_lr10(device, link, &data);

    bit = DRF_NUM(_EGRESS, _ERR_STATUS_0, _EGRESSBUFERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_EGRESSBUFERR, "egress crossbar overflow", LW_TRUE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_EGRESSBUFERR, data);

        buffer_data.data[0] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _BUFFER_POINTERS0);
        buffer_data.data[1] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _BUFFER_POINTERS1);
        buffer_data.data[2] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _BUFFER_POINTERS2);
        buffer_data.data[3] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _BUFFER_POINTERS3);
        buffer_data.data[4] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _BUFFER_POINTERS4);
        buffer_data.data[5] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _BUFFER_POINTERS5);
        buffer_data.data[6] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _BUFFER_POINTERS6);
        buffer_data.data[7] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _BUFFER_POINTERS7);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_EGRESSBUFERR, buffer_data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_EGRESS, _ERR_STATUS_0, _PKTROUTEERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_PKTROUTEERR, "egress packet route", LW_TRUE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_PKTROUTEERR, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_EGRESS, _ERR_STATUS_0, _SEQIDERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_SEQIDERR, "egress sequence ID error", LW_TRUE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_SEQIDERR, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_EGRESS, _ERR_STATUS_0, _NXBAR_HDR_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_NXBAR_HDR_ECC_DBE_ERR, "egress input ECC DBE error", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_NXBAR_HDR_ECC_DBE_ERR, data);
        lwswitch_clear_flags(&unhandled, bit);

        _lwswitch_construct_ecc_error_event(&err_event,
            LWSWITCH_ERR_HW_NPORT_EGRESS_NXBAR_HDR_ECC_DBE_ERR, link, LW_FALSE, 0,
            LW_TRUE, 1);

        lwswitch_inforom_ecc_log_err_event(device, &err_event);

        // Clear associated LIMIT_ERR interrupt
        if (report.raw_pending & DRF_NUM(_EGRESS, _ERR_STATUS_0, _NXBAR_HDR_ECC_LIMIT_ERR, 1))
        {
            LWSWITCH_NPORT_WR32_LR10(device, link, _EGRESS, _ERR_STATUS_0,
                DRF_NUM(_EGRESS, _ERR_STATUS_0, _NXBAR_HDR_ECC_LIMIT_ERR, 1));
        }
    }

    bit = DRF_NUM(_EGRESS, _ERR_STATUS_0, _RAM_OUT_HDR_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LwBool bAddressValid = LW_FALSE;
        LwU32 address = 0;
        LwU32 addressValid = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS,
                _ERR_RAM_OUT_ECC_ERROR_ADDRESS_VALID);

        if (FLD_TEST_DRF(_EGRESS_ERR_RAM_OUT, _ECC_ERROR_ADDRESS_VALID, _VALID, _VALID,
                         addressValid))
        {
            address = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS,
                                               _ERR_RAM_OUT_ECC_ERROR_ADDRESS);
            bAddressValid = LW_TRUE;
        }

        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_RAM_OUT_HDR_ECC_DBE_ERR, "egress output ECC DBE error", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_RAM_OUT_HDR_ECC_DBE_ERR, data);
        lwswitch_clear_flags(&unhandled, bit);

        _lwswitch_construct_ecc_error_event(&err_event,
            LWSWITCH_ERR_HW_NPORT_EGRESS_RAM_OUT_HDR_ECC_DBE_ERR, link, bAddressValid,
            address, LW_TRUE, 1);

        lwswitch_inforom_ecc_log_err_event(device, &err_event);

        // Clear associated LIMIT_ERR interrupt
        if (report.raw_pending & DRF_NUM(_EGRESS, _ERR_STATUS_0, _RAM_OUT_HDR_ECC_LIMIT_ERR, 1))
        {
            LWSWITCH_NPORT_WR32_LR10(device, link, _EGRESS, _ERR_STATUS_0,
                DRF_NUM(_EGRESS, _ERR_STATUS_0, _RAM_OUT_HDR_ECC_LIMIT_ERR, 1));
        }
    }

    bit = DRF_NUM(_EGRESS, _ERR_STATUS_0, _NCISOCCREDITOVFL, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_NCISOCCREDITOVFL, "egress credit overflow", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_NCISOCCREDITOVFL, data);

        credit_data.data[0] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _NCISOC_CREDIT0);
        credit_data.data[1] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _NCISOC_CREDIT1);
        credit_data.data[2] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _NCISOC_CREDIT2);
        credit_data.data[3] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _NCISOC_CREDIT3);
        credit_data.data[4] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _NCISOC_CREDIT4);
        credit_data.data[5] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _NCISOC_CREDIT5);
        credit_data.data[6] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _NCISOC_CREDIT6);
        credit_data.data[7] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _NCISOC_CREDIT7);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_NCISOCCREDITOVFL, credit_data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_EGRESS, _ERR_STATUS_0, _REQTGTIDMISMATCHERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_REQTGTIDMISMATCHERR, "egress destination request ID error", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_REQTGTIDMISMATCHERR, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_EGRESS, _ERR_STATUS_0, _RSPREQIDMISMATCHERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_RSPREQIDMISMATCHERR, "egress destination response ID error", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_RSPREQIDMISMATCHERR, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_EGRESS, _ERR_STATUS_0, _NXBAR_HDR_PARITY_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_NXBAR_HDR_PARITY_ERR, "egress control parity error", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_NXBAR_HDR_PARITY_ERR, data);
        lwswitch_clear_flags(&unhandled, bit);

        _lwswitch_construct_ecc_error_event(&err_event,
            LWSWITCH_ERR_HW_NPORT_EGRESS_NXBAR_HDR_PARITY_ERR, link, LW_FALSE, 0,
            LW_TRUE, 1);

        lwswitch_inforom_ecc_log_err_event(device, &err_event);
    }

    bit = DRF_NUM(_EGRESS, _ERR_STATUS_0, _NCISOC_CREDIT_PARITY_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_NCISOC_CREDIT_PARITY_ERR, "egress credit parity error", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_NCISOC_CREDIT_PARITY_ERR, data);

        credit_data.data[0] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _NCISOC_CREDIT0);
        credit_data.data[1] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _NCISOC_CREDIT1);
        credit_data.data[2] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _NCISOC_CREDIT2);
        credit_data.data[3] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _NCISOC_CREDIT3);
        credit_data.data[4] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _NCISOC_CREDIT4);
        credit_data.data[5] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _NCISOC_CREDIT5);
        credit_data.data[6] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _NCISOC_CREDIT6);
        credit_data.data[7] = LWSWITCH_NPORT_RD32_LR10(device, link, _EGRESS, _NCISOC_CREDIT7);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_NCISOC_CREDIT_PARITY_ERR, credit_data);
        lwswitch_clear_flags(&unhandled, bit);

        _lwswitch_construct_ecc_error_event(&err_event,
            LWSWITCH_ERR_HW_NPORT_EGRESS_NCISOC_CREDIT_PARITY_ERR, link, LW_FALSE, 0,
            LW_TRUE, 1);

        lwswitch_inforom_ecc_log_err_event(device, &err_event);
    }

    bit = DRF_NUM(_EGRESS, _ERR_STATUS_0, _NXBAR_FLITTYPE_MISMATCH_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_NXBAR_FLITTYPE_MISMATCH_ERR, "egress flit type mismatch", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_NXBAR_FLITTYPE_MISMATCH_ERR, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_EGRESS, _ERR_STATUS_0, _CREDIT_TIME_OUT_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_CREDIT_TIME_OUT_ERR, "egress credit timeout", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_CREDIT_TIME_OUT_ERR, data);
        lwswitch_clear_flags(&unhandled, bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    // This helps prevent an interrupt storm if HW keeps triggering unnecessary stream of interrupts.
    if (device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_NPORT_WR32_LR10(device, link, _EGRESS, _ERR_FATAL_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    if (report.raw_first & report.mask)
    {
        LWSWITCH_NPORT_WR32_LR10(device, link, _EGRESS, _ERR_FIRST_0,
            report.raw_first & report.mask);
    }

    LWSWITCH_NPORT_WR32_LR10(device, link, _EGRESS, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_sourcetrack_nonfatal_lr10
(
    lwswitch_device *device,
    LwU32           link
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, unhandled;
    INFOROM_LWS_ECC_ERROR_EVENT err_event = {0};

    report.raw_pending = LWSWITCH_NPORT_RD32_LR10(device, link,
                            _SOURCETRACK, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_NPORT_RD32_LR10(device, link,
                            _SOURCETRACK, _ERR_NON_FATAL_REPORT_EN_0);
    report.mask = report.raw_enable & chip_device->intr_mask.sourcetrack.nonfatal;

    pending = report.raw_pending & report.mask;
    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;
    report.raw_first = LWSWITCH_NPORT_RD32_LR10(device, link, _SOURCETRACK, _ERR_FIRST_0);

    bit = DRF_NUM(_SOURCETRACK, _ERR_STATUS_0, _CREQ_TCEN0_CRUMBSTORE_ECC_LIMIT_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        // Ignore LIMIT error if DBE is pending
        if (!(lwswitch_test_flags(report.raw_pending,
                DRF_NUM(_SOURCETRACK, _ERR_STATUS_0, _CREQ_TCEN0_CRUMBSTORE_ECC_DBE_ERR, 1))))
        {
            LwBool bAddressValid = LW_FALSE;
            LwU32 address = 0;
            LwU32 addressValid = LWSWITCH_NPORT_RD32_LR10(device, link, _SOURCETRACK,
                    _ERR_CREQ_TCEN0_CRUMBSTORE_ECC_ERROR_ADDRESS_VALID);

            if (FLD_TEST_DRF(_SOURCETRACK_ERR_CREQ_TCEN0_CRUMBSTORE, _ECC_ERROR_ADDRESS_VALID,
                             _VALID, _VALID, addressValid))
            {
                address = LWSWITCH_NPORT_RD32_LR10(device, link, _SOURCETRACK,
                                                   _ERR_CREQ_TCEN0_CRUMBSTORE_ECC_ERROR_ADDRESS);
                bAddressValid = LW_TRUE;
            }

            report.data[0] = LWSWITCH_NPORT_RD32_LR10(device, link, _SOURCETRACK,
                                _ERR_CREQ_TCEN0_CRUMBSTORE_ECC_ERROR_COUNTER);
            report.data[1] = LWSWITCH_NPORT_RD32_LR10(device, link, _SOURCETRACK,
                                _ERR_CREQ_TCEN0_CRUMBSTORE_ECC_ERROR_ADDRESS);
            report.data[2] = LWSWITCH_NPORT_RD32_LR10(device, link, _SOURCETRACK,
                                _ERR_CREQ_TCEN0_CRUMBSTORE_ECC_ERROR_ADDRESS_VALID);
            LWSWITCH_REPORT_NONFATAL(_HW_NPORT_SOURCETRACK_CREQ_TCEN0_CRUMBSTORE_ECC_LIMIT_ERR,
                                    "sourcetrack TCEN0 crumbstore ECC limit err");

            _lwswitch_construct_ecc_error_event(&err_event,
                LWSWITCH_ERR_HW_NPORT_SOURCETRACK_CREQ_TCEN0_CRUMBSTORE_ECC_LIMIT_ERR, link,
                bAddressValid, address, LW_FALSE, 1);

            lwswitch_inforom_ecc_log_err_event(device, &err_event);
        }

        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_SOURCETRACK, _ERR_STATUS_0, _CREQ_TCEN1_CRUMBSTORE_ECC_LIMIT_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        // Ignore LIMIT error if DBE is pending
        if (!(lwswitch_test_flags(report.raw_pending,
                DRF_NUM(_SOURCETRACK, _ERR_STATUS_0, _CREQ_TCEN1_CRUMBSTORE_ECC_DBE_ERR, 1))))
            {
            LwBool bAddressValid = LW_FALSE;
            LwU32 address = 0;
            LwU32 addressValid = LWSWITCH_NPORT_RD32_LR10(device, link, _SOURCETRACK,
                    _ERR_CREQ_TCEN1_CRUMBSTORE_ECC_ERROR_ADDRESS_VALID);

            if (FLD_TEST_DRF(_SOURCETRACK_ERR_CREQ_TCEN1_CRUMBSTORE, _ECC_ERROR_ADDRESS_VALID,
                             _VALID, _VALID, addressValid))
            {
                address = LWSWITCH_NPORT_RD32_LR10(device, link, _SOURCETRACK,
                                                   _ERR_CREQ_TCEN1_CRUMBSTORE_ECC_ERROR_ADDRESS);
                bAddressValid = LW_TRUE;
            }

            report.data[0] = LWSWITCH_NPORT_RD32_LR10(device, link, _SOURCETRACK,
                                _ERR_CREQ_TCEN1_CRUMBSTORE_ECC_ERROR_COUNTER);
            report.data[1] = LWSWITCH_NPORT_RD32_LR10(device, link, _SOURCETRACK,
                                _ERR_CREQ_TCEN1_CRUMBSTORE_ECC_ERROR_ADDRESS);
            report.data[2] = LWSWITCH_NPORT_RD32_LR10(device, link, _SOURCETRACK,
                                _ERR_CREQ_TCEN1_CRUMBSTORE_ECC_ERROR_ADDRESS_VALID);
            LWSWITCH_REPORT_NONFATAL(_HW_NPORT_SOURCETRACK_CREQ_TCEN1_CRUMBSTORE_ECC_LIMIT_ERR,
                                    "sourcetrack TCEN1 crumbstore ECC limit err");

            _lwswitch_construct_ecc_error_event(&err_event,
                LWSWITCH_ERR_HW_NPORT_SOURCETRACK_CREQ_TCEN1_CRUMBSTORE_ECC_LIMIT_ERR, link,
                bAddressValid, address, LW_FALSE, 1);

            lwswitch_inforom_ecc_log_err_event(device, &err_event);
        }

        lwswitch_clear_flags(&unhandled, bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    //
    // Disable interrupts that have oclwrred after fatal error.
    // This helps prevent an interrupt storm if HW keeps triggering unnecessary stream of interrupts.
    //
    if (device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_NPORT_WR32_LR10(device, link, _SOURCETRACK, _ERR_NON_FATAL_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    if (report.raw_first & report.mask)
    {
        LWSWITCH_NPORT_WR32_LR10(device, link, _SOURCETRACK, _ERR_FIRST_0,
                report.raw_first & report.mask);
    }

    LWSWITCH_NPORT_WR32_LR10(device, link, _SOURCETRACK, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_sourcetrack_fatal_lr10
(
    lwswitch_device *device,
    LwU32            link
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, contain, unhandled;
    INFOROM_LWS_ECC_ERROR_EVENT err_event = {0};

    report.raw_pending = LWSWITCH_NPORT_RD32_LR10(device, link,
                            _SOURCETRACK, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_NPORT_RD32_LR10(device, link,
                            _SOURCETRACK, _ERR_FATAL_REPORT_EN_0);
    report.mask = report.raw_enable & chip_device->intr_mask.sourcetrack.fatal;
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;
    report.raw_first = LWSWITCH_NPORT_RD32_LR10(device, link, _SOURCETRACK, _ERR_FIRST_0);
    contain = LWSWITCH_NPORT_RD32_LR10(device, link, _SOURCETRACK, _ERR_CONTAIN_EN_0);

    bit = DRF_NUM(_SOURCETRACK, _ERR_STATUS_0, _CREQ_TCEN0_CRUMBSTORE_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LwBool bAddressValid = LW_FALSE;
        LwU32 address = 0;
        LwU32 addressValid = LWSWITCH_NPORT_RD32_LR10(device, link, _SOURCETRACK,
                _ERR_CREQ_TCEN0_CRUMBSTORE_ECC_ERROR_ADDRESS_VALID);

        if (FLD_TEST_DRF(_SOURCETRACK_ERR_CREQ_TCEN0_CRUMBSTORE, _ECC_ERROR_ADDRESS_VALID,
                         _VALID, _VALID, addressValid))
        {
            address = LWSWITCH_NPORT_RD32_LR10(device, link, _SOURCETRACK,
                                               _ERR_CREQ_TCEN0_CRUMBSTORE_ECC_ERROR_ADDRESS);
            bAddressValid = LW_TRUE;
        }

        report.data[0] = LWSWITCH_NPORT_RD32_LR10(device, link, _SOURCETRACK,
                            _ERR_CREQ_TCEN0_CRUMBSTORE_ECC_ERROR_ADDRESS);
        report.data[1] = LWSWITCH_NPORT_RD32_LR10(device, link, _SOURCETRACK,
                            _ERR_CREQ_TCEN0_CRUMBSTORE_ECC_ERROR_ADDRESS_VALID);
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_SOURCETRACK_CREQ_TCEN0_CRUMBSTORE_ECC_DBE_ERR,
                                "sourcetrack TCEN0 crumbstore DBE", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);

        _lwswitch_construct_ecc_error_event(&err_event,
            LWSWITCH_ERR_HW_NPORT_SOURCETRACK_CREQ_TCEN0_CRUMBSTORE_ECC_DBE_ERR,
            link, bAddressValid, address, LW_TRUE, 1);

        lwswitch_inforom_ecc_log_err_event(device, &err_event);

        // Clear associated LIMIT_ERR interrupt
        if (report.raw_pending & DRF_NUM(_SOURCETRACK, _ERR_STATUS_0, _CREQ_TCEN0_CRUMBSTORE_ECC_LIMIT_ERR, 1))
        {
            LWSWITCH_NPORT_WR32_LR10(device, link, _SOURCETRACK, _ERR_STATUS_0,
                DRF_NUM(_SOURCETRACK, _ERR_STATUS_0, _CREQ_TCEN0_CRUMBSTORE_ECC_LIMIT_ERR, 1));
        }
    }

    bit = DRF_NUM(_SOURCETRACK, _ERR_STATUS_0, _CREQ_TCEN1_CRUMBSTORE_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LwBool bAddressValid = LW_FALSE;
        LwU32 address = 0;
        LwU32 addressValid = LWSWITCH_NPORT_RD32_LR10(device, link, _SOURCETRACK,
                _ERR_CREQ_TCEN1_CRUMBSTORE_ECC_ERROR_ADDRESS_VALID);

        if (FLD_TEST_DRF(_SOURCETRACK_ERR_CREQ_TCEN1_CRUMBSTORE, _ECC_ERROR_ADDRESS_VALID,
                         _VALID, _VALID, addressValid))
        {
            address = LWSWITCH_NPORT_RD32_LR10(device, link, _SOURCETRACK,
                                               _ERR_CREQ_TCEN1_CRUMBSTORE_ECC_ERROR_ADDRESS);
            bAddressValid = LW_TRUE;
        }

        report.data[0] = LWSWITCH_NPORT_RD32_LR10(device, link, _SOURCETRACK,
                            _ERR_CREQ_TCEN1_CRUMBSTORE_ECC_ERROR_ADDRESS);
        report.data[1] = LWSWITCH_NPORT_RD32_LR10(device, link, _SOURCETRACK,
                            _ERR_CREQ_TCEN1_CRUMBSTORE_ECC_ERROR_ADDRESS_VALID);
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_SOURCETRACK_CREQ_TCEN1_CRUMBSTORE_ECC_DBE_ERR,
                                "sourcetrack TCEN1 crumbstore DBE", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);

        _lwswitch_construct_ecc_error_event(&err_event,
            LWSWITCH_ERR_HW_NPORT_SOURCETRACK_CREQ_TCEN1_CRUMBSTORE_ECC_DBE_ERR,
            link, bAddressValid, address, LW_TRUE, 1);

        lwswitch_inforom_ecc_log_err_event(device, &err_event);

        // Clear associated LIMIT_ERR interrupt
        if (report.raw_pending & DRF_NUM(_SOURCETRACK, _ERR_STATUS_0, _CREQ_TCEN1_CRUMBSTORE_ECC_LIMIT_ERR, 1))
        {
            LWSWITCH_NPORT_WR32_LR10(device, link, _SOURCETRACK, _ERR_STATUS_0,
                DRF_NUM(_SOURCETRACK, _ERR_STATUS_0, _CREQ_TCEN1_CRUMBSTORE_ECC_LIMIT_ERR, 1));
        }
    }

    bit = DRF_NUM(_SOURCETRACK, _ERR_STATUS_0, _SOURCETRACK_TIME_OUT_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_SOURCETRACK_SOURCETRACK_TIME_OUT_ERR,
                                "sourcetrack timeout error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    //
    // Disable interrupts that have oclwrred after fatal error.
    // This helps prevent an interrupt storm if HW keeps triggering unnecessary stream of interrupts.
    //
    if (device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_NPORT_WR32_LR10(device, link, _SOURCETRACK, _ERR_FATAL_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    if (report.raw_first & report.mask)
    {
        LWSWITCH_NPORT_WR32_LR10(device, link, _SOURCETRACK, _ERR_FIRST_0,
            report.raw_first & report.mask);
    }

    LWSWITCH_NPORT_WR32_LR10(device, link, _SOURCETRACK, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;

}

static LwlStatus
_lwswitch_service_nport_fatal_lr10
(
    lwswitch_device *device,
    LwU32            link
)
{
    LwlStatus status[5];

    status[0] = _lwswitch_service_route_fatal_lr10(device, link);
    status[1] = _lwswitch_service_ingress_fatal_lr10(device, link);
    status[2] = _lwswitch_service_egress_fatal_lr10(device, link);
    status[3] = _lwswitch_service_tstate_fatal_lr10(device, link);
    status[4] = _lwswitch_service_sourcetrack_fatal_lr10(device, link);

    if ((status[0] != LWL_SUCCESS) &&
        (status[1] != LWL_SUCCESS) &&
        (status[2] != LWL_SUCCESS) &&
        (status[3] != LWL_SUCCESS) &&
        (status[4] != LWL_SUCCESS))
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_npg_fatal_lr10
(
    lwswitch_device *device,
    LwU32            npg
)
{
    LwU32 pending, mask, bit, unhandled;
    LwU32 nport;
    LwU32 link;

    pending = LWSWITCH_NPG_RD32_LR10(device, npg, _NPG, _NPG_INTERRUPT_STATUS);
 
    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    mask = 
        DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV0_INT_STATUS, _FATAL) |
        DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV1_INT_STATUS, _FATAL) |
        DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV2_INT_STATUS, _FATAL) |
        DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV3_INT_STATUS, _FATAL);
    pending &= mask;
    unhandled = pending;

    for (nport = 0; nport < LWSWITCH_NPORT_PER_NPG; nport++)
    {
        switch (nport)
        {
            case 0:
                bit = DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV0_INT_STATUS, _FATAL);
                break;
            case 1:
                bit = DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV1_INT_STATUS, _FATAL);
                break;
            case 2:
                bit = DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV2_INT_STATUS, _FATAL);
                break;
            case 3:
                bit = DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV3_INT_STATUS, _FATAL);
                break;
            default:
                bit = 0;
                LWSWITCH_ASSERT(0);
                break;
        }
        if (lwswitch_test_flags(pending, bit))
        {
            link = NPORT_TO_LINK(device, npg, nport);
            if (LWSWITCH_ENG_VALID_LR10(device, NPORT, link))
            {
                if (_lwswitch_service_nport_fatal_lr10(device, link) == LWL_SUCCESS)
                {
                    lwswitch_clear_flags(&unhandled, bit);
                }
            }
        }
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_nport_nonfatal_lr10
(
    lwswitch_device *device,
    LwU32            link
)
{
    LwlStatus status[5];

    status[0] = _lwswitch_service_route_nonfatal_lr10(device, link);
    status[1] = _lwswitch_service_ingress_nonfatal_lr10(device, link);
    status[2] = _lwswitch_service_egress_nonfatal_lr10(device, link);
    status[3] = _lwswitch_service_tstate_nonfatal_lr10(device, link);
    status[4] = _lwswitch_service_sourcetrack_nonfatal_lr10(device, link);

    if ((status[0] != LWL_SUCCESS) &&
        (status[1] != LWL_SUCCESS) &&
        (status[2] != LWL_SUCCESS) &&
        (status[3] != LWL_SUCCESS) &&
        (status[4] != LWL_SUCCESS))
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_npg_nonfatal_lr10
(
    lwswitch_device *device,
    LwU32 npg
)
{
    LwU32 pending, mask, bit, unhandled;
    LwU32 nport;
    LwU32 link;

    pending = LWSWITCH_NPG_RD32_LR10(device, npg, _NPG, _NPG_INTERRUPT_STATUS);

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    mask = 
        DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV0_INT_STATUS, _NONFATAL) |
        DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV1_INT_STATUS, _NONFATAL) |
        DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV2_INT_STATUS, _NONFATAL) |
        DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV3_INT_STATUS, _NONFATAL);
    pending &= mask;
    unhandled = pending;

    for (nport = 0; nport < LWSWITCH_NPORT_PER_NPG; nport++)
    {
        switch (nport)
        {
            case 0:
                bit = DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV0_INT_STATUS, _NONFATAL);
                break;
            case 1:
                bit = DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV1_INT_STATUS, _NONFATAL);
                break;
            case 2:
                bit = DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV2_INT_STATUS, _NONFATAL);
                break;
            case 3:
                bit = DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV3_INT_STATUS, _NONFATAL);
                break;
            default:
                bit = 0;
                LWSWITCH_ASSERT(0);
                break;
        }
        if (lwswitch_test_flags(pending, bit))
        {
            link = NPORT_TO_LINK(device, npg, nport);
            if (LWSWITCH_ENG_VALID_LR10(device, NPORT, link))
            {
                if (_lwswitch_service_nport_nonfatal_lr10(device, link) == LWL_SUCCESS)
                {
                    lwswitch_clear_flags(&unhandled, bit);
                }
            }
        }
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_service_minion_link_lr10
(
    lwswitch_device *device,
    LwU32 instance
)
{
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, unhandled, minionIntr, linkIntr, reg, enabledLinks, bit;
    LwU32 localLinkIdx, link;

    //
    // _MINION_MINION_INTR shows all interrupts lwrrently at the host on this minion
    // Note: _MINIO_MINION_INTR is not used to clear link specific interrupts
    //
    minionIntr = LWSWITCH_MINION_RD32_LR10(device, instance, _MINION, _MINION_INTR);

    // get all possible interrupting links associated with this minion
    report.raw_pending = DRF_VAL(_MINION, _MINION_INTR, _LINK, minionIntr);

    // read in the enaled minion interrupts on this minion
    reg = LWSWITCH_MINION_RD32_LR10(device, instance, _MINION, _MINION_INTR_STALL_EN);

    // get the links with enabled interrupts on this minion
    enabledLinks = DRF_VAL(_MINION, _MINION_INTR_STALL_EN, _LINK, reg);

    report.raw_enable = enabledLinks;
    report.mask = report.raw_enable;

    // pending bit field contains interrupting links after being filtered
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;

    FOR_EACH_INDEX_IN_MASK(32, localLinkIdx, pending)
    {
        link = (instance * LWSWITCH_LINKS_PER_LWLIPT) + localLinkIdx;
        bit = LWBIT(localLinkIdx);

        // read in the interrupt register for the given link
        linkIntr = LWSWITCH_MINION_LINK_RD32_LR10(device, link, _MINION, _LWLINK_LINK_INTR(localLinkIdx));

        // _STATE must be set for _CODE to be valid
        if (!DRF_VAL(_MINION, _LWLINK_LINK_INTR, _STATE, linkIntr))
        {
            continue;
        }

        report.data[0] = linkIntr;

        switch(DRF_VAL(_MINION, _LWLINK_LINK_INTR, _CODE, linkIntr))
        {
            case LW_MINION_LWLINK_LINK_INTR_CODE_NA:
                LWSWITCH_REPORT_FATAL(_HW_MINION_FATAL_LINK_INTR, "Minion Link NA interrupt", LW_FALSE);
                break;
            case LW_MINION_LWLINK_LINK_INTR_CODE_SWREQ:
                LWSWITCH_PRINT(device, INFO,
                      "%s: Received MINION Link SW Generate interrupt on MINION %d : link %d.\n",
                      __FUNCTION__, instance, link);
                break;
            case LW_MINION_LWLINK_LINK_INTR_CODE_DLREQ:
                LWSWITCH_REPORT_NONFATAL(_HW_MINION_NONFATAL, "Minion Link DLREQ interrupt");
                break;
            case LW_MINION_LWLINK_LINK_INTR_CODE_PMDISABLED:
                LWSWITCH_REPORT_NONFATAL(_HW_MINION_NONFATAL, "Minion Link PMDISABLED interrupt");
                break;
            case LW_MINION_LWLINK_LINK_INTR_CODE_DLCMDFAULT:
                LWSWITCH_REPORT_FATAL(_HW_MINION_FATAL_LINK_INTR, "Minion Link DLCMDFAULT interrupt", LW_FALSE);
                break;
            case LW_MINION_LWLINK_LINK_INTR_CODE_TLREQ:
                LWSWITCH_REPORT_NONFATAL(_HW_MINION_NONFATAL, "Minion Link TLREQ interrupt");
                break;
            case LW_MINION_LWLINK_LINK_INTR_CODE_NOINIT:
                LWSWITCH_REPORT_FATAL(_HW_MINION_FATAL_LINK_INTR, "Minion Link NOINIT interrupt", LW_FALSE);
                break;
            case LW_MINION_LWLINK_LINK_INTR_CODE_NOTIFY:
                LWSWITCH_PRINT(device, INFO,
                      "%s: Received MINION NOTIFY interrupt on MINION %d : link %d.\n",
                      __FUNCTION__, instance, link);
                break;
            case LW_MINION_LWLINK_LINK_INTR_CODE_LOCAL_CONFIG_ERR:
                LWSWITCH_REPORT_FATAL(_HW_MINION_FATAL_LINK_INTR, "Minion Link Local-Config-Error interrupt", LW_FALSE);
                break;
            case LW_MINION_LWLINK_LINK_INTR_CODE_NEGOTIATION_CONFIG_ERR:
                LWSWITCH_REPORT_FATAL(_HW_MINION_FATAL_LINK_INTR, "Minion Link Negotiation Config Err Interrupt", LW_FALSE);
                break;
            case LW_MINION_LWLINK_LINK_INTR_CODE_BADINIT: 
                LWSWITCH_REPORT_FATAL(_HW_MINION_FATAL_LINK_INTR, "Minion Link BADINIT interrupt", LW_FALSE);
                break;
            case LW_MINION_LWLINK_LINK_INTR_CODE_PMFAIL:
                LWSWITCH_REPORT_FATAL(_HW_MINION_FATAL_LINK_INTR, "Minion Link PMFAIL interrupt", LW_FALSE);
                break;
            default:
                LWSWITCH_REPORT_FATAL(_HW_MINION_FATAL_LINK_INTR, "Minion Interrupt code unknown", LW_FALSE);
        }
        lwswitch_clear_flags(&unhandled, bit);

        // Disable interrupt bit for the given link - fatal error olwrred before
        if (device->link[link].fatal_error_oclwrred)
        {
            enabledLinks &= ~bit;
            reg = DRF_NUM(_MINION, _MINION_INTR_STALL_EN, _LINK, enabledLinks);
            LWSWITCH_MINION_LINK_WR32_LR10(device, link, _MINION, _MINION_INTR_STALL_EN, reg);
        }

        //
        // _MINION_INTR_LINK is a read-only register field for the host
        // Host must write 1 to _LWLINK_LINK_INTR_STATE to clear the interrupt on the link
        //
        reg = DRF_NUM(_MINION, _LWLINK_LINK_INTR, _STATE, 1);
        LWSWITCH_MINION_WR32_LR10(device, instance, _MINION, _LWLINK_LINK_INTR(localLinkIdx), reg);
    }
    FOR_EACH_INDEX_IN_MASK_END;

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_lwldl_nonfatal_link_lr10
(
    lwswitch_device *device,
    LwU32 link
)
{
    LwU32 pending, bit, unhandled;
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };

    report.raw_pending = LWSWITCH_LINK_RD32_LR10(device, link, LWLDL, _LWLDL_TOP, _INTR);
    report.raw_enable = LWSWITCH_LINK_RD32_LR10(device, link, LWLDL, _LWLDL_TOP, _INTR_NONSTALL_EN);
    report.mask = report.raw_enable;
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _TX_REPLAY, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_DLPL_TX_REPLAY, "TX Replay Error");
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _TX_RECOVERY_SHORT, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_DLPL_TX_RECOVERY_SHORT, "TX Recovery Short");
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _RX_SHORT_ERROR_RATE, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_DLPL_RX_SHORT_ERROR_RATE, "RX Short Error Rate");
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _RX_LONG_ERROR_RATE, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_DLPL_RX_LONG_ERROR_RATE, "RX Long Error Rate");
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _RX_ILA_TRIGGER, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_DLPL_RX_ILA_TRIGGER, "RX ILA Trigger");
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _RX_CRC_COUNTER, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_DLPL_RX_CRC_COUNTER, "RX CRC Counter");
        lwswitch_clear_flags(&unhandled, bit);

        //
        // Mask CRC counter after first oclwrrance - otherwise, this interrupt
        // will continue to fire once the CRC counter has hit the threshold
        // See Bug 3341528
        //
        report.raw_enable = report.raw_enable & (~bit);
        LWSWITCH_LINK_WR32_LR10(device, link, LWLDL, _LWLDL_TOP, _INTR_NONSTALL_EN,
            report.raw_enable);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    LWSWITCH_LINK_WR32_LR10(device, link, LWLDL, _LWLDL_TOP, _INTR, pending);

    if (unhandled != 0)
    {
        LWSWITCH_PRINT(device, WARN,
                "%s: Unhandled LWLDL nonfatal interrupts, link: %d pending: 0x%x enabled: 0x%x.\n",
                 __FUNCTION__, link, pending, report.raw_enable);
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_lwldl_nonfatal_lr10
(
    lwswitch_device *device,
    LwU32 lwlipt_instance
)
{
    LwU64 enabledLinkMask, localLinkMask, localEnabledLinkMask;
    LwU32 i;
    lwlink_link *link;
    LwlStatus status = -LWL_MORE_PROCESSING_REQUIRED;

    enabledLinkMask = lwswitch_get_enabled_link_mask(device);
    localLinkMask = LWSWITCH_LWLIPT_GET_LOCAL_LINK_MASK64(lwlipt_instance);
    localEnabledLinkMask = enabledLinkMask & localLinkMask;

    FOR_EACH_INDEX_IN_MASK(64, i, localEnabledLinkMask)
    {
        link = lwswitch_get_link(device, i);
        if (link == NULL)
        {
            // An interrupt on an invalid link should never occur
            LWSWITCH_ASSERT(link != NULL);
            continue;
        }

        if (LWSWITCH_GET_LINK_ENG_INST(device, i, LWLIPT) != lwlipt_instance)
        {
            LWSWITCH_ASSERT(0);
            break;
        }

        if (lwswitch_is_link_in_reset(device, link))
        {
            continue;
        }

        if (_lwswitch_service_lwldl_nonfatal_link_lr10(device, i) == LWL_SUCCESS)
        {
            status = LWL_SUCCESS;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return status;
}

static LwlStatus
_lwswitch_service_lwltlc_rx_lnk_nonfatal_0_lr10
(
    lwswitch_device *device,
    LwU32 lwlipt_instance,
    LwU32 link
)
{
    LwU32 pending, bit, unhandled;
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    INFOROM_LWLINK_ERROR_EVENT error_event;
    LwU32 injected;
#endif

    report.raw_pending = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_NON_FATAL_REPORT_EN_0);
    report.mask = report.raw_enable;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    error_event.lwliptInstance = (LwU8) lwlipt_instance;
    error_event.localLinkIdx   = (LwU8) LWSWITCH_LWLIPT_GET_LOCAL_LINK_ID_LR10(link);
#endif

    pending = report.raw_pending & report.mask;
    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;
    report.raw_first = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_FIRST_0);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    injected = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_REPORT_INJECT_0);
#endif

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_0, _RXRSPSTATUS_PRIV_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_LWLTLC_RX_LNK_RXRSPSTATUS_PRIV_ERR, "RX Rsp Status PRIV Error");
        lwswitch_clear_flags(&unhandled, bit);

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        if (FLD_TEST_DRF_NUM(_LWLTLC_RX_LNK, _ERR_REPORT_INJECT_0, _RXRSPSTATUS_PRIV_ERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_RX_RSP_STATUS_PRIV_ERR_NONFATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    if (report.raw_first & report.mask)
    {
        LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_FIRST_0,
                report.raw_first & report.mask);
    }
    LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        LWSWITCH_PRINT(device, WARN,
                "%s: Unhandled LWLTLC_RX_LNK _0 interrupts, link: %d pending: 0x%x enabled: 0x%x.\n",
                 __FUNCTION__, link, pending, report.raw_enable);
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_lwltlc_tx_lnk_nonfatal_0_lr10
(
    lwswitch_device *device,
    LwU32 lwlipt_instance,
    LwU32 link
)
{
    LwU32 pending, bit, unhandled;
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    INFOROM_LWLINK_ERROR_EVENT error_event;
    LwU32 injected;
#endif

    report.raw_pending = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_TX_LNK, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_TX_LNK, _ERR_NON_FATAL_REPORT_EN_0);
    report.mask = report.raw_enable;
    pending = report.raw_pending & report.mask;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    error_event.lwliptInstance = (LwU8) lwlipt_instance;
    error_event.localLinkIdx   = (LwU8) LWSWITCH_LWLIPT_GET_LOCAL_LINK_ID_LR10(link);
#endif

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;
    report.raw_first = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_TX_LNK, _ERR_FIRST_0);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    injected = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_TX_LNK, _ERR_REPORT_INJECT_0);
#endif

    bit = DRF_NUM(_LWLTLC_TX_LNK, _ERR_STATUS_0, _CREQ_RAM_DAT_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_LWLTLC_TX_LNK_CREQ_RAM_DAT_ECC_DBE_ERR, "CREQ RAM DAT ECC DBE Error");
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC_TX_LNK, _ERR_REPORT_INJECT_0, _CREQ_RAM_DAT_ECC_DBE_ERR, 0x0, injected))
        {
            // TODO 3014908 log these in the LWL object until we have ECC object support
            error_event.error = INFOROM_LWLINK_TLC_TX_CREQ_DAT_RAM_ECC_DBE_NONFATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_TX_LNK, _ERR_STATUS_0, _CREQ_RAM_ECC_LIMIT_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_LWLTLC_TX_LNK_CREQ_RAM_ECC_LIMIT_ERR, "CREQ RAM DAT ECC Limit Error");
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_LWLTLC_TX_LNK, _ERR_STATUS_0, _RSP_RAM_DAT_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_LWLTLC_TX_LNK_RSP_RAM_DAT_ECC_DBE_ERR, "Response RAM DAT ECC DBE Error");
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_LWLTLC_TX_LNK, _ERR_STATUS_0, _RSP_RAM_ECC_LIMIT_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_LWLTLC_TX_LNK_RSP_RAM_ECC_LIMIT_ERR, "Response RAM ECC Limit Error");
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_LWLTLC_TX_LNK, _ERR_STATUS_0, _COM_RAM_DAT_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_LWLTLC_TX_LNK_COM_RAM_DAT_ECC_DBE_ERR, "COM RAM DAT ECC DBE Error");
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC_TX_LNK, _ERR_REPORT_INJECT_0, _COM_RAM_DAT_ECC_DBE_ERR, 0x0, injected))
        {
            // TODO 3014908 log these in the LWL object until we have ECC object support
            error_event.error = INFOROM_LWLINK_TLC_TX_COM_DAT_RAM_ECC_DBE_NONFATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_TX_LNK, _ERR_STATUS_0, _COM_RAM_ECC_LIMIT_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_LWLTLC_TX_LNK_COM_RAM_ECC_LIMIT_ERR, "COM RAM ECC Limit Error");
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_LWLTLC_TX_LNK, _ERR_STATUS_0, _RSP1_RAM_ECC_LIMIT_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_LWLTLC_TX_LNK_RSP1_RAM_ECC_LIMIT_ERR, "RSP1 RAM ECC Limit Error");
        lwswitch_clear_flags(&unhandled, bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_TX_LNK, _ERR_NON_FATAL_REPORT_EN_0,
            report.raw_enable ^ pending);
    }

    if (report.raw_first & report.mask)
    {
        LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_TX_LNK, _ERR_FIRST_0,
                report.raw_first & report.mask);
    }
    LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_TX_LNK, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        LWSWITCH_PRINT(device, WARN,
                "%s: Unhandled LWLTLC_TX_LNK _0 interrupts, link: %d pending: 0x%x enabled: 0x%x.\n",
                 __FUNCTION__, link, pending, report.raw_enable);
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_lwltlc_rx_lnk_nonfatal_1_lr10
(
    lwswitch_device *device,
    LwU32 lwlipt_instance,
    LwU32 link
)
{
    LwU32 pending, bit, unhandled, injected;
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    INFOROM_LWLINK_ERROR_EVENT error_event;
#endif

    report.raw_pending = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_STATUS_1);
    report.raw_enable = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_NON_FATAL_REPORT_EN_1);
    report.mask = report.raw_enable;
    pending = report.raw_pending & report.mask;
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

    error_event.lwliptInstance = (LwU8) lwlipt_instance;
    error_event.localLinkIdx   = (LwU8) LWSWITCH_LWLIPT_GET_LOCAL_LINK_ID_LR10(link);
#endif

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;
    report.raw_first = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_FIRST_1);
    injected = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_REPORT_INJECT_1);

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_1, _AN1_HEARTBEAT_TIMEOUT_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_LWLTLC_RX_LNK_AN1_HEARTBEAT_TIMEOUT_ERR, "AN1 Heartbeat Timeout Error");
        lwswitch_clear_flags(&unhandled, bit);

        if (FLD_TEST_DRF_NUM(_LWLTLC, _RX_LNK_ERR_REPORT_INJECT_1, _AN1_HEARTBEAT_TIMEOUT_ERR, 0x0, injected))
        {
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
            error_event.error = INFOROM_LWLINK_TLC_RX_AN1_HEARTBEAT_TIMEOUT_NONFATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
#endif
            //
            // WAR Bug 200627368: Mask off HBTO to avoid a storm
            // During the start of reset_and_drain, all links on the GPU
            // will go into contain, causing HBTO on other switch links connected
            // to that GPU. For the switch side, these interrupts are not fatal,
            // but until we get to reset_and_drain for this link, HBTO will continue
            // to fire repeatedly. After reset_and_drain, HBTO will be re-enabled
            // by MINION after links are trained.
            //
            report.raw_enable = report.raw_enable & (~bit);
            LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_NON_FATAL_REPORT_EN_1,
                report.raw_enable);
        }
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_NON_FATAL_REPORT_EN_1,
            report.raw_enable & (~pending));
    }

    if (report.raw_first & report.mask)
    {
        LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_FIRST_1,
                report.raw_first & report.mask);
    }
    LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_STATUS_1, pending);

    if (unhandled != 0)
    {
        LWSWITCH_PRINT(device, WARN,
                "%s: Unhandled LWLTLC_RX_LNK _1 interrupts, link: %d pending: 0x%x enabled: 0x%x.\n",
                 __FUNCTION__, link, pending, report.raw_enable);
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_lwltlc_tx_lnk_nonfatal_1_lr10
(
    lwswitch_device *device,
    LwU32 lwlipt_instance,
    LwU32 link
)
{
    LwU32 pending, bit, unhandled;
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    INFOROM_LWLINK_ERROR_EVENT error_event = { 0 };
    LwU32 injected;
#endif

    report.raw_pending = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_TX_LNK, _ERR_STATUS_1);
    report.raw_enable = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_TX_LNK, _ERR_NON_FATAL_REPORT_EN_1);
    report.mask = report.raw_enable;
    pending = report.raw_pending & report.mask;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    error_event.lwliptInstance = (LwU8) lwlipt_instance;
    error_event.localLinkIdx   = (LwU8) LWSWITCH_LWLIPT_GET_LOCAL_LINK_ID_LR10(link);
#endif

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;
    report.raw_first = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_TX_LNK, _ERR_FIRST_1);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    injected = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_TX_LNK, _ERR_REPORT_INJECT_1);
#endif

    bit = DRF_NUM(_LWLTLC_TX_LNK, _ERR_STATUS_1, _AN1_TIMEOUT_VC0, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_LWLTLC_TX_LNK_AN1_TIMEOUT_VC0, "AN1 Timeout VC0");
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _TX_LNK_ERR_REPORT_INJECT_1, _AN1_TIMEOUT_VC0, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_TX_AN1_TIMEOUT_VC0_NONFATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_TX_LNK, _ERR_STATUS_1, _AN1_TIMEOUT_VC1, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_LWLTLC_TX_LNK_AN1_TIMEOUT_VC1, "AN1 Timeout VC1");
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _TX_LNK_ERR_REPORT_INJECT_1, _AN1_TIMEOUT_VC1, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_TX_AN1_TIMEOUT_VC1_NONFATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_TX_LNK, _ERR_STATUS_1, _AN1_TIMEOUT_VC2, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_LWLTLC_TX_LNK_AN1_TIMEOUT_VC2, "AN1 Timeout VC2");
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _TX_LNK_ERR_REPORT_INJECT_1, _AN1_TIMEOUT_VC2, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_TX_AN1_TIMEOUT_VC2_NONFATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_TX_LNK, _ERR_STATUS_1, _AN1_TIMEOUT_VC3, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_LWLTLC_TX_LNK_AN1_TIMEOUT_VC3, "AN1 Timeout VC3");
        lwswitch_clear_flags(&unhandled, bit);

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        if (FLD_TEST_DRF_NUM(_LWLTLC, _TX_LNK_ERR_REPORT_INJECT_1, _AN1_TIMEOUT_VC3, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_TX_AN1_TIMEOUT_VC3_NONFATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_TX_LNK, _ERR_STATUS_1, _AN1_TIMEOUT_VC4, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_LWLTLC_TX_LNK_AN1_TIMEOUT_VC4, "AN1 Timeout VC4");
        lwswitch_clear_flags(&unhandled, bit);

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        if (FLD_TEST_DRF_NUM(_LWLTLC, _TX_LNK_ERR_REPORT_INJECT_1, _AN1_TIMEOUT_VC4, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_TX_AN1_TIMEOUT_VC4_NONFATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_TX_LNK, _ERR_STATUS_1, _AN1_TIMEOUT_VC5, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_LWLTLC_TX_LNK_AN1_TIMEOUT_VC5, "AN1 Timeout VC5");
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _TX_LNK_ERR_REPORT_INJECT_1, _AN1_TIMEOUT_VC5, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_TX_AN1_TIMEOUT_VC5_NONFATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_TX_LNK, _ERR_STATUS_1, _AN1_TIMEOUT_VC6, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_LWLTLC_TX_LNK_AN1_TIMEOUT_VC6, "AN1 Timeout VC6");
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _TX_LNK_ERR_REPORT_INJECT_1, _AN1_TIMEOUT_VC6, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_TX_AN1_TIMEOUT_VC6_NONFATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_TX_LNK, _ERR_STATUS_1, _AN1_TIMEOUT_VC7, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_LWLTLC_TX_LNK_AN1_TIMEOUT_VC7, "AN1 Timeout VC7");
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _TX_LNK_ERR_REPORT_INJECT_1, _AN1_TIMEOUT_VC7, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_TX_AN1_TIMEOUT_VC7_NONFATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_TX_LNK, _ERR_NON_FATAL_REPORT_EN_1,
                report.raw_enable ^ pending);
    }

    if (report.raw_first & report.mask)
    {
        LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_TX_LNK, _ERR_FIRST_1,
                report.raw_first & report.mask);
    }
    LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_TX_LNK, _ERR_STATUS_1, pending);

    if (unhandled != 0)
    {
        LWSWITCH_PRINT(device, WARN,
                "%s: Unhandled LWLTLC_TX_LNK _1 interrupts, link: %d pending: 0x%x enabled: 0x%x.\n",
                 __FUNCTION__, link, pending, report.raw_enable);
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_lwltlc_nonfatal_lr10
(
    lwswitch_device *device,
    LwU32 lwlipt_instance
)
{
    LwU64 enabledLinkMask, localLinkMask, localEnabledLinkMask;
    LwU32 i;
    lwlink_link *link;
    LwlStatus status = -LWL_MORE_PROCESSING_REQUIRED;

    enabledLinkMask = lwswitch_get_enabled_link_mask(device);
    localLinkMask = LWSWITCH_LWLIPT_GET_LOCAL_LINK_MASK64(lwlipt_instance);
    localEnabledLinkMask = enabledLinkMask & localLinkMask;

    FOR_EACH_INDEX_IN_MASK(64, i, localEnabledLinkMask)
    {
        link = lwswitch_get_link(device, i);
        if (link == NULL)
        {
            // An interrupt on an invalid link should never occur
            LWSWITCH_ASSERT(link != NULL);
            continue;
        }

        if (LWSWITCH_GET_LINK_ENG_INST(device, i, LWLIPT) != lwlipt_instance)
        {
            LWSWITCH_ASSERT(0);
            break;
        }

        if (lwswitch_is_link_in_reset(device, link))
        {
            continue;
        }

        if (_lwswitch_service_lwltlc_rx_lnk_nonfatal_0_lr10(device, lwlipt_instance, i) == LWL_SUCCESS)
        {
            status = LWL_SUCCESS;
        }

        if (_lwswitch_service_lwltlc_tx_lnk_nonfatal_0_lr10(device, lwlipt_instance, i) == LWL_SUCCESS)
        {
            status = LWL_SUCCESS;
        }

        if (_lwswitch_service_lwltlc_rx_lnk_nonfatal_1_lr10(device, lwlipt_instance, i) == LWL_SUCCESS)
        {
            status = LWL_SUCCESS;
        }

        if (_lwswitch_service_lwltlc_tx_lnk_nonfatal_1_lr10(device, lwlipt_instance, i) == LWL_SUCCESS)
        {
            status = LWL_SUCCESS;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    return status;
}

static LwlStatus
_lwswitch_service_lwlipt_lnk_nonfatal_lr10
(
    lwswitch_device *device,
    LwU32 lwlipt_instance,
    LwU32 link
)
{
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, unhandled;
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    INFOROM_LWLINK_ERROR_EVENT error_event = { 0 };
    LwU32 injected;
#endif

    report.raw_pending = LWSWITCH_LINK_RD32_LR10(device, link, LWLIPT_LNK, _LWLIPT_LNK, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_LINK_RD32_LR10(device, link, LWLIPT_LNK, _LWLIPT_LNK, _ERR_NON_FATAL_REPORT_EN_0);
    report.mask = report.raw_enable;
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

    error_event.lwliptInstance = (LwU8) lwlipt_instance;
    error_event.localLinkIdx   = (LwU8) LWSWITCH_LWLIPT_GET_LOCAL_LINK_ID_LR10(link);
#endif

    pending = report.raw_pending & report.mask;
    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;
    report.raw_first = LWSWITCH_LINK_RD32_LR10(device, link, LWLIPT_LNK, _LWLIPT_LNK, _ERR_FIRST_0);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    injected = LWSWITCH_LINK_RD32_LR10(device, link, LWLIPT_LNK, _LWLIPT_LNK, _ERR_REPORT_INJECT_0);
#endif

    bit = DRF_NUM(_LWLIPT_LNK, _ERR_STATUS_0, _ILLEGALLINKSTATEREQUEST, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_LWLIPT_LNK_ILLEGALLINKSTATEREQUEST, "_HW_LWLIPT_LNK_ILLEGALLINKSTATEREQUEST");
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLIPT_LNK, _ERR_REPORT_INJECT_0, _ILLEGALLINKSTATEREQUEST, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_LWLIPT_ILLEGAL_LINK_STATE_REQUEST_NONFATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLIPT_LNK, _ERR_STATUS_0, _FAILEDMINIONREQUEST, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_LWLIPT_LNK_FAILEDMINIONREQUEST, "_FAILEDMINIONREQUEST");
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLIPT_LNK, _ERR_REPORT_INJECT_0, _FAILEDMINIONREQUEST, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_LWLIPT_FAILED_MINION_REQUEST_NONFATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLIPT_LNK, _ERR_STATUS_0, _RESERVEDREQUESTVALUE, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_LWLIPT_LNK_RESERVEDREQUESTVALUE, "_RESERVEDREQUESTVALUE");
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLIPT_LNK, _ERR_REPORT_INJECT_0, _RESERVEDREQUESTVALUE, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_LWLIPT_RESERVED_REQUEST_VALUE_NONFATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLIPT_LNK, _ERR_STATUS_0, _LINKSTATEWRITEWHILEBUSY, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_LWLIPT_LNK_LINKSTATEWRITEWHILEBUSY, "_LINKSTATEWRITEWHILEBUSY");
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLIPT_LNK, _ERR_REPORT_INJECT_0, _LINKSTATEWRITEWHILEBUSY, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_LWLIPT_LINK_STATE_WRITE_WHILE_BUSY_NONFATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLIPT_LNK, _ERR_STATUS_0, _LINK_STATE_REQUEST_TIMEOUT, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_LWLIPT_LNK_LINK_STATE_REQUEST_TIMEOUT, "_LINK_STATE_REQUEST_TIMEOUT");
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLIPT_LNK, _ERR_REPORT_INJECT_0, _LINK_STATE_REQUEST_TIMEOUT, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_LWLIPT_LINK_STATE_REQUEST_TIMEOUT_NONFATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLIPT_LNK, _ERR_STATUS_0, _WRITE_TO_LOCKED_SYSTEM_REG_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_LWLIPT_LNK_WRITE_TO_LOCKED_SYSTEM_REG_ERR, "_WRITE_TO_LOCKED_SYSTEM_REG_ERR");
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLIPT_LNK, _ERR_REPORT_INJECT_0, _WRITE_TO_LOCKED_SYSTEM_REG_ERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_LWLIPT_WRITE_TO_LOCKED_SYSTEM_REG_NONFATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    if (report.raw_first & report.mask)
    {
        LWSWITCH_LINK_WR32_LR10(device, link, LWLIPT_LNK, _LWLIPT_LNK, _ERR_FIRST_0,
            report.raw_first & report.mask);
    }
    LWSWITCH_LINK_WR32_LR10(device, link, LWLIPT_LNK, _LWLIPT_LNK, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        LWSWITCH_PRINT(device, WARN,
                "%s: Unhandled LWLIPT_LNK NON_FATAL interrupts, pending: 0x%x enabled: 0x%x.\n",
                 __FUNCTION__, pending, report.raw_enable);
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_lwlipt_link_nonfatal_lr10
(
    lwswitch_device *device,
    LwU32 lwlipt_instance
)
{
    LwU32 i, intrLink;
    LwU64 enabledLinkMask, localLinkMask, localEnabledLinkMask, interruptingLinks = 0;

    enabledLinkMask = lwswitch_get_enabled_link_mask(device);
    localLinkMask = LWSWITCH_LWLIPT_GET_LOCAL_LINK_MASK64(lwlipt_instance);
    localEnabledLinkMask = enabledLinkMask & localLinkMask;

    FOR_EACH_INDEX_IN_MASK(64, i, localEnabledLinkMask)
    {
        intrLink = LWSWITCH_LINK_RD32_LR10(device, i, LWLIPT_LNK, _LWLIPT_LNK, _ERR_STATUS_0);

        if(intrLink)
        {
            interruptingLinks |= LWBIT(i);
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    if(interruptingLinks)
    {
        FOR_EACH_INDEX_IN_MASK(64, i, interruptingLinks)
        {
            if( _lwswitch_service_lwlipt_lnk_nonfatal_lr10(device, lwlipt_instance, i) != LWL_SUCCESS)
            {
                return -LWL_MORE_PROCESSING_REQUIRED;
            }
        }
        FOR_EACH_INDEX_IN_MASK_END;
        return LWL_SUCCESS;
    }
    else
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }
}

static LwlStatus
_lwswitch_service_lwlipt_nonfatal_lr10
(
    lwswitch_device *device,
    LwU32 instance
)
{
    LwlStatus status[4];

    //
    // MINION LINK interrupts trigger both INTR_FATAL and INTR_NONFATAL
    // trees (Bug 3037835). Because of this, we must service them in both the
    // fatal and nonfatal handlers
    //
    status[0] = device->hal.lwswitch_service_minion_link(device, instance);
    status[1] = _lwswitch_service_lwldl_nonfatal_lr10(device, instance);
    status[2] = _lwswitch_service_lwltlc_nonfatal_lr10(device, instance);
    status[3] = _lwswitch_service_lwlipt_link_nonfatal_lr10(device, instance);

    if (status[0] != LWL_SUCCESS &&
        status[1] != LWL_SUCCESS &&
        status[2] != LWL_SUCCESS &&
        status[3] != LWL_SUCCESS)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_soe_fatal_lr10
(
    lwswitch_device *device
)
{
    // We only support 1 SOE as of LR10.
    if (soeService_HAL(device, (PSOE)device->pSoe) != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_saw_legacy_lr10
(
    lwswitch_device *device
)
{
    //TODO : SAW Legacy interrupts

    return -LWL_MORE_PROCESSING_REQUIRED;
}

static LwlStatus
_lwswitch_service_saw_nonfatal_lr10
(
    lwswitch_device *device
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LwU32 pending, bit, unhandled;
    LwU32 i;

    pending = LWSWITCH_SAW_RD32_LR10(device, _LWLSAW_LWSPMC, _INTR_NONFATAL);
    pending &= chip_device->intr_enable_nonfatal;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;

    for (i = 0; i < NUM_NPG_ENGINE_LR10; i++)
    {
        if (!LWSWITCH_ENG_VALID_LR10(device, NPG, i))
            continue;

        bit = DRF_NUM(_LWLSAW_LWSPMC, _INTR_NONFATAL, _NPG_0, 1) << i;
        if (lwswitch_test_flags(pending, bit))
        {
            if (_lwswitch_service_npg_nonfatal_lr10(device, i) == LWL_SUCCESS)
            {
                lwswitch_clear_flags(&unhandled, bit);
            }
        }
    }

    for (i = 0; i < NUM_LWLIPT_ENGINE_LR10; i++)
    {
        if (!LWSWITCH_ENG_VALID_LR10(device, LWLIPT, i))
        {
            continue;
        }

        bit = DRF_NUM(_LWLSAW_LWSPMC, _INTR_NONFATAL, _LWLIPT_0, 1) << i;

        if (lwswitch_test_flags(pending, bit))
        {
            if (_lwswitch_service_lwlipt_nonfatal_lr10(device, i) == LWL_SUCCESS)
            {
                lwswitch_clear_flags(&unhandled, bit);
            }
        }
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_nxbar_tile_lr10
(
    lwswitch_device *device,
    LwU32 link
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LwU32 pending, bit, unhandled;
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };

    report.raw_pending = LWSWITCH_TILE_RD32_LR10(device, link, _NXBAR_TILE, _ERR_STATUS);
    report.raw_enable = LWSWITCH_TILE_RD32_LR10(device, link, _NXBAR_TILE, _ERR_FATAL_INTR_EN);
    report.mask = chip_device->intr_mask.tile.fatal;
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;
    report.raw_first = LWSWITCH_TILE_RD32_LR10(device, link, _NXBAR_TILE, _ERR_FIRST);

    bit = DRF_NUM(_NXBAR_TILE, _ERR_STATUS, _INGRESS_BUFFER_OVERFLOW, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_NXBAR_TILE_INGRESS_BUFFER_OVERFLOW, "ingress SRC-VC buffer overflow", LW_TRUE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_NXBAR_TILE, _ERR_STATUS, _INGRESS_BUFFER_UNDERFLOW, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_NXBAR_TILE_INGRESS_BUFFER_UNDERFLOW, "ingress SRC-VC buffer underflow", LW_TRUE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_NXBAR_TILE, _ERR_STATUS, _EGRESS_CREDIT_OVERFLOW, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_NXBAR_TILE_EGRESS_CREDIT_OVERFLOW, "egress DST-VC credit overflow", LW_TRUE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_NXBAR_TILE, _ERR_STATUS, _EGRESS_CREDIT_UNDERFLOW, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_NXBAR_TILE_EGRESS_CREDIT_UNDERFLOW, "egress DST-VC credit underflow", LW_TRUE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_NXBAR_TILE, _ERR_STATUS, _INGRESS_NON_BURSTY_PKT, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_NXBAR_TILE_INGRESS_NON_BURSTY_PKT, "ingress packet burst error", LW_TRUE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_NXBAR_TILE, _ERR_STATUS, _INGRESS_NON_STICKY_PKT, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_NXBAR_TILE_INGRESS_NON_STICKY_PKT, "ingress packet sticky error", LW_TRUE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_NXBAR_TILE, _ERR_STATUS, _INGRESS_BURST_GT_9_DATA_VC, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_NXBAR_TILE_INGRESS_BURST_GT_9_DATA_VC, "possible bubbles at ingress", LW_TRUE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_NXBAR_TILE, _ERR_STATUS, _INGRESS_PKT_ILWALID_DST, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_NXBAR_TILE_INGRESS_PKT_ILWALID_DST, "ingress packet invalid dst error", LW_TRUE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_NXBAR_TILE, _ERR_STATUS, _INGRESS_PKT_PARITY_ERROR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_NXBAR_TILE_INGRESS_PKT_PARITY_ERROR, "ingress packet parity error", LW_TRUE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    if (report.raw_first & report.mask)
    {
        LWSWITCH_TILE_WR32_LR10(device, link, _NXBAR_TILE, _ERR_FIRST,
            report.raw_first & report.mask);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    // This helps prevent an interrupt storm if HW keeps triggering unnecessary stream of interrupts.
    LWSWITCH_TILE_WR32_LR10(device, link, _NXBAR_TILE, _ERR_FATAL_INTR_EN,
                            report.raw_enable ^ pending);

    LWSWITCH_TILE_WR32_LR10(device, link, _NXBAR_TILE, _ERR_STATUS, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_nxbar_tileout_lr10
(
    lwswitch_device *device,
    LwU32 link,
    LwU32 tileout
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LwU32 pending, bit, unhandled;
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };

    report.raw_pending = LWSWITCH_NXBAR_RD32_LR10(device, link, _NXBAR_TC_TILEOUT, _ERR_STATUS(tileout));
    report.raw_enable = LWSWITCH_NXBAR_RD32_LR10(device, link, _NXBAR_TC_TILEOUT, _ERR_FATAL_INTR_EN(tileout));
    report.mask = chip_device->intr_mask.tileout.fatal;
    report.data[0] = tileout;
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;
    report.raw_first = LWSWITCH_NXBAR_RD32_LR10(device, link, _NXBAR_TC_TILEOUT, _ERR_FIRST(tileout));

    bit = DRF_NUM(_NXBAR, _TC_TILEOUT0_ERR_STATUS, _INGRESS_BUFFER_OVERFLOW, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_NXBAR_TILEOUT_INGRESS_BUFFER_OVERFLOW, "ingress SRC-VC buffer overflow", LW_TRUE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_NXBAR, _TC_TILEOUT0_ERR_STATUS, _INGRESS_BUFFER_UNDERFLOW, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_NXBAR_TILEOUT_INGRESS_BUFFER_UNDERFLOW, "ingress SRC-VC buffer underflow", LW_TRUE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_NXBAR, _TC_TILEOUT0_ERR_STATUS, _EGRESS_CREDIT_OVERFLOW, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_NXBAR_TILEOUT_EGRESS_CREDIT_OVERFLOW, "egress DST-VC credit overflow", LW_TRUE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_NXBAR, _TC_TILEOUT0_ERR_STATUS, _EGRESS_CREDIT_UNDERFLOW, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_NXBAR_TILEOUT_EGRESS_CREDIT_UNDERFLOW, "egress DST-VC credit underflow", LW_TRUE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_NXBAR, _TC_TILEOUT0_ERR_STATUS, _INGRESS_NON_BURSTY_PKT, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_NXBAR_TILEOUT_INGRESS_NON_BURSTY_PKT, "ingress packet burst error", LW_TRUE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_NXBAR, _TC_TILEOUT0_ERR_STATUS, _INGRESS_NON_STICKY_PKT, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_NXBAR_TILEOUT_INGRESS_NON_STICKY_PKT, "ingress packet sticky error", LW_TRUE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_NXBAR, _TC_TILEOUT0_ERR_STATUS, _INGRESS_BURST_GT_9_DATA_VC, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_NXBAR_TILEOUT_INGRESS_BURST_GT_9_DATA_VC, "possible bubbles at ingress", LW_TRUE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_NXBAR, _TC_TILEOUT0_ERR_STATUS, _EGRESS_CDT_PARITY_ERROR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_NXBAR_TILEOUT_EGRESS_CDT_PARITY_ERROR, "ingress credit parity error", LW_TRUE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    if (report.raw_first & report.mask)
    {
        LWSWITCH_NXBAR_WR32_LR10(device, link, _NXBAR_TC_TILEOUT, _ERR_FIRST(tileout),
            report.raw_first & report.mask);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    // This helps prevent an interrupt storm if HW keeps triggering unnecessary stream of interrupts.
    LWSWITCH_NXBAR_WR32_LR10(device, link, _NXBAR_TC_TILEOUT, _ERR_FATAL_INTR_EN(tileout),
                            report.raw_enable ^ pending);

    LWSWITCH_NXBAR_WR32_LR10(device, link, _NXBAR_TC_TILEOUT, _ERR_STATUS(tileout), pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_nxbar_fatal_lr10
(
    lwswitch_device *device,
    LwU32 nxbar
)
{
    LwU32 pending, bit, unhandled;
    LwU32 link;
    LwU32 tile, tileout;

    pending = LWSWITCH_NXBAR_RD32_LR10(device, nxbar, _NXBAR, _TC_ERROR_STATUS);
    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;

    for (tile = 0; tile < NUM_NXBAR_TILES_PER_TC_LR10; tile++)
    {
        bit = DRF_NUM(_NXBAR, _TC_ERROR_STATUS, _TILE0, 1) << tile;
        if (lwswitch_test_flags(pending, bit))
        {
            link = TILE_TO_LINK(device, nxbar, tile);
            if (LWSWITCH_ENG_VALID_LR10(device, TILE, link))
            {
                if (_lwswitch_service_nxbar_tile_lr10(device, link) == LWL_SUCCESS)
                {
                    lwswitch_clear_flags(&unhandled, bit);
                }
            }
        }
    }

    for (tileout = 0; tileout < NUM_NXBAR_TILEOUTS_PER_TC_LR10; tileout++)
    {
        bit = DRF_NUM(_NXBAR, _TC_ERROR_STATUS, _TILEOUT0, 1) << tileout;
        if (lwswitch_test_flags(pending, bit))
        {
            if (_lwswitch_service_nxbar_tileout_lr10(device, nxbar, tileout) == LWL_SUCCESS)
            {
                lwswitch_clear_flags(&unhandled, bit);
            }
        }
    }

    // TODO: Perform hot_reset to recover NXBAR

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);


    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

LwlStatus
_lwswitch_service_minion_fatal_lr10
(
    lwswitch_device *device,
    LwU32 instance
)
{
    LwU32 pending, bit, unhandled, mask;

    pending = LWSWITCH_MINION_RD32_LR10(device, instance, _MINION, _MINION_INTR);
    mask =  LWSWITCH_MINION_RD32_LR10(device, instance, _MINION, _MINION_INTR_STALL_EN);

    // Don't consider MINION Link interrupts in this handler
    mask &= ~(DRF_NUM(_MINION, _MINION_INTR_STALL_EN, _LINK, LW_MINION_MINION_INTR_STALL_EN_LINK_ENABLE_ALL));

    pending &= mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending; 

    bit = DRF_NUM(_MINION, _MINION_INTR, _FALCON_STALL, 0x1);
    if (lwswitch_test_flags(pending, bit))
    {
        if (lwswitch_minion_service_falcon_interrupts_lr10(device, instance) == LWL_SUCCESS)
        {
            lwswitch_clear_flags(&unhandled, bit);
        }
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_service_lwldl_fatal_link_lr10
(
    lwswitch_device *device,
    LwU32 lwlipt_instance,
    LwU32 link
)
{
    LwU32 pending, bit, unhandled;
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    INFOROM_LWLINK_ERROR_EVENT error_event;
#endif

    report.raw_pending = LWSWITCH_LINK_RD32_LR10(device, link, LWLDL, _LWLDL_TOP, _INTR);
    report.raw_enable = LWSWITCH_LINK_RD32_LR10(device, link, LWLDL, _LWLDL_TOP, _INTR_STALL_EN);
    report.mask = report.raw_enable;
    pending = report.raw_pending & report.mask;
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

    error_event.lwliptInstance = (LwU8) lwlipt_instance;
    error_event.localLinkIdx   = (LwU8) LWSWITCH_LWLIPT_GET_LOCAL_LINK_ID_LR10(link);
#endif

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _TX_FAULT_RAM, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_TX_FAULT_RAM, "TX Fault Ram", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        error_event.error = INFOROM_LWLINK_DL_TX_FAULT_RAM_FATAL;
        lwswitch_inforom_lwlink_log_error_event(device, &error_event);
#endif
    }

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _TX_FAULT_INTERFACE, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_TX_FAULT_INTERFACE, "TX Fault Interface", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        error_event.error = INFOROM_LWLINK_DL_TX_FAULT_INTERFACE_FATAL;
        lwswitch_inforom_lwlink_log_error_event(device, &error_event);
#endif
    }

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _TX_FAULT_SUBLINK_CHANGE, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_TX_FAULT_SUBLINK_CHANGE, "TX Fault Sublink Change", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        error_event.error = INFOROM_LWLINK_DL_TX_FAULT_SUBLINK_CHANGE_FATAL;
        lwswitch_inforom_lwlink_log_error_event(device, &error_event);
#endif
    }

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _RX_FAULT_SUBLINK_CHANGE, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_RX_FAULT_SUBLINK_CHANGE, "RX Fault Sublink Change", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        error_event.error = INFOROM_LWLINK_DL_RX_FAULT_SUBLINK_CHANGE_FATAL;
        lwswitch_inforom_lwlink_log_error_event(device, &error_event);
#endif
    }

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _RX_FAULT_DL_PROTOCOL, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_RX_FAULT_DL_PROTOCOL, "RX Fault DL Protocol", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        error_event.error = INFOROM_LWLINK_DL_RX_FAULT_DL_PROTOCOL_FATAL;
        lwswitch_inforom_lwlink_log_error_event(device, &error_event);
#endif
    }

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _LTSSM_FAULT_DOWN, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_LTSSM_FAULT_DOWN, "LTSSM Fault Down", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        error_event.error = INFOROM_LWLINK_DL_LTSSM_FAULT_DOWN_FATAL;
        lwswitch_inforom_lwlink_log_error_event(device, &error_event);
#endif
    }

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _LTSSM_FAULT_UP, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_LTSSM_FAULT_UP, "LTSSM Fault Up", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        error_event.error = INFOROM_LWLINK_DL_LTSSM_FAULT_UP_FATAL;
        lwswitch_inforom_lwlink_log_error_event(device, &error_event);
#endif
    }

    bit = DRF_NUM(_LWLDL_TOP, _INTR, _LTSSM_PROTOCOL, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_LTSSM_PROTOCOL, "LTSSM Protocol Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);

        // TODO 2827793 this should be logged to the InfoROM as fatal
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_LINK_WR32_LR10(device, link, LWLDL, _LWLDL_TOP, _INTR_STALL_EN,
                report.raw_enable ^ pending);
    }

    LWSWITCH_LINK_WR32_LR10(device, link, LWLDL, _LWLDL_TOP, _INTR, pending);

    if (unhandled != 0)
    {
        LWSWITCH_PRINT(device, WARN,
                "%s: Unhandled LWLDL fatal interrupts, link: %d pending: 0x%x enabled: 0x%x.\n",
                 __FUNCTION__, link, pending, report.raw_enable);
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

LwlStatus
_lwswitch_service_lwldl_fatal_lr10
(
    lwswitch_device *device,
    LwU32 lwlipt_instance
)
{
    LwU64 enabledLinkMask, localLinkMask, localEnabledLinkMask, runtimeErrorMask = 0;
    LwU32 i;
    lwlink_link *link;
    LwlStatus status = -LWL_MORE_PROCESSING_REQUIRED;
    LWSWITCH_LINK_TRAINING_ERROR_INFO linkTrainingErrorInfo = { 0 };
    LWSWITCH_LINK_RUNTIME_ERROR_INFO linkRuntimeErrorInfo = { 0 };

    enabledLinkMask = lwswitch_get_enabled_link_mask(device);
    localLinkMask = LWSWITCH_LWLIPT_GET_LOCAL_LINK_MASK64(lwlipt_instance);
    localEnabledLinkMask = enabledLinkMask & localLinkMask;

    FOR_EACH_INDEX_IN_MASK(64, i, localEnabledLinkMask)
    {
        link = lwswitch_get_link(device, i);
        if (link == NULL)
        {
            // An interrupt on an invalid link should never occur
            LWSWITCH_ASSERT(link != NULL);
            continue;
        }

        if (LWSWITCH_GET_LINK_ENG_INST(device, i, LWLIPT) != lwlipt_instance)
        {
            LWSWITCH_ASSERT(0);
            break;
        }

        if (lwswitch_is_link_in_reset(device, link))
        {
            continue;
        }

        if (device->hal.lwswitch_service_lwldl_fatal_link(device, lwlipt_instance, i) == LWL_SUCCESS)
        {
            runtimeErrorMask |= LWBIT64(i);
            status = LWL_SUCCESS;
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    linkTrainingErrorInfo.isValid = LW_FALSE;
    linkRuntimeErrorInfo.isValid  = LW_TRUE;
    linkRuntimeErrorInfo.mask0    = runtimeErrorMask;

    if (lwswitch_smbpbi_set_link_error_info(device, &linkTrainingErrorInfo, &linkRuntimeErrorInfo) !=
        LWL_SUCCESS)
    {
        LWSWITCH_PRINT(device, ERROR,
                       "%s: Unable to send Runtime Error bitmask: 0x%llx,\n",
                       __FUNCTION__, runtimeErrorMask);
    }

    return status;
}

static LwlStatus
_lwswitch_service_lwltlc_tx_sys_fatal_lr10
(
    lwswitch_device *device,
    LwU32 lwlipt_instance,
    LwU32 link
)
{
    LwU32 pending, bit, unhandled;
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    INFOROM_LWLINK_ERROR_EVENT error_event = { 0 };
    LwU32 injected;
#endif

    report.raw_pending = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_TX_SYS, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_TX_SYS, _ERR_FATAL_REPORT_EN_0);
    report.mask = report.raw_enable;
    pending = report.raw_pending & report.mask;
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    error_event.lwliptInstance = (LwU8) lwlipt_instance;
    error_event.localLinkIdx   = (LwU8) LWSWITCH_LWLIPT_GET_LOCAL_LINK_ID_LR10(link);
#endif

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;
    report.raw_first = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_TX_SYS, _ERR_FIRST_0);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    injected = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_TX_SYS, _ERR_REPORT_INJECT_0);
#endif

    bit = DRF_NUM(_LWLTLC_TX_SYS, _ERR_STATUS_0, _NCISOC_PARITY_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_TX_SYS_NCISOC_PARITY_ERR, "NCISOC Parity Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        if (FLD_TEST_DRF_NUM(_LWLTLC, _TX_SYS_ERR_REPORT_INJECT_0, _NCISOC_PARITY_ERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_TX_NCISOC_PARITY_ERR_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_TX_SYS, _ERR_STATUS_0, _NCISOC_HDR_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_TX_SYS_NCISOC_HDR_ECC_DBE_ERR, "NCISOC HDR ECC DBE Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        if (FLD_TEST_DRF_NUM(_LWLTLC, _TX_SYS_ERR_REPORT_INJECT_0, _NCISOC_HDR_ECC_DBE_ERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_TX_NCISOC_HDR_ECC_DBE_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_TX_SYS, _ERR_STATUS_0, _NCISOC_DAT_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_TX_SYS_NCISOC_DAT_ECC_DBE_ERR, "NCISOC DAT ECC DBE Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_LWLTLC_TX_SYS, _ERR_STATUS_0, _NCISOC_ECC_LIMIT_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_TX_SYS_NCISOC_ECC_LIMIT_ERR, "NCISOC ECC Limit Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_LWLTLC_TX_SYS, _ERR_STATUS_0, _TXPOISONDET, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_TXPOISONDET, "Poison Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_LWLTLC_TX_SYS, _ERR_STATUS_0, _TXRSPSTATUS_HW_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_TX_SYS_TXRSPSTATUS_HW_ERR, "TX Response Status HW Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_LWLTLC_TX_SYS, _ERR_STATUS_0, _TXRSPSTATUS_UR_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_TX_SYS_TXRSPSTATUS_UR_ERR, "TX Response Status UR Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_LWLTLC_TX_SYS, _ERR_STATUS_0, _TXRSPSTATUS_PRIV_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_TX_SYS_TXRSPSTATUS_PRIV_ERR, "TX Response Status PRIV Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    if (report.raw_first & report.mask)
    {
        LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_TX_SYS, _ERR_FIRST_0,
            report.raw_first & report.mask);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_TX_SYS, _ERR_FATAL_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_TX_SYS, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        LWSWITCH_PRINT(device, WARN,
                "%s: Unhandled LWLTLC_TX_SYS interrupts, link: %d pending: 0x%x enabled: 0x%x.\n",
                 __FUNCTION__, link, pending, report.raw_enable);
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_lwltlc_rx_sys_fatal_lr10
(
    lwswitch_device *device,
    LwU32 lwlipt_instance,
    LwU32 link
)
{
    LwU32 pending, bit, unhandled;
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    INFOROM_LWLINK_ERROR_EVENT error_event = { 0 };
    LwU32 injected;
#endif

    report.raw_pending = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_RX_SYS, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_RX_SYS, _ERR_FATAL_REPORT_EN_0);
    report.mask = report.raw_enable;
    pending = report.raw_pending & report.mask;
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    error_event.lwliptInstance = (LwU8) lwlipt_instance;
    error_event.localLinkIdx   = (LwU8) LWSWITCH_LWLIPT_GET_LOCAL_LINK_ID_LR10(link);
#endif

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;
    report.raw_first = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_RX_SYS, _ERR_FIRST_0);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    injected = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_RX_SYS, _ERR_REPORT_INJECT_0);
#endif

    bit = DRF_NUM(_LWLTLC_RX_SYS, _ERR_STATUS_0, _NCISOC_PARITY_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RX_SYS_NCISOC_PARITY_ERR, "NCISOC Parity Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_LWLTLC_RX_SYS, _ERR_STATUS_0, _HDR_RAM_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RX_SYS_HDR_RAM_ECC_DBE_ERR, "HDR RAM ECC DBE Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
        if (FLD_TEST_DRF_NUM(_LWLTLC_RX_SYS, _ERR_REPORT_INJECT_0, _HDR_RAM_ECC_DBE_ERR, 0x0, injected))
        {
            // TODO 3014908 log these in the LWL object until we have ECC object support
            error_event.error = INFOROM_LWLINK_TLC_RX_HDR_RAM_ECC_DBE_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_RX_SYS, _ERR_STATUS_0, _HDR_RAM_ECC_LIMIT_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RX_SYS_HDR_RAM_ECC_LIMIT_ERR, "HDR RAM ECC Limit Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_LWLTLC_RX_SYS, _ERR_STATUS_0, _DAT0_RAM_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RX_SYS_DAT0_RAM_ECC_DBE_ERR, "DAT0 RAM ECC DBE Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC_RX_SYS, _ERR_REPORT_INJECT_0, _DAT0_RAM_ECC_DBE_ERR, 0x0, injected))
        {
            // TODO 3014908 log these in the LWL object until we have ECC object support
            error_event.error = INFOROM_LWLINK_TLC_RX_DAT0_RAM_ECC_DBE_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_RX_SYS, _ERR_STATUS_0, _DAT0_RAM_ECC_LIMIT_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RX_SYS_DAT0_RAM_ECC_LIMIT_ERR, "DAT0 RAM ECC Limit Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_LWLTLC_RX_SYS, _ERR_STATUS_0, _DAT1_RAM_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RX_SYS_DAT1_RAM_ECC_DBE_ERR, "DAT1 RAM ECC DBE Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC_RX_SYS, _ERR_REPORT_INJECT_0, _DAT1_RAM_ECC_DBE_ERR, 0x0, injected))
        {
            // TODO 3014908 log these in the LWL object until we have ECC object support
            error_event.error = INFOROM_LWLINK_TLC_RX_DAT1_RAM_ECC_DBE_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_RX_SYS, _ERR_STATUS_0, _DAT1_RAM_ECC_LIMIT_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RX_SYS_DAT1_RAM_ECC_LIMIT_ERR, "DAT1 RAM ECC Limit Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    if (report.raw_first & report.mask)
    {
        LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_RX_SYS, _ERR_FIRST_0,
            report.raw_first & report.mask);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_RX_SYS, _ERR_FATAL_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_RX_SYS, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        LWSWITCH_PRINT(device, WARN,
                "%s: Unhandled LWLTLC_RX_SYS interrupts, link: %d pending: 0x%x enabled: 0x%x.\n",
                 __FUNCTION__, link, pending, report.raw_enable);
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_lwltlc_tx_lnk_fatal_0_lr10
(
    lwswitch_device *device,
    LwU32 lwlipt_instance,
    LwU32 link
)
{
    LwU32 pending, bit, unhandled;
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    INFOROM_LWLINK_ERROR_EVENT error_event = { 0 };
    LwU32 injected;
#endif

    report.raw_pending = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_TX_LNK, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_TX_LNK, _ERR_FATAL_REPORT_EN_0);
    report.mask = report.raw_enable;
    pending = report.raw_pending & report.mask;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    error_event.lwliptInstance = (LwU8) lwlipt_instance;
    error_event.localLinkIdx   = (LwU8) LWSWITCH_LWLIPT_GET_LOCAL_LINK_ID_LR10(link);
#endif
    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;
    report.raw_first = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_TX_LNK, _ERR_FIRST_0);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    injected = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_TX_LNK, _ERR_REPORT_INJECT_0);
#endif

    bit = DRF_NUM(_LWLTLC_TX_LNK, _ERR_STATUS_0, _TXDLCREDITPARITYERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_TXDLCREDITPARITYERR, "TX DL Credit Parity Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _TX_LNK_ERR_REPORT_INJECT_0, _TXDLCREDITPARITYERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_TX_DL_CREDIT_PARITY_ERR_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_TX_LNK, _ERR_STATUS_0, _CREQ_RAM_HDR_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_TX_LNK_CREQ_RAM_HDR_ECC_DBE_ERR, "CREQ RAM HDR ECC DBE Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_LWLTLC_TX_LNK, _ERR_STATUS_0, _RSP_RAM_HDR_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_TX_LNK_RSP_RAM_HDR_ECC_DBE_ERR, "Response RAM HDR ECC DBE Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_LWLTLC_TX_LNK, _ERR_STATUS_0, _COM_RAM_HDR_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_TX_LNK_COM_RAM_HDR_ECC_DBE_ERR, "COM RAM HDR ECC DBE Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_LWLTLC_TX_LNK, _ERR_STATUS_0, _RSP1_RAM_HDR_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_TX_LNK_RSP1_RAM_HDR_ECC_DBE_ERR, "RSP1 RAM HDR ECC DBE Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    bit = DRF_NUM(_LWLTLC_TX_LNK, _ERR_STATUS_0, _RSP1_RAM_DAT_ECC_DBE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_TX_LNK_RSP1_RAM_DAT_ECC_DBE_ERR, "RSP1 RAM DAT ECC DBE Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC_TX_LNK, _ERR_REPORT_INJECT_0, _RSP1_RAM_DAT_ECC_DBE_ERR, 0x0, injected))
        {
            // TODO 3014908 log these in the LWL object until we have ECC object support
            error_event.error = INFOROM_LWLINK_TLC_TX_RSP1_DAT_RAM_ECC_DBE_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_TX_LNK, _ERR_FATAL_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    if (report.raw_first & report.mask)
    {
        LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_TX_LNK, _ERR_FIRST_0,
                report.raw_first & report.mask);
    }
    LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_TX_LNK, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        LWSWITCH_PRINT(device, WARN,
                "%s: Unhandled LWLTLC_TX_LNK _0 interrupts, link: %d pending: 0x%x enabled: 0x%x.\n",
                 __FUNCTION__, link, pending, report.raw_enable);
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_lwltlc_rx_lnk_fatal_0_lr10
(
    lwswitch_device *device,
    LwU32 lwlipt_instance,
    LwU32 link
)
{
    LwU32 pending, bit, unhandled;
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    INFOROM_LWLINK_ERROR_EVENT error_event = { 0 };
    LwU32 injected;
#endif

    report.raw_pending = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_FATAL_REPORT_EN_0);
    report.mask = report.raw_enable;
    pending = report.raw_pending & report.mask;
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

    error_event.lwliptInstance = (LwU8) lwlipt_instance;
    error_event.localLinkIdx   = (LwU8) LWSWITCH_LWLIPT_GET_LOCAL_LINK_ID_LR10(link);
#endif
    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;
    report.raw_first = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_FIRST_0);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    injected = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_REPORT_INJECT_0);
#endif

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_0, _RXDLHDRPARITYERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RXDLHDRPARITYERR, "RX DL HDR Parity Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _RX_LNK_ERR_REPORT_INJECT_0, _RXDLHDRPARITYERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_RX_DL_HDR_PARITY_ERR_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_0, _RXDLDATAPARITYERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RXDLDATAPARITYERR, "RX DL Data Parity Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _RX_LNK_ERR_REPORT_INJECT_0, _RXDLDATAPARITYERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_RX_DL_DATA_PARITY_ERR_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_0, _RXDLCTRLPARITYERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RXDLCTRLPARITYERR, "RX DL Ctrl Parity Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _RX_LNK_ERR_REPORT_INJECT_0, _RXDLCTRLPARITYERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_RX_DL_CTRL_PARITY_ERR_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_0, _RXILWALIDAEERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RXILWALIDAEERR, "RX Invalid DAE Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _RX_LNK_ERR_REPORT_INJECT_0, _RXILWALIDAEERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_RX_ILWALID_AE_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_0, _RXILWALIDBEERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RXILWALIDBEERR, "RX Invalid BE Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _RX_LNK_ERR_REPORT_INJECT_0, _RXILWALIDBEERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_RX_ILWALID_BE_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_0, _RXILWALIDADDRALIGNERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RXILWALIDADDRALIGNERR, "RX Invalid Addr Align Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _RX_LNK_ERR_REPORT_INJECT_0, _RXILWALIDADDRALIGNERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_RX_ILWALID_ADDR_ALIGN_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_0, _RXPKTLENERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RXPKTLENERR, "RX Packet Length Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _RX_LNK_ERR_REPORT_INJECT_0, _RXPKTLENERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_RX_PKTLEN_ERR_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_0, _RSVCMDENCERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RSVCMDENCERR, "RSV Cmd Encoding Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _RX_LNK_ERR_REPORT_INJECT_0, _RSVCMDENCERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_RX_RSVD_CMD_ENC_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_0, _RSVDATLENENCERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RSVDATLENENCERR, "RSV Data Length Encoding Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _RX_LNK_ERR_REPORT_INJECT_0, _RSVDATLENENCERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_RX_RSVD_DAT_LEN_ENC_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_0, _RSVPKTSTATUSERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RSVPKTSTATUSERR, "RSV Packet Status Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _RX_LNK_ERR_REPORT_INJECT_0, _RSVPKTSTATUSERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_RX_RSVD_PACKET_STATUS_ERR_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_0, _RSVCACHEATTRPROBEREQERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RSVCACHEATTRPROBEREQERR, "RSV Packet Status Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _RX_LNK_ERR_REPORT_INJECT_0, _RSVCACHEATTRPROBEREQERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_RX_RSVD_CACHE_ATTR_PROBE_REQ_ERR_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_0, _RSVCACHEATTRPROBERSPERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RSVCACHEATTRPROBERSPERR, "RSV CacheAttr Probe Rsp Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _RX_LNK_ERR_REPORT_INJECT_0, _RSVCACHEATTRPROBERSPERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_RX_RSVD_CACHE_ATTR_PROBE_RSP_ERR_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_0, _DATLENGTRMWREQMAXERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_DATLENGTRMWREQMAXERR, "Data Length RMW Req Max Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _RX_LNK_ERR_REPORT_INJECT_0, _DATLENGTRMWREQMAXERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_RX_DATLEN_GT_RMW_REQ_MAX_ERR_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_0, _DATLENLTATRRSPMINERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_DATLENLTATRRSPMINERR, "Data Len Lt ATR RSP Min Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _RX_LNK_ERR_REPORT_INJECT_0, _DATLENLTATRRSPMINERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_RX_DATLEN_LT_ATR_RSP_MIN_ERR_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_0, _ILWALIDCACHEATTRPOERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_ILWALIDCACHEATTRPOERR, "Invalid Cache Attr PO Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _RX_LNK_ERR_REPORT_INJECT_0, _ILWALIDCACHEATTRPOERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_RX_ILWALID_PO_FOR_CACHE_ATTR_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_0, _ILWALIDCRERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_ILWALIDCRERR, "Invalid CR Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _RX_LNK_ERR_REPORT_INJECT_0, _ILWALIDCRERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_RX_ILWALID_CR_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_0, _RXRSPSTATUS_HW_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RX_LNK_RXRSPSTATUS_HW_ERR, "RX Rsp Status HW Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        // TODO 200564153 _RX_RSPSTATUS_HW_ERR should be reported as non-fatal
        if (FLD_TEST_DRF_NUM(_LWLTLC, _RX_LNK_ERR_REPORT_INJECT_0, _RXRSPSTATUS_HW_ERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_RX_RSP_STATUS_HW_ERR_NONFATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_0, _RXRSPSTATUS_UR_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RX_LNK_RXRSPSTATUS_UR_ERR, "RX Rsp Status UR Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        // TODO 200564153 _RX_RSPSTATUS_UR_ERR should be reported as non-fatal
        if (FLD_TEST_DRF_NUM(_LWLTLC, _RX_LNK_ERR_REPORT_INJECT_0, _RXRSPSTATUS_UR_ERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_RX_RSP_STATUS_UR_ERR_NONFATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_0, _ILWALID_COLLAPSED_RESPONSE_ERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RX_LNK_ILWALID_COLLAPSED_RESPONSE_ERR, "Invalid Collapsed Response Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _RX_LNK_ERR_REPORT_INJECT_0, _ILWALID_COLLAPSED_RESPONSE_ERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_RX_ILWALID_COLLAPSED_RESPONSE_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_FATAL_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    if (report.raw_first & report.mask)
    {
        LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_FIRST_0,
                report.raw_first & report.mask);
    }
    LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        LWSWITCH_PRINT(device, WARN,
                "%s: Unhandled LWLTLC_RX_LNK _0 interrupts, link: %d pending: 0x%x enabled: 0x%x.\n",
                 __FUNCTION__, link, pending, report.raw_enable);
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_lwltlc_rx_lnk_fatal_1_lr10
(
    lwswitch_device *device,
    LwU32 lwlipt_instance,
    LwU32 link
)
{
    LwU32 pending, bit, unhandled;
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    INFOROM_LWLINK_ERROR_EVENT error_event = { 0 };
    LwU32 injected;
#endif

    report.raw_pending = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_STATUS_1);
    report.raw_enable = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_FATAL_REPORT_EN_1);
    report.mask = report.raw_enable;
    pending = report.raw_pending & report.mask;
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

    error_event.lwliptInstance = (LwU8) lwlipt_instance;
    error_event.localLinkIdx   = (LwU8) LWSWITCH_LWLIPT_GET_LOCAL_LINK_ID_LR10(link);
#endif

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;
    report.raw_first = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_FIRST_1);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    injected = LWSWITCH_LINK_RD32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_REPORT_INJECT_1);
#endif

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_1, _RXHDROVFERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RXHDROVFERR, "RX HDR OVF Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _RX_LNK_ERR_REPORT_INJECT_1, _RXHDROVFERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_RX_HDR_OVERFLOW_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_1, _RXDATAOVFERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RXDATAOVFERR, "RX Data OVF Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _RX_LNK_ERR_REPORT_INJECT_1, _RXDATAOVFERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_RX_DATA_OVERFLOW_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_1, _STOMPDETERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_STOMPDETERR, "Stomp Det Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLTLC, _RX_LNK_ERR_REPORT_INJECT_1, _STOMPDETERR, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_TLC_RX_STOMP_DETECTED_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLTLC_RX_LNK, _ERR_STATUS_1, _RXPOISONERR, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLTLC_RXPOISONERR, "RX Poison Error", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_FATAL_REPORT_EN_1,
                report.raw_enable ^ pending);
    }

    if (report.raw_first & report.mask)
    {
        LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_FIRST_1,
                report.raw_first & report.mask);
    }
    LWSWITCH_LINK_WR32_LR10(device, link, LWLTLC, _LWLTLC_RX_LNK, _ERR_STATUS_1, pending);

    if (unhandled != 0)
    {
        LWSWITCH_PRINT(device, WARN,
                "%s: Unhandled LWLTLC_RX_LNK _1 interrupts, link: %d pending: 0x%x enabled: 0x%x.\n",
                 __FUNCTION__, link, pending, report.raw_enable);
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

LwlStatus
_lwswitch_service_lwltlc_fatal_lr10
(
    lwswitch_device *device,
    LwU32 lwlipt_instance
)
{
    LwU64 enabledLinkMask, localLinkMask, localEnabledLinkMask;
    LwU32 i;
    lwlink_link *link;
    LwlStatus status = -LWL_MORE_PROCESSING_REQUIRED;

    enabledLinkMask = lwswitch_get_enabled_link_mask(device);
    localLinkMask = LWSWITCH_LWLIPT_GET_LOCAL_LINK_MASK64(lwlipt_instance);
    localEnabledLinkMask = enabledLinkMask & localLinkMask;

    FOR_EACH_INDEX_IN_MASK(64, i, localEnabledLinkMask)
    {
        link = lwswitch_get_link(device, i);
        if (link == NULL)
        {
            // An interrupt on an invalid link should never occur
            LWSWITCH_ASSERT(link != NULL);
            continue;
        }

        if (LWSWITCH_GET_LINK_ENG_INST(device, i, LWLIPT) != lwlipt_instance)
        {
            LWSWITCH_ASSERT(0);
            break;
        }

        if (lwswitch_is_link_in_reset(device, link))
        {
            continue;
        }

        if (_lwswitch_service_lwltlc_tx_sys_fatal_lr10(device, lwlipt_instance, i) == LWL_SUCCESS)
        {
            status = LWL_SUCCESS;
        }

        if (_lwswitch_service_lwltlc_rx_sys_fatal_lr10(device, lwlipt_instance, i) == LWL_SUCCESS)
        {
            status = LWL_SUCCESS;
        }

        if (_lwswitch_service_lwltlc_tx_lnk_fatal_0_lr10(device, lwlipt_instance, i) == LWL_SUCCESS)
        {
            status = LWL_SUCCESS;
        }

        if (_lwswitch_service_lwltlc_rx_lnk_fatal_0_lr10(device, lwlipt_instance, i) == LWL_SUCCESS)
        {
            status = LWL_SUCCESS;
        }

        if (_lwswitch_service_lwltlc_rx_lnk_fatal_1_lr10(device, lwlipt_instance, i) == LWL_SUCCESS)
        {
            status = LWL_SUCCESS;
        }

    }
    FOR_EACH_INDEX_IN_MASK_END;

    return status;
}

static LwlStatus
_lwswitch_service_lwlipt_common_fatal_lr10
(
    lwswitch_device *device,
    LwU32 instance
)
{
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, contain, unhandled;
    LwU32 link, local_link_idx;
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    INFOROM_LWLINK_ERROR_EVENT error_event = { 0 };
    LwU32 injected;
#endif

    report.raw_pending = LWSWITCH_LWLIPT_RD32_LR10(device, instance, _LWLIPT_COMMON, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_LWLIPT_RD32_LR10(device, instance, _LWLIPT_COMMON, _ERR_FATAL_REPORT_EN_0);
    report.mask = report.raw_enable &
        (DRF_NUM(_LWLIPT_COMMON, _ERR_STATUS_0, _CLKCTL_ILLEGAL_REQUEST, 1) |
            DRF_NUM(_LWLIPT_COMMON, _ERR_STATUS_0, _RSTSEQ_PLL_TIMEOUT, 1) |
            DRF_NUM(_LWLIPT_COMMON, _ERR_STATUS_0, _RSTSEQ_PHYARB_TIMEOUT, 1));

    pending = report.raw_pending & report.mask;
    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

    error_event.lwliptInstance = (LwU8) instance;
#endif

    unhandled = pending;
    report.raw_first = LWSWITCH_LWLIPT_RD32_LR10(device, instance, _LWLIPT_COMMON, _ERR_FIRST_0);
    contain = LWSWITCH_LWLIPT_RD32_LR10(device, instance, _LWLIPT_COMMON, _ERR_CONTAIN_EN_0);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    injected = LWSWITCH_LWLIPT_RD32_LR10(device, instance, _LWLIPT_COMMON, _ERR_REPORT_INJECT_0);
#endif

    bit = DRF_NUM(_LWLIPT_COMMON, _ERR_STATUS_0, _CLKCTL_ILLEGAL_REQUEST, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        for (local_link_idx = 0; local_link_idx < LWSWITCH_LINKS_PER_LWLIPT; local_link_idx++)
        {
            link = (instance * LWSWITCH_LINKS_PER_LWLIPT) + local_link_idx;
            if (lwswitch_is_link_valid(device, link))
            {
                LWSWITCH_REPORT_CONTAIN(_HW_LWLIPT_CLKCTL_ILLEGAL_REQUEST, "CLKCTL_ILLEGAL_REQUEST", LW_FALSE);
            }
        }

        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLIPT_COMMON, _ERR_REPORT_INJECT_0, _CLKCTL_ILLEGAL_REQUEST, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_LWLIPT_CLKCTL_ILLEGAL_REQUEST_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLIPT_COMMON, _ERR_STATUS_0, _RSTSEQ_PLL_TIMEOUT, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        for (local_link_idx = 0; local_link_idx < LWSWITCH_LINKS_PER_LWLIPT; local_link_idx++)
        {
            link = (instance * LWSWITCH_LINKS_PER_LWLIPT) + local_link_idx;
            if (lwswitch_is_link_valid(device, link))
            {
                LWSWITCH_REPORT_CONTAIN(_HW_LWLIPT_RSTSEQ_PLL_TIMEOUT, "RSTSEQ_PLL_TIMEOUT", LW_FALSE);
            }
        }

        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLIPT_COMMON, _ERR_REPORT_INJECT_0, _RSTSEQ_PLL_TIMEOUT, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_LWLIPT_RSTSEQ_PLL_TIMEOUT_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLIPT_COMMON, _ERR_STATUS_0, _RSTSEQ_PHYARB_TIMEOUT, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        for (local_link_idx = 0; local_link_idx < LWSWITCH_LINKS_PER_LWLIPT; local_link_idx++)
        {
            link = (instance * LWSWITCH_LINKS_PER_LWLIPT) + local_link_idx;
            if (lwswitch_is_link_valid(device, link))
            {
                LWSWITCH_REPORT_CONTAIN(_HW_LWLIPT_RSTSEQ_PHYARB_TIMEOUT, "RSTSEQ_PHYARB_TIMEOUT", LW_FALSE);
            }
        }

        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLIPT_COMMON, _ERR_REPORT_INJECT_0, _RSTSEQ_PHYARB_TIMEOUT, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_LWLIPT_RSTSEQ_PHYARB_TIMEOUT_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    for (local_link_idx = 0; local_link_idx < LWSWITCH_LINKS_PER_LWLIPT; local_link_idx++)
    {
        link = (instance * LWSWITCH_LINKS_PER_LWLIPT) + local_link_idx;
        if (lwswitch_is_link_valid(device, link) &&
            (device->link[link].fatal_error_oclwrred))
        {
            LWSWITCH_LWLIPT_WR32_LR10(device, instance, _LWLIPT_COMMON, _ERR_FATAL_REPORT_EN_0,
                report.raw_enable ^ pending);
            break;
        }
    }

    // clear the interrupts
    if (report.raw_first & report.mask)
    {
        LWSWITCH_LWLIPT_WR32_LR10(device, instance, _LWLIPT_COMMON, _ERR_FIRST_0,
            report.raw_first & report.mask);
    }
    LWSWITCH_LWLIPT_WR32_LR10(device, instance, _LWLIPT_COMMON, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        LWSWITCH_PRINT(device, WARN,
                "%s: Unhandled LWLIPT_COMMON FATAL interrupts, pending: 0x%x enabled: 0x%x.\n",
                 __FUNCTION__, pending, report.raw_enable);
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_lwlipt_lnk_fatal_lr10
(
    lwswitch_device *device,
    LwU32 lwlipt_instance,
    LwU32 link
)
{
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, unhandled;
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    INFOROM_LWLINK_ERROR_EVENT error_event = { 0 };
    LwU32 injected;
#endif

    report.raw_pending = LWSWITCH_LINK_RD32_LR10(device, link, LWLIPT_LNK, _LWLIPT_LNK, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_LINK_RD32_LR10(device, link, LWLIPT_LNK, _LWLIPT_LNK, _ERR_FATAL_REPORT_EN_0);
    report.mask = report.raw_enable;

    pending = report.raw_pending & report.mask;
    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

    error_event.lwliptInstance = (LwU8) lwlipt_instance;
    error_event.localLinkIdx   = (LwU8) LWSWITCH_LWLIPT_GET_LOCAL_LINK_ID_LR10(link);
#endif

    unhandled = pending;
    report.raw_first = LWSWITCH_LINK_RD32_LR10(device, link, LWLIPT_LNK, _LWLIPT_LNK, _ERR_FIRST_0);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
    injected = LWSWITCH_LINK_RD32_LR10(device, link, LWLIPT_LNK, _LWLIPT_LNK, _ERR_REPORT_INJECT_0);
#endif

    bit = DRF_NUM(_LWLIPT_LNK, _ERR_STATUS_0, _SLEEPWHILEACTIVELINK, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLIPT_LNK_SLEEPWHILEACTIVELINK, "No non-empty link is detected", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLIPT_LNK, _ERR_REPORT_INJECT_0, _SLEEPWHILEACTIVELINK, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_LWLIPT_SLEEP_WHILE_ACTIVE_LINK_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLIPT_LNK, _ERR_STATUS_0, _RSTSEQ_PHYCTL_TIMEOUT, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLIPT_LNK_RSTSEQ_PHYCTL_TIMEOUT, "Reset sequencer timed out waiting for a handshake from PHYCTL", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLIPT_LNK, _ERR_REPORT_INJECT_0, _RSTSEQ_PHYCTL_TIMEOUT, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_LWLIPT_RSTSEQ_PHYCTL_TIMEOUT_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    bit = DRF_NUM(_LWLIPT_LNK, _ERR_STATUS_0, _RSTSEQ_CLKCTL_TIMEOUT, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        LWSWITCH_REPORT_FATAL(_HW_LWLIPT_LNK_RSTSEQ_CLKCTL_TIMEOUT, "Reset sequencer timed out waiting for a handshake from CLKCTL", LW_FALSE);
        lwswitch_clear_flags(&unhandled, bit);
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

        if (FLD_TEST_DRF_NUM(_LWLIPT_LNK, _ERR_REPORT_INJECT_0, _RSTSEQ_CLKCTL_TIMEOUT, 0x0, injected))
        {
            error_event.error = INFOROM_LWLINK_LWLIPT_RSTSEQ_CLKCTL_TIMEOUT_FATAL;
            lwswitch_inforom_lwlink_log_error_event(device, &error_event);
        }
#endif
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_LINK_WR32_LR10(device, link, LWLIPT_LNK, _LWLIPT_LNK, _ERR_FATAL_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    // clear interrupts
    if (report.raw_first & report.mask)
    {
        LWSWITCH_LINK_WR32_LR10(device, link, LWLIPT_LNK, _LWLIPT_LNK, _ERR_FIRST_0,
                report.raw_first & report.mask);
    }
    LWSWITCH_LINK_WR32_LR10(device, link, LWLIPT_LNK, _LWLIPT_LNK, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        LWSWITCH_PRINT(device, WARN,
                "%s: Unhandled LWLIPT_LNK FATAL interrupts, pending: 0x%x enabled: 0x%x.\n",
                 __FUNCTION__, pending, report.raw_enable);
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_lwlipt_link_fatal_lr10
(
    lwswitch_device *device,
    LwU32 lwlipt_instance
)
{
    LwU32 i, intrLink;
    LwU64 enabledLinkMask, localLinkMask, localEnabledLinkMask, interruptingLinks = 0;

    enabledLinkMask = lwswitch_get_enabled_link_mask(device);
    localLinkMask = LWSWITCH_LWLIPT_GET_LOCAL_LINK_MASK64(lwlipt_instance);
    localEnabledLinkMask = enabledLinkMask & localLinkMask;

    FOR_EACH_INDEX_IN_MASK(64, i, localEnabledLinkMask)
    {
        intrLink = LWSWITCH_LINK_RD32_LR10(device, i, LWLIPT_LNK, _LWLIPT_LNK, _ERR_STATUS_0);

        if(intrLink)
        {
            interruptingLinks |= LWBIT(i);
        }
    }
    FOR_EACH_INDEX_IN_MASK_END;

    if(interruptingLinks)
    {
        FOR_EACH_INDEX_IN_MASK(64, i, interruptingLinks)
        {
            if( _lwswitch_service_lwlipt_lnk_fatal_lr10(device, lwlipt_instance, i) != LWL_SUCCESS)
            {
                return -LWL_MORE_PROCESSING_REQUIRED;
            }
        }
        FOR_EACH_INDEX_IN_MASK_END;
        return LWL_SUCCESS;
    }
    else
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }
}

static LwlStatus
_lwswitch_service_lwlipt_fatal_lr10
(
    lwswitch_device *device,
    LwU32 instance
)
{
    LwlStatus status[6];

    //
    // MINION LINK interrupts trigger both INTR_FATAL and INTR_NONFATAL
    // trees (Bug 3037835). Because of this, we must service them in both the
    // fatal and nonfatal handlers
    //
    status[0] = device->hal.lwswitch_service_minion_link(device, instance);
    status[1] = _lwswitch_service_lwldl_fatal_lr10(device, instance);
    status[2] = _lwswitch_service_lwltlc_fatal_lr10(device, instance);
    status[3] = _lwswitch_service_minion_fatal_lr10(device, instance);
    status[4] = _lwswitch_service_lwlipt_common_fatal_lr10(device, instance);
    status[5] = _lwswitch_service_lwlipt_link_fatal_lr10(device, instance);

    if (status[0] != LWL_SUCCESS &&
        status[1] != LWL_SUCCESS &&
        status[2] != LWL_SUCCESS &&
        status[3] != LWL_SUCCESS &&
        status[4] != LWL_SUCCESS &&
        status[5] != LWL_SUCCESS)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_saw_fatal_lr10
(
    lwswitch_device *device
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LwU32 pending, bit, unhandled;
    LwU32 i;

    pending = LWSWITCH_SAW_RD32_LR10(device, _LWLSAW_LWSPMC, _INTR_FATAL);
    pending &= chip_device->intr_enable_fatal;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;

    for (i = 0; i < NUM_NPG_ENGINE_LR10; i++)
    {
        if (!LWSWITCH_ENG_VALID_LR10(device, NPG, i))
        {
            continue;
        }

        bit = DRF_NUM(_LWLSAW_LWSPMC, _INTR_FATAL, _NPG_0, 1) << i;
        if (lwswitch_test_flags(pending, bit))
        {
            if (_lwswitch_service_npg_fatal_lr10(device, i) == LWL_SUCCESS)
            {
                lwswitch_clear_flags(&unhandled, bit);
            }
        }
    }

    for (i = 0; i < NUM_NXBAR_ENGINE_LR10; i++)
    {
        if (!LWSWITCH_ENG_VALID_LR10(device, NXBAR, i))
            continue;

        bit = DRF_NUM(_LWLSAW_LWSPMC, _INTR_FATAL, _NXBAR_0, 1) << i;
        if (lwswitch_test_flags(pending, bit))
        {
            if (_lwswitch_service_nxbar_fatal_lr10(device, i) == LWL_SUCCESS)
            {
                lwswitch_clear_flags(&unhandled, bit);
            }
        }
    }

    for (i = 0; i < NUM_LWLIPT_ENGINE_LR10; i++)
    {
        if (!LWSWITCH_ENG_VALID_LR10(device, LWLIPT, i))
        {
            continue;
        }

        bit = DRF_NUM(_LWLSAW_LWSPMC, _INTR_FATAL, _LWLIPT_0, 1) << i;
        if (lwswitch_test_flags(pending, bit))
        {
            if (_lwswitch_service_lwlipt_fatal_lr10(device, i) == LWL_SUCCESS)
            {
                lwswitch_clear_flags(&unhandled, bit);
            }
        }
    }

    if (LWSWITCH_ENG_VALID_LR10(device, SOE, 0))
    {
        bit = DRF_NUM(_LWLSAW_LWSPMC, _INTR_FATAL, _SOE, 1);
        if (lwswitch_test_flags(pending, bit))
        {
            if (_lwswitch_service_soe_fatal_lr10(device) == LWL_SUCCESS)
            {
                lwswitch_clear_flags(&unhandled, bit);
            }
        }
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_saw_lr10
(
    lwswitch_device *device
)
{
    LwlStatus status[4];

    status[0] = _lwswitch_service_saw_legacy_lr10(device);
    status[1] = _lwswitch_service_saw_fatal_lr10(device);
    status[2] = _lwswitch_service_saw_nonfatal_lr10(device);

    if ((status[0] != LWL_SUCCESS) &&
        (status[1] != LWL_SUCCESS) &&
        (status[2] != LWL_SUCCESS))
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_legacy_lr10
(
    lwswitch_device *device
)
{
    lr10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_LR10(device);
    LwU32 pending, bit, unhandled;

    pending = LWSWITCH_REG_RD32(device, _PSMC, _INTR_LEGACY);
    pending &= chip_device->intr_enable_legacy;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    unhandled = pending;

    bit = DRF_NUM(_PSMC, _INTR_LEGACY, _SAW, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        if (_lwswitch_service_saw_lr10(device) == LWL_SUCCESS)
        {
            lwswitch_clear_flags(&unhandled, bit);
        }
    }

    bit = DRF_NUM(_PSMC, _INTR_LEGACY, _PRIV_RING, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        if (_lwswitch_service_priv_ring_lr10(device) == LWL_SUCCESS)
        {
            lwswitch_clear_flags(&unhandled, bit);
        }
    }

    bit = DRF_NUM(_PSMC, _INTR_LEGACY, _PBUS, 1);
    if (lwswitch_test_flags(pending, bit))
    {
        if (_lwswitch_service_pbus_lr10(device) == LWL_SUCCESS)
        {
            lwswitch_clear_flags(&unhandled, bit);
        }
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

//
// Service interrupt and re-enable interrupts. Interrupts should disabled when
// this is called.
//
LwlStatus
lwswitch_lib_service_interrupts_lr10
(
    lwswitch_device *device
)
{
    LwlStatus status;

    status = _lwswitch_service_legacy_lr10(device);

    /// @todo remove LWL_NOT_FOUND from the condition below, it was added as a WAR until Bug 2856055 is fixed.
    if ((status != LWL_SUCCESS) && (status != -LWL_NOT_FOUND))
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    _lwswitch_rearm_msi_lr10(device);

    return LWL_SUCCESS;
}

