/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2016-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include "common_lwswitch.h"
#include "intr_lwswitch.h"
#include "sv10/sv10.h"
#include "sv10/minion_sv10.h"

#include "lwswitch/svnp01/dev_lws.h"
#include "lwswitch/svnp01/dev_lws_master.h"
#include "lwswitch/svnp01/dev_pri_ringmaster.h"
#include "lwswitch/svnp01/dev_pri_ringstation_sys.h"
#include "lwswitch/svnp01/dev_pri_ringstation_prt.h"
#include "lwswitch/svnp01/dev_timer.h"
#include "lwswitch/svnp01/dev_afs_ip.h"
#include "lwswitch/svnp01/dev_nport_ip.h"
#include "lwswitch/svnp01/dev_npg_ip.h"
#include "lwswitch/svnp01/dev_npg_ip_addendum.h"
#include "lwswitch/svnp01/dev_lwlipt_ip.h"
#include "lwswitch/svnp01/dev_lwlipt_ip_addendum.h"
#include "lwswitch/svnp01/dev_lwlctrl_ip.h"
#include "lwswitch/svnp01/dev_lwlctrl_ip_addendum.h"
#include "lwswitch/svnp01/dev_lwltlc_ip.h"
#include "lwswitch/svnp01/dev_lwltlc_ip_addendum.h"
#include "lwswitch/svnp01/dev_ingress_ip.h"
#include "lwswitch/svnp01/dev_egress_ip.h"
#include "lwswitch/svnp01/dev_route_ip.h"
#include "lwswitch/svnp01/dev_ftstate_ip.h"
#include "lwswitch/svnp01/dev_swx_ip.h"
#include "lwswitch/svnp01/dev_lwl_ip.h"
#include "lwswitch/svnp01/dev_lw_xve.h"
#include "lwswitch/svnp01/dev_minion_ip.h"

//
// Unit to link colwersion
//
#define NPORT_TO_LINK(_device, _npg, _nport) \
     LWSWITCH_GET_CHIP_DEVICE_SV10(_device)->subengNPG[_npg].subengNPORT[_nport].instance
#define SIOCTRL_TO_LINK(_device, _sioctrl, _link) \
     LWSWITCH_GET_CHIP_DEVICE_SV10(_device)->subengSIOCTRL[_sioctrl].subengDLPL[_link].instance
#define AFS_TO_LINK(_device, _swx, _afs) \
     LWSWITCH_GET_CHIP_DEVICE_SV10(_device)->subengSWX[_swx].subengAFS[_afs].instance

/*
 * @Brief : Enable top level HW interrupts.
 *
 * @Description :
 *
 * @param[in] device        operate on this device
 */
void
lwswitch_lib_enable_interrupts_sv10
(
    lwswitch_device *device
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    LWSWITCH_FLUSH_MMIO(device);

    LWSWITCH_REG_WR32(device, _PSMC, _INTR_EN_SET_LEGACY, chip_device->intr_enable_legacy);
    LWSWITCH_REG_WR32(device, _PSMC, _INTR_EN_SET_CORRECTABLE, chip_device->intr_enable_corr);
    LWSWITCH_REG_WR32(device, _PSMC, _INTR_EN_SET_FATAL, chip_device->intr_enable_uncorr);
    LWSWITCH_REG_WR32(device, _PSMC, _INTR_EN_SET_NONFATAL, 0);

    LWSWITCH_FLUSH_MMIO(device);
}

/*
 * The MSI interrupt block must be re-armed after servicing interrupts. This
 * write generates an EOI, which allows further MSIs to be triggered.
 */
static void
_lwswitch_rearm_msi
(
    lwswitch_device *device
)
{
    LWSWITCH_ENG_WR32_SV10(device, XVE, , 0, uc, _XVE_CYA, _2, 0xff);

    LWSWITCH_FLUSH_MMIO(device);
}

/*
 * @Brief : Disable top level HW interrupts.
 *
 * @Description :
 *
 * @param[in] device        operate on this device
 */
void
lwswitch_lib_disable_interrupts_sv10
(
    lwswitch_device *device
)
{
    LWSWITCH_REG_WR32(device, _PSMC, _INTR_EN_CLR_LEGACY, 0xffffffff);
    LWSWITCH_REG_WR32(device, _PSMC, _INTR_EN_CLR_FATAL, 0xffffffff);
    LWSWITCH_REG_WR32(device, _PSMC, _INTR_EN_CLR_CORRECTABLE, 0xffffffff);
    LWSWITCH_FLUSH_MMIO(device);

    //
    // Need a bit more time to ensure interrupt de-asserts, on
    // RTL simulation under FSF. Part of BUG 1869204 and 1881361.
    //
    if (IS_RTLSIM(device))
    {
        (void)LWSWITCH_REG_RD32(device, _PSMC, _INTR_EN_CLR_CORRECTABLE);
    }
}

static void
_lwswitch_build_top_interrupt_mask
(
    lwswitch_device *device
)
{
    LwU32 i;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    chip_device->intr_enable_legacy =
        DRF_NUM(_PSMC, _INTR_EN_SET_LEGACY, _PTIMER, 0) |
        DRF_NUM(_PSMC, _INTR_EN_SET_LEGACY, _PMGR, 0) |
        DRF_NUM(_PSMC, _INTR_EN_SET_LEGACY, _DECODE_TRAP_PRIV_LEVEL_VIOLATION, 0) |
        DRF_NUM(_PSMC, _INTR_EN_SET_LEGACY, _DECODE_TRAP_WRITE_DROPPED, 0) |
        DRF_NUM(_PSMC, _INTR_EN_SET_LEGACY, _RING_MANAGE_SUCCESS, 0) |
        DRF_NUM(_PSMC, _INTR_EN_SET_LEGACY, _PBUS, 1) |
        DRF_NUM(_PSMC, _INTR_EN_SET_LEGACY, _XVE, 0) |
        DRF_NUM(_PSMC, _INTR_EN_SET_LEGACY, _PRIV_RING, 1);

    chip_device->intr_enable_uncorr = 0;
    chip_device->intr_enable_corr = 0;

    for (i = 0; i < NUM_SWX_ENGINE_SV10; i++)
    {
        if (chip_device->engSWX[i].valid)
        {
            // SWX only has fatal interrupts
            LWSWITCH_ASSERT((chip_device->intr_enable_uncorr & (1 << chip_device->engSWX[i].intr_bit)) == 0);
            chip_device->intr_enable_uncorr |= 1 << chip_device->engSWX[i].intr_bit;
        }
    }

    for (i = 0; i < NUM_NPG_ENGINE_SV10; i++)
    {
        if (chip_device->engNPG[i].valid)
        {
            LWSWITCH_ASSERT((chip_device->intr_enable_uncorr & (1 << chip_device->engNPG[i].intr_bit)) == 0);
            LWSWITCH_ASSERT((chip_device->intr_enable_corr & (1 << chip_device->engNPG[i].intr_bit)) == 0);
            chip_device->intr_enable_uncorr |= 1 << chip_device->engNPG[i].intr_bit;
            chip_device->intr_enable_corr |= 1 << chip_device->engNPG[i].intr_bit;
        }
    }

    for (i = 0; i < NUM_SIOCTRL_ENGINE_SV10; i++)
    {
        if (chip_device->engSIOCTRL[i].valid)
        {
            LWSWITCH_ASSERT((chip_device->intr_enable_uncorr & (1 << chip_device->engSIOCTRL[i].intr_bit)) == 0);
            LWSWITCH_ASSERT((chip_device->intr_enable_corr & (1 << chip_device->engSIOCTRL[i].intr_bit)) == 0);
            chip_device->intr_enable_uncorr |= 1 << chip_device->engSIOCTRL[i].intr_bit;
            chip_device->intr_enable_corr |= 1 << chip_device->engSIOCTRL[i].intr_bit;
        }
    }
}

/*
 * @Brief : Enable MINION Falcon interrupts. These are not link related interrupts.
 *          These are interrupts related to Falcon operation.
 *
 * @param[in] device : Enable Falcon interrupts for MINIONs on this device.
 *
 */
static void
_lwswitch_minion_enable_falcon_interrupts
(
    lwswitch_device *device
)
{
    LwU32 i, flcnDefaultIntrMask, flcnDefaultIntrDest, intrEn;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    flcnDefaultIntrMask = (DRF_DEF(_CMINION, _FALCON_IRQMSET, _WDTMR,  _SET) |
                           DRF_DEF(_CMINION, _FALCON_IRQMSET, _HALT,   _SET) |
                           DRF_DEF(_CMINION, _FALCON_IRQMSET, _EXTERR, _SET) |
                           DRF_DEF(_CMINION, _FALCON_IRQMSET, _SWGEN0, _SET) |
                           DRF_DEF(_CMINION, _FALCON_IRQMSET, _SWGEN1, _SET));

    flcnDefaultIntrDest = (DRF_DEF(_CMINION, _FALCON_IRQDEST, _HOST_WDTMR,  _HOST) |
                           DRF_DEF(_CMINION, _FALCON_IRQDEST, _HOST_HALT,   _HOST) |
                           DRF_DEF(_CMINION, _FALCON_IRQDEST, _HOST_EXTERR, _HOST) |
                           DRF_DEF(_CMINION, _FALCON_IRQDEST, _HOST_SWGEN0, _HOST) |
                           DRF_DEF(_CMINION, _FALCON_IRQDEST, _HOST_SWGEN1,   _HOST)        |
                           DRF_DEF(_CMINION, _FALCON_IRQDEST, _TARGET_WDTMR,  _HOST_NORMAL) |
                           DRF_DEF(_CMINION, _FALCON_IRQDEST, _TARGET_HALT,   _HOST_NORMAL) |
                           DRF_DEF(_CMINION, _FALCON_IRQDEST, _TARGET_EXTERR, _HOST_NORMAL) |
                           DRF_DEF(_CMINION, _FALCON_IRQDEST, _TARGET_SWGEN0, _HOST_NORMAL) |
                           DRF_DEF(_CMINION, _FALCON_IRQDEST, _TARGET_SWGEN1, _HOST_NORMAL));

    // There is one MINION instance per SIOCTRL in Willow. one MINION controls 2 links.
    for (i = 0; i < (chip_device->numSIOCTRL * NUM_MINION_INSTANCES_SV10) ; i++)
    {
        if (!chip_device->engSIOCTRL[i].valid || !chip_device->subengSIOCTRL[i].subengMINION[0].valid)
        {
            continue;
        }

        if (!chip_device->subengSIOCTRL[i].subengMINION[0].initialized)
        {
            continue;
        }

        LWSWITCH_MINION_WR32_SV10(device, i, _CMINION, _FALCON_IRQMSET, flcnDefaultIntrMask);
        LWSWITCH_MINION_WR32_SV10(device, i, _CMINION, _FALCON_IRQDEST, flcnDefaultIntrDest);

        // Enable MINION "intr" stallable interrupt tree.
        intrEn = LWSWITCH_MINION_RD32_SV10(device, i, _MINION, _MINION_INTR_STALL_EN);
        intrEn = FLD_SET_DRF(_MINION, _MINION_INTR_STALL_EN, _FATAL, _ENABLE, intrEn);
        intrEn = FLD_SET_DRF(_MINION, _MINION_INTR_STALL_EN, _FALCON_STALL, _ENABLE, intrEn);
        LWSWITCH_MINION_WR32_SV10(device, i, _MINION, _MINION_INTR_STALL_EN, intrEn);
    }

    LWSWITCH_FLUSH_MMIO(device);
}


static void
_lwswitch_initialize_lwlipt_interrupts
(
    lwswitch_device *device
)
{
    LwU32 val;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    // Route all non-correctable "err" interrupts through fatal tree.
    val = DRF_NUM(_LWLIPT, _ERR_UC_SEVERITY_LINK0, _DLPROTOCOL, 1)         |
        DRF_NUM(_LWLIPT, _ERR_UC_SEVERITY_LINK0, _DATAPOISONED, 1)         |
        DRF_NUM(_LWLIPT, _ERR_UC_SEVERITY_LINK0, _FLOWCONTROL, 1)          |
        DRF_NUM(_LWLIPT, _ERR_UC_SEVERITY_LINK0, _RESPONSETIMEOUT, 1)      |
        DRF_NUM(_LWLIPT, _ERR_UC_SEVERITY_LINK0, _TARGETERROR, 1)          |
        DRF_NUM(_LWLIPT, _ERR_UC_SEVERITY_LINK0, _UNEXPECTEDRESPONSE, 1)   |
        DRF_NUM(_LWLIPT, _ERR_UC_SEVERITY_LINK0, _RECEIVEROVERFLOW, 1)     |
        DRF_NUM(_LWLIPT, _ERR_UC_SEVERITY_LINK0, _MALFORMEDPACKET, 1)      |
        DRF_NUM(_LWLIPT, _ERR_UC_SEVERITY_LINK0, _STOMPEDPACKETRECEIVED, 1)|
        DRF_NUM(_LWLIPT, _ERR_UC_SEVERITY_LINK0, _UNSUPPORTEDREQUEST, 1)   |
        DRF_NUM(_LWLIPT, _ERR_UC_SEVERITY_LINK0, _UCINTERNAL, 1);
    LWSWITCH_LWLIPT_BCAST_WR32_SV10(device, _LWLIPT, _ERR_UC_SEVERITY_LINK0, val);
    LWSWITCH_LWLIPT_BCAST_WR32_SV10(device, _LWLIPT, _ERR_UC_SEVERITY_LINK1, val);

    //  0 is enabled, 1 is disabled
    val = DRF_NUM(_LWLIPT, _ERR_UC_MASK_LINK0, _DLPROTOCOL, 1)          | // Ignore DL "err"
        DRF_NUM(_LWLIPT, _ERR_UC_MASK_LINK0, _DATAPOISONED, 1)          | // silent poison
        DRF_NUM(_LWLIPT, _ERR_UC_MASK_LINK0, _FLOWCONTROL, 0)           |
        DRF_NUM(_LWLIPT, _ERR_UC_MASK_LINK0, _RESPONSETIMEOUT, 1)       | // not in RTL
        DRF_NUM(_LWLIPT, _ERR_UC_MASK_LINK0, _TARGETERROR, 0)           |
        DRF_NUM(_LWLIPT, _ERR_UC_MASK_LINK0, _UNEXPECTEDRESPONSE, 1)    | // not in RTL
        DRF_NUM(_LWLIPT, _ERR_UC_MASK_LINK0, _RECEIVEROVERFLOW, 0)      |
        DRF_NUM(_LWLIPT, _ERR_UC_MASK_LINK0, _MALFORMEDPACKET, 0)       |
        DRF_NUM(_LWLIPT, _ERR_UC_MASK_LINK0, _STOMPEDPACKETRECEIVED, 0) |
        DRF_NUM(_LWLIPT, _ERR_UC_MASK_LINK0, _UNSUPPORTEDREQUEST, 0)    |
        DRF_NUM(_LWLIPT, _ERR_UC_MASK_LINK0, _UCINTERNAL, 0);
    LWSWITCH_LWLIPT_BCAST_WR32_SV10(device, _LWLIPT, _ERR_UC_MASK_LINK0, val);
    LWSWITCH_LWLIPT_BCAST_WR32_SV10(device, _LWLIPT, _ERR_UC_MASK_LINK1, val);
    chip_device->intr_mask.lwlipt_uc = val;

    // All common LWLIPT errors are tied off to 0 in the RTL
    val = 0xffffffff;
    LWSWITCH_LWLIPT_BCAST_WR32_SV10(device, _LWLIPT, _ERR_UC_MASK_COMMON, val);

    // 0 is enabled, 1 is disabled
    val = DRF_NUM(_LWLIPT, _ERR_C_MASK_LINK0, _PHYRECEIVER, 1) | // Not implemented in RTL
        DRF_NUM(_LWLIPT, _ERR_C_MASK_LINK0, _BADAN0PKT, 1)     | // Ignore DL "err"
        DRF_NUM(_LWLIPT, _ERR_C_MASK_LINK0, _REPLAYTIMEOUT, 1) | // Ignore DL "err"
        DRF_NUM(_LWLIPT, _ERR_C_MASK_LINK0, _ADVISORYERROR, 0) |
        DRF_NUM(_LWLIPT, _ERR_C_MASK_LINK0, _CINTERNAL, 0)     |
        DRF_NUM(_LWLIPT, _ERR_C_MASK_LINK0, _HEADEROVERFLOW, 0);
    LWSWITCH_LWLIPT_BCAST_WR32_SV10(device, _LWLIPT, _ERR_C_MASK_LINK0, val);
    LWSWITCH_LWLIPT_BCAST_WR32_SV10(device, _LWLIPT, _ERR_C_MASK_LINK1, val);
    chip_device->intr_mask.lwlipt_c = val;

    // Enable correctable + fatal trees
    val = DRF_NUM(_LWLIPT, _ERR_CONTROL_LINK0, _CORRECTABLEENABLE, 1) |
        DRF_NUM(_LWLIPT, _ERR_CONTROL_LINK0, _FATALENABLE, 1)         |
        DRF_NUM(_LWLIPT, _ERR_CONTROL_LINK0, _NONFATALENABLE, 0);
    LWSWITCH_LWLIPT_BCAST_WR32_SV10(device, _LWLIPT, _ERR_CONTROL_LINK0, val);
    LWSWITCH_LWLIPT_BCAST_WR32_SV10(device, _LWLIPT, _ERR_CONTROL_LINK1, val);

    // Enable common stall interrupts. This is where fatal Minion interrupts are routed.
    val = DRF_NUM(_LWLIPT, _INTR_CONTROL_COMMON, _STALLENABLE, 1);
    LWSWITCH_LWLIPT_BCAST_WR32_SV10(device, _LWLIPT, _INTR_CONTROL_COMMON, val);

    // Enable LWLIPT "intr" link stallable interrupt tree.  Routed to fatal tree.
    val = DRF_NUM(_LWLIPT, _INTR_CONTROL_LINK0, _STALLENABLE,   1)         |
        DRF_NUM(_LWLIPT, _INTR_CONTROL_LINK0, _NOSTALLENABLE, 0);
    LWSWITCH_LWLIPT_BCAST_WR32_SV10(device, _LWLIPT, _INTR_CONTROL_LINK0, val);
    LWSWITCH_LWLIPT_BCAST_WR32_SV10(device, _LWLIPT, _INTR_CONTROL_LINK1, val);

    val = DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_LOG_EN_0, _INGRESSECCSOFTLIMITERR, 1)   |
        DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_LOG_EN_0, _INGRESSECCHDRDOUBLEBITERR, 1)  |
        DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_LOG_EN_0, _INGRESSECCDATADOUBLEBITERR, 1) |
        DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_LOG_EN_0, _INGRESSBUFFERERR, 1)           |
        DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_LOG_EN_0, _EGRESSECCSOFTLIMITERR, 1)      |
        DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_LOG_EN_0, _EGRESSECCHDRDOUBLEBITERR, 1)   |
        DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_LOG_EN_0, _EGRESSECCDATADOUBLEBITERR, 1)  |
        DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_LOG_EN_0, _EGRESSBUFFERERR, 1);
    LWSWITCH_SIOCTRL_BCAST_WR32_SV10(device, _LWLCTRL, _CLKCROSS_0_ERR_LOG_EN_0, val);
    LWSWITCH_SIOCTRL_BCAST_WR32_SV10(device, _LWLCTRL, _CLKCROSS_0_ERR_REPORT_EN_0, val);
    LWSWITCH_SIOCTRL_BCAST_WR32_SV10(device, _LWLCTRL, _CLKCROSS_1_ERR_LOG_EN_0, val);
    LWSWITCH_SIOCTRL_BCAST_WR32_SV10(device, _LWLCTRL, _CLKCROSS_1_ERR_REPORT_EN_0, val);
    chip_device->intr_mask.clkcross = val;

    val = DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_CONTAIN_EN_0, _INGRESSECCSOFTLIMITERR, 0)   |
        DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_CONTAIN_EN_0, _INGRESSECCHDRDOUBLEBITERR, 1)  |
        DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_CONTAIN_EN_0, _INGRESSECCDATADOUBLEBITERR, 0) |
        DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_CONTAIN_EN_0, _INGRESSBUFFERERR, 0)           |
        DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_CONTAIN_EN_0, _EGRESSECCSOFTLIMITERR, 0)      |
        DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_CONTAIN_EN_0, _EGRESSECCHDRDOUBLEBITERR, 1)   |
        DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_CONTAIN_EN_0, _EGRESSECCDATADOUBLEBITERR, 0)  |
        DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_CONTAIN_EN_0, _EGRESSBUFFERERR, 0);
    LWSWITCH_SIOCTRL_BCAST_WR32_SV10(device, _LWLCTRL, _CLKCROSS_0_ERR_CONTAIN_EN_0, val);
    LWSWITCH_SIOCTRL_BCAST_WR32_SV10(device, _LWLCTRL, _CLKCROSS_1_ERR_CONTAIN_EN_0, val);

    // MINION is closely related to lwlipt in HW. So enable MINION interrupts here.
    _lwswitch_minion_enable_falcon_interrupts(device);

    LWSWITCH_FLUSH_MMIO(device);
}

/*
 * Initialize interrupt tree HW for all units.
 *
 * Init and servicing both depend on bits matching across STATUS/MASK
 * and IErr STATUS/LOG/REPORT/CONTAIN registers.
 */
void
lwswitch_initialize_interrupt_tree_sv10
(
    lwswitch_device *device
)
{
    LwU32 val;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    LWSWITCH_FLUSH_MMIO(device);

    _lwswitch_build_top_interrupt_mask(device);

    // Initialize legacy interrupt tree - depends on reset to disabled for
    // unused interrupts
    LWSWITCH_REG_WR32(device, _PBUS, _INTR_0, 0xffffffff);

    // Clear prior saved PRI error data
    LWSWITCH_REG_WR32(device, _PTIMER, _PRI_TIMEOUT_SAVE_0,
        DRF_DEF(_PTIMER, _PRI_TIMEOUT_SAVE_0, _TO, _CLEAR));

    LWSWITCH_REG_WR32(device, _PBUS, _INTR_EN_0,
            DRF_DEF(_PBUS, _INTR_EN_0, _PRI_TIMEOUT, _ENABLED) |
            DRF_DEF(_PBUS, _INTR_EN_0, _PRI_SQUASH, _DISABLED) |
            DRF_DEF(_PBUS, _INTR_EN_0, _PRI_FECSERR, _ENABLED) |
            DRF_DEF(_PBUS, _INTR_EN_0, _SW, _ENABLED));

    // SWX - 0 is enabled, 1 is disabled
    LWSWITCH_AFS_MC_BCAST_WR32_SV10(device, _AFS, _ERR_MASK_UC,
            DRF_NUM(_AFS, _ERR_MASK_UC, _INGRESS_CREDIT_OVERFLOW, 0) |
            DRF_NUM(_AFS, _ERR_MASK_UC, _INGRESS_CREDIT_UNDERFLOW, 0) |
            DRF_NUM(_AFS, _ERR_MASK_UC, _EGRESS_CREDIT_OVERFLOW, 0) |
            DRF_NUM(_AFS, _ERR_MASK_UC, _EGRESS_CREDIT_UNDERFLOW, 0) |
            DRF_NUM(_AFS, _ERR_MASK_UC, _INGRESS_NON_BURSTY_PKT_DETECTED, 0) |
            DRF_NUM(_AFS, _ERR_MASK_UC, _INGRESS_NON_STICKY_PKT_DETECTED, 0) |
            DRF_NUM(_AFS, _ERR_MASK_UC, _INGRESS_BURST_GT_17_DATA_VC_DETECTED, 0) |
            DRF_NUM(_AFS, _ERR_MASK_UC, _INGRESS_BURST_GT_1_NONDATA_VC_DETECTED, 0) |
            DRF_NUM(_AFS, _ERR_MASK_UC, _ILWALID_DST, 0) |
            DRF_NUM(_AFS, _ERR_MASK_UC, _PKT_MISROUTE, 0));
    LWSWITCH_AFS_MC_BCAST_WR32_SV10(device, _AFS, _ERR_CTRL,
            DRF_NUM(_AFS, _ERR_CTRL, _AFS_FATAL_ENABLE, 1));

    // NPG/NPORT

    // Route all non-correctable interrupts through fatal tree.
    val = DRF_NUM(_NPORT, _ERR_UC_SEVERITY_NPORT, _DLPROTOCOL, 1) |
        DRF_NUM(_NPORT, _ERR_UC_SEVERITY_NPORT, _DATAPOISONED, 1) |
        DRF_NUM(_NPORT, _ERR_UC_SEVERITY_NPORT, _FLOWCONTROL, 1) |
        DRF_NUM(_NPORT, _ERR_UC_SEVERITY_NPORT, _RESPONSETIMEOUT, 1) |
        DRF_NUM(_NPORT, _ERR_UC_SEVERITY_NPORT, _TARGETERROR, 1) |
        DRF_NUM(_NPORT, _ERR_UC_SEVERITY_NPORT, _UNEXPECTEDRESPONSE, 1) |
        DRF_NUM(_NPORT, _ERR_UC_SEVERITY_NPORT, _RECEIVEROVERFLOW, 1) |
        DRF_NUM(_NPORT, _ERR_UC_SEVERITY_NPORT, _MALFORMEDPACKET, 1) |
        DRF_NUM(_NPORT, _ERR_UC_SEVERITY_NPORT, _STOMPEDPACKETRECEIVED, 1) |
        DRF_NUM(_NPORT, _ERR_UC_SEVERITY_NPORT, _UNSUPPORTEDREQUEST, 1) |
        DRF_NUM(_NPORT, _ERR_UC_SEVERITY_NPORT, _UCINTERNAL, 1);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _NPORT, _ERR_UC_SEVERITY_NPORT, val);

    val = DRF_NUM(_NPORT, _ERR_CONTROL_NPORT, _CORRECTABLEENABLE, 1) |
        DRF_NUM(_NPORT, _ERR_CONTROL_NPORT, _FATALENABLE, 1) |
        DRF_NUM(_NPORT, _ERR_CONTROL_NPORT, _NONFATALENABLE, 0);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _NPORT, _ERR_CONTROL_NPORT, val);

    // 0 is enabled, 1 is disabled
    val = DRF_NUM(_NPORT, _ERR_UC_MASK_NPORT, _DLPROTOCOL, 1) |             // not in RTL
        DRF_NUM(_NPORT, _ERR_UC_MASK_NPORT, _DATAPOISONED, 0) |             // allow leaf ECC intr
        DRF_NUM(_NPORT, _ERR_UC_MASK_NPORT, _FLOWCONTROL, 1) |              // not in RTL
        DRF_NUM(_NPORT, _ERR_UC_MASK_NPORT, _RESPONSETIMEOUT, 1) |          // not in RTL
        DRF_NUM(_NPORT, _ERR_UC_MASK_NPORT, _TARGETERROR, 1) |              // not in RTL
        DRF_NUM(_NPORT, _ERR_UC_MASK_NPORT, _UNEXPECTEDRESPONSE, 1) |       // not in RTL
        DRF_NUM(_NPORT, _ERR_UC_MASK_NPORT, _RECEIVEROVERFLOW, 1) |         // not in RTL
        DRF_NUM(_NPORT, _ERR_UC_MASK_NPORT, _MALFORMEDPACKET, 1) |          // not in RTL
        DRF_NUM(_NPORT, _ERR_UC_MASK_NPORT, _STOMPEDPACKETRECEIVED, 1) |    // not in RTL
        DRF_NUM(_NPORT, _ERR_UC_MASK_NPORT, _UNSUPPORTEDREQUEST, 1) |       // not in RTL
        DRF_NUM(_NPORT, _ERR_UC_MASK_NPORT, _UCINTERNAL, 0);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _NPORT, _ERR_UC_MASK_NPORT, val);
    chip_device->intr_mask.nport_uc = val;

    // 0 is enabled, 1 is disabled
    val = DRF_NUM(_NPORT, _ERR_C_MASK_NPORT, _PHYRECEIVER, 1) |             // not in RTL
        DRF_NUM(_NPORT, _ERR_C_MASK_NPORT, _BADAN0PKT, 1) |                 // not in RTL
        DRF_NUM(_NPORT, _ERR_C_MASK_NPORT, _REPLAYTIMEOUT, 1) |             // not in RTL
        DRF_NUM(_NPORT, _ERR_C_MASK_NPORT, _ADVISORYERROR, 1) |             // not in RTL
        DRF_NUM(_NPORT, _ERR_C_MASK_NPORT, _CINTERNAL, 0) |
        DRF_NUM(_NPORT, _ERR_C_MASK_NPORT, _HEADEROVERFLOW, 1);             // not in RTL
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _NPORT, _ERR_C_MASK_NPORT, val);
    chip_device->intr_mask.nport_c = val;

    // Other nport intrs - enable log, report and set contain
    val = DRF_NUM(_TSTATE, _ERR_LOG_EN_0, _TAGPOOLBUFERR, 1) |
        DRF_NUM(_TSTATE, _ERR_LOG_EN_0, _CRUMBSTOREBUFERR, 1) |
        DRF_NUM(_TSTATE, _ERR_LOG_EN_0, _SINGLEBITECCLIMITERR_CRUMBSTORE, 1) |
        DRF_NUM(_TSTATE, _ERR_LOG_EN_0, _UNCORRECTABLEECCERR_CRUMBSTORE, 1) |
        DRF_NUM(_TSTATE, _ERR_LOG_EN_0, _SINGLEBITECCLIMITERR_TAGSTORE, 1) |
        DRF_NUM(_TSTATE, _ERR_LOG_EN_0, _UNCORRECTABLEECCERR_TAGSTORE, 1);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _TSTATE, _ERR_LOG_EN_0, val);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _TSTATE, _ERR_REPORT_EN_0, val);
    chip_device->intr_mask.tstate = val;

    val = DRF_NUM(_TSTATE, _ERR_CONTAIN_EN_0, _TAGPOOLBUFERR, 1) |
        DRF_NUM(_TSTATE, _ERR_CONTAIN_EN_0, _CRUMBSTOREBUFERR, 1) |
        DRF_NUM(_TSTATE, _ERR_CONTAIN_EN_0, _SINGLEBITECCLIMITERR_CRUMBSTORE, 0) |
        DRF_NUM(_TSTATE, _ERR_CONTAIN_EN_0, _UNCORRECTABLEECCERR_CRUMBSTORE, 1) |
        DRF_NUM(_TSTATE, _ERR_CONTAIN_EN_0, _SINGLEBITECCLIMITERR_TAGSTORE, 0) |
        DRF_NUM(_TSTATE, _ERR_CONTAIN_EN_0, _UNCORRECTABLEECCERR_TAGSTORE, 1);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _TSTATE, _ERR_CONTAIN_EN_0, val);

    val = DRF_NUM(_FSTATE, _ERR_LOG_EN_0, _TAGPOOLBUFERR, 1) |
        DRF_NUM(_FSTATE, _ERR_LOG_EN_0, _CRUMBSTOREBUFERR, 1) |
        DRF_NUM(_FSTATE, _ERR_LOG_EN_0, _SINGLEBITECCLIMITERR_CRUMBSTORE, 1) |
        DRF_NUM(_FSTATE, _ERR_LOG_EN_0, _UNCORRECTABLEECCERR_CRUMBSTORE, 1) |
        DRF_NUM(_FSTATE, _ERR_LOG_EN_0, _SINGLEBITECCLIMITERR_TAGSTORE, 1) |
        DRF_NUM(_FSTATE, _ERR_LOG_EN_0, _UNCORRECTABLEECCERR_TAGSTORE, 1) |
        DRF_NUM(_FSTATE, _ERR_LOG_EN_0, _SINGLEBITECCLIMITERR_FLUSHREQSTORE, 1) |
        DRF_NUM(_FSTATE, _ERR_LOG_EN_0, _UNCORRECTABLEECCERR_FLUSHREQSTORE, 1);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _FSTATE, _ERR_LOG_EN_0, val);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _FSTATE, _ERR_REPORT_EN_0, val);
    chip_device->intr_mask.fstate = val;

    val = DRF_NUM(_FSTATE, _ERR_CONTAIN_EN_0, _TAGPOOLBUFERR, 1) |
        DRF_NUM(_FSTATE, _ERR_CONTAIN_EN_0, _CRUMBSTOREBUFERR, 1) |
        DRF_NUM(_FSTATE, _ERR_CONTAIN_EN_0, _SINGLEBITECCLIMITERR_CRUMBSTORE, 0) |
        DRF_NUM(_FSTATE, _ERR_CONTAIN_EN_0, _UNCORRECTABLEECCERR_CRUMBSTORE, 1) |
        DRF_NUM(_FSTATE, _ERR_CONTAIN_EN_0, _SINGLEBITECCLIMITERR_TAGSTORE, 0) |
        DRF_NUM(_FSTATE, _ERR_CONTAIN_EN_0, _UNCORRECTABLEECCERR_TAGSTORE, 1) |
        DRF_NUM(_FSTATE, _ERR_CONTAIN_EN_0, _SINGLEBITECCLIMITERR_FLUSHREQSTORE, 0) |
        DRF_NUM(_FSTATE, _ERR_CONTAIN_EN_0, _UNCORRECTABLEECCERR_FLUSHREQSTORE, 1);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _FSTATE, _ERR_CONTAIN_EN_0, val);

    val = DRF_DEF(_INGRESS, _ERR_LOG_EN_0, _CMDDECODEERR, __PROD) |
        DRF_DEF(_INGRESS, _ERR_LOG_EN_0, _BDFMISMATCHERR, __PROD) |
        DRF_DEF(_INGRESS, _ERR_LOG_EN_0, _BUBBLEDETECT, __PROD) |
        DRF_DEF(_INGRESS, _ERR_LOG_EN_0, _ACLFAIL, __PROD) |
        DRF_DEF(_INGRESS, _ERR_LOG_EN_0, _PKTPOISONSET, __PROD) |  // silent poison interrupt
        DRF_DEF(_INGRESS, _ERR_LOG_EN_0, _ECCSOFTLIMITERR, _DISABLE) | // See Bug 1973042.
        DRF_DEF(_INGRESS, _ERR_LOG_EN_0, _ECCHDRDOUBLEBITERR, __PROD) |
        DRF_DEF(_INGRESS, _ERR_LOG_EN_0, _ILWALIDCMD, __PROD) |
        DRF_DEF(_INGRESS, _ERR_LOG_EN_0, _ILWALIDVCSET, __PROD);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _INGRESS, _ERR_LOG_EN_0, val);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _INGRESS, _ERR_REPORT_EN_0, val);
    chip_device->intr_mask.ingress = val;

#define LWSWITCH_INGRESS_CONTAIN_EN \
    DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _CMDDECODEERR, __PROD) | \
    DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _BDFMISMATCHERR, __PROD) | \
    DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _BUBBLEDETECT, __PROD) | \
    DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _ACLFAIL, __PROD) | \
    DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _PKTPOISONSET, __PROD) | \
    DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _ECCSOFTLIMITERR, __PROD) | \
    DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _ECCHDRDOUBLEBITERR, __PROD) | \
    DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _ILWALIDCMD, __PROD) | \
    DRF_DEF(_INGRESS, _ERR_CONTAIN_EN_0, _ILWALIDVCSET, __PROD)

    val = LWSWITCH_INGRESS_CONTAIN_EN;
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _INGRESS, _ERR_CONTAIN_EN_0, val);

    val = DRF_NUM(_EGRESS, _ERR_LOG_EN_0, _EGRESSBUFERR, 1) |
        DRF_NUM(_EGRESS, _ERR_LOG_EN_0, _PKTROUTEERR, 1) |
        DRF_NUM(_EGRESS, _ERR_LOG_EN_0, _ECCSINGLEBITLIMITERR0, 1) |
        DRF_NUM(_EGRESS, _ERR_LOG_EN_0, _ECCHDRDOUBLEBITERR0, 1) |
        DRF_NUM(_EGRESS, _ERR_LOG_EN_0, _ECCDATADOUBLEBITERR0, 1) |
        DRF_NUM(_EGRESS, _ERR_LOG_EN_0, _ECCSINGLEBITLIMITERR1, 1) |
        DRF_NUM(_EGRESS, _ERR_LOG_EN_0, _ECCHDRDOUBLEBITERR1, 1) |
        DRF_NUM(_EGRESS, _ERR_LOG_EN_0, _ECCDATADOUBLEBITERR1, 1) |
        DRF_NUM(_EGRESS, _ERR_LOG_EN_0, _NCISOCHDRCREDITOVFL, 1) |
        DRF_NUM(_EGRESS, _ERR_LOG_EN_0, _NCISOCDATACREDITOVFL, 1) |
        DRF_NUM(_EGRESS, _ERR_LOG_EN_0, _ADDRMATCHERR, 1) |
        DRF_NUM(_EGRESS, _ERR_LOG_EN_0, _TAGCOUNTERR, 1) |
        DRF_NUM(_EGRESS, _ERR_LOG_EN_0, _FLUSHRSPERR, 1) |
        DRF_NUM(_EGRESS, _ERR_LOG_EN_0, _DROPNPURRSPERR, 1) |
        DRF_NUM(_EGRESS, _ERR_LOG_EN_0, _POISONERR, 0);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _EGRESS, _ERR_LOG_EN_0, val);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _EGRESS, _ERR_REPORT_EN_0, val);
    chip_device->intr_mask.egress = val;

#define LWSWITCH_EGRESS_CONTAIN_EN \
    DRF_NUM(_EGRESS, _ERR_CONTAIN_EN_0, _EGRESSBUFERR, 0) | \
    DRF_NUM(_EGRESS, _ERR_CONTAIN_EN_0, _PKTROUTEERR, 0) | \
    DRF_NUM(_EGRESS, _ERR_CONTAIN_EN_0, _ECCSINGLEBITLIMITERR0, 0) | \
    DRF_NUM(_EGRESS, _ERR_CONTAIN_EN_0, _ECCHDRDOUBLEBITERR0, 1) | \
    DRF_NUM(_EGRESS, _ERR_CONTAIN_EN_0, _ECCDATADOUBLEBITERR0, 0) | \
    DRF_NUM(_EGRESS, _ERR_CONTAIN_EN_0, _ECCSINGLEBITLIMITERR1, 0) | \
    DRF_NUM(_EGRESS, _ERR_CONTAIN_EN_0, _ECCHDRDOUBLEBITERR1, 1) | \
    DRF_NUM(_EGRESS, _ERR_CONTAIN_EN_0, _ECCDATADOUBLEBITERR1, 0) | \
    DRF_NUM(_EGRESS, _ERR_CONTAIN_EN_0, _NCISOCHDRCREDITOVFL, 1) | \
    DRF_NUM(_EGRESS, _ERR_CONTAIN_EN_0, _NCISOCDATACREDITOVFL, 1) | \
    DRF_NUM(_EGRESS, _ERR_CONTAIN_EN_0, _ADDRMATCHERR, 1) | \
    DRF_NUM(_EGRESS, _ERR_CONTAIN_EN_0, _TAGCOUNTERR, 1) | \
    /* FLUSHRSPERR is not contained, but proper setting is debatable */ \
    DRF_NUM(_EGRESS, _ERR_CONTAIN_EN_0, _FLUSHRSPERR, 0) | \
    DRF_NUM(_EGRESS, _ERR_CONTAIN_EN_0, _DROPNPURRSPERR, 1) | \
    DRF_NUM(_EGRESS, _ERR_CONTAIN_EN_0, _POISONERR, 0)

    val = LWSWITCH_EGRESS_CONTAIN_EN;
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _EGRESS, _ERR_CONTAIN_EN_0, val);

    val = DRF_DEF(_ROUTE, _ERR_LOG_EN_0, _ROUTEBUFERR, _DETECTED) |
        DRF_DEF(_ROUTE, _ERR_LOG_EN_0, _NOPORTDEFINEDERR, _DETECTED) |
        DRF_DEF(_ROUTE, _ERR_LOG_EN_0, _ILWALIDROUTEPOLICYERR, _DETECTED) |
        DRF_DEF(_ROUTE, _ERR_LOG_EN_0, _ECCLIMITERR, _DETECTED) |
        DRF_DEF(_ROUTE, _ERR_LOG_EN_0, _UNCORRECTABLEECCERR, _DETECTED) |
        DRF_DEF(_ROUTE, _ERR_LOG_EN_0, _TRANSDONERESVERR, _DETECTED);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _ROUTE, _ERR_LOG_EN_0, val);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _ROUTE, _ERR_REPORT_EN_0, val);
    chip_device->intr_mask.route = val;

    val = DRF_NUM(_ROUTE, _ERR_CONTAIN_EN_0, _ROUTEBUFERR, 1) |
        DRF_NUM(_ROUTE, _ERR_CONTAIN_EN_0, _NOPORTDEFINEDERR, 0) |
        DRF_NUM(_ROUTE, _ERR_CONTAIN_EN_0, _ILWALIDROUTEPOLICYERR, 0) |
        DRF_NUM(_ROUTE, _ERR_CONTAIN_EN_0, _ECCLIMITERR, 0) |
        DRF_NUM(_ROUTE, _ERR_CONTAIN_EN_0, _UNCORRECTABLEECCERR, 1) |
        DRF_NUM(_ROUTE, _ERR_CONTAIN_EN_0, _TRANSDONERESVERR, 1);
    LWSWITCH_NPORT_MC_BCAST_WR32_SV10(device, _ROUTE, _ERR_CONTAIN_EN_0, val);

    _lwswitch_initialize_lwlipt_interrupts(device);

    LWSWITCH_FLUSH_MMIO(device);
}

static LwlStatus
_lwswitch_service_priv_ring
(
    lwswitch_device *device
)
{
    LwU32 pending;

    pending = LWSWITCH_REG_RD32(device, _PPRIV_MASTER, _RING_INTERRUPT_STATUS0);

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    if (FLD_TEST_DRF_NUM(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0,
            _GBL_WRITE_ERROR_SYS, 1, pending))
    {
        LWSWITCH_PRI_ERROR_LOG_TYPE pri_error;

        pri_error.addr = LWSWITCH_REG_RD32(device, _PPRIV_SYS, _PRIV_ERROR_ADR);
        pri_error.data = LWSWITCH_REG_RD32(device, _PPRIV_SYS, _PRIV_ERROR_WRDAT);
        pri_error.info = LWSWITCH_REG_RD32(device, _PPRIV_SYS, _PRIV_ERROR_INFO);
        pri_error.code = LWSWITCH_REG_RD32(device, _PPRIV_SYS, _PRIV_ERROR_CODE);

        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_HOST_PRIV_ERROR, 
            "Fatal, SYS PRI write error\n");
        LWSWITCH_LOG_FATAL_DATA(device, _HW, _HW_HOST_PRIV_ERROR, 0, 2, LW_FALSE, &pri_error);

        LWSWITCH_PRINT(device, ERROR,
            "SYS PRI write error addr: 0x%08x data: 0x%08x info: 0x%08x code: 0x%08x\n",
            pri_error.addr, pri_error.data,
            pri_error.info, pri_error.code);

        pending = FLD_SET_DRF_NUM(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0,
            _GBL_WRITE_ERROR_SYS, 0, pending);
    }

    if (DRF_VAL(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0,
            _GBL_WRITE_ERROR_FBP, pending) & LWBIT(0))
    {
        LWSWITCH_PRI_ERROR_LOG_TYPE pri_error;

        pri_error.addr = LWSWITCH_REG_RD32(device, _PPRIV_PRT_PRT0, _PRIV_ERROR_ADR);
        pri_error.data = LWSWITCH_REG_RD32(device, _PPRIV_PRT_PRT0, _PRIV_ERROR_WRDAT);
        pri_error.info = LWSWITCH_REG_RD32(device, _PPRIV_PRT_PRT0, _PRIV_ERROR_INFO);
        pri_error.code = LWSWITCH_REG_RD32(device, _PPRIV_PRT_PRT0, _PRIV_ERROR_CODE);

        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_HOST_PRIV_ERROR, 
            "Fatal, PRT0 PRI write error\n");
        LWSWITCH_LOG_FATAL_DATA(device, _HW, _HW_HOST_PRIV_ERROR, 0, 0, LW_FALSE, &pri_error);

        LWSWITCH_PRINT(device, ERROR,
            "PRT0 PRI write error addr: 0x%08x data: 0x%08x info: 0x%08x code: 0x%08x\n",
            pri_error.addr, pri_error.data,
            pri_error.info, pri_error.code);

        pending &= ~DRF_NUM(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0,
            _GBL_WRITE_ERROR_FBP, LWBIT(0));
    }

    if (DRF_VAL(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0,
            _GBL_WRITE_ERROR_FBP, pending) & LWBIT(1))
    {
        LWSWITCH_PRI_ERROR_LOG_TYPE pri_error;

        pri_error.addr = LWSWITCH_REG_RD32(device, _PPRIV_PRT_PRT1, _PRIV_ERROR_ADR);
        pri_error.data = LWSWITCH_REG_RD32(device, _PPRIV_PRT_PRT1, _PRIV_ERROR_WRDAT);
        pri_error.info = LWSWITCH_REG_RD32(device, _PPRIV_PRT_PRT1, _PRIV_ERROR_INFO);
        pri_error.code = LWSWITCH_REG_RD32(device, _PPRIV_PRT_PRT1, _PRIV_ERROR_CODE);

        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_HOST_PRIV_ERROR, 
            "Fatal, PRT1 PRI write error\n");
        LWSWITCH_LOG_FATAL_DATA(device, _HW, _HW_HOST_PRIV_ERROR, 0, 1, LW_FALSE, &pri_error);

        LWSWITCH_PRINT(device, ERROR,
            "PRT1 PRI write error addr: 0x%08x data: 0x%08x info: 0x%08x code: 0x%08x\n",
            pri_error.addr, pri_error.data,
            pri_error.info, pri_error.code);

        pending &= ~DRF_NUM(_PPRIV_MASTER, _RING_INTERRUPT_STATUS0,
            _GBL_WRITE_ERROR_FBP, LWBIT(1));
    }

    if (pending != 0)
    {
        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_HOST_PRIV_ERROR, 
            "Fatal, Unexpected PRI error\n");
        LWSWITCH_LOG_FATAL_DATA(device, _HW, _HW_HOST_PRIV_ERROR, 1, 0, LW_FALSE, &pending);

        LWSWITCH_PRINT(device, ERROR,
            "Unexpected PRI error 0x%08x\n", pending);
    }

    // TODO reset the priv ring like GPU driver?

    // acknowledge the interrupt to the ringmaster
    lwswitch_ring_master_cmd_sv10(device,
        DRF_DEF(_PPRIV_MASTER, _RING_COMMAND, _CMD, _ACK_INTERRUPT));

    return LWL_SUCCESS;
}

/**
 * @brief Service MINION Falcon interrupts on the requested interrupt tree
 *
 * @param[in] device   MINION on this device
 * @param[in] instance MINION instance
 *
 */
LwlStatus
lwswitch_minion_service_falcon_interrupts_sv10
(
    lwswitch_device *device,
    LwU32           instance
)
{
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, unhandled, intr;
    LwU8  link;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    report.raw_pending = LWSWITCH_MINION_RD32_SV10(device, instance, _CMINION, _FALCON_IRQSTAT);
    report.raw_enable = LWSWITCH_MINION_RD32_SV10(device, instance, _CMINION, _FALCON_IRQDEST);
    report.mask = LWSWITCH_MINION_RD32_SV10(device, instance, _CMINION, _FALCON_IRQMASK);
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    // Report fatal interrupts on even link of the associated MINION.
    link = (LwU8)(instance * 2);

    if (LWSWITCH_PENDING(DRF_NUM(_CMINION_FALCON, _IRQSTAT, _WDTMR, 1)))
    {
        LWSWITCH_REPORT_FATAL(_HW_MINION_WATCHDOG, "Minion Watchdog Timer ran out", LW_TRUE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_CMINION_FALCON, _IRQSTAT, _HALT, 1)))
    {
        LWSWITCH_REPORT_FATAL(_HW_MINION_HALT, "Minion HALT", LW_TRUE);
        // TODO : Print Falcon Debug Info
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_CMINION_FALCON, _IRQSTAT, _EXTERR, 1)))
    {
        LWSWITCH_REPORT_FATAL(_HW_MINION_EXTERR, "Minion EXTERR", LW_TRUE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_CMINION_FALCON, _IRQSTAT, _SWGEN0, 1)))
    {
        LWSWITCH_PRINT(device, INFO,
                       "%s: Received MINION Falcon SWGEN0 interrupt on MINION %d.\n",
                       __FUNCTION__, instance);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_CMINION_FALCON, _IRQSTAT, _SWGEN1, 1)))
    {
        LWSWITCH_PRINT(device, INFO,
                       "%s: Received MINION Falcon SWGEN1 interrupt on MINION %d.\n",
                       __FUNCTION__, instance);
        LWSWITCH_HANDLED(bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable fatal interrupt
    if (chip_device->link[link].fatal_error_oclwrred)
    {
        intr = LWSWITCH_MINION_RD32_SV10(device, instance, _MINION, _MINION_INTR_STALL_EN);
        intr = FLD_SET_DRF(_MINION, _MINION_INTR_STALL_EN, _FATAL, _DISABLE, intr);
        intr = FLD_SET_DRF(_MINION, _MINION_INTR_STALL_EN, _FALCON_STALL, _DISABLE, intr);
        LWSWITCH_MINION_WR32_SV10(device, instance, _MINION, _MINION_INTR_STALL_EN, intr);
    }

    // Clear interrupt (W1C)
    LWSWITCH_MINION_WR32_SV10(device, instance, _CMINION, _FALCON_IRQSCLR, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_minion_service_link_interrupts
(
    lwswitch_device *device,
    LwU32           linkNumber
)
{
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32              pending, bit, unhandled, reg;
    LwU32              dlpl_idx = linkNumber % 2;
    LwU32              link = linkNumber, intr;
    LwU32              instance;
    sv10_device        *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    instance = chip_device->link[linkNumber].engMINION->instance;

    reg = LWSWITCH_MINION_LINK_RD32_SV10(device, linkNumber, _MINION, _LWLINK_LINK_INTR(dlpl_idx));
    report.data[0]     = reg;
    report.raw_pending = DRF_NUM(_MINION, _LWLINK_LINK_INTR, _STATE, reg);

    reg = LWSWITCH_MINION_LINK_RD32_SV10(device, linkNumber, _MINION, _MINION_INTR_STALL_EN);
    reg = DRF_NUM(_MINION, _MINION_INTR_STALL_EN, _LINK, reg);

    report.raw_enable = reg & LWBIT(dlpl_idx);
    report.mask = report.raw_enable;

    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    if (LWSWITCH_PENDING(DRF_NUM(_MINION, _LWLINK_LINK_INTR, _STATE, report.data[0])))
    {
        switch(DRF_VAL(_MINION, _LWLINK_LINK_INTR, _CODE, report.data[0]))
        {
            // The following are considered NON-FATAL by arch:
            case LW_MINION_LWLINK_LINK_INTR_CODE_SWREQ:
                LWSWITCH_REPORT_NONFATAL(_HW_MINION_NONFATAL, "MINION Link SWREQ Interrupt");
                break;

            //
            // As per current POR we shouldn't be seeing this interrupt and
            // we won't be handling it. Ignore this interrupt. This is a
            // non-fatal interrupt.
            //
            case LW_MINION_LWLINK_LINK_INTR_CODE_PMDISABLED:
                LWSWITCH_REPORT_NONFATAL(_HW_MINION_NONFATAL, "MINION Link PMDISABLED Interrupt");
                break;

            // The following are considered FATAL by arch:
            case LW_MINION_LWLINK_LINK_INTR_CODE_NA:
                LWSWITCH_REPORT_FATAL(_HW_MINION_FATAL_LINK_INTR, "MINION Link NA Interrupt", LW_TRUE);
                break;

            case LW_MINION_LWLINK_LINK_INTR_CODE_DLREQ:
                LWSWITCH_REPORT_FATAL(_HW_MINION_FATAL_LINK_INTR, "MINION Link DLREQ Interrupt", LW_TRUE);
                break;

            case LW_MINION_LWLINK_LINK_INTR_CODE_DLCMDFAULT:
                LWSWITCH_REPORT_FATAL(_HW_MINION_DLCMD_FAULT, "MINION Link DLCMDFAULT Interrupt", LW_TRUE);
                break;

            case LW_MINION_LWLINK_LINK_INTR_CODE_NOINIT:
                LWSWITCH_REPORT_FATAL(_HW_MINION_FATAL_LINK_INTR, "MINION Link NOINIT Interrupt", LW_TRUE);
                break;

            case LW_MINION_LWLINK_LINK_INTR_CODE_BADINIT:
                LWSWITCH_REPORT_FATAL(_HW_MINION_FATAL_LINK_INTR, "MINION Link BADINIT Interrupt", LW_TRUE);
                break;

            case LW_MINION_LWLINK_LINK_INTR_CODE_PMFAIL:
                LWSWITCH_REPORT_FATAL(_HW_MINION_FATAL_LINK_INTR, "MINION Link PMFAIL Interrupt", LW_TRUE);
                break;

            default:
                LWSWITCH_REPORT_FATAL(_HW_MINION_FATAL_LINK_INTR, "MINION Link unknown Interrupt", LW_TRUE);
                break;
        }
        LWSWITCH_HANDLED(bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // On fatal interrupts, disable interrupts for that link
    if (chip_device->link[link].fatal_error_oclwrred)
    {
        reg = LWSWITCH_MINION_LINK_RD32_SV10(device, linkNumber, _MINION, _MINION_INTR_STALL_EN);

        // Disable interrupt bit for the given link
        intr = DRF_VAL(_MINION, _MINION_INTR_STALL_EN, _LINK, reg);
        intr &= ~LWBIT(dlpl_idx);

        reg = FLD_SET_DRF_NUM(_MINION, _MINION_INTR_STALL_EN, _LINK, intr, reg);
        LWSWITCH_MINION_LINK_WR32_SV10(device, linkNumber, _MINION, _MINION_INTR_STALL_EN, reg);
    }

    // Clear the interrupt state and move on
    reg = FLD_SET_DRF_NUM(_MINION, _LWLINK_LINK_INTR, _STATE, 1, report.data[0]);
    LWSWITCH_MINION_WR32_SV10(device, instance, _MINION, _LWLINK_LINK_INTR(dlpl_idx), reg);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_intr_dlpl_uncorr
(
    lwswitch_device *device,
    LwU32 link
)
{
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, unhandled;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    if (!LWSWITCH_IS_LINK_ENG_VALID_SV10(device, DLPL, link))
    {
        return -LWL_ERR_ILWALID_STATE;
    }

    report.raw_pending = LWSWITCH_LINK_RD32_SV10(device, link, DLPL, _PLWL, _INTR);
    report.raw_enable = LWSWITCH_LINK_RD32_SV10(device, link, DLPL, _PLWL, _INTR_STALL_EN);
    report.mask = report.raw_enable;
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    report.data[0] = LWSWITCH_LINK_RD32_SV10(device, link, DLPL, _PLWL, _ERROR_COUNT1);

    if (LWSWITCH_PENDING(DRF_NUM(_PLWL, _INTR, _TX_REPLAY, 0x1)))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_DLPL_TX_REPLAY, "DL TX Replay");
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_PLWL, _INTR, _TX_RECOVERY_SHORT, 0x1)))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_DLPL_TX_RECOVERY_SHORT,
                "DL TX Recovery Short");
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_PLWL, _INTR, _TX_RECOVERY_LONG, 0x1)))
    {
        //
        // This error indicates that HW has seen a significant number of bit
        // errors and needs help to recover. Disable all DLPL interrupts and
        // report the SW (fabric manager) the error condition to ilwoke link
        // re-training.
        //
        // Disabling all the interrupts prevents potential interrupt storm in
        // case SW fails to recover HW.
        //
        // Note, we disable the interrupts first and then log the error.
        // This forcefully synchronizes SW's action to re-enable interrupts
        // after link re-training.
        //
        LWSWITCH_LINK_WR32_SV10(device, link, DLPL, _PLWL, _INTR_STALL_EN, 0);

        LWSWITCH_REPORT_NONFATAL(_HW_DLPL_TX_RECOVERY_LONG, "DL TX Recovery Long");
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_PLWL, _INTR, _TX_FAULT_RAM, 0x1)))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_TX_FAULT_RAM, "DL TX Fault RAM", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_PLWL, _INTR, _TX_FAULT_INTERFACE, 0x1)))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_TX_FAULT_INTERFACE, "DL TX Fault Interface", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_PLWL, _INTR, _TX_FAULT_SUBLINK_CHANGE, 0x1)))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_TX_FAULT_SUBLINK_CHANGE,
                "DL TX Fault Sublink Change", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_PLWL, _INTR, _RX_FAULT_SUBLINK_CHANGE, 0x1)))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_RX_FAULT_SUBLINK_CHANGE,
                "DL RX Fault Sublink Change", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_PLWL, _INTR, _RX_FAULT_DL_PROTOCOL, 0x1)))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_RX_FAULT_DL_PROTOCOL,
                "DL RX Fault DL Protocol", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_PLWL, _INTR, _RX_SHORT_ERROR_RATE, 0x1)))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_DLPL_RX_SHORT_ERROR_RATE,
                "DL RX Short Error Rate");
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_PLWL, _INTR, _RX_LONG_ERROR_RATE, 0x1)))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_DLPL_RX_LONG_ERROR_RATE,
                "DL RX Long Error Rate Change");
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_PLWL, _INTR, _RX_ILA_TRIGGER, 0x1)))
    {
        LWSWITCH_REPORT_CORRECTABLE(_HW_DLPL_RX_ILA_TRIGGER, "DL ILA");
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_PLWL, _INTR, _RX_CRC_COUNTER, 0x1)))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_DLPL_RX_CRC_COUNTER, "DL RX CRC counter");
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_PLWL, _INTR, _LTSSM_FAULT, 0x1)))
    {
        LWSWITCH_REPORT_FATAL(_HW_DLPL_LTSSM_FAULT, "DL LTSSM Fault", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_PLWL, _INTR, _LTSSM_PROTOCOL, 0x1)))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_DLPL_LTSSM_PROTOCOL, "DL LTSSM Protocol");
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_PLWL, _INTR, _MINION_REQUEST, 0x1)))
    {
        LWSWITCH_REPORT_NONFATAL(_HW_DLPL_MINION_REQUEST, "DL unexpected MINION request");
        LWSWITCH_HANDLED(bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (chip_device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_LINK_WR32_SV10(device, link, DLPL, _PLWL, _INTR_STALL_EN,
                report.raw_enable ^ pending);
    }

    LWSWITCH_LINK_WR32_SV10(device, link, DLPL, _PLWL, _INTR, pending);

    //
    // Always clear SW2 to cover sideband "err" interfaces to LWLIPT.
    // Interrupts mapped to UCINTERNAL cannot be masked by SW.
    //
    LWSWITCH_LINK_WR32_SV10(device, link, DLPL, _PLWL, _INTR_SW2, 0xffffffff);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

// DLPL "intr"s are wire ORed into the top level interrupt w/o fan-out status bits.
static LwlStatus
_lwswitch_service_intr_dlpl_common_uncorr
(
    lwswitch_device *device,
    LwU32            sioctrl
)
{
    LwU32 link_base = SIOCTRL_TO_LINK(device, sioctrl, 0);
    LwlStatus status[2];

    status[0] = _lwswitch_service_intr_dlpl_uncorr(device, link_base + 0);
    status[1] = _lwswitch_service_intr_dlpl_uncorr(device, link_base + 1);

    if ((status[0] != LWL_SUCCESS) &&
        (status[1] != LWL_SUCCESS))
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_afs_uncorr
(
    lwswitch_device *device,
    LwU32            swx,
    LwU32            afs
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    LwU32 link = AFS_TO_LINK(device, swx, afs);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, unhandled;

    report.raw_pending = LWSWITCH_AFS_RD32_SV10(device, swx, afs, _AFS, _ERR_STATUS_UC);
    report.raw_enable = LWSWITCH_AFS_RD32_SV10(device, swx, afs, _AFS, _ERR_MASK_UC);
    report.mask = ~report.raw_enable;
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    report.raw_first = LWSWITCH_AFS_RD32_SV10(device, swx, afs, _AFS, _ERR_FIRST_UC);

    // AFS error log has fields for different errors.  Always record them.
    report.data[0] = LWSWITCH_AFS_RD32_SV10(device, swx, afs, _AFS, _ERR_LOG0);
    report.data[1] = LWSWITCH_AFS_RD32_SV10(device, swx, afs, _AFS, _ERR_LOG1);
    report.data[2] = LWSWITCH_AFS_RD32_SV10(device, swx, afs, _AFS, _ERR_LOG2);
    report.data[3] = LWSWITCH_AFS_RD32_SV10(device, swx, afs, _AFS, _ERR_LOG3);

    if (LWSWITCH_PENDING(DRF_NUM(_AFS, _ERR_STATUS_UC, _INGRESS_CREDIT_OVERFLOW, 1)))
    {
        LWSWITCH_REPORT_FATAL(_HW_AFS_UC_INGRESS_CREDIT_OVERFLOW,
                "AFS ingress credit overflow", LW_TRUE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_AFS, _ERR_STATUS_UC, _INGRESS_CREDIT_UNDERFLOW, 1)))
    {
        LWSWITCH_REPORT_FATAL(_HW_AFS_UC_INGRESS_CREDIT_UNDERFLOW,
                "AFS ingress credit underflow", LW_TRUE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_AFS, _ERR_STATUS_UC, _EGRESS_CREDIT_OVERFLOW, 1)))
    {
        LWSWITCH_REPORT_FATAL(_HW_AFS_UC_EGRESS_CREDIT_OVERFLOW,
                "AFS egress credit overflow", LW_TRUE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_AFS, _ERR_STATUS_UC, _EGRESS_CREDIT_UNDERFLOW, 1)))
    {
        LWSWITCH_REPORT_FATAL(_HW_AFS_UC_EGRESS_CREDIT_UNDERFLOW,
                "AFS egress credit underflow", LW_TRUE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_AFS, _ERR_STATUS_UC, _INGRESS_NON_BURSTY_PKT_DETECTED, 1)))
    {
        LWSWITCH_REPORT_FATAL(_HW_AFS_UC_INGRESS_NON_BURSTY_PKT_DETECTED,
                "AFS ingress non-bursty packet", LW_TRUE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_AFS, _ERR_STATUS_UC, _INGRESS_NON_STICKY_PKT_DETECTED, 1)))
    {
        LWSWITCH_REPORT_FATAL(_HW_AFS_UC_INGRESS_NON_STICKY_PKT_DETECTED,
                "AFS ingress non-sticky packet", LW_TRUE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_AFS, _ERR_STATUS_UC, _INGRESS_BURST_GT_17_DATA_VC_DETECTED, 1)))
    {
        LWSWITCH_REPORT_FATAL(_HW_AFS_UC_INGRESS_BURST_GT_17_DATA_VC_DETECTED,
                "AFS ingress data burst > 17", LW_TRUE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_AFS, _ERR_STATUS_UC, _INGRESS_BURST_GT_1_NONDATA_VC_DETECTED, 1)))
    {
        LWSWITCH_REPORT_FATAL(_HW_AFS_UC_INGRESS_BURST_GT_1_NONDATA_VC_DETECTED,
                "AFS ingress non-data burst > 1", LW_TRUE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_AFS, _ERR_STATUS_UC, _ILWALID_DST, 1)))
    {
        LWSWITCH_REPORT_FATAL(_HW_AFS_UC_ILWALID_DST, "AFS invalid destination", LW_TRUE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_AFS, _ERR_STATUS_UC, _PKT_MISROUTE, 1)))
    {
        LWSWITCH_REPORT_FATAL(_HW_AFS_UC_PKT_MISROUTE, "AFS packet misroute", LW_TRUE);
        LWSWITCH_HANDLED(bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (chip_device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_AFS_WR32_SV10(device, swx, afs, _AFS, _ERR_MASK_UC,
                report.raw_enable ^ pending);
    }

    // Clear interrupts
    LWSWITCH_AFS_WR32_SV10(device, swx, afs, _AFS, _ERR_FIRST_UC,
            report.raw_first & report.mask);
    LWSWITCH_AFS_WR32_SV10(device, swx, afs, _AFS, _ERR_STATUS_UC, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_swx_uncorr
(
    lwswitch_device *device,
    LwU32            swx
)
{
    LwU32 pending, bit, unhandled;
    LwU32 afs;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    pending = LWSWITCH_SWX_RD32_SV10(device, swx, _SWX, _ERR_FATAL_STATUS);
    pending &= DRF_NUM(_SWX, _ERR_FATAL_STATUS, _AFS_VAR0, 1) |
        DRF_NUM(_SWX, _ERR_FATAL_STATUS, _AFS_VAR1, 1) |
        DRF_NUM(_SWX, _ERR_FATAL_STATUS, _AFS_VAR2, 1) |
        DRF_NUM(_SWX, _ERR_FATAL_STATUS, _AFS_VAR3, 1) |
        DRF_NUM(_SWX, _ERR_FATAL_STATUS, _AFS_VAR4, 1) |
        DRF_NUM(_SWX, _ERR_FATAL_STATUS, _AFS_VAR5, 1) |
        DRF_NUM(_SWX, _ERR_FATAL_STATUS, _AFS_VAR6, 1) |
        DRF_NUM(_SWX, _ERR_FATAL_STATUS, _AFS_VAR7, 1) |
        DRF_NUM(_SWX, _ERR_FATAL_STATUS, _AFS_VAR8, 1);

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    // Loop over all AFS in this SWX
    for (afs = 0; pending && (afs < chip_device->subengSWX[swx].numAFS); afs++)
    {
        LwU32 intr_mask = 1 << chip_device->subengSWX[swx].subengAFS[afs].intr_bit;

        if (LWSWITCH_PENDING(intr_mask))
        {
            if (_lwswitch_service_afs_uncorr(device, swx, afs) == LWL_SUCCESS)
            {
                LWSWITCH_HANDLED(bit);
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

static void
_lwswitch_save_egress_packet_header
(
    lwswitch_device    *device,
    LwU32               npg,
    LwU32               nport,
    LwU32               valid,
    LWSWITCH_RAW_ERROR_LOG_TYPE *data
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 link = NPORT_TO_LINK(device, npg, nport);

    lwswitch_os_memset(data, 0, sizeof(*data));

    //
    // These errors latch _BAD_PACKET_HEADER* the first time they occur for the
    // following four errors when first is clear.  The manual states that HEADER7
    // valid will clear when first is cleared, but we must do a debug_reset in
    // order to clear it.
    //
    // HW BUG 1842500 + 1849499 - details reset behavior of debug_reset.
    //
    if (valid && (LW_FALSE == chip_device->link[link].egress_packet_latched))
    {
        data->data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _BAD_PACKET_HEADER0);
        data->data[1] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _BAD_PACKET_HEADER1);
        data->data[2] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _BAD_PACKET_HEADER2);
        data->data[3] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _BAD_PACKET_HEADER3);
        data->data[4] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _BAD_PACKET_HEADER4);
        data->data[5] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _BAD_PACKET_HEADER5);
        data->data[6] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _BAD_PACKET_HEADER6);
        data->data[7] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _BAD_PACKET_HEADER7);
        if (FLD_TEST_DRF_NUM(_EGRESS, _BAD_PACKET_HEADER7, _VALID, 1, data->data[7]))
        {
            chip_device->link[link].egress_packet_latched = LW_TRUE;
        }
    }
    else
    {
        data->data[0] = 0xdeadbeef;
    }
}

static void
_lwswitch_save_ingress_errorinfo
(
    lwswitch_device    *device,
    LwU32               npg,
    LwU32               nport,
    LWSWITCH_RAW_ERROR_LOG_TYPE *data
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 link = NPORT_TO_LINK(device, npg, nport);

    lwswitch_os_memset(data, 0, sizeof(*data));

    //
    // All ingress errors latch _ERRORINFO*. The manual states that ERRORINFO7
    // valid will clear when first is cleared, but we must do a debug_reset in
    // order to clear it.
    //
    // HW BUG 1842500 + 1849499 - details reset behavior of debug_reset.
    //
    if (LW_FALSE == chip_device->link[link].ingress_packet_latched)
    {
        data->data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _INGRESS, _ERRORINFO0);
        data->data[1] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _INGRESS, _ERRORINFO1);
        data->data[2] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _INGRESS, _ERRORINFO2);
        data->data[3] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _INGRESS, _ERRORINFO3);
        data->data[4] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _INGRESS, _ERRORINFO4);
        data->data[5] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _INGRESS, _ERRORINFO5);
        data->data[6] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _INGRESS, _ERRORINFO6);
        data->data[7] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _INGRESS, _ERRORINFO7);
        if (FLD_TEST_DRF_NUM(_INGRESS, _ERRORINFO7, _VALID, 1, data->data[7]))
        {
            chip_device->link[link].ingress_packet_latched = LW_TRUE;
        }
    }
    else
    {
        data->data[0] = 0xdeadfeed;
    }
}

static LwlStatus
_lwswitch_service_tag_state_uncorr
(
    lwswitch_device *device,
    LwU32            npg,
    LwU32            nport
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 link = NPORT_TO_LINK(device, npg, nport);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, contain, unhandled;
    LWSWITCH_RAW_ERROR_LOG_TYPE data = {{0}};

    report.raw_pending = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _TSTATE, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _TSTATE, _ERR_REPORT_EN_0);
    report.mask = report.raw_enable &
        (DRF_NUM(_TSTATE, _ERR_STATUS_0, _TAGPOOLBUFERR, 1) |
         DRF_NUM(_TSTATE, _ERR_STATUS_0, _CRUMBSTOREBUFERR, 1) |
         DRF_NUM(_TSTATE, _ERR_STATUS_0, _UNCORRECTABLEECCERR_CRUMBSTORE, 1) |
         DRF_NUM(_TSTATE, _ERR_STATUS_0, _UNCORRECTABLEECCERR_TAGSTORE, 1));
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    report.raw_first = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _TSTATE, _ERR_FIRST_0);
    contain = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _TSTATE, _ERR_CONTAIN_EN_0);

    //
    // Depending on the error type flush state can latch the packet header in
    // ingress, egress or both.  This is briefly mentioned in the IAS, but see
    // ftstate_error_capture and capture_tserror in the RTL.
    //
    if (LWSWITCH_PENDING(DRF_NUM(_TSTATE, _ERR_STATUS_0, _TAGPOOLBUFERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_TSTATE_TAGPOOLBUFERR,
                "TS pointer crossover", LW_FALSE);
        _lwswitch_save_ingress_errorinfo(device, npg, nport, &data);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_INGRESS_ERRORINFO, data);
        _lwswitch_save_egress_packet_header(device, npg, nport, LW_TRUE, &data);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_PACKET_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_TSTATE, _ERR_STATUS_0, _CRUMBSTOREBUFERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_TSTATE_CRUMBSTOREBUFERR,
                "TS crumbstore", LW_FALSE);
        _lwswitch_save_ingress_errorinfo(device, npg, nport, &data);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_INGRESS_ERRORINFO, data);
        _lwswitch_save_egress_packet_header(device, npg, nport, LW_TRUE, &data);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_PACKET_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_TSTATE, _ERR_STATUS_0, _UNCORRECTABLEECCERR_CRUMBSTORE, 1)))
    {
        report.data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _TSTATE, _ECC_ERROR_COUNT_CRUMBSTORE);
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_TSTATE_UNCORRECTABLEECCERR_CRUMBSTORE,
                "TS crumbstore fatal ECC", LW_FALSE);
        _lwswitch_save_ingress_errorinfo(device, npg, nport, &data);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_INGRESS_ERRORINFO, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_TSTATE, _ERR_STATUS_0, _UNCORRECTABLEECCERR_TAGSTORE, 1)))
    {
        report.data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _TSTATE, _ECC_ERROR_COUNT_TAGSTORE);
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_TSTATE_UNCORRECTABLEECCERR_TAGSTORE,
                "TS tag store fatal ECC", LW_FALSE);
        _lwswitch_save_egress_packet_header(device, npg, nport, LW_TRUE, &data);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_PACKET_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (chip_device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _TSTATE, _ERR_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _TSTATE, _ERR_FIRST_0,
            report.raw_first & report.mask);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _TSTATE, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_tag_state_corr
(
    lwswitch_device *device,
    LwU32            npg,
    LwU32            nport
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 link = NPORT_TO_LINK(device, npg, nport);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, unhandled;
    LWSWITCH_RAW_ERROR_LOG_TYPE data;

    report.raw_pending = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _TSTATE, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _TSTATE, _ERR_REPORT_EN_0);
    report.mask = report.raw_enable &
        (DRF_NUM(_TSTATE, _ERR_STATUS_0, _SINGLEBITECCLIMITERR_CRUMBSTORE, 1) |
         DRF_NUM(_TSTATE, _ERR_STATUS_0, _SINGLEBITECCLIMITERR_TAGSTORE, 1));
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    report.raw_first = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _TSTATE, _ERR_FIRST_0);

    if (LWSWITCH_PENDING(DRF_NUM(_TSTATE, _ERR_STATUS_0, _SINGLEBITECCLIMITERR_CRUMBSTORE, 1)))
    {
        report.data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _TSTATE, _ECC_ERROR_COUNT_CRUMBSTORE);
        LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _TSTATE, _ECC_ERROR_COUNT_CRUMBSTORE,
                DRF_DEF(_TSTATE, _ECC_ERROR_COUNT_CRUMBSTORE, _ERROR_COUNT, _INIT) |
                DRF_DEF(_TSTATE, _ECC_ERROR_LIMIT_CRUMBSTORE, _ERROR_LIMIT, _INIT));
        LWSWITCH_REPORT_CORRECTABLE(_HW_NPORT_TSTATE_SINGLEBITECCLIMITERR_CRUMBSTORE,
                "TS crumbstore single-bit threshold");
        _lwswitch_save_ingress_errorinfo(device, npg, nport, &data);
        LWSWITCH_REPORT_DATA(_HW_NPORT_INGRESS_ERRORINFO, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_TSTATE, _ERR_STATUS_0, _SINGLEBITECCLIMITERR_TAGSTORE, 1)))
    {
        report.data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _TSTATE, _ECC_ERROR_COUNT_TAGSTORE);
        LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _TSTATE, _ECC_ERROR_COUNT_TAGSTORE,
                DRF_DEF(_TSTATE, _ECC_ERROR_COUNT_TAGSTORE, _ERROR_COUNT, _INIT) |
                DRF_DEF(_TSTATE, _ECC_ERROR_LIMIT_TAGSTORE, _ERROR_LIMIT, _INIT));
        LWSWITCH_REPORT_CORRECTABLE(_HW_NPORT_TSTATE_SINGLEBITECCLIMITERR_TAGSTORE,
                "TS tag store single-bit threshold");
        _lwswitch_save_egress_packet_header(device, npg, nport, LW_TRUE, &data);
        LWSWITCH_REPORT_DATA(_HW_NPORT_EGRESS_PACKET_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (chip_device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _TSTATE, _ERR_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _TSTATE, _ERR_FIRST_0,
            report.raw_first & report.mask);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _TSTATE, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_flush_state_uncorr
(
    lwswitch_device *device,
    LwU32            npg,
    LwU32            nport
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 link = NPORT_TO_LINK(device, npg, nport);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, contain, unhandled;
    LWSWITCH_RAW_ERROR_LOG_TYPE data;

    report.raw_pending = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _FSTATE, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _FSTATE, _ERR_REPORT_EN_0);
    report.mask = report.raw_enable &
        (DRF_NUM(_FSTATE, _ERR_STATUS_0, _TAGPOOLBUFERR, 1) |
         DRF_NUM(_FSTATE, _ERR_STATUS_0, _CRUMBSTOREBUFERR, 1) |
         DRF_NUM(_FSTATE, _ERR_STATUS_0, _UNCORRECTABLEECCERR_CRUMBSTORE, 1) |
         DRF_NUM(_FSTATE, _ERR_STATUS_0, _UNCORRECTABLEECCERR_TAGSTORE, 1) |
         DRF_NUM(_FSTATE, _ERR_STATUS_0, _UNCORRECTABLEECCERR_FLUSHREQSTORE, 1));
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    report.raw_first = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _FSTATE, _ERR_FIRST_0);
    contain = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _FSTATE, _ERR_CONTAIN_EN_0);

    //
    // Depending on the error type flush state can latch the packet header in
    // ingress, egress or both.  This is briefly mentioned in the IAS, but see
    // ftstate_error_capture and capture_fserror in the RTL.
    //
    if (LWSWITCH_PENDING(DRF_NUM(_FSTATE, _ERR_STATUS_0, _TAGPOOLBUFERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_FSTATE_TAGPOOLBUFERR,
                "FS pointer crossover", LW_FALSE);
        _lwswitch_save_ingress_errorinfo(device, npg, nport, &data);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_INGRESS_ERRORINFO, data);
        _lwswitch_save_egress_packet_header(device, npg, nport, LW_TRUE, &data);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_PACKET_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_FSTATE, _ERR_STATUS_0, _CRUMBSTOREBUFERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_FSTATE_CRUMBSTOREBUFERR,
                "FS crumbstore", LW_FALSE);
        _lwswitch_save_ingress_errorinfo(device, npg, nport, &data);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_INGRESS_ERRORINFO, data);
        _lwswitch_save_egress_packet_header(device, npg, nport, LW_TRUE, &data);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_PACKET_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_FSTATE, _ERR_STATUS_0, _UNCORRECTABLEECCERR_CRUMBSTORE, 1)))
    {
        report.data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _FSTATE, _ECC_ERROR_COUNT_CRUMBSTORE);
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_FSTATE_UNCORRECTABLEECCERR_CRUMBSTORE,
                "FS crumbstore fatal ECC", LW_FALSE);
        _lwswitch_save_egress_packet_header(device, npg, nport, LW_TRUE, &data);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_PACKET_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_FSTATE, _ERR_STATUS_0, _UNCORRECTABLEECCERR_TAGSTORE, 1)))
    {
        report.data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _FSTATE, _ECC_ERROR_COUNT_TAGSTORE);
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_FSTATE_UNCORRECTABLEECCERR_TAGSTORE,
                "FS tag store fatal ECC", LW_FALSE);
        _lwswitch_save_ingress_errorinfo(device, npg, nport, &data);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_INGRESS_ERRORINFO, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_FSTATE, _ERR_STATUS_0, _UNCORRECTABLEECCERR_FLUSHREQSTORE, 1)))
    {
        report.data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _FSTATE, _ECC_ERROR_COUNT_FLUSHREQ);
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_FSTATE_UNCORRECTABLEECCERR_FLUSHREQSTORE,
                "FS flush req fatal ECC", LW_FALSE);
        _lwswitch_save_egress_packet_header(device, npg, nport, LW_TRUE, &data);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_PACKET_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (chip_device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _FSTATE, _ERR_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _FSTATE, _ERR_FIRST_0,
            report.raw_first & report.mask);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _FSTATE, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_flush_state_corr
(
    lwswitch_device *device,
    LwU32            npg,
    LwU32            nport
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 link = NPORT_TO_LINK(device, npg, nport);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, unhandled;
    LWSWITCH_RAW_ERROR_LOG_TYPE data;

    report.raw_pending = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _FSTATE, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _FSTATE, _ERR_REPORT_EN_0);
    report.mask = report.raw_enable &
        (DRF_NUM(_FSTATE, _ERR_STATUS_0, _SINGLEBITECCLIMITERR_CRUMBSTORE, 1) |
         DRF_NUM(_FSTATE, _ERR_STATUS_0, _SINGLEBITECCLIMITERR_TAGSTORE, 1) |
         DRF_NUM(_FSTATE, _ERR_STATUS_0, _SINGLEBITECCLIMITERR_FLUSHREQSTORE, 1));
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    report.raw_first = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _FSTATE, _ERR_FIRST_0);

    if (LWSWITCH_PENDING(DRF_NUM(_FSTATE, _ERR_STATUS_0, _SINGLEBITECCLIMITERR_CRUMBSTORE, 1)))
    {
        report.data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _FSTATE, _ECC_ERROR_COUNT_CRUMBSTORE);
        LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _FSTATE, _ECC_ERROR_COUNT_CRUMBSTORE, 0);
        LWSWITCH_REPORT_CORRECTABLE(_HW_NPORT_FSTATE_SINGLEBITECCLIMITERR_CRUMBSTORE,
                "FS crumbstore single-bit limit");
        _lwswitch_save_egress_packet_header(device, npg, nport, LW_TRUE, &data);
        LWSWITCH_REPORT_DATA(_HW_NPORT_EGRESS_PACKET_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_FSTATE, _ERR_STATUS_0, _SINGLEBITECCLIMITERR_TAGSTORE, 1)))
    {
        report.data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _FSTATE, _ECC_ERROR_COUNT_TAGSTORE);
        LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _FSTATE, _ECC_ERROR_COUNT_TAGSTORE, 0);
        LWSWITCH_REPORT_CORRECTABLE(_HW_NPORT_FSTATE_SINGLEBITECCLIMITERR_TAGSTORE,
                "FS tag store single-bit limit");
        _lwswitch_save_ingress_errorinfo(device, npg, nport, &data);
        LWSWITCH_REPORT_DATA(_HW_NPORT_INGRESS_ERRORINFO, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_FSTATE, _ERR_STATUS_0, _SINGLEBITECCLIMITERR_FLUSHREQSTORE, 1)))
    {
        report.data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _FSTATE, _ECC_ERROR_COUNT_FLUSHREQ);
        LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _FSTATE, _ECC_ERROR_COUNT_FLUSHREQ, 0);
        LWSWITCH_REPORT_CORRECTABLE(_HW_NPORT_FSTATE_SINGLEBITECCLIMITERR_FLUSHREQSTORE,
                "FS flush req single-bit limit");
        _lwswitch_save_egress_packet_header(device, npg, nport, LW_TRUE, &data);
        LWSWITCH_REPORT_DATA(_HW_NPORT_EGRESS_PACKET_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (chip_device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _FSTATE, _ERR_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _FSTATE, _ERR_FIRST_0,
            report.raw_first & report.mask);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _FSTATE, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_ingress_uncorr
(
    lwswitch_device *device,
    LwU32            npg,
    LwU32            nport
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 link = NPORT_TO_LINK(device, npg, nport);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, contain, unhandled;
    LWSWITCH_RAW_ERROR_LOG_TYPE data;

    report.raw_pending = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _INGRESS, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _INGRESS, _ERR_REPORT_EN_0);
    report.mask = report.raw_enable &
        (DRF_NUM(_INGRESS, _ERR_STATUS_0, _CMDDECODEERR, 1) |
         DRF_NUM(_INGRESS, _ERR_STATUS_0, _BDFMISMATCHERR, 1) |
         DRF_NUM(_INGRESS, _ERR_STATUS_0, _BUBBLEDETECT, 1) |
         DRF_NUM(_INGRESS, _ERR_STATUS_0, _ACLFAIL, 1) |
         DRF_NUM(_INGRESS, _ERR_STATUS_0, _PKTPOISONSET, 1) |
         DRF_NUM(_INGRESS, _ERR_STATUS_0, _ECCHDRDOUBLEBITERR, 1) |
         DRF_NUM(_INGRESS, _ERR_STATUS_0, _ILWALIDCMD, 1) |
         DRF_NUM(_INGRESS, _ERR_STATUS_0, _ILWALIDVCSET, 1));
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    report.raw_first = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _INGRESS, _ERR_FIRST_0);
    contain = LWSWITCH_INGRESS_CONTAIN_EN;

    _lwswitch_save_ingress_errorinfo(device, npg, nport, &data);

    if (LWSWITCH_PENDING(DRF_NUM(_INGRESS, _ERR_STATUS_0, _CMDDECODEERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_INGRESS_CMDDECODEERR,
                "ingress invalid command", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_NPORT_INGRESS_ERRORINFO, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_INGRESS, _ERR_STATUS_0, _BDFMISMATCHERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_INGRESS_BDFMISMATCHERR,
                "ingress BDF CAM lookup", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_NPORT_INGRESS_ERRORINFO, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_INGRESS, _ERR_STATUS_0, _BUBBLEDETECT, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_INGRESS_BUBBLEDETECT,
                "ingress detected intra packet bubble", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_NPORT_INGRESS_ERRORINFO, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_INGRESS, _ERR_STATUS_0, _ACLFAIL, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_INGRESS_ACLFAIL,
                "ingress invalid ACL", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_NPORT_INGRESS_ERRORINFO, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_INGRESS, _ERR_STATUS_0, _PKTPOISONSET, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_INGRESS_PKTPOISONSET,
                "ingress recieved poison packet", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_NPORT_INGRESS_ERRORINFO, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_INGRESS, _ERR_STATUS_0, _ECCHDRDOUBLEBITERR, 1)))
    {
        // Save ECC info.  Count will be cleared when it passes limit.
        report.data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _INGRESS, _ECC_ERROR_COUNT);
        report.data[1] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _INGRESS, _ECC_ERROR_ADDRESS);
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_INGRESS_ECCHDRDOUBLEBITERR,
                "ingress header multi-bit ECC", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_NPORT_INGRESS_ERRORINFO, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_INGRESS, _ERR_STATUS_0, _ILWALIDCMD, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_INGRESS_ILWALIDCMD,
                "ingress invalid ID/CMD", LW_FALSE);
        // HW BUG 1848374 - HW does not capture the header in some cases.
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_NPORT_INGRESS_ERRORINFO, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_INGRESS, _ERR_STATUS_0, _ILWALIDVCSET, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_INGRESS_ILWALIDVCSET,
                "ingress invalid VCSet", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_NPORT_INGRESS_ERRORINFO, data);
        LWSWITCH_HANDLED(bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (chip_device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _INGRESS, _ERR_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    // Also clear LW_INGRESS_ERR_FIRST_0_ENCODEDVC bits
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _INGRESS, _ERR_FIRST_0,
            (report.raw_first & report.mask) |
            DRF_NUM(_INGRESS, _ERR_FIRST_0, _ENCODEDVC, 0xff));
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _INGRESS, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_ingress_corr
(
    lwswitch_device *device,
    LwU32            npg,
    LwU32            nport
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 link = NPORT_TO_LINK(device, npg, nport);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, unhandled;
    LWSWITCH_RAW_ERROR_LOG_TYPE data;

    report.raw_pending = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _INGRESS, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _INGRESS, _ERR_REPORT_EN_0);
    report.mask = report.raw_enable &
        DRF_NUM(_INGRESS, _ERR_STATUS_0, _ECCSOFTLIMITERR, 1);
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    report.raw_first = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _INGRESS, _ERR_FIRST_0);

    _lwswitch_save_ingress_errorinfo(device, npg, nport, &data);

    //
    // _ECCSOFTLIMITERR has been disabled due to HW bug 1973042. So, we shouldn't
    // reach here. _ECCSOFTLIMITERR is being handled by _lwswitch_ingress_ecc_writeback.
    //
    if (LWSWITCH_PENDING(DRF_NUM(_INGRESS, _ERR_STATUS_0, _ECCSOFTLIMITERR, 1)))
    {
        LWSWITCH_ASSERT(0);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (chip_device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _INGRESS, _ERR_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    // Correctable errors do not set LW_INGRESS_ERR_FIRST_0_ENCODEDVC
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _INGRESS, _ERR_FIRST_0,
            report.raw_first & report.mask);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _INGRESS, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_egress_uncorr
(
    lwswitch_device *device,
    LwU32            npg,
    LwU32            nport
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 link = NPORT_TO_LINK(device, npg, nport);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, contain, unhandled;
    LWSWITCH_RAW_ERROR_LOG_TYPE data = {{0}};

    report.raw_pending = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _ERR_REPORT_EN_0);
    report.mask = report.raw_enable &
        (DRF_NUM(_EGRESS, _ERR_STATUS_0, _EGRESSBUFERR, 1) |
         DRF_NUM(_EGRESS, _ERR_STATUS_0, _PKTROUTEERR, 1) |
         DRF_NUM(_EGRESS, _ERR_STATUS_0, _ECCHDRDOUBLEBITERR0, 1) |
         DRF_NUM(_EGRESS, _ERR_STATUS_0, _ECCHDRDOUBLEBITERR1, 1) |
         DRF_NUM(_EGRESS, _ERR_STATUS_0, _NCISOCHDRCREDITOVFL, 1) |
         DRF_NUM(_EGRESS, _ERR_STATUS_0, _NCISOCDATACREDITOVFL, 1) |
         DRF_NUM(_EGRESS, _ERR_STATUS_0, _ADDRMATCHERR, 1) |
         DRF_NUM(_EGRESS, _ERR_STATUS_0, _TAGCOUNTERR, 1) |
         DRF_NUM(_EGRESS, _ERR_STATUS_0, _FLUSHRSPERR, 1) |
         DRF_NUM(_EGRESS, _ERR_STATUS_0, _DROPNPURRSPERR, 1) |
         DRF_NUM(_EGRESS, _ERR_STATUS_0, _POISONERR, 1));
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    report.raw_first = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _ERR_FIRST_0);
    contain = LWSWITCH_EGRESS_CONTAIN_EN;

    if (LWSWITCH_PENDING(DRF_NUM(_EGRESS, _ERR_STATUS_0, _EGRESSBUFERR, 1)))
    {
        data.data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _BUFFER_POINTERS0);
        data.data[1] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _BUFFER_POINTERS1);
        data.data[2] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _BUFFER_POINTERS2);
        data.data[3] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _BUFFER_POINTERS3);
        data.data[4] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _BUFFER_POINTERS4);
        data.data[5] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _BUFFER_POINTERS5);
        data.data[6] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _BUFFER_POINTERS6);
        data.data[7] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _BUFFER_POINTERS7);
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_EGRESSBUFERR,
                "egress crossbar overflow", LW_TRUE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_BUFFER_DATA, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_EGRESS, _ERR_STATUS_0, _ECCHDRDOUBLEBITERR0, 1)))
    {
        // No combined ECC count, but save correctable anyway
        report.data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _ECC_CORRECTABLE_COUNT_0);
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_ECCHDRDOUBLEBITERR0,
                "egress multi-bit header ECC 0", LW_TRUE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_EGRESS, _ERR_STATUS_0, _ECCHDRDOUBLEBITERR1, 1)))
    {
        // No combined ECC count, but save correctable anyway
        report.data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _ECC_CORRECTABLE_COUNT_1);
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_ECCHDRDOUBLEBITERR1,
                "egress multi-bit header ECC 1", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_EGRESS, _ERR_STATUS_0, _NCISOCHDRCREDITOVFL, 1) |
                DRF_NUM(_EGRESS, _ERR_STATUS_0, _NCISOCDATACREDITOVFL, 1)))
    {
        data.data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _NCISOC_CREDIT0);
        data.data[1] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _NCISOC_CREDIT1);
        data.data[2] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _NCISOC_CREDIT2);
        data.data[3] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _NCISOC_CREDIT3);
        data.data[4] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _NCISOC_CREDIT4);
        data.data[5] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _NCISOC_CREDIT5);
        data.data[6] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _NCISOC_CREDIT6);
        data.data[7] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _NCISOC_CREDIT7);

        if (LWSWITCH_PENDING(DRF_NUM(_EGRESS, _ERR_STATUS_0, _NCISOCHDRCREDITOVFL, 1)))
        {
            LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_NCISOCHDRCREDITOVFL,
                    "egress header credit overflow", LW_FALSE);
            LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_NCISOC_CREDITS, data);
            LWSWITCH_HANDLED(bit);
        }
        if (LWSWITCH_PENDING(DRF_NUM(_EGRESS, _ERR_STATUS_0, _NCISOCDATACREDITOVFL, 1)))
        {
            LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_NCISOCDATACREDITOVFL,
                    "egress data credit overflow", LW_FALSE);
            LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_NCISOC_CREDITS, data);
            LWSWITCH_HANDLED(bit);
        }
    }
    if (LWSWITCH_PENDING(DRF_NUM(_EGRESS, _ERR_STATUS_0, _TAGCOUNTERR, 1)))
    {
        data.data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _TAGCOUNTS0);
        data.data[1] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _TAGCOUNTS1);
        data.data[2] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _TAGCOUNTS2);
        data.data[3] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _TAGCOUNTS3);
        data.data[4] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _TAGCOUNTS4);
        data.data[5] = 0;
        data.data[6] = 0;
        data.data[7] = 0;
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_TAGCOUNTERR,
                "egress tag counter underflow", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA(_HW_NPORT_EGRESS_TAG_DATA, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_EGRESS, _ERR_STATUS_0, _POISONERR, 1)))
    {
        // This interrupt should not be enabled
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_POISONERR,
                "egress data poison packet", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }

    //
    // These errors latch _BAD_PACKET_HEADER*.  See helper function for caveats.
    //
    if (LWSWITCH_PENDING(DRF_NUM(_EGRESS, _ERR_STATUS_0, _PKTROUTEERR, 1) |
                DRF_NUM(_EGRESS, _ERR_STATUS_0, _ADDRMATCHERR, 1) |
                DRF_NUM(_EGRESS, _ERR_STATUS_0, _FLUSHRSPERR, 1) |
                DRF_NUM(_EGRESS, _ERR_STATUS_0, _DROPNPURRSPERR, 1)))
    {
        _lwswitch_save_egress_packet_header(device, npg, nport, report.raw_first & bit, &data);

        if (LWSWITCH_PENDING(DRF_NUM(_EGRESS, _ERR_STATUS_0, _PKTROUTEERR, 1)))
        {
            LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_PKTROUTEERR,
                    "egress packet route", LW_TRUE);
            LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_NPORT_EGRESS_PACKET_HEADER, data);
            LWSWITCH_HANDLED(bit);
        }
        if (LWSWITCH_PENDING(DRF_NUM(_EGRESS, _ERR_STATUS_0, _ADDRMATCHERR, 1)))
        {
            LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_ADDRMATCHERR,
                    "egress address out of range", LW_FALSE);
            LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_NPORT_EGRESS_PACKET_HEADER, data);
            LWSWITCH_HANDLED(bit);
        }
        if (LWSWITCH_PENDING(DRF_NUM(_EGRESS, _ERR_STATUS_0, _FLUSHRSPERR, 1)))
        {
            LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_FLUSHRSPERR,
                    "egress flush error response", LW_FALSE);
            LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_NPORT_EGRESS_PACKET_HEADER, data);
            LWSWITCH_HANDLED(bit);
        }
        if (LWSWITCH_PENDING(DRF_NUM(_EGRESS, _ERR_STATUS_0, _DROPNPURRSPERR, 1)))
        {
            LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_DROPNPURRSPERR,
                    "egress non-posted UR", LW_FALSE);
            LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_NPORT_EGRESS_PACKET_HEADER, data);
            LWSWITCH_HANDLED(bit);
        }
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (chip_device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _EGRESS, _ERR_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _EGRESS, _ERR_FIRST_0,
            report.raw_first & report.mask);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _EGRESS, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_egress_corr
(
    lwswitch_device *device,
    LwU32            npg,
    LwU32            nport
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 link = NPORT_TO_LINK(device, npg, nport);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, unhandled;

    report.raw_pending = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _ERR_REPORT_EN_0);
    report.mask = report.raw_enable &
        (DRF_NUM(_EGRESS, _ERR_STATUS_0, _ECCSINGLEBITLIMITERR0, 1) |
         DRF_NUM(_EGRESS, _ERR_STATUS_0, _ECCSINGLEBITLIMITERR1, 1));
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    report.raw_first = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _ERR_FIRST_0);

    if (LWSWITCH_PENDING(DRF_NUM(_EGRESS, _ERR_STATUS_0, _ECCSINGLEBITLIMITERR0, 1)))
    {
        report.data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _ECC_CORRECTABLE_COUNT_0);
        LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _EGRESS, _ECC_CORRECTABLE_COUNT_0, 0);
        LWSWITCH_REPORT_CORRECTABLE(_HW_NPORT_EGRESS_ECCSINGLEBITLIMITERR0,
                "egress single-bit limit 0");
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_EGRESS, _ERR_STATUS_0, _ECCSINGLEBITLIMITERR1, 1)))
    {
        report.data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _ECC_CORRECTABLE_COUNT_1);
        LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _EGRESS, _ECC_CORRECTABLE_COUNT_1, 0);
        LWSWITCH_REPORT_CORRECTABLE(_HW_NPORT_EGRESS_ECCSINGLEBITLIMITERR1,
                "egress single-bit limit 1");
        LWSWITCH_HANDLED(bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (chip_device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _EGRESS, _ERR_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _EGRESS, _ERR_FIRST_0,
            report.raw_first & report.mask);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _EGRESS, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_egress_poisoned
(
    lwswitch_device *device,
    LwU32            npg,
    LwU32            nport
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 link = NPORT_TO_LINK(device, npg, nport);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, contain, unhandled;

    report.raw_pending = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _ERR_REPORT_EN_0);
    report.mask = report.raw_enable &
        (DRF_NUM(_EGRESS, _ERR_STATUS_0, _ECCDATADOUBLEBITERR0, 1) |
         DRF_NUM(_EGRESS, _ERR_STATUS_0, _ECCDATADOUBLEBITERR1, 1));
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    report.raw_first = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _ERR_FIRST_0);
    contain = LWSWITCH_EGRESS_CONTAIN_EN;

    if (LWSWITCH_PENDING(DRF_NUM(_EGRESS, _ERR_STATUS_0, _ECCDATADOUBLEBITERR0, 1)))
    {
        // No combined ECC count, but save correctable anyway
        report.data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _ECC_CORRECTABLE_COUNT_0);
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_ECCDATADOUBLEBITERR0,
                "egress multi-bit data ECC 0", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_EGRESS, _ERR_STATUS_0, _ECCDATADOUBLEBITERR1, 1)))
    {
        // No combined ECC count, but save correctable anyway
        report.data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _EGRESS, _ECC_CORRECTABLE_COUNT_1);
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_EGRESS_ECCDATADOUBLEBITERR1,
                "egress multi-bit header ECC 1", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (chip_device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _EGRESS, _ERR_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _EGRESS, _ERR_FIRST_0,
            report.raw_first & report.mask);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _EGRESS, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_route_uncorr
(
    lwswitch_device *device,
    LwU32            npg,
    LwU32            nport
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 link = NPORT_TO_LINK(device, npg, nport);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, contain, unhandled;
    LWSWITCH_RAW_ERROR_LOG_TYPE data = {{0}};

    report.raw_pending = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _ERR_REPORT_EN_0);
    report.mask = report.raw_enable &
        (DRF_DEF(_ROUTE, _ERR_STATUS_0, _ROUTEBUFERR, _DETECTED) |
         DRF_DEF(_ROUTE, _ERR_STATUS_0, _NOPORTDEFINEDERR, _DETECTED) |
         DRF_DEF(_ROUTE, _ERR_STATUS_0, _ILWALIDROUTEPOLICYERR, _DETECTED) |
         DRF_DEF(_ROUTE, _ERR_STATUS_0, _UNCORRECTABLEECCERR, _DETECTED) |
         DRF_DEF(_ROUTE, _ERR_STATUS_0, _TRANSDONERESVERR, _DETECTED));
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    report.raw_first = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _ERR_FIRST_0);
    contain = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _ERR_CONTAIN_EN_0);

    //
    // The bad packet header applies to to the following UC errors:
    //    _NOPORTDEFINEDERR, _UNCORRECTABLEECCER, _ILWALIDROUTEPOLICYERR
    //
    data.data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _BADPACKETHEADER0);
    data.data[1] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _BADPACKETHEADER1);
    data.data[2] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _BADPACKETHEADER2);
    data.data[3] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _BADPACKETHEADER3);
    data.data[4] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _BADPACKETHEADER4);
    data.data[5] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _BADPACKETHEADER5);
    data.data[6] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _BADPACKETHEADER6);
    data.data[7] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _ERRORINFO);

    if (LWSWITCH_PENDING(DRF_DEF(_ROUTE, _ERR_STATUS_0, _ROUTEBUFERR, _DETECTED)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_ROUTE_ROUTEBUFERR,
                "route buffer over/underflow", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_DEF(_ROUTE, _ERR_STATUS_0, _NOPORTDEFINEDERR, _DETECTED)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_ROUTE_NOPORTDEFINEDERR,
                "route undefined route", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_NPORT_ROUTE_PACKET_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_DEF(_ROUTE, _ERR_STATUS_0, _ILWALIDROUTEPOLICYERR, _DETECTED)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_ROUTE_ILWALIDROUTEPOLICYERR,
                "route invalid policy", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_NPORT_ROUTE_PACKET_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_DEF(_ROUTE, _ERR_STATUS_0, _UNCORRECTABLEECCERR, _DETECTED)))
    {
        // Route ganged-link DBE ECC error
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_ROUTE_UNCORRECTABLEECCERR,
                "route multi-bit ECC", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_NPORT_ROUTE_PACKET_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_DEF(_ROUTE, _ERR_STATUS_0, _TRANSDONERESVERR, _DETECTED)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_NPORT_ROUTE_TRANSDONERESVERR,
                "route transdone over/underflow", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (chip_device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _ROUTE, _ERR_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    // Only clear first if our error is first to preserve _ENCODEDVC.
    if (report.raw_first & report.mask)
    {
        LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _ROUTE, _ERR_FIRST_0,
                (report.raw_first & report.mask) |
                DRF_NUM(_ROUTE, _ERR_FIRST_0, _ENCODEDVC, 0xff));
    }
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _ROUTE, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_route_corr
(
    lwswitch_device *device,
    LwU32            npg,
    LwU32            nport
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 link = NPORT_TO_LINK(device, npg, nport);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, unhandled;
    LWSWITCH_RAW_ERROR_LOG_TYPE data = {{0}};

    report.raw_pending = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _ERR_REPORT_EN_0);
    report.mask = report.raw_enable &
        DRF_DEF(_ROUTE, _ERR_STATUS_0, _ECCLIMITERR, _DETECTED);
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    report.raw_first = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _ERR_FIRST_0);

    data.data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _BADPACKETHEADER0);
    data.data[1] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _BADPACKETHEADER1);
    data.data[2] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _BADPACKETHEADER2);
    data.data[3] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _BADPACKETHEADER3);
    data.data[4] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _BADPACKETHEADER4);
    data.data[5] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _BADPACKETHEADER5);
    data.data[6] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _BADPACKETHEADER6);
    data.data[7] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _ERRORINFO);

    if (LWSWITCH_PENDING(DRF_DEF(_ROUTE, _ERR_STATUS_0, _ECCLIMITERR, _DETECTED)))
    {
        // Route ganged-link SBE ECC error
        report.data[0] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _ECC_ERROR_COUNT);
        report.data[1] = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _ROUTE, _ECC_ERROR_ADDRESS);

        if (FLD_TEST_DRF(_ROUTE, _ECC_ERROR_ADDRESS, _VALID, _VALID, report.data[1]))
        {
            //
            // ROUTE unit ECC write-back WAR. See Bug: 1836454.
            // See IAS "7.5.11.1.1 Monitoring and Correction Single Bit ECC
            // Errors" for the WAR details.
            //
            LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _NPORT, _CTRL_STOP,
                DRF_DEF(_NPORT, _CTRL_STOP, _INGRESS_STOP, _STOP));

            LWSWITCH_FLUSH_MMIO(device);

            lwswitch_set_ganged_link_table_sv10(device, link,
                    DRF_VAL(_ROUTE, _ECC_ERROR_ADDRESS, _ADDRESS, report.data[1]), 1);

            LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _NPORT, _CTRL_STOP,
                DRF_DEF(_NPORT, _CTRL_STOP, _INGRESS_STOP, _ALLOWTRAFFIC));
        }

        LWSWITCH_REPORT_CORRECTABLE(_HW_NPORT_ROUTE_ECCLIMITERR,
                "route single-bit limit");
        LWSWITCH_REPORT_DATA_FIRST(_HW_NPORT_ROUTE_PACKET_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (chip_device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _ROUTE, _ERR_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    // Only clear first if our error is first to preserve _ENCODEDVC.
    if (report.raw_first & report.mask)
    {
        LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _ROUTE, _ERR_FIRST_0,
                (report.raw_first & report.mask) |
                DRF_NUM(_ROUTE, _ERR_FIRST_0, _ENCODEDVC, 0xff));
    }

    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _ROUTE, _ERR_STATUS_0, pending);

    //
    // Note, when traffic is flowing, if we reset ERR_COUNT before ERR_STATUS
    // register, we won't see an interrupt again until counter wraps around.
    // In that case, we will miss writing back many ECC victim entries. Hence,
    // always clear _ERR_COUNT only after _ERR_STATUS register is cleared!
    //
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _ROUTE, _ECC_ERROR_COUNT, 0x0);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_nport_uncorr
(
    lwswitch_device *device,
    LwU32            npg,
    LwU32            nport
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 link = NPORT_TO_LINK(device, npg, nport);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, unhandled;
    LwlStatus status[5];

    report.raw_pending = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _NPORT, _ERR_UC_STATUS_NPORT);
    report.raw_enable = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _NPORT, _ERR_UC_MASK_NPORT);
    report.mask = ~report.raw_enable;
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    report.raw_first = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _NPORT, _ERR_UC_FIRST_NPORT);

    // Only these two bits are implemented in RTL
    if (LWSWITCH_PENDING(DRF_NUM(_NPORT, _ERR_UC_STATUS_NPORT, _DATAPOISONED, 1)))
    {
        LWSWITCH_REPORT_TREE(_HW_NPORT_DATAPOISONED);
        if (_lwswitch_service_egress_poisoned(device, npg, nport) == LWL_SUCCESS)
        {
            LWSWITCH_HANDLED(bit);
        }
    }

    if (LWSWITCH_PENDING(DRF_NUM(_NPORT, _ERR_UC_STATUS_NPORT, _UCINTERNAL, 1)))
    {
        LWSWITCH_REPORT_TREE(_HW_NPORT_UCINTERNAL);

        status[0] = _lwswitch_service_tag_state_uncorr(device, npg, nport);
        status[1] = _lwswitch_service_flush_state_uncorr(device, npg, nport);
        status[2] = _lwswitch_service_ingress_uncorr(device, npg, nport);
        status[3] = _lwswitch_service_egress_uncorr(device, npg, nport);
        status[4] = _lwswitch_service_route_uncorr(device, npg, nport);

        if ((status[0] == LWL_SUCCESS) ||
            (status[1] == LWL_SUCCESS) ||
            (status[2] == LWL_SUCCESS) ||
            (status[3] == LWL_SUCCESS) ||
            (status[4] == LWL_SUCCESS))
        {
            LWSWITCH_HANDLED(bit);
        }
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (chip_device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _NPORT, _ERR_UC_MASK_NPORT,
                report.raw_enable ^ pending);
    }

    // Clear NPORT interrupts
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _NPORT, _ERR_UC_FIRST_NPORT,
            report.raw_first & report.mask);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _NPORT, _ERR_UC_STATUS_NPORT, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_npg_uncorr
(
    lwswitch_device *device,
    LwU32            npg
)
{
    LwU32 pending, mask, bit, unhandled;
    LwU32 nport;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    pending = LWSWITCH_NPG_RD32_SV10(device, npg, _NPG, _NPG_INTERRUPT_STATUS);

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    mask = DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV0_FATAL, _PENDING) |
        DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV1_FATAL, _PENDING) |
        DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV2_FATAL, _PENDING) |
        DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV3_FATAL, _PENDING);
    pending &= mask;
    LWSWITCH_UNHANDLED_INIT(pending);

    for (nport = 0; nport < chip_device->subengNPG[npg].numNPORT; nport++)
    {
        if (chip_device->subengNPG[npg].subengNPORT[nport].valid)
        {
            // Shift fatal interrupt bit into the correct DEV0-DEV3 position
            if (LWSWITCH_PENDING(DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV0_FATAL, _PENDING) <<
                chip_device->subengNPG[npg].subengNPORT[nport].intr_bit))
            {
                if (_lwswitch_service_nport_uncorr(device, npg, nport) == LWL_SUCCESS)
                {
                    LWSWITCH_HANDLED(bit);
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
_lwswitch_service_nport_corr
(
    lwswitch_device *device,
    LwU32            npg,
    LwU32            nport
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 link = NPORT_TO_LINK(device, npg, nport);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, unhandled;
    LwlStatus status[5];

    report.raw_pending = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _NPORT, _ERR_C_STATUS_NPORT);
    report.raw_enable = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _NPORT, _ERR_C_MASK_NPORT);
    report.mask = ~report.raw_enable;
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    report.raw_first = LWSWITCH_NPORT_RD32_SV10(device, npg, nport, _NPORT, _ERR_C_FIRST_NPORT);

    if (LWSWITCH_PENDING(DRF_NUM(_NPORT, _ERR_C_STATUS_NPORT, _CINTERNAL, 1)))
    {
        LWSWITCH_REPORT_TREE(_HW_NPORT_CINTERNAL);

        status[0] = _lwswitch_service_tag_state_corr(device, npg, nport);
        status[1] = _lwswitch_service_flush_state_corr(device, npg, nport);
        status[2] = _lwswitch_service_ingress_corr(device, npg, nport);
        status[3] = _lwswitch_service_egress_corr(device, npg, nport);
        status[4] = _lwswitch_service_route_corr(device, npg, nport);

        if ((status[0] == LWL_SUCCESS) ||
            (status[1] == LWL_SUCCESS) ||
            (status[2] == LWL_SUCCESS) ||
            (status[3] == LWL_SUCCESS) ||
            (status[4] == LWL_SUCCESS))
        {
            LWSWITCH_HANDLED(bit);
        }
    }

    //
    // Unimplemented in RTL:
    // _PHYRECEIVER, _BADAN0PKT, _REPLAYTIMEOUT, _ADVISORYERROR, _HEADEROVERFLOW
    //

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (chip_device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _NPORT, _ERR_C_MASK_NPORT,
                report.raw_enable ^ pending);
    }

    // Clear interrupts
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _NPORT, _ERR_C_FIRST_NPORT,
            report.raw_first & report.mask);
    LWSWITCH_NPORT_WR32_SV10(device, npg, nport, _NPORT, _ERR_C_STATUS_NPORT, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_npg_corr
(
    lwswitch_device *device,
    LwU32 npg
)
{
    LwU32 pending, mask, bit, unhandled;
    LwU32 nport;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    pending = LWSWITCH_NPG_RD32_SV10(device, npg, _NPG, _NPG_INTERRUPT_STATUS);
    mask = DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV0_CORRECTABLE, _PENDING) |
        DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV1_CORRECTABLE, _PENDING) |
        DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV2_CORRECTABLE, _PENDING) |
        DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV3_CORRECTABLE, _PENDING);
    pending &= mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    for (nport = 0; nport < chip_device->subengNPG[npg].numNPORT; nport++)
    {
        if (chip_device->subengNPG[npg].subengNPORT[nport].valid)
        {
            // Shift correctable interrupt bit into the correct DEV0-DEV3 position
            if (LWSWITCH_PENDING(DRF_DEF(_NPG, _NPG_INTERRUPT_STATUS, _DEV0_CORRECTABLE, _PENDING) <<
                chip_device->subengNPG[npg].subengNPORT[nport].intr_bit))
            {
                if (_lwswitch_service_nport_corr(device, npg, nport) == LWL_SUCCESS)
                {
                    LWSWITCH_HANDLED(bit);
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
_lwswitch_service_clkcross_uncorr
(
    lwswitch_device *device,
    LwU32            sioctrl,
    LwU32            local_link
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 link = SIOCTRL_TO_LINK(device, sioctrl, local_link);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, contain, unhandled;

    report.raw_pending = LWSWITCH_SIOCTRL_OFF_RD32_SV10(device, sioctrl,
            LW_LWLCTRL_CLKCROSS_ERR_STATUS_0(local_link));
    report.raw_enable = LWSWITCH_SIOCTRL_OFF_RD32_SV10(device, sioctrl,
            LW_LWLCTRL_CLKCROSS_ERR_REPORT_EN_0(local_link));
    report.mask = report.raw_enable &
        (DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_STATUS_0, _INGRESSECCHDRDOUBLEBITERR, 1) |
         DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_STATUS_0, _INGRESSECCDATADOUBLEBITERR, 1) |
         DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_STATUS_0, _INGRESSBUFFERERR, 1) |
         DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_STATUS_0, _EGRESSECCHDRDOUBLEBITERR, 1) |
         DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_STATUS_0, _EGRESSECCDATADOUBLEBITERR, 1) |
         DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_STATUS_0, _EGRESSBUFFERERR, 1));
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    report.raw_first = LWSWITCH_SIOCTRL_OFF_RD32_SV10(device, sioctrl,
            LW_LWLCTRL_CLKCROSS_ERR_FIRST_0(local_link));
    contain = LWSWITCH_SIOCTRL_OFF_RD32_SV10(device, sioctrl,
            LW_LWLCTRL_CLKCROSS_ERR_CONTAIN_EN_0(local_link));

    if (LWSWITCH_PENDING(DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_STATUS_0, _INGRESSECCHDRDOUBLEBITERR, 1)))
    {
        // Record current ecc count to be cleared by limit error
        report.data[0] = LWSWITCH_SIOCTRL_OFF_RD32_SV10(device, sioctrl,
                LW_LWLCTRL_CLKCROSS_INGRESS_ECC_ERROR_COUNT(local_link));
        LWSWITCH_REPORT_CONTAIN(_HW_LWLCTRL_INGRESSECCHDRDOUBLEBITERR,
                "clkcross ingress header multi-bit", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_STATUS_0, _INGRESSECCDATADOUBLEBITERR, 1)))
    {
        // Record current ecc count to be cleared by limit error
        report.data[0] = LWSWITCH_SIOCTRL_OFF_RD32_SV10(device, sioctrl,
                LW_LWLCTRL_CLKCROSS_INGRESS_ECC_ERROR_COUNT(local_link));
        LWSWITCH_REPORT_CONTAIN(_HW_LWLCTRL_INGRESSECCDATADOUBLEBITERR,
                "clkcross ingress data multi-bit", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_STATUS_0, _INGRESSBUFFERERR, 1)))
    {
        report.data[0] = 0;
        LWSWITCH_REPORT_CONTAIN(_HW_LWLCTRL_INGRESSBUFFERERR,
                "clkcross ingress buffer overflow", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_STATUS_0, _EGRESSECCHDRDOUBLEBITERR, 1)))
    {
        // Record current ecc count to be cleared by limit error
        report.data[0] = LWSWITCH_SIOCTRL_OFF_RD32_SV10(device, sioctrl,
                LW_LWLCTRL_CLKCROSS_EGRESS_ECC_ERROR_COUNT(local_link));
        LWSWITCH_REPORT_CONTAIN(_HW_LWLCTRL_EGRESSECCHDRDOUBLEBITERR,
                "clkcross egress header multi-bit", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_STATUS_0, _EGRESSECCDATADOUBLEBITERR, 1)))
    {
        // Record current ecc count to be cleared by limit error
        report.data[0] = LWSWITCH_SIOCTRL_OFF_RD32_SV10(device, sioctrl,
                LW_LWLCTRL_CLKCROSS_EGRESS_ECC_ERROR_COUNT(local_link));
        LWSWITCH_REPORT_CONTAIN(_HW_LWLCTRL_EGRESSECCDATADOUBLEBITERR,
                "clkcross egress data multi-bit", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_STATUS_0, _EGRESSBUFFERERR, 1)))
    {
        report.data[0] = 0;
        LWSWITCH_REPORT_CONTAIN(_HW_LWLCTRL_EGRESSBUFFERERR,
                "clkcross egress buffer overflow", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (chip_device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_SIOCTRL_OFF_WR32_SV10(device, sioctrl,
                LW_LWLCTRL_CLKCROSS_ERR_REPORT_EN_0(local_link),
                report.raw_enable ^ pending);
    }

    //
    // Clockcross samples the register insted of write pulse and needs
    // to be explicitly cleared.
    //
    LWSWITCH_SIOCTRL_OFF_WR32_SV10(device, sioctrl,
            LW_LWLCTRL_CLKCROSS_ERR_INJECT_0(local_link), 0);
    LWSWITCH_SIOCTRL_OFF_WR32_SV10(device, sioctrl,
            LW_LWLCTRL_CLKCROSS_ERR_FIRST_0(local_link),
            report.raw_first & report.mask);
    LWSWITCH_SIOCTRL_OFF_WR32_SV10(device, sioctrl,
            LW_LWLCTRL_CLKCROSS_ERR_STATUS_0(local_link), pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_clkcross_corr
(
    lwswitch_device *device,
    LwU32            sioctrl,
    LwU32            local_link
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 link = SIOCTRL_TO_LINK(device, sioctrl, local_link);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, unhandled;

    report.raw_pending = LWSWITCH_SIOCTRL_OFF_RD32_SV10(device, sioctrl,
            LW_LWLCTRL_CLKCROSS_ERR_STATUS_0(local_link));
    report.raw_enable = LWSWITCH_SIOCTRL_OFF_RD32_SV10(device, sioctrl,
            LW_LWLCTRL_CLKCROSS_ERR_REPORT_EN_0(local_link));
    report.mask = report.raw_enable &
        (DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_LOG_EN_0, _INGRESSECCSOFTLIMITERR, 1) |
         DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_LOG_EN_0, _EGRESSECCSOFTLIMITERR, 1));
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    report.raw_first = LWSWITCH_SIOCTRL_OFF_RD32_SV10(device, sioctrl,
            LW_LWLCTRL_CLKCROSS_ERR_FIRST_0(local_link));

    if (LWSWITCH_PENDING(DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_STATUS_0, _INGRESSECCSOFTLIMITERR, 1)))
    {
        // Save running ECC count - is both SBE and DBE
        report.data[0] = LWSWITCH_SIOCTRL_OFF_RD32_SV10(device, sioctrl,
                LW_LWLCTRL_CLKCROSS_INGRESS_ECC_ERROR_COUNT(local_link));
        LWSWITCH_SIOCTRL_OFF_WR32_SV10(device, sioctrl,
                LW_LWLCTRL_CLKCROSS_INGRESS_ECC_ERROR_COUNT(local_link), 0);
        LWSWITCH_REPORT_CORRECTABLE(_HW_LWLCTRL_INGRESSECCSOFTLIMITERR,
                "clkcross ingress ECC threshold");
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLCTRL, _CLKCROSS_0_ERR_STATUS_0, _EGRESSECCSOFTLIMITERR, 1)))
    {
        // Save running ECC count - is both SBE and DBE
        report.data[0] = LWSWITCH_SIOCTRL_OFF_RD32_SV10(device, sioctrl,
                LW_LWLCTRL_CLKCROSS_EGRESS_ECC_ERROR_COUNT(local_link));
        LWSWITCH_SIOCTRL_OFF_WR32_SV10(device, sioctrl,
                LW_LWLCTRL_CLKCROSS_EGRESS_ECC_ERROR_COUNT(local_link), 0);
        LWSWITCH_REPORT_CORRECTABLE(_HW_LWLCTRL_EGRESSECCSOFTLIMITERR,
                "clkcross egress ECC threshold");
        LWSWITCH_HANDLED(bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (chip_device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_SIOCTRL_OFF_WR32_SV10(device, sioctrl,
            LW_LWLCTRL_CLKCROSS_ERR_REPORT_EN_0(local_link),
            report.raw_enable ^ pending);
    }

    //
    // Clockcross samples the register insted of write pulse and needs
    // to be explicitly cleared.
    //
    LWSWITCH_SIOCTRL_OFF_WR32_SV10(device, sioctrl,
            LW_LWLCTRL_CLKCROSS_ERR_INJECT_0(local_link), 0);
    LWSWITCH_SIOCTRL_OFF_WR32_SV10(device, sioctrl,
            LW_LWLCTRL_CLKCROSS_ERR_FIRST_0(local_link),
            report.raw_first & report.mask);
    LWSWITCH_SIOCTRL_OFF_WR32_SV10(device, sioctrl,
            LW_LWLCTRL_CLKCROSS_ERR_STATUS_0(local_link), pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_lwltlc_tx_uncorr
(
    lwswitch_device *device,
    LwU32            sioctrl,
    LwU32            local_link,
    LwU32            mask
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 link = SIOCTRL_TO_LINK(device, sioctrl, local_link);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, contain, unhandled;
    LWSWITCH_RAW_ERROR_LOG_TYPE data = {{0}};

    report.raw_pending = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_TX, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_TX, _ERR_REPORT_EN_0);
    report.mask = report.raw_enable & mask;
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    report.raw_first = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_TX, _ERR_FIRST_0);
    contain = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_TX, _ERR_CONTAIN_EN_0);

    // lwlipt _FLOWCONTROL errors
    {
        if (LWSWITCH_PENDING(DRF_DEF(_LWLTLC_TX, _ERR_STATUS_0, _TXHDRCREDITOVFERR, _PENDING)))
        {
            // 8 bits cover VC0 to VC7 - record in data payload
            report.data[0] = bit;
            LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_TXHDRCREDITOVFERR,
                    "LWLTLC TX header credit overflow", LW_FALSE);
            report.data[0] = 0;
            LWSWITCH_HANDLED(bit);
        }
        if (LWSWITCH_PENDING(DRF_DEF(_LWLTLC_TX, _ERR_STATUS_0, _TXDATACREDITOVFERR, _PENDING)))
        {
            // 8 bits cover VC0 to VC7 - record in data payload
            report.data[0] = bit >> DRF_SHIFT(LW_LWLTLC_TX_ERR_STATUS_0_TXDATACREDITOVFERR);
            LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_TXDATACREDITOVFERR,
                    "LWLTLC TX data credit overflow", LW_FALSE);
            report.data[0] = 0;
            LWSWITCH_HANDLED(bit);
        }
        if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_TX, _ERR_STATUS_0, _TXDLCREDITOVFERR, 1)))
        {
            LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_TXDLCREDITOVFERR,
                    "LWLTLC TX DL credit overflow", LW_FALSE);
            LWSWITCH_HANDLED(bit);
        }
    }

    // lwlipt UCINTERNAL errors
    {
        if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_TX, _ERR_STATUS_0, _TXDLCREDITPARITYERR, 1)))
        {
            LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_TXDLCREDITPARITYERR,
                    "LWLTLC TX credit parity", LW_FALSE);
            LWSWITCH_HANDLED(bit);
        }
        if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_TX, _ERR_STATUS_0, _TXRAMHDRPARITYERR, 1)))
        {
            LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_TXRAMHDRPARITYERR,
                    "LWLTLC TX RAM header parity", LW_FALSE);
            LWSWITCH_HANDLED(bit);
        }
        if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_TX, _ERR_STATUS_0, _TXRAMDATAPARITYERR, 1)))
        {
            LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_TXRAMDATAPARITYERR,
                    "LWLTLC TX RAM data parity", LW_FALSE);
            LWSWITCH_HANDLED(bit);
        }
        if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_TX, _ERR_STATUS_0, _TXUNSUPVCOVFERR, 1)))
        {
            LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_TXUNSUPVCOVFERR,
                    "LWLTLC TX unsupported VC overflow", LW_FALSE);
            LWSWITCH_HANDLED(bit);
        }
        if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_TX, _ERR_STATUS_0, _TXSTOMPDET, 1)))
        {
            LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_TXSTOMPDET,
                    "LWLTLC TX stomp detect", LW_FALSE);
            LWSWITCH_HANDLED(bit);
        }
        if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_TX, _ERR_STATUS_0, _TXPOISONDET, 1)))
        {
            LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_TXPOISONDET,
                    "LWLTLC TX poison detect", LW_FALSE);
            LWSWITCH_HANDLED(bit);
        }
    }

    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_TX, _ERR_STATUS_0, _UNSUPPORTEDREQUESTERR, 1) |
                DRF_NUM(_LWLTLC_TX, _ERR_STATUS_0, _TARGETERR, 1)))
    {
        data.data[0] = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_TX, _ERR_HEADER_LOG_0);
        data.data[1] = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_TX, _ERR_HEADER_LOG_1);
        data.data[2] = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_TX, _ERR_HEADER_LOG_2);
        data.data[3] = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_TX, _ERR_HEADER_LOG_3);
        data.data[4] = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_TX, _ERR_AE_LOG_0);
        data.data[5] = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_TX, _ERR_AE_LOG_1);
        data.data[6] = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_TX, _ERR_AE_LOG_2);
        data.data[7] = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_TX, _ERR_AE_LOG_3);

        // lwlipt _UNSUPPORTEDREQUEST error
        if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_TX, _ERR_STATUS_0, _UNSUPPORTEDREQUESTERR, 1)))
        {
            LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_UNSUPPORTEDREQUESTERR,
                    "LWLTLC TX unsupported request", LW_FALSE);
            LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_LWLTLC_TX_PACKET_HEADER, data);
            LWSWITCH_HANDLED(bit);
        }

        // lwlipt _TARGETERROR error
        if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_TX, _ERR_STATUS_0, _TARGETERR, 1)))
        {
            LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_TARGETERR, "LWLTLC TX target", LW_FALSE);
            LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_LWLTLC_TX_PACKET_HEADER, data);
            LWSWITCH_HANDLED(bit);
        }
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (chip_device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_LWLTLC_WR32_SV10(device, sioctrl, local_link, _LWLTLC_TX, _ERR_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    LWSWITCH_LWLTLC_WR32_SV10(device, sioctrl, local_link, _LWLTLC_TX, _ERR_FIRST_0,
            report.raw_first & report.mask);
    LWSWITCH_LWLTLC_WR32_SV10(device, sioctrl, local_link, _LWLTLC_TX, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_lwltlc_rx_0_uncorr
(
    lwswitch_device *device,
    LwU32            sioctrl,
    LwU32            local_link,
    LwU32            mask
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 link = SIOCTRL_TO_LINK(device, sioctrl, local_link);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, contain, unhandled;
    LWSWITCH_RAW_ERROR_LOG_TYPE data = {{0}};

    report.raw_pending = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_STATUS_0);
    report.raw_enable = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_REPORT_EN_0);
    report.mask = report.raw_enable & mask;
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    report.raw_first = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_FIRST_0);
    contain = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_CONTAIN_EN_0);

    /*
     * The manual says these are latched for rx protocol/overflow errors, but the assignment
     * of rx_log_valid_headervalid0 in the RTL is very different.
     */
    data.data[0] = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_HEADER_LOG_0);
    data.data[1] = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_HEADER_LOG_1);
    data.data[2] = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_HEADER_LOG_2);
    data.data[3] = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_HEADER_LOG_3);
    data.data[4] = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_AE_LOG_0);
    data.data[5] = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_AE_LOG_1);
    data.data[6] = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_AE_LOG_2);
    data.data[7] = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_AE_LOG_3);

    // lwlipt UCINTERNAL error

    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RXDLHDRPARITYERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_RXDLHDRPARITYERR,
                "LWLTLC RX DL header parity", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RXDLDATAPARITYERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_RXDLDATAPARITYERR,
                "LWLTLC RX DL data parity", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RXDLCTRLPARITYERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_RXDLCTRLPARITYERR,
                    "LWLTLC RX DL control parity", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RXRAMDATAPARITYERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_RXRAMDATAPARITYERR,
                "LWLTLC RX RAM data parity", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RXRAMHDRPARITYERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_RXRAMHDRPARITYERR,
                "LWLTLC RX RAM header parity", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RXILWALIDAEERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_RXILWALIDAEERR,
                "LWLTLC RX invalid AE flit", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_LWLTLC_RX_ERR_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }

    // lwlipt _MALFORMEDPACKET error

    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RXILWALIDBEERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_RXILWALIDBEERR,
                "LWLTLC RX invalid BE flit", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_LWLTLC_RX_ERR_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RXILWALIDADDRALIGNERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_RXILWALIDADDRALIGNERR,
                "LWLTLC RX invalid address alignment", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_LWLTLC_RX_ERR_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RXPKTLENERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_RXPKTLENERR,
                "LWLTLC RX invalid packet length", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_LWLTLC_RX_ERR_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RSVCMDENCERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_RSVCMDENCERR,
                "LWLTLC RX command encode", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_LWLTLC_RX_ERR_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RSVDATLENENCERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_RSVDATLENENCERR,
                "LWLTLC RX reserved data length", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_LWLTLC_RX_ERR_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RSVADDRTYPEERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_RSVADDRTYPEERR,
                "LWLTLC RX reserved addr type", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_LWLTLC_RX_ERR_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RSVRSPSTATUSERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_RSVRSPSTATUSERR,
                "LWLTLC RX reserved response encoding", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_LWLTLC_RX_ERR_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RSVPKTSTATUSERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_RSVPKTSTATUSERR,
                "LWLTLC RX reserved packet status encoding", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_LWLTLC_RX_ERR_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RSVCACHEATTRPROBEREQERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_RSVCACHEATTRPROBEREQERR,
                "LWLTLC RX reserved cacheAddr encoding", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_LWLTLC_RX_ERR_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RSVCACHEATTRPROBERSPERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_RSVCACHEATTRPROBERSPERR,
                "LWLTLC RX reserved probe response", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_LWLTLC_RX_ERR_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _DATLENGTATOMICREQMAXERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_DATLENGTATOMICREQMAXERR,
                "LWLTLC RX atomic too big", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_LWLTLC_RX_ERR_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _DATLENGTRMWREQMAXERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_DATLENGTRMWREQMAXERR,
                "LWLTLC RX DataLen too big", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_LWLTLC_RX_ERR_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _DATLENLTATRRSPMINERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_DATLENLTATRRSPMINERR,
                "LWLTLC RX DataLen < ATR response", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_LWLTLC_RX_ERR_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _ILWALIDCACHEATTRPOERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_ILWALIDCACHEATTRPOERR, "LWLTLC RX", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_LWLTLC_RX_ERR_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _ILWALIDCRERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_ILWALIDCRERR,
                "LWLTLC RX CacheAttr/P0 mismatch", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RXRESPSTATUSTARGETERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_RXRESPSTATUSTARGETERR,
                "LWLTLC RX invalid compressed response", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_LWLTLC_RX_ERR_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RXRESPSTATUSUNSUPPORTEDREQUESTERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_RXRESPSTATUSUNSUPPORTEDREQUESTERR,
                "LWLTLC RX TE error in response status", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_LWLTLC_RX_ERR_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (chip_device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_LWLTLC_WR32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_REPORT_EN_0,
                report.raw_enable ^ pending);
    }

    LWSWITCH_LWLTLC_WR32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_FIRST_0,
            report.raw_first & report.mask);
    LWSWITCH_LWLTLC_WR32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_STATUS_0, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_lwltlc_rx_1_uncorr
(
    lwswitch_device *device,
    LwU32            sioctrl,
    LwU32            local_link,
    LwU32            mask
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 link = SIOCTRL_TO_LINK(device, sioctrl, local_link);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, contain, unhandled;

    report.raw_pending = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_STATUS_1);
    report.raw_enable = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_REPORT_EN_1);
    report.mask = report.raw_enable & mask;
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    report.raw_first = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_FIRST_1);
    contain = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_CONTAIN_EN_1);

    // lwlipt _RECEIVEROVERFLOW errors
    {
        if (LWSWITCH_PENDING(DRF_DEF(_LWLTLC_RX, _ERR_STATUS_1, _RXHDROVFERR, _PENDING)))
        {
            // 8 bits cover VC0 to VC7 - record in data payload
            LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_RXHDROVFERR,
                    "LWLTLC RX header overflow", LW_FALSE);
            LWSWITCH_HANDLED(bit);
        }
        if (LWSWITCH_PENDING(DRF_DEF(_LWLTLC_RX, _ERR_STATUS_1, _RXDATAOVFERR, _PENDING)))
        {
            // 8 bits cover VC0 to VC7 - record in data payload
            LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_RXDATAOVFERR,
                    "LWLTLC RX data overflow", LW_FALSE);
            LWSWITCH_HANDLED(bit);
        }
        if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_1, _RXUNSUPVCOVFERR, 1)))
        {
            LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_RXUNSUPVCOVFERR,
                    "LWLTLC RX unupported VC overflow", LW_FALSE);
            LWSWITCH_HANDLED(bit);
        }
    }

    // lwlipt _STOMPEDPACKETRECEIVED error
    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_1, _STOMPDETERR, 1)))
    {
        LWSWITCH_RAW_ERROR_LOG_TYPE data = {{0}};

        data.data[0] = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_HEADER_LOG_0);
        data.data[1] = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_HEADER_LOG_1);
        data.data[2] = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_HEADER_LOG_2);
        data.data[3] = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_HEADER_LOG_3);
        data.data[4] = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_AE_LOG_0);
        data.data[5] = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_AE_LOG_1);
        data.data[6] = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_AE_LOG_2);
        data.data[7] = LWSWITCH_LWLTLC_RD32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_AE_LOG_3);

        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_STOMPDETERR,
                "LWLTLC RX stomp detected", LW_FALSE);
        LWSWITCH_REPORT_CONTAIN_DATA_FIRST(_HW_LWLTLC_RX_ERR_HEADER, data);
        LWSWITCH_HANDLED(bit);
    }

    // lwlipt _DATAPOISONED error
    if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_1, _RXPOISONERR, 1)))
    {
        LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_RXPOISONERR,
                "LWLTLC RX poison", LW_FALSE);
        LWSWITCH_HANDLED(bit);
    }

    // LW_LWLTLC_RX_ERR_STATUS_1_CORRECTABLEINTERNALERR is forced to 0 in the RTL

    // lwlipt _FLOWCONTROL error
    {
        if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_1, _RXUNSUPLWLINKCREDITRELERR, 1)))
        {
            LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_RXUNSUPLWLINKCREDITRELERR,
                    "LWLTLC RX unsupported lwlink credit release", LW_FALSE);
            LWSWITCH_HANDLED(bit);
        }
        if (LWSWITCH_PENDING(DRF_NUM(_LWLTLC_RX, _ERR_STATUS_1, _RXUNSUPNCISOCCREDITRELERR, 1)))
        {
            LWSWITCH_REPORT_CONTAIN(_HW_LWLTLC_RXUNSUPNCISOCCREDITRELERR,
                    "LWLTLC RX unsupported NCISOC credit release", LW_FALSE);
            LWSWITCH_HANDLED(bit);
        }
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (chip_device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_LWLTLC_WR32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_REPORT_EN_1,
                report.raw_enable ^ pending);
    }

    LWSWITCH_LWLTLC_WR32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_FIRST_1,
            report.raw_first & report.mask);
    LWSWITCH_LWLTLC_WR32_SV10(device, sioctrl, local_link, _LWLTLC_RX, _ERR_STATUS_1, pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_lwlipt_err_uncorr
(
    lwswitch_device *device,
    LwU32            sioctrl,
    LwU32            local_link
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 link = SIOCTRL_TO_LINK(device, sioctrl, local_link);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, unhandled;

    report.raw_pending = LWSWITCH_LWLIPT_OFF_RD32_SV10(device, sioctrl,
            LW_LWLIPT_ERR_UC_STATUS(local_link));
    report.raw_enable = LWSWITCH_LWLIPT_OFF_RD32_SV10(device, sioctrl,
            LW_LWLIPT_ERR_UC_MASK(local_link));
    report.mask = ~report.raw_enable;
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    report.raw_first = LWSWITCH_LWLIPT_OFF_RD32_SV10(device, sioctrl,
            LW_LWLIPT_ERR_UC_FIRST(local_link));

    // _DLPROTOCOL is handled via DL "intr" and is disabled.
    if (LWSWITCH_PENDING(DRF_NUM(_LWLIPT, _ERR_UC_STATUS_LINK0, _DATAPOISONED, 1)))
    {
        LWSWITCH_REPORT_TREE(_HW_LWLIPT_DATAPOISONED);
        if (_lwswitch_service_lwltlc_rx_1_uncorr(device, sioctrl, local_link,
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_1, _RXPOISONERR, 1)) == LWL_SUCCESS)
        {
            LWSWITCH_HANDLED(bit);
        }
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLIPT, _ERR_UC_STATUS_LINK0, _FLOWCONTROL, 1)))
    {
        LWSWITCH_REPORT_TREE(_HW_LWLIPT_FLOWCONTROL);
        if (_lwswitch_service_lwltlc_tx_uncorr(device, sioctrl, local_link,
                DRF_DEF(_LWLTLC_TX, _ERR_STATUS_0, _TXHDRCREDITOVFERR, _PENDING) |
                DRF_DEF(_LWLTLC_TX, _ERR_STATUS_0, _TXDATACREDITOVFERR, _PENDING) |
                DRF_NUM(_LWLTLC_TX, _ERR_STATUS_0, _TXDLCREDITOVFERR, 1)) == LWL_SUCCESS)
        {
            LWSWITCH_HANDLED(bit);
        }

        if (_lwswitch_service_lwltlc_rx_1_uncorr(device, sioctrl, local_link,
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_1, _RXUNSUPLWLINKCREDITRELERR, 1) |
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_1, _RXUNSUPNCISOCCREDITRELERR, 1)) == LWL_SUCCESS)
        {
            LWSWITCH_HANDLED(bit);
        }
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLIPT, _ERR_UC_STATUS_LINK0, _TARGETERROR, 1)))
    {
        LWSWITCH_REPORT_TREE(_HW_LWLIPT_TARGETERROR);
        if (_lwswitch_service_lwltlc_tx_uncorr(device, sioctrl, local_link,
                DRF_NUM(_LWLTLC_TX, _ERR_STATUS_0, _TARGETERR, 1)) == LWL_SUCCESS)
        {
            LWSWITCH_HANDLED(bit);
        }
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLIPT, _ERR_UC_STATUS_LINK0, _RECEIVEROVERFLOW, 1)))
    {
        LWSWITCH_REPORT_TREE(_HW_LWLIPT_RECEIVEROVERFLOW);
        if (_lwswitch_service_lwltlc_rx_1_uncorr(device, sioctrl, local_link,
                DRF_DEF(_LWLTLC_RX, _ERR_STATUS_1, _RXHDROVFERR, _PENDING) |
                DRF_DEF(_LWLTLC_RX, _ERR_STATUS_1, _RXDATAOVFERR, _PENDING) |
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_1, _RXUNSUPVCOVFERR, 1)) == LWL_SUCCESS)
        {
            LWSWITCH_HANDLED(bit);
        }
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLIPT, _ERR_UC_STATUS_LINK0, _MALFORMEDPACKET, 1)))
    {
        LWSWITCH_REPORT_TREE(_HW_LWLIPT_MALFORMEDPACKET);
        if (_lwswitch_service_lwltlc_rx_0_uncorr(device, sioctrl, local_link,
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RXILWALIDBEERR, 1) |
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RXILWALIDADDRALIGNERR, 1) |
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RXPKTLENERR, 1) |
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RSVCMDENCERR, 1) |
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RSVDATLENENCERR, 1) |
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RSVADDRTYPEERR, 1) |
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RSVRSPSTATUSERR, 1) |
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RSVPKTSTATUSERR, 1) |
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RSVCACHEATTRPROBEREQERR, 1) |
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RSVCACHEATTRPROBERSPERR, 1) |
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _DATLENGTATOMICREQMAXERR, 1) |
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _DATLENGTRMWREQMAXERR, 1) |
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _DATLENLTATRRSPMINERR, 1) |
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _ILWALIDCACHEATTRPOERR, 1) |
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _ILWALIDCRERR, 1) |
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RXRESPSTATUSTARGETERR, 1) |
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RXILWALIDAEERR, 1)) == LWL_SUCCESS)
        {
            LWSWITCH_HANDLED(bit);
        }
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLIPT, _ERR_UC_STATUS_LINK0, _STOMPEDPACKETRECEIVED, 1)))
    {
        LWSWITCH_REPORT_TREE(_HW_LWLIPT_STOMPEDPACKETRECEIVED);
        if (_lwswitch_service_lwltlc_rx_1_uncorr(device, sioctrl, local_link,
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_1, _STOMPDETERR, 1)) == LWL_SUCCESS)
        {
            LWSWITCH_HANDLED(bit);
        }
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLIPT, _ERR_UC_STATUS_LINK0, _UNSUPPORTEDREQUEST, 1)))
    {
        LWSWITCH_REPORT_TREE(_HW_LWLIPT_UNSUPPORTEDREQUEST);
        if (_lwswitch_service_lwltlc_tx_uncorr(device, sioctrl, local_link,
                DRF_NUM(_LWLTLC_TX, _ERR_STATUS_0, _UNSUPPORTEDREQUESTERR, 1)) == LWL_SUCCESS)
        {
            LWSWITCH_HANDLED(bit);
        }

        if (_lwswitch_service_lwltlc_rx_0_uncorr(device, sioctrl, local_link,
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RXRESPSTATUSUNSUPPORTEDREQUESTERR, 1)) == LWL_SUCCESS)
        {
            LWSWITCH_HANDLED(bit);
        }
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLIPT, _ERR_UC_STATUS_LINK0, _UCINTERNAL, 1)))
    {
        LWSWITCH_REPORT_TREE(_HW_LWLIPT_UCINTERNAL);
        // DL may trigger this, but should already be serviced
        if (_lwswitch_service_clkcross_uncorr(device, sioctrl, local_link) == LWL_SUCCESS)
        {
            LWSWITCH_HANDLED(bit);
        }

        if (_lwswitch_service_lwltlc_tx_uncorr(device, sioctrl, local_link,
                DRF_NUM(_LWLTLC_TX, _ERR_STATUS_0, _TARGETERR, 1) |
                DRF_NUM(_LWLTLC_TX, _ERR_STATUS_0, _TXDLCREDITPARITYERR, 1) |
                DRF_NUM(_LWLTLC_TX, _ERR_STATUS_0, _TXRAMHDRPARITYERR, 1) |
                DRF_NUM(_LWLTLC_TX, _ERR_STATUS_0, _TXRAMDATAPARITYERR, 1) |
                DRF_NUM(_LWLTLC_TX, _ERR_STATUS_0, _TXUNSUPVCOVFERR, 1) |
                DRF_NUM(_LWLTLC_TX, _ERR_STATUS_0, _TXSTOMPDET, 1) |
                DRF_NUM(_LWLTLC_TX, _ERR_STATUS_0, _TXPOISONDET, 1)) == LWL_SUCCESS)
        {
            LWSWITCH_HANDLED(bit);
        }

        if (_lwswitch_service_lwltlc_rx_0_uncorr(device, sioctrl, local_link,
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RXDLHDRPARITYERR, 1) |
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RXDLDATAPARITYERR, 1) |
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RXDLCTRLPARITYERR, 1) |
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RXRAMDATAPARITYERR, 1) |
                DRF_NUM(_LWLTLC_RX, _ERR_STATUS_0, _RXRAMHDRPARITYERR, 1)) == LWL_SUCCESS)
        {
            LWSWITCH_HANDLED(bit);
        }
    }
    // Unimplemented in RTL:  _RESPONSETIMEOUT, _UNEXPECTEDRESPONSE

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (chip_device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_LWLIPT_OFF_WR32_SV10(device, sioctrl,
                LW_LWLIPT_ERR_UC_MASK(local_link),
                report.raw_enable ^ pending);
    }

    LWSWITCH_LWLIPT_OFF_WR32_SV10(device, sioctrl,
            LW_LWLIPT_ERR_UC_FIRST(local_link),
            report.raw_first & report.mask);
    LWSWITCH_LWLIPT_OFF_WR32_SV10(device, sioctrl,
            LW_LWLIPT_ERR_UC_STATUS(local_link),
            pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_minion_interrupts
(
    lwswitch_device *device,
    LwU32            sioctrl
)
{
    LwU32       pending, mask, bit, unhandled, linkNumber;
    LwU16       interruptingLinks;

    pending  = LWSWITCH_MINION_RD32_SV10(device, sioctrl, _MINION, _MINION_INTR);
    mask     = LWSWITCH_MINION_RD32_SV10(device, sioctrl, _MINION, _MINION_INTR_STALL_EN);
    pending &= mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    if (LWSWITCH_PENDING(DRF_NUM(_MINION, _MINION_INTR, _FATAL, 0x1) |
                         DRF_NUM(_MINION, _MINION_INTR, _FALCON_STALL, 0x1)))
    {
        if (lwswitch_minion_service_falcon_interrupts_sv10(device, sioctrl) == LWL_SUCCESS)
        {
            LWSWITCH_HANDLED(bit);
        }
    }

    interruptingLinks = DRF_VAL(_MINION, _MINION_INTR, _LINK, pending);

    if (interruptingLinks)
    {
        if (LWSWITCH_PENDING(DRF_NUM(_MINION, _MINION_INTR, _LINK, LWBIT(0))))
        {
            linkNumber = SIOCTRL_TO_LINK(device, sioctrl, 0);
            if (_lwswitch_minion_service_link_interrupts(device, linkNumber) == LWL_SUCCESS)
            {
                LWSWITCH_HANDLED(bit);
            }
        }

        if (LWSWITCH_PENDING(DRF_NUM(_MINION, _MINION_INTR, _LINK, LWBIT(1))))
        {
            linkNumber = SIOCTRL_TO_LINK(device, sioctrl, 1);
            if (_lwswitch_minion_service_link_interrupts(device, linkNumber) == LWL_SUCCESS)
            {
                LWSWITCH_HANDLED(bit);
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
_lwswitch_service_lwlipt_common_uncorr
(
    lwswitch_device *device,
    LwU32            sioctrl
)
{
    LwU32 pending, mask, bit, unhandled;

    pending = LWSWITCH_SIOCTRL_RD32_SV10(device, sioctrl, _LWLCTRL, _LWLIPT_INTERRUPT_STATUS);
    mask = DRF_DEF(_LWLCTRL, _LWLIPT_INTERRUPT_STATUS, _DEV0_FATAL, _PENDING) |
        DRF_DEF(_LWLCTRL, _LWLIPT_INTERRUPT_STATUS, _DEV1_FATAL, _PENDING);
    pending &= mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    if (LWSWITCH_PENDING(DRF_DEF(_LWLCTRL, _LWLIPT_INTERRUPT_STATUS, _DEV0_FATAL, _PENDING)))
    {
        if (_lwswitch_service_lwlipt_err_uncorr(device, sioctrl, 0) == LWL_SUCCESS)
        {
            LWSWITCH_HANDLED(bit);
        }
    }
    if (LWSWITCH_PENDING(DRF_DEF(_LWLCTRL, _LWLIPT_INTERRUPT_STATUS, _DEV1_FATAL, _PENDING)))
    {
        if (_lwswitch_service_lwlipt_err_uncorr(device, sioctrl, 1) == LWL_SUCCESS)
        {
            LWSWITCH_HANDLED(bit);
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
_lwswitch_service_sioctrl_uncorr
(
    lwswitch_device *device,
    LwU32            sioctrl
)
{
    LwlStatus status[3];

    status[0] = _lwswitch_service_minion_interrupts(device, sioctrl);
    status[1] = _lwswitch_service_lwlipt_common_uncorr(device, sioctrl);
    status[2] = _lwswitch_service_intr_dlpl_common_uncorr(device, sioctrl);

    if ((status[0] != LWL_SUCCESS) &&
        (status[1] != LWL_SUCCESS) &&
        (status[2] != LWL_SUCCESS))
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_lwlipt_err_corr
(
    lwswitch_device *device,
    LwU32            sioctrl,
    LwU32            local_link
)
{
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);
    LwU32 link = SIOCTRL_TO_LINK(device, sioctrl, local_link);
    LWSWITCH_INTERRUPT_LOG_TYPE report = { 0 };
    LwU32 pending, bit, unhandled;

    report.raw_pending = LWSWITCH_LWLIPT_OFF_RD32_SV10(device, sioctrl,
            LW_LWLIPT_ERR_C_STATUS(local_link));
    report.raw_enable = LWSWITCH_LWLIPT_OFF_RD32_SV10(device, sioctrl,
            LW_LWLIPT_ERR_C_MASK(local_link));
    report.mask = ~report.raw_enable;
    pending = report.raw_pending & report.mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    report.raw_first = LWSWITCH_LWLIPT_OFF_RD32_SV10(device, sioctrl,
            LW_LWLIPT_ERR_C_FIRST(local_link));

    //
    // Bit offset/defines are constant across LINK0/LINK1/COMMON
    //
    // _PHYRECEIVER is not implemented in RTL
    // _BADAN0PKT is handled via DL "intr" and is disabled.
    // _REPLAYTIMEOUT is handled via DL "intr" and is disabled.
    //
    if (LWSWITCH_PENDING(DRF_NUM(_LWLIPT, _ERR_C_STATUS_LINK0, _ADVISORYERROR, 1)))
    {
        LWSWITCH_REPORT_CORRECTABLE(_HW_LWLIPT_ADVISORYERROR, "LWLIPT advisory");
        LWSWITCH_HANDLED(bit);
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLIPT, _ERR_C_STATUS_LINK0, _CINTERNAL, 1)))
    {
        LWSWITCH_REPORT_TREE(_HW_LWLIPT_CINTERNAL);
        if (_lwswitch_service_clkcross_corr(device, sioctrl, local_link) == LWL_SUCCESS)
        {
            LWSWITCH_HANDLED(bit);
        }
    }
    if (LWSWITCH_PENDING(DRF_NUM(_LWLIPT, _ERR_C_STATUS_LINK0, _HEADEROVERFLOW, 1)))
    {
        // TLC RX received 2nd error before service and could not log header flit
        LWSWITCH_REPORT_CORRECTABLE(_HW_LWLIPT_HEADEROVERFLOW,
                "header log overflow");
        LWSWITCH_HANDLED(bit);
    }

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

    // Disable interrupts that have oclwrred after fatal error.
    if (chip_device->link[link].fatal_error_oclwrred)
    {
        LWSWITCH_LWLIPT_OFF_WR32_SV10(device, sioctrl,
                LW_LWLIPT_ERR_C_MASK(local_link),
                report.raw_enable ^ pending);
    }

    LWSWITCH_LWLIPT_OFF_WR32_SV10(device, sioctrl,
            LW_LWLIPT_ERR_C_FIRST(local_link),
            report.raw_first & report.mask);
    LWSWITCH_LWLIPT_OFF_WR32_SV10(device, sioctrl,
            LW_LWLIPT_ERR_C_STATUS(local_link),
            pending);

    if (unhandled != 0)
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    return LWL_SUCCESS;
}

static LwlStatus
_lwswitch_service_sioctrl_corr
(
    lwswitch_device *device,
    LwU32 sioctrl
)
{
    LwU32 pending, mask, bit, unhandled;

    pending = LWSWITCH_SIOCTRL_RD32_SV10(device, sioctrl, _LWLCTRL, _LWLIPT_INTERRUPT_STATUS);
    mask = DRF_DEF(_LWLCTRL, _LWLIPT_INTERRUPT_STATUS, _DEV0_CORRECTABLE, _PENDING) |
        DRF_DEF(_LWLCTRL, _LWLIPT_INTERRUPT_STATUS, _DEV1_CORRECTABLE, _PENDING);
    pending &= mask;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    if (LWSWITCH_PENDING(DRF_DEF(_LWLCTRL, _LWLIPT_INTERRUPT_STATUS, _DEV0_CORRECTABLE, _PENDING)))
    {
        if (_lwswitch_service_lwlipt_err_corr(device, sioctrl, 0) == LWL_SUCCESS)
        {
            LWSWITCH_HANDLED(bit);
        }
    }
    if (LWSWITCH_PENDING(DRF_DEF(_LWLCTRL, _LWLIPT_INTERRUPT_STATUS, _DEV1_CORRECTABLE, _PENDING)))
    {
        if (_lwswitch_service_lwlipt_err_corr(device, sioctrl, 1) == LWL_SUCCESS)
        {
            LWSWITCH_HANDLED(bit);
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
_lwswitch_service_pbus
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

    LWSWITCH_UNHANDLED_INIT(pending);

    if (LWSWITCH_PENDING(DRF_DEF(_PBUS, _INTR_0, _PRI_TIMEOUT, _PENDING) |
                         DRF_DEF(_PBUS, _INTR_0, _PRI_FECSERR, _PENDING) |
                         DRF_DEF(_PBUS, _INTR_0, _PRI_SQUASH, _PENDING)))
    {

        // PRI timeout is likely not recoverable
        LWSWITCH_REG_WR32(device, _PBUS, _INTR_0,
            DRF_DEF(_PBUS, _INTR_0, _PRI_TIMEOUT, _RESET));

        save0 = LWSWITCH_REG_RD32(device, _PTIMER, _PRI_TIMEOUT_SAVE_0);
        save1 = LWSWITCH_REG_RD32(device, _PTIMER, _PRI_TIMEOUT_SAVE_1);
        save3 = LWSWITCH_REG_RD32(device, _PTIMER, _PRI_TIMEOUT_SAVE_3);
        errCode = LWSWITCH_REG_RD32(device, _PTIMER, _PRI_TIMEOUT_FECS_ERRCODE);

        pri_timeout.addr  = DRF_VAL(_PTIMER, _PRI_TIMEOUT_SAVE_0, _ADDR, save0) * 4;
        pri_timeout.data  = DRF_VAL(_PTIMER, _PRI_TIMEOUT_SAVE_1, _DATA, save1);
        pri_timeout.write = DRF_VAL(_PTIMER, _PRI_TIMEOUT_SAVE_0, _WRITE, save0);
        pri_timeout.dest  = DRF_VAL(_PTIMER, _PRI_TIMEOUT_SAVE_0, _TO, save0);
        pri_timeout.subId = DRF_VAL(_PTIMER, _PRI_TIMEOUT_SAVE_3, _SUBID, save3);
        pri_timeout.errCode = DRF_VAL(_PTIMER, _PRI_TIMEOUT_FECS_ERRCODE, _DATA, errCode);

        // Dump register values as well
        pri_timeout.raw_data[0] = save0;
        pri_timeout.raw_data[1] = save1;
        pri_timeout.raw_data[2] = save3;
        pri_timeout.raw_data[3] = errCode;

        LWSWITCH_PRINT_SXID(device, LWSWITCH_ERR_HW_HOST_PRIV_TIMEOUT, 
            "Fatal, PBUS PRI timeout error\n");
        LWSWITCH_LOG_FATAL_DATA(device, _HW, _HW_HOST_PRIV_TIMEOUT, 0, 0, LW_FALSE, &pri_timeout);

        LWSWITCH_PRINT(device, ERROR,
                    "PBUS PRI timeout: %s offset: 0x%x data: 0x%x to: %d, "
                    "subId: 0x%x, FECS errCode: 0x%x\n",
                    pri_timeout.write ? "write" : "read",
                    pri_timeout.addr,
                    pri_timeout.data,
                    pri_timeout.dest,
                    pri_timeout.subId,
                    pri_timeout.errCode);

        if (FLD_TEST_DRF(_PBUS, _INTR_0, _PRI_SQUASH, _PENDING, bit))
        {
            LWSWITCH_PRINT(device, ERROR, "PRI_SQUASH: "
                "PBUS PRI error due to pri access while target block is in reset\n");
        }

        if (FLD_TEST_DRF(_PBUS, _INTR_0, _PRI_FECSERR, _PENDING, bit))
        {
            LWSWITCH_PRINT(device, ERROR, "PRI_FECSERR: "
                "FECS detected the error while processing a PRI request\n");
        }

        if (FLD_TEST_DRF(_PBUS, _INTR_0, _PRI_TIMEOUT, _PENDING, bit))
        {
            LWSWITCH_PRINT(device, ERROR, "PRI_TIMEOUT: "
                "PBUS PRI error due non-existent host register or timeout waiting for FECS\n");
        }

        // allow next error to latch
        LWSWITCH_REG_WR32(device, _PTIMER, _PRI_TIMEOUT_SAVE_0,
            FLD_SET_DRF(_PTIMER, _PRI_TIMEOUT_SAVE_0, _TO, _CLEAR, save0));
        
        LWSWITCH_HANDLED(bit);
    }

    if (LWSWITCH_PENDING(DRF_DEF(_PBUS, _INTR_0, _SW, _PENDING)))
    {
        // Useful for debugging SW interrupts
        LWSWITCH_PRINT(device, INFO, "SW intr\n");
        LWSWITCH_HANDLED(bit);
    }

    LWSWITCH_REG_WR32(device, _PBUS, _INTR_0, pending);  // W1C with _RESET

    LWSWITCH_UNHANDLED_CHECK(device, unhandled);

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
// FSF/RTLsim does not model interrupts correctly.  The interrupt is shared
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
lwswitch_lib_check_interrupts_sv10
(
    lwswitch_device *device
)
{
    LwU32 pending;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    if (IS_RTLSIM(device) || IS_FMODEL(device))
    {
        pending = LWSWITCH_REG_RD32(device, _PSMC, _INTR_LEGACY);
        pending &= chip_device->intr_enable_legacy;
        if (pending)
        {
            return -LWL_MORE_PROCESSING_REQUIRED;
        }

        // Non-correctable errors are all routed through the "fatal" interrupt tree
        pending = LWSWITCH_REG_RD32(device, _PSMC, _INTR_FATAL);
        pending &= chip_device->intr_enable_uncorr;
        if (pending)
        {
            return -LWL_MORE_PROCESSING_REQUIRED;
        }

        // Correctable interrupts
        pending = LWSWITCH_REG_RD32(device, _PSMC, _INTR_CORRECTABLE);
        pending &= chip_device->intr_enable_corr;
        if (pending)
        {
            return -LWL_MORE_PROCESSING_REQUIRED;
        }

        return LWL_SUCCESS;
    }
    else
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }
}

static LwlStatus
_lwswitch_service_legacy_interrupts
(
    lwswitch_device *device
)
{
    LwU32 pending;
    LwU32 unhandled;
    LwU32 bit;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    pending = LWSWITCH_REG_RD32(device, _PSMC, _INTR_LEGACY);
    pending &= chip_device->intr_enable_legacy;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    if (LWSWITCH_PENDING(DRF_NUM(_PSMC, _INTR_LEGACY, _PRIV_RING, 1)))
    {
        if (_lwswitch_service_priv_ring(device) == LWL_SUCCESS)
        {
            LWSWITCH_HANDLED(bit);
        }
    }

    if (LWSWITCH_PENDING(DRF_NUM(_PSMC, _INTR_LEGACY, _PBUS, 1)))
    {
        if (_lwswitch_service_pbus(device) == LWL_SUCCESS)
        {
            LWSWITCH_HANDLED(bit);
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
_lwswitch_service_uncorr_interrupts
(
    lwswitch_device *device
)
{
    LwU32 i, bit, pending, unhandled;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    // Non-correctable errors are all routed through the "fatal" interrupt tree
    pending = LWSWITCH_REG_RD32(device, _PSMC, _INTR_FATAL);
    pending &= chip_device->intr_enable_uncorr;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    for (i = 0; i < NUM_SWX_ENGINE_SV10; i++)
    {
        if (!chip_device->engSWX[i].valid)
        {
            continue;
        }

        if (LWSWITCH_PENDING(LWBIT(chip_device->engSWX[i].intr_bit)))
        {
            if (_lwswitch_service_swx_uncorr(device, i) == LWL_SUCCESS)
            {
                LWSWITCH_HANDLED(bit);
            }
        }
    }

    for (i = 0; i < NUM_NPG_ENGINE_SV10; i++)
    {
        if (!chip_device->engNPG[i].valid)
        {
            continue;
        }

        if (LWSWITCH_PENDING(LWBIT(chip_device->engNPG[i].intr_bit)))
        {
            if (_lwswitch_service_npg_uncorr(device, i) == LWL_SUCCESS)
            {
                LWSWITCH_HANDLED(bit);
            }
        }
    }

    for (i = 0; i < NUM_SIOCTRL_ENGINE_SV10; i++)
    {
        if (!chip_device->engSIOCTRL[i].valid)
        {
            continue;
        }

        if (LWSWITCH_PENDING(LWBIT(chip_device->engSIOCTRL[i].intr_bit)))
        {
            if (_lwswitch_service_sioctrl_uncorr(device, i) == LWL_SUCCESS)
            {
                LWSWITCH_HANDLED(bit);
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
_lwswitch_service_corr_interrupts
(
    lwswitch_device *device
)
{
    LwU32 i, bit, pending, unhandled;
    sv10_device *chip_device = LWSWITCH_GET_CHIP_DEVICE_SV10(device);

    // Correctable errors are all routed through the "correctable" interrupt tree
    pending = LWSWITCH_REG_RD32(device, _PSMC, _INTR_CORRECTABLE);
    pending &= chip_device->intr_enable_corr;

    if (pending == 0)
    {
        return -LWL_NOT_FOUND;
    }

    LWSWITCH_UNHANDLED_INIT(pending);

    for (i = 0; i < NUM_NPG_ENGINE_SV10; i++)
    {
        if (!chip_device->engNPG[i].valid)
        {
            continue;
        }

        if (LWSWITCH_PENDING(LWBIT(chip_device->engNPG[i].intr_bit)))
        {
            if (_lwswitch_service_npg_corr(device, i) == LWL_SUCCESS)
            {
                LWSWITCH_HANDLED(bit);
            }
        }
    }

    for (i = 0; i < NUM_SIOCTRL_ENGINE_SV10; i++)
    {
        if (!chip_device->engSIOCTRL[i].valid)
        {
            continue;
        }

        if (LWSWITCH_PENDING(LWBIT(chip_device->engSIOCTRL[i].intr_bit)))
        {
            if (_lwswitch_service_sioctrl_corr(device, i) == LWL_SUCCESS)
            {
                LWSWITCH_HANDLED(bit);
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

//
// Service interrupt and re-enable interrupts. Interrupts should be disabled
// when this is called.
//
LwlStatus
lwswitch_lib_service_interrupts_sv10
(
    lwswitch_device *device
)
{
    LwlStatus status[3];

    status[0] = _lwswitch_service_legacy_interrupts(device);
    status[1] = _lwswitch_service_uncorr_interrupts(device);
    status[2] = _lwswitch_service_corr_interrupts(device);

    if ((status[0] != LWL_SUCCESS) &&
        (status[1] != LWL_SUCCESS) &&
        (status[2] != LWL_SUCCESS))
    {
        return -LWL_MORE_PROCESSING_REQUIRED;
    }

    _lwswitch_rearm_msi(device);

    return LWL_SUCCESS;
}

LwlStatus
lwswitch_service_lwldl_fatal_link_sv10
(
    lwswitch_device *device,
    LwU32 lwlipt_instance,
    LwU32 link
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}

LwlStatus
lwswitch_service_minion_link_sv10
(
    lwswitch_device *device,
    LwU32 instance
)
{
    return -LWL_ERR_NOT_SUPPORTED;
}
