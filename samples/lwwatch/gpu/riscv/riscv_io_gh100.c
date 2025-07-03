/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019-2020 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include <stdio.h>
#include <limits.h>
#include <os.h>

#include "riscv_prv.h"
#include "riscv_dbgint.h"
#include "hopper/gh100/dev_gsp.h"
#include "hopper/gh100/dev_sec_pri.h"
#include "hopper/gh100/dev_pwr_pri.h"
#include "hopper/gh100/dev_minion_ip.h"
#include "hopper/gh100/dev_riscv_pri.h"
#include "hopper/gh100/dev_falcon_v4.h"
#include "hopper/gh100/dev_fsp_csb.h"
#include "hopper/gh100/dev_fsp_pri.h"
#include "hopper/gh100/dev_lwdec_csb.h"
#include "hopper/gh100/dev_lwdec_pri.h"
#include "lwswitch/ls10/dev_soe_ip.h"
#include "lwswitch/ls10/dev_riscv_pri.h"
// For the outstanding holes in hwref
#include "riscv_regs.h"

#include "g_riscv_private.h"

/*!
 * @copydoc riscvIsInstanceSupported_TU10X
 */
LwBool riscvIsInstanceSupported_GH100(RiscvInstanceType type)
{
    switch (type)
    {
        case RISCV_INSTANCE_GSP:
        case RISCV_INSTANCE_SEC2:
        case RISCV_INSTANCE_PMU:
        case RISCV_INSTANCE_MINION:
        case RISCV_INSTANCE_LWDEC0:
        case RISCV_INSTANCE_LWDEC1:
        case RISCV_INSTANCE_LWDEC2:
        case RISCV_INSTANCE_LWDEC3:
        case RISCV_INSTANCE_LWDEC4:
        case RISCV_INSTANCE_LWDEC5:
        case RISCV_INSTANCE_LWDEC6:
        case RISCV_INSTANCE_LWDEC7:
        case RISCV_INSTANCE_FSP:
        case RISCV_INSTANCE_SOE:
            return LW_TRUE;
        default:
            break;
    }
    return LW_FALSE;
}

/*!
 * @copydoc  riscvPopulateCoreInfo_TU10X
 * @note     Minion uses RISCV starting GH100.
 */
void
riscvPopulateCoreInfo_GH100
(
    RiscVInstance *info,
    RiscvInstanceType engine
)
{
    LwU32 fhwcfg, hwcfg, lwdecInstOffset;

    if (!pRiscv[indexGpu].riscvIsInstanceSupported(engine))
    {
        info->name = NULL;
        return;
    }

    info->name             = riscvInstanceNames[engine];
    info->instance_no      = engine;
    info->riscv_imem_start = pRiscv[indexGpu].getRiscvDef(engine, "LW_RISCV_AMAP_IMEM_START");
    info->riscv_dmem_start = pRiscv[indexGpu].getRiscvDef(engine, "LW_RISCV_AMAP_DMEM_START");
    info->riscv_emem_start = pRiscv[indexGpu].getRiscvDef(engine, "LW_RISCV_AMAP_EMEM_START");
    info->riscv_fb_start   = pRiscv[indexGpu].getRiscvDef(engine, "LW_RISCV_AMAP_FBGPA_START");
    info->riscv_fb_size    = pRiscv[indexGpu].getRiscvDef(engine, "LW_RISCV_AMAP_FBGPA_SIZE");
    info->riscv_priv_start = pRiscv[indexGpu].getRiscvDef(engine, "LW_RISCV_AMAP_PRIV_START");
    info->riscv_priv_size  = pRiscv[indexGpu].getRiscvDef(engine, "LW_RISCV_AMAP_PRIV_SIZE");

    switch (engine)
    {
        case RISCV_INSTANCE_GSP:
            fhwcfg = GPU_REG_RD32(LW_PGSP_FALCON_HWCFG3);
            hwcfg = GPU_REG_RD32(LW_PGSP_HWCFG);
            info->bar0Base             = LW_FALCON_GSP_BASE;
            info->riscvBase            = LW_FALCON2_GSP_BASE;
            info->defaultPort          = 4444;
            // hwcfg gives mem sizes in units of 256 bytes.
            info->riscv_imem_size      = DRF_VAL(_PGSP, _FALCON_HWCFG3, _IMEM_TOTAL_SIZE, fhwcfg) * FALCON_PAGE_SIZE;
            info->riscv_dmem_size      = DRF_VAL(_PGSP, _FALCON_HWCFG3, _DMEM_TOTAL_SIZE, fhwcfg) * FALCON_PAGE_SIZE;
            info->riscv_emem_size      = DRF_VAL(_PGSP, _HWCFG, _EMEM_SIZE, hwcfg) * FALCON_PAGE_SIZE;
            break;
        case RISCV_INSTANCE_SEC2:
            fhwcfg = GPU_REG_RD32(LW_PSEC_FALCON_HWCFG3);
            hwcfg = GPU_REG_RD32(LW_PSEC_HWCFG);
            // @todo  LW_FALCON_SEC_BASE is sourced from riscv_regs.h at the moment instead of dev_falcon_v1.h
            info->bar0Base             = LW_FALCON_SEC_BASE;
            info->riscvBase            = LW_FALCON2_SEC_BASE;
            info->defaultPort          = 5555;
            info->riscv_imem_size      = DRF_VAL(_PSEC, _FALCON_HWCFG3, _IMEM_TOTAL_SIZE, fhwcfg) * FALCON_PAGE_SIZE;
            info->riscv_dmem_size      = DRF_VAL(_PSEC, _FALCON_HWCFG3, _DMEM_TOTAL_SIZE, fhwcfg) * FALCON_PAGE_SIZE;
            info->riscv_emem_size      = DRF_VAL(_PSEC, _HWCFG, _EMEM_SIZE, hwcfg) * FALCON_PAGE_SIZE;
            break;
        case RISCV_INSTANCE_PMU:
            fhwcfg = GPU_REG_RD32(LW_PPWR_FALCON_HWCFG3);
            info->bar0Base             = LW_FALCON_PWR_BASE;
            info->riscvBase            = LW_FALCON2_PWR_BASE;
            info->defaultPort          = 4545;
            info->riscv_imem_size      = DRF_VAL(_PPWR, _FALCON_HWCFG3, _IMEM_TOTAL_SIZE, fhwcfg) * FALCON_PAGE_SIZE;
            info->riscv_dmem_size      = DRF_VAL(_PPWR, _FALCON_HWCFG3, _DMEM_TOTAL_SIZE, fhwcfg) * FALCON_PAGE_SIZE;
            info->riscv_emem_start     = 0;
            info->riscv_emem_size      = 0;
            break;
        case RISCV_INSTANCE_MINION:
            // @todo LW_FALCON_MINION_BASE is actually wrong; bug 2908447 tracking
            info->bar0Base             = 0xa04000;
            info->riscvBase            = LW_FALCON2_MINION_BASE;
            info->defaultPort          = 5454;
            fhwcfg = bar0ReadLegacy(LW_PFALCON_FALCON_HWCFG3);
            info->riscv_imem_size      = DRF_VAL(_PFALCON, _FALCON_HWCFG3, _IMEM_TOTAL_SIZE, fhwcfg) * FALCON_PAGE_SIZE;
            info->riscv_dmem_size      = DRF_VAL(_PFALCON, _FALCON_HWCFG3, _DMEM_TOTAL_SIZE, fhwcfg) * FALCON_PAGE_SIZE;
            info->riscv_fb_start       = 0;
            info->riscv_fb_size        = 0;
            info->riscv_emem_start     = 0;
            info->riscv_emem_size      = 0;
            break;
        case RISCV_INSTANCE_FSP:
            fhwcfg = GPU_REG_RD32(LW_PFSP_FALCON_HWCFG3);
            hwcfg = GPU_REG_RD32(LW_PFSP_HWCFG);
            // @todo LW_FALCON_FSP_BASE does not exist.
            info->bar0Base             = LW_PFSP_FALCON_IRQSSET;
            info->riscvBase            = LW_FALCON2_FSP_BASE;
            info->defaultPort          = 5455;
            info->riscv_imem_size      = DRF_VAL(_PFSP, _FALCON_HWCFG3, _IMEM_TOTAL_SIZE, fhwcfg) * FALCON_PAGE_SIZE;
            info->riscv_dmem_size      = DRF_VAL(_PFSP, _FALCON_HWCFG3, _DMEM_TOTAL_SIZE, fhwcfg) * FALCON_PAGE_SIZE;
            info->riscv_emem_size      = DRF_VAL(_PFSP, _HWCFG, _EMEM_SIZE, hwcfg) * FALCON_PAGE_SIZE;
            break;
        case RISCV_INSTANCE_SOE:
            // Hardcoding the base address since there's no SOE register
            info->bar0Base             = 0x840000;
            info->riscvBase            = LW_FALCON2_SOE_BASE;
            info->defaultPort          = 5456; // dont know
            fhwcfg = bar0ReadLegacy(LW_SOE_FALCON_HWCFG3);
            hwcfg = bar0ReadLegacy(LW_SOE_HWCFG);
            info->riscv_imem_size      = DRF_VAL(_SOE, _FALCON_HWCFG3, _IMEM_TOTAL_SIZE, fhwcfg) * FALCON_PAGE_SIZE;
            info->riscv_dmem_size      = DRF_VAL(_SOE, _FALCON_HWCFG3, _DMEM_TOTAL_SIZE, fhwcfg) * FALCON_PAGE_SIZE;
            info->riscv_emem_size      = DRF_VAL(_SOE, _HWCFG, _EMEM_SIZE, hwcfg) * FALCON_PAGE_SIZE;
            break;
        case RISCV_INSTANCE_LWDEC0:
        case RISCV_INSTANCE_LWDEC1:
        case RISCV_INSTANCE_LWDEC2:
        case RISCV_INSTANCE_LWDEC3:
        case RISCV_INSTANCE_LWDEC4:
        case RISCV_INSTANCE_LWDEC5:
        case RISCV_INSTANCE_LWDEC6:
        case RISCV_INSTANCE_LWDEC7:
            // @todo ask to fix manuals to add static defines for all 8 lwdecs (lwrrently only 5 lwdec instances are defined)
            lwdecInstOffset            = (engine - RISCV_INSTANCE_LWDEC0) * 0x4000;
            info->bar0Base             = LW_FALCON_LWDEC0_BASE  + lwdecInstOffset;
            info->riscvBase            = LW_FALCON2_LWDEC0_BASE + lwdecInstOffset;
            info->defaultPort          = 5456;
            fhwcfg = bar0ReadLegacy(LW_PFALCON_FALCON_HWCFG3);
            info->riscv_imem_size      = DRF_VAL(_PFALCON, _FALCON_HWCFG3, _IMEM_TOTAL_SIZE, fhwcfg) * FALCON_PAGE_SIZE;
            info->riscv_dmem_size      = DRF_VAL(_PFALCON, _FALCON_HWCFG3, _DMEM_TOTAL_SIZE, fhwcfg) * FALCON_PAGE_SIZE;
            info->riscv_emem_start     = 0;
            info->riscv_emem_size      = 0;
            break;
        default:
            break;
    }

    info->riscv_dmesg_hdr_addr = info->riscv_dmem_size - sizeof(RiscvDbgDmesgHdr);
}

/*!
 * @brief    Issue reset to the core and wait for reset and IMEM and DMEM scrubbing to finish.
 *
 * @details  Uses LW_PFALCON_* spaces and assume @ref pRiscvInstance set properly.
 *
 * @param[in]  bHammer  whether to issue an engine reset
 * @return     LW_OK on success
 *             LW_ERR_TIMEOUT or LW_ERR_NOT_SUPPORTED on failures
 */
LW_STATUS riscvReset_GH100(LwBool bHammer)
{
    LW_STATUS status = LW_OK;
    LwU32 scrubberMask;
    int timeout = RISCV_SCRUBBING_TIMEOUT_MS;

    // Reset both cores (not possible to reset RISCV separately)
    if (bHammer)
    {
        dprintf("Issuing ENGINE reset to %s...\n", pRiscvInstance->name);
        bar0WriteLegacy(LW_PFALCON_FALCON_ENGINE,
            DRF_DEF(_PFALCON_FALCON, _ENGINE, _RESET, _TRUE));
        riscvDelay(1);
        bar0WriteLegacy(LW_PFALCON_FALCON_ENGINE,
            DRF_DEF(_PFALCON_FALCON, _ENGINE, _RESET, _FALSE));

        // TODO: Remove the condition when FSP implements reset.
        if (!riscvCoreIsFsp())
        {
            // Wait for LW_PFALCON_FALCON_ENGINE_RESET_STATUS to be deasserted
            while (!FLD_TEST_DRF(_PFALCON_FALCON, _ENGINE, _RESET_STATUS, _DEASSERTED,
                bar0ReadLegacy(LW_PFALCON_FALCON_ENGINE)))
            {
                riscvDelay(1);
                timeout--;
                if (!timeout)
                {
                    dprintf("Timeout waiting for LW_PFALCON_FALCON_ENGINE_RESET_STATUS to be deasserted.\n");
                    return LW_ERR_TIMEOUT;
                }
            }
        }
    }
    else
    {
        dprintf("Issuing reset to %s core...\n", pRiscvInstance->name);
        bar0WriteLegacy(LW_PFALCON_FALCON_CPUCTL,
            DRF_DEF(_PFALCON_FALCON, _CPUCTL, _HRESET, _TRUE));
    }

    // Wait for reset to complete
    dprintf("Waiting for start...\n");
    timeout = RISCV_SCRUBBING_TIMEOUT_MS;
    while (FLD_TEST_DRF(_PFALCON_FALCON, _CPUCTL, _HRESET, _TRUE,
        bar0ReadLegacy(LW_PFALCON_FALCON_CPUCTL)))
    {
        riscvDelay(1);
        timeout--;
        if (!timeout)
        {
            dprintf("Timeout waiting for reset.\n");
            return LW_ERR_TIMEOUT;
        }
    }

    // Wait for memory scrubbing
    scrubberMask = DRF_DEF(_PFALCON_FALCON, _DMACTL, _IMEM_SCRUBBING, _PENDING) |
                   DRF_DEF(_PFALCON_FALCON, _DMACTL, _DMEM_SCRUBBING, _PENDING);

    timeout = RISCV_SCRUBBING_TIMEOUT_MS;
    while (bar0ReadLegacy(LW_PFALCON_FALCON_DMACTL) & scrubberMask)
    {
        riscvDelay(1);
        timeout--;
        if (!timeout)
        {
            dprintf("Timeout waiting for memory scrubber.\n");
            return LW_ERR_TIMEOUT;
        }
    }

    return status;
}

LwBool riscvIsBusy_GH100(void)
{
    switch (pRiscvInstance->instance_no)
    {
        case RISCV_INSTANCE_MINION:
            return FLD_TEST_DRF(_CMINION_FALCON, _IDLESTATE, _RISCV_BUSY, _TRUE,
                bar0ReadLegacy(LW_CMINION_FALCON_IDLESTATE));
        case RISCV_INSTANCE_FSP:
            return FLD_TEST_DRF(_CFSP_FALCON, _IDLESTATE, _RISCV_BUSY, _TRUE,
                bar0ReadLegacy(LW_CFSP_FALCON_IDLESTATE));
        case RISCV_INSTANCE_SOE:
            return FLD_TEST_DRF(_SOE_FALCON, _IDLESTATE, _RISCV_BUSY, _TRUE,
                bar0ReadLegacy(LW_SOE_FALCON_IDLESTATE));
        case RISCV_INSTANCE_LWDEC0:
        case RISCV_INSTANCE_LWDEC1:
        case RISCV_INSTANCE_LWDEC2:
        case RISCV_INSTANCE_LWDEC3:
        case RISCV_INSTANCE_LWDEC4:
        case RISCV_INSTANCE_LWDEC5:
        case RISCV_INSTANCE_LWDEC6:
        case RISCV_INSTANCE_LWDEC7: 
            return FLD_TEST_DRF(_CLWDEC_FALCON, _IDLESTATE, _RISCV_BUSY, _TRUE,
                bar0ReadLegacy(LW_CLWDEC_FALCON_IDLESTATE));
        default:
            return riscvIsBusy_GA100();
    }
}

#define dumpStatePrint(X) dprintf("[%08x] %s = %08x\n", X, #X, bar0ReadLegacy(X))
void riscvDumpState_GH100(void)
{
    riscvDumpState_GA10X();

    switch (pRiscvInstance->instance_no)
    {
        case RISCV_INSTANCE_MINION:
            dumpStatePrint(LW_CMINION_FALCON_IDLESTATE);
            break;
        case RISCV_INSTANCE_FSP:
            dumpStatePrint(LW_CFSP_FALCON_IDLESTATE);
            dumpStatePrint(LW_CFSP_FBIF_CTL);
            dumpStatePrint(LW_CFSP_FBIF_CTL2);
            break;
        case RISCV_INSTANCE_SOE:
            dumpStatePrint(LW_SOE_FALCON_IDLESTATE);
            dumpStatePrint(LW_SOE_FBIF_CTL);
            dumpStatePrint(LW_SOE_FBIF_CTL2);
            break;
        case RISCV_INSTANCE_LWDEC0:
        case RISCV_INSTANCE_LWDEC1:
        case RISCV_INSTANCE_LWDEC2:
        case RISCV_INSTANCE_LWDEC3:
        case RISCV_INSTANCE_LWDEC4:
        case RISCV_INSTANCE_LWDEC5:
        case RISCV_INSTANCE_LWDEC6:
        case RISCV_INSTANCE_LWDEC7:
             dumpStatePrint(LW_CLWDEC_FALCON_IDLESTATE);
            dumpStatePrint(LW_CLWDEC_FBIF_CTL);
            // LWDEC doesn't have this register
            // dumpStatePrint(LW_CLWDEC_FBIF_CTL2);
            break;
        default:
            break;
    }
}

LW_STATUS riscvBoot_GH100(LwU64 startAddr, LwBool bStartInIcd)
{
    LwU32 dbgctl;

    switch(pRiscvInstance->instance_no)
    {
        case RISCV_INSTANCE_FSP:
            // Enable FBIF, allow physical addresses
            GPU_REG_WR32(LW_PFSP_FBIF_CTL,
                         DRF_DEF(_PFSP, _FBIF_CTL, _ENABLE, _TRUE) |
                         DRF_DEF(_PFSP, _FBIF_CTL, _ALLOW_PHYS_NO_CTX, _ALLOW));
            // Setup NACK as ACK that should help if we try invalid read
            GPU_REG_WR32(LW_PFSP_FBIF_CTL2, DRF_DEF(_PFSP, _FBIF_CTL2, _NACK_MODE, _NACK_AS_ACK));
            break;
        case RISCV_INSTANCE_SOE:
            // Enable FBIF, allow physical addresses
            bar0WriteLegacy(LW_SOE_FBIF_CTL,
                         DRF_DEF(_SOE, _FBIF_CTL, _ENABLE, _TRUE) |
                         DRF_DEF(_SOE, _FBIF_CTL, _ALLOW_PHYS_NO_CTX, _ALLOW));
            // Setup NACK as ACK that should help if we try invalid read
            bar0WriteLegacy(LW_SOE_FBIF_CTL2, DRF_DEF(_SOE, _FBIF_CTL2, _NACK_MODE, _NACK_AS_ACK));
            break;
        case RISCV_INSTANCE_LWDEC0:
        case RISCV_INSTANCE_LWDEC1:
        case RISCV_INSTANCE_LWDEC2:
        case RISCV_INSTANCE_LWDEC3:
        case RISCV_INSTANCE_LWDEC4:
        case RISCV_INSTANCE_LWDEC5:
        case RISCV_INSTANCE_LWDEC6:
        case RISCV_INSTANCE_LWDEC7: 
            // Enable FBIF, allow physical addresses
            bar0WriteLegacy(LW_CLWDEC_FBIF_CTL,
                            DRF_DEF(_CLWDEC, _FBIF_CTL, _ENABLE, _TRUE) |
                            DRF_DEF(_CLWDEC, _FBIF_CTL, _ALLOW_PHYS_NO_CTX, _ALLOW));
            // LWDEC doesn't have this register
            // bar0WriteLegacy(LW_CLWDEC_FBIF_CTL2, DRF_DEF(_CLWDEC, _FBIF_CTL2, _NACK_MODE, _NACK_AS_ACK));
            break;
        default:
            return riscvBoot_GA10X(startAddr, bStartInIcd);
    }

    // TODO: allow using default value from bootrom, {RISCV_PA_BR_START_HI, RISCV_PA_BR_START_LO}
    bar0Write(LW_PRISCV_RISCV_BOOT_VECTOR_HI, LwU64_HI32(startAddr));
    bar0Write(LW_PRISCV_RISCV_BOOT_VECTOR_LO, LwU64_LO32(startAddr));

    // write core start configuration
    // BRFETCH set to false now because I don't really know what it does
    bar0Write(LW_PRISCV_RISCV_BCR_CTRL, DRF_DEF(_PRISCV_RISCV,_BCR_CTRL,_CORE_SELECT,_RISCV) |
                                        DRF_DEF(_PRISCV_RISCV,_BCR_CTRL,_BRFETCH,_FALSE));

    dbgctl = bar0Read(LW_PRISCV_RISCV_DBGCTL);
    if (bStartInIcd)
    {
        dprintf("Starting in DEBUG mode.\n");
        bar0Write(LW_PRISCV_RISCV_DBGCTL, dbgctl | DRF_DEF(_PRISCV_RISCV,_DBGCTL,_START_IN_ICD,_TRUE));
    } else
    {
        bar0Write(LW_PRISCV_RISCV_DBGCTL, dbgctl & ~(DRF_DEF(_PRISCV_RISCV,_DBGCTL,_START_IN_ICD,_TRUE)));
    }

    dprintf("Starting RISC-V @ "LwU64_FMT"\n", startAddr);
    bar0Write(LW_PRISCV_RISCV_CPUCTL, DRF_DEF(_PRISCV_RISCV,_CPUCTL,_STARTCPU,_TRUE));

    return LW_OK;
}

LwBool riscvIsFbBusy_GH100(void)
{
    if (pRiscvInstance->riscv_fb_size == 0)
    {
        dprintf("%s: No FB access on %s.\n", __FUNCTION__, pRiscvInstance->name);
        return LW_FALSE;
    }
    // TODO: `LW_FALCON_IDLESTATE_FBIF_BUSY` is still a hack (hole in hwref)
    // TODO: This may or may not work, and idk how to test
    switch (pRiscvInstance->instance_no) {
    case RISCV_INSTANCE_FSP:
        return (FLD_TEST_DRF(_FALCON, _IDLESTATE, _FBIF_BUSY, _TRUE,
                             GPU_REG_RD32(LW_PFSP_FALCON_IDLESTATE)));
    case RISCV_INSTANCE_SOE:
        return (FLD_TEST_DRF(_FALCON, _IDLESTATE, _FBIF_BUSY, _TRUE,
                             bar0ReadLegacy(LW_SOE_FALCON_IDLESTATE)));
    case RISCV_INSTANCE_LWDEC0:
    case RISCV_INSTANCE_LWDEC1:
    case RISCV_INSTANCE_LWDEC2:
    case RISCV_INSTANCE_LWDEC3:
    case RISCV_INSTANCE_LWDEC4:
    case RISCV_INSTANCE_LWDEC5:
    case RISCV_INSTANCE_LWDEC6:
    case RISCV_INSTANCE_LWDEC7: 
        return (FLD_TEST_DRF(_FALCON, _IDLESTATE, _FBIF_BUSY, _TRUE,
                             bar0ReadLegacy(LW_CLWDEC_FALCON_IDLESTATE)));
    default:
        return riscvIsFbBusy_GA10X();
    }
}
