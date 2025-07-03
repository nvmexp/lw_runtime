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
#include "ampere/ga102/dev_gsp.h"
#include "ampere/ga102/dev_sec_pri.h"
#include "ampere/ga102/dev_pwr_pri.h"
#include "ampere/ga102/dev_riscv_pri.h"
#include "ampere/ga102/dev_falcon_v4.h"
#include "ampere/ga102/lw_pmu_riscv_address_map.h"
// For the outstanding holes in hwref
#include "riscv_regs.h"

#include "g_riscv_private.h"

LW_STATUS riscvBoot_GA10X(LwU64 startAddr, LwBool bStartInIcd)
{
    LwU32 dbgctl;

    switch(pRiscvInstance->instance_no) {
        case RISCV_INSTANCE_GSP:
            // Enable FBIF, allow physical addresses
            GPU_REG_WR32(LW_PGSP_FBIF_CTL,
                         DRF_DEF(_PGSP, _FBIF_CTL, _ENABLE, _TRUE) |
                         DRF_DEF(_PGSP, _FBIF_CTL, _ALLOW_PHYS_NO_CTX, _ALLOW));
            // Setup NACK as ACK that should help if we try invalid read
            GPU_REG_WR32(LW_PGSP_FBIF_CTL2, DRF_DEF(_PGSP, _FBIF_CTL2, _NACK_MODE, _NACK_AS_ACK));
            break;
        case RISCV_INSTANCE_SEC2:
            // Enable FBIF, allow physical addresses
            GPU_REG_WR32(LW_PSEC_FBIF_CTL,
                         DRF_DEF(_PSEC, _FBIF_CTL, _ENABLE, _TRUE) |
                         DRF_DEF(_PSEC, _FBIF_CTL, _ALLOW_PHYS_NO_CTX, _ALLOW));
            // Setup NACK as ACK that should help if we try invalid read
            GPU_REG_WR32(LW_PSEC_FBIF_CTL2, DRF_DEF(_PSEC, _FBIF_CTL2, _NACK_MODE, _NACK_AS_ACK));
            break;
        case RISCV_INSTANCE_PMU:
            // enable full address range of dmem
            // see https://lwbugswb.lwpu.com/LwBugs5/HWBug.aspx?bugid=2489286
            // NOTE: This SHOULD NOT WORK on GA10X
            // TODO investigate and remove
            GPU_REG_WR32(LW_PPWR_FALCON_DMVACTL, (LwU32)(pRiscvInstance->riscv_dmem_size / FALCON_PAGE_SIZE));

            // Enable FBIF, allow physical addresses
            GPU_REG_WR32(LW_PPWR_FBIF_CTL,
                         DRF_DEF(_PPWR, _FBIF_CTL, _ENABLE, _TRUE) |
                         DRF_DEF(_PPWR, _FBIF_CTL, _ALLOW_PHYS_NO_CTX, _ALLOW));
            // Setup NACK as ACK that should help if we try invalid read
            GPU_REG_WR32(LW_PPWR_FBIF_CTL2, DRF_DEF(_PPWR, _FBIF_CTL2, _NACK_MODE, _NACK_AS_ACK));
            break;
        case RISCV_INSTANCE_MINION:
            // Minion has no FB access so no action here.
            break;
        case RISCV_INSTANCE_TSEC:
            // @todo  check what should be done here
            break;
        default:
            return LW_ERR_NOT_SUPPORTED;
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

#define pr(X) dprintf("[%08x] %s = %08x\n", X, #X, bar0Read(X))
#define prl(X) dprintf("[%08x] %s = %08x\n", X, #X, GPU_REG_RD32(X))

void riscvDumpState_GA10X(void)
{
    pr(LW_PRISCV_RISCV_BOOT_VECTOR_HI);
    pr(LW_PRISCV_RISCV_BOOT_VECTOR_LO);
    pr(LW_PRISCV_RISCV_CPUCTL);
    pr(LW_PRISCV_RISCV_IRQMASK);
    pr(LW_PRISCV_RISCV_IRQDEST);
    pr(LW_PRISCV_RISCV_DBGCTL);
    switch (pRiscvInstance->instance_no) {
        case RISCV_INSTANCE_GSP:
            prl(LW_PGSP_FALCON_IDLESTATE);
            prl(LW_PGSP_FBIF_CTL);
            prl(LW_PGSP_FBIF_CTL2);
            break;
        case RISCV_INSTANCE_SEC2:
            prl(LW_PSEC_FALCON_IDLESTATE);
            prl(LW_PSEC_FBIF_CTL);
            prl(LW_PSEC_FBIF_CTL2);
            break;
        case RISCV_INSTANCE_PMU:
            prl(LW_PPWR_FALCON_IDLESTATE);
            prl(LW_PPWR_FBIF_CTL);
            prl(LW_PPWR_FBIF_CTL2);
            break;
        default:
            break;
    }
}

LwBool riscvIsFbBusy_GA10X(void)
{
    if (pRiscvInstance->riscv_fb_size == 0)
    {
        dprintf("%s: No FB access on %s.\n", __FUNCTION__, pRiscvInstance->name);
        return LW_FALSE;
    }
    // TODO: `LW_FALCON_IDLESTATE_FBIF_BUSY` is still a hack (hole in hwref)
    // TODO: This may or may not work, and idk how to test
    switch (pRiscvInstance->instance_no) {
    case RISCV_INSTANCE_GSP:
        return (FLD_TEST_DRF(_FALCON, _IDLESTATE, _FBIF_BUSY, _TRUE,
                             GPU_REG_RD32(LW_PGSP_FALCON_IDLESTATE)));
    case RISCV_INSTANCE_SEC2:
        return (FLD_TEST_DRF(_FALCON, _IDLESTATE, _FBIF_BUSY, _TRUE,
                             GPU_REG_RD32(LW_PSEC_FALCON_IDLESTATE)));
    case RISCV_INSTANCE_PMU:
        return (FLD_TEST_DRF(_FALCON, _IDLESTATE, _FBIF_BUSY, _TRUE,
                             GPU_REG_RD32(LW_PPWR_FALCON_IDLESTATE)));
    default:
        dprintf("%s: Config conflict on %s.\n", __FUNCTION__, pRiscvInstance->name);
        return LW_FALSE;
    }
}

void riscvPopulateCoreInfo_GA10X(RiscVInstance *info, RiscvInstanceType engine) {
    LwU32 fhwcfg, hwcfg;

    if (!pRiscv[indexGpu].riscvIsInstanceSupported(engine))
    {
        info->name = NULL;
        return;
    }

    info->name                 = riscvInstanceNames[engine];
    info->instance_no          = engine;
    info->riscv_imem_start     = LW_RISCV_AMAP_IMEM_START;
    info->riscv_dmem_start     = LW_RISCV_AMAP_DMEM_START;
    switch (engine) {
    case RISCV_INSTANCE_GSP:
        fhwcfg = GPU_REG_RD32(LW_PGSP_FALCON_HWCFG3);
        hwcfg = GPU_REG_RD32(LW_PGSP_HWCFG);
        info->bar0Base             = LW_FALCON_GSP_BASE;
        info->riscvBase            = LW_FALCON2_GSP_BASE;
        info->defaultPort          = 4444;
        // hwcfg gives mem sizes in units of 256 bytes.
        info->riscv_imem_size      = DRF_VAL(_PGSP, _FALCON_HWCFG3, _IMEM_TOTAL_SIZE, fhwcfg) * FALCON_PAGE_SIZE;
        info->riscv_dmem_size      = DRF_VAL(_PGSP, _FALCON_HWCFG3, _DMEM_TOTAL_SIZE, fhwcfg) * FALCON_PAGE_SIZE;
        info->riscv_emem_start     = LW_RISCV_AMAP_EMEM_START;
        info->riscv_emem_size      = DRF_VAL(_PGSP, _HWCFG, _EMEM_SIZE, hwcfg) * FALCON_PAGE_SIZE;
        info->riscv_fb_start       = LW_RISCV_AMAP_FBGPA_START;
        info->riscv_fb_size        = LW_RISCV_AMAP_FBGPA_SIZE;
        info->riscv_priv_start     = LW_RISCV_AMAP_PRIV_START;
        info->riscv_priv_size      = LW_RISCV_AMAP_PRIV_SIZE;
        info->riscv_dmesg_hdr_addr = info->riscv_dmem_size - sizeof(RiscvDbgDmesgHdr);
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
        info->riscv_emem_start     = LW_RISCV_AMAP_EMEM_START;
        info->riscv_emem_size      = DRF_VAL(_PSEC, _HWCFG, _EMEM_SIZE, hwcfg) * FALCON_PAGE_SIZE;
        info->riscv_fb_start       = LW_RISCV_AMAP_FBGPA_START;
        info->riscv_fb_size        = LW_RISCV_AMAP_FBGPA_SIZE;
        info->riscv_priv_start     = LW_RISCV_AMAP_PRIV_START;
        info->riscv_priv_size      = LW_RISCV_AMAP_PRIV_SIZE;
        info->riscv_dmesg_hdr_addr = info->riscv_dmem_size - sizeof(RiscvDbgDmesgHdr);
        break;
    case RISCV_INSTANCE_PMU:
        fhwcfg = GPU_REG_RD32(LW_PPWR_FALCON_HWCFG3);
        info->bar0Base             = LW_FALCON_PWR_BASE;
        info->riscvBase            = LW_FALCON2_PWR_BASE;
        info->defaultPort          = 4545;
        info->riscv_imem_size      = DRF_VAL(_PPWR, _FALCON_HWCFG3, _IMEM_TOTAL_SIZE, fhwcfg) * FALCON_PAGE_SIZE;
        info->riscv_dmem_size      = DRF_VAL(_PPWR, _FALCON_HWCFG3, _DMEM_TOTAL_SIZE, fhwcfg) * FALCON_PAGE_SIZE;
        info->riscv_emem_start     = 0;
        info->riscv_emem_size      = 0x0;
        info->riscv_fb_start       = LW_RISCV_AMAP_FBGPA_START;
        info->riscv_fb_size        = LW_RISCV_AMAP_FBGPA_SIZE;
        info->riscv_priv_start     = LW_RISCV_AMAP_PRIV_START;
        info->riscv_priv_size      = LW_RISCV_AMAP_PRIV_SIZE;
        info->riscv_dmesg_hdr_addr = info->riscv_dmem_size - sizeof(RiscvDbgDmesgHdr);
        break;
    default:
        break;
    }
}
