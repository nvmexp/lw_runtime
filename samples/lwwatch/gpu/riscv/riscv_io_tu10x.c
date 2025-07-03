/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All information
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
#include "turing/tu102/dev_gsp.h"
#include "turing/tu102/dev_sec_pri.h"
#include "turing/tu102/dev_pwr_pri.h"
#include "turing/tu102/dev_riscv_pri.h"
#include "turing/tu102/dev_falcon_v4.h"
#include "turing/tu102/lw_gsp_riscv_address_map.h"
// For the outstanding holes in hwref
#include "riscv_regs.h"

#include "g_riscv_private.h"

/*!
 * @brief    Return if the given instance is supported.
 * @details  Fork a new version when new RISCV engines are supported.
 *           Only necessary to use this function when switching core.
 *           For now, referencing @ref riscvPopulateCoreInfo_TU10X.
 *
 * @param[in]  type  RISCV instance type.
 *
 * @return     LW_TURE if supported.
 * @return     LW_FALSE otherwise.
 */
LwBool riscvIsInstanceSupported_TU10X(RiscvInstanceType type)
{
    switch (type)
    {
        case RISCV_INSTANCE_GSP:
        case RISCV_INSTANCE_SEC2:
            return LW_TRUE;
        default:
            break;
    }
    return LW_FALSE;
}

LW_STATUS riscvBoot_TU10X(LwU64 startAddr, LwBool bStartInIcd)
{
    switch (pRiscvInstance->instance_no) {
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
    default:
        return LW_ERR_NOT_SUPPORTED;
    }

    bar0Write(LW_PRISCV_RISCV_BOOT_VECTOR_HI, LwU64_HI32(startAddr));
    bar0Write(LW_PRISCV_RISCV_BOOT_VECTOR_LO, LwU64_LO32(startAddr));

    if (!bStartInIcd)
    {
        dprintf("Starting RISC-V @ "LwU64_FMT"\n", startAddr);
        bar0Write(LW_PRISCV_RISCV_CPUCTL, DRF_DEF(_PRISCV_RISCV,_CPUCTL,_STARTCPU,_TRUE));
    }
    else
    {
        dprintf("Starting RISC-V in DEBUG mode @ "LwU64_FMT"\n", startAddr);
        bar0Write(LW_PRISCV_RISCV_CPUCTL, DRF_DEF(_PRISCV_RISCV,_CPUCTL,_START_IN_ICD,_TRUE) |
                                          DRF_DEF(_PRISCV_RISCV, _CPUCTL, _STARTCPU, _TRUE));
    }

    return LW_OK;
}

LW_STATUS riscvReset_TU10X(LwBool bHammer)
{
    LwU32 scrubberMask;
    int timeout = RISCV_SCRUBBING_TIMEOUT_MS;

    switch (pRiscvInstance->instance_no) {
    case RISCV_INSTANCE_GSP:
        // Reset both cores (not possible to reset RISCV separately)
        if (bHammer)
        {
            dprintf("Issuing ENGINE reset to %s...\n", pRiscvInstance->name);
            GPU_REG_WR32(LW_PGSP_FALCON_ENGINE, LW_PGSP_FALCON_ENGINE_RESET_TRUE);
            riscvDelay(1);
            GPU_REG_WR32(LW_PGSP_FALCON_ENGINE, LW_PGSP_FALCON_ENGINE_RESET_FALSE);
        } else
        {
            dprintf("Issuing reset to %s core...\n", pRiscvInstance->name);
            GPU_REG_WR32(LW_PGSP_FALCON_CPUCTL, DRF_DEF(_PGSP_FALCON, _CPUCTL, _HRESET, _TRUE));
        }

        dprintf("Waiting for start...\n");
        // Wait for reset to complete
        while (FLD_TEST_DRF(_PGSP_FALCON,_CPUCTL,_HRESET,_TRUE, GPU_REG_RD32(LW_PGSP_FALCON_CPUCTL)))
        {
            riscvDelay(1);
            timeout --;
            if (!timeout)
            {
                dprintf("Timeout waiting for reset.\n");
                return LW_ERR_TIMEOUT;
            }
        }
        // Wait for memory scrubbing
        scrubberMask = DRF_DEF(_PGSP_FALCON, _DMACTL, _IMEM_SCRUBBING, _PENDING) |
            DRF_DEF(_PGSP_FALCON, _DMACTL, _DMEM_SCRUBBING, _PENDING);

        while (GPU_REG_RD32(LW_PGSP_FALCON_DMACTL) & scrubberMask)
        {
            riscvDelay(1);
            timeout --;
            if (!timeout)
            {
                dprintf("Timeout waiting for memory scrubber.\n");
                return LW_ERR_TIMEOUT;
            }
        }
        break;
    case RISCV_INSTANCE_SEC2:
        // Reset both cores (not possible to reset RISCV separately)
        if (bHammer)
        {
            dprintf("Issuing ENGINE reset to %s...\n", pRiscvInstance->name);
            GPU_REG_WR32(LW_PSEC_FALCON_ENGINE, LW_PSEC_FALCON_ENGINE_RESET_TRUE);
            riscvDelay(1);
            GPU_REG_WR32(LW_PSEC_FALCON_ENGINE, LW_PSEC_FALCON_ENGINE_RESET_FALSE);
        } else
        {
            dprintf("Issuing reset to %s core...\n", pRiscvInstance->name);
            GPU_REG_WR32(LW_PSEC_FALCON_CPUCTL, DRF_DEF(_PSEC_FALCON, _CPUCTL, _HRESET, _TRUE));
        }

        dprintf("Waiting for start...\n");
        // Wait for reset to complete
        while (FLD_TEST_DRF(_PSEC_FALCON,_CPUCTL,_HRESET,_TRUE, GPU_REG_RD32(LW_PSEC_FALCON_CPUCTL)))
        {
            riscvDelay(1);
            timeout --;
            if (!timeout)
            {
                dprintf("Timeout waiting for reset.\n");
                return LW_ERR_TIMEOUT;
            }
        }
        // Wait for memory scrubbing
        scrubberMask = DRF_DEF(_PSEC_FALCON, _DMACTL, _IMEM_SCRUBBING, _PENDING) |
            DRF_DEF(_PSEC_FALCON, _DMACTL, _DMEM_SCRUBBING, _PENDING);

        while (GPU_REG_RD32(LW_PSEC_FALCON_DMACTL) & scrubberMask)
        {
            riscvDelay(1);
            timeout --;
            if (!timeout)
            {
                dprintf("Timeout waiting for memory scrubber.\n");
                return LW_ERR_TIMEOUT;
            }
        }
        break;
    default:
        return LW_ERR_NOT_SUPPORTED;
    }

    return LW_OK;
}

#define pr(X) dprintf("[%08x] %s = %08x\n", X, #X, bar0Read(X))
#define prl(X) dprintf("[%08x] %s = %08x\n", X, #X, GPU_REG_RD32(X))

void riscvDumpState_TU10X(void)
{
    pr(LW_PRISCV_RISCV_CORE_SWITCH_RISCV_STATUS);
    pr(LW_PRISCV_RISCV_CORE_SWITCH_FALCON_STATUS);
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
    default:
        dprintf("Instance not supported.\n");
    }
}

LwBool riscvIsFbBusy_TU10X(void)
{
    // TODO: `LW_FALCON_IDLESTATE_FBIF_BUSY` is still a hack (hole in hwref)
    switch (pRiscvInstance->instance_no) {
    case RISCV_INSTANCE_GSP:
        return (FLD_TEST_DRF(_FALCON, _IDLESTATE, _FBIF_BUSY, _TRUE,
                             GPU_REG_RD32(LW_PGSP_FALCON_IDLESTATE)));
    case RISCV_INSTANCE_SEC2:
        return (FLD_TEST_DRF(_FALCON, _IDLESTATE, _FBIF_BUSY, _TRUE,
                             GPU_REG_RD32(LW_PSEC_FALCON_IDLESTATE)));
    default:
        return LW_TRUE;
    }
}

LwBool riscvIsBusy_TU10X(void)
{
    switch (pRiscvInstance->instance_no) {
    case RISCV_INSTANCE_GSP:
        return (FLD_TEST_DRF(_PGSP_FALCON, _IDLESTATE, _RISCV_BUSY, _TRUE,
                             GPU_REG_RD32(LW_PGSP_FALCON_IDLESTATE)));
    case RISCV_INSTANCE_SEC2:
        return (FLD_TEST_DRF(_PSEC_FALCON, _IDLESTATE, _RISCV_BUSY, _TRUE,
                             GPU_REG_RD32(LW_PSEC_FALCON_IDLESTATE)));
    default:
        return LW_TRUE;
    }
}

/*!
 * @brief      Populate structure @ref RiscVInstance for a given engine.
 *
 * @param[out] info    Pointer to the core info structure for the engine.
 * @param[in]  engine  The engine to populate the info for.
 */
void riscvPopulateCoreInfo_TU10X(RiscVInstance *info, RiscvInstanceType engine) {
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
        fhwcfg = GPU_REG_RD32(LW_PGSP_FALCON_HWCFG);
        hwcfg = GPU_REG_RD32(LW_PGSP_HWCFG);
        info->bar0Base             = LW_FALCON_GSP_BASE;
        info->riscvBase            = LW_FALCON2_GSP_BASE;
        info->defaultPort          = 4444;
        // hwcfg gives mem sizes in units of 256 bytes.
        info->riscv_imem_size      = DRF_VAL(_PGSP, _FALCON_HWCFG, _IMEM_SIZE, fhwcfg) * FALCON_PAGE_SIZE;
        info->riscv_dmem_size      = DRF_VAL(_PGSP, _FALCON_HWCFG, _DMEM_SIZE, fhwcfg) * FALCON_PAGE_SIZE;
        info->riscv_emem_start     = LW_RISCV_AMAP_EMEM_START;
        info->riscv_emem_size      = DRF_VAL(_PGSP, _HWCFG, _EMEM_SIZE, hwcfg) * FALCON_PAGE_SIZE;
        info->riscv_fb_start       = LW_RISCV_AMAP_FBGPA_START;
        info->riscv_fb_size        = LW_RISCV_AMAP_FBGPA_SIZE;
        info->riscv_priv_start     = LW_RISCV_AMAP_PRIV_START;
        info->riscv_priv_size      = LW_RISCV_AMAP_PRIV_SIZE;
        info->riscv_dmesg_hdr_addr = info->riscv_dmem_size - sizeof(RiscvDbgDmesgHdr);
        break;
    case RISCV_INSTANCE_SEC2:
        fhwcfg = GPU_REG_RD32(LW_PSEC_FALCON_HWCFG);
        hwcfg = GPU_REG_RD32(LW_PSEC_HWCFG);
        info->bar0Base             = LW_FALCON_SEC_BASE;
        info->riscvBase            = LW_FALCON2_SEC_BASE;
        info->defaultPort          = 5555;
        info->riscv_imem_size      = DRF_VAL(_PSEC, _FALCON_HWCFG, _IMEM_SIZE, fhwcfg) * FALCON_PAGE_SIZE;
        info->riscv_dmem_size      = DRF_VAL(_PSEC, _FALCON_HWCFG, _DMEM_SIZE, fhwcfg) * FALCON_PAGE_SIZE;
        info->riscv_emem_start     = LW_RISCV_AMAP_EMEM_START;
        info->riscv_emem_size      = DRF_VAL(_PSEC, _HWCFG, _EMEM_SIZE, hwcfg) * FALCON_PAGE_SIZE;
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
