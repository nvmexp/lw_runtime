/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All information
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

#include "t23x/t234/dev_sec_pri.h"
#include "t23x/t234/dev_riscv_pri.h"
#include "t23x/t234/dev_falcon_v4.h"
// For the outstanding holes in hwref
#include "riscv_regs.h"

#include "g_riscv_private.h"

/*!
 * @copydoc riscvIsInstanceSupported_TU10X
 */
LwBool riscvIsInstanceSupported_T234(RiscvInstanceType type)
{
    switch (type)
    {
        case RISCV_INSTANCE_PMU:
        case RISCV_INSTANCE_TSEC:
        case RISCV_INSTANCE_LWDEC0:
            return LW_TRUE;
        default:
            break;
    }
    return LW_FALSE;
}

/*!
 * @copydoc  riscvPopulateCoreInfo_TU10X
 * @note     TSEC uses RISCV starting T234.
 */
void
riscvPopulateCoreInfo_T234
(
    RiscVInstance *info,
    RiscvInstanceType engine
)
{
    LwU32 falconHwcfg3;

    if (!pRiscv[indexGpu].riscvIsInstanceSupported(engine))
    {
        info->name = NULL;
        return;
    }

    info->name             = riscvInstanceNames[engine];
    info->instance_no      = engine;
    info->riscv_imem_start = pRiscv[indexGpu].getRiscvDef(engine, "LW_RISCV_AMAP_IMEM_START");
    info->riscv_dmem_start = pRiscv[indexGpu].getRiscvDef(engine, "LW_RISCV_AMAP_DMEM_START");
    info->riscv_fb_start   = pRiscv[indexGpu].getRiscvDef(engine, "LW_RISCV_AMAP_FBGPA_START");
    info->riscv_fb_size    = pRiscv[indexGpu].getRiscvDef(engine, "LW_RISCV_AMAP_FBGPA_SIZE");
    info->riscv_emem_start = pRiscv[indexGpu].getRiscvDef(engine, "LW_RISCV_AMAP_EMEM_START");
    info->riscv_emem_size  = (LwU32) pRiscv[indexGpu].getRiscvDef(engine, "LW_RISCV_AMAP_EMEM_SIZE");

    switch (engine) {
        case RISCV_INSTANCE_PMU:
            info->bar0Base             = LW_FALCON_PWR_BASE;
            info->riscvBase            = LW_FALCON2_PWR_BASE;
            falconHwcfg3               = bar0ReadLegacy(LW_PFALCON_FALCON_HWCFG3);
            info->defaultPort          = 4545;
            info->riscv_imem_size      = DRF_VAL(_PFALCON, _FALCON_HWCFG3, _IMEM_TOTAL_SIZE, falconHwcfg3) * FALCON_PAGE_SIZE;
            info->riscv_dmem_size      = DRF_VAL(_PFALCON, _FALCON_HWCFG3, _DMEM_TOTAL_SIZE, falconHwcfg3) * FALCON_PAGE_SIZE;
            info->riscv_emem_start     = 0;
            info->riscv_emem_size      = 0;
            info->riscv_priv_start     = 0;
            info->riscv_priv_size      = 0;
            break;
        case RISCV_INSTANCE_TSEC:
            // @todo  drop hard coded values
            info->bar0Base             = 0x1000;
            info->riscvBase            = 0x2000;
            info->defaultPort          = 5544;
            falconHwcfg3               = bar0ReadLegacy(LW_PFALCON_FALCON_HWCFG3);
            info->riscv_imem_size      = DRF_VAL(_PFALCON, _FALCON_HWCFG3, _IMEM_TOTAL_SIZE, falconHwcfg3) * FALCON_PAGE_SIZE;
            info->riscv_dmem_size      = DRF_VAL(_PFALCON, _FALCON_HWCFG3, _DMEM_TOTAL_SIZE, falconHwcfg3) * FALCON_PAGE_SIZE;
            info->riscv_priv_start     = 0;
            info->riscv_priv_size      = 0;
            break;
        case RISCV_INSTANCE_LWDEC0:
            // @todo  drop hard coded values
            info->bar0Base             = 0x1000;
            info->riscvBase            = 0x4000;
            info->defaultPort          = 5454;
            falconHwcfg3               = bar0ReadLegacy(LW_PFALCON_FALCON_HWCFG3);
            info->riscv_imem_size      = DRF_VAL(_PFALCON, _FALCON_HWCFG3, _IMEM_TOTAL_SIZE, falconHwcfg3) * FALCON_PAGE_SIZE;
            info->riscv_dmem_size      = DRF_VAL(_PFALCON, _FALCON_HWCFG3, _DMEM_TOTAL_SIZE, falconHwcfg3) * FALCON_PAGE_SIZE;
            info->riscv_priv_start     = 0;
            info->riscv_priv_size      = 0;
            break;
        default:
            break;
    }

    info->riscv_dmesg_hdr_addr = info->riscv_dmem_size - sizeof(RiscvDbgDmesgHdr);
}

LwBool riscvIsBusy_T234(void)
{
    switch (pRiscvInstance->instance_no)
    {
        case RISCV_INSTANCE_PMU:
        case RISCV_INSTANCE_TSEC:
        case RISCV_INSTANCE_LWDEC0:
            return FLD_TEST_DRF(_PFALCON_FALCON, _IDLESTATE, _RISCV_BUSY, _TRUE,
                bar0ReadLegacy(LW_PFALCON_FALCON_IDLESTATE));
        default:
            return LW_TRUE;
    }
}
