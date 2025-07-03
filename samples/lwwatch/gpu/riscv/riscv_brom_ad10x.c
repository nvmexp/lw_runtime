/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <string.h>
#include <lwtypes.h>
#include <lwmisc.h>
#include <ada/ad102/lw_pmu_riscv_address_map.h>
#include <ada/ad102/dev_riscv_pri.h>
#include <ada/ad102/dev_falcon_v4.h>
#include <ada/ad102/dev_se_pri.h>

#include "riscv_prv.h"
#include "riscv_porting.h"

// Not in falcon manuals :/
#define ENGINE_RESET_OFFSET         0x3c0
#define ENGINE_RESET_PLM_OFFSET     0x3c4

LW_STATUS riscvBrReset_AD10X(LwBool bWait)
{
    LwU32 bcr = 0;

    // Check if you can reset engine
    if ((bar0ReadLegacy(ENGINE_RESET_PLM_OFFSET) & 0xff) != 0xff)
    {
        dprintf("Can't reset %s - PLM prohibits ENGINE_RESET.\n", pRiscvInstance->name);
        return LW_FALSE;
    }

    dprintf("Issuing reset to %s... ", pRiscvInstance->name);
    bar0WriteLegacy(ENGINE_RESET_OFFSET, 1);
    riscvDelay(10);
    bar0WriteLegacy(ENGINE_RESET_OFFSET, 0);
    dprintf("Done.\n");
    if ((GPU_REG_RD32(LW_PSE_LWPKA_CTRL_RESET__PRIV_LEVEL_MASK) & 0xff) == 0xff)
    {
        dprintf("Issuing reset to SE... ");
        GPU_REG_WR32(LW_PSE_LWPKA_CTRL_RESET,
                    DRF_DEF(_PSE_LWPKA, _CTRL_RESET, _SOFT_RESET, _TRIGGER));
        if (bWait)
        {
            LwU32 timeoutMs = 10000;
            while(!FLD_TEST_DRF(_PSE_LWPKA, _CTRL_RESET, _SOFT_RESET, _DONE,
                                GPU_REG_RD32(LW_PSE_LWPKA_CTRL_RESET)))
            {
                dprintf(".");
                if (timeoutMs > 0)
                {
                    if (timeoutMs >= 10)
                    {
                        riscvDelay(10);
                        timeoutMs -= 10;
                    } else
                    {
                        riscvDelay(timeoutMs);
                        timeoutMs = 0;
                    }
                } else
                {
                    dprintf("Timeout waiting for BROM.\n");
                    return LW_ERR_TIMEOUT;
                }
            }
            dprintf(" Done.\n");
        } else {
            dprintf("Not waiting for reset to complete. SE may be unusable.\n");
        }
    }
    else
    {
        dprintf("Can't reset %s - PLM prohibits SE_RESET. Boot may fail.\n", pRiscvInstance->name);
    }

    bcr = DRF_DEF(_PRISCV_RISCV, _BCR_CTRL, _CORE_SELECT, _RISCV)   |
          DRF_DEF(_PRISCV_RISCV, _BCR_CTRL, _VALID,       _TRUE)    |
          DRF_NUM(_PRISCV_RISCV, _BCR_CTRL, _BRFETCH,     0);
    dprintf("Configuring bcr...\n");
    bar0Write(LW_PRISCV_RISCV_BCR_CTRL, bcr);

    return LW_OK;
}
