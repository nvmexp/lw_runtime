/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include <string.h>
#include <lwtypes.h>
#include <lwmisc.h>
#include <hopper/gh100/dev_riscv_pri.h>
#include <hopper/gh100/dev_falcon_v4.h>
#include <hopper/gh100/dev_se_pri.h>

#include "riscv_prv.h"
#include "riscv_porting.h"

LW_STATUS riscvBrReset_GH100(LwBool bWait)
{
    LwU32 bcr = 0;
    LwU32 timeoutMs = 10000;

    // Check if you can reset engine
    // @todo  Question: minion has no priv mask?
    // if (((bar0ReadLegacy(LW_PFALCON_FALCON_ENGINE__PRIV_LEVEL_MASK) & 0xff) != 0xff) |
    //     ((GPU_REG_RD32(LW_PSE_SE_CTRL_RESET__PRIV_LEVEL_MASK) & 0xff) != 0xff)
    //    )
    // {
    //     dprintf("Can't reset %s - PLM prohibits.\n", pRiscvInstance->name);
    //     return LW_FALSE;
    // }
    dprintf("Issuing reset to %s... ", pRiscvInstance->name);
    bar0WriteLegacy(LW_PFALCON_FALCON_ENGINE,
        DRF_DEF(_PFALCON_FALCON, _ENGINE, _RESET, _TRUE));
    riscvDelay(10);
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
            timeoutMs--;
            if (!timeoutMs)
            {
                dprintf("Timeout waiting for LW_PFALCON_FALCON_ENGINE_RESET_STATUS to be deasserted.\n");
                return LW_ERR_TIMEOUT;
            }
        }
    }

    dprintf("Done.\n");
    dprintf("Issuing reset to SE... ");
    GPU_REG_WR32(LW_PSE_LWPKA_CTRL_RESET,
                DRF_DEF(_PSE, _LWPKA_CTRL_RESET, _SOFT_RESET, _TRIGGER));
    if (bWait)
    {
        timeoutMs = 10000;
        while(!FLD_TEST_DRF(_PSE, _LWPKA_CTRL_RESET, _SOFT_RESET, _DONE,
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

    bcr = DRF_DEF(_PRISCV_RISCV, _BCR_CTRL, _CORE_SELECT, _RISCV)   |
          DRF_DEF(_PRISCV_RISCV, _BCR_CTRL, _VALID,       _TRUE)    |
          DRF_NUM(_PRISCV_RISCV, _BCR_CTRL, _BRFETCH,     0);
    dprintf("Configuring bcr...\n");
    bar0Write(LW_PRISCV_RISCV_BCR_CTRL, bcr);

    return LW_OK;
}
