
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch debug extension
// fbgp100.c
//
//*****************************************************

//
// includes
//
#include "pascal/gp100/dev_pri_ringmaster.h"
#include "pascal/gp100/dev_fbpa.h"
#include "pascal/gp100/dev_ltc.h"
#include "pascal/gp100/dev_graphics_nobundle.h"
#include "chip.h"
#include "fb.h"
#include "priv.h"
#include "pascal/gp100/hwproject.h"

/*!
 *  Callwlates the total fb RAM
 *  Total = #fb partitions * RAM per fbpa
 *
 *  @return total fb memory in MB.
 */
LwU32 fbGetMemSizeMb_GP100( void )
{
    LwU32   data32 = GPU_REG_RD_DRF(_PFB_FBPA, _CSTATUS, _RAMAMOUNT);
    LwU32   fbps = GPU_REG_RD_DRF(_PPRIV_MASTER, _RING_ENUMERATE_RESULTS_FBP, _COUNT);
    return (fbps * data32);
}

setup_writes_t * fbGetPMEnableWrites_GP100( void )
{
    static setup_writes_t PmEnableWrites_GP100[] =
    {
        { "LW_PFB_FBPA_PM_ENABLE"  , LW_PFB_FBPA_PM, 0x00000001, 0x00000001},
        { "LW_PFB_FBPA_PM_SELECT"  , LW_PFB_FBPA_PM, 0x00000110, 0x000003f0},
        { "LW_PFB_FBPA_PM_ENABLE"  , LW_PFB_FBPA_PM, 0x00000001, 0x00000001}, 
        { "LW_PFB_FBPA_PM_SELECT"  , LW_PFB_FBPA_PM, 0x00000110, 0x000003f0},
        {NULL}
    };

    return(PmEnableWrites_GP100);
}

LwU32 fbGetFBIOBroadcastDDRMode_GP100( void )
{
    return GPU_REG_RD_DRF(_PFB_FBPA, _FBIO_BROADCAST, _DDR_MODE);
}

LW_STATUS fbL2StateForCacheLines_GP100
(
    LwU32 numActiveLTCs,
    LwU32 numLTSPerLTC,
    LwU32 numLinesPerLTS
)
{
    LwU32 i, j, k, l;
    LwU32 val;
    LwU32 valOrig;
    LwU32 val2;

    dprintf("lw:\tReading out cache tag stage:\n");
    for (i = 0; i < numActiveLTCs; i++)
    {
        dprintf("lw:\tLTC %d\n", i);

        for (j = 0; j < numLTSPerLTC; j++)
        {
            dprintf("lw:\t\tLTS %d\n", j);

            val = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_TSTG_CST_RDI_INDEX +
                              (LW_LTC_PRI_STRIDE * i) +
                              (LW_LTS_PRI_STRIDE * j));
            valOrig = val;

            // Loops through cache lines...
            for (k = 0; k < numLinesPerLTS; k++ )
            {
                //
                // The value of _ADDRESS controls which L2 cache line tag and
                // state to access.
                //
                val &= ~DRF_SHIFTMASK(LW_PLTCG_LTC0_LTS0_TSTG_CST_RDI_INDEX_ADDRESS);
                val |=  DRF_NUM(_PLTCG, _LTC0_LTS0_TSTG_CST_RDI_INDEX, _ADDRESS, k);
                GPU_REG_WR32(LW_PLTCG_LTC0_LTS0_TSTG_CST_RDI_INDEX +
                            (LW_LTC_PRI_STRIDE * i) +
                            (LW_LTS_PRI_STRIDE * j),
                            val);

                dprintf("lw:\t\t\tLine %d\n", k);
                dprintf("lw:\t\t\t\t");
                for (l = 0; l < 3; l++)
                {
                    val2 = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_TSTG_CST_RDI_DATA(l) +
                                       (LW_LTC_PRI_STRIDE * i) +
                                       (LW_LTS_PRI_STRIDE * j));
                    dprintf(" [%d]:0x%x", l, val2);
                }
                dprintf("\n");
            }

            // Restore the original value.
            GPU_REG_WR32(LW_PLTCG_LTC0_LTS0_TSTG_CST_RDI_INDEX +
                        (LW_LTC_PRI_STRIDE * i) +
                        (LW_LTS_PRI_STRIDE * j),
                         valOrig);
        }
    }

    return LW_OK;
}
