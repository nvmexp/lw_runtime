/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2015-2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "acr.h"
#include "rmlsfm.h"
#include "os.h"

#include "chip.h"
#include "disp.h"
#include "pmu.h"
#include "sig.h"
#include "fb.h"
#include "fifo.h"
#include "inst.h"
#include "clk.h"
#include "smbpbi.h"
#include "falcphys.h"

#include "g_acr_private.h"          // (rmconfig) hal/obj setup
#include "pascal/gp102/dev_pwr_pri.h"
#include "pascal/gp102/dev_master.h"
#include "pascal/gp102/dev_falcon_v1.h"
#include "pascal/gp102/dev_fb.h"

void acrGetFalconProp_GP10X(LSFALCPROP *pFalc, LwU32  falconId, LwU32 indexGpu)
{
    switch(falconId)
    {
        case LSF_FALCON_ID_PMU:
                pFalc[falconId].name           = "PMU   ";
                pFalc[falconId].available      = LW_TRUE;
                pFalc[falconId].regBase        = LW_FALCON_PWR_BASE;
                pFalc[falconId].bFalconEnabled = TRUE;
                break;
        default:
            acrGetFalconProp_GM20X(pFalc, falconId, indexGpu);
    }
}


//-------------------------------------------------------------------------------------------------------------
// acrRegionStatus_GP10X - L0 Sanity testing - Verify that read/write is not possible through NS client
//-------------------------------------------------------------------------------------------------------------
LW_STATUS acrRegionStatus_GP10X( LwU32 indexGpu )
{
    LW_STATUS   status   = LW_OK;
    LwU32       value;
    LwU32       oldVal;
    LwU32       newVal;
    LwU32       cmd;
    LwU32       regionId;
    LwU32       readMask[LW_PFB_PRI_MMU_VPR_WPR_WRITE_ALLOW_READ_WPR__SIZE_1];
    LwU32       writeMask[LW_PFB_PRI_MMU_VPR_WPR_WRITE_ALLOW_READ_WPR__SIZE_1];
    LwU64       startAddr[LW_PFB_PRI_MMU_VPR_WPR_WRITE_ALLOW_READ_WPR__SIZE_1];

    dprintf("----------------------------------\n");
    dprintf("||  regionId  ||  Status        ||\n");
    dprintf("----------------------------------\n");

    for (regionId=1; regionId <= (LW_PFB_PRI_MMU_VPR_WPR_WRITE_ALLOW_READ_WPR__SIZE_1-1); regionId++)
    {
        // Read start address
        cmd = FLD_SET_DRF_IDX(_PFB, _PRI_MMU_WPR, _INFO_INDEX, _WPR_ADDR_LO, regionId, 0);
        GPU_REG_WR32(LW_PFB_PRI_MMU_WPR_INFO, cmd);
        startAddr[regionId] = GPU_REG_RD_DRF(_PFB, _PRI_MMU_WPR_INFO, _DATA);
        startAddr[regionId] = startAddr[regionId] << LW_PFB_PRI_MMU_WPR_INFO_ADDR_ALIGNMENT;

        // Read bit0 ReadMask
        cmd = FLD_SET_DRF(_PFB, _PRI_MMU_WPR, _INFO_INDEX, _ALLOW_READ, 0);
        GPU_REG_WR32(LW_PFB_PRI_MMU_WPR_INFO, cmd);

        readMask[regionId] = DRF_IDX_OFFSET_VAL(_PFB, _PRI_MMU_WPR_INFO, _ALLOW_READ_WPR_SELWRE,
                                                regionId, 0, GPU_REG_RD32(LW_PFB_PRI_MMU_WPR_INFO));
        // Read bit0 WriteMask
        cmd = FLD_SET_DRF(_PFB, _PRI_MMU_WPR, _INFO_INDEX, _ALLOW_WRITE, 0);
        GPU_REG_WR32(LW_PFB_PRI_MMU_WPR_INFO, cmd);

        writeMask[regionId] = DRF_IDX_OFFSET_VAL(_PFB, _PRI_MMU_WPR_INFO, _ALLOW_WRITE_WPR_SELWRE,
                                                 regionId, 0, GPU_REG_RD32(LW_PFB_PRI_MMU_WPR_INFO));

        dprintf("||  %d         ||  ", regionId);

        // The hardware manuals don't define any enum values for these
        // registers, but 1 means "secure" (i.e., no access).
        if (readMask[regionId])
        {
            dprintf("Failed - Read mask configuration error        ||\n");
            continue;
        }
        else if (writeMask[regionId])
        {
            dprintf("Failed - Write mask configuration error        ||\n");
            continue;
        }
        else
        {
            pFb[indexGpu].fbRead(startAddr[regionId], &oldVal, sizeof(LwU32));

            // Check if reading selwred region returns value 0xBAD0xxxx
            if(0xBAD0 != (oldVal>>16) )
            {
                dprintf("Failed - Read error        ||\n");
                continue;
            }

            // Generating a new value by toggling bits;
            value = oldVal ^ 0xffffffff;

            pFb[indexGpu].fbWrite(startAddr[regionId], &value, sizeof(LwU32));
            pFb[indexGpu].fbRead(startAddr[regionId], &newVal, sizeof(LwU32));

            if((newVal>>16) != (oldVal>>16))
            {
                dprintf("Failed - Write error       ||\n");

                // Restore previous value
                pFb[indexGpu].fbWrite(startAddr[regionId], &oldVal, sizeof(LwU32));
                continue;
            }
        }

        dprintf("Success        ||\n");
    }

    dprintf("----------------------------------\n");
    return status;
}
