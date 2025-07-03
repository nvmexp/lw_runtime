
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2005-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch debug extension
// fbgm200.c
//
//*****************************************************

//
// includes
//
#include "maxwell/gm200/dev_ltc.h"
#include "maxwell/gm200/dev_fuse.h"
#include "maxwell/gm200/dev_pri_ringmaster.h"
#include "maxwell/gm200/dev_top.h"
#include "maxwell/gm200/hwproject.h"
#include "fb.h"
#include "sig.h"

#define VIDMEMBIT   0
#define SYSMEMBIT   1

LW_STATUS fbIsMemReq_GM200(LwU32 nFbp)
{
    LwU32 i, j, val, data32;
    char* access[2] = { "YES", "NO" };
    char* statusVid;
    char* statusSys;
    LwU32 numLTCSlices = pFb[indexGpu].fbGetLTSPerLTCCountLwW();

    dprintf("lw: Checking for pending mem requests \n");
    for (i=0; i<nFbp; i++)
    {
        dprintf("lw: \n");
        dprintf("lw: Partition %d\n", i);
        for (j=0; j<numLTCSlices; j++)
        {
            dprintf("lw: \n");
            dprintf("lw:    Slice %d\n", j);
            data32 = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_TSTG_STATUS + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j);
            dprintf("lw:         _LTC%d_LTS%d_TSTG_STATUS: 0x%x\n", i, j, data32);

            if (!(data32 & DRF_SHIFTMASK(LW_PLTCG_LTC0_LTS0_TSTG_STATUS_CACHELINES_PINNED)))
            {
                dprintf("lw:         _LTS0_TSTG_STATUS_CACHELINES_PINNED 0\n");
            }

            data32 = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_IQ_CFG_0 + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j);
            data32 &= ~DRF_SHIFTMASK(LW_PLTCG_LTC0_LTS0_IQ_CFG_0_STATUS_SELECT);
            data32 |= DRF_NUM(_PLTCG_LTC0, _LTS0_IQ_CFG_0, _STATUS_SELECT, 3);
            GPU_REG_WR32(LW_PLTCG_LTC0_LTS0_IQ_CFG_0 + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j, data32);

            data32 = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_IQ_CFG_0 + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j);
            dprintf("lw: Written 3 to _STATUS_SELECT\n");
            dprintf("lw: LW_PLTCG_LTC%d_LTS%d_IQ_CFG_0 : 0x%x\n", i, j, data32);

            val = GPU_REG_RD32(LW_PLTCG_LTC0_LTS0_TSTG_SRV_STATUS_0 + LW_LTC_PRI_STRIDE*i + LW_LTS_PRI_STRIDE*j);
            dprintf("lw: LW_PLTCG_LTC%d_LTS%d_TSTG_SRV_STATUS_0 : 0x%x\n", i, j, val);
            val = DRF_VAL(_PLTCG_LTC0_LTS0, _TSTG_SRV_STATUS_0, _STATE, val);

            //if bit is 0 ; we have pending access
            statusVid = (val & BIT(VIDMEMBIT))? access[1]: access[0];  //[1] for NO, [0] for YES
            statusSys = (val & BIT(SYSMEMBIT))? access[1]: access[0];
            dprintf("lw:         VID: %s\n", statusVid);
            dprintf("lw:         SYS: %s\n", statusSys);
        }
    }
    return LW_OK;
}


void fbL2BypassEnable_GM200(BOOL bEnable)
{
    LwU32 regVal = GPU_REG_RD32(LW_PLTCG_LTCS_LTSS_TSTG_SET_MGMT_2);

    if(bEnable)
    {
        regVal = FLD_SET_DRF(_PLTCG, _LTCS_LTSS_TSTG_SET_MGMT_2, _L2_BYPASS_MODE, _ENABLED, regVal);

        GPU_REG_WR32(LW_PLTCG_LTCS_LTSS_TSTG_SET_MGMT_2, regVal);
    }
    else
    {
        regVal = FLD_SET_DRF(_PLTCG, _LTCS_LTSS_TSTG_SET_MGMT_2, _L2_BYPASS_MODE, _DISABLED, regVal);

        GPU_REG_WR32(LW_PLTCG_LTCS_LTSS_TSTG_SET_MGMT_2, regVal);
    }
}

/*!
 * @brief Gets the active LTC count.
 *
 * NOTE: If MODS is built with INCLUDE_LWWATCH=true, MODS may load the lwwatch
 * library before librm in which case, RM will jump to the wrong function by
 * accident if LwWatch and RM have the exact same function names. Thus, suffix
 * this function with LwW to avoid such name conflicts.
 *
 * @return  The active LTC count.
 */
LwU32
fbGetActiveLTCCountLwW_GM200()
{
    //
    // Starting with GM20X, there may not be a 1:1 correspondence of LTCs and
    // FBPs. Thus, use the new ROP_L2 (LTC) enumeration register to determine
    // the number of LTCs (as opposed to the FBP enumeration register).
    //
    return GPU_REG_RD_DRF(_PPRIV_MASTER,
                          _RING_ENUMERATE_RESULTS_ROP_L2,
                          _COUNT);
}

LwU32
fbGetActiveLtcMaskforFbp_GM200(LwU32 fbpIdx)
{
    LwU32 regVal;
    LwU32 numLTCPerFBP = GPU_REG_RD_DRF(_PTOP, _SCAL_NUM_LTC_PER_FBP, _VALUE);

    regVal = GPU_REG_IDX_RD_DRF (_FUSE, _STATUS_OPT_ROP_L2_FBP, fbpIdx, _DATA);
    return ~regVal & (LWBIT32(numLTCPerFBP) - 1);
}

/*!
 * @brief Gets the LTS per LTC count.
 *
 * NOTE: If MODS is built with INCLUDE_LWWATCH=true, MODS may load the lwwatch
 * library before librm in which case, RM will jump to the wrong function by
 * accident if LwWatch and RM have the exact same function names. Thus, suffix
 * this function with LwW to avoid such name conflicts.
 *
 * @return  The LTS per LTC count.
 */
LwU32
fbGetLTSPerLTCCountLwW_GM200()
{
    return GPU_REG_RD_DRF(_PLTCG, _LTC0_LTS0_CBC_PARAM, _SLICES_PER_LTC);
}

/*!
 * @brief Gets the lines per LTS count.
 *
 * NOTE: If MODS is built with INCLUDE_LWWATCH=true, MODS may load the lwwatch
 * library before librm in which case, RM will jump to the wrong function by
 * accident if LwWatch and RM have the exact same function names. Thus, suffix
 * this function with LwW to avoid such name conflicts.
 *
 * @return  The lines per LTS count.
 */
LwU32
fbGetLinesPerLTSCountLwW_GM200()
{
    return LW_SCAL_LITTER_NUM_LTC_LTS_SETS * LW_SCAL_LITTER_NUM_LTC_LTS_WAYS;
}
