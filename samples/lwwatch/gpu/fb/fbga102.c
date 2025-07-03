/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch debug extension
// fbga102.c
//
//*****************************************************

//
// includes
//
#include "fb.h"
#include "g_fb_private.h"
#include "ampere/ga102/dev_fuse.h"
#include "ampere/ga102/dev_ltc.h"
#include "ampere/ga102/dev_pri_ringmaster.h"
#include "ampere/ga102/dev_top.h"
#include "ampere/ga102/hwproject.h"

#include "hal.h"

LwU32
fbGetActiveLTCCountLwW_GA102()
{
    LwU32 numActiveLTCs = GPU_REG_RD_DRF(_PPRIV, _MASTER_RING_ENUMERATE_RESULTS_L2, _COUNT);
    return numActiveLTCs;
}


LwU32
fbGetActiveLtsMaskForLTC_GA102(LwU32 ltcIdx)
{
    LwU32 ltsDisableMask;
    ltsDisableMask = GPU_REG_RD32(LW_PLTCG_LTC0_MISC_LTC_CFG + (ltcIdx * LW_LTC_PRI_STRIDE));
    ltsDisableMask = DRF_VAL(_PLTCG, _LTC0_MISC_LTC_CFG, _FS_SLICE_DISABLE, ltsDisableMask);
    return ~ltsDisableMask & (LWBIT32(pFb[indexGpu].fbGetLTSPerLTCCountLwW()) - 1);
}

LwU32
fbGetActiveLtcMaskforFbp_GA102(LwU32 fbpIdx)
{
    LwU32 regVal;
    LwU32 numLTCPerFBP = GPU_REG_RD_DRF(_PTOP, _SCAL_NUM_LTC_PER_FBP, _VALUE);
    LW_ASSERT(numLTCPerFBP > 0);
    regVal = GPU_REG_IDX_RD_DRF(_FUSE, _STATUS_OPT_LTC_FBP, fbpIdx, _DATA);
    return ~regVal & (LWBIT32(numLTCPerFBP) - 1) ;
}