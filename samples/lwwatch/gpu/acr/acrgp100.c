/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "acr.h"
#include "rmlsfm.h"
#include "g_acr_private.h"          // (rmconfig) hal/obj setup

#include "pascal/gp100/dev_falcon_v4.h"
#include "pascal/gp100/dev_master.h"

void acrGetLwenc2FalconProp_GP100(LSFALCPROP *pFalc)
{
    pFalc[LSF_FALCON_ID_LWENC2].available       = LW_TRUE;
    pFalc[LSF_FALCON_ID_LWENC2].regBase         = LW_FALCON_LWENC2_BASE;
    pFalc[LSF_FALCON_ID_LWENC2].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_LWENC2), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));
}
