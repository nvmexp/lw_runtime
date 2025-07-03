/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All information
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

#include "volta/gv100/dev_pwr_pri.h"
#include "volta/gv100/dev_master.h"
#include "volta/gv100/dev_falcon_v1.h"
#include "volta/gv100/dev_fbfalcon_pri.h"
#include "volta/gv100/dev_minion.h"
#include "volta/gv100/dev_gsp.h"
#include "volta/gv100/dev_lwenc_pri_sw.h"
#include "volta/gv100/dev_lwdec_pri.h"
#include "volta/gv100/dev_fb.h"
#include "volta/gv100/dev_falcon_v4.h"
#include "volta/gv100/dev_graphics_nobundle.h"
#include "volta/gv100/dev_sec_pri.h"
#include "dpu/v02_05/dev_disp_falcon.h"


void acrGetFalconProp_GV100(LSFALCPROP *pFalc, LwU32  falconId, LwU32 indexGpu)
{
    switch(falconId)
    {
        case LSF_FALCON_ID_PMU:
                pFalc[falconId].name            = "PMU   ";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = LW_FALCON_PWR_BASE;
                pFalc[falconId].bFalconEnabled  = TRUE;
                break;
        case LSF_FALCON_ID_DPU:
                pFalc[falconId].name            = "GSPLITE";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = DEVICE_BASE(LW_PGSP);
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_PDISP), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));
                break;
        case LSF_FALCON_ID_FECS:
                pFalc[falconId].name            = "FECS  ";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = LW_PGRAPH_PRI_FECS_FALCON_IRQSSET;
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_PGRAPH), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));

                break;
        case LSF_FALCON_ID_GPCCS:
                pFalc[falconId].name            = "GPCCS ";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = LW_PGRAPH_PRI_GPC0_GPCCS_FALCON_IRQSSET;
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_PGRAPH), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));
                break;
        case LSF_FALCON_ID_LWDEC:
                pFalc[falconId].name            = "LWDEC ";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = LW_FALCON_LWDEC_BASE;
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_LWDEC), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));
                break;
        case LSF_FALCON_ID_LWENC0:
                pFalc[falconId].name            = "LWENC0";
                pFalc[falconId].regBase         = LW_FALCON_LWENC0_BASE;
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_LWENC0), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));
                break;
        case LSF_FALCON_ID_LWENC1:
                pFalc[falconId].name            = "LWENC1";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = LW_FALCON_LWENC1_BASE;
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_LWENC1), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));
                break;
        case LSF_FALCON_ID_SEC2:
                pFalc[falconId].name            = "SEC   ";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = LW_PSEC_FALCON_IRQSSET;
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_SEC), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));
                break;
        case LSF_FALCON_ID_LWENC2:
                pFalc[falconId].name            = "LWENC2";
                pAcr[indexGpu].acrGetLwenc2FalconProp(pFalc);
                break;
        case LSF_FALCON_ID_MINION:
                pFalc[falconId].name            = "MINION";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = DEVICE_BASE(LW_PMINION_FALCON);
                pFalc[falconId].bFalconEnabled  = FLD_IDX_TEST_DRF(_PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_LWLINK), _ENABLE,
                                                                   GPU_REG_RD32(LW_PMC_ENABLE));
                break;
        case LSF_FALCON_ID_FBFALCON:
                pFalc[falconId].name            = "FBFALCON";
                pFalc[falconId].available       = LW_TRUE;
                pFalc[falconId].regBase         = DEVICE_BASE(LW_PFBFALCON);
                pFalc[falconId].bFalconEnabled  = LW_TRUE;
                break;
        default:
                break;
    }
}
