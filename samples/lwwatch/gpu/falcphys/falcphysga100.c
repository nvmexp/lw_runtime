/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "falcphys.h"
#include "os.h"
#include "rmlsfm.h"
#include "lwtypes.h"
#include "acr.h"

#include "g_falcphys_private.h"          // (rmconfig) hal/obj setup
#include "g_acr_hal.h"

#include "ampere/ga100/dev_fuse.h"
#include "ampere/ga100/dev_master.h"


#define LW_FUSE_STATUS_OPT_LWDEC_DATA_ENABLE      0x00000000 /* R---V */
#define LW_PMC_ENABLE_PGRAPH                                  12:12 /*       */
#define LW_PMC_ENABLE_PGRAPH_DISABLED                    0x00000000 /*       */
#define LW_PMC_ENABLE_PGRAPH_ENABLED                     0x00000001 /*       */

//-----------------------------------------------------
// falcphysIsSupported_GA100
//-----------------------------------------------------
BOOL falcphysIsSupported_GA100( LwU32 indexGpu )
{
    LwU32 reg;

    // Skip IsSupported check in case of fmodel
    if (pAcr[indexGpu].acrIsFmodel())
    {
        return TRUE;
    }

    // Check PRIV_SEC fuse, if fuse is not blown then PRIV sec feature in not present
    reg = GPU_REG_RD32(LW_FUSE_OPT_PRIV_SEC_EN);
    if (DRF_VAL(_FUSE, _OPT_PRIV_SEC_EN, _DATA, reg) == LW_FUSE_OPT_PRIV_SEC_EN_DATA_NO)
    {
        dprintf("Need PRIV_SEC fuse blown for priv security\n");
        return FALSE;
    }

    // Check WPR fuse, if fuse is not blown, then wpr feature is not present
    reg = GPU_REG_RD32(LW_FUSE_OPT_WPR_ENABLED);
    if (DRF_VAL(_FUSE, _OPT_WPR_ENABLED_, DATA, reg) == LW_FUSE_OPT_WPR_ENABLED_DATA_NO)
    {
        dprintf("Need WPR fuse blown for wpr feature\n");
        return FALSE;
    }

    return TRUE;
}

LwBool falcphysCheckEngineIsPresent_GA100(LwU32 falconId)
{
    switch (falconId)
    {
        case FECS_FALCON_ID:
            if (GPU_REG_RD_DRF_IDX( _PMC, _ENABLE, _DEVICE, DRF_BASE(LW_PMC_ENABLE_PGRAPH)) == LW_PMC_ENABLE_DEVICE_ENABLE)
            {
                return LW_TRUE;
            }
            break;
        case PMU_FALCON_ID:
            return LW_TRUE;
            break;
        case SEC_FALCON_ID:
            if (DRF_VAL( _FUSE, _STATUS_OPT_SEC, _DATA, GPU_REG_RD32(LW_FUSE_STATUS_OPT_SEC)) == LW_FUSE_STATUS_OPT_SEC_DATA_ENABLE)
            {
                return LW_TRUE;
            }
            break;
        case DPU_FALCON_ID:
        case HDA_FALCON_ID:
            if (DRF_VAL( _FUSE, _STATUS_OPT_DISPLAY, _DATA, GPU_REG_RD32(LW_FUSE_STATUS_OPT_DISPLAY)) == LW_FUSE_STATUS_OPT_DISPLAY_DATA_ENABLE)
            {
                return LW_TRUE;
            }
            break;
        case LWDEC_FALCON_ID:
            if (DRF_VAL( _FUSE, _STATUS_OPT_LWDEC, _DATA, GPU_REG_RD32(LW_FUSE_STATUS_OPT_LWDEC)) == LW_FUSE_STATUS_OPT_LWDEC_DATA_ENABLE)
            {
                return LW_TRUE;
            }
            break;
        case LWENC0_FALCON_ID:
            if (DRF_IDX_VAL( _FUSE, _STATUS_OPT_LWENC, _IDX, 0, GPU_REG_RD32(LW_FUSE_STATUS_OPT_LWENC)) == LW_FUSE_STATUS_OPT_LWENC_IDX_ENABLE)
            {
                return LW_TRUE;
            }
            break;
        case LWENC1_FALCON_ID:
            if (DRF_IDX_VAL( _FUSE, _STATUS_OPT_LWENC, _IDX, 1, GPU_REG_RD32(LW_FUSE_STATUS_OPT_LWENC)) == LW_FUSE_STATUS_OPT_LWENC_IDX_ENABLE)
            {
                return LW_TRUE;
            }
            break;
        case LWENC2_FALCON_ID:
            if (DRF_IDX_VAL( _FUSE, _STATUS_OPT_LWENC, _IDX, 2, GPU_REG_RD32(LW_FUSE_STATUS_OPT_LWENC)) == LW_FUSE_STATUS_OPT_LWENC_IDX_ENABLE)
            {
                return LW_TRUE;
            }
            break;
        default:
            dprintf(" Invalid falconId ");
    }
    return LW_FALSE;
}
