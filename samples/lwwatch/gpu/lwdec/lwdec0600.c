/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2021 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

//-----------------------------------------------------
//
// lwdec0600.c - LWDEC 6.0 routines
//
//-----------------------------------------------------

#include "lwdec.h"
#include "chip.h"
#include "g_lwdec_private.h"     // (rmconfig)  implementation prototypes
#include "class/clc9b0.h"
#include "lwdec0600.h"

#include "ada/ad102/dev_pri_ringstation_sys.h"

//-----------------------------------------------------
// lwdecIsValidEngineId_v06_00
//-----------------------------------------------------
BOOL lwdecIsValidEngineId_v06_00(LwU32 indexGpu, LwU32 engineId)
{
    switch(engineId)
    {
        case LWWATCH_LWDEC_0:
        case LWWATCH_LWDEC_1:
        case LWWATCH_LWDEC_2:
        case LWWATCH_LWDEC_3:
             break;
        default:
             dprintf("ad102 supports upto 4 lwdec instances only\n");
             return FALSE;
    }

    return TRUE;
}

//-----------------------------------------------------
// lwdecIsSupported_v06_00
//-----------------------------------------------------
BOOL lwdecIsSupported_v06_00(LwU32 indexGpu, LwU32 engineId)
{
    if (!pLwdec[indexGpu].lwdecIsValidEngineId(indexGpu, engineId))
        return FALSE;

    switch(engineId)
    {
        case LWWATCH_LWDEC_0:
             pLwdecPrivReg[LWWATCH_LWDEC_0] = lwdecPrivReg_v06_00_eng0;
             break;
        case LWWATCH_LWDEC_1:
             pLwdecPrivReg[LWWATCH_LWDEC_1] = lwdecPrivReg_v06_00_eng1;
             break;
        case LWWATCH_LWDEC_2:
             pLwdecPrivReg[LWWATCH_LWDEC_2] = lwdecPrivReg_v06_00_eng2;
             break;
        case LWWATCH_LWDEC_3:
             pLwdecPrivReg[LWWATCH_LWDEC_3] = lwdecPrivReg_v06_00_eng3;
             break;
       default:
             return FALSE;
    }

    pLwdecMethodTable = lwdecMethodTable_v06_00;

    return TRUE;
}

//-----------------------------------------------------
// lwdecIsPrivBlocked_v06_00
//-----------------------------------------------------
BOOL lwdecIsPrivBlocked_v06_00(LwU32 indexGpu, LwU32 engineId)
{
    LwU32 idx;
    LwU32 bitmask;
    LwU32 regSysPrivFsConfig;

    if (!pLwdec[indexGpu].lwdecIsValidEngineId(indexGpu, engineId))
        return TRUE;

    // Bit-fields within LW_PPRIV_SYS_PRIV_FS_CONFIG denote priv access for video.
    // All video engines must have priv access for lwdec command support.
    switch(engineId)
    {
        case LWWATCH_LWDEC_0:
             idx = LW_PPRIV_SYS_PRI_MASTER_fecs2lwdec_pri0 >> 5;
             regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(idx));
             bitmask = BIT(LW_PPRIV_SYS_PRI_MASTER_fecs2lwdec_pri0 - (idx << 5));
             break;
        case LWWATCH_LWDEC_1:
             idx = LW_PPRIV_SYS_PRI_MASTER_fecs2lwdec_pri1 >> 5;
             regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(idx));
             bitmask = BIT(LW_PPRIV_SYS_PRI_MASTER_fecs2lwdec_pri1 - (idx << 5));
             break;
        case LWWATCH_LWDEC_2:
             idx = LW_PPRIV_SYS_PRI_MASTER_fecs2lwdec_pri2 >> 5;
             regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(idx));
             bitmask = BIT(LW_PPRIV_SYS_PRI_MASTER_fecs2lwdec_pri2 - (idx << 5));
             break;
        case LWWATCH_LWDEC_3:
             idx = LW_PPRIV_SYS_PRI_MASTER_fecs2lwdec_pri3 >> 5;
             regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(idx));
             bitmask = BIT(LW_PPRIV_SYS_PRI_MASTER_fecs2lwdec_pri3 - (idx << 5));
             break;

        default:
             return TRUE;
    }

    return ((regSysPrivFsConfig & bitmask) != bitmask);
}

//-----------------------------------------------------
// lwdecGetClassId_v06_00
//-----------------------------------------------------
LwU32
lwdecGetClassId_v06_00 (void)
{
    return LWC9B0_VIDEO_DECODER;
}

