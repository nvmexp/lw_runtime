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
// msenc0802.c - LWENC routines
//
//-----------------------------------------------------

#include "ada/ad102/dev_lwenc_pri_sw.h"
#include "ada/ad102/dev_falcon_v4.h"
#include "ada/ad102/dev_pri_ringstation_sys.h"
#include "ada/ad102/dev_fifo.h"
#include "ada/ad102/dev_master.h"
#include "class/clc9b7.h"

#include "msenc.h"
#include "hwref/lwutil.h"
#include "g_msenc_private.h"     // (rmconfig)  implementation prototypes

#define USE_LWENC_8_2

#if defined(USE_LWENC_8_2)
#include "msenc0802.h"
#endif

//-----------------------------------------------------
// msencIsValidEngineId_v08_02
//-----------------------------------------------------
BOOL msencIsValidEngineId_v08_02(LwU32 indexGpu)
{
    switch (lwencId)
    {
    case LWWATCH_MSENC_0:
    case LWWATCH_MSENC_1:
    case LWWATCH_MSENC_2:
        break;
    default:
        dprintf("AD10x supports up to 3 msenc instances only\n");
        return FALSE;
    }

    return TRUE;
}

//-----------------------------------------------------
// msencIsSupported_v08_02
//-----------------------------------------------------
BOOL msencIsSupported_v08_02( LwU32 indexGpu )
{
    if (!pMsenc[indexGpu].msencIsValidEngineId(indexGpu))
        return FALSE;

    switch (lwencId)
    {
    case LWWATCH_MSENC_0:
        pMsencPrivReg[LWWATCH_MSENC_0] = msencPrivReg_v08_02_eng0;
        break;
    case LWWATCH_MSENC_1:
        pMsencPrivReg[LWWATCH_MSENC_1] = msencPrivReg_v08_02_eng1;
        break;
    case LWWATCH_MSENC_2:
        pMsencPrivReg[LWWATCH_MSENC_2] = msencPrivReg_v08_02_eng2;
        break;
    default:
        dprintf("AD10x has 3 msenc instances (0-2).\n");
        return FALSE;
        break;
    }

    pMsencFuseReg = msencFuseReg_v08_02;
    pMsencMethodTable = msencMethodTable_v08_02;
    cmnMethodArraySize = CMNMETHODARRAYSIZEC9B7;
    appMethodArraySize = APPMETHODARRAYSIZEC9B7;

    engineId = lwencId;

    return TRUE;
}

//-----------------------------------------------------
// msencIsPrivBlocked_v08_02//-----------------------------------------------------
BOOL msencIsPrivBlocked_v08_02(LwU32 indexGpu)
{
    LwU32 idx;
    LwU32 bitmask;
    LwU32 regSysPrivFsConfig;

    if (!pMsenc[indexGpu].msencIsValidEngineId(indexGpu))
        return TRUE;

    // Bit-fields within LW_PPRIV_SYS_PRIV_FS_CONFIG denote priv access for video.
    // All video engines must have priv access for lwenc command support.
    switch (lwencId)
    {
    case LWWATCH_MSENC_0:
        idx = LW_PPRIV_SYS_PRI_MASTER_sys_pri_hub2lwenc_pri0 >> 5;
        regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(idx));
        bitmask = BIT(LW_PPRIV_SYS_PRI_MASTER_sys_pri_hub2lwenc_pri0 - (idx << 5));
        break;
    case LWWATCH_MSENC_1:
        idx = LW_PPRIV_SYS_PRI_MASTER_sys_pri_hub2lwenc_pri1 >> 5;
        regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(idx));
        bitmask = BIT(LW_PPRIV_SYS_PRI_MASTER_sys_pri_hub2lwenc_pri1 - (idx << 5));
        break;
    case LWWATCH_MSENC_2:
        idx = LW_PPRIV_SYS_PRI_MASTER_sys_pri_hub2lwenc_pri2 >> 5;
        regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(idx));
        bitmask = BIT(LW_PPRIV_SYS_PRI_MASTER_sys_pri_hub2lwenc_pri2 - (idx << 5));
        break;
    default:
        return TRUE;
    }

    return ((regSysPrivFsConfig & bitmask) != bitmask);
}

//-----------------------------------------------------
// msencGetClassId_v08_02 - Returns Class ID supported
//                          for IP 08.2
//-----------------------------------------------------
LwU32
msencGetClassId_v08_02 (void)
{
    return LWC9B7_VIDEO_ENCODER;
}

//----------------------------------------------------------
// msencDumpFuse_v08_02 - Dumps LWENC related fuse registers
//----------------------------------------------------------
LW_STATUS msencDumpFuse_v08_02(LwU32 indexGpu)
{
    LwU32 u;

    if (!pMsencFuseReg)
    {
        dprintf("lw: -- Gpu %u MSENC error: fuse reg array uninitialized\n", indexGpu);
        return LW_ERR_ILWALID_PARAMETER;
    }

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u MSENC related fuse registers -- \n", indexGpu);
    dprintf("lw:\n");

    for (u = 0; ; u++)
    {
        if (pMsencFuseReg[u].m_id == 0)
        {
            break;
        }
        pMsenc[indexGpu].msencPrintPriv(70, pMsencFuseReg[u].m_tag,
            pMsencFuseReg[u].m_id);
    }
    return LW_OK;
}
