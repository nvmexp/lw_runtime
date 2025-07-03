
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2008-2014 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// MSENC routines 
// 
//*****************************************************

#include "hal.h"
#include "g_msenc_hal.h"
#include "msenc.h"
#include "exts.h"

dbg_msenc_v01_01 *pMsencMethodTable;
dbg_msenc_v01_01 *pMsencFuseReg = { 0 };
dbg_msenc_v01_01 *pMsencPrivReg[LWWATCH_MAX_MSENC] = { 0 };

LwU32 lwencId;
LwU32 engineId;
LwU32 cmnMethodArraySize;
LwU32 appMethodArraySize;

//-----------------------------------------------------
// msencIsSupported - Determines if MSENC is supported
//-----------------------------------------------------
BOOL msencIsSupported(LwU32 indexGpu)
{
    // Check first if all video engines have priv access.
    if (pMsenc[indexGpu].msencIsPrivBlocked(indexGpu))
    {
        dprintf("\n");
        dprintf("====================\n");
        dprintf("lw: MSENC%d priv access blocked on GPU%d. Cannot read registers.\n", lwencId, indexGpu);
        dprintf("lw: MSENC%d not supported on GPU%d.\n", lwencId, indexGpu);
        dprintf("====================\n");
        return FALSE;
    }

    if (!pMsenc[indexGpu].msencIsSupported(indexGpu))
    {
        dprintf("lw: MSENC%d not supported on GPU%d.\n", lwencId, indexGpu);
        return FALSE;
    }
    else
    {
        dprintf("lw: MSENC%d supported on GPU%d.\n", lwencId, indexGpu);
        return TRUE;
    }
}

//-----------------------------------------------------
// msencDumpPriv - Dumps MSENC priv reg space
//-----------------------------------------------------
LW_STATUS msencDumpPriv(LwU32 indexGpu)
{
    if (!msencIsSupported(indexGpu))
    {
        return LW_ERR_NOT_SUPPORTED;
    }
    return pMsenc[indexGpu].msencDumpPriv(indexGpu);
}

// msencDumpFuse - Dumps MSENC related fuse registers
//-----------------------------------------------------
LW_STATUS msencDumpFuse(LwU32 indexGpu)
{
    if (!msencIsSupported(indexGpu))
    {
        return LW_ERR_NOT_SUPPORTED;
    }

    return pMsenc[indexGpu].msencDumpFuse(indexGpu);
}

//-----------------------------------------------------
// msencDumpImem - Dumps MSENC instruction memory
//-----------------------------------------------------
LW_STATUS msencDumpImem(LwU32 indexGpu, LwU32 imemSize)
{
    if (!msencIsSupported(indexGpu))
    {
        return LW_ERR_NOT_SUPPORTED;
    }
    return pMsenc[indexGpu].msencDumpImem(indexGpu, imemSize);
}

//-----------------------------------------------------
// msencDumpDmem - Dumps MSENC data memory
//-----------------------------------------------------
LW_STATUS msencDumpDmem(LwU32 indexGpu, LwU32 dmemSize)
{
    if (!msencIsSupported(indexGpu))
    {
        return LW_ERR_NOT_SUPPORTED;
    }
    return pMsenc[indexGpu].msencDumpDmem(indexGpu, dmemSize);
}

//-----------------------------------------------------
// msencTestState - Test basic MSENC state
//-----------------------------------------------------
LW_STATUS msencTestState(LwU32 indexGpu)
{
    if (!msencIsSupported(indexGpu))
    {
        return LW_ERR_NOT_SUPPORTED;
    }
    return pMsenc[indexGpu].msencTestState(indexGpu);
}

//-----------------------------------------------------
// msencDisplayHwcfg - Display MSENC HW config state
//-----------------------------------------------------
LW_STATUS msencDisplayHwcfg(LwU32 indexGpu)
{
    if (!msencIsSupported(indexGpu))
    {
        return LW_ERR_NOT_SUPPORTED;
    }
    return pMsenc[indexGpu].msencDisplayHwcfg(indexGpu);
}

//-----------------------------------------------------
// msencDisplayHelp - Display related help info
//-----------------------------------------------------
void msencDisplayHelp(void)
{
    dprintf("MSENC commands:\n");
    dprintf(" msenc \"-help\"                       - Displays the MSENC related help menu\n");
    dprintf(" msenc \"-supported <msencId>\"        - Determines if MSENC is supported on available GPUs, need to specify MSENCID either 0 or 1(GM20X_and_later) or 2(AD102)\n");
    dprintf(" msenc \"-hwcfg <msencId>\"            - Displays hardware config info for MSENC, need to specify MSENCID either 0 or 1(GM20X_and_later) or 2(AD102)\n");
    dprintf(" msenc \"-priv <msencId>\"             - Dumps MSENC priv registers, need to specify MSENCID either 0 or 1(GM20X_and_later) or 2(AD102)\n");
    dprintf(" msenc \"-fuse\"                       - Dumps MSENC related fuse registers\n");
    dprintf(" msenc \"-imem <imemsize> <msencId>\"  - Dumps MSENC instruction memory, need to specify imemsize and MSENCID either 0 or 1(GM20X_and_later) or 2(AD102)\n");
    dprintf(" msenc \"-dmem <dmemsize> <msencId>\"  - Dumps MSENC data memory , need to specify dmemsize and MSENCID either 0 or 1(GM20X_and_later) or 2(AD102)\n");
    dprintf(" msenc \"-state <msencId>\"            - Checks the current state of MSENC, need to specify MSENCID either 0 or 1(GM20X_and_later) or 2(AD102)\n");
    dprintf(" msenc \"-spr <msencId>\"              - Dumps Flcn Special Purpose Registers like PC,CSW,SP using ICD, need to specify MSENCID either 0 or 1(GM20X_and_later) or 2(AD102)\n");
}

//-------------------------------------------------------------------
// msencDisplayHwcfg - Display MSENC Falcon Special Purpose Registers
//--------------------------------------------------------------------
LW_STATUS msencDisplayFlcnSPR(LwU32 indexGpu)
{
    if (!msencIsSupported(indexGpu))
    {
        return LW_ERR_NOT_SUPPORTED;
    }
    return pMsenc[indexGpu].msencDisplayFlcnSPR(indexGpu);
}
