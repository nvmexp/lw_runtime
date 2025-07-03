
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2015 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// LWDEC routines 
// 
//*****************************************************

#include "hal.h"
#include "g_lwdec_hal.h"
#include "lwdec.h"
#include "exts.h"
#include "chip.h"
#include "maxwell/gm107/dev_pri_ringstation_sys.h"

dbg_lwdec_v01_01 *pLwdecMethodTable;
dbg_lwdec_v01_01 *pLwdecFuseReg = { 0 };
dbg_lwdec_v01_01 *pLwdecPrivReg[LWWATCH_MAX_LWDEC] = { 0 };

//-----------------------------------------------------
// lwdecIsSupported - Determines if LWDEC is supported
//-----------------------------------------------------
BOOL lwdecIsSupported(LwU32 indexGpu, LwU32 engineId)
{
    // Check first if all video engines have priv access.
    if (pLwdec[indexGpu].lwdecIsPrivBlocked(indexGpu, engineId))
    {
        dprintf("\n");
        dprintf("====================\n");
        dprintf("lw: LWDEC %d priv access blocked on GPU %d. Cannot read registers.\n", engineId, indexGpu);
        dprintf("lw: LWDEC %d not supported on GPU %d.\n", engineId, indexGpu);
        dprintf("====================\n");
        return FALSE;
    }

    /*
     * TODO: Remove this WAR once MODS move to new backport LWDEC ucode , Bug 1643686
     */
#ifndef LW_MODS
    if (IsGM107() || IsGM200() || IsGM204() || IsGM206())
    {
        if (!lwdecIsSupported_v03_00(indexGpu, engineId))
        {
            dprintf("lw: LWDEC %d not supported on GPU %d.\n", engineId, indexGpu);
            return FALSE;
        }
        else
        {
            dprintf("lw: LWDEC %d supported on GPU %d.\n", engineId, indexGpu);
            return TRUE;
        }
    }
#endif
    if (!pLwdec[indexGpu].lwdecIsSupported(indexGpu, engineId))
    {
        dprintf("lw: LWDEC %d not supported on GPU %d.\n", engineId, indexGpu);
        return FALSE;
    }
    else
    {
        dprintf("lw: LWDEC %d supported on GPU %d.\n", engineId, indexGpu);
        return TRUE;
    }
}

//-----------------------------------------------------
// lwdecDumpPriv - Dumps LWDEC priv reg space
//-----------------------------------------------------
LW_STATUS lwdecDumpPriv(LwU32 indexGpu, LwU32 engineId)
{
    if (!lwdecIsSupported(indexGpu, engineId))
    {
        return LW_ERR_NOT_SUPPORTED;
    }

    return pLwdec[indexGpu].lwdecDumpPriv(indexGpu, engineId);
}

// lwdecDumpFuse - Dumps LWDEC related fuse registers
//-----------------------------------------------------
LW_STATUS lwdecDumpFuse(LwU32 indexGpu)
{
    if (!lwdecIsSupported(indexGpu, 0))
    {
        return LW_ERR_NOT_SUPPORTED;
    }

    return pLwdec[indexGpu].lwdecDumpFuse(indexGpu);
}

//-----------------------------------------------------
// lwdecDumpImem - Dumps LWDEC instruction memory
//-----------------------------------------------------
LW_STATUS lwdecDumpImem(LwU32 indexGpu, LwU32 engineId, LwU32 imemSize)
{
    if (!lwdecIsSupported(indexGpu, engineId))
    {
        return LW_ERR_NOT_SUPPORTED;
    }

    return pLwdec[indexGpu].lwdecDumpImem(indexGpu, engineId, imemSize);
}

//-----------------------------------------------------
// lwdecDumpDmem - Dumps LWDEC data memory
//-----------------------------------------------------
LW_STATUS lwdecDumpDmem(LwU32 indexGpu, LwU32 engineId, LwU32 dmemSize)
{
    if (!lwdecIsSupported(indexGpu, engineId))
    {
        return LW_ERR_NOT_SUPPORTED;
    }

    /*
     * TODO: Remove this WAR once MODS move to new backport LWDEC ucode , Bug 1643686
     */
#ifndef LW_MODS
    if (IsGM107() || IsGM200() || IsGM204() || IsGM206())
    {
        return lwdecDumpDmem_v03_00(indexGpu, engineId, dmemSize);
    }
#endif

    return pLwdec[indexGpu].lwdecDumpDmem(indexGpu, engineId, dmemSize);
}

//-----------------------------------------------------
// lwdecTestState - Test basic LWDEC state
//-----------------------------------------------------
LW_STATUS lwdecTestState(LwU32 indexGpu, LwU32 engineId)
{
    if (!lwdecIsSupported(indexGpu, engineId))
    {
        return LW_ERR_NOT_SUPPORTED;
    }

    return pLwdec[indexGpu].lwdecTestState(indexGpu, engineId);
}

//-----------------------------------------------------
// lwdecDisplayHwcfg - Display LWDEC HW config state
//-----------------------------------------------------
LW_STATUS lwdecDisplayHwcfg(LwU32 indexGpu, LwU32 engineId)
{
    if (!lwdecIsSupported(indexGpu, engineId))
    {
        return LW_ERR_NOT_SUPPORTED;
    }

    return pLwdec[indexGpu].lwdecDisplayHwcfg(indexGpu, engineId);
}

//-----------------------------------------------------
// lwdecDisplayHelp - Display related help info
//-----------------------------------------------------
void lwdecDisplayHelp(void)
{
    dprintf("LWDEC commands:\n");
    dprintf(" lwdec \"-help\"                             - Displays the LWDEC related help menu\n");
    dprintf(" lwdec \"-supported <engineId>\"              - Determines if LWDEC is supported on available GPUs, engineId=0,1,or2 Tu10x_or_later\n");
    dprintf(" lwdec \"-hwcfg <engineId>\"                  - Display hardware config info for LWDEC, engineId=0,1,or2 Tu10x_or_later\n");
    dprintf(" lwdec \"-priv <engineId>\"                   - Dumps LWDEC priv registers, engineId=0,1,or2 Tu10x_or_later\n");
    dprintf(" lwdec \"-fuse\"                              - Dumps LWDEC related fuse registers\n");
    dprintf(" lwdec \"-imem  <imemsize> <engineId>\"       - Dumps LWDEC instruction memory, need to  specify imemsize, engineId=0,1,or2 Tu10x_or_later\n");
    dprintf(" lwdec \"-dmem  <dmemsize> <engineId>\"       - Dumps LWDEC data memory , need to specify dmemsize, engineId=0,1,or2 Tu10x_or_later\n");
    dprintf(" lwdec \"-state <engineId>\"                  - Checks the current state of LWDEC, engineId=0,1,or2 Tu10x_or_later\n");
    dprintf(" lwdec \"-spr <engineId>\"                    - Dumps Flcn Special Purpose Registers like PC,CSW,SP using ICD, engineId=0,1,or2 Tu10x_or_later\n");
}

//-------------------------------------------------------------------
// lwdecDisplayFlcnSPR - Display LWDEC Falcon Special Purpose Registers
//--------------------------------------------------------------------
LW_STATUS lwdecDisplayFlcnSPR(LwU32 indexGpu, LwU32 engineId)
{
    if (!lwdecIsSupported(indexGpu, engineId))
    {
        return LW_ERR_NOT_SUPPORTED;
    }

    return pLwdec[indexGpu].lwdecDisplayFlcnSPR(indexGpu, engineId);
}
