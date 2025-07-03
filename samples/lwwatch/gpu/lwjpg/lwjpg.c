
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// LWJPG routines 
// 
//*****************************************************

#include "hal.h"
#include "g_lwjpg_hal.h"
#include "lwjpg.h"
#include "exts.h"
#include "chip.h"
#include "ampere/ga100/dev_pri_ringstation_sys.h"

dbg_lwjpg_v02_00 *pLwjpgMethodTable;
dbg_lwjpg_v02_00 *pLwjpgFuseReg = { 0 };
dbg_lwjpg_v02_00 *pLwjpgPrivReg[LWWATCH_MAX_LWJPG] = { 0 };

//-----------------------------------------------------
// lwjpgIsSupported - Determines if LWJPG is supported
//-----------------------------------------------------
BOOL lwjpgIsSupported(LwU32 indexGpu, LwU32 engineId)
{
    // Check first if all video engines have priv access.
    if (pLwjpg[indexGpu].lwjpgIsPrivBlocked(indexGpu, engineId))
    {
        dprintf("\n");
        dprintf("====================\n");
        dprintf("lw: LWJPG %d priv access blocked on GPU %d. Cannot read registers.\n", engineId, indexGpu);
        dprintf("lw: LWJPG %d not supported on GPU %d.\n", engineId, indexGpu);
        dprintf("====================\n");
        return FALSE;
    }

    if (!pLwjpg[indexGpu].lwjpgIsSupported(indexGpu, engineId))
    {
        dprintf("lw: LWJPG %d not supported on GPU %d.\n", engineId, indexGpu);
        return FALSE;
    }
    else
    {
        dprintf("lw: LWJPG %d supported on GPU %d.\n", engineId, indexGpu);
        return TRUE;
    }
}

//-----------------------------------------------------
// lwjpgDumpPriv - Dumps LWJPG priv reg space
//-----------------------------------------------------
LW_STATUS lwjpgDumpPriv(LwU32 indexGpu, LwU32 engineId)
{
    if (!lwjpgIsSupported(indexGpu, engineId))
    {
        return LW_ERR_NOT_SUPPORTED;
    }

    return pLwjpg[indexGpu].lwjpgDumpPriv(indexGpu, engineId);
}

//-----------------------------------------------------
// lwjpgDumpFuse - Dumps LWJPG related fuse registers
//-----------------------------------------------------
LW_STATUS lwjpgDumpFuse(LwU32 indexGpu, LwU32 engineId)
{
    if (!lwjpgIsSupported(indexGpu, engineId))
    {
        return LW_ERR_NOT_SUPPORTED;
    }

    return pLwjpg[indexGpu].lwjpgDumpFuse(indexGpu, engineId);
}

//-----------------------------------------------------
// lwjpgDumpImem - Dumps LWJPG instruction memory
//-----------------------------------------------------
LW_STATUS lwjpgDumpImem(LwU32 indexGpu, LwU32 engineId, LwU32 imemSize)
{
    if (!lwjpgIsSupported(indexGpu, engineId))
    {
        return LW_ERR_NOT_SUPPORTED;
    }

    return pLwjpg[indexGpu].lwjpgDumpImem(indexGpu, engineId, imemSize);
}

//-----------------------------------------------------
// lwjpgDumpDmem - Dumps LWJPG data memory
//-----------------------------------------------------
LW_STATUS lwjpgDumpDmem(LwU32 indexGpu, LwU32 engineId, LwU32 dmemSize)
{
    if (!lwjpgIsSupported(indexGpu, engineId))
    {
        return LW_ERR_NOT_SUPPORTED;
    }

    return pLwjpg[indexGpu].lwjpgDumpDmem(indexGpu, engineId, dmemSize);
}

//-----------------------------------------------------
// lwjpgTestState - Test basic LWJPG state
//-----------------------------------------------------
LW_STATUS lwjpgTestState(LwU32 indexGpu, LwU32 engineId)
{
    if (!lwjpgIsSupported(indexGpu, engineId))
    {
        return LW_ERR_NOT_SUPPORTED;
    }

    return pLwjpg[indexGpu].lwjpgTestState(indexGpu, engineId);
}

//-----------------------------------------------------
// lwjpgDisplayHwcfg - Display LWJPG HW config state
//-----------------------------------------------------
LW_STATUS lwjpgDisplayHwcfg(LwU32 indexGpu, LwU32 engineId)
{
    if (!lwjpgIsSupported(indexGpu, engineId))
    {
        return LW_ERR_NOT_SUPPORTED;
    }

    return pLwjpg[indexGpu].lwjpgDisplayHwcfg(indexGpu, engineId);
}

//-----------------------------------------------------
// lwjpgDisplayHelp - Display related help info
//-----------------------------------------------------
void lwjpgDisplayHelp(void)
{
    dprintf("LWJPG commands:\n");
    dprintf(" lwjpg \"-help\"                   - Displays the LWJPG related help menu\n");
    dprintf(" lwjpg \"-supported <engineId>\"              - Determines if LWJPG is supported on available GPUs, engineId=0 to 7 on GH100_or_later\n");
    dprintf(" lwjpg \"-hwcfg <engineId>\"                  - Display hardware config info for LWJPG\n");
    dprintf(" lwjpg \"-priv <engineId>\"                   - Dumps LWJPG priv registers\n");
    dprintf(" lwjpg \"-fuse <engineId>\"                   - Dumps LWJPG fuse related registers\n");
    dprintf(" lwjpg \"-imem  <imemsize> <engineId>\"       - Dumps LWJPG instruction memory, need to  specify imemsize\n");
    dprintf(" lwjpg \"-dmem  <dmemsize> <engineId>\"       - Dumps LWJPG data memory , need to specify dmemsize \n");
    dprintf(" lwjpg \"-state <engineId>\"                  - Checks the current state of LWJPG\n");
    dprintf(" lwjpg \"-spr <engineId>\"                    - Dumps Flcn Special Purpose Registers like PC,CSW,SP using ICD\n");
}

//-------------------------------------------------------------------
// lwjpgDisplayHwcfg - Display LWJPG Falcon Special Purpose Registers
//--------------------------------------------------------------------
LW_STATUS lwjpgDisplayFlcnSPR(LwU32 indexGpu, LwU32 engineId)
{
    if (!lwjpgIsSupported(indexGpu, engineId))
    {
        return LW_ERR_NOT_SUPPORTED;
    }

    return pLwjpg[indexGpu].lwjpgDisplayFlcnSPR(indexGpu, engineId);
}
