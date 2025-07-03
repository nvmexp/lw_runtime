
/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2008-2016 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// VIC routines 
// 
//*****************************************************

#include "lwwatch.h"
#include "hal.h"
#include "g_vic_hal.h"
#include "vic.h"
#include "exts.h"


static void vicDisplayPowerGatedMessage(void);

dbg_vic *pVicMethodTable;
dbg_vic *pVicPrivReg;


//-----------------------------------------------------
// vicIsSupported - Determines if VIC is supported
//-----------------------------------------------------
BOOL vicIsSupported(LwU32 indexGpu)
{
    if (!pVic[indexGpu].vicIsSupported(indexGpu))
    {
        dprintf("lw: VIC not supported on GPU %d.\n", indexGpu);
        return FALSE;
    }
    else
    {
        dprintf("lw: VIC supported on GPU %d.\n", indexGpu);
        return TRUE;
    }
}

//-----------------------------------------------------
// vicDumpPriv - Dumps VIC priv reg space
//-----------------------------------------------------
LW_STATUS vicDumpPriv(LwU32 indexGpu)
{
    if (!pVic[indexGpu].vicIsSupported(indexGpu))
    {
        dprintf("lw: vicDumpPriv: VIC not supported on GPU %d.\n", indexGpu);
        return LW_ERR_NOT_SUPPORTED;
    }
    
    if (pVic[indexGpu].vicIsPrivBlocked(indexGpu))
    {
        vicDisplayPowerGatedMessage();
        return LW_OK;
    }
    
    return pVic[indexGpu].vicDumpPriv(indexGpu);
}

//-----------------------------------------------------
// vicDumpImem - Dumps VIC instruction memory
//-----------------------------------------------------
LW_STATUS vicDumpImem(LwU32 indexGpu)
{
    if (!pVic[indexGpu].vicIsSupported(indexGpu))
    {
        dprintf("lw: vicDumpImem: VIC not supported on GPU %d.\n", indexGpu);
        return LW_ERR_NOT_SUPPORTED;
    }
    
    if (pVic[indexGpu].vicIsPrivBlocked(indexGpu))
    {
        vicDisplayPowerGatedMessage();
        return LW_OK;
    }

    return pVic[indexGpu].vicDumpImem(indexGpu);
}

//-----------------------------------------------------
// vicDumpDmem - Dumps VIC data memory
//-----------------------------------------------------
LW_STATUS vicDumpDmem(LwU32 indexGpu)
{
    if (!pVic[indexGpu].vicIsSupported(indexGpu))
    {
        dprintf("lw: vicDumpDmem: VIC not supported on GPU %d.\n", indexGpu);
        return LW_ERR_NOT_SUPPORTED;
    }
    
    if (pVic[indexGpu].vicIsPrivBlocked(indexGpu))
    {
        vicDisplayPowerGatedMessage();
        return LW_OK;
    }

    return pVic[indexGpu].vicDumpDmem(indexGpu);
}

//-----------------------------------------------------
// vicTestState - Test basic VIC state
//-----------------------------------------------------
LW_STATUS vicTestState(LwU32 indexGpu)
{
    /* In case of error, R/W of VIC, not supported */
    if (!IsAndroid())
        return LW_ERR_NOT_SUPPORTED;

    if (!pVic[indexGpu].vicIsSupported(indexGpu))
    {
        dprintf("lw: vicTestState: VIC not supported on GPU %d.\n", indexGpu);
        return LW_ERR_NOT_SUPPORTED;
    }
    
    if (pVic[indexGpu].vicIsPrivBlocked(indexGpu))
    {
        vicDisplayPowerGatedMessage();
        return LW_OK;
    }

    return pVic[indexGpu].vicTestState(indexGpu);
}

//-----------------------------------------------------
// vicDisplayHwcfg - Display VIC HW config state
//-----------------------------------------------------
LW_STATUS vicDisplayHwcfg(LwU32 indexGpu)
{
    if (!pVic[indexGpu].vicIsSupported(indexGpu))
    {
        dprintf("lw: vicDisplayHwcfg: VIC not supported on GPU %d.\n", indexGpu);
        return LW_ERR_NOT_SUPPORTED;
    }
    
    if (pVic[indexGpu].vicIsPrivBlocked(indexGpu))
    {
        vicDisplayPowerGatedMessage();
        return LW_OK;
    }

    return pVic[indexGpu].vicDisplayHwcfg(indexGpu);
}

//-----------------------------------------------------
// vicDisplayHelp - Display related help info
//-----------------------------------------------------
void vicDisplayHelp(void)
{
    if (LWWATCHCFG_IS_PLATFORM(UNIX) || LWWATCHCFG_FEATURE_ENABLED(MODS_UNIX))
    {
        dprintf("VIC commands:\n");
        dprintf(" lws vic \"-help\"                   - Displays the VIC related help menu\n");
        dprintf(" lws vic \"-supported\"              - Determines if VIC is supported on available GPUs\n");
        dprintf(" lws vic \"-hwcfg\"                  - Display hardware config info for VIC\n");
        dprintf(" lws vic \"-priv\"                   - Dumps VIC priv registers\n");
        dprintf(" lws vic \"-imem\"                   - Dumps VIC instruction memory\n");
        dprintf(" lws vic \"-dmem\"                   - Dumps VIC data memory\n");
        dprintf(" lws vic \"-state\"                  - Checks the current state of VIC\n");
    }
    else
    {
        dprintf("VIC commands:\n");
        dprintf(" vic \"-help\"                   - Displays the VIC related help menu\n");
        dprintf(" vic \"-supported\"              - Determines if VIC is supported on available GPUs\n");
        dprintf(" vic \"-hwcfg\"                  - Display hardware config info for VIC\n");
        dprintf(" vic \"-priv\"                   - Dumps VIC priv registers\n");
        dprintf(" vic \"-imem\"                   - Dumps VIC instruction memory\n");
        dprintf(" vic \"-dmem\"                   - Dumps VIC data memory\n");
        dprintf(" vic \"-state\"                  - Checks the current state of VIC\n");
    }
}

//-----------------------------------------------------
// vicDisplayPowerGatedMessage -
//-----------------------------------------------------
static void vicDisplayPowerGatedMessage(void)
{
    dprintf("lw: VIC is lwrrently power gated on GPU %d.\n", indexGpu);
    dprintf("lw: The priv interface is  blocked and its unsafe to read VIC registers.\n");
    dprintf("lw: Under this condition, VIC commands are disabled.\n");
    dprintf("lw: Disabling ELPG will disable VIC power gating.\n");
}

//-------------------------------------------------------------------
// vicDisplayFlcnSPR - Display VIC Falcon Special Purpose Registers
//--------------------------------------------------------------------
LW_STATUS vicDisplayFlcnSPR(LwU32 indexGpu)
{
    if (!pVic[indexGpu].vicIsSupported(indexGpu))
    {
        dprintf("lw: vicDisplayFlcnSPR VIC not supported on GPU %d.\n", indexGpu);
        return LW_ERR_NOT_SUPPORTED;
    }
    
    return pVic[indexGpu].vicDisplayFlcnSPR(indexGpu);
}
