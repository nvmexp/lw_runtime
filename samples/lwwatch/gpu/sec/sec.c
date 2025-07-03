
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
// SEC routines 
// 
//*****************************************************

#include "hal.h"
#include "g_cipher_hal.h"
#include "sec.h"
#include "exts.h"

dbg_sec_t114 *pSecMethodTable;
dbg_sec_t114 *pSecPrivReg;


//-----------------------------------------------------
// secIsSupported - Determines if SEC is supported
//-----------------------------------------------------
BOOL secIsSupported(LwU32 indexGpu)
{
    if (!pCipher[indexGpu].secIsSupported(indexGpu))
    {
        dprintf("lw: SEC not supported on GPU %d.\n", indexGpu);
        return FALSE;
    }
    else
    {
        dprintf("lw: SEC supported on GPU %d.\n", indexGpu);
        return TRUE;
    }
}

//-----------------------------------------------------
// secDumpPriv - Dumps SEC priv reg space
//-----------------------------------------------------
LW_STATUS secDumpPriv(LwU32 indexGpu)
{
    if (!pCipher[indexGpu].secIsSupported(indexGpu))
    {
        dprintf("lw: secDumpPriv: SEC not supported on GPU %d.\n", indexGpu);
        return LW_ERR_NOT_SUPPORTED;
    }

    return pCipher[indexGpu].secDumpPriv(indexGpu);
}

//-----------------------------------------------------
// secDumpImem - Dumps SEC instruction memory
//-----------------------------------------------------
LW_STATUS secDumpImem(LwU32 indexGpu, LwU32 imemSize)
{
    if (!pCipher[indexGpu].secIsSupported(indexGpu))
    {
        dprintf("lw: secDumpImem: SEC not supported on GPU %d.\n", indexGpu);
        return LW_ERR_NOT_SUPPORTED;
    }
    
    return pCipher[indexGpu].secDumpImem(indexGpu, imemSize);
}

//-----------------------------------------------------
// secDumpDmem - Dumps SEC data memory
//-----------------------------------------------------
LW_STATUS secDumpDmem(LwU32 indexGpu, LwU32 dmemSize)
{
    if (!pCipher[indexGpu].secIsSupported(indexGpu))
    {
        dprintf("lw: secDumpDmem: SEC not supported on GPU %d.\n", indexGpu);
        return LW_ERR_NOT_SUPPORTED;
    }
    
    return pCipher[indexGpu].secDumpDmem(indexGpu, dmemSize);
}

//-----------------------------------------------------
// secTestState - Test basic SEC state
//-----------------------------------------------------
LW_STATUS secTestState(LwU32 indexGpu)
{
    if (!pCipher[indexGpu].secIsSupported(indexGpu))
    {
        dprintf("lw: secTestState: SEC not supported on GPU %d.\n", indexGpu);
        return LW_ERR_NOT_SUPPORTED;
    }

    return pCipher[indexGpu].secTestState(indexGpu);
}

//-----------------------------------------------------
// secDisplayHwcfg - Display SEC HW config state
//-----------------------------------------------------
LW_STATUS secDisplayHwcfg(LwU32 indexGpu)
{
    if (!pCipher[indexGpu].secIsSupported(indexGpu))
    {
        dprintf("lw: secDisplayHwcfg: SEC not supported on GPU %d.\n", indexGpu);
        return LW_ERR_NOT_SUPPORTED;
    }
    
    return pCipher[indexGpu].secDisplayHwcfg(indexGpu);
}

//-----------------------------------------------------
// secDisplayHelp - Display related help info
//-----------------------------------------------------
void secDisplayHelp(void)
{
    dprintf("SEC commands:\n");
    dprintf(" sec \"-help\"                   - Displays the SEC related help menu\n");
    dprintf(" sec \"-supported\"              - Determines if SEC is supported on available GPUs\n");
    dprintf(" sec \"-hwcfg\"                  - Display hardware config info for SEC\n");
    dprintf(" sec \"-priv\"                   - Dumps SEC priv registers\n");
    dprintf(" sec \"-imem  <imemsize>\"       - Dumps SEC instruction memory, need to  specify imemsize\n");
    dprintf(" sec \"-dmem  <dmemsize>\"       - Dumps SEC data memory , need to specify dmemsize \n");
    dprintf(" sec \"-state\"                  - Checks the current state of SEC\n");
    dprintf(" sec \"-spr\"                    - Dumps Flcn Special Purpose Registers like PC,CSW,SP using ICD\n");
}

//-------------------------------------------------------------------
// secDisplayHwcfg - Display SEC Falcon Special Purpose Registers
//--------------------------------------------------------------------
LW_STATUS secDisplayFlcnSPR(LwU32 indexGpu)
{
    if (!pCipher[indexGpu].secIsSupported(indexGpu))
    {
        dprintf("lw: secDisplayHwcfg: SEC not supported on GPU %d.\n", indexGpu);
        return LW_ERR_NOT_SUPPORTED;
    }
    
    return pCipher[indexGpu].secDisplayFlcnSPR(indexGpu);
}
