/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2018 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/
#include "ofa.h"
#include "hal.h"
#include "lwsym.h"
#include "g_ofa_hal.h"
#include "exts.h"

static OBJFLCN ofaFlcn[MAX_GPUS];
static int ofaObjInitialized[MAX_GPUS] = { 0 };
dbg_ofa_v01_00 *pOfaPrivReg[LWWATCH_MAX_OFA];
dbg_ofa_v01_00 *pOfaFuseReg = { 0 };
dbg_ofa_v01_00 *pOfaMethodTable;

LwU32 ofaId = 0;


POBJFLCN ofaGetFalconObject(void)
{
    if (!pOfa[indexGpu].ofaIsSupported())
    {
        // OFA is not supported
        return NULL;
    }

    // Initialize the object if it is not done yet
    if (ofaObjInitialized[indexGpu] == 0)
    {
        ofaFlcn[indexGpu].pFCIF = pOfa[indexGpu].ofaGetFalconCoreIFace();
        ofaFlcn[indexGpu].pFEIF = pOfa[indexGpu].ofaGetFalconEngineIFace();
        if (ofaFlcn[indexGpu].pFEIF)
        {
            ofaFlcn[indexGpu].engineName = ofaFlcn[indexGpu].pFEIF->flcnEngGetEngineName();
            ofaFlcn[indexGpu].engineBase = ofaFlcn[indexGpu].pFEIF->flcnEngGetFalconBase();
        }
        pOfa[indexGpu].ofaFillSymPath(ofaFlcn);
        ofaObjInitialized[indexGpu] = 1;
    }

    return &ofaFlcn[indexGpu];
}

LwU32 ofaGetDmemAccessPort(void)
{
    return 0;
}

//-----------------------------------------------------
// ofaIsGpuSupported - Determines if OFA is supported
//-----------------------------------------------------
BOOL ofaIsGpuSupported(LwU32 indexGpu)
{
    if (!pOfa[indexGpu].ofaIsGpuSupported(indexGpu))
    {
        return FALSE;
    }
    else
    {
        return TRUE;
    }
}

//-----------------------------------------------------
// ofaDumpPriv - Dumps OFA priv reg space
//-----------------------------------------------------
LW_STATUS ofaDumpPriv(LwU32 indexGpu)
{
    if (!pOfa[indexGpu].ofaIsGpuSupported(indexGpu))
    {
        return LW_ERR_NOT_SUPPORTED;
    }
    return pOfa[indexGpu].ofaDumpPriv(indexGpu);
}

//-----------------------------------------------------
// ofaDumpFuse - Dumps OFA priv reg space
//-----------------------------------------------------
LW_STATUS ofaDumpFuse(LwU32 indexGpu)
{
    if (!pOfa[indexGpu].ofaIsGpuSupported(indexGpu))
    {
        return LW_ERR_NOT_SUPPORTED;
    }
    return pOfa[indexGpu].ofaDumpFuse(indexGpu);
}

//-----------------------------------------------------
// ofaDumpDmem - Dumps OFA data memory
//-----------------------------------------------------
LW_STATUS ofaDumpDmem(LwU32 indexGpu, LwU32 dmemSize, LwU32 offs2MthdOffs)
{
    if (!pOfa[indexGpu].ofaIsGpuSupported(indexGpu))
    {
        return LW_ERR_NOT_SUPPORTED;
    }
    return pOfa[indexGpu].ofaDumpDmem(indexGpu, dmemSize, offs2MthdOffs);
}

//-----------------------------------------------------
// ofaDumpImem - Dumps OFA instruction memory
//-----------------------------------------------------
LW_STATUS ofaDumpImem(LwU32 indexGpu, LwU32 imemSize)
{
    if (!pOfa[indexGpu].ofaIsGpuSupported(indexGpu))
    {
        return LW_ERR_NOT_SUPPORTED;
    }
    return pOfa[indexGpu].ofaDumpImem(indexGpu, imemSize);
}

//-----------------------------------------------------
// ofaTestState - Test basic OFA state
//-----------------------------------------------------
LW_STATUS ofaTestState(LwU32 indexGpu)
{
    if (!pOfa[indexGpu].ofaIsGpuSupported(indexGpu))
    {
        return LW_ERR_NOT_SUPPORTED;
    }
    return pOfa[indexGpu].ofaTestState(indexGpu);
}

//-----------------------------------------------------
// ofaDisplayHwcfg - Display OFA HW config state
//-----------------------------------------------------
LW_STATUS ofaDisplayHwcfg(LwU32 indexGpu)
{
    if (!pOfa[indexGpu].ofaIsGpuSupported(indexGpu))
    {
        return LW_ERR_NOT_SUPPORTED;
    }
    return pOfa[indexGpu].ofaDisplayHwcfg(indexGpu);
}

//-----------------------------------------------------
// ofaDisplayHelp - Display related help info
//-----------------------------------------------------
void ofaDisplayHelp(void)
{
    dprintf("OFA commands:\n");
    dprintf(" ofa \"-help\"                     - Displays the OFA related help menu\n");
    dprintf(" ofa \"-supported <ofaId>\"        - Determines if OFA is supported on available GPUs, Default OFAID is set to 0 as OFA Engine is supported only on (v01_00_and_later)\n");
    dprintf(" ofa \"-hwcfg <ofaId>\"            - Displays hardware config info for OFA,  Default OFAID is set to 0 as OFA Engine is supported only on (v01_00_and_later)\n");
    dprintf(" ofa \"-priv <ofaId>\"             - Dumps OFA priv registers,  Default OFAID is set to 0 as OFA Engine is supported only on (v01_00_and_later)\n");
    dprintf(" ofa \"-fuse <ofaId>\"             - Dumps OFA related fuse registers,  Default OFAID is set to 0 as OFA Engine is supported only on (v01_00_and_later)\n");
    dprintf(" ofa \"-imem <imemsize> <ofaId>\"  - Dumps OFA instruction memory,  Default OFAID is set to 0 as OFA Engine is supported only on (v01_00_and_later)\n");
    dprintf(" ofa \"-dmem <dmemsize> [-offs2mthdoffs <offset>] <ofaId>\"\n");
    dprintf("                                   - Dumps OFA data memory and methods,  off2mthdoffs is offset, in bytes, of the DWORD in DMEM which stores the method array offset\n");
    dprintf("                                     Default OFAID is set to 0 as OFA Engine is supported only on (v01_00_and_later)\n");
    dprintf(" ofa \"-state <ofaId>\"            - Checks the current state of OFA, Default OFAID is set to 0 as OFA Engine is supported only on (v01_00_and_later)\n");
    dprintf(" ofa \"-spr <ofaId>\"              - Dumps Flcn Special Purpose Registers like PC,CSW,SP using ICD, Default OFAID is set to 0 as OFA Engine is supported only on (v01_00_and_later)\n");
}

//-------------------------------------------------------------------
// ofaDisplayHwcfg - Display OFA Falcon Special Purpose Registers
//--------------------------------------------------------------------
LW_STATUS ofaDisplayFlcnSPR(LwU32 indexGpu)
{
    if (!pOfa[indexGpu].ofaIsGpuSupported(indexGpu))
    {
        return LW_ERR_NOT_SUPPORTED;
    }
    return pOfa[indexGpu].ofaDisplayFlcnSPR(indexGpu);
}
