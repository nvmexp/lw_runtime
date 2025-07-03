/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "vpr.h"
#include "exts.h"
#include "g_vpr_hal.h"

//-----------------------------------------------------
// vprDisplayHelp - Displays help of VPR extension and available arguments
//-----------------------------------------------------
void vprDisplayHelp(void)
{
    dprintf("VPR commands:\n");
    dprintf("    vpr -help                                             - Displays the VPR related help menu\n");
    dprintf("    vpr -supported                                        - Determines if VPR is supported on available GPUs\n");
    dprintf("    vpr -lwrrangeinmmu                                    - Prints current VPR range in mmu\n");
    dprintf("    vpr -maxrangeinbsi                                    - Prints max VPR range in bsi\n");
    dprintf("    vpr -lwrrangeinbsi                                    - Prints current VPR range in bsi\n");
    dprintf("    vpr -getmemlockstatus                                 - Prints global memory lock status\n");
    dprintf("    vpr -memlockrange                                     - Prints memory lock range\n");
    dprintf("    vpr -getHdcpType1LockStatusInBSISelwreScratch         - Prints type1 lock status of BSI Scratch\n");
    dprintf("    vpr -hwfuseversions                                   - Prints various HW fuse versions\n");
    dprintf("    vpr -ucodeversions                                    - Prints various ucode versions\n");
}

//-----------------------------------------------------
// vprIsSupported - Determines if vpr is supported by gpu
//-----------------------------------------------------
LwBool vprIsSupported(LwU32 indexGpu)
{
    if (pVpr[indexGpu].vprIsSupported(indexGpu))
    {
        dprintf("\lw: VPR is supported on GPU %d.\n\n", indexGpu);
        return LW_TRUE;
    }

    return LW_FALSE;
}

//-------------------------------------------------------------------
// vprMmuLwrrentRangeInfo - Prints current vpr range in mmu
//-------------------------------------------------------------------
LW_STATUS vprMmuLwrrentRangeInfo(LwU32 indexGpu, LwBool bPrintline)
{
    LW_STATUS status = LW_OK;

    if (pVpr[indexGpu].vprIsSupported(indexGpu))
    {
        dprintf("\n|| Lwr Start: MMU || Lwr End: MMU ||\n");
        dprintf("||");
        status = pVpr[indexGpu].vprMmuLwrrentRangeInfo(indexGpu);
        if(bPrintline)
        {
            dprintf("\n");
        }
        return status;

    }
    return LW_ERR_NOT_SUPPORTED;
    
}

//-------------------------------------------------------------------
// vprBsiMaxRangeInfo - Prints max vpr range in bsi
//-------------------------------------------------------------------
LW_STATUS vprBsiMaxRangeInfo(LwU32 indexGpu, LwBool bPrintline)
{
    LW_STATUS status = LW_OK;

    if (pVpr[indexGpu].vprIsSupported(indexGpu))
    {
        dprintf("\n||   Max Size: BSI    ||\n");
        dprintf("||");
        status = pVpr[indexGpu].vprBsiMaxRangeInfo(indexGpu);
        if(bPrintline)
        {
            dprintf("\n");
        }
        return status;
    }
    return LW_ERR_NOT_SUPPORTED;
}

//-------------------------------------------------------------------
// vprBsiLwrrentRangeInfo - Prints current vpr range in bsi
//-------------------------------------------------------------------
LW_STATUS vprBsiLwrrentRangeInfo(LwU32 indexGpu, LwBool bPrintline)
{
    LW_STATUS status = LW_OK;

    if (pVpr[indexGpu].vprIsSupported(indexGpu))
    {
        dprintf("\n|| Lwr Start: BSI || Lwr End: BSI ||\n");
        dprintf("||");
        status = pVpr[indexGpu].vprBsiLwrrentRangeInfo(indexGpu);
        if(bPrintline)
        {
            dprintf("\n");
        }
        return status;
    }
    return LW_ERR_NOT_SUPPORTED;
}

//-------------------------------------------------------------------
// vprPrintMemLockStatus - Prints global memory lock status
//-------------------------------------------------------------------
void vprPrintMemLockStatus(LwU32 indexGpu, LwBool bPrintline)
{
    if (pVpr[indexGpu].vprIsSupported(indexGpu))
    {
        dprintf("\n|| GlobalMemLock State  ||\n");
        dprintf("||");
        pVpr[indexGpu].vprPrintMemLockStatus(indexGpu);
        if(bPrintline)
        {
            dprintf("\n");
        }
    }
}

//-------------------------------------------------------------------
// vprMemLockRangeInfo - Prints memory lock range
//-------------------------------------------------------------------
LW_STATUS vprMemLockRangeInfo(LwU32 indexGpu, LwBool bPrintline)
{
    LW_STATUS status = LW_OK;

    if (pVpr[indexGpu].vprIsSupported(indexGpu))
    {
        dprintf("\n|| Mem Lock: Start  || Mem Lock: End ||\n");
        dprintf("||");
        status = pVpr[indexGpu].vprMemLockRangeInfo(indexGpu);
        if(bPrintline)
        {
            dprintf("\n");
        }
        return status;
    }
    return LW_ERR_NOT_SUPPORTED;
}

//--------------------------------------------------------------------
// vprPrintBsiType1LockStatus - Prints type1 lock status of BSI Scratch
//--------------------------------------------------------------------
void vprPrintBsiType1LockStatus(LwU32 indexGpu, LwBool bPrintline)
{
    if (pVpr[indexGpu].vprIsSupported(indexGpu))
    {
        dprintf("\n|| Type1LockStatus: BSI Scratch ||\n");
        dprintf("||");
        pVpr[indexGpu].vprPrintBsiType1LockStatus(indexGpu);
        if(bPrintline)
        {
            dprintf("\n");
        }
    }
}

//-------------------------------------------------------------------
// vprGetAllInfo - Prints all the info
//-------------------------------------------------------------------
LW_STATUS vprGetAllInfo(LwU32 indexGpu)
{
    LW_STATUS status = LW_OK;

    if(pVpr[indexGpu].vprIsSupported(indexGpu))
    {
        dprintf("\n|| Lwr Start: MMU || Lwr End: MMU ||   Max Size: BSI    || Lwr Start: BSI || Lwr End: BSI "
                  "|| GlobalMemLock State  || Mem Lock: Start  || Mem Lock: End || Type1LockStatus: BSI Scratch ||\n||");

        pVpr[indexGpu].vprMmuLwrrentRangeInfo(indexGpu);
        pVpr[indexGpu].vprBsiMaxRangeInfo(indexGpu);
        pVpr[indexGpu].vprBsiLwrrentRangeInfo(indexGpu);
        pVpr[indexGpu].vprPrintMemLockStatus(indexGpu);
        pVpr[indexGpu].vprMemLockRangeInfo(indexGpu);
        pVpr[indexGpu].vprPrintBsiType1LockStatus(indexGpu);
        vprGetHwFuseVersions(indexGpu);
        vprGetUcodeVersions(indexGpu);
    }
    else
    {
        status = LW_ERR_GENERIC;
    }

    return status;
}

//-------------------------------------------------------------------
// vprGetHwFuseVersions - Prints all the hw fuse versions
//-------------------------------------------------------------------
LW_STATUS vprGetHwFuseVersions(LwU32 indexGpu)
{
    LW_STATUS status = LW_OK;
    LwU32 vFuseAcr, vFuseCtxsw, vFuseLwdec, vFuseScrubber, vFuseSec2, vFuseUde, vFuseVprApp;

    if(pVpr[indexGpu].vprIsSupported(indexGpu))
    {
        vFuseAcr      = pVpr[indexGpu].vprGetFuseVersionAcr(indexGpu);
        vFuseCtxsw    = pVpr[indexGpu].vprGetFuseVersionCtxsw(indexGpu);
        vFuseLwdec    = pVpr[indexGpu].vprGetFuseVersionLwdec(indexGpu);
        vFuseScrubber = pVpr[indexGpu].vprGetFuseVersionScrubber(indexGpu);
        vFuseSec2     = pVpr[indexGpu].vprGetFuseVersionSec2(indexGpu);
        vFuseUde      = pVpr[indexGpu].vprGetFuseVersionUde(indexGpu);
        vFuseVprApp   = pVpr[indexGpu].vprGetFuseVersiolwprApp(indexGpu);

        dprintf("\n\nHW Fuse Versions:");
        
        dprintf("\n|| ACR || CTXSW || LWDEC || SCRUBBER || SEC2 || UDE || VPR APP ||");
        
        dprintf("\n|| %#3x || %#3x   || %#3x   ||  %#3x     || %#3x  || %#3x ||  %#3x    ||\n",
                  vFuseAcr, vFuseCtxsw, vFuseLwdec, vFuseScrubber, vFuseSec2, vFuseUde, vFuseVprApp);
    }
    else
    {
        status = LW_ERR_GENERIC;
    }

    return status;
}

//-------------------------------------------------------------------
// vprGetUcodeVersions - Prints all the ucode versions
//-------------------------------------------------------------------
LW_STATUS vprGetUcodeVersions(LwU32 indexGpu)
{
    LW_STATUS status = LW_OK;
    LwU32 vUcodeAcr, vUcodeScrubber, vUcodeUde;

    if(pVpr[indexGpu].vprIsSupported(indexGpu))
    {
        vUcodeAcr      = pVpr[indexGpu].vprGetUcodeVersionAcr(indexGpu);
        vUcodeScrubber = pVpr[indexGpu].vprGetUcodeVersionScrubber(indexGpu);
        vUcodeUde      = pVpr[indexGpu].vprGetUcodeVersionUde(indexGpu);

        dprintf("\n\nUcode Versions:");
        
        dprintf("\n|| ACR || SCRUBBER || UDE ||");
        
        dprintf("\n|| %#3x ||   %#3x    ||%#4x ||\n", vUcodeAcr, vUcodeScrubber, vUcodeUde);
    }
    else
    {
        status = LW_ERR_GENERIC;
    }

    return status;
}
