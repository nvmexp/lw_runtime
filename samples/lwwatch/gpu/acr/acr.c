/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2018 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "acr.h"
#include "exts.h"
#include "g_acr_hal.h"

//-----------------------------------------------------
// acrIsSupported - Determines if acr is supported
//-----------------------------------------------------
BOOL acrIsSupported(LwU32 indexGpu)
{
    if (!pAcr[indexGpu].acrIsSupported(indexGpu))
    {
        dprintf("lw: ACR not supported on GPU %d.\n", indexGpu);
        return FALSE;
    }
    else
    {
        dprintf("lw: ACR supported on GPU %d.\n", indexGpu);
        return TRUE;
    }
}

//-------------------------------------------------------------------
// acrLsfStatus - Prints PRIV level of falcons, if engine is enabled
//-------------------------------------------------------------------
LW_STATUS acrLsfStatus(LwU32 indexGpu)
{
    if (!pAcr[indexGpu].acrIsSupported(indexGpu))
    {
        dprintf("lw: acrLsfStatus: ACR not supported on GPU %d.\n", indexGpu);
        return LW_ERR_NOT_SUPPORTED;
    }
    dprintf("Prints PRIV level of falcons, if engine is enabled\n");
    return pAcr[indexGpu].acrLsfStatus(indexGpu);
}

//-----------------------------------------------------
// acrGetRegionInfo - Get ACR region info
//-----------------------------------------------------
LW_STATUS acrGetRegionInfo(LwU32 indexGpu)
{
    if (!pAcr[indexGpu].acrIsSupported(indexGpu))
    {
        dprintf("lw: acrRegionStatus: ACR not supported on GPU %d.\n", indexGpu);
        return LW_ERR_NOT_SUPPORTED;
    }
    dprintf("Get ACR region info\n");
    return pAcr[indexGpu].acrGetRegionInfo(indexGpu);
}

//--------------------------------------------------------------------------------------------------
// acrRegionStatus - L0 Sanity testing - Verify if write is possible through NS client
//--------------------------------------------------------------------------------------------------
LW_STATUS acrRegionStatus(LwU32 indexGpu)
{
    if (!pAcr[indexGpu].acrIsSupported(indexGpu))
    {
        dprintf("lw: acrRegionStatus: ACR not supported on GPU %d.\n", indexGpu);
        return LW_ERR_NOT_SUPPORTED;
    }
    dprintf("L0 Sanity testing - Verify if write is possible through NS client\n");
    return pAcr[indexGpu].acrRegionStatus(indexGpu);
}

//--------------------------------------------------------------------------------------------------
// acrDmemProtection - Verify DMEM protection
//--------------------------------------------------------------------------------------------------
LW_STATUS acrDmemProtection(LwU32 indexGpu)
{
    if (!pAcr[indexGpu].acrIsSupported(indexGpu))
    {
        dprintf("lw: acrRegionStatus: ACR not supported on GPU %d.\n", indexGpu);
        return LW_ERR_NOT_SUPPORTED;
    }
    dprintf("Verify DMEM protection\n");
    return pAcr[indexGpu].acrDmemProtection(indexGpu);
}

//--------------------------------------------------------------------------------------------------
// acrImemProtection - Verify IMEM protection
//--------------------------------------------------------------------------------------------------
LW_STATUS acrImemProtection(LwU32 indexGpu)
{
    if (!pAcr[indexGpu].acrIsSupported(indexGpu))
    {
        dprintf("lw: acrRegionStatus: ACR not supported on GPU %d.\n", indexGpu);
        return LW_ERR_NOT_SUPPORTED;
    }
    dprintf("Verify IMEM protection\n");
    return pAcr[indexGpu].acrImemProtection(indexGpu);
}

//-----------------------------------------------------
// acrGetMultipleWprInfo - Get ACR Multiple WPR info
//-----------------------------------------------------
LW_STATUS acrGetMultipleWprInfo(LwU32 indexGpu)
{
    if (!pAcr[indexGpu].acrIsSupported(indexGpu))
    {
        dprintf("lw: acrMultipleWprInfo: ACR not supported on GPU %d.\n", indexGpu);
        return LW_ERR_NOT_SUPPORTED;
    }
    dprintf("Get ACR Multiple WPR info\n");
    return pAcr[indexGpu].acrGetMultipleWprInfo(indexGpu);
}

//-----------------------------------------------------------
// acrVerifyMultipleWprStatus - Verify Multiple WPR configuration
//-----------------------------------------------------------
LW_STATUS acrVerifyMultipleWprStatus(LwU32 indexGpu)
{
    if (!pAcr[indexGpu].acrIsSupported(indexGpu))
    {
        dprintf("lw: acrVerifyMultipleWprStatus: ACR not supported on GPU %d.\n", indexGpu);
        return LW_ERR_NOT_SUPPORTED;
    }
    dprintf("Verify ACR Multiple WPR settings\n");
    return pAcr[indexGpu].acrVerifyMultipleWprStatus(indexGpu);
}

//--------------------------------------------------------
// acrGetSharedWprStatus - Verify shared WPRs configuration
//--------------------------------------------------------
LW_STATUS acrGetSharedWprStatus(LwU32 indexGpu)
{
    if (!pAcr[indexGpu].acrIsSupported(indexGpu))
    {
        dprintf("lw: acrGetSharedWprStatus: ACR not supported on GPU %d.\n", indexGpu);
        return LW_ERR_NOT_SUPPORTED;
    }
    dprintf("Verify Shared subWPRs configuration\n");
    return pAcr[indexGpu].acrGetSharedWprStatus(indexGpu);
}

//-----------------------------------------------------
// acrDisplayHelp - Display related help info
//-----------------------------------------------------
void acrDisplayHelp(void)
{
    dprintf("ACR commands:\n");
    dprintf(" acr \"-help\"                   - Displays the ACR related help menu\n");
    dprintf(" acr \"-supported\"              - Determines if ACR is supported on available GPUs\n");
    dprintf(" acr \"-lsfstatus\"              - Checks the PRIV level of falcons, if engine is enabled\n");
    dprintf(" acr \"-getregioninfo\"          - Get ACR region info\n");
    dprintf(" acr \"-regionstatus\"           - L0 Sanity testing - Verify if write is possible through NS client\n");
    dprintf(" acr \"-dmemprotection\"         - Verify DMEM protection\n");
    dprintf(" acr \"-imemprotection\"         - Verify IMEM protection\n");
    dprintf(" acr \"-getmwprinfo\"            - Get ACR mWPR region info\n");
    dprintf(" acr \"-mwprstatus\"             - Verify mWPR regions configuration\n");
    dprintf(" acr \"-sharedwprstatus\"        - Verify configuration of shared WPR regions\n");
}
