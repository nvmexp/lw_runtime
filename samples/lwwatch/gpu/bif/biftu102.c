/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */


//*****************************************************
//
// lwwatch extension
// biftu102.c
//
//*****************************************************

//
// includes
//
#include "turing/tu102/dev_lw_xve.h"
#include "gpuanalyze.h"

#include "g_bif_private.h"     // (rmconfig)  implementation prototypes


void bifGetMsiInfo_TU102(void)
{
    LwBool bIntxMode = LW_TRUE;
    LwU32 val = GPU_REG_RD32(DEVICE_BASE(LW_PCFG) + LW_XVE_MSI_CTRL);
    dprintf("lw: LW_XVE_MSI_CTRL:   0x%08x\n", val);
    if (FLD_TEST_DRF(_XVE, _MSI_CTRL, _MSI, _ENABLE, val))
    {
        dprintf("lw:  +  _MSI_ENABLE\n");
        bIntxMode = LW_FALSE;
    }
    else
    {
        dprintf("lw:  +  _MSI_DISABLE\n");
    }

    val = GPU_REG_RD32(DEVICE_BASE(LW_PCFG) + LW_XVE_MSIX_CAP_HDR);
    dprintf("lw: LW_XVE_MSIX_CAP_HDR:   0x%08x\n", val);
    if (FLD_TEST_DRF(_XVE, _MSIX_CAP_HDR, _ENABLE, _ENABLED, val))
    {
        dprintf("lw:  +  _MSIX_CAP_HDR_ENABLE = _ENABLED\n");
        bIntxMode = LW_FALSE;
    }
    else
    {
        dprintf("lw:  +  _MSIX_CAP_HDR_ENABLE = _DISABLED\n");
    }

    if (bIntxMode)
    {
        dprintf("lw: INTx mode active\n");
    }
}

