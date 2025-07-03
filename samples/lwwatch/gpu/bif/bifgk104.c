/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2010-2018 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */


//*****************************************************
//
// lwwatch extension
// bifgk104.c
//
//*****************************************************

//
// includes
//
#include "kepler/gk104/dev_lw_xve.h"
#include "gpuanalyze.h"

#include "g_bif_private.h"     // (rmconfig)  implementation prototypes


void bifGetMsiInfo_GK104(void)
{
    LwU32 data32 = GPU_REG_RD32(DEVICE_BASE(LW_PCFG) + LW_XVE_MSI_CTRL);
    dprintf("lw: LW_XVE_MSI_CTRL:   0x%08x\n", data32);
    if (data32 & DRF_DEF(_XVE, _MSI_CTRL, _MSI, _ENABLE))
    {
        dprintf("lw:  +  _MSI_ENABLE\n");
    }
    else
    {
        dprintf("lw:  +  _MSI_DISABLE\n");
        dprintf("lw: INTx mode active\n");
    }
}

