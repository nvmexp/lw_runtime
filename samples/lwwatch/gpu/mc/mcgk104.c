/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2010-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
 
//*****************************************************
//
// mcgk104.c
//
//*****************************************************

//
// includes
//
#include "hal.h"
#include "kepler/gk104/dev_therm.h"
#include "kepler/gk104/dev_fuse.h"

#include "g_mc_private.h"     // (rmconfig)  implementation prototypes


void
mcReadPgOnBootStatus_GK104()
{
    LwU32   data32;
    
    // Read the fuse state first
    data32 = GPU_REG_RD32(LW_FUSE_OPT_PGOB);
    if (!FLD_TEST_DRF(_FUSE, _OPT_PGOB, _DATA, _ENABLE, data32))
    {
        dprintf("lw: LW_FUSE_OPT_PGOB is NOT enabled : 0x%x\n", data32);
        dprintf("lw: This fuse should be enabled for PGOB\n");
        return;
    }
    else
    {
        dprintf("lw: LW_FUSE_OPT_PGOB IS enabled : 0x%x\n", data32);
    }
    
    // try to guess the present state we are in
    data32 = GPU_REG_RD32(LW_THERM_CTRL_1);
    dprintf("lw: LW_THERM_CTRL_1: 0x%x\n", data32);
    
    if (FLD_TEST_DRF(_THERM, _CTRL_1, _PGOB_OVERRIDE, _ENABLED, data32))
        dprintf("lw: LW_THERM_CTRL_1_PGOB_OVERRIDE : _ENABLED\n");
    else
        dprintf("lw: LW_THERM_CTRL_1_PGOB_OVERRIDE : _DISABLED\n");
    
    if (FLD_TEST_DRF(_THERM, _CTRL_1, _PGOB_OVERRIDE_VALUE, _OFF, data32))
        dprintf("lw: LW_THERM_CTRL_1_PGOB_OVERRIDE_VALUE : _OFF\n");
    else
        dprintf("lw: LW_THERM_CTRL_1_PGOB_OVERRIDE_VALUE : _ON\n");
}
