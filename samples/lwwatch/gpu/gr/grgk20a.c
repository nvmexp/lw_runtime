/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2003-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// grgk20a.c
//
//*****************************************************

//
// includes
//
#include "chip.h"
#include "inst.h"
#include "print.h"
#include "gpuanalyze.h"
#include "kepler/gk20a/hwproject.h"
#include "kepler/gk20a/dev_pri_ringmaster.h"
#include "kepler/gk20a/dev_pri_ringstation_sys.h"
#include "kepler/gk20a/dev_top.h"
#include "gr.h"

#include "g_gr_private.h"       // (rmconfig) implementation prototypes


LwBool grCheckPrivAccess_GK20A(LwBool bForceEnable)
{
    LwU32 regSysPrivFsConfig;
    LwU32 sysMask;

    //
    // Check first if the graphics engine has priv access.
    // these bits is sys denotes this for graphics.
    //
    sysMask = BIT(LW_PPRIV_SYS_PRI_MASTER_fecs2ds_pri)      |
              BIT(LW_PPRIV_SYS_PRI_MASTER_fecs2pd_pri)      |
              BIT(LW_PPRIV_SYS_PRI_MASTER_fecs2pdb_pri)     |
              BIT(LW_PPRIV_SYS_PRI_MASTER_fecs2rastwod_pri) |
              BIT(LW_PPRIV_SYS_PRI_MASTER_fecs2scc_pri)     |
              BIT(LW_PPRIV_SYS_PRI_MASTER_fecs2fe_pri);

    regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(0));

    // Check that all the bits are enabled
    if ((regSysPrivFsConfig & sysMask) != sysMask)
    {
        // Check for forced enable access
        if (bForceEnable)
        {
            // Try to enable the bits
            GPU_REG_WR32(LW_PPRIV_SYS_PRIV_FS_CONFIG(0), (regSysPrivFsConfig | sysMask));

            // Check once again to see if access is enabled
            regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYS_PRIV_FS_CONFIG(0));
            if ((regSysPrivFsConfig & sysMask) != sysMask)
            {
                return FALSE;
            }
        }
        else
        {
            return FALSE;
        }
    }
    return TRUE;
}

LwU32 grGetMaxTpcPerGpc_GK20A()
{
    return LW_SCAL_LITTER_NUM_TPC_PER_GPC;
}

LwU32 grGetMaxGpc_GK20A()
{
    return LW_SCAL_LITTER_NUM_GPCS;
}

LwU32 grGetMaxFbp_GK20A(void)
{
    return LW_SCAL_LITTER_NUM_FBPS;
}

