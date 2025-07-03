/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2003-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// grGK208.c
//
//*****************************************************

//
// includes
//
#include "gr.h"
#include "kepler/gk208/hwproject.h"

#include "g_gr_private.h"       // (rmconfig) implementation prototypes


LwU32 grGetMaxTpcPerGpc_GK208()
{
    return LW_SCAL_LITTER_NUM_TPC_PER_GPC;
}

LwU32 grGetMaxGpc_GK208()
{
    return LW_SCAL_LITTER_NUM_GPCS;
}

LwU32 grGetMaxFbp_GK208(void)
{
    return LW_SCAL_LITTER_NUM_FBPS;
}
