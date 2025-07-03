/*
 * _LWRM_COPYRIGHT_START_
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
// grgf108.c
//
//*****************************************************

//
// includes
//
#include "fermi/gf108/dev_graphics_nobundle.h"
#include "gr.h"
#include "fermi/gf108/hwproject.h"
#include "fermi/gf108/dev_top.h"

#include "g_gr_private.h"       // (rmconfig) implementation prototypes


LwU32 grGetNumTpcPerGpc_GF108()
{
   LwU32  val;
   val = GPU_REG_RD32(LW_PTOP_SCAL_NUM_TPC_PER_GPC);
   return DRF_VAL(_PTOP_SCAL, _NUM_TPC_PER_GPC, _VALUE, val);
}

LwU32 grGetMaxTpcPerGpc_GF108()
{
    return LW_SCAL_LITTER_NUM_TPC_PER_GPC;
}

LwU32 grGetMaxGpc_GF108()
{
    return LW_SCAL_LITTER_NUM_GPCS;
}

LwU32 grGetMaxFbp_GF108()
{
    return LW_SCAL_LITTER_NUM_FBPS;
}
