/*
 * _LWRM_COPYRIGHT_START_
 *
 * Copyright 2011-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// grgf119.c
//
//*****************************************************

//
// includes
//
#include "fermi/gf119/dev_graphics_nobundle.h"
#include "gr.h"
#include "fermi/gf119/hwproject.h"
#include "fermi/gf119/dev_top.h"

#include "g_gr_private.h"       // (rmconfig) implementation prototypes


LwU32 grGetMaxTpcPerGpc_GF119()
{
    return LW_SCAL_LITTER_NUM_TPC_PER_GPC;
}
