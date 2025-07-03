/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
 
//*****************************************************
//
// DPU Hal Functions
// dpu0205.c
//
//*****************************************************


/* ------------------------ Includes --------------------------------------- */
#include "hal.h"
#include "falcon.h"
#include "dpu.h"
#include "dpu/v02_05/dev_disp_falcon.h"

#include "g_dpu_private.h"     // (rmconfig)  implementation prototypes

/*!
 *@brief Returns the maximum number of breakpoints
 */
LwU32
dpuFlcngdbMaxBreakpointsGet_v02_05()
{
    return 5;
}

