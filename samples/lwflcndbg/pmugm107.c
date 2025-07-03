/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch PMU helper.  
// pmugm107.c
//
//*****************************************************

//
// includes
//
#include "hal.h"
#include "falcon.h"
#include "maxwell/gm107/dev_pwr_pri.h"

/*!
 @brief Returns the maximum of breakpoint registers supported
 */
LwU32
pmuFlcngdbMaxBreakpointsGet_GM107()
{
    return 5;
}
