/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file  pmgr.c
 * @brief WinDbg Extension for PMGR.
 */

/* ------------------------ Includes --------------------------------------- */
#include "os.h"
#include "hal.h"
#include "pmgr.h"
#include "print.h"
#include "lwsym.h"

/* ------------------------ Types definitions ------------------------------ */
/* ------------------------ Static variables ------------------------------- */
//static OBJFLCN pmuFlcn = {0};

/* ------------------------ Function Prototypes ---------------------------- */
/* ------------------------ Defines ---------------------------------------- */


/*!
 * Return string of engine name
 *
 * @return Engine Name
 */
const char*
pmgrGetEngineName()
{
    return "PMGR";
}

