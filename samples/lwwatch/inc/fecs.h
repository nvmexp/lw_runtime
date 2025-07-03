/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// Non-Hal public functions for FECS
// fecs.h
//
//*****************************************************

#ifndef _FECS_H_
#define _FECS_H_

/* ------------------------ Includes --------------------------------------- */
#include "os.h"
#include "hal.h"
#include "falcon.h"

/* ------------------------ Function Prototypes ---------------------------- */
const char*             fecsGetEngineName(void);
POBJFLCN                fecsGetFalconObject(void);


/* ------------------------ Common Defines --------------------------------- */
/* ------------------------ Types definitions ------------------------------ */
/* ------------------------ Static variables ------------------------------- */

#include "g_fecs_hal.h"     // (rmconfig)  public interfaces

#endif // _FECS_H_
