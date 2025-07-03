/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file   flcn0501_lwswitch.c
 * @brief  Provides the implementation for all falcon 5.1 HAL interfaces.
 */

#include "flcn/flcn_lwswitch.h"


/**
 * @brief   set hal function pointers for functions defined in v05_01 (i.e. this file)
 *
 * this function has to be at the end of the file so that all the
 * other functions are already defined.
 *
 * @param[in] pFlcn   The flcn for which to set hals
 */
void
flcnSetupHal_v05_01
(
    PFLCN pFlcn
)
{
    // default to using definitions from v04_00
    flcnSetupHal_v04_00(pFlcn);
}
