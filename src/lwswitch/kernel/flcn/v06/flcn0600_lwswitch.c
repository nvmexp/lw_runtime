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
 * @file   flcn0600_lwswitch.c
 * @brief  Provides the implementation for all falcon 06.00 HAL interfaces.
 */

#include "flcn/flcn_lwswitch.h"


/**
 * @brief   set hal function pointers for functions defined in v06_00 (i.e. this file)
 *
 * this function has to be at the end of the file so that all the
 * other functions are already defined.
 *
 * @param[in] pFlcn   The flcn for which to set hals
 */
void
flcnSetupHal_v06_00
(
    PFLCN pFlcn
)
{
    // default to using definitions from v05_01
    flcnSetupHal_v05_01(pFlcn);
}
