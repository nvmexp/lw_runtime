/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2021 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#include "turing/tu102/dev_top.h"
#include "turing/tu102/dev_ce.h"

#include "hwref/lwutil.h"
#include "fifo.h"
#include "deviceinfo.h"
#include "ce.h"
#include "g_ce_hal.h"

/*!
 *  Get the LW_CE_PCE2LCE_CONFIG__SIZE_1 value
 *
 *  @return LW_CE_PCE2LCE_CONFIG__SIZE_1 value
 */
LwU32 ceGetPceToLceConfigSize_TU102(void)
{
    return LW_CE_PCE2LCE_CONFIG__SIZE_1;
}

/*!
 *  Get the LW_CE_LCE_STATUS__SIZE_1 value
 *
 *  @return LW_CE_LCE_STATUS__SIZE_1 value
 */
LwU32 ceGetCeLceStatusSize_TU102(void)
{
    return LW_CE_LCE_STATUS__SIZE_1;
}
