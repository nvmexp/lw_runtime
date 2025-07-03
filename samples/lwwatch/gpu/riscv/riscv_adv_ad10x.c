/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <lwtypes.h>
#include "riscv_prv.h"

#include "ada/ad102/dev_riscv_pri.h"

LwBool riscvIsActive_AD10X(void)
{
    LwU64 r;

    r = bar0Read(LW_PRISCV_RISCV_CPUCTL);
    return FLD_TEST_DRF(_PRISCV_RISCV, _CPUCTL, _ACTIVE_STAT, _ACTIVE, r);
}
