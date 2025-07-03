/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include <stdio.h>
#include <limits.h>
#include <lwtypes.h>
#include <os.h>
#include <chip.h>

#include "riscv_prv.h"

//
// Read/write RISC-V engine-agnostic registers
//

LwU32 bar0Read(LwUPtr offset)
{
    if (IsTegra())
    {
        return DEV_REG_RD32(pRiscvInstance->riscvBase + offset, pRiscvInstance->name, 0);
    }
    else
    {
        return GPU_REG_RD32(pRiscvInstance->riscvBase + offset);
    }
}

void bar0Write(LwUPtr offset, LwU32 value)
{
    if (IsTegra())
    {
        DEV_REG_WR32(pRiscvInstance->riscvBase + offset, value, pRiscvInstance->name, 0);
    }
    else
    {
        GPU_REG_WR32(pRiscvInstance->riscvBase + offset, value);
    }
}

//
// Read/write Falcon engine-agnostic registers
//

LwU32 bar0ReadLegacy(LwUPtr offset)
{
    if (IsTegra())
    {
        return DEV_REG_RD32(pRiscvInstance->bar0Base + offset, pRiscvInstance->name, 0);
    }
    else
    {
        return GPU_REG_RD32(pRiscvInstance->bar0Base + offset);
    }
}

void bar0WriteLegacy(LwUPtr offset, LwU32 value)
{
    if (IsTegra())
    {
        DEV_REG_WR32(pRiscvInstance->bar0Base + offset, value, pRiscvInstance->name, 0);
    }
    else
    {
        GPU_REG_WR32(pRiscvInstance->bar0Base + offset, value);
    }
}
