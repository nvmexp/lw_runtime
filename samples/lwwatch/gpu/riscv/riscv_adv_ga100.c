/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include <string.h>
#include "lwmisc.h"
#include "riscv_prv.h"
#include "riscv_porting.h"

#include "ampere/ga100/dev_falcon_v4.h"
#include "ampere/ga100/dev_riscv_pri.h"
#include "ampere/ga100/dev_pmu_riscv_csr_64.h"

#include "riscv_regs.h"
#include "riscv_csrs_ga100.h"

#include "riscv_dbgint.h"
#include "g_riscv_private.h"
#include "hal.h"

/*
 * Decode symbolic name of CSR. Returns -1 if CSR not found.
 */
LwS16 riscvDecodeCsr_GA100(const char *name, size_t nameLen)
{
    const struct NamedCsr *pCsr = &_csrs[0];
    size_t nLen;


    if (!name)
        return -1;

    if (nameLen)
        nLen = nameLen;
    else
        nLen = strlen(name);

    if (!nLen)
        return -1;

    while (pCsr->name)
    {
        if (!strncasecmp(pCsr->name, name, nLen))
            return pCsr->address;
        pCsr++;
    }

    return -1;
}
