/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <lwtypes.h>
#include <print.h>

#include "riscv_prv.h"
#include "g_riscv_private.h"

#include "turing/tu102/dev_gsp_riscv_csr_64.h"

#define pcsr(r) riscvIcdRcsr_TU10X(r, &reg); dprintf("[%3x] %s = %016llx\n", r, #r, reg);
void riscvDumpCsr_TU10X(void)
{
    LwU64 reg;

    if (!riscvIsInIcd_TU10X())
    {
        dprintf("Core must be halted.\n");
        return;
    }

    //rcsr
    pcsr(LW_RISCV_CSR_MISA); // 0x2 == ISA
    pcsr(LW_RISCV_CSR_MVENDORID); // 0x6e7669646961 == LWPU
    pcsr(LW_RISCV_CSR_MARCHID);
    pcsr(LW_RISCV_CSR_MIMPID);
    pcsr(LW_RISCV_CSR_MHARTID);
    pcsr(LW_RISCV_CSR_TSELECT);
    pcsr(LW_RISCV_CSR_TDATA1);
    pcsr(LW_RISCV_CSR_TDATA2);
    pcsr(LW_RISCV_CSR_TDATA3);
}
