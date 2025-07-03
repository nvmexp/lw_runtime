/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <ctype.h>
#include <limits.h>

#include <os.h>
#include <utils/lwassert.h>

#include "riscv_prv.h"
#include "riscv_gdbmon.h"
#include "riscv_porting.h"

#include "turing/tu102/dev_gsp_riscv_csr_64.h"

#include "g_riscv_private.h"

void _monitorReadComms_TU10X(void)
{
    LW_STATUS ret;
    int ofs = 0;
    LwU64 tohost, fromhost;

    ret = riscvIcdRcsr_TU10X(LW_RISCV_CSR_MFROMHOST, &fromhost);
    if (ret == LW_OK)
        ofs = sprintf(monitorReplyBuf, "mfromhost = "LwU64_FMT"    ", fromhost);
    else
        ofs = sprintf(monitorReplyBuf, "mfromhost = ?? Err: %x  ", ret);

    ret = riscvIcdRcsr_TU10X(LW_RISCV_CSR_MTOHOST, &tohost);
    if (ret == LW_OK)
        sprintf(monitorReplyBuf + ofs, "mtohost = "LwU64_FMT"\n", tohost);
    else
        sprintf(monitorReplyBuf + ofs, "mtohost = ?? Err: %x\n", ret);
}

void _monitorWriteHost_TU10X(void)
{
    LwU64 val;
    LW_STATUS ret;
    char *pVs, *pErr;
    char *pRbp = monitorReplyBuf;

    pVs = strtok_r(NULL, " \t", &pMonitorSavedPtr);
    if (!pVs)
    {
        sprintf(pRbp, "Usage: wh [number]\n");
        return;
    }
    val = strtoull(pVs, &pErr, 0);
    if (pErr == pVs)
    {
        sprintf(pRbp, "Usage: wh [number]\n");
        return;
    }

    ret = riscvIcdWcsr_TU10X(LW_RISCV_CSR_MFROMHOST, val);
    if (ret == LW_OK)
        sprintf(pRbp, "mfromhost = "LwU64_FMT"\n", val);
    else
        sprintf(pRbp, "mfromhost = ?? Err:%x\n", ret);
}
