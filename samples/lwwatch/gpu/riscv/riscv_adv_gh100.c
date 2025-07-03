/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019-2020 by LWPU Corporation.  All rights reserved.  All information
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

#include "hopper/gh100/dev_top.h"
#include "hopper/gh100/dev_bus.h"
#include "hopper/gh100/dev_falcon_v4.h"
#include "hopper/gh100/dev_riscv_pri.h"

#include "riscv_csrs_gh100.h"

#include "riscv_dbgint.h"
#include "g_riscv_private.h"
#include "hal.h"

#include "lwport/lwport.h"

static const unsigned char pmpModes[][6] = {"off  ", "tor  ", "na4  ", "napot"};
static const unsigned char pmpLock[][10] = {" unlocked", "   locked"};
static const unsigned char iopmpMasters[][6] = { "fbdma", "cpdma", "sha", "pmb" };

static LwU64 getPmpMode(LwU8 lwrrFlag, LwBool isIoPmp)
{
    return isIoPmp ? lwrrFlag & 0x3 :
           DRF_VAL64(_RISCV, _CSR, _PMPCFG0_PMP0A, lwrrFlag);
}

static void decodeIoPmpFlags_GH100(LwU8 pmpCfg, LwU32 ioPmpCfg)
{
    LwU32 iopmp_masters;
    iopmp_masters = DRF_SHIFTMASK(LW_PRISCV_RISCV_IOPMP_CFG_MASTER) & ioPmpCfg;

    dprintf("%s mode:%s %c%c masters:%s%s%s%s%s%s",
        FLD_TEST_DRF64(_PRISCV_RISCV, _IOPMP_CFG, _LOCK, _LOCKED, ioPmpCfg) ? pmpLock[1] : pmpLock[0],
        pmpModes[getPmpMode(pmpCfg, LW_TRUE)],
        FLD_TEST_DRF64(_PRISCV_RISCV, _IOPMP_CFG, _WRITE, _ENABLE, ioPmpCfg) ? 'w' : '-',
        FLD_TEST_DRF64(_PRISCV_RISCV, _IOPMP_CFG, _READ, _ENABLE, ioPmpCfg) ? 'r' : '-',
        iopmp_masters == 0 ? " none" : "",
        FLD_TEST_DRF64(_PRISCV_RISCV, _IOPMP_CFG, _MASTER_FBDMA_IMEM, _ENABLE, iopmp_masters) ? " FBDMA_IMEM" : "",
        FLD_TEST_DRF64(_PRISCV_RISCV, _IOPMP_CFG, _MASTER_FBDMA_DMEM, _ENABLE, iopmp_masters) ? " FBDMA_DMEM" : "",
        FLD_TEST_DRF64(_PRISCV_RISCV, _IOPMP_CFG, _MASTER_CPDMA, _ENABLE, iopmp_masters) ? " CPDMA" : "",
        FLD_TEST_DRF64(_PRISCV_RISCV, _IOPMP_CFG, _MASTER_SHA, _ENABLE, iopmp_masters) ? " SHA" : "",
        iopmp_masters >> 8 ? " PMB:0x" : "");

    if (iopmp_masters >> 8)
        dprintf("%02x", iopmp_masters >> 8);

    dprintf("\n");

    return;
}

static void decodePmpFlags(LwU8 pmpCfg, LwU8 nextPmpCfg)
{
    LwU64 mode;
    LwU64 lockStatus;
    LwU64 nextMode;
    LwU64 nextLockStatus;

    mode = getPmpMode(pmpCfg, LW_FALSE);
    lockStatus = FLD_TEST_DRF(_RISCV, _CSR, _PMPCFG0_PMP0L, _LOCK, pmpCfg);

    nextMode = getPmpMode(nextPmpCfg, LW_FALSE);
    nextLockStatus = FLD_TEST_DRF(_RISCV, _CSR, _PMPCFG0_PMP0L, _LOCK, nextPmpCfg);

    dprintf("%s mode:%s %c%c%c\n",
        FLD_TEST_DRF64(_RISCV, _CSR_PMPCFG0, _PMP0L, _LOCK, pmpCfg) ? pmpLock[1] : pmpLock[0],
        pmpModes[mode],
        FLD_TEST_DRF64(_RISCV, _CSR_PMPCFG0, _PMP0R, _PERMITTED, pmpCfg) ? 'r' : '-',
        FLD_TEST_DRF64(_RISCV, _CSR_PMPCFG0, _PMP0W, _PERMITTED, pmpCfg) ? 'w' : '-',
        FLD_TEST_DRF64(_RISCV, _CSR_PMPCFG0, _PMP0X, _PERMITTED, pmpCfg) ? 'x' : '-');

    // If next PMP register is in TOR mode
    // AND it is locked
    // AND this entry is not locked
    // the RESULT is that this entry's ADDR is locked IMPLICITLY. Warn user.
    if ((nextMode == 1) && (nextLockStatus) && (!lockStatus))
    {
        dprintf("Warning, above entry's address locked by entry above it.\n");
    }
    return;
}

static LwU8 getPmpFlag(LwU64 *pmpFlag, LwU64 idx)
{
    if (idx == 32)
    { // dummy value needed for nextFlag in case of _TOR
        return 0;
    }
    else if (idx > 32)
    {
        dprintf("getPmpFlag: invalid index");
        return -1;
    }
    return (LwU8)(pmpFlag[idx / 8] >> (8 * (idx % 8)));
}

static LW_STATUS parsePmpRegisters_GH100(LwU64 *pmpFlag, LwU64 *pmpAddr, LwU32 *ioPmpCfg, LwBool isIoPmp)
{
    unsigned int i;
    LwU64 entries = 32; // We have properly extended Core PMP!

    for (i = 0; i < entries; ++i)
    {
        LwU8 lwrrFlag, nextFlag;
        LwU64 addrL, addrH;
        LwU64 mode;

        addrL = pmpAddr[i] * 4;
        lwrrFlag = getPmpFlag(pmpFlag, i);
        nextFlag = isIoPmp ? 0 : getPmpFlag(pmpFlag, i+1);
        mode = getPmpMode(lwrrFlag, isIoPmp);

        if (mode == 0) // off
        {
            addrH = 0;
        }
        else if (mode == 1) // TOR
        {
            if (isIoPmp)
            {
                dprintf("TOR not supported for IOPMP.\n");
                return LW_ERR_ILWALID_STATE;
            }
            addrH = addrL;
            addrL = (i == 0) ? 0 : pmpAddr[i-1] * 4;
        }
        else if (mode == 2) // NA4
        {
            if (isIoPmp)
            {
                dprintf("NA4 not supported for IOPMP.\n");
                return LW_ERR_ILWALID_STATE;
            }
            addrH = addrL + 4;
        }
        else if (mode == 3) // NAPOT
        {
            LwU64 size;
            LwU64 ilwert;
            ilwert = ~addrL>>2;
            size = (ilwert == 0) ? 8ULL << 62 : 8ULL << portUtilCountTrailingZeros64(ilwert);
            addrL = addrL & ~(size - 1ULL);
            addrH = addrL + size;
        }
        else
            return LW_ERR_ILWALID_STATE;

        if (mode == 0)
            dprintf("%2u "LwU64_FMT"                    ", i, addrL);
        else
            dprintf("%2u "LwU64_FMT"-"LwU64_FMT" ", i, addrL, addrH - 1);

        if (isIoPmp)
            decodeIoPmpFlags_GH100(lwrrFlag, ioPmpCfg[i]);
        else
            decodePmpFlags(lwrrFlag, nextFlag);
    }
    return LW_OK;
}

LW_STATUS riscvDumpPmp_GH100(int regions)
{
    LwU64 pmpFlag[4];
    LwU64 pmpAddr[32];
    int i;

    dprintf("Core PMP registers:\n");

    CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_PMPCFG0, &pmpFlag[0]));
    CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_PMPCFG2, &pmpFlag[1]));
    for (i=0; i<16; i++)
        CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_PMPADDR(i), &pmpAddr[i]));

    CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_MEXTPMPCFG0, &pmpFlag[2]));
    CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_MEXTPMPCFG2, &pmpFlag[3]));
    for (i=0; i<16; i++)
        CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_MEXTPMPADDR(i), &pmpAddr[i+16]));

    CHECK_SUCCESS_OR_RETURN(parsePmpRegisters_GH100(pmpFlag, pmpAddr, NULL, LW_FALSE));

    return LW_OK;
}

LW_STATUS riscvDumpIoPmp_GH100(LwU64 vaOffset)
{
    struct
    {
        LwU64 addr;
        LwU8 master;
        LwU8 read;
        LwU8 entry;
    } iopmpFault;

    LwU64 pmpFlag[4];
    LwU8 *pmpFlagU8;
    LwU64 pmpAddr[32];
    LwU32 pmpCfg[32];
    LwU32 pmpAddrL, pmpAddrH;
    LwU32 iopmp_mode;
    LwU32 iopmp_err_stat;
    LwU32 iopmp_err_info;
    int i, j;

    if (!pRiscv[indexGpu].riscvIsInIcd())
    {
        dprintf("IO-PMP registers are priv protected. Core must be in ICD to proceed.\n");
        return LW_ERR_ILWALID_STATE;
    }

    pmpFlagU8 = (LwU8 *)&pmpFlag[0];
    dprintf("IO-PMP registers:\n");

    if (vaOffset == 0) // Read from physical address.
        vaOffset = pRiscvInstance->riscvBase + pRiscvInstance->riscv_priv_start;
    else // Read from virtual address specified by user.
        vaOffset += pRiscvInstance->riscvBase;

    for (i=0; i<2; ++i)
    {
        riscvMemRead((vaOffset + LW_PRISCV_RISCV_IOPMP_MODE(i)), 4, &iopmp_mode, LW_TRUE);
        for (j=0; j<16; ++j)
        {
            pmpFlagU8[16*i+j] = (LwU8) DRF_VAL(_PRISCV, _RISCV, _IOPMP_MODE_VAL_ENTRY(j), iopmp_mode);
        }
    }

    for (i=0; i<32; ++i)
    {
        riscvMemWrite((vaOffset + LW_PRISCV_RISCV_IOPMP_INDEX), 4, &i, LW_TRUE);
        riscvMemRead((vaOffset + LW_PRISCV_RISCV_IOPMP_CFG), 4, &pmpCfg[i], LW_TRUE);
        riscvMemRead((vaOffset + LW_PRISCV_RISCV_IOPMP_ADDR_HI), 4, &pmpAddrH, LW_TRUE);
        riscvMemRead((vaOffset + LW_PRISCV_RISCV_IOPMP_ADDR_LO), 4, &pmpAddrL, LW_TRUE);
        pmpAddr[i] = ((LwU64)pmpAddrH)<<32 | pmpAddrL;
    }

    CHECK_SUCCESS_OR_RETURN(parsePmpRegisters_GH100(pmpFlag, pmpAddr, pmpCfg, LW_TRUE));

    riscvMemRead((vaOffset + LW_PRISCV_RISCV_IOPMP_ERR_STAT), 4, &iopmp_err_stat, LW_TRUE);

    if (DRF_VAL(_PRISCV, _RISCV, _IOPMP_ERR_STAT_VALID, iopmp_err_stat))
    {
        dprintf("IO-PMP violation detected:\n");
        riscvMemRead((vaOffset + LW_PRISCV_RISCV_IOPMP_ERR_INFO), 4, &iopmp_err_info, LW_TRUE);
        riscvMemRead((vaOffset + LW_PRISCV_RISCV_IOPMP_ERR_ADDR_HI), 4, &pmpAddrH, LW_TRUE);
        riscvMemRead((vaOffset + LW_PRISCV_RISCV_IOPMP_ERR_ADDR_LO), 4, &pmpAddrL, LW_TRUE);
        iopmpFault.master = DRF_VAL(_PRISCV, _RISCV, _IOPMP_ERR_INFO_MASTER, iopmp_err_info);
        iopmpFault.read = DRF_VAL(_PRISCV, _RISCV, _IOPMP_ERR_INFO_READ, iopmp_err_info);
        iopmpFault.entry = DRF_VAL(_PRISCV, _RISCV, _IOPMP_ERR_INFO_ENTRY, iopmp_err_info);
        iopmpFault.addr = ((LwU64)pmpAddrH << 32ULL) | (LwU64)pmpAddrL;

        printf("Entry %u, %s fault @ "LwU64_FMT", by %s.\n", iopmpFault.entry,
            iopmpFault.read ? "read" : "write", 4ULL*iopmpFault.addr,
            iopmpFault.master <= 3 ? iopmpMasters[iopmpFault.master] : iopmpMasters[3]);
    }

    return LW_OK;
}
