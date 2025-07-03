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

#include "ampere/ga102/dev_top.h"
#include "ampere/ga102/dev_bus.h"
#include "ampere/ga102/dev_falcon_v4.h"
#include "ampere/ga102/dev_riscv_pri.h"
#include "ampere/ga102/dev_pmu_riscv_csr_64.h"
#include "ampere/ga102/lw_pmu_riscv_address_map.h"

#include "riscv_csrs_ga10x.h"

#include "riscv_dbgint.h"
#include "g_riscv_private.h"
#include "hal.h"

#include "lwport/lwport.h"

LwBool riscvIsActive_GA10X(void)
{
    LwU64 r;

    /*
     * Older emulation netlists have Turing core switch registers
     */
    r = GPU_REG_RD32(LW_PTOP_PLATFORM);
    if (FLD_TEST_DRF(_PTOP, _PLATFORM, _TYPE, _EMU, r) &&
        (GPU_REG_RD32(LW_PBUS_EMULATION_REV0) <= 9))
    {
        return riscvIsActive_TU10X();
    }

    r = bar0Read(LW_PRISCV_RISCV_CPUCTL);
    return FLD_TEST_DRF(_PRISCV_RISCV, _CPUCTL, _ACTIVE_STAT, _ACTIVE, r);
}

LwBool riscvHasMpuEnabled_GA10X(void)
{
    LwU64 mstatus;
    LwU64 satp;
    int ret;

    ret = riscvIcdRcsr_TU10X(LW_RISCV_CSR_SATP, &satp);
    if (ret != LW_OK)
    {
        dprintf("Failed reading satp.\n");
        return LW_FALSE;
    }

    if (FLD_TEST_DRF64(_RISCV, _CSR_SATP, _MODE, _LWMPU, satp)) {
        return LW_TRUE;
    }

    ret = riscvIcdRcsr_TU10X(LW_RISCV_CSR_MSTATUS, &mstatus);
    if (ret != LW_OK)
    {
        dprintf("Failed reading mstatus.\n");
        return LW_FALSE;
    }

    if (FLD_TEST_DRF64(_RISCV, _CSR_MSTATUS, _MPRV, _ENABLE, mstatus)) {
        return LW_TRUE;
    }

    return LW_FALSE;
}

static const unsigned char pmpModes[][6] = {"off  ", "tor  ", "na4  ", "napot"};
static const unsigned char pmpLock[][10] = {" unlocked", "   locked"};
static const unsigned char iopmpMasters[][6] = { "fbdma", "cpdma", "sha", "pmb" };

static LwU64 getPmpMode(LwU8 lwrrFlag, LwBool isIoPmp)
{
    return isIoPmp ? lwrrFlag & 0x3 :
           DRF_VAL64(_RISCV, _CSR, _PMPCFG0_PMP0A, lwrrFlag);
}

static void decodeIoPmpFlags(LwU8 pmpCfg, LwU32 ioPmpCfg)
{
    LwU32 iopmp_masters;
    iopmp_masters = DRF_SHIFTMASK(LW_PRISCV_RISCV_IOPMP_CFG_MASTER) & ioPmpCfg;

    dprintf("%s mode:%s %c%c masters:%s%s%s%s%s",
        FLD_TEST_DRF64(_PRISCV_RISCV, _IOPMP_CFG, _LOCK, _LOCKED, ioPmpCfg) ? pmpLock[1] : pmpLock[0],
        pmpModes[getPmpMode(pmpCfg, LW_TRUE)],
        FLD_TEST_DRF64(_PRISCV_RISCV, _IOPMP_CFG, _WRITE, _ENABLE, ioPmpCfg) ? 'w' : '-',
        FLD_TEST_DRF64(_PRISCV_RISCV, _IOPMP_CFG, _READ, _ENABLE, ioPmpCfg) ? 'r' : '-',
        iopmp_masters == 0 ? " none" : "",
        FLD_TEST_DRF64(_PRISCV_RISCV, _IOPMP_CFG, _MASTER_FBDMA, _ENABLE, iopmp_masters) ? " FBDMA" : "",
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

static LW_STATUS parsePmpRegisters(LwU64 *pmpFlag, LwU64 *pmpAddr, LwU32 *ioPmpCfg, LwBool isIoPmp)
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
            decodeIoPmpFlags(lwrrFlag, ioPmpCfg[i]);
        else
            decodePmpFlags(lwrrFlag, nextFlag);
    }
    return LW_OK;
}

LW_STATUS riscvDumpPmp_GA10X(int regions)
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

    CHECK_SUCCESS_OR_RETURN(parsePmpRegisters(pmpFlag, pmpAddr, NULL, LW_FALSE));

    return LW_OK;
}

LW_STATUS riscvDumpIoPmp_GA10X(LwU64 vaOffset)
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
        vaOffset = pRiscvInstance->riscvBase + LW_RISCV_AMAP_PRIV_START;
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

    CHECK_SUCCESS_OR_RETURN(parsePmpRegisters(pmpFlag, pmpAddr, pmpCfg, LW_TRUE));

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

LW_STATUS riscvDumpMpu_GA10X(int regions)
{
    int reg;
    LwU64 smpuidxOrigValue;
    LwU64 smpuctl;
    int numRegions;
    LwBool isLsMode = riscvIsInLs_TU10X();

    if (!riscvIsInIcd_TU10X())
    {
        dprintf("Core must be in ICD to dump MPU mappings.\n");
        return LW_ERR_ILWALID_STATE;
    }

    if (!riscvHasMpuEnabled_GA10X())
    {
        // We may be in M-mode and still want to examine MPU, so don't return here.
        dprintf("MPU is not enabled. VA == PA.\n");
    }

    CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_SMPUCTL, &smpuctl));
    numRegions = DRF_VAL64(_RISCV,_CSR_SMPUCTL,_ENTRY_COUNT, smpuctl);
    if (regions == 0)
        regions = numRegions;

    if (regions > numRegions)
    {
        dprintf("MPU partition has only %d MPU regions.\n", numRegions);
        regions = numRegions;
    }

    CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_SMPUIDX, &smpuidxOrigValue));
    for (reg = 0; reg < regions; ++reg)
    {
        LwU64 smpuva, smpupa, smpurng, smpuattr;
        LwBool isEnabled;
        LwU64 wprid;
        char wpr[16];

        CHECK_SUCCESS_OR_RETURN(riscvIcdWcsr_TU10X(LW_RISCV_CSR_SMPUIDX, reg));
        CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_SMPUVA, &smpuva));
        CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_SMPUATTR, &smpuattr));

        isEnabled = (smpuva & 0x1) ? LW_TRUE : LW_FALSE;

        // Heuristic for empty entry: (Entry is disabled) && (MPU ATTR is zero)
        if ((isEnabled == LW_FALSE) && (smpuattr == 0))
        {
            continue;
        }

        CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_SMPUPA, &smpupa));
        CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_SMPURNG, &smpurng));

        wprid = (smpuattr & DRF_SHIFTMASK64(LW_RISCV_CSR_SMPUATTR_WPR)) >> DRF_SHIFT64(LW_RISCV_CSR_SMPUATTR_WPR);
        sprintf(wpr, "WPRid:%2"LwU64_fmtu" ", wprid);
        dprintf("%2d "LwU64_FMT"-"LwU64_FMT" %s "LwU64_FMT" "LwU64_FMT
                " %s%s user:%c%c%c supervisor:%c%c%c targets: %s%s\n",
                reg,
                DRF_SHIFTMASK64(LW_RISCV_CSR_SMPUVA_BASE) & smpuva,
                (DRF_SHIFTMASK64(LW_RISCV_CSR_SMPUVA_BASE) & smpuva) +
                    (DRF_SHIFTMASK64(LW_RISCV_CSR_SMPURNG_RANGE) & smpurng),
                isEnabled == LW_TRUE ? "->" : "XX",
                DRF_SHIFTMASK64(LW_RISCV_CSR_SMPUPA_BASE) & smpupa,
                DRF_SHIFTMASK64(LW_RISCV_CSR_SMPURNG_RANGE) & smpurng,
                isLsMode ? wpr : "",
                DRF_SHIFTMASK64(LW_RISCV_CSR_SMPUATTR_CACHEABLE) & smpuattr ? "cached  " : "uncached",
                DRF_SHIFTMASK64(LW_RISCV_CSR_SMPUATTR_UR) & smpuattr ? 'r' : '-',
                DRF_SHIFTMASK64(LW_RISCV_CSR_SMPUATTR_UW) & smpuattr ? 'w' : '-',
                DRF_SHIFTMASK64(LW_RISCV_CSR_SMPUATTR_UX) & smpuattr ? 'x' : '-',
                DRF_SHIFTMASK64(LW_RISCV_CSR_SMPUATTR_SR) & smpuattr ? 'r' : '-',
                DRF_SHIFTMASK64(LW_RISCV_CSR_SMPUATTR_SW) & smpuattr ? 'w' : '-',
                DRF_SHIFTMASK64(LW_RISCV_CSR_SMPUATTR_SX) & smpuattr ? 'x' : '-',
                _riscvTarget_GA10X(smpupa, smpurng),
                isEnabled == LW_TRUE ? "" : " **DISABLED**");
    }
    CHECK_SUCCESS_OR_RETURN(riscvIcdWcsr_TU10X(LW_RISCV_CSR_SMPUIDX, smpuidxOrigValue));

    return LW_OK;
}

// doesn't work on GA10x
LW_STATUS riscvGetLsInfo_GA10X(void)
{
    LwBool isLsMode = riscvIsInLs_TU10X();

    dprintf("This doesn't work on GA10x. Use CSRs (m/s)spm and (m/s)rsp to examine core state.\n");

    if (!isLsMode)
    {
        dprintf("Core is not in LS mode.\n");
    }
    else
    {
        dprintf("Core is in LS mode.\n");
    }

    if (!riscvIsInIcd_TU10X())
    {
        dprintf("Core is not in ICD, cannot check RISC-V PRIV privilege level CSRs.\n");
        return LW_OK;
    }
    else
    {
        LwU64 mrsp, mspm, mdbgctl;
        LwU64 m_privmask, m_privlvl;
        LwU64 u_privmask, u_privlvl;
        LwU8 uprivmask[4];
        LwU8 mprivmask[4];
        LwU64 i;
        CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_MRSP, &mrsp));
        CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_MSPM, &mspm));
        CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_MDBGCTL, &mdbgctl));

        m_privmask = (mspm & DRF_SHIFTMASK64(LW_RISCV_CSR_MSPM_MPLM)) >> DRF_SHIFT64(LW_RISCV_CSR_MSPM_MPLM);
        u_privmask = (mspm & DRF_SHIFTMASK64(LW_RISCV_CSR_MSPM_UPLM)) >> DRF_SHIFT64(LW_RISCV_CSR_MSPM_UPLM);
        m_privlvl = (mrsp & DRF_SHIFTMASK64(LW_RISCV_CSR_MRSP_MRPL)) >> DRF_SHIFT64(LW_RISCV_CSR_MRSP_MRPL);
        u_privlvl = (mrsp & DRF_SHIFTMASK64(LW_RISCV_CSR_MRSP_URPL)) >> DRF_SHIFT64(LW_RISCV_CSR_MRSP_URPL);
        for (i = 0; i < 4; ++i)
        {
            uprivmask[i] = u_privmask & (1ULL<<i) ? 'y' : 'n';
            mprivmask[i] = m_privmask & (1ULL<<i) ? 'y' : 'n';
        }
        dprintf("Core U PRIV level mask: L0:%c L1:%c L2:%c L3:%c\n",
                uprivmask[0], uprivmask[1], uprivmask[2], uprivmask[3]);
        dprintf("Core M PRIV level mask: L0:%c L1:%c L2:%c L3:%c\n",
                mprivmask[0], mprivmask[1], mprivmask[2], mprivmask[3]);
        dprintf("Core PRIV accesses in U mode: L%"LwU64_fmtu" / M mode: L%"LwU64_fmtu" / ICD: %s\n",
                u_privlvl, m_privlvl, (mdbgctl & DRF_SHIFTMASK64(LW_RISCV_CSR_MDBGCTL_ICDPL)) == 0 ? "L0" : "Same as core");
    }
    return LW_OK;
}

LW_STATUS riscvDumpBreakpoint_GA10X(int regions)
{
    LwBool dumpAll = LW_FALSE;
    LwU64 tselectOrigValue;
    int reg;

    if (!riscvIsInIcd_TU10X())
    {
        dprintf("Core must be in ICD to dump breakpoints.\n");
        return LW_ERR_ILWALID_STATE;
    }

    if (regions == 0)
    {
        regions = TRIGGERS_MAX;
    }

    if (regions > TRIGGERS_MAX)
    {
        dumpAll = LW_TRUE;
        dprintf("RISC-V core has only %d triggers.\n", TRIGGERS_MAX);
        regions = TRIGGERS_MAX;
    }

    CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_TSELECT, &tselectOrigValue));
    for (reg = 0; reg < regions; ++reg)
    {
        LwU64 bp_flags, bp_va;
        LwBool isEnabled = LW_FALSE;

        CHECK_SUCCESS_OR_RETURN(riscvIcdWcsr_TU10X(LW_RISCV_CSR_TSELECT, reg));
        CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_TDATA1, &bp_flags));
        CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_TDATA2, &bp_va));

        isEnabled = riscvCheckBreakpointFlagIsEnabled_GA10X(bp_flags) ? LW_TRUE : LW_FALSE;

        if (dumpAll || isEnabled) // Filter by enabled breakpoints.
        {
            dprintf("%2d "LwU64_FMT" mode:%c%c%c priv:%c%c%c action:%s\n",
                    reg, bp_va,
                    DRF_VAL64(_RISCV, _CSR_TDATA1, _LOAD, bp_flags) ? 'L' : '-',
                    DRF_VAL64(_RISCV, _CSR_TDATA1, _STORE, bp_flags) ? 'S' : '-',
                    DRF_VAL64(_RISCV, _CSR_TDATA1, _EXELWTE, bp_flags) ? 'X' : '-',
                    FLD_TEST_DRF64(_RISCV, _CSR_TDATA1, _U, _ENABLE, bp_flags) ? 'U' : '-',
                    FLD_TEST_DRF64(_RISCV, _CSR_TDATA1, _S, _ENABLE, bp_flags) ? 'S' : '-',
                    FLD_TEST_DRF64(_RISCV, _CSR_TDATA1, _M, _ENABLE, bp_flags) ? 'M' : '-',
                    isEnabled ? ((FLD_TEST_DRF64(_RISCV, _CSR_TDATA1, _ACTION, _ICD, bp_flags)) ?
                                 "ICD halt" : "Exception") : "Invalid");
        }
    }
    CHECK_SUCCESS_OR_RETURN(riscvIcdWcsr_TU10X(LW_RISCV_CSR_TSELECT, tselectOrigValue));

    return LW_OK;
}

LwU64 riscvDefaultBpFlags_GA10X(void)
{
    return (DRF_DEF64(_RISCV, _CSR_TDATA1, _ACTION, _ICD) |
            DRF_SHIFTMASK64(LW_RISCV_CSR_TDATA1_EXELWTE)  |
            DRF_DEF64(_RISCV, _CSR_TDATA1, _M, _ENABLE)   |
            DRF_DEF64(_RISCV, _CSR_TDATA1, _S, _ENABLE)   |
            DRF_DEF64(_RISCV, _CSR_TDATA1, _U, _ENABLE));
}

LwBool riscvCheckBreakpointFlagIsEnabled_GA10X(LwU64 bp_flags)
{
    // Flags: one of (L / S / X) + one of (U / S / M).
    return ((bp_flags & (DRF_SHIFTMASK64(LW_RISCV_CSR_TDATA1_LOAD)  |
                         DRF_SHIFTMASK64(LW_RISCV_CSR_TDATA1_STORE) |
                         DRF_SHIFTMASK64(LW_RISCV_CSR_TDATA1_EXELWTE))) &&
            (bp_flags & (DRF_DEF64(_RISCV, _CSR_TDATA1, _M, _ENABLE) |
                         DRF_DEF64(_RISCV, _CSR_TDATA1, _S, _ENABLE) |
                         DRF_DEF64(_RISCV, _CSR_TDATA1, _U, _ENABLE))));
}

/*
 * Note on register mappings for RISC-V GDB
 * The way registers are organized is as follows:
 * - GDB keeps all registers (including CSR) in single address space
 * - First 32 registers are GPR (X0-X31)
 * - Register 33 is PC
 * - Another 32 registers are floating point registers (that we don't have)
 * - At the end there are CSR, their address is offset, so CSR0 is mapped as
 *   register 65, CSR1 as 66 and so on.
 */

LW_STATUS riscvRegReadGdb_GA10X(unsigned reg, LwU64 *pValue)
{
    if (reg < 32) // GPR
    {
        return riscvIcdRReg_TU10X(reg, pValue);
    } else if (reg < 33) // PC
    {
        return riscvIcdRPc_TU10X(pValue);
    } else if (reg < 65) // fp
    {
        LwU32 r;
        LW_STATUS ret = riscvIcdRFReg_GA10X((reg - 33) & 0xFFFF, &r);
        *pValue = r;
        return ret;
    } else // CSR
    {
        return riscvIcdRcsr_TU10X((reg - 65) & 0xFFFF, pValue);
    }
}

LW_STATUS riscvRegWriteGdb_GA10X(unsigned reg, LwU64 value)
{
    if (reg < 32) // GPR
    {
        return riscvIcdWReg_TU10X(reg, value);
    } else if (reg < 33) // PC
    {
        return riscvIcdJump(value);
    } else if (reg < 65) // fp
    {
        return riscvIcdWFReg_GA10X((reg - 33) & 0xFFFF, (LwU32)value);
    } else // CSR
    {
        return riscvIcdWcsr_TU10X((reg - 65) & 0xFFFF, value);
    }
}


static int _inRange(LwU64 min, LwU64 max, LwU64 addr, LwU64 size)
{
    if (!size)
        return 0;
    if ((addr + size) < min || addr >= max)
        return 0;
    return 1;
}

const char *_riscvTarget_GA10X(LwU64 base, LwU64 size)
{
    static char buf[256];

    if (!size)
        return strcpy(buf, "Unknown");

    buf[0]=0;
    if (_inRange((LW_RISCV_AMAP_IROM_START),
                 (LW_RISCV_AMAP_IROM_START) + (LW_RISCV_AMAP_IROM_SIZE), base, size))
        strcat(buf, "IROM ");
    if (_inRange((LW_RISCV_AMAP_IMEM_START),
                 (LW_RISCV_AMAP_IMEM_START) + (LW_RISCV_AMAP_IMEM_SIZE), base, size))
        strcat(buf, "ITCM ");
    if (_inRange((LW_RISCV_AMAP_DMEM_START),
                 (LW_RISCV_AMAP_DMEM_START) + (LW_RISCV_AMAP_DMEM_SIZE), base, size))
        strcat(buf, "DTCM ");
    if (_inRange((LW_RISCV_AMAP_EMEM_START),
                 (LW_RISCV_AMAP_EMEM_START) + (LW_RISCV_AMAP_EMEM_SIZE), base, size))
        strcat(buf, "EMEM ");
    if (_inRange((LW_RISCV_AMAP_PRIV_START),
                 (LW_RISCV_AMAP_PRIV_START) + (LW_RISCV_AMAP_PRIV_SIZE), base, size))
        strcat(buf, "PRIV ");
    if (_inRange((LW_RISCV_AMAP_FBGPA_START),
                 (LW_RISCV_AMAP_FBGPA_START) + (LW_RISCV_AMAP_FBGPA_SIZE), base, size))
        strcat(buf, "FBGPA ");
    if (_inRange((LW_RISCV_AMAP_SYSGPA_START),
                 (LW_RISCV_AMAP_SYSGPA_START) + (LW_RISCV_AMAP_SYSGPA_SIZE), base, size))
        strcat(buf, "SYSGPA ");
    if (_inRange((LW_RISCV_AMAP_GVA_START),
                 (LW_RISCV_AMAP_GVA_START) + (LW_RISCV_AMAP_GVA_SIZE), base, size))
        strcat(buf, "GVA ");

    return buf;
}

/*
 * Decode symbolic name of CSR. Returns -1 if CSR not found.
 */
LwS16 riscvDecodeCsr_GA10X(const char *name, size_t nameLen)
{
    const struct NamedCsr *pCsr = &_csrs[0];
    size_t nLen;
    unsigned long index = 0;

    if (!name)
        return -1;

    if (nameLen)
        nLen = nameLen;
    else
        nLen = strlen(name);

    if (!nLen)
        return -1;

    // check if this is an indexed CSR
    if (name[nLen-1] == ')') {
        const char *beginIndex = name + nLen - 2;
        // find matching '('
        for (beginIndex = name + nLen - 2; beginIndex > name; beginIndex--) {
            if (*beginIndex == '(') {
                nLen = beginIndex - name; // exclude (index) from name match
                beginIndex++; // advance past '(' to start of index
                index = strtoul(beginIndex, NULL, 0);
                break;
            }
        }
    }

    if (index > 0xFFF) {
        dprintf("CSR index out of bounds.\n");
        return -1;
    }

    while (pCsr->name)
    {
        if (!strncasecmp(pCsr->name, name, nLen))
            return pCsr->address + (LwU16)index;
        pCsr++;
    }

    return -1;
}
