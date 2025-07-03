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

#include "turing/tu102/dev_falcon_v4.h"
#include "turing/tu102/dev_riscv_pri.h"
#include "turing/tu102/dev_gsp_riscv_csr_64.h"
#include "turing/tu102/lw_gsp_riscv_address_map.h"

#include "riscv_regs.h"
#include "riscv_csrs_tu10x.h"

#include "riscv_dbgint.h"
#include "g_riscv_private.h"
#include "hal.h"

// Hack to determine number of MPU regions from hwref manuals
// 64 on TU10X + GA100
#define RISCV_MPU_REGION_NUM DRF_MASK(LW_RISCV_CSR_MMPUIDX_INDEX) + 1

LwBool riscvIsSupported_TU10X(void)
{
    return LW_TRUE;
}

LwBool riscvIsActive_TU10X(void)
{
    LwU64 r;

    r = bar0Read(LW_PRISCV_RISCV_CORE_SWITCH_RISCV_STATUS);
    return FLD_TEST_DRF(_PRISCV_RISCV, _CORE_SWITCH_RISCV_STATUS, _ACTIVE_STAT, _ACTIVE, r);
}

LwBool riscvIsInIcd_TU10X(void)
{
    LwU64 r;
    int ret;

    if (!riscvIsActive())
        return LW_FALSE;

    ret = riscvIcdReadRstat(4, &r);
    if (ret != LW_OK)
    {
        dprintf("Failed reading RSTAT4\n");
        return LW_FALSE;
    }

    return FLD_TEST_DRF(_PRISCV_RISCV,_ICD_RDATA0_RSTAT4,_ICD_STATE, _ICD, r);
}

LwBool riscvIsInLs_TU10X(void)
{
    LwU32 sctl = bar0ReadLegacy(LW_PFALCON_FALCON_SCTL);
    LwBool bAllowedPrivilege = LW_FALSE;

    bAllowedPrivilege = (FLD_TEST_DRF_NUM(_PFALCON_FALCON,_SCTL,_LSMODE_LEVEL, 1, sctl) ||
                         FLD_TEST_DRF_NUM(_PFALCON_FALCON,_SCTL,_LSMODE_LEVEL, 2, sctl));

    return (FLD_TEST_DRF(_PFALCON_FALCON,_SCTL,_LSMODE, _TRUE, sctl) &&
            FLD_TEST_DRF(_PFALCON_FALCON,_SCTL,_HSMODE, _FALSE, sctl) &&
            bAllowedPrivilege);
}

LwBool riscvHasMpuEnabled_TU10X(void)
{
    LwU64 mstatus;
    int ret;

    ret = riscvIcdRcsr_TU10X(LW_RISCV_CSR_MSTATUS, &mstatus);
    if (ret != LW_OK)
    {
        dprintf("Failed reading mstatus.\n");
        return LW_FALSE;
    }

    // TODO: `_VM_MPU` is not in the hwref. The options for `_MSTATUS_VM` are `_VM_BARE`=0x0 or `_VM_LWGPU`=0x3
    return ((mstatus >> DRF_SHIFT64(LW_RISCV_CSR_MSTATUS_VM)) & DRF_MASK64(LW_RISCV_CSR_MSTATUS_VM)) ==
        LW_RISCV_CSR_MSTATUS_VM_MPU;
}

LW_STATUS riscvDumpBreakpoint_TU10X(int regions)
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

        isEnabled = riscvCheckBreakpointFlagIsEnabled_TU10X(bp_flags) ? LW_TRUE : LW_FALSE;

        if (dumpAll || isEnabled) // Filter by enabled breakpoints.
        {
            dprintf("%2d "LwU64_FMT" mode:%c%c%c priv:%c%c action:%s\n",
                    reg, bp_va,
                    DRF_VAL64(_RISCV, _CSR_TDATA1, _LOAD, bp_flags) ? 'L' : '-',
                    DRF_VAL64(_RISCV, _CSR_TDATA1, _STORE, bp_flags) ? 'S' : '-',
                    DRF_VAL64(_RISCV, _CSR_TDATA1, _EXELWTE, bp_flags) ? 'X' : '-',
                    FLD_TEST_DRF64(_RISCV, _CSR_TDATA1, _U, _ENABLE, bp_flags) ? 'U' : '-',
                    FLD_TEST_DRF64(_RISCV, _CSR_TDATA1, _M, _ENABLE, bp_flags) ? 'M' : '-',
                    isEnabled ? ((FLD_TEST_DRF64(_RISCV, _CSR_TDATA1, _ACTION, _ICD, bp_flags)) ?
                                 "ICD halt" : "Exception") : "Invalid");
        }
    }
    CHECK_SUCCESS_OR_RETURN(riscvIcdWcsr_TU10X(LW_RISCV_CSR_TSELECT, tselectOrigValue));

    return LW_OK;
}

LwU64 riscvDefaultBpFlags_TU10X(void)
{
    return (DRF_DEF64(_RISCV, _CSR_TDATA1, _ACTION, _ICD) |
            DRF_SHIFTMASK64(LW_RISCV_CSR_TDATA1_EXELWTE)  |
            DRF_DEF64(_RISCV, _CSR_TDATA1, _M, _ENABLE)   |
            DRF_DEF64(_RISCV, _CSR_TDATA1, _U, _ENABLE));
}

LW_STATUS riscvSetBreakpoint_TU10X(int index, LwU64 addr, LwU64 flags)
{
    LwU64 tselectOrigValue;

    if (index >= TRIGGERS_MAX)
        return LW_ERR_ILWALID_ARGUMENT;

    if (riscvCheckBreakpointFlagIsEnabled_TU10X(flags))
    {
        CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_TSELECT, &tselectOrigValue));

        CHECK_SUCCESS_OR_RETURN(riscvIcdWcsr_TU10X(LW_RISCV_CSR_TSELECT, index));
        CHECK_SUCCESS_OR_RETURN(riscvIcdWcsr_TU10X(LW_RISCV_CSR_TDATA1, flags));
        CHECK_SUCCESS_OR_RETURN(riscvIcdWcsr_TU10X(LW_RISCV_CSR_TDATA2, addr));

        CHECK_SUCCESS_OR_RETURN(riscvIcdWcsr_TU10X(LW_RISCV_CSR_TSELECT, tselectOrigValue));
    }
    else
        return LW_ERR_ILWALID_ARGUMENT;

    return LW_OK;
}

LW_STATUS riscvClearBreakpoint_TU10X(int index)
{
    LwU64 tselectOrigValue;

    if (index >= TRIGGERS_MAX)
        return LW_ERR_ILWALID_ARGUMENT;

    CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_TSELECT, &tselectOrigValue));

    CHECK_SUCCESS_OR_RETURN(riscvIcdWcsr_TU10X(LW_RISCV_CSR_TSELECT, index));
    CHECK_SUCCESS_OR_RETURN(riscvIcdWcsr_TU10X(LW_RISCV_CSR_TDATA1, 0));
    CHECK_SUCCESS_OR_RETURN(riscvIcdWcsr_TU10X(LW_RISCV_CSR_TDATA2, 0));

    CHECK_SUCCESS_OR_RETURN(riscvIcdWcsr_TU10X(LW_RISCV_CSR_TSELECT, tselectOrigValue));

    return LW_OK;
}

LW_STATUS riscvDumpMpu_TU10X(int regions)
{
    int reg;
    LwU64 mmpuidxOrigValue;
    LwBool isLsMode = riscvIsInLs_TU10X();

    if (!riscvIsInIcd_TU10X())
    {
        dprintf("Core must be in ICD to dump MPU mappings.\n");
        return LW_ERR_ILWALID_STATE;
    }

    if (!riscvHasMpuEnabled_TU10X())
    {
        dprintf("MPU is not enabled. VA == PA.\n");
        return LW_OK;
    }

    if (regions == 0)
        regions = RISCV_MPU_REGION_NUM;

    if (regions > RISCV_MPU_REGION_NUM)
    {
        dprintf("RISC-V core has only %d MPU regions.\n", RISCV_MPU_REGION_NUM);
        regions = RISCV_MPU_REGION_NUM;
    }

    CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_MMPUIDX, &mmpuidxOrigValue));
    for (reg = 0; reg < regions; ++reg)
    {
        LwU64 mmpuva, mmpupa, mmpurng, mmpuattr;
        LwBool isEnabled;
        LwU64 wprid;
        char wpr[16];

        CHECK_SUCCESS_OR_RETURN(riscvIcdWcsr_TU10X(LW_RISCV_CSR_MMPUIDX, reg));
        CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_MMPUVA, &mmpuva));
        CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_MMPUATTR, &mmpuattr));

        isEnabled = DRF_VAL64(_RISCV, _CSR_MMPUVA, _VLD, mmpuva) ? LW_TRUE : LW_FALSE;

        // Heuristic for empty entry: (Entry is disabled) && (MPU ATTR is zero)
        if ((isEnabled == LW_FALSE) && (mmpuattr == 0))
        {
            continue;
        }

        CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_MMPUPA, &mmpupa));
        CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_MMPURNG, &mmpurng));

        wprid = (mmpuattr & DRF_SHIFTMASK64(LW_RISCV_CSR_MMPUATTR_WPR)) >> DRF_SHIFT64(LW_RISCV_CSR_MMPUATTR_WPR);
        sprintf(wpr, "WPRid:%2"LwU64_fmtu" ", wprid);
        dprintf("%2d "LwU64_FMT"-"LwU64_FMT" %s "LwU64_FMT" "LwU64_FMT
                " %s%s user:%c%c%c machine:%c%c%c targets: %s%s\n",
                reg,
                DRF_SHIFTMASK64(LW_RISCV_CSR_MMPUVA_BASE) & mmpuva,
                (DRF_SHIFTMASK64(LW_RISCV_CSR_MMPUVA_BASE) & mmpuva) +
                    (DRF_SHIFTMASK64(LW_RISCV_CSR_MMPURNG_RANGE) & mmpurng),
                isEnabled == LW_TRUE ? "->" : "XX",
                DRF_SHIFTMASK64(LW_RISCV_CSR_MMPUPA_BASE) & mmpupa,
                DRF_SHIFTMASK64(LW_RISCV_CSR_MMPURNG_RANGE) & mmpurng,
                isLsMode ? wpr : "",
                DRF_SHIFTMASK64(LW_RISCV_CSR_MMPUATTR_CACHEABLE) & mmpuattr ? "cached  " : "uncached",
                DRF_SHIFTMASK64(LW_RISCV_CSR_MMPUATTR_UR) & mmpuattr ? 'r' : '-',
                DRF_SHIFTMASK64(LW_RISCV_CSR_MMPUATTR_UW) & mmpuattr ? 'w' : '-',
                DRF_SHIFTMASK64(LW_RISCV_CSR_MMPUATTR_UX) & mmpuattr ? 'x' : '-',
                DRF_SHIFTMASK64(LW_RISCV_CSR_MMPUATTR_MR) & mmpuattr ? 'r' : '-',
                DRF_SHIFTMASK64(LW_RISCV_CSR_MMPUATTR_MW) & mmpuattr ? 'w' : '-',
                DRF_SHIFTMASK64(LW_RISCV_CSR_MMPUATTR_MX) & mmpuattr ? 'x' : '-',
                _riscvTarget_TU10X(mmpupa, mmpurng),
                isEnabled == LW_TRUE ? "" : " **DISABLED**");
    }
    CHECK_SUCCESS_OR_RETURN(riscvIcdWcsr_TU10X(LW_RISCV_CSR_MMPUIDX, mmpuidxOrigValue));

    return LW_OK;
}

LW_STATUS riscvGetLsInfo_TU10X(void)
{
    LwBool isLsMode = riscvIsInLs_TU10X();

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
        LwU64 mrsp, mspm;
        LwU64 m_privmask, m_privlvl;
        LwU64 u_privmask, u_privlvl;
        LwU8 uprivmask[4];
        LwU8 mprivmask[4];
        LwU64 i;
        CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_MRSP, &mrsp));
        CHECK_SUCCESS_OR_RETURN(riscvIcdRcsr_TU10X(LW_RISCV_CSR_MSPM, &mspm));

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
                u_privlvl, m_privlvl, (FLD_TEST_DRF64(_RISCV, _CSR_MRSP, _ICD_PL, _USE_PL0, mrsp)) ?
                "L0" : "Same as core");

    }
    return LW_OK;
}

LwBool riscvCheckBreakpointFlagIsEnabled_TU10X(LwU64 bp_flags)
{
    // Flags: one of (L / S / X) + one of (U / M).
    return ((bp_flags & (DRF_SHIFTMASK64(LW_RISCV_CSR_TDATA1_LOAD)  |
                         DRF_SHIFTMASK64(LW_RISCV_CSR_TDATA1_STORE) |
                         DRF_SHIFTMASK64(LW_RISCV_CSR_TDATA1_EXELWTE))) &&
            (bp_flags & (DRF_DEF64(_RISCV, _CSR_TDATA1, _M, _ENABLE) |
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

LW_STATUS riscvRegReadGdb_TU10X(unsigned reg, LwU64 *pValue)
{
    if (reg < 32) // GPR
    {
        return riscvIcdRReg_TU10X(reg, pValue);
    } else if (reg < 33) // PC
    {
        return riscvIcdRPc_TU10X(pValue);
    } else if (reg < 65) // fp
    {
        dprintf("Floating point registers not implemented.\n");
        return LW_ERR_NOT_SUPPORTED;
    } else // CSR
    {
        return riscvIcdRcsr_TU10X((reg - 65) & 0xFFFF, pValue);
    }
}

LW_STATUS riscvRegWriteGdb_TU10X(unsigned reg, LwU64 value)
{
    if (reg < 32) // GPR
    {
        return riscvIcdWReg_TU10X(reg, value);
    } else if (reg < 33) // PC
    {
        return riscvIcdJump(value);
    } else if (reg < 65) // fp
    {
        dprintf("Floating point registers not implemented.\n");
        return LW_ERR_NOT_SUPPORTED;
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

const char *_riscvTarget_TU10X(LwU64 base, LwU64 size)
{
    static char buf[256];

    if (!size)
        return strcpy(buf, "Unknown");

    buf[0]=0;

    if (_inRange((LW_RISCV_AMAP_IMEM_START), (LW_RISCV_AMAP_IMEM_START) +
                 (LW_RISCV_AMAP_IMEM_SIZE), base, size))
        strcat(buf, "ITCM ");
    if (_inRange((LW_RISCV_AMAP_DMEM_START), (LW_RISCV_AMAP_DMEM_START) +
                 (LW_RISCV_AMAP_DMEM_SIZE), base, size))
        strcat(buf, "DTCM ");
    if (_inRange((LW_RISCV_AMAP_EMEM_START), (LW_RISCV_AMAP_EMEM_START) +
                 (LW_RISCV_AMAP_EMEM_SIZE), base, size))
        strcat(buf, "EMEM ");
    if (_inRange((LW_RISCV_AMAP_PRIV_START), (LW_RISCV_AMAP_PRIV_START) +
                 (LW_RISCV_AMAP_PRIV_SIZE), base, size))
        strcat(buf, "PRIV ");
    if (_inRange((LW_RISCV_AMAP_FBGPA_START), (LW_RISCV_AMAP_FBGPA_START) +
                 (LW_RISCV_AMAP_FBGPA_SIZE), base, size))
        strcat(buf, "FBGPA ");
    if (_inRange((LW_RISCV_AMAP_SYSGPA_START), (LW_RISCV_AMAP_SYSGPA_START) +
                 (LW_RISCV_AMAP_SYSGPA_SIZE), base, size))
        strcat(buf, "SYSGPA ");
    if (_inRange((LW_RISCV_AMAP_GVA_START), (LW_RISCV_AMAP_GVA_START) +
                 (LW_RISCV_AMAP_GVA_SIZE), base, size))
        strcat(buf, "GVA ");

    return buf;
}

/*
 * Decode symbolic name of CSR. Returns -1 if CSR not found.
 */
LwS16 riscvDecodeCsr_TU10X(const char *name, size_t nameLen)
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
