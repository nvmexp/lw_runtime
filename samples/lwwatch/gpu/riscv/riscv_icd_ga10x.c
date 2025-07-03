/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <utils/lwassert.h>
#include <os.h>

#include "riscv_prv.h"

#include "ampere/ga102/dev_riscv_pri.h"
#include "ampere/ga102/dev_pmu_riscv_csr_64.h"

#include "g_riscv_private.h"

LW_STATUS _icdRead_GA10X(ICD_REGS reg, LwU64 *pValue, LwBool b32BitAccess)
{
    if (reg > _ICD_END || !pValue)
        return LW_ERR_ILWALID_ARGUMENT;

    switch (reg)
    {
        case ICD_CMD:
            *pValue = bar0Read(LW_PRISCV_RISCV_ICD_CMD);
            break;
        case ICD_ADDR:
            *pValue = bar0Read(LW_PRISCV_RISCV_ICD_ADDR0);
            if (!b32BitAccess)
                *pValue |= (((LwU64)bar0Read(LW_PRISCV_RISCV_ICD_ADDR1)) << 32);
            break;
        case ICD_RDATA:
            *pValue = bar0Read(LW_PRISCV_RISCV_ICD_RDATA0);
            if (!b32BitAccess)
                *pValue |= (((LwU64)bar0Read(LW_PRISCV_RISCV_ICD_RDATA1)) << 32);
            break;
        case ICD_WDATA:
            *pValue = bar0Read(LW_PRISCV_RISCV_ICD_WDATA0);
            if (!b32BitAccess)
                *pValue |= (((LwU64)bar0Read(LW_PRISCV_RISCV_ICD_WDATA1)) << 32);
            break;
        default:
            return LW_ERR_ILWALID_ARGUMENT;
    }
    return LW_OK;
}

LW_STATUS _icdWrite_GA10X(ICD_REGS reg, LwU64 value, LwBool b32BitAccess)
{
    if (reg > _ICD_END)
        return LW_ERR_ILWALID_ARGUMENT;

    switch (reg)
    {
        case ICD_CMD:
            bar0Write(LW_PRISCV_RISCV_ICD_CMD, LwU64_LO32(value));
            break;
        case ICD_ADDR:
            bar0Write(LW_PRISCV_RISCV_ICD_ADDR0, LwU64_LO32(value));
            if (!b32BitAccess)
                bar0Write(LW_PRISCV_RISCV_ICD_ADDR1, LwU64_HI32(value));
            break;
        case ICD_WDATA:
            bar0Write(LW_PRISCV_RISCV_ICD_WDATA0, LwU64_LO32(value));
            if (!b32BitAccess)
                bar0Write(LW_PRISCV_RISCV_ICD_WDATA1, LwU64_HI32(value));
            break;
        case ICD_RDATA:
            bar0Write(LW_PRISCV_RISCV_ICD_RDATA0, LwU64_LO32(value));
            if (!b32BitAccess)
                bar0Write(LW_PRISCV_RISCV_ICD_RDATA1, LwU64_HI32(value));
            break;
        default:
            return LW_ERR_ILWALID_ARGUMENT;
    }

    return LW_OK;
}

LW_STATUS riscvIcdRFReg_GA10X(unsigned reg, LwU32 *pValue)
{
    LW_STATUS ret;
    LwU64 val64;

    if (reg > 31)
    {
        dprintf("No such register: f%u\n", reg);
    }

    if (!pValue)
    {
        dprintf("Invalid value.\n");
        return LW_ERR_ILWALID_ARGUMENT;
    }

    ret = _icdWriteCommand(CMD_RFREG | CMD_WIDTH(ICD_WIDTH_32) | CMD_REG(reg));
    if (ret) {
        dprintf("Failed to read FPU register f%u.\n", reg);
        return ret;
    }

    TGT_DEBUG("CMD_RFREG(%u)\n", reg);

    ret = _icdRead_GA10X(ICD_RDATA, &val64, LW_TRUE);
    *pValue = LwU64_LO32(val64);
    return ret;
}

LW_STATUS riscvIcdWFReg_GA10X(unsigned reg, LwU32 value)
{
    union {
        LwU32 u32;
        LwF32 f32;
    } val = {value}; // value is assigned to .u32

    CHECK_TARGET_IS_HALTED_OR_RETURN;

    if (reg > 31)
    {
        dprintf("No such register: f%u\n", reg);
        return LW_ERR_ILWALID_ARGUMENT;
    }

    CHECK_SUCCESS_OR_RETURN(_icdWrite_GA10X(ICD_WDATA, val.u32, LW_TRUE));

    TGT_DEBUG("CMD_WFREG(%u)=0x%08x (%f)\n", reg, val.u32, val.f32);

    return _icdWriteCommand(CMD_WFREG | CMD_WIDTH(ICD_WIDTH_32) | CMD_REG(reg));
}

void riscvIcdDumpRegs_GA10X(void)
{
    int i;
    LwU64 reg;
    LwU32 freg;
    LwU64 mstatus = 0;
    LW_STATUS ret;

    if (!pRiscv[indexGpu].riscvIsInIcd())
    {
        dprintf("RISC-V is not halted: cannot dump GPR, FPR, or hi-bits of PC\n");
        reg = bar0Read(LW_PRISCV_RISCV_RPC);
        dprintf("PC  = --------%08x\n", LwU64_LO32(reg));
        return;
    }
    else
    {
        ret = pRiscv[indexGpu].riscvIcdRPc(&reg);
        if (ret == LW_OK)
            dprintf("PC  = %16"LwU64_fmtx"\n", reg);
        else
            dprintf("PC  = ?? Err: %x\n", ret);
    }

    for (i=0; i<32; ++i)
    {
        ret = riscvIcdRReg_TU10X(i, &reg);
        if (ret == LW_OK)
            dprintf("%s = %016"LwU64_fmtx" ", reg_name[i], reg);
        else
            dprintf("%s = ?? ERR: %x ", reg_name[i], ret);

        if (i % 4 == 3)
            dprintf("\n");
    }

    // check if FPU is enabled
    if (riscvIcdRcsr_TU10X(LW_RISCV_CSR_MSTATUS, &mstatus) != LW_OK)
        return;

    if (!FLD_TEST_DRF(_RISCV, _CSR_MSTATUS, _FS, _OFF, mstatus))
    {
        LwU64 fcsr;
        if (riscvIcdRcsr_TU10X(LW_RISCV_CSR_FCSR, &fcsr) != LW_OK)
            return;

        dprintf("fcsr = "LwU64_FMT"\n", fcsr);
        for (i=0; i<32; ++i)
        {
            float f;
            char fstring[20];
            ret = riscvIcdRFReg_GA10X(i, &freg);
            if (ret == LW_OK) {
                f = *((float *)(&freg));
                // special case printing registers that are actually equal to 0.0
                sprintf(fstring, "%16.8f", f);
                dprintf("%s = %s ", fpreg_name[i], ((f == 0.0)? "               0" : fstring));
            }
            else
                dprintf("%s = ?? ERR: %x ", fpreg_name[i], ret);

            if (i % 4 == 3)
                dprintf("\n");
        }
    }
    else
        dprintf("FPU is disabled, cannot dump registers.\n");

}
