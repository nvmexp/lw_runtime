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

#include "turing/tu102/dev_riscv_pri.h"

#include "g_riscv_private.h"

LW_STATUS _icdRead_TU10X(ICD_REGS reg, LwU64 *pValue, LwBool b32BitAccess)
{
    if (reg > _ICD_END || !pValue)
        return LW_ERR_ILWALID_ARGUMENT;

    switch (reg)
    {
        case ICD_CMD:
            *pValue = bar0Read(LW_PRISCV_RISCV_ICD_CMD + (reg << 2));
            break;
        case ICD_ADDR:
        case ICD_RDATA:
        case ICD_WDATA:
            *pValue = bar0Read(LW_PRISCV_RISCV_ICD_CMD + (reg << 2));
            if (!b32BitAccess)
                *pValue |= (((LwU64)bar0Read(LW_PRISCV_RISCV_ICD_CMD + ((reg + 3) << 2))) << 32);
            break;
        default:
            return LW_ERR_ILWALID_ARGUMENT;
    }
    return LW_OK;
}

LW_STATUS _icdWrite_TU10X(ICD_REGS reg, LwU64 value, LwBool b32BitAccess)
{
    if (reg > _ICD_END)
        return LW_ERR_ILWALID_ARGUMENT;

    switch (reg)
    {
        case ICD_CMD:
            bar0Write(LW_PRISCV_RISCV_ICD_CMD + (reg << 2), LwU64_LO32(value));
            break;
        case ICD_ADDR:
        case ICD_WDATA:
        case ICD_RDATA:
            bar0Write(LW_PRISCV_RISCV_ICD_CMD + (reg << 2), LwU64_LO32(value));
            if (!b32BitAccess)
                bar0Write(LW_PRISCV_RISCV_ICD_CMD + ((reg + 3) << 2), LwU64_HI32(value));
            break;
        default:
            return LW_ERR_ILWALID_ARGUMENT;
    }

    return LW_OK;
}

LW_STATUS _icdWaitForCompletion_TU10X(void)
{
    LwU32 status;
    int timeout = RISCV_ICD_TIMEOUT_MS;

    do
    {
        CHECK_SUCCESS_OR_RETURN(_icdRead32(ICD_CMD, &status));

        //ERROR bit is set
        if (FLD_TEST_DRF(_PRISCV_RISCV, _ICD_CMD, _ERROR, _TRUE, status))
        {
            dprintf("Command failed: %x (cmd=%hhu)\n", status,
                    (LwU8)DRF_VAL(_PRISCV_RISCV, _ICD_CMD, _OPC, status));
            return LW_ERR_GENERIC;
        }

        if (FLD_TEST_DRF(_PRISCV_RISCV, _ICD_CMD, _BUSY, _FALSE, status))
            break;

        timeout--;
        if (timeout == 0)
        {
            dprintf("Command timeout: %x (cmd=%hhu)\n", status,
                    (LwU8)DRF_VAL(_PRISCV_RISCV, _ICD_CMD, _OPC, status));
            return LW_ERR_TIMEOUT;
        }

        _icdDelay();
    } while FLD_TEST_DRF(_PRISCV_RISCV, _ICD_CMD, _BUSY, _TRUE, status);

    return LW_OK;
}

LW_STATUS riscvIcdRReg_TU10X(unsigned reg, LwU64 *pValue)
{
    LW_STATUS ret;

    if (reg > 31)
    {
        dprintf("No such register: x%d\n", reg);
        return LW_ERR_ILWALID_ARGUMENT;
    }

    if (!pValue)
    {
        dprintf("Invalid value.\n");
        return LW_ERR_ILWALID_ARGUMENT;
    }

    ret = _icdWriteCommand(CMD_RREG | CMD_WIDTH(ICD_WIDTH_64) | CMD_REG(reg));
    if (ret)
    {
        dprintf("Failed to read register x%d.\n", reg);
        return ret;
    }

    TGT_DEBUG("CMD_RREG(%d)\n", reg);

    return _icdRead64(ICD_RDATA, pValue);
}

LW_STATUS riscvIcdWReg_TU10X(unsigned reg, LwU64 value)
{
    CHECK_TARGET_IS_HALTED_OR_RETURN;

    if (reg > 31)
    {
        dprintf("No such register: x%d\n", reg);
        return LW_ERR_ILWALID_ARGUMENT;
    }

    CHECK_SUCCESS_OR_RETURN(_icdWrite64(ICD_WDATA, value));

    TGT_DEBUG("CMD_WREG(%d)="LwU64_FMT"\n", reg, value);

    return _icdWriteCommand(CMD_WREG | CMD_WIDTH(ICD_WIDTH_64) | CMD_REG(reg));
}

LW_STATUS _riscvIcdRdmInt_TU10X(LwU64 address, void *pValue, ICD_ACCESS_WIDTH width, LwBool bCached)
{
    LwU64 rv;

    if (!pValue)
        return LW_ERR_ILWALID_ARGUMENT;

    if (!bCached)
        CHECK_TARGET_IS_HALTED_OR_RETURN;

    CHECK_SUCCESS_OR_RETURN(_icdWriteAddress(address, bCached));

    TGT_DEBUG("CMD_RDM[%d]\n", address);

    CHECK_SUCCESS_OR_RETURN(_icdWriteCommand(CMD_RDM | CMD_WIDTH(width) | CMD_PARM(CMD_PARM_DM_ACCESS_VA)));

    if (width == ICD_WIDTH_64)
    {
        CHECK_SUCCESS_OR_RETURN(_icdRead64(ICD_RDATA, &rv));
    }
    else
    {
        LwU32 rv32;
        CHECK_SUCCESS_OR_RETURN(_icdRead32(ICD_RDATA, &rv32));
        rv = rv32;
    }

    switch(width)
    {
        case ICD_WIDTH_8:
            *((LwU8*)pValue) = rv & 0xFF;
            break;
        case ICD_WIDTH_16:
            *((LwU16*)pValue) = rv & 0xFFFF;
            break;
        case ICD_WIDTH_32:
            *((LwU32*)pValue) = rv & 0xFFFFFFFF;
            break;
        case ICD_WIDTH_64:
            *((LwU64*)pValue) = rv;
            break;
        default:
            return LW_ERR_ILWALID_ARGUMENT;
    }

    return LW_OK;
}

LW_STATUS _riscvIcdWdmInt_TU10X(LwU64 address, LwU64 value, ICD_ACCESS_WIDTH width, LwBool bCached)
{
    if (!bCached)
        CHECK_TARGET_IS_HALTED_OR_RETURN;

    CHECK_SUCCESS_OR_RETURN(_icdWriteAddress(address, bCached));
    CHECK_SUCCESS_OR_RETURN(_icdWriteWdata(value, width, bCached));

    TGT_DEBUG("CMD_WDM[%d]\n", address);

    return _icdWriteCommand(CMD_WDM | CMD_WIDTH(width) | CMD_PARM(CMD_PARM_DM_ACCESS_VA));
}

LW_STATUS riscvIcdRcm_TU10X(LwU64 address, LwU64 *pValue)
{
    CHECK_TARGET_IS_HALTED_OR_RETURN;
    if (!pValue)
        return LW_ERR_ILWALID_ARGUMENT;

    CHECK_SUCCESS_OR_RETURN(_icdWrite64(ICD_ADDR, address));

    TGT_DEBUG("CMD_RCM[%d]\n", address);

    CHECK_SUCCESS_OR_RETURN(_icdWriteCommand(CMD_RCM | CMD_WIDTH(ICD_WIDTH_64)));

    return _icdRead64(ICD_RDATA, pValue);
}

// CSB?
LW_STATUS riscvIcdWcm_TU10X(LwU64 address, LwU64 value)
{
    CHECK_TARGET_IS_HALTED_OR_RETURN;

    CHECK_SUCCESS_OR_RETURN(_icdWrite64(ICD_ADDR, address));
    CHECK_SUCCESS_OR_RETURN(_icdWrite64(ICD_WDATA, value));

    TGT_DEBUG("CMD_WCM[%d]\n", address);

    return _icdWriteCommand(CMD_WCM | CMD_WIDTH(ICD_WIDTH_64));
}

LW_STATUS riscvIcdRcsr_TU10X(LwU16 address, LwU64 *pValue)
{
    CHECK_TARGET_IS_HALTED_OR_RETURN;
    if (!pValue)
        return LW_ERR_ILWALID_ARGUMENT;

    TGT_DEBUG("CMD_RCSR[%d]\n", address);

    CHECK_SUCCESS_OR_RETURN(_icdWriteCommand(CMD_RCSR | CMD_WIDTH(ICD_WIDTH_64) | CMD_PARM(address)));

    return _icdRead64(ICD_RDATA, pValue);
}

LW_STATUS riscvIcdWcsr_TU10X(LwU16 address, LwU64 value)
{
    CHECK_TARGET_IS_HALTED_OR_RETURN;

    CHECK_SUCCESS_OR_RETURN(_icdWrite64(ICD_WDATA, value));

    TGT_DEBUG("CMD_WCSR[%d]\n", address);

    return _icdWriteCommand(CMD_WCSR | CMD_WIDTH(ICD_WIDTH_64) | CMD_PARM(address));
}

LW_STATUS riscvIcdRPc_TU10X(LwU64 *pValue)
{
    LW_STATUS ret;

    ret = _icdWriteCommand(CMD_RPC | CMD_WIDTH(ICD_WIDTH_64));
    if (ret)
    {
        dprintf("Failed to read PC.\n");
        return ret;
    }

    TGT_DEBUG("CMD_RPC()\n");

    return _icdRead64(ICD_RDATA, pValue);
}

void riscvIcdDumpRegs_TU10X(void)
{
    int i;
    LwU64 reg;
    LW_STATUS ret;

    if (!pRiscv[indexGpu].riscvIsInIcd())
    {
        dprintf("Can't dump GPR and PC - RISC-V is not halted.\n");
        return;
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

    ret = pRiscv[indexGpu].riscvIcdRPc(&reg);
    if (ret == LW_OK)
        dprintf("PC =  %16"LwU64_fmtx"\n", reg);
    else
        dprintf("PC =  ?? Err: %x\n", ret);
}

LW_STATUS _icdReadRstat_TU10X(ICD_RSTAT no, LwU64 *pValue)
{
    LW_STATUS ret = 0;

    ret = _icdWriteCommand(CMD_RSTAT | CMD_REG(no) | CMD_WIDTH(ICD_WIDTH_32));
    if (ret)
        return ret;

    ret = _icdRead64(ICD_RDATA, pValue);

    return ret;
}
