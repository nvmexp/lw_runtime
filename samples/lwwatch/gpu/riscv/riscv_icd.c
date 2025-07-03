/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include <utils/lwassert.h>
#include <os.h>

#include "riscv_prv.h"

#include "hal.h"
#include "g_riscv_hal.h"

const char reg_name[][4] =
{ "x0 ", "ra ", "sp ", "gp ", "tp ", "t0 ", "t1 ", "t2 ",
  "s0 ", "s1 ", "a0 ", "a1 ", "a2 ", "a3 ", "a4 ", "a5 ",
  "a6 ", "a7 ", "s2 ", "s3 ", "s4 ", "s5 ", "s6 ", "s7 ",
  "s8 ", "s9 ", "s10", "s11", "t3 ", "t4 ", "t5 ", "t6 " };

const char fpreg_name[][5] =
{"ft0 ","ft1 ","ft2 ","ft3 ","ft4 ","ft5 ","ft6 ","ft7 ",
 "fs0 ","fs1 ","fa0 ","fa1 ","fa2 ","fa3 ","fa4 ","fa5 ",
 "fa6 ","fa7 ","fs2 ","fs3 ","fs4 ","fs5 ","fs6 ","fs7 ",
 "fs8 ","fs9 ","fs10","fs11","ft8 ","ft9 ","ft10","ft11" };

void _icdDelay(void)
{
    riscvDelay(1);
}

LW_STATUS _icdRead32(ICD_REGS reg, LwU32 *value)
{
    LwU64 val64;
    LW_STATUS ret;

    ret = pRiscv[indexGpu]._icdRead(reg, &val64, LW_TRUE);
    *value = LwU64_LO32(val64);

    return ret;
}

LW_STATUS _icdRead64(ICD_REGS reg, LwU64 *pValue)
{
    return pRiscv[indexGpu]._icdRead(reg, pValue, LW_FALSE);
}

static LW_STATUS _icdWrite32(ICD_REGS reg, LwU32 value)
{
    return pRiscv[indexGpu]._icdWrite(reg, value, LW_TRUE);
}

LW_STATUS _icdWrite64(ICD_REGS reg, LwU64 value)
{
    return pRiscv[indexGpu]._icdWrite(reg, value, LW_FALSE);
}

// Write ICD command, return true if it was accepted
LW_STATUS _icdWriteCommand(LwU32 cmd)
{
    LW_STATUS ret;
    CHECK_SUCCESS_OR_RETURN(_icdWrite32(ICD_CMD, cmd));
    ret = pRiscv[indexGpu]._icdWaitForCompletion();

    if (ret != LW_OK && pRiscv[indexGpu].riscvIsFbBusy())
        dprintf("FBIF is busy. Core reset is only way to recover if there "
                "was illegal request to FB\n");

    return ret;
}

////////////////////////////////////////////////////////////////////////////////
// EXTERNAL INTERFACE                                                         //
////////////////////////////////////////////////////////////////////////////////

LW_STATUS riscvIcdStop(void)
{
    TGT_DEBUG("CMD_STOP\n");
    return _icdWriteCommand(CMD_STOP);
}

LW_STATUS riscvIcdRun(void)
{
    CHECK_TARGET_IS_HALTED_OR_RETURN;

    TGT_DEBUG("CMD_RUN\n");
    return _icdWriteCommand(CMD_RUN);
}

LW_STATUS riscvIcdJump(LwU64 addr)
{
    CHECK_TARGET_IS_HALTED_OR_RETURN;

    TGT_DEBUG("CMD_JUMP\n");
    CHECK_SUCCESS_OR_RETURN(_icdWrite64(ICD_ADDR, addr));
    return _icdWriteCommand(CMD_JUMP);
}

LW_STATUS riscvIcdStep(void)
{
    CHECK_TARGET_IS_HALTED_OR_RETURN;

    TGT_DEBUG("CMD_STEP\n");
    CHECK_SUCCESS_OR_RETURN(_icdWriteCommand(CMD_STEP));
    return riscvWaitForHalt(RISCV_ICD_TIMEOUT_MS);
}

LW_STATUS riscvIcdSetEmask(LwU64 mask)
{
    CHECK_SUCCESS_OR_RETURN(_icdWrite64(ICD_WDATA, mask));

    TGT_DEBUG("CMD_EMASK(0x"LwU64_FMT")\n", mask);
    return _icdWriteCommand(CMD_EMASK);
}

/*
 * We want to cache parts of rdm/wdm to improve performance on fmodel
 */
static LwU64 _icd_mem_cached_address;
static LwU64 _icd_mem_cached_wdata;
static LwBool _icd_mem_cached_address_valid = LW_FALSE;
static LwBool _icd_mem_cached_wdata_lo_valid = LW_FALSE;
static LwBool _icd_mem_cached_wdata_hi_valid = LW_FALSE;

LW_STATUS _icdWriteAddress(LwU64 address, LwBool bCached)
{
    if (!bCached)
        _icd_mem_cached_address_valid = LW_FALSE;

    if (!_icd_mem_cached_address_valid)
    {
      CHECK_SUCCESS_OR_RETURN(_icdWrite64(ICD_ADDR, address));
    } else
    {
        if (address != _icd_mem_cached_address)
        {
            _icd_mem_cached_address_valid = LW_FALSE;
            if (LwU64_HI32(address) == LwU64_HI32(_icd_mem_cached_address))
                CHECK_SUCCESS_OR_RETURN(_icdWrite32(ICD_ADDR, LwU64_LO32(address)));
            else
                CHECK_SUCCESS_OR_RETURN(_icdWrite64(ICD_ADDR, address));
        }
    }
    _icd_mem_cached_address = address;
    _icd_mem_cached_address_valid = LW_TRUE;

    return LW_OK;
}

LW_STATUS _icdWriteWdata(LwU64 value, ICD_ACCESS_WIDTH width, LwBool bCached)
{
    if (!bCached)
    {
        _icd_mem_cached_wdata_lo_valid = LW_FALSE;
        _icd_mem_cached_wdata_hi_valid = LW_FALSE;
    }
    else
    {
        if (LwU64_LO32(value) != LwU64_LO32(_icd_mem_cached_wdata))
            _icd_mem_cached_wdata_lo_valid = LW_FALSE;

        if (width == ICD_WIDTH_64)
        {
            if (LwU64_HI32(value) != LwU64_HI32(_icd_mem_cached_wdata))
                _icd_mem_cached_wdata_hi_valid  = LW_FALSE;
        }
    }


    if (!_icd_mem_cached_wdata_hi_valid && width == ICD_WIDTH_64)
    {
        CHECK_SUCCESS_OR_RETURN(_icdWrite64(ICD_WDATA, value));
        _icd_mem_cached_wdata_hi_valid = LW_TRUE;
        _icd_mem_cached_wdata_lo_valid = LW_TRUE;
        _icd_mem_cached_wdata = value;
    }
    else if (!_icd_mem_cached_wdata_lo_valid)
    {
        CHECK_SUCCESS_OR_RETURN(_icdWrite32(ICD_WDATA, LwU64_LO32(value)));
        _icd_mem_cached_wdata_lo_valid = LW_TRUE;
        _icd_mem_cached_wdata = (_icd_mem_cached_wdata & (0xFFFFFFFFULL << 32)) | LwU64_LO32(value);
    }
    // else -> cache valid, do nothing
    return LW_OK;
}


LW_STATUS riscvIcdRdm(LwU64 address, void *pValue, ICD_ACCESS_WIDTH width)
{
    return pRiscv[indexGpu]._riscvIcdRdmInt(address, pValue, width, LW_FALSE);
}

LW_STATUS riscvIcdWdm(LwU64 address, LwU64 value, ICD_ACCESS_WIDTH width)
{
    return pRiscv[indexGpu]._riscvIcdWdmInt(address, value, width, LW_FALSE);
}

LW_STATUS riscvIcdRdmFast(LwU64 address, void *pValue, ICD_ACCESS_WIDTH width)
{
    return pRiscv[indexGpu]._riscvIcdRdmInt(address, pValue, width, LW_TRUE);
}

LW_STATUS riscvIcdWdmFast(LwU64 address, LwU64 value, ICD_ACCESS_WIDTH width)
{
    return pRiscv[indexGpu]._riscvIcdWdmInt(address, value, width, LW_TRUE);
}

LW_STATUS riscvIcdReadRstat(ICD_RSTAT no, LwU64 *pValue)
{
    if (no > ICD_RSTAT_END)
    {
        dprintf("Invalid RSTAT requested (%d).\n", no);
    }

    TGT_DEBUG("CMD_READ_RSTAT(%d)\n", no);
    return pRiscv[indexGpu]._icdReadRstat(no, pValue);
}

LW_STATUS riscvIcdCmdSbu(void)
{
    dprintf("CMD_SBU: unsupported\n");
    return LW_ERR_NOT_SUPPORTED;
}
