/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2020 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include <utils/lwassert.h>
#include <lwstatus.h>
#include "print.h"

#include "riscv.h"
#include "riscv_prv.h"

#define MEM_READ    LW_TRUE
#define MEM_WRITE   LW_FALSE

static LW_STATUS _riscvMemTransfer(LwU64 addr, unsigned len, void *pBuffer, LwBool bRead)
{
    LwBool transferCached = LW_FALSE;

    // Check alignment for addr + len
    if ((addr & 0xFFFFFFFF) + len > 0xFFFFFFFF)
    {
        dprintf("Mem transfer crosses memory domains.\n");
        return LW_ERR_ILWALID_ARGUMENT;
    }

    while (len)
    {
        ICD_ACCESS_WIDTH width;
        LwU64 wval;

        if (len >= 8 && ((addr & 0x7) == 0))
        {
            width = ICD_WIDTH_64;
            wval = *(LwU64*)pBuffer; // We can always read buffer, even if it's needed for write
        } else if (len >= 4 && ((addr & 0x3) == 0))
        {
            width = ICD_WIDTH_32;
            wval = *(LwU32*)pBuffer; // We can always read buffer, even if it's needed for write
        } else if (len >= 2 && ((addr & 0x1) == 0))
        {
            width = ICD_WIDTH_16;
            wval = *(LwU16*)pBuffer; // We can always read buffer, even if it's needed for write
        } else
        {
            width = ICD_WIDTH_8;
            wval = *(LwU8*)pBuffer; // We can always read buffer, even if it's needed for write
        }

        //
        // For larger transfers, it is beneficial to remember values written
        // to ICD registers. That way we can decrease number of bar0 accesses
        // significantly.
        // Because core state may change due to various factors (it may be reset
        // by lwwatch, it may reset itself etc.), we always use uncached access
        // for the first word, hoping core will not reset in the middle of
        // transfer.
        //
        if (!transferCached)
        {
            if (bRead)
                CHECK_SUCCESS_OR_RETURN(riscvIcdRdm(addr, pBuffer, width));
            else
                CHECK_SUCCESS_OR_RETURN(riscvIcdWdm(addr, wval, width));
            transferCached = LW_TRUE;
        } else
        {
            if (bRead)
                CHECK_SUCCESS_OR_RETURN(riscvIcdRdmFast(addr, pBuffer, width));
            else
                CHECK_SUCCESS_OR_RETURN(riscvIcdWdmFast(addr, wval, width));
        }

        len -= ICD_ACCESS_WIDTH_TO_BYTES(width);
        addr += ICD_ACCESS_WIDTH_TO_BYTES(width);
        pBuffer = (LwU8*)pBuffer + ICD_ACCESS_WIDTH_TO_BYTES(width);
    }
    return LW_OK;
}

LW_STATUS riscvMemRead(LwU64 addr, unsigned len, void *pBuffer, LwBool bForceIcd)
{
    // For small transfers and when forced use ICD
    // The split is there to avoid bar accesses in MPU check
    LwBool usedIcd = LW_TRUE;
    LW_STATUS retVal;

    if (len <= 32 || bForceIcd)
    {
        usedIcd = LW_TRUE;
    }
    else if (!pRiscv[indexGpu].riscvHasMpuEnabled())
    {
        //
        // For rest, try to use mem backdoor as it's ~2x faster.
        // We can do that only if MPU is disabled, as mem backdoor access physical
        // addresses (contrary to ICD)
        //
        const char *memType = pRiscv[indexGpu]._riscvTarget(addr, len);
        if (strstr(memType, "ITCM"))
        {
            usedIcd = LW_FALSE;
            retVal = pRiscv[indexGpu].riscvImemRead(LwU64_LO32(addr - pRiscvInstance->riscv_imem_start), len, pBuffer, BACKDOOR_NO);
        }
        else if (strstr(memType, "DTCM"))
        {
            usedIcd = LW_FALSE;
            retVal = pRiscv[indexGpu].riscvDmemRead(LwU64_LO32(addr - pRiscvInstance->riscv_dmem_start), len, pBuffer, BACKDOOR_NO);
        }
        else if (strstr(memType, "EMEM"))
        {
            usedIcd = LW_FALSE;
            retVal = pRiscv[indexGpu].riscvEmemRead(LwU64_LO32(addr - pRiscvInstance->riscv_emem_start), len, pBuffer, BACKDOOR_NO);
        }
    }

    // Lastly just do memtransfer
    if (usedIcd == LW_TRUE)
    {
        retVal = _riscvMemTransfer(addr, len, pBuffer, MEM_READ);
    }
    return retVal;
}

LW_STATUS riscvMemWrite(LwU64 addr, unsigned len, void *pBuffer, LwBool bForceIcd)
{
    // For small transfers and when forced use ICD
    // The split is there to avoid bar accesses in MPU check
    LwBool usedIcd = LW_TRUE;
    LW_STATUS retVal;

    if (len <= 32 || bForceIcd)
    {
        usedIcd = LW_TRUE;
    }
    else if (!pRiscv[indexGpu].riscvHasMpuEnabled())
    {
        //
        // For rest, try to use mem backdoor as it's ~2x faster.
        // We can do that only if MPU is disabled, as mem backdoor access physical
        // addresses (contrary to ICD)
        //
        const char *memType = pRiscv[indexGpu]._riscvTarget(addr, len);
        if (strstr(memType, "ITCM"))
        {
            usedIcd = LW_FALSE;
            retVal = pRiscv[indexGpu].riscvImemWrite(LwU64_LO32(addr - pRiscvInstance->riscv_imem_start), len, pBuffer, BACKDOOR_NO, LW_FALSE);
        }
        else if (strstr(memType, "DTCM"))
        {
            usedIcd = LW_FALSE;
            retVal = pRiscv[indexGpu].riscvDmemWrite(LwU64_LO32(addr - pRiscvInstance->riscv_dmem_start), len, pBuffer, BACKDOOR_NO);
        }
        else if (strstr(memType, "EMEM"))
        {
            usedIcd = LW_FALSE;
            retVal = pRiscv[indexGpu].riscvEmemWrite(LwU64_LO32(addr - pRiscvInstance->riscv_emem_start), len, pBuffer, BACKDOOR_NO);
        }
    }

    // Lastly just do memtransfer
    if (usedIcd == LW_TRUE)
    {
        retVal = _riscvMemTransfer(addr, len, pBuffer, MEM_WRITE);
    }
    return retVal;
}
