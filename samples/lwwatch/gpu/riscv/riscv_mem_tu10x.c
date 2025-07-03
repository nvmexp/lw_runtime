/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All information
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

#include "turing/tu102/dev_gsp.h"
#include "turing/tu102/dev_sec_pri.h"
#include "turing/tu102/dev_riscv_pri.h"
#include "turing/tu102/dev_falcon_v4.h"

#include "g_riscv_private.h"

#define TO_PFALCON_OFFSET(x) ((LwU32)((x) - pRiscvInstance->bar0Base))

static void _memReadByte_TU10X(LwU32 addr, LwU8 *pBuffer, const LwU32 MemcBase, const LwU32 MemdBase)
{
    LwU32 val;
    LwU32 addressShift;

    addr = addr & 0x00FFFFFFU; //24 bits for [blk(16)|offset(6)|00]
    addressShift = 8 * (addr & 0x3);
    addr = addr & ~0x3;

    bar0WriteLegacy(MemcBase, addr);

    val = bar0ReadLegacy(MemdBase);
    *pBuffer = (val >> addressShift) & 0xFF;
}

static LW_STATUS _riscvReadTcm_TU10X(LwU32 offset, LwU32 len, void *pBuffer,
                                     const LwU32 MemcBase, const LwU32 MemdBase, const LwU32 MemtBase)
{
    // noop
    if (len == 0)
        return LW_OK;

    //
    // Write *MEMT, that is required at least once after reboot, as
    // IMEMT is XXX (has no reset value) on all GPUs
    // DMEMT is 0 on reset, but that may change (default value is not in refman)
    // EMEMT does not exist.
    //
    if (MemtBase != (LwU32) -1)
        bar0WriteLegacy(MemtBase, 0);

    // Do expensive reads until we hit 4-byte alignment
    if (offset & 3)
        dprintf("%s: doing unaligned head reads (%u bytes).\n",
                __FUNCTION__, offset & 3);
    while (offset & 0x3)
    {
        _memReadByte_TU10X(offset, (LwU8*)pBuffer, MemcBase, MemdBase);
        offset++;
        pBuffer = (LwU8*)pBuffer + 1;

        if (--len == 0)
            break;
    }

    // Do fast read for 4-byte aligned reads
    if (len >= 4)
    {
        LwU32 data;
        // LW_PFALCON_*MEMC_AINCR are all the same, but probably not great to rely on this
        bar0WriteLegacy(MemcBase, DRF_DEF(_PFALCON_FALCON, _IMEMC, _AINCR, _TRUE) | offset);
        while (len >= 4)
        {
            //
            // We must copy with memcpy, as at this point pBuffer
            // may be unaligned.
            //
            data = bar0ReadLegacy(MemdBase);
            memcpy(pBuffer, &data, sizeof(data));
            pBuffer = ((LwU8*)pBuffer) + 4;
            len -= 4;
            offset += 4;
        }
    }

    if (len)
        dprintf("%s: doing unaligned tail reads (%u bytes).\n",
                __FUNCTION__, len);
    // Do expensive reads for the "tail"
    while (len)
    {
        _memReadByte_TU10X(offset, (LwU8*)pBuffer, MemcBase, MemdBase);
        offset++;
        pBuffer = (LwU8*)pBuffer + 1;
        len --;
    }

    return LW_OK;
}

LW_STATUS riscvImemRead_TU10X(LwU32 offset, unsigned len, void *pBuffer, int backdoor_no)
{
    // Sanity check on offset and size
    if (len > pRiscvInstance->riscv_imem_size)
    {
        dprintf("%s: Too much to read.\n", __FUNCTION__);
        return LW_ERR_ILWALID_ARGUMENT;
    }
    if (len > (pRiscvInstance->riscv_imem_size - offset))
    {
        dprintf("%s: Too much to read.\n", __FUNCTION__);
        return LW_ERR_ILWALID_ARGUMENT;
    }
    if (offset >= pRiscvInstance->riscv_imem_size)
    {
        dprintf("%s: Address past end of memory.\n", __FUNCTION__);
        return LW_ERR_ILWALID_ARGUMENT;
    }
    return _riscvReadTcm_TU10X(offset, len, pBuffer, LW_PFALCON_FALCON_IMEMC(backdoor_no),
                               LW_PFALCON_FALCON_IMEMD(backdoor_no), LW_PFALCON_FALCON_IMEMT(backdoor_no));
}

LW_STATUS riscvDmemRead_TU10X(LwU32 offset, unsigned len, void *pBuffer, int backdoor_no)
{
    // Sanity check on offset and size
    if (len > pRiscvInstance->riscv_dmem_size)
    {
        dprintf("%s: Too much to read.\n", __FUNCTION__);
        return LW_ERR_ILWALID_ARGUMENT;
    }
    if (len > (pRiscvInstance->riscv_dmem_size - offset))
    {
        dprintf("%s: Too much to read.\n", __FUNCTION__);
        return LW_ERR_ILWALID_ARGUMENT;
    }
    if (offset >= pRiscvInstance->riscv_dmem_size)
    {
        dprintf("%s: Address past end of memory.\n", __FUNCTION__);
        return LW_ERR_ILWALID_ARGUMENT;
    }
    return _riscvReadTcm_TU10X(offset, len, pBuffer, LW_PFALCON_FALCON_DMEMC(backdoor_no),
                               LW_PFALCON_FALCON_DMEMD(backdoor_no), LW_PFALCON_FALCON_DMEMT(backdoor_no));
}

LW_STATUS riscvEmemRead_TU10X(LwU32 offset, unsigned len, void *pBuffer, int backdoor_no)
{
    if (pRiscvInstance->riscv_emem_size == 0)
    {
        dprintf("%s: No EMEM on %s.\n", __FUNCTION__, pRiscvInstance->name);
        return LW_ERR_NOT_SUPPORTED;
    }

    // Sanity check on offset and size
    if (len > pRiscvInstance->riscv_emem_size)
    {
        dprintf("%s: Too much to read.\n", __FUNCTION__);
        return LW_ERR_ILWALID_ARGUMENT;
    }
    if (len > (pRiscvInstance->riscv_emem_size - offset))
    {
        dprintf("%s: Too much to read.\n", __FUNCTION__);
        return LW_ERR_ILWALID_ARGUMENT;
    }
    if (offset >= pRiscvInstance->riscv_emem_size)
    {
        dprintf("%s: Address past end of memory.\n", __FUNCTION__);
        return LW_ERR_ILWALID_ARGUMENT;
    }
    switch (pRiscvInstance->instance_no) {
    case RISCV_INSTANCE_GSP:
        return _riscvReadTcm_TU10X(offset, len, pBuffer, TO_PFALCON_OFFSET(LW_PGSP_EMEMC(backdoor_no)),
                                   TO_PFALCON_OFFSET(LW_PGSP_EMEMD(backdoor_no)), (LwU32) -1);
    case RISCV_INSTANCE_SEC2:
        return _riscvReadTcm_TU10X(offset, len, pBuffer, TO_PFALCON_OFFSET(LW_PSEC_EMEMC(backdoor_no)),
                                   TO_PFALCON_OFFSET(LW_PSEC_EMEMD(backdoor_no)), (LwU32) -1);
    default:
        return LW_ERR_ILWALID_ARGUMENT;
    }
}

static void _memWriteByte_TU10X(LwU32 addr, LwU8 byte, const LwU32 MemcBase, const LwU32 MemdBase)
{
    LwU32 val;
    LwU32 as;

    addr = addr & 0x00FFFFFFU; //24 bits for [blk(16)|offset(6)|00]
    as = 8 * (addr & 0x3);
    addr = addr & ~0x3;

    bar0WriteLegacy(MemcBase, addr);
    // Backup bytes not to write
    val = bar0ReadLegacy(MemdBase);

    val &= (~(0xFF << as)); // mask out other data
    val |= (((LwU32)byte) << as);

    bar0WriteLegacy(MemdBase, val);
}

static LW_STATUS _riscvWriteTcm_TU10X(LwU32 offset, LwU32 len, const void *pBuffer,
                                      const LwU32 MemcBase, const LwU32 MemdBase,
                                      const LwU32 MemtBase, LwBool bSelwre)
{
    // noop
    if (len == 0)
        return LW_OK;

    //
    // Write *MEMT, that is required at least once after reboot, as
    // IMEMT is XXX (has no reset value) on all GPUs
    // DMEMT is 0 on reset, but that may change (default value is not in refman)
    // EMEMT does not exist.
    //
    if (MemtBase != (LwU32) -1)
        bar0WriteLegacy(MemtBase, 0);

    // Sanity check for secure writes - in that case we can only write whole pages
    if (bSelwre)
    {
        if ((offset & FALCON_PAGE_MASK) || (len & FALCON_PAGE_MASK))
        {
            dprintf("%s: In secure mode, it is possible to program whole pages only.\n",
                    __FUNCTION__);
            return LW_ERR_ILWALID_STATE;
        }
    }

    // Do expensive writes until we hit 4-byte alignment
    if (offset & 3)
    {
        dprintf("%s: doing unaligned head writes (%u bytes).\n",
                __FUNCTION__, offset & 3);
    }
    while (offset & 0x3)
    {
        _memWriteByte_TU10X(offset, *(const LwU8*)pBuffer, MemcBase, MemdBase);
        offset++;
        pBuffer = (const LwU8*)pBuffer + 1;

        if (--len == 0)
            break;
    }

    // Do fast write for 4-byte aligned reads
    if (len >= 4)
    {
        LwU32 data;

        bar0WriteLegacy(MemcBase, DRF_DEF(_PFALCON_FALCON, _IMEMC, _AINCW, _TRUE) | offset |
                  DRF_NUM(_PFALCON_FALCON, _IMEMC, _SELWRE, bSelwre? 1:0));
        while (len >= 4)
        {
            //
            // We must copy with memcpy, as at this point pBuffer
            // may be unaligned.
            //
            memcpy(&data, pBuffer, sizeof(data));
            bar0WriteLegacy(MemdBase, data);
            pBuffer = ((const LwU8*)pBuffer) + 4;
            len -= 4;
            offset += 4;
        }
    }

    if (len)
        dprintf("%s: doing unaligned tail writes (%u bytes).\n",
                __FUNCTION__, len);
    // Do expensive writes for the "tail"
    while (len)
    {
        _memWriteByte_TU10X(offset, *(const LwU8*)pBuffer, MemcBase, MemdBase);
        offset++;
        pBuffer = (const LwU8*)pBuffer + 1;
        len --;
    }

    /*
     * HW Trick:
     * IMEM/DMEM page is marked as pending until _last_ word is written.
     * It is enough to read/write last word, even if not full page was written.
     * Otherwise core will stall when trying to access that page.
     */
    if (offset & FALCON_PAGE_MASK)
    {
        LwU32 writeAddr = (offset & (~FALCON_PAGE_MASK)) | (FALCON_PAGE_SIZE - 0x4);
        LwU32 data;

        dprintf("Write didn't complete at end of page. Triggering noop write at 0x%x.\n", writeAddr);
        bar0WriteLegacy(MemcBase, writeAddr);
        data = bar0ReadLegacy(MemdBase);
        bar0WriteLegacy(MemdBase, data);
    }

    return LW_OK;
}

LW_STATUS riscvImemWrite_TU10X(LwU32 offset, unsigned len, const void *pBuffer, int backdoor_no, LwBool bSelwre)
{
    // Sanity check on offset and size
    if (len > pRiscvInstance->riscv_imem_size)
    {
        dprintf("%s: Too much to write.\n", __FUNCTION__);
        return LW_ERR_ILWALID_ARGUMENT;
    }
    if (len > (pRiscvInstance->riscv_imem_size - offset))
    {
        dprintf("%s: Too much to write.\n", __FUNCTION__);
        return LW_ERR_ILWALID_ARGUMENT;
    }
    if (offset >= pRiscvInstance->riscv_imem_size)
    {
        dprintf("%s: Address past end of memory.\n", __FUNCTION__);
        return LW_ERR_ILWALID_ARGUMENT;
    }
    return _riscvWriteTcm_TU10X(offset, len, pBuffer, LW_PFALCON_FALCON_IMEMC(backdoor_no),
                                LW_PFALCON_FALCON_IMEMD(backdoor_no),
                                LW_PFALCON_FALCON_IMEMT(backdoor_no), bSelwre);
}

LW_STATUS riscvDmemWrite_TU10X(LwU32 offset, unsigned len, const void *pBuffer, int backdoor_no)
{
    // Sanity check on offset and size
    if (len > pRiscvInstance->riscv_dmem_size)
    {
        dprintf("%s: Too much to write.\n", __FUNCTION__);
        return LW_ERR_ILWALID_ARGUMENT;
    }
    if (len > (pRiscvInstance->riscv_dmem_size - offset))
    {
        dprintf("%s: Too much to write.\n", __FUNCTION__);
        return LW_ERR_ILWALID_ARGUMENT;
    }
    if (offset >= pRiscvInstance->riscv_dmem_size)
    {
        dprintf("%s: Address past end of memory.\n", __FUNCTION__);
        return LW_ERR_ILWALID_ARGUMENT;
    }
    return _riscvWriteTcm_TU10X(offset, len, pBuffer, LW_PFALCON_FALCON_DMEMC(backdoor_no),
                                LW_PFALCON_FALCON_DMEMD(backdoor_no),
                                LW_PFALCON_FALCON_DMEMT(backdoor_no), LW_FALSE);
}

LW_STATUS riscvEmemWrite_TU10X(LwU32 offset, unsigned len, const void *pBuffer, int backdoor_no)
{
    if (pRiscvInstance->riscv_emem_size == 0)
    {
        dprintf("%s: No EMEM on %s.\n", __FUNCTION__, pRiscvInstance->name);
        return LW_ERR_NOT_SUPPORTED;
    }

    // Sanity check on offset and size
    if (len > pRiscvInstance->riscv_emem_size)
    {
        dprintf("%s: Too much to write.\n", __FUNCTION__);
        return LW_ERR_ILWALID_ARGUMENT;
    }
    if (len > (pRiscvInstance->riscv_emem_size - offset))
    {
        dprintf("%s: Too much to write.\n", __FUNCTION__);
        return LW_ERR_ILWALID_ARGUMENT;
    }
    if (offset >= pRiscvInstance->riscv_emem_size)
    {
        dprintf("%s: Address past end of memory.\n", __FUNCTION__);
        return LW_ERR_ILWALID_ARGUMENT;
    }
    switch (pRiscvInstance->instance_no) {
    case RISCV_INSTANCE_GSP:
        return _riscvWriteTcm_TU10X(offset, len, pBuffer, TO_PFALCON_OFFSET(LW_PGSP_EMEMC(backdoor_no)),
                                    TO_PFALCON_OFFSET(LW_PGSP_EMEMD(backdoor_no)), (LwU32) -1, LW_FALSE);
    case RISCV_INSTANCE_SEC2:
        return _riscvWriteTcm_TU10X(offset, len, pBuffer, TO_PFALCON_OFFSET(LW_PSEC_EMEMC(backdoor_no)),
                                    TO_PFALCON_OFFSET(LW_PSEC_EMEMD(backdoor_no)), (LwU32) -1, LW_FALSE);
    default:
        return LW_ERR_ILWALID_ARGUMENT;
    }
}
